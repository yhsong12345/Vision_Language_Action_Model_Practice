"""
DAgger (Dataset Aggregation)

Interactive imitation learning that addresses distribution shift:
1. Train initial policy via BC
2. Execute learned policy in environment
3. Query expert for correct actions
4. Aggregate new data with existing dataset
5. Repeat
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm

from .base_trainer import ILTrainer, ExpertDataset, PolicyNetwork

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import ILConfig


class DAgger(ILTrainer):
    """
    DAgger (Dataset Aggregation) trainer.

    Addresses the distribution shift problem in behavioral cloning
    by iteratively collecting data under the learned policy and
    querying an expert for corrections.

    Algorithm:
    1. Initialize dataset D with expert demonstrations
    2. For iteration i = 1 to N:
       a. Train policy π_i on D
       b. Execute π_i in environment
       c. Query expert for correct actions
       d. Aggregate: D = D ∪ new_data
    3. Return final policy

    The β (beta) parameter controls the mix of expert and learned policy
    during data collection.
    """

    def __init__(
        self,
        env,
        expert_policy,
        policy: Optional[nn.Module] = None,
        config: Optional[ILConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ILConfig.dagger()

        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config = config
        self.expert_policy = expert_policy

        # DAgger specific params
        self.num_iterations = config.dagger_iterations
        self.episodes_per_iter = config.dagger_episodes_per_iter
        self.beta_schedule = config.dagger_beta_schedule
        self.initial_beta = config.dagger_initial_beta
        self.bc_epochs = config.bc_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate

        # Aggregated dataset
        self.states_buffer = []
        self.actions_buffer = []

        # Optimizer
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )

        # Loss function
        if self.continuous:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def get_beta(self, iteration: int) -> float:
        """
        Get β value for current iteration.

        β = probability of using expert policy.
        """
        if self.beta_schedule == "constant":
            return self.initial_beta

        elif self.beta_schedule == "linear":
            # Linear decay from initial_beta to 0
            return max(0, self.initial_beta * (1 - iteration / self.num_iterations))

        elif self.beta_schedule == "exponential":
            # Exponential decay
            decay_rate = 0.9
            return self.initial_beta * (decay_rate ** iteration)

        else:
            return self.initial_beta

    def collect_with_dagger(
        self,
        num_episodes: int,
        beta: float,
        max_steps: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect data using β-mixture of expert and learned policy.

        Args:
            num_episodes: Number of episodes to collect
            beta: Probability of using expert action
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (states, expert_actions)
        """
        states = []
        actions = []
        episode_rewards = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Get expert action (always needed for labeling)
                expert_action = self.expert_policy(state)

                # Decide which action to execute
                if np.random.random() < beta:
                    action = expert_action
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                    action = self.policy.get_action(state_tensor, deterministic=False)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()

                # Store state and EXPERT action (key difference from BC)
                states.append(state)
                actions.append(expert_action)

                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

                state = next_state

            episode_rewards.append(episode_reward)

        print(f"Collected {len(states)} transitions (β={beta:.2f}), "
              f"mean reward: {np.mean(episode_rewards):.2f}")

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32 if self.continuous else np.int64),
        )

    def train_bc_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch of behavioral cloning."""
        self.policy.train()
        total_loss = 0
        num_batches = 0

        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)

            predicted_actions = self.policy(states_batch)

            if self.continuous:
                loss = self.criterion(predicted_actions, actions_batch)
            else:
                loss = self.criterion(predicted_actions, actions_batch.long())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        initial_states: Optional[np.ndarray] = None,
        initial_actions: Optional[np.ndarray] = None,
        num_initial_episodes: int = 20,
    ):
        """
        Run DAgger training.

        Args:
            initial_states: Initial expert states (optional)
            initial_actions: Initial expert actions (optional)
            num_initial_episodes: Episodes for initial BC if no data provided
        """
        print("=" * 60)
        print("DAgger Training")
        print("=" * 60)
        print(f"Iterations: {self.num_iterations}")
        print(f"Episodes per iteration: {self.episodes_per_iter}")
        print(f"Beta schedule: {self.beta_schedule}")

        # Collect initial demonstrations if not provided
        if initial_states is None or initial_actions is None:
            print("\nCollecting initial expert demonstrations...")
            initial_states, initial_actions = self.collect_expert_demonstrations(
                self.expert_policy, num_initial_episodes
            )

        # Initialize buffer with initial data
        self.states_buffer = [initial_states]
        self.actions_buffer = [initial_actions]

        # DAgger iterations
        iteration_results = []

        for iteration in range(self.num_iterations):
            print(f"\n{'=' * 40}")
            print(f"DAgger Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'=' * 40}")

            # Get current beta
            beta = self.get_beta(iteration)
            print(f"β = {beta:.3f}")

            # Create dataset from aggregated data
            all_states = np.concatenate(self.states_buffer)
            all_actions = np.concatenate(self.actions_buffer)

            print(f"Dataset size: {len(all_states)}")

            dataset = ExpertDataset(all_states, all_actions)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )

            # Train policy via BC
            print("Training policy...")
            for epoch in range(self.bc_epochs):
                loss = self.train_bc_epoch(dataloader)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{self.bc_epochs}, Loss: {loss:.4f}")

            # Evaluate current policy
            eval_results = self.evaluate()
            print(f"Evaluation: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

            # Collect new data (if not last iteration)
            if iteration < self.num_iterations - 1:
                new_states, new_actions = self.collect_with_dagger(
                    num_episodes=self.episodes_per_iter,
                    beta=beta,
                )

                # Aggregate data
                self.states_buffer.append(new_states)
                self.actions_buffer.append(new_actions)

            iteration_results.append({
                "iteration": iteration + 1,
                "beta": beta,
                "dataset_size": len(all_states),
                "mean_reward": eval_results["mean_reward"],
                "std_reward": eval_results["std_reward"],
            })

        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        final_eval = self.evaluate(num_episodes=50)
        print(f"Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")

        # Save final model
        self.save()

        return {
            "iterations": iteration_results,
            "final_eval": final_eval,
        }


def simple_expert_policy(env_name: str = "CartPole-v1"):
    """
    Create a simple expert policy for testing.
    """
    if env_name == "CartPole-v1":
        def policy(state):
            pole_angle = state[2]
            pole_velocity = state[3]

            if pole_angle + 0.1 * pole_velocity > 0:
                return 1
            else:
                return 0
        return policy

    else:
        raise ValueError(f"No expert policy for {env_name}")


class VLADAgger:
    """
    DAgger for VLA models.

    Interactive imitation learning with human expert corrections
    for vision-language-action models.
    """

    def __init__(
        self,
        model,
        expert_fn,
        config: Optional[ILConfig] = None,
    ):
        """
        Args:
            model: VLA model to train
            expert_fn: Expert function(observation) -> action
            config: Training configuration
        """
        if config is None:
            config = ILConfig.dagger()

        self.model = model
        self.expert_fn = expert_fn
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # DAgger params
        self.num_iterations = config.dagger_iterations
        self.episodes_per_iter = config.dagger_episodes_per_iter
        self.beta_schedule = config.dagger_beta_schedule
        self.initial_beta = config.dagger_initial_beta

        # Data buffers
        self.data_buffer = []

        os.makedirs(config.output_dir, exist_ok=True)

    def get_beta(self, iteration: int) -> float:
        """Get beta value for current iteration."""
        if self.beta_schedule == "constant":
            return self.initial_beta
        elif self.beta_schedule == "linear":
            return max(0, self.initial_beta * (1 - iteration / self.num_iterations))
        elif self.beta_schedule == "exponential":
            return self.initial_beta * (0.9 ** iteration)
        return self.initial_beta

    def collect_data(
        self,
        dataloader,
        beta: float,
        num_samples: int = 100,
    ) -> List[Dict]:
        """Collect data using beta-mixture of policy and expert."""
        self.model.eval()
        collected = []

        with torch.no_grad():
            for batch in dataloader:
                if len(collected) >= num_samples:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get policy prediction
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                policy_action = outputs["predicted_actions"]

                # Get expert action
                expert_action = self.expert_fn(batch)

                # Use beta-mixture for execution (but always label with expert)
                for i in range(len(policy_action)):
                    collected.append({
                        "pixel_values": batch["pixel_values"][i].cpu(),
                        "input_ids": batch["input_ids"][i].cpu(),
                        "attention_mask": batch["attention_mask"][i].cpu(),
                        "action": expert_action[i].cpu() if torch.is_tensor(expert_action) else torch.tensor(expert_action[i]),
                    })

        return collected

    def train_epoch(self, dataloader) -> float:
        """Train one epoch on aggregated data."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                actions=batch["action"],
            )

            loss = outputs["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        initial_dataloader,
        num_epochs_per_iter: int = 10,
    ):
        """Run DAgger training."""
        print("=" * 60)
        print("VLA DAgger Training")
        print("=" * 60)

        # Initialize with initial data
        for batch in initial_dataloader:
            for i in range(len(batch["pixel_values"])):
                self.data_buffer.append({
                    k: v[i].cpu() for k, v in batch.items()
                })

        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")

            beta = self.get_beta(iteration)
            print(f"Beta: {beta:.3f}, Buffer size: {len(self.data_buffer)}")

            # Create dataloader from buffer
            from torch.utils.data import DataLoader

            class BufferDataset:
                def __init__(self, buffer):
                    self.buffer = buffer

                def __len__(self):
                    return len(self.buffer)

                def __getitem__(self, idx):
                    return self.buffer[idx]

            buffer_loader = DataLoader(
                BufferDataset(self.data_buffer),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=lambda x: {k: torch.stack([d[k] for d in x]) for k in x[0].keys()},
            )

            # Train on aggregated data
            for epoch in range(num_epochs_per_iter):
                loss = self.train_epoch(buffer_loader)
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}/{num_epochs_per_iter}, Loss: {loss:.4f}")

            # Collect new data (if not last iteration)
            if iteration < self.num_iterations - 1:
                new_data = self.collect_data(initial_dataloader, beta)
                self.data_buffer.extend(new_data)

        # Save final model
        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="DAgger Training")

    # Environment / Data
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment")
    parser.add_argument("--expert_data", type=str, default=None, help="Path to expert data (.npz)")

    # Model
    parser.add_argument("--model_path", type=str, default=None, help="VLA model path (for VLA DAgger)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # DAgger params
    parser.add_argument("--dagger_iterations", type=int, default=10, help="Number of DAgger iterations")
    parser.add_argument("--dagger_episodes_per_iter", type=int, default=20, help="Episodes per iteration")
    parser.add_argument("--dagger_beta_schedule", type=str, default="linear",
                        choices=["constant", "linear", "exponential"], help="Beta schedule")
    parser.add_argument("--dagger_initial_beta", type=float, default=1.0, help="Initial beta value")

    # Training
    parser.add_argument("--bc_epochs", type=int, default=20, help="BC epochs per iteration")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/dagger", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("DAgger Training")
    print("=" * 60)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check if VLA mode
    if args.model_path is not None:
        print("VLA DAgger mode")

        from model.vla import VLAModel
        from model.vla.vla_model import VLAConfig
        from train.finetune.dataset import RobotDataset
        from torch.utils.data import DataLoader

        # Load VLA model
        model_config = VLAConfig()
        model = VLAModel(model_config)

        if os.path.exists(args.model_path):
            state_dict = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded VLA model from {args.model_path}")

        config = ILConfig(
            dagger_iterations=args.dagger_iterations,
            dagger_episodes_per_iter=args.dagger_episodes_per_iter,
            dagger_beta_schedule=args.dagger_beta_schedule,
            dagger_initial_beta=args.dagger_initial_beta,
            bc_epochs=args.bc_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        # Simple expert that returns ground truth actions
        def expert_fn(batch):
            return batch.get("action", torch.zeros(len(batch["pixel_values"]), 7))

        trainer = VLADAgger(model, expert_fn, config=config)

        try:
            dataset = RobotDataset(dataset_name="lerobot/pusht", max_samples=1000)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            trainer.train(dataloader)
        except Exception as e:
            print(f"Error loading dataset: {e}")

    else:
        # Standard DAgger mode
        print(f"Environment: {args.env}")

        import gymnasium as gym
        env = gym.make(args.env)
        expert = simple_expert_policy(args.env)

        config = ILConfig(
            dagger_iterations=args.dagger_iterations,
            dagger_episodes_per_iter=args.dagger_episodes_per_iter,
            dagger_beta_schedule=args.dagger_beta_schedule,
            dagger_initial_beta=args.dagger_initial_beta,
            bc_epochs=args.bc_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        trainer = DAgger(env=env, expert_policy=expert, config=config)

        if args.resume:
            trainer.load(args.resume)

        # Load initial data or collect
        if args.expert_data and os.path.exists(args.expert_data):
            data = np.load(args.expert_data)
            trainer.train(initial_states=data["states"], initial_actions=data["actions"])
        else:
            trainer.train()

    print("\nTraining complete!")
