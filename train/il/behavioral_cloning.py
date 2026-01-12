"""
Behavioral Cloning (BC)

The simplest form of imitation learning:
- Collect expert demonstrations
- Train policy via supervised learning
- No environment interaction during training
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

from .base_trainer import ILTrainer, ExpertDataset, PolicyNetwork

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import ILConfig


class BehavioralCloning(ILTrainer):
    """
    Behavioral Cloning trainer.

    Simple supervised learning approach:
    1. Collect expert demonstrations (state, action) pairs
    2. Train policy to predict actions from states
    3. Uses MSE loss for continuous, CE for discrete actions

    Pros:
    - Simple and easy to implement
    - No environment interaction during training
    - Works well with high-quality demonstrations

    Cons:
    - Distribution shift (covariate shift) problem
    - Requires large amounts of expert data
    - Cannot improve beyond expert performance
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        config: Optional[ILConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ILConfig.behavioral_cloning()

        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config = config

        # Training params
        self.num_epochs = config.bc_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.val_split = config.bc_validation_split

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

    def train(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        expert_policy=None,
        num_expert_episodes: int = None,
    ):
        """
        Train policy using behavioral cloning.

        Args:
            states: Expert states (optional if expert_policy provided)
            actions: Expert actions (optional if expert_policy provided)
            expert_policy: Expert policy function for collecting demonstrations
            num_expert_episodes: Number of episodes to collect
        """
        print("=" * 60)
        print("Behavioral Cloning Training")
        print("=" * 60)

        # Collect demonstrations if not provided
        if states is None or actions is None:
            if expert_policy is None:
                raise ValueError("Must provide either (states, actions) or expert_policy")

            if num_expert_episodes is None:
                num_expert_episodes = self.config.num_expert_episodes

            states, actions = self.collect_expert_demonstrations(
                expert_policy, num_expert_episodes
            )

        # Create dataset
        dataset = ExpertDataset(states, actions)

        # Split into train/val
        train_size = int(len(dataset) * (1 - self.val_split))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")

        # Training loop
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            # Training
            self.policy.train()
            epoch_train_loss = 0
            num_batches = 0

            for states_batch, actions_batch in train_loader:
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)

                # Forward pass
                predicted_actions = self.policy(states_batch)

                # Compute loss
                if self.continuous:
                    loss = self.criterion(predicted_actions, actions_batch)
                else:
                    loss = self.criterion(predicted_actions, actions_batch.long())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation
            self.policy.eval()
            epoch_val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for states_batch, actions_batch in val_loader:
                    states_batch = states_batch.to(self.device)
                    actions_batch = actions_batch.to(self.device)

                    predicted_actions = self.policy(states_batch)

                    if self.continuous:
                        loss = self.criterion(predicted_actions, actions_batch)
                    else:
                        loss = self.criterion(predicted_actions, actions_batch.long())

                    epoch_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / num_val_batches
            val_losses.append(avg_val_loss)

            # Logging
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save(os.path.join(self.config.output_dir, "best_policy.pt"))

        # Final evaluation
        print("\nFinal Evaluation:")
        eval_results = self.evaluate()
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

        # Save final model
        self.save()

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "eval_results": eval_results,
        }


class VLABehavioralCloning:
    """
    Behavioral Cloning for VLA models.

    Uses the robot manipulation dataset format with:
    - Images
    - Language instructions
    - Actions
    """

    def __init__(
        self,
        model,
        config: Optional[ILConfig] = None,
    ):
        if config is None:
            config = ILConfig.behavioral_cloning()

        self.model = model
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

        os.makedirs(config.output_dir, exist_ok=True)

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
    ):
        """
        Train VLA model using behavioral cloning.

        Args:
            train_dataloader: DataLoader with robot manipulation data
            val_dataloader: Optional validation DataLoader
        """
        print("=" * 60)
        print("VLA Behavioral Cloning")
        print("=" * 60)

        best_val_loss = float("inf")

        for epoch in range(self.config.bc_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.bc_epochs}",
            )

            for batch in progress_bar:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )

                loss = outputs["loss"]

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

            # Validation
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                print(f"Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save("best_model.pt")

        # Save final model
        self._save("final_model.pt")

    def _validate(self, val_dataloader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )

                total_loss += outputs["loss"].item()
                num_batches += 1

        return total_loss / num_batches

    def _save(self, filename: str):
        """Save model."""
        path = os.path.join(self.config.output_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral Cloning Training")

    # Environment / Data
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment")
    parser.add_argument("--expert_data", type=str, default=None, help="Path to expert data (.npz)")
    parser.add_argument("--num_expert_episodes", type=int, default=50, help="Episodes to collect")

    # Model
    parser.add_argument("--model_path", type=str, default=None, help="VLA model path (for VLA BC)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Training
    parser.add_argument("--bc_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/bc", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def create_simple_expert(env_name: str):
    """Create simple expert policies for common environments."""
    if env_name == "CartPole-v1":
        def policy(state):
            pole_angle = state[2]
            pole_velocity = state[3]
            return 1 if pole_angle + 0.1 * pole_velocity > 0 else 0
        return policy
    elif env_name == "Pendulum-v1":
        def policy(state):
            theta = np.arctan2(state[1], state[0])
            return np.array([-2.0 * theta - 0.1 * state[2]])
        return policy
    else:
        raise ValueError(f"No simple expert for {env_name}. Provide --expert_data instead.")


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Behavioral Cloning Training")
    print("=" * 60)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check if VLA mode
    if args.model_path is not None:
        print("VLA Behavioral Cloning mode")

        from model.vla import VLAModel
        from model.vla.vla_model import VLAConfig

        # Load VLA model
        model_config = VLAConfig()
        model = VLAModel(model_config)

        if os.path.exists(args.model_path):
            state_dict = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded VLA model from {args.model_path}")

        config = ILConfig(
            bc_epochs=args.bc_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        trainer = VLABehavioralCloning(model, config=config)

        # Load dataset
        from train.finetune.dataset import RobotDataset

        try:
            dataset = RobotDataset(dataset_name="lerobot/pusht", max_samples=1000)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            trainer.train(dataloader)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please provide a valid robot dataset")

    else:
        # Standard BC mode
        print(f"Environment: {args.env}")

        import gymnasium as gym
        env = gym.make(args.env)

        config = ILConfig(
            bc_epochs=args.bc_epochs,
            bc_validation_split=args.validation_split,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_expert_episodes=args.num_expert_episodes,
            output_dir=args.output_dir,
        )

        trainer = BehavioralCloning(env, config=config)

        if args.resume:
            trainer.load(args.resume)

        # Load or collect expert data
        if args.expert_data and os.path.exists(args.expert_data):
            data = np.load(args.expert_data)
            states, actions = data["states"], data["actions"]
            print(f"Loaded {len(states)} expert transitions from {args.expert_data}")
            trainer.train(states=states, actions=actions)
        else:
            expert_policy = create_simple_expert(args.env)
            trainer.train(expert_policy=expert_policy, num_expert_episodes=args.num_expert_episodes)

    print("\nTraining complete!")
