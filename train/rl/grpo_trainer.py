"""
GRPO (Group Relative Policy Optimization) Trainer

Implements GRPO algorithm for VLA fine-tuning:
- Sample multiple outputs per prompt
- Rank by reward
- Optimize with relative ranking
- Particularly suited for language model based policies

This is the RL method used by DeepSeek and similar models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import RLConfig


@dataclass
class GRPOSample:
    """A single sample for GRPO training."""
    observation: torch.Tensor
    instruction: str
    actions: List[torch.Tensor]  # Multiple action samples
    rewards: List[float]  # Reward for each sample
    log_probs: List[torch.Tensor]  # Log probs for each sample


class GRPOTrainer:
    """
    GRPO Trainer for VLA models.

    GRPO (Group Relative Policy Optimization) works by:
    1. Sampling multiple action sequences per observation
    2. Computing rewards for each sequence
    3. Ranking samples by reward
    4. Optimizing policy to increase probability of high-reward samples

    This is particularly effective for VLA models where the action
    space is continuous and high-dimensional.

    Reference: DeepSeek-R1 and similar papers on relative reward optimization.
    """

    def __init__(
        self,
        model,
        reward_fn,
        config: Optional[RLConfig] = None,
        reference_model: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: VLA model to train
            reward_fn: Function(obs, actions) -> reward
            config: Training configuration
            reference_model: Optional frozen reference for KL penalty
        """
        if config is None:
            config = RLConfig.grpo_vla()

        self.model = model
        self.reward_fn = reward_fn
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Reference model for KL divergence
        if reference_model is not None:
            self.ref_model = reference_model.to(self.device)
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = None

        # GRPO specific params
        self.group_size = config.grpo_group_size  # Samples per prompt
        self.kl_coef = config.grpo_kl_coef

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def sample_actions(
        self,
        observation: Dict[str, torch.Tensor],
        num_samples: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample multiple actions from the model.

        Args:
            observation: Dict with pixel_values, input_ids, attention_mask
            num_samples: Number of samples to generate

        Returns:
            Tuple of (actions_list, log_probs_list)
        """
        self.model.eval()
        actions_list = []
        log_probs_list = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass with stochastic sampling
                outputs = self.model(
                    pixel_values=observation["pixel_values"],
                    input_ids=observation["input_ids"],
                    attention_mask=observation["attention_mask"],
                )

                # Get predicted action
                action_mean = outputs["predicted_actions"]

                # Add noise for exploration
                action_std = 0.1  # Can be learned
                noise = torch.randn_like(action_mean) * action_std
                action = action_mean + noise

                # Compute log probability (Gaussian)
                log_prob = -0.5 * ((action - action_mean) / action_std).pow(2).sum(-1)
                log_prob -= 0.5 * action_mean.shape[-1] * np.log(2 * np.pi)
                log_prob -= action_mean.shape[-1] * np.log(action_std)

                actions_list.append(action)
                log_probs_list.append(log_prob)

        self.model.train()
        return actions_list, log_probs_list

    def compute_advantages(
        self,
        rewards: List[float],
        baseline: str = "mean",
    ) -> torch.Tensor:
        """
        Compute advantages from rewards using group baseline.

        Args:
            rewards: List of rewards for each sample in the group
            baseline: "mean" or "min" for baseline computation

        Returns:
            advantages: Normalized advantages tensor
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        if baseline == "mean":
            baseline_value = rewards_tensor.mean()
        elif baseline == "min":
            baseline_value = rewards_tensor.min()
        else:
            baseline_value = 0

        advantages = rewards_tensor - baseline_value

        # Normalize
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def grpo_loss(
        self,
        observation: Dict[str, torch.Tensor],
        actions: List[torch.Tensor],
        old_log_probs: List[torch.Tensor],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GRPO loss.

        Args:
            observation: Input observation
            actions: List of sampled actions
            old_log_probs: Log probabilities from sampling
            advantages: Computed advantages

        Returns:
            GRPO loss
        """
        total_loss = 0

        for i, (action, old_lp, adv) in enumerate(zip(actions, old_log_probs, advantages)):
            # Compute current log probability
            outputs = self.model(
                pixel_values=observation["pixel_values"],
                input_ids=observation["input_ids"],
                attention_mask=observation["attention_mask"],
            )

            action_mean = outputs["predicted_actions"]
            action_std = 0.1

            new_log_prob = -0.5 * ((action - action_mean) / action_std).pow(2).sum(-1)
            new_log_prob -= 0.5 * action_mean.shape[-1] * np.log(2 * np.pi)
            new_log_prob -= action_mean.shape[-1] * np.log(action_std)

            # Policy gradient loss with importance sampling
            ratio = torch.exp(new_log_prob - old_lp.to(self.device))

            # Clip ratio for stability
            ratio_clipped = torch.clamp(ratio, 0.8, 1.2)

            policy_loss = -torch.min(
                ratio * adv.to(self.device),
                ratio_clipped * adv.to(self.device),
            ).mean()

            # KL penalty
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        pixel_values=observation["pixel_values"],
                        input_ids=observation["input_ids"],
                        attention_mask=observation["attention_mask"],
                    )
                    ref_mean = ref_outputs["predicted_actions"]

                kl_div = 0.5 * ((action_mean - ref_mean) / action_std).pow(2).sum(-1).mean()
                policy_loss += self.kl_coef * kl_div

            total_loss += policy_loss

        return total_loss / len(actions)

    def train_step(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.

        Args:
            observation: Input observation with image and instruction

        Returns:
            Metrics dictionary
        """
        # Sample multiple actions
        actions, old_log_probs = self.sample_actions(observation, self.group_size)

        # Get rewards for each action
        rewards = []
        for action in actions:
            reward = self.reward_fn(observation, action)
            rewards.append(reward)

        # Compute advantages
        advantages = self.compute_advantages(rewards)

        # Compute loss
        loss = self.grpo_loss(observation, actions, old_log_probs, advantages)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "reward_std": np.std(rewards),
        }

    def train(
        self,
        train_dataloader,
        num_epochs: int = None,
    ):
        """
        Run GRPO training.

        Args:
            train_dataloader: DataLoader providing observations
            num_epochs: Number of training epochs
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print("=" * 60)
        print("GRPO Training")
        print("=" * 60)
        print(f"Group size: {self.group_size}")
        print(f"KL coefficient: {self.kl_coef}")

        best_reward = float("-inf")
        global_step = 0

        for epoch in range(num_epochs):
            epoch_metrics = {"loss": [], "mean_reward": [], "max_reward": []}

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            )

            for batch in progress_bar:
                # Move to device
                observation = {
                    k: v.to(self.device) for k, v in batch.items()
                }

                # Training step
                metrics = self.train_step(observation)

                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

                global_step += 1

                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.3f}",
                })

                # Save best model
                if metrics["mean_reward"] > best_reward:
                    best_reward = metrics["mean_reward"]
                    self.save(os.path.join(self.config.output_dir, "best_model.pt"))

            # Epoch summary
            avg_reward = np.mean(epoch_metrics["mean_reward"])
            print(f"Epoch {epoch + 1} - Avg Reward: {avg_reward:.4f}")

        # Save final model
        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model from {path}")


class SimpleRewardFunction:
    """
    Simple reward function for VLA training.

    Can be used for:
    - Task completion rewards
    - Distance-based rewards
    - Safety constraint rewards
    """

    def __init__(
        self,
        target_action: Optional[torch.Tensor] = None,
        reward_type: str = "distance",
    ):
        self.target_action = target_action
        self.reward_type = reward_type

    def __call__(
        self,
        observation: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> float:
        if self.reward_type == "distance" and self.target_action is not None:
            # Negative distance to target
            distance = torch.norm(action - self.target_action.to(action.device))
            return -distance.item()

        elif self.reward_type == "smoothness":
            # Reward smooth actions (low magnitude)
            return -torch.norm(action).item()

        else:
            # Default: random reward for testing
            return np.random.randn()


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training for VLA")

    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to VLA model")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # GRPO parameters
    parser.add_argument("--grpo_group_size", type=int, default=8, help="Samples per prompt")
    parser.add_argument("--grpo_kl_coef", type=float, default=0.1, help="KL divergence coefficient")
    parser.add_argument("--num_epochs", type=int, default=10, help="Training epochs")

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # Data
    parser.add_argument("--dataset", type=str, default="lerobot/pusht", help="Dataset name")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples")

    # Reward
    parser.add_argument("--reward_type", type=str, default="smoothness",
                        choices=["distance", "smoothness"], help="Reward function type")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/grpo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("GRPO Training for VLA")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Import VLA model
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from model.vla import VLAModel
    from model.vla.vla_model import VLAConfig

    # Load VLA model
    model_config = VLAConfig()
    model = VLAModel(model_config)

    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VLA model from {args.model_path}")
    else:
        print(f"Warning: Model path {args.model_path} not found, using random init")

    config = RLConfig(
        algorithm="grpo",
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        grpo_group_size=args.grpo_group_size,
        grpo_kl_coef=args.grpo_kl_coef,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    reward_fn = SimpleRewardFunction(reward_type=args.reward_type)
    trainer = GRPOTrainer(model, reward_fn, config=config)

    if args.resume:
        trainer.load(args.resume)

    # Load dataset
    try:
        from train.finetune.dataset import RobotDataset
        from torch.utils.data import DataLoader

        dataset = RobotDataset(dataset_name=args.dataset, max_samples=args.max_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        print(f"Dataset: {args.dataset}, Samples: {len(dataset)}")
        trainer.train(dataloader, num_epochs=args.num_epochs)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy data for testing...")

        from torch.utils.data import DataLoader, TensorDataset

        dummy_dataset = TensorDataset(
            torch.randn(100, 3, 224, 224),
            torch.randint(0, 1000, (100, 32)),
            torch.ones(100, 32),
        )

        class DictDataLoader:
            def __init__(self, loader):
                self.loader = loader

            def __iter__(self):
                for pv, ids, mask in self.loader:
                    yield {"pixel_values": pv, "input_ids": ids, "attention_mask": mask}

            def __len__(self):
                return len(self.loader)

        dataloader = DictDataLoader(DataLoader(dummy_dataset, batch_size=args.batch_size))
        trainer.train(dataloader, num_epochs=args.num_epochs)

    print("\nTraining complete!")
