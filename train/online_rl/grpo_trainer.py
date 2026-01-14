"""
GRPO (Group Relative Policy Optimization) Trainer - Online RL

Implements GRPO algorithm for VLA online fine-tuning:
- Sample multiple outputs per prompt
- Rank by reward from environment
- Optimize with relative ranking
- Particularly suited for language model based policies
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


class GRPOTrainer:
    """
    Online GRPO Trainer for VLA models.

    GRPO (Group Relative Policy Optimization) works by:
    1. Sampling multiple action sequences per observation from environment
    2. Computing rewards for each sequence via environment interaction
    3. Ranking samples by reward
    4. Optimizing policy to increase probability of high-reward samples

    This version interacts with the environment to get real rewards.
    """

    def __init__(
        self,
        model,
        env,
        config: Optional[RLConfig] = None,
        reference_model: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: VLA model to train
            env: Environment for getting rewards
            config: Training configuration
            reference_model: Optional frozen reference for KL penalty
        """
        if config is None:
            config = RLConfig.grpo_vla()

        self.model = model
        self.env = env
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
        self.group_size = config.grpo_group_size
        self.kl_coef = config.grpo_kl_coef
        self.total_timesteps = config.total_timesteps

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Statistics
        self.episode_rewards = []

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
                action_std = 0.1
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

    def get_env_reward(
        self,
        observation: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> float:
        """
        Execute action in environment and get reward.

        Args:
            observation: Current observation
            action: Action to execute

        Returns:
            reward: Environment reward
        """
        action_np = action.cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np[0]

        _, reward, _, _, _ = self.env.step(action_np)
        return reward

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
        Perform one online GRPO training step.

        Args:
            observation: Input observation with image and instruction

        Returns:
            Metrics dictionary
        """
        # Sample multiple actions
        actions, old_log_probs = self.sample_actions(observation, self.group_size)

        # Get rewards from environment for each action
        rewards = []
        for action in actions:
            reward = self.get_env_reward(observation, action)
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
        num_episodes: int = 1000,
    ):
        """
        Run online GRPO training with environment interaction.

        Args:
            num_episodes: Number of episodes to train
        """
        print("=" * 60)
        print("Online GRPO Training")
        print("=" * 60)
        print(f"Group size: {self.group_size}")
        print(f"KL coefficient: {self.kl_coef}")

        best_reward = float("-inf")
        global_step = 0

        progress_bar = tqdm(range(num_episodes), desc="Training")

        for episode in progress_bar:
            # Reset environment
            obs, _ = self.env.reset()

            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Convert observation to model input format
                # This depends on your specific observation format
                observation = self._prepare_observation(obs)

                # Training step
                metrics = self.train_step(observation)

                # Take best action for environment step
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(
                        pixel_values=observation["pixel_values"],
                        input_ids=observation["input_ids"],
                        attention_mask=observation["attention_mask"],
                    )
                    action = outputs["predicted_actions"]
                    self.model.train()

                # Environment step
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1
                global_step += 1

            self.episode_rewards.append(episode_reward)

            progress_bar.set_postfix({
                "reward": f"{episode_reward:.2f}",
                "avg_reward": f"{np.mean(self.episode_rewards[-100:]):.2f}",
            })

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save(os.path.join(self.config.output_dir, "best_model.pt"))

            # Periodic checkpoint
            if episode % 100 == 0:
                self.save(os.path.join(self.config.output_dir, f"model_{episode}.pt"))

        # Save final model
        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def _prepare_observation(self, obs) -> Dict[str, torch.Tensor]:
        """
        Prepare environment observation for model input.

        Override this method based on your specific observation format.
        """
        # Default implementation assumes obs is an image
        if isinstance(obs, np.ndarray):
            # Assume image observation
            pixel_values = torch.tensor(obs, dtype=torch.float32)
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.permute(2, 0, 1)  # HWC -> CHW
            pixel_values = pixel_values.unsqueeze(0).to(self.device)

            # Dummy text input
            input_ids = torch.zeros(1, 32, dtype=torch.long, device=self.device)
            attention_mask = torch.ones(1, 32, device=self.device)
        else:
            # Handle dict observation
            pixel_values = torch.tensor(obs.get("image", np.zeros((3, 224, 224))),
                                       dtype=torch.float32).unsqueeze(0).to(self.device)
            input_ids = torch.zeros(1, 32, dtype=torch.long, device=self.device)
            attention_mask = torch.ones(1, 32, device=self.device)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model from {path}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Online GRPO Training for VLA")

    # Model
    parser.add_argument("--model_path", type=str, default=None, help="Path to VLA model")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment")

    # GRPO parameters
    parser.add_argument("--grpo_group_size", type=int, default=8, help="Samples per prompt")
    parser.add_argument("--grpo_kl_coef", type=float, default=0.1, help="KL divergence coefficient")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Training episodes")

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/online_grpo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Online GRPO Training for VLA")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create simple test model and environment
    import gymnasium as gym
    env = gym.make(args.env)

    print(f"Environment: {args.env}")
    print(f"Number of episodes: {args.num_episodes}")

    print("\nNote: Full VLA GRPO training requires VLA model and appropriate environment.")
    print("This script demonstrates the online GRPO training loop.")
