"""
TD3+BC Trainer - Offline RL

Implements TD3+BC algorithm for offline reinforcement learning:
- TD3 (Twin Delayed DDPG) as base algorithm
- Behavioral cloning regularization to stay close to data
- Simple and effective offline RL method
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import copy

from .base_trainer import OfflineRLTrainer, OfflineRLConfig, OfflineReplayBuffer


class DeterministicPolicy(nn.Module):
    """Deterministic policy for TD3."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs) * self.max_action


class TwinQNetwork(nn.Module):
    """Twin Q-networks for TD3."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


class TD3BCTrainer(OfflineRLTrainer):
    """
    TD3+BC (TD3 with Behavioral Cloning) Trainer.

    Combines TD3 with BC regularization:
    - Standard TD3 Q-learning with target networks
    - BC term to keep policy close to data
    - Normalized BC coefficient based on Q-values

    Reference: Fujimoto & Gu, "A Minimalist Approach to Offline RL"
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[OfflineRLConfig] = None,
        alpha: float = 2.5,  # BC coefficient
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        max_action: float = 1.0,
    ):
        if config is None:
            config = OfflineRLConfig()

        # Create policy
        policy = DeterministicPolicy(obs_dim, action_dim, config.hidden_dim, max_action)

        super().__init__(config, policy)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # TD3+BC specific
        self.alpha = alpha
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action

        # Networks
        self.q_network = TwinQNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)

        # Target networks
        self.policy_target = copy.deepcopy(self.policy)
        self.q_target = copy.deepcopy(self.q_network)

        for param in self.policy_target.parameters():
            param.requires_grad = False
        for param in self.q_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
        self.q_optimizer = Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Training counter
        self.total_it = 0

    def train(self, buffer: OfflineReplayBuffer):
        """Run TD3+BC offline training."""
        print("=" * 60)
        print("TD3+BC Offline Training")
        print("=" * 60)
        print(f"BC Alpha: {self.alpha}")
        print(f"Dataset size: {buffer.size}")

        # Normalize data
        self.obs_mean, self.obs_std = buffer.normalize()

        num_batches = buffer.size // self.config.batch_size
        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = {"q_loss": [], "policy_loss": [], "bc_loss": []}

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for _ in progress_bar:
                batch = buffer.sample(self.config.batch_size)
                metrics = self.train_step(batch)

                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

                progress_bar.set_postfix({
                    "q_loss": f"{metrics['q_loss']:.3f}",
                    "bc": f"{metrics.get('bc_loss', 0):.3f}",
                })

            # Epoch summary
            avg_q_loss = np.mean(epoch_metrics["q_loss"])
            avg_bc_loss = np.mean([x for x in epoch_metrics["bc_loss"] if x > 0])
            print(f"Epoch {epoch+1} - Q Loss: {avg_q_loss:.4f}, BC Loss: {avg_bc_loss:.4f}")

            # Save best
            if avg_q_loss < best_loss:
                best_loss = avg_q_loss
                self.save(os.path.join(self.config.output_dir, "best_model.pt"))

            # Periodic save
            if (epoch + 1) % self.config.save_freq == 0:
                self.save(os.path.join(self.config.output_dir, f"model_epoch_{epoch+1}.pt"))

        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one TD3+BC training step."""
        self.total_it += 1

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        # === Q-function update ===
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.policy_target(next_observations) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Target Q-value
            q1_next, q2_next = self.q_target(next_observations, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards.unsqueeze(-1) + self.config.discount_gamma * (1 - dones.unsqueeze(-1)) * q_next

        q1, q2 = self.q_network(observations, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        metrics = {
            "q_loss": q_loss.item(),
            "policy_loss": 0.0,
            "bc_loss": 0.0,
        }

        # === Delayed policy update ===
        if self.total_it % self.policy_freq == 0:
            # Policy actions
            policy_actions = self.policy(observations)

            # Q-value for policy actions
            q_value = self.q_network.q1_forward(observations, policy_actions)

            # Normalize Q-values for BC coefficient
            lmbda = self.alpha / q_value.abs().mean().detach()

            # BC loss (MSE to dataset actions)
            bc_loss = F.mse_loss(policy_actions, actions)

            # Combined loss: maximize Q - BC penalty
            policy_loss = -lmbda * q_value.mean() + bc_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            metrics["policy_loss"] = policy_loss.item()
            metrics["bc_loss"] = bc_loss.item()

            # Soft target update
            with torch.no_grad():
                for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )

                for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )

        return metrics

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select action for evaluation."""
        # Normalize observation
        if hasattr(self, 'obs_mean'):
            obs = (obs - torch.tensor(self.obs_mean, device=self.device)) / torch.tensor(self.obs_std, device=self.device)

        return self.policy(obs)

    def save(self, path: str = None):
        """Save all networks."""
        if path is None:
            path = os.path.join(self.config.output_dir, "model.pt")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "q_network_state_dict": self.q_network.state_dict(),
            "obs_mean": getattr(self, 'obs_mean', None),
            "obs_std": getattr(self, 'obs_std', None),
            "config": self.config,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load all networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])

        self.policy_target = copy.deepcopy(self.policy)
        self.q_target = copy.deepcopy(self.q_network)

        if checkpoint.get("obs_mean") is not None:
            self.obs_mean = checkpoint["obs_mean"]
            self.obs_std = checkpoint["obs_std"]

        print(f"Loaded model from {path}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="TD3+BC Offline Training")

    parser.add_argument("--dataset", type=str, default="hopper-medium-v2", help="D4RL dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=2.5, help="BC coefficient")
    parser.add_argument("--output_dir", type=str, default="./output/td3bc", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("TD3+BC Offline Training")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = OfflineRLConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    trainer = TD3BCTrainer(
        obs_dim=11,
        action_dim=3,
        config=config,
        alpha=args.alpha,
    )

    # Load dataset
    from .base_trainer import create_dummy_dataset
    dataset = create_dummy_dataset(obs_dim=11, action_dim=3)

    buffer = OfflineReplayBuffer(obs_dim=11, action_dim=3, device=str(trainer.device))
    buffer.load_dataset(dataset)

    trainer.train(buffer)

    print("\nTraining complete!")
