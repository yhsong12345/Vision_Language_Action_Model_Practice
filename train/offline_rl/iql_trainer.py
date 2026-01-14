"""
IQL (Implicit Q-Learning) Trainer - Offline RL

Implements IQL algorithm for offline reinforcement learning:
- Avoids querying OOD actions by using expectile regression
- Implicit policy extraction through advantage-weighted BC
- Simple and stable offline RL method
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


class ValueNetwork(nn.Module):
    """State value function V(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class QNetwork(nn.Module):
    """Q-function Q(s, a)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


class TwinQNetwork(nn.Module):
    """Twin Q-networks."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class GaussianPolicy(nn.Module):
    """Gaussian policy for IQL."""

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.network(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return normal.log_prob(action).sum(-1, keepdim=True)

    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return normal.rsample()

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            return mean
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return normal.sample()


class IQLTrainer(OfflineRLTrainer):
    """
    Implicit Q-Learning (IQL) Trainer.

    IQL avoids querying OOD actions by:
    1. Learning V(s) using expectile regression on Q(s, a)
    2. Learning Q(s, a) with standard Bellman backup using V(s')
    3. Extracting policy via advantage-weighted behavioral cloning

    Reference: Kostrikov et al., "Offline Reinforcement Learning with
               Implicit Q-Learning"
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[OfflineRLConfig] = None,
        expectile: float = 0.7,
        temperature: float = 3.0,
    ):
        if config is None:
            config = OfflineRLConfig()

        # Create policy
        policy = GaussianPolicy(obs_dim, action_dim, config.hidden_dim)

        super().__init__(config, policy)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # IQL specific
        self.expectile = expectile  # tau for expectile regression
        self.temperature = temperature  # beta for advantage weighting

        # Networks
        self.v_network = ValueNetwork(obs_dim, config.hidden_dim).to(self.device)
        self.q_network = TwinQNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_network)

        for param in self.q_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.v_optimizer = Adam(self.v_network.parameters(), lr=config.learning_rate)
        self.q_optimizer = Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)

    def train(self, buffer: OfflineReplayBuffer):
        """Run IQL offline training."""
        print("=" * 60)
        print("IQL Offline Training")
        print("=" * 60)
        print(f"Expectile: {self.expectile}")
        print(f"Temperature: {self.temperature}")
        print(f"Dataset size: {buffer.size}")

        num_batches = buffer.size // self.config.batch_size
        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = {"v_loss": [], "q_loss": [], "policy_loss": []}

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for _ in progress_bar:
                batch = buffer.sample(self.config.batch_size)
                metrics = self.train_step(batch)

                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

                progress_bar.set_postfix({
                    "v_loss": f"{metrics['v_loss']:.3f}",
                    "q_loss": f"{metrics['q_loss']:.3f}",
                })

            # Epoch summary
            avg_v_loss = np.mean(epoch_metrics["v_loss"])
            avg_q_loss = np.mean(epoch_metrics["q_loss"])
            print(f"Epoch {epoch+1} - V Loss: {avg_v_loss:.4f}, Q Loss: {avg_q_loss:.4f}")

            # Save best
            total_loss = avg_v_loss + avg_q_loss
            if total_loss < best_loss:
                best_loss = total_loss
                self.save(os.path.join(self.config.output_dir, "best_model.pt"))

            # Periodic save
            if (epoch + 1) % self.config.save_freq == 0:
                self.save(os.path.join(self.config.output_dir, f"model_epoch_{epoch+1}.pt"))

        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one IQL training step."""
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        # === Value function update (expectile regression) ===
        with torch.no_grad():
            q_target = self.q_target.min_q(observations, actions)

        v = self.v_network(observations)
        v_loss = self._expectile_loss(q_target - v)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # === Q-function update ===
        with torch.no_grad():
            next_v = self.v_network(next_observations)
            q_target_value = rewards.unsqueeze(-1) + self.config.discount_gamma * (1 - dones.unsqueeze(-1)) * next_v

        q1, q2 = self.q_network(observations, actions)
        q_loss = F.mse_loss(q1, q_target_value) + F.mse_loss(q2, q_target_value)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # === Policy update (advantage-weighted BC) ===
        with torch.no_grad():
            v_pred = self.v_network(observations)
            q_pred = self.q_target.min_q(observations, actions)
            advantage = q_pred - v_pred

            # Clip advantages for stability
            weight = torch.exp(advantage * self.temperature)
            weight = torch.clamp(weight, max=100.0)

        log_prob = self.policy.log_prob(observations, actions)
        policy_loss = -(weight * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # === Soft target update ===
        with torch.no_grad():
            for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_advantage": advantage.mean().item(),
            "mean_weight": weight.mean().item(),
        }

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric expectile loss.

        L_tau(u) = |tau - I(u < 0)| * u^2
        """
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * (diff ** 2)).mean()

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select action for evaluation."""
        return self.policy.get_action(obs, deterministic=deterministic)

    def save(self, path: str = None):
        """Save all networks."""
        if path is None:
            path = os.path.join(self.config.output_dir, "model.pt")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "v_network_state_dict": self.v_network.state_dict(),
            "q_network_state_dict": self.q_network.state_dict(),
            "config": self.config,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load all networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.v_network.load_state_dict(checkpoint["v_network_state_dict"])
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.q_target = copy.deepcopy(self.q_network)
        print(f"Loaded model from {path}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="IQL Offline Training")

    parser.add_argument("--dataset", type=str, default="hopper-medium-v2", help="D4RL dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--expectile", type=float, default=0.7, help="Expectile tau")
    parser.add_argument("--temperature", type=float, default=3.0, help="AWR temperature")
    parser.add_argument("--output_dir", type=str, default="./output/iql", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("IQL Offline Training")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = OfflineRLConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    trainer = IQLTrainer(
        obs_dim=11,
        action_dim=3,
        config=config,
        expectile=args.expectile,
        temperature=args.temperature,
    )

    # Load dataset
    from .base_trainer import create_dummy_dataset
    dataset = create_dummy_dataset(obs_dim=11, action_dim=3)

    buffer = OfflineReplayBuffer(obs_dim=11, action_dim=3, device=str(trainer.device))
    buffer.load_dataset(dataset)

    trainer.train(buffer)

    print("\nTraining complete!")
