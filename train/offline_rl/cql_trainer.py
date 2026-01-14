"""
CQL (Conservative Q-Learning) Trainer - Offline RL

Implements CQL algorithm for offline reinforcement learning:
- Learns conservative Q-function that lower-bounds true Q
- Penalizes Q-values for out-of-distribution actions
- No environment interaction during training
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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class GaussianPolicy(nn.Module):
    """Gaussian policy for CQL."""

    LOG_STD_MIN = -20
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

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return torch.tanh(normal.sample())


class TwinQNetwork(nn.Module):
    """Twin Q-networks for CQL."""

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


class CQLTrainer(OfflineRLTrainer):
    """
    Conservative Q-Learning (CQL) Trainer.

    CQL learns a conservative Q-function by:
    1. Minimizing Q-values for OOD actions (sampled from policy)
    2. Maximizing Q-values for dataset actions
    3. Standard Bellman backup

    Reference: Kumar et al., "Conservative Q-Learning for Offline RL"
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[OfflineRLConfig] = None,
        cql_alpha: float = 5.0,
        cql_num_samples: int = 10,
        use_automatic_alpha: bool = True,
        target_action_gap: float = 10.0,
    ):
        if config is None:
            config = OfflineRLConfig()

        # Create policy
        policy = GaussianPolicy(obs_dim, action_dim, config.hidden_dim)

        super().__init__(config, policy)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # CQL specific
        self.cql_alpha = cql_alpha
        self.cql_num_samples = cql_num_samples
        self.use_automatic_alpha = use_automatic_alpha
        self.target_action_gap = target_action_gap

        # Automatic alpha
        if use_automatic_alpha:
            self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.cql_alpha_optimizer = Adam([self.log_cql_alpha], lr=3e-4)

        # Q-networks
        self.q_network = TwinQNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_network)

        for param in self.q_target.parameters():
            param.requires_grad = False

        # SAC temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = 0.2
        self.target_entropy = -action_dim

        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
        self.q_optimizer = Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.alpha_optimizer = Adam([self.log_alpha], lr=3e-4)

    def train(self, buffer: OfflineReplayBuffer):
        """Run CQL offline training."""
        print("=" * 60)
        print("CQL Offline Training")
        print("=" * 60)
        print(f"CQL Alpha: {self.cql_alpha}")
        print(f"Dataset size: {buffer.size}")

        num_batches = buffer.size // self.config.batch_size
        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = {"q_loss": [], "policy_loss": [], "cql_loss": []}

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for _ in progress_bar:
                batch = buffer.sample(self.config.batch_size)
                metrics = self.train_step(batch)

                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

                progress_bar.set_postfix({
                    "q_loss": f"{metrics['q_loss']:.3f}",
                    "cql": f"{metrics['cql_loss']:.3f}",
                })

            # Epoch summary
            avg_q_loss = np.mean(epoch_metrics["q_loss"])
            avg_cql_loss = np.mean(epoch_metrics["cql_loss"])
            print(f"Epoch {epoch+1} - Q Loss: {avg_q_loss:.4f}, CQL Loss: {avg_cql_loss:.4f}")

            # Save best
            if avg_q_loss < best_loss:
                best_loss = avg_q_loss
                self.save(os.path.join(self.config.output_dir, "best_model.pt"))

            # Periodic save
            if (epoch + 1) % self.config.save_freq == 0:
                self.save(os.path.join(self.config.output_dir, f"model_epoch_{epoch+1}.pt"))

        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one CQL training step."""
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        # Get current CQL alpha
        if self.use_automatic_alpha:
            cql_alpha = self.log_cql_alpha.exp().item()
        else:
            cql_alpha = self.cql_alpha

        # === Q-function update ===
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_observations)
            q1_next, q2_next = self.q_target(next_observations, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards.unsqueeze(-1) + self.config.discount_gamma * (1 - dones.unsqueeze(-1)) * q_next

        q1, q2 = self.q_network(observations, actions)
        bellman_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # CQL loss: penalize high Q-values for OOD actions
        cql_loss = self._compute_cql_loss(observations, actions, q1, q2)

        q_loss = bellman_loss + cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # === Policy update ===
        actions_new, log_probs = self.policy.sample(observations)
        q1_new, q2_new = self.q_network(observations, actions_new)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # === Temperature update ===
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # === CQL alpha update ===
        if self.use_automatic_alpha:
            cql_alpha_loss = self.log_cql_alpha * (cql_loss.detach() - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()

        # === Soft target update ===
        with torch.no_grad():
            for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )

        return {
            "q_loss": q_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
            "cql_alpha": cql_alpha,
        }

    def _compute_cql_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CQL regularization loss.

        CQL loss = E[log(sum_a exp(Q(s,a)))] - E[Q(s, a_data)]
        """
        batch_size = observations.shape[0]

        # Sample random actions
        random_actions = torch.FloatTensor(
            batch_size * self.cql_num_samples, self.action_dim
        ).uniform_(-1, 1).to(self.device)

        # Sample actions from current policy
        obs_repeated = observations.unsqueeze(1).repeat(1, self.cql_num_samples, 1)
        obs_repeated = obs_repeated.view(-1, self.obs_dim)

        with torch.no_grad():
            policy_actions, policy_log_probs = self.policy.sample(obs_repeated)

        # Compute Q-values for random actions
        q1_random, q2_random = self.q_network(obs_repeated, random_actions)
        q1_random = q1_random.view(batch_size, self.cql_num_samples)
        q2_random = q2_random.view(batch_size, self.cql_num_samples)

        # Compute Q-values for policy actions
        q1_policy, q2_policy = self.q_network(obs_repeated, policy_actions)
        q1_policy = q1_policy.view(batch_size, self.cql_num_samples)
        q2_policy = q2_policy.view(batch_size, self.cql_num_samples)

        # Log-sum-exp for OOD actions
        random_density = np.log(0.5 ** self.action_dim)
        policy_log_probs = policy_log_probs.view(batch_size, self.cql_num_samples)

        q1_cat = torch.cat([q1_random - random_density, q1_policy - policy_log_probs], dim=1)
        q2_cat = torch.cat([q2_random - random_density, q2_policy - policy_log_probs], dim=1)

        cql_q1 = torch.logsumexp(q1_cat, dim=1).mean() - q1.mean()
        cql_q2 = torch.logsumexp(q2_cat, dim=1).mean() - q2.mean()

        return cql_q1 + cql_q2

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select action for evaluation."""
        return self.policy.get_action(obs, deterministic=deterministic)


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="CQL Offline Training")

    parser.add_argument("--dataset", type=str, default="hopper-medium-v2", help="D4RL dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cql_alpha", type=float, default=5.0, help="CQL alpha")
    parser.add_argument("--output_dir", type=str, default="./output/cql", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("CQL Offline Training")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = OfflineRLConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    # Create trainer
    trainer = CQLTrainer(
        obs_dim=11,  # Hopper
        action_dim=3,
        config=config,
        cql_alpha=args.cql_alpha,
    )

    # Load dataset
    from .base_trainer import create_dummy_dataset
    dataset = create_dummy_dataset(obs_dim=11, action_dim=3)

    buffer = OfflineReplayBuffer(obs_dim=11, action_dim=3, device=str(trainer.device))
    buffer.load_dataset(dataset)

    # Train
    trainer.train(buffer)

    print("\nTraining complete!")
