"""
SAC (Soft Actor-Critic) Trainer

Implements SAC algorithm for continuous robot control:
- Maximum entropy RL
- Twin Q-networks
- Automatic temperature tuning
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

from .base_trainer import RLTrainer, ReplayBuffer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import RLConfig


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous action spaces."""

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

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        # Log prob with tanh squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = torch.tanh(normal.sample())
        return action


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2
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


class SACTrainer(RLTrainer):
    """
    SAC Trainer for continuous robot control.

    Features:
    - Maximum entropy RL for exploration
    - Twin Q-networks to reduce overestimation
    - Automatic temperature (alpha) tuning
    - Soft target updates
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        config: Optional[RLConfig] = None,
        **kwargs,
    ):
        # Default config
        if config is None:
            config = RLConfig.sac_default()

        # Get dimensions
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Create default policy if not provided
        if policy is None:
            policy = GaussianPolicy(obs_dim, action_dim)

        super().__init__(
            env=env,
            policy=policy,
            output_dir=config.output_dir,
            gamma=config.discount_gamma,
            total_timesteps=config.total_timesteps,
            **kwargs,
        )

        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # SAC specific params
        self.tau = config.sac_tau
        self.buffer_size = config.sac_buffer_size
        self.learning_starts = config.sac_learning_starts
        self.batch_size = config.batch_size

        # Create Q-networks
        self.q_network = TwinQNetwork(obs_dim, action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_network)

        # Freeze target network
        for param in self.q_target.parameters():
            param.requires_grad = False

        # Automatic temperature
        if config.sac_target_entropy == "auto":
            self.target_entropy = -action_dim
        else:
            self.target_entropy = float(config.sac_target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = config.sac_init_temperature

        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
        self.q_optimizer = Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.alpha_optimizer = Adam([self.log_alpha], lr=config.learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=str(self.device),
        )

    def train(self):
        """Run SAC training."""
        print("=" * 60)
        print("SAC Training")
        print("=" * 60)

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        best_reward = float("-inf")

        progress_bar = tqdm(total=self.total_timesteps, desc="Training")

        for timestep in range(self.total_timesteps):
            # Select action
            if timestep < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    action = self.policy.get_action(obs_tensor).cpu().numpy()

            # Environment step
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                obs=torch.tensor(obs, dtype=torch.float32),
                action=torch.tensor(action, dtype=torch.float32),
                reward=reward,
                next_obs=torch.tensor(next_obs, dtype=torch.float32),
                done=done,
            )

            episode_reward += reward
            episode_length += 1

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs

            # Training
            if timestep >= self.learning_starts:
                metrics = self.learn_step()

                if timestep % self.log_freq == 0:
                    progress_bar.set_postfix({
                        "reward": f"{np.mean(list(self.episode_rewards)):.1f}" if self.episode_rewards else "N/A",
                        "q_loss": f"{metrics['q_loss']:.3f}",
                        "alpha": f"{self.alpha:.3f}",
                    })

            # Evaluation
            if timestep % self.eval_freq == 0 and timestep > 0:
                eval_results = self.evaluate()
                print(f"\nStep {timestep}: Mean Reward = {eval_results['mean_reward']:.2f}")

                if eval_results["mean_reward"] > best_reward:
                    best_reward = eval_results["mean_reward"]
                    self.save(os.path.join(self.output_dir, "best_policy.pt"))

            # Save checkpoint
            if timestep % self.save_freq == 0 and timestep > 0:
                self.save(os.path.join(self.output_dir, f"policy_{timestep}.pt"))

            progress_bar.update(1)

        progress_bar.close()
        self.save()

    def learn_step(self) -> Dict[str, float]:
        """Perform one SAC update."""
        batch = self.buffer.sample(self.batch_size)

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_observations)
            q1_next, q2_next = self.q_target(next_observations, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * q_next

        q1, q2 = self.q_network(observations, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        actions_new, log_probs = self.policy.sample(observations)
        q1_new, q2_new = self.q_network(observations, actions_new)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        with torch.no_grad():
            for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }

    def _store_transition(self, obs, action, reward, done, value, log_prob):
        """Not used - SAC stores transitions directly."""
        pass


def train_sac(
    env_name: str = "Pendulum-v1",
    total_timesteps: int = 100000,
    **kwargs,
):
    """
    Convenience function for SAC training.

    Args:
        env_name: Gymnasium environment name (continuous action space)
        total_timesteps: Total training timesteps
        **kwargs: Additional config arguments
    """
    import gymnasium as gym

    env = gym.make(env_name)
    config = RLConfig(
        algorithm="sac",
        total_timesteps=total_timesteps,
        **kwargs,
    )

    trainer = SACTrainer(env, config=config)
    trainer.train()

    return trainer.policy


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="SAC Training")

    # Environment
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gymnasium environment (continuous)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # SAC parameters
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--sac_tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--sac_buffer_size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--sac_learning_starts", type=int, default=1000, help="Steps before learning")
    parser.add_argument("--sac_target_entropy", type=str, default="auto", help="Target entropy")
    parser.add_argument("--sac_init_temperature", type=float, default=0.2, help="Initial temperature")

    # Training
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--discount_gamma", type=float, default=0.99, help="Discount factor")

    # Logging
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=10000, help="Checkpoint frequency")
    parser.add_argument("--log_freq", type=int, default=1000, help="Logging frequency")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/sac", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("SAC Training")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import gymnasium as gym
    env = gym.make(args.env)

    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")

    config = RLConfig(
        algorithm="sac",
        total_timesteps=args.total_timesteps,
        sac_tau=args.sac_tau,
        sac_buffer_size=args.sac_buffer_size,
        sac_learning_starts=args.sac_learning_starts,
        sac_target_entropy=args.sac_target_entropy,
        sac_init_temperature=args.sac_init_temperature,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        discount_gamma=args.discount_gamma,
        output_dir=args.output_dir,
    )

    trainer = SACTrainer(env, config=config)

    if args.resume:
        trainer.load(args.resume)

    trainer.train()

    print("\nTraining complete!")
