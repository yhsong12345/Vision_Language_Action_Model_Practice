"""
Base Online RL Trainer

Abstract base class for online reinforcement learning trainers.
Provides common functionality for environment interaction and learning.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from collections import deque
import json


class RolloutBuffer:
    """Buffer for storing on-policy rollout data (PPO, A2C, etc.)."""

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device

        # Storage
        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute GAE returns and advantages."""
        last_gae = 0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        """Generate random batches."""
        indices = np.random.permutation(self.size)

        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                "observations": self.observations[batch_indices].to(self.device),
                "actions": self.actions[batch_indices].to(self.device),
                "returns": self.returns[batch_indices].to(self.device),
                "advantages": self.advantages[batch_indices].to(self.device),
                "log_probs": self.log_probs[batch_indices].to(self.device),
                "values": self.values[batch_indices].to(self.device),
            }

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms (SAC, TD3, etc.)."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.device = device

        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.next_observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ):
        """Add a transition."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[indices].to(self.device),
            "actions": self.actions[indices].to(self.device),
            "rewards": self.rewards[indices].to(self.device),
            "next_observations": self.next_observations[indices].to(self.device),
            "dones": self.dones[indices].to(self.device),
        }


class OnlineRLTrainer(ABC):
    """
    Abstract base class for online RL trainers.

    Provides common functionality:
    - Environment interaction
    - Logging
    - Checkpointing
    - Evaluation
    """

    def __init__(
        self,
        env,
        policy,
        output_dir: str = "./online_rl_output",
        device: str = "auto",
        seed: int = 42,
        gamma: float = 0.99,
        total_timesteps: int = 1000000,
        eval_episodes: int = 10,
        eval_freq: int = 10000,
        log_freq: int = 1000,
        save_freq: int = 50000,
    ):
        self.env = env
        self.policy = policy
        self.output_dir = output_dir
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.save_freq = save_freq

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy = self.policy.to(self.device)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        """Run training."""
        pass

    @abstractmethod
    def learn_step(self) -> Dict[str, float]:
        """Perform one learning step."""
        pass

    def collect_rollout(self, num_steps: int) -> Tuple[float, int]:
        """Collect experience by interacting with environment."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for _ in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                action, value, log_prob = self.policy.get_action_value(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            # Store transition
            self._store_transition(obs, action, reward, done, value, log_prob)

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

        return np.mean(self.episode_rewards), np.mean(self.episode_lengths)

    @abstractmethod
    def _store_transition(self, obs, action, reward, done, value, log_prob):
        """Store transition in buffer."""
        pass

    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """Evaluate the policy."""
        if num_episodes is None:
            num_episodes = self.eval_episodes

        self.policy.eval()
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    action = self.policy.get_action(obs_tensor, deterministic=True)

                action_np = action.cpu().numpy()
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        self.policy.train()

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
        }

    def save(self, path: str = None):
        """Save the policy."""
        if path is None:
            path = os.path.join(self.output_dir, "policy.pt")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "episode_rewards": list(self.episode_rewards),
        }, path)

        print(f"Saved policy to {path}")

    def load(self, path: str):
        """Load the policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from {path}")


class ActorCritic(nn.Module):
    """Simple Actor-Critic network for continuous control."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)
        mean = self.actor_mean(features)
        value = self.critic(features)
        return mean, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, _ = self.forward(obs)

        if deterministic:
            return mean

        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.sample()

    def get_action_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.actor_log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, value.squeeze(-1), log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.actor_log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return value.squeeze(-1), log_prob, entropy


if __name__ == "__main__":
    print("Online RL Base Trainer")
    print("Abstract class providing common online RL functionality")
