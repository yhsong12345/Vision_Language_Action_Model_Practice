"""
Base Offline RL Trainer

Abstract base class for offline reinforcement learning trainers.
Provides common functionality for learning from static datasets.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
from collections import deque
import json
from dataclasses import dataclass


@dataclass
class OfflineRLConfig:
    """Configuration for offline RL training."""
    # Data
    dataset_path: str = ""
    dataset_name: str = "d4rl_hopper_medium"

    # Training
    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Model
    hidden_dim: int = 256
    num_layers: int = 3

    # Algorithm specific
    discount_gamma: float = 0.99
    tau: float = 0.005  # Soft update

    # Output
    output_dir: str = "./offline_rl_output"
    eval_freq: int = 10
    save_freq: int = 50
    log_freq: int = 1

    # Seed
    seed: int = 42


class OfflineReplayBuffer:
    """
    Replay buffer for offline RL.

    Loads data from static dataset and provides batched sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 1000000,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = device

        # Storage
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

        # For sequence-based methods
        self.timesteps = np.zeros(max_size, dtype=np.int64)
        self.episode_starts = []

        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        timestep: int = 0,
    ):
        """Add single transition."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.timesteps[self.ptr] = timestep

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def load_dataset(self, dataset: Dict[str, np.ndarray]):
        """
        Load entire dataset at once.

        Args:
            dataset: Dict with observations, actions, rewards, next_observations, dones
        """
        n_samples = len(dataset["observations"])
        n_samples = min(n_samples, self.max_size)

        self.observations[:n_samples] = dataset["observations"][:n_samples]
        self.actions[:n_samples] = dataset["actions"][:n_samples]
        self.rewards[:n_samples] = dataset["rewards"][:n_samples]
        self.next_observations[:n_samples] = dataset["next_observations"][:n_samples]
        self.dones[:n_samples] = dataset["dones"][:n_samples]

        if "timesteps" in dataset:
            self.timesteps[:n_samples] = dataset["timesteps"][:n_samples]

        self.size = n_samples
        print(f"Loaded {n_samples} transitions")

        # Find episode boundaries
        self._find_episode_starts()

    def _find_episode_starts(self):
        """Find episode start indices."""
        self.episode_starts = [0]
        for i in range(self.size - 1):
            if self.dones[i]:
                self.episode_starts.append(i + 1)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": torch.tensor(self.observations[indices], device=self.device),
            "actions": torch.tensor(self.actions[indices], device=self.device),
            "rewards": torch.tensor(self.rewards[indices], device=self.device),
            "next_observations": torch.tensor(self.next_observations[indices], device=self.device),
            "dones": torch.tensor(self.dones[indices], device=self.device),
        }

    def sample_trajectories(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample trajectory segments for sequence models.

        Args:
            batch_size: Number of trajectories
            seq_len: Length of each trajectory segment

        Returns:
            Dict with batched trajectories
        """
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_timesteps = []
        batch_returns_to_go = []

        for _ in range(batch_size):
            # Sample random start point
            start_idx = np.random.randint(0, max(1, self.size - seq_len))

            # Get trajectory segment
            end_idx = min(start_idx + seq_len, self.size)
            actual_len = end_idx - start_idx

            obs = self.observations[start_idx:end_idx]
            actions = self.actions[start_idx:end_idx]
            rewards = self.rewards[start_idx:end_idx]
            dones = self.dones[start_idx:end_idx]
            timesteps = self.timesteps[start_idx:end_idx]

            # Compute returns-to-go
            rtg = self._compute_returns_to_go(rewards, dones)

            # Pad if necessary
            if actual_len < seq_len:
                pad_len = seq_len - actual_len
                obs = np.pad(obs, ((0, pad_len), (0, 0)))
                actions = np.pad(actions, ((0, pad_len), (0, 0)))
                rewards = np.pad(rewards, (0, pad_len))
                dones = np.pad(dones, (0, pad_len), constant_values=1)
                timesteps = np.pad(timesteps, (0, pad_len))
                rtg = np.pad(rtg, (0, pad_len))

            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            batch_timesteps.append(timesteps)
            batch_returns_to_go.append(rtg)

        return {
            "observations": torch.tensor(np.array(batch_obs), device=self.device),
            "actions": torch.tensor(np.array(batch_actions), device=self.device),
            "rewards": torch.tensor(np.array(batch_rewards), device=self.device),
            "dones": torch.tensor(np.array(batch_dones), device=self.device),
            "timesteps": torch.tensor(np.array(batch_timesteps), device=self.device),
            "returns_to_go": torch.tensor(np.array(batch_returns_to_go), device=self.device),
        }

    def _compute_returns_to_go(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Compute returns-to-go for each timestep."""
        rtg = np.zeros_like(rewards)
        rtg[-1] = rewards[-1]

        for t in reversed(range(len(rewards) - 1)):
            if dones[t]:
                rtg[t] = rewards[t]
            else:
                rtg[t] = rewards[t] + gamma * rtg[t + 1]

        return rtg

    def normalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize observations and return mean/std."""
        obs_mean = self.observations[:self.size].mean(axis=0)
        obs_std = self.observations[:self.size].std(axis=0) + 1e-6

        self.observations[:self.size] = (self.observations[:self.size] - obs_mean) / obs_std
        self.next_observations[:self.size] = (self.next_observations[:self.size] - obs_mean) / obs_std

        return obs_mean, obs_std

    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics."""
        return {
            "num_transitions": self.size,
            "num_episodes": len(self.episode_starts),
            "mean_reward": self.rewards[:self.size].mean(),
            "std_reward": self.rewards[:self.size].std(),
            "mean_episode_length": self.size / max(1, len(self.episode_starts)),
        }


class OfflineRLTrainer(ABC):
    """
    Abstract base class for offline RL trainers.

    Provides common functionality:
    - Dataset loading
    - Training loop
    - Logging
    - Checkpointing
    - Evaluation
    """

    def __init__(
        self,
        config: OfflineRLConfig,
        policy: nn.Module,
        device: str = "auto",
    ):
        self.config = config

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy = policy.to(self.device)

        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Logging
        self.train_metrics = []

        os.makedirs(config.output_dir, exist_ok=True)

    @abstractmethod
    def train(self, buffer: OfflineReplayBuffer):
        """Run offline training."""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        pass

    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate policy in environment.

        Args:
            env: Gymnasium environment
            num_episodes: Number of evaluation episodes

        Returns:
            Dict with evaluation metrics
        """
        self.policy.eval()
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action = self.select_action(obs_tensor, deterministic=True)

                action_np = action.cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = env.step(action_np)
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
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }

    @abstractmethod
    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select action given observation."""
        pass

    def save(self, path: str = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(self.config.output_dir, "model.pt")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "config": self.config,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded model from {path}")


def load_d4rl_dataset(env_name: str) -> Dict[str, np.ndarray]:
    """
    Load D4RL dataset.

    Args:
        env_name: D4RL environment name (e.g., "hopper-medium-v2")

    Returns:
        Dataset dictionary
    """
    try:
        import d4rl
        import gymnasium as gym

        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)

        return {
            "observations": dataset["observations"],
            "actions": dataset["actions"],
            "rewards": dataset["rewards"],
            "next_observations": dataset["next_observations"],
            "dones": dataset["terminals"].astype(np.float32),
        }
    except ImportError:
        print("D4RL not installed. Creating dummy dataset.")
        return create_dummy_dataset()


def create_dummy_dataset(
    num_samples: int = 10000,
    obs_dim: int = 11,
    action_dim: int = 3,
) -> Dict[str, np.ndarray]:
    """Create dummy dataset for testing."""
    return {
        "observations": np.random.randn(num_samples, obs_dim).astype(np.float32),
        "actions": np.random.randn(num_samples, action_dim).astype(np.float32),
        "rewards": np.random.randn(num_samples).astype(np.float32),
        "next_observations": np.random.randn(num_samples, obs_dim).astype(np.float32),
        "dones": (np.random.rand(num_samples) < 0.01).astype(np.float32),
    }


if __name__ == "__main__":
    print("Offline RL Base Trainer")
    print("Abstract class providing common offline RL functionality")

    # Test buffer
    buffer = OfflineReplayBuffer(obs_dim=11, action_dim=3)
    dataset = create_dummy_dataset()
    buffer.load_dataset(dataset)

    print(f"\nBuffer statistics: {buffer.get_statistics()}")

    batch = buffer.sample(64)
    print(f"Sample batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
