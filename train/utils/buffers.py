"""
Experience Buffers for RL Training

Unified buffer implementations:
- RolloutBuffer: On-policy algorithms (PPO, A2C)
- ReplayBuffer: Off-policy algorithms (SAC, TD3)
- OfflineBuffer: Offline RL from static datasets
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Iterator, Tuple


class BaseBuffer(ABC):
    """Abstract base class for experience buffers."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.ptr = 0
        self.size = 0

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add transition to buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer."""
        pass

    def clear(self) -> None:
        """Clear buffer."""
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size


class RolloutBuffer(BaseBuffer):
    """
    Buffer for on-policy rollout data (PPO, A2C).

    Stores complete trajectories and computes GAE advantages.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        super().__init__(buffer_size, obs_dim, action_dim, device)

        # Storage
        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add transition to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
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
        adv = self.advantages[:self.size]
        self.advantages[:self.size] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._get_batch(indices)

    def get_batches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate batches over all data."""
        indices = np.random.permutation(self.size)

        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield self._get_batch(batch_indices)

    def _get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Get batch by indices."""
        return {
            "observations": self.observations[indices].to(self.device),
            "actions": self.actions[indices].to(self.device),
            "returns": self.returns[indices].to(self.device),
            "advantages": self.advantages[indices].to(self.device),
            "log_probs": self.log_probs[indices].to(self.device),
            "values": self.values[indices].to(self.device),
        }


class ReplayBuffer(BaseBuffer):
    """
    Experience replay buffer for off-policy algorithms (SAC, TD3).

    Stores (s, a, r, s', done) transitions for random sampling.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        super().__init__(buffer_size, obs_dim, action_dim, device)

        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.next_observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        """Add transition."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[indices].to(self.device),
            "actions": self.actions[indices].to(self.device),
            "rewards": self.rewards[indices].to(self.device),
            "next_observations": self.next_observations[indices].to(self.device),
            "dones": self.dones[indices].to(self.device),
        }


class OfflineBuffer(BaseBuffer):
    """
    Buffer for offline RL from static datasets.

    Supports loading full datasets and trajectory sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 1000000,
        device: str = "cpu",
    ):
        super().__init__(max_size, obs_dim, action_dim, device)

        # Use numpy for memory efficiency with large datasets
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.timesteps = np.zeros(max_size, dtype=np.int64)
        self.episode_starts = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        timestep: int = 0,
    ) -> None:
        """Add single transition."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.timesteps[self.ptr] = timestep

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        """Load entire dataset at once."""
        n = min(len(dataset["observations"]), self.buffer_size)

        self.observations[:n] = dataset["observations"][:n]
        self.actions[:n] = dataset["actions"][:n]
        self.rewards[:n] = dataset["rewards"][:n]
        self.next_observations[:n] = dataset["next_observations"][:n]
        self.dones[:n] = dataset["dones"][:n]

        if "timesteps" in dataset:
            self.timesteps[:n] = dataset["timesteps"][:n]

        self.size = n
        self._find_episode_starts()
        print(f"Loaded {n:,} transitions")

    def _find_episode_starts(self) -> None:
        """Find episode start indices."""
        self.episode_starts = [0]
        for i in range(self.size - 1):
            if self.dones[i]:
                self.episode_starts.append(i + 1)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions."""
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
        """Sample trajectory segments for sequence models."""
        batch = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "timesteps": [],
            "returns_to_go": [],
        }

        for _ in range(batch_size):
            start = np.random.randint(0, max(1, self.size - seq_len))
            end = min(start + seq_len, self.size)
            length = end - start

            obs = self.observations[start:end]
            actions = self.actions[start:end]
            rewards = self.rewards[start:end]
            dones = self.dones[start:end]
            timesteps = self.timesteps[start:end]
            rtg = self._compute_returns_to_go(rewards, dones)

            # Pad if needed
            if length < seq_len:
                pad = seq_len - length
                obs = np.pad(obs, ((0, pad), (0, 0)))
                actions = np.pad(actions, ((0, pad), (0, 0)))
                rewards = np.pad(rewards, (0, pad))
                dones = np.pad(dones, (0, pad), constant_values=1)
                timesteps = np.pad(timesteps, (0, pad))
                rtg = np.pad(rtg, (0, pad))

            batch["observations"].append(obs)
            batch["actions"].append(actions)
            batch["rewards"].append(rewards)
            batch["dones"].append(dones)
            batch["timesteps"].append(timesteps)
            batch["returns_to_go"].append(rtg)

        return {k: torch.tensor(np.array(v), device=self.device) for k, v in batch.items()}

    def _compute_returns_to_go(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Compute returns-to-go."""
        rtg = np.zeros_like(rewards)
        rtg[-1] = rewards[-1]

        for t in reversed(range(len(rewards) - 1)):
            rtg[t] = rewards[t] + (0 if dones[t] else gamma * rtg[t + 1])

        return rtg

    def normalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize observations, return mean/std."""
        obs = self.observations[:self.size]
        mean = obs.mean(axis=0)
        std = obs.std(axis=0) + 1e-6

        self.observations[:self.size] = (obs - mean) / std
        self.next_observations[:self.size] = (self.next_observations[:self.size] - mean) / std

        return mean, std

    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics."""
        return {
            "num_transitions": self.size,
            "num_episodes": len(self.episode_starts),
            "mean_reward": float(self.rewards[:self.size].mean()),
            "std_reward": float(self.rewards[:self.size].std()),
            "mean_episode_length": self.size / max(1, len(self.episode_starts)),
        }


if __name__ == "__main__":
    print("Testing buffers...")

    # Test RolloutBuffer
    rollout = RolloutBuffer(100, obs_dim=4, action_dim=2)
    for _ in range(50):
        rollout.add(
            obs=torch.randn(4),
            action=torch.randn(2),
            reward=np.random.randn(),
            done=np.random.rand() < 0.1,
            value=np.random.randn(),
            log_prob=np.random.randn(),
        )
    print(f"RolloutBuffer size: {len(rollout)}")

    # Test ReplayBuffer
    replay = ReplayBuffer(1000, obs_dim=4, action_dim=2)
    for _ in range(100):
        replay.add(
            obs=torch.randn(4),
            action=torch.randn(2),
            reward=np.random.randn(),
            next_obs=torch.randn(4),
            done=np.random.rand() < 0.1,
        )
    batch = replay.sample(32)
    print(f"ReplayBuffer sample shapes: {batch['observations'].shape}")

    # Test OfflineBuffer
    offline = OfflineBuffer(obs_dim=4, action_dim=2, max_size=1000)
    dataset = {
        "observations": np.random.randn(500, 4).astype(np.float32),
        "actions": np.random.randn(500, 2).astype(np.float32),
        "rewards": np.random.randn(500).astype(np.float32),
        "next_observations": np.random.randn(500, 4).astype(np.float32),
        "dones": (np.random.rand(500) < 0.05).astype(np.float32),
    }
    offline.load_dataset(dataset)
    print(f"OfflineBuffer stats: {offline.get_statistics()}")

    print("\nAll buffer tests passed!")
