"""
Base Imitation Learning Trainer

Abstract base class for imitation learning trainers.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque

# Import shared utilities
from train.utils import get_device, evaluate_policy as _evaluate_policy


class ExpertDataset(Dataset):
    """Dataset for expert demonstrations."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class PolicyNetwork(nn.Module):
    """Simple MLP policy for imitation learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        continuous: bool = True,
    ):
        super().__init__()

        self.continuous = continuous

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        if continuous:
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)

        if self.continuous:
            return self.mean_head(features)
        else:
            return self.action_head(features)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            if self.continuous:
                mean = self.forward(state)
                if deterministic:
                    action = mean
                else:
                    std = self.log_std.exp()
                    action = mean + torch.randn_like(mean) * std
            else:
                logits = self.forward(state)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze(-1)

        return action.squeeze(0)

    def get_action_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probability of action given state."""
        if self.continuous:
            mean = self.forward(state)
            std = self.log_std.exp()

            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            return log_prob
        else:
            logits = self.forward(state)
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)


class ILTrainer(ABC):
    """
    Abstract base class for Imitation Learning trainers.

    Provides common functionality:
    - Expert demonstration handling
    - Policy training
    - Evaluation
    - Logging
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        output_dir: str = "./il_output",
        device: str = "auto",
        seed: int = 42,
    ):
        self.env = env
        self.output_dir = output_dir

        # Get dimensions
        self.state_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'shape'):
            self.action_dim = env.action_space.shape[0]
            self.continuous = True
        else:
            self.action_dim = env.action_space.n
            self.continuous = False

        # Create policy if not provided
        if policy is None:
            policy = PolicyNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                continuous=self.continuous,
            )

        self.policy = policy

        # Set device using shared utility
        self.device = get_device(device)
        self.policy = self.policy.to(self.device)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        """Run training."""
        pass

    def collect_expert_demonstrations(
        self,
        expert_policy,
        num_episodes: int = 100,
        max_steps: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect demonstrations from an expert policy.

        Args:
            expert_policy: Expert policy with get_action method
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (states, actions) numpy arrays
        """
        states = []
        actions = []
        episode_rewards = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = expert_policy(state)

                states.append(state)
                actions.append(action)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

                state = next_state

            episode_rewards.append(episode_reward)

        print(f"Collected {len(states)} transitions from {num_episodes} episodes")
        print(f"Expert mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32 if self.continuous else np.int64)

        return states, actions

    def evaluate(
        self,
        num_episodes: int = 20,
        max_steps: int = 1000,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the learned policy.

        Args:
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic actions

        Returns:
            Evaluation metrics
        """
        self.policy.eval()
        episode_rewards = []
        episode_lengths = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                action = self.policy.get_action(state_tensor, deterministic=deterministic)

                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()

                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

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

    def save(self, path: str = None):
        """Save the policy."""
        if path is None:
            path = os.path.join(self.output_dir, "policy.pt")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "continuous": self.continuous,
        }, path)

        print(f"Saved policy to {path}")

    def load(self, path: str):
        """Load a saved policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from {path}")


if __name__ == "__main__":
    print("IL Base Trainer")
    print("Abstract class for imitation learning")
