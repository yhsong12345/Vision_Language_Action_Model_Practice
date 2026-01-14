"""
Evaluation Utilities

Shared evaluation functions for IL and RL trainers:
- Policy evaluation in environments
- Metrics computation
- Standardized evaluation loops
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Callable, Any, Union
from collections import defaultdict


def evaluate_policy(
    policy: nn.Module,
    env,
    num_episodes: int = 10,
    max_steps: int = 1000,
    deterministic: bool = True,
    device: Union[str, torch.device] = "cpu",
    render: bool = False,
) -> Dict[str, float]:
    """
    Evaluate policy in environment.

    Args:
        policy: Policy with get_action method
        env: Gymnasium environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        deterministic: Use deterministic actions
        device: Device for inference
        render: Whether to render environment

    Returns:
        Dict with evaluation metrics
    """
    if isinstance(device, str):
        device = torch.device(device)

    policy.eval()
    episode_rewards = []
    episode_lengths = []
    successes = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            # Get action
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action = policy.get_action(obs_tensor, deterministic=deterministic)

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check for success (if available)
        if "success" in info:
            successes.append(info["success"])
        elif "is_success" in info:
            successes.append(info["is_success"])

    policy.train()

    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    if successes:
        metrics["success_rate"] = float(np.mean(successes))

    return metrics


def evaluate_in_env(
    get_action_fn: Callable[[np.ndarray], np.ndarray],
    env,
    num_episodes: int = 10,
    max_steps: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate using a custom action function.

    Args:
        get_action_fn: Function that takes obs and returns action
        env: Gymnasium environment
        num_episodes: Number of episodes
        max_steps: Max steps per episode

    Returns:
        Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            action = get_action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
    }


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute common prediction metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth
        masks: Optional mask for valid entries

    Returns:
        Dict with MSE, MAE, etc.
    """
    if masks is not None:
        predictions = predictions[masks]
        targets = targets[masks]

    mse = ((predictions - targets) ** 2).mean().item()
    mae = (predictions - targets).abs().mean().item()
    rmse = np.sqrt(mse)

    # Per-dimension metrics
    if predictions.dim() > 1:
        per_dim_mse = ((predictions - targets) ** 2).mean(dim=0)
        per_dim_mae = (predictions - targets).abs().mean(dim=0)
    else:
        per_dim_mse = torch.tensor([mse])
        per_dim_mae = torch.tensor([mae])

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "per_dim_mse": per_dim_mse.tolist(),
        "per_dim_mae": per_dim_mae.tolist(),
    }


def compute_action_metrics(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    action_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute action-specific metrics.

    Args:
        pred_actions: Predicted actions (batch, action_dim)
        gt_actions: Ground truth actions (batch, action_dim)
        action_names: Optional names for each action dimension

    Returns:
        Dict with action metrics
    """
    metrics = compute_metrics(pred_actions, gt_actions)

    action_dim = pred_actions.shape[-1]
    if action_names is None:
        action_names = [f"action_{i}" for i in range(action_dim)]

    # Per-action dimension errors
    for i, name in enumerate(action_names):
        dim_mse = ((pred_actions[:, i] - gt_actions[:, i]) ** 2).mean().item()
        metrics[f"{name}_mse"] = dim_mse

    return metrics


class EpisodeStats:
    """Track episode statistics during training."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []

    def add_episode(
        self,
        reward: float,
        length: int,
        success: Optional[bool] = None,
    ) -> None:
        """Add episode result."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        if success is not None:
            self.successes.append(success)

        # Keep window size
        if len(self.episode_rewards) > self.window_size:
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            if self.successes:
                self.successes.pop(0)

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        if not self.episode_rewards:
            return {}

        stats = {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "num_episodes": len(self.episode_rewards),
        }

        if self.successes:
            stats["success_rate"] = float(np.mean(self.successes))

        return stats


if __name__ == "__main__":
    print("Testing evaluation utilities...")

    # Test compute_metrics
    pred = torch.randn(100, 7)
    target = torch.randn(100, 7)
    metrics = compute_metrics(pred, target)
    print(f"Metrics: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    # Test EpisodeStats
    stats = EpisodeStats(window_size=10)
    for i in range(20):
        stats.add_episode(reward=np.random.randn(), length=100 + i, success=np.random.rand() > 0.5)

    print(f"Episode stats: {stats.get_stats()}")

    print("\nEvaluation utilities test passed!")
