"""
Policy Network Architectures

Shared policy implementations for IL and RL trainers:
- MLPPolicy: Simple deterministic policy
- GaussianMLPPolicy: Stochastic policy with Gaussian output
- ActorCritic: Combined actor-critic for PPO/A2C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Union


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
    output_activation: bool = False,
) -> nn.Sequential:
    """
    Build MLP network with configurable architecture.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ("relu", "gelu", "tanh", "silu")
        dropout: Dropout probability
        output_activation: Whether to apply activation on output

    Returns:
        nn.Sequential MLP
    """
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    act_cls = activations.get(activation, nn.ReLU)

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation:
        layers.append(act_cls())

    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    """
    Simple MLP policy for imitation learning.

    Supports both continuous and discrete action spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.1,
        continuous: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous

        self.network = build_mlp(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action or logits."""
        return self.network(obs)

    def get_action(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Get action for inference."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            output = self.forward(obs)

            if self.continuous:
                action = output
            else:
                if deterministic:
                    action = torch.argmax(output, dim=-1)
                else:
                    probs = F.softmax(output, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze(-1)

        return action.squeeze(0)


class GaussianMLPPolicy(nn.Module):
    """
    Gaussian MLP policy for stochastic continuous control.

    Outputs mean and log_std for a diagonal Gaussian distribution.
    Used for PPO, SAC, and other policy gradient methods.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        state_dependent_std: bool = False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.state_dependent_std = state_dependent_std

        # Build backbone
        self.backbone = build_mlp(
            input_dim=obs_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            output_activation=True,
        )

        # Mean head
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)

        # Log std (state-dependent or learned parameter)
        if state_dependent_std:
            self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action: Sampled or mean action
            log_prob: Log probability of action
            mean: Mean of distribution
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            noise = torch.randn_like(mean)
            action = mean + std * noise
            log_prob = self._log_prob(action, mean, std)

        return action, log_prob, mean

    def _log_prob(
        self,
        action: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of action."""
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var
            + 2 * torch.log(std)
            + np.log(2 * np.pi)
        )
        return log_prob.sum(dim=-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log prob and entropy for given actions.

        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of distribution
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)

        log_prob = self._log_prob(actions, mean, std)
        entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_std).sum(dim=-1)

        return log_prob, entropy

    def get_action(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action for inference."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            action, _, _ = self.forward(obs, deterministic=deterministic)

        return action.squeeze(0)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO/A2C.

    Shares feature extractor between actor and critic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Shared feature extractor
        self.feature_extractor = build_mlp(
            input_dim=obs_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            output_activation=True,
        )

        # Actor heads
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean action and value."""
        features = self.feature_extractor(obs)
        mean = self.actor_mean(features)
        value = self.critic(features)
        return mean, value

    def get_action(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action for inference."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            mean, _ = self.forward(obs)

            if deterministic:
                action = mean
            else:
                std = torch.exp(self.actor_log_std)
                action = mean + torch.randn_like(mean) * std

        return action.squeeze(0)

    def get_action_value(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, value, and log prob for rollout collection."""
        mean, value = self.forward(obs)
        log_std = torch.clamp(self.actor_log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, value.squeeze(-1), log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        mean, value = self.forward(obs)
        log_std = torch.clamp(self.actor_log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return value.squeeze(-1), log_prob, entropy


if __name__ == "__main__":
    print("Testing policy networks...")

    obs_dim = 11
    action_dim = 3
    batch_size = 32

    obs = torch.randn(batch_size, obs_dim)

    # Test MLPPolicy
    mlp = MLPPolicy(obs_dim, action_dim)
    action = mlp.get_action(obs[0])
    print(f"MLPPolicy action shape: {action.shape}")

    # Test GaussianMLPPolicy
    gaussian = GaussianMLPPolicy(obs_dim, action_dim)
    action, log_prob, mean = gaussian(obs)
    print(f"GaussianMLPPolicy action: {action.shape}, log_prob: {log_prob.shape}")

    # Test ActorCritic
    ac = ActorCritic(obs_dim, action_dim)
    action, value, log_prob = ac.get_action_value(obs)
    print(f"ActorCritic action: {action.shape}, value: {value.shape}")

    value, log_prob, entropy = ac.evaluate_actions(obs, action)
    print(f"ActorCritic evaluate: value={value.shape}, entropy={entropy.shape}")

    print("\nAll policy tests passed!")
