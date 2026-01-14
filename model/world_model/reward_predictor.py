"""
Reward Predictor Module

Predicts rewards and values from states/actions.
Key components:
- RewardPredictor: Predicts immediate reward
- CriticNetwork: Twin Q-network for value estimation
- ValuePredictor: State value function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import copy


@dataclass
class RewardPredictorConfig:
    """Configuration for reward predictor."""
    state_dim: int = 768
    action_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    use_layer_norm: bool = True
    output_activation: str = "none"  # "none", "tanh", "sigmoid"


class RewardPredictor(nn.Module):
    """
    Predicts immediate reward from state and action.

    Can be used for:
    - Model-based planning reward estimation
    - Intrinsic reward prediction
    - Reward shaping
    """

    def __init__(self, config: RewardPredictorConfig):
        super().__init__()
        self.config = config

        # Build network
        layers = []
        input_dim = config.state_dim + config.action_dim

        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else 1
            layers.append(nn.Linear(input_dim if i == 0 else config.hidden_dim, out_dim))

            if i < config.num_layers - 1:
                if config.use_layer_norm:
                    layers.append(nn.LayerNorm(config.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))

        self.network = nn.Sequential(*layers)

        # Output activation
        if config.output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif config.output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict reward.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)

        Returns:
            reward: (batch_size,)
        """
        x = torch.cat([state, action], dim=-1)
        reward = self.network(x).squeeze(-1)
        return self.output_activation(reward)

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward prediction loss."""
        pred = self.forward(state, action)
        return F.mse_loss(pred, target_reward)


class CriticNetwork(nn.Module):
    """
    Twin Q-network for SAC-style critic.

    Estimates Q(s, a) using two networks to reduce overestimation.
    """

    def __init__(self, config: RewardPredictorConfig):
        super().__init__()
        self.config = config

        # Q1 network
        self.q1 = self._build_network()

        # Q2 network
        self.q2 = self._build_network()

    def _build_network(self) -> nn.Module:
        layers = []
        input_dim = self.config.state_dim + self.config.action_dim

        for i in range(self.config.num_layers):
            out_dim = self.config.hidden_dim if i < self.config.num_layers - 1 else 1
            layers.append(nn.Linear(
                input_dim if i == 0 else self.config.hidden_dim, out_dim
            ))

            if i < self.config.num_layers - 1:
                if self.config.use_layer_norm:
                    layers.append(nn.LayerNorm(self.config.hidden_dim))
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values.

        Returns:
            q1, q2: (batch_size,) Q-values from both networks
        """
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return q1, q2

    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q1 value only."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1)

    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum of Q1 and Q2."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ValuePredictor(nn.Module):
    """
    State value function V(s).

    Estimates expected return from a state.
    """

    def __init__(self, config: RewardPredictorConfig):
        super().__init__()
        self.config = config

        layers = []
        for i in range(config.num_layers):
            in_dim = config.state_dim if i == 0 else config.hidden_dim
            out_dim = config.hidden_dim if i < config.num_layers - 1 else 1

            layers.append(nn.Linear(in_dim, out_dim))

            if i < config.num_layers - 1:
                if config.use_layer_norm:
                    layers.append(nn.LayerNorm(config.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Args:
            state: (batch_size, state_dim)

        Returns:
            value: (batch_size,)
        """
        return self.network(state).squeeze(-1)


class TargetNetwork:
    """
    Target network wrapper for soft updates.

    Maintains a slowly-updating copy of a network for stable training.
    """

    def __init__(self, network: nn.Module, tau: float = 0.005):
        self.network = network
        self.target = copy.deepcopy(network)
        self.tau = tau

        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

    def soft_update(self):
        """Soft update target network."""
        with torch.no_grad():
            for param, target_param in zip(
                self.network.parameters(),
                self.target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def hard_update(self):
        """Hard update (copy) target network."""
        self.target.load_state_dict(self.network.state_dict())

    def __call__(self, *args, **kwargs):
        """Forward through target network."""
        return self.target(*args, **kwargs)


class EnsembleValuePredictor(nn.Module):
    """
    Ensemble of value predictors for uncertainty estimation.

    Useful for:
    - Exploration bonus (high uncertainty = explore)
    - Conservative value estimation
    """

    def __init__(
        self,
        config: RewardPredictorConfig,
        num_ensemble: int = 5,
    ):
        super().__init__()
        self.models = nn.ModuleList([
            ValuePredictor(config) for _ in range(num_ensemble)
        ])

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ensemble value estimates.

        Returns:
            Dict with mean, std, min, max values
        """
        values = torch.stack([m(state) for m in self.models], dim=0)

        return {
            "mean": values.mean(0),
            "std": values.std(0),
            "min": values.min(0).values,
            "max": values.max(0).values,
            "all": values,
        }

    def conservative_value(
        self,
        state: torch.Tensor,
        penalty_coef: float = 1.0,
    ) -> torch.Tensor:
        """
        Get conservative value estimate (mean - std).

        Useful for offline RL to avoid overestimation.
        """
        outputs = self.forward(state)
        return outputs["mean"] - penalty_coef * outputs["std"]


class CostPredictor(nn.Module):
    """
    Predicts constraint costs for safe RL.

    Estimates expected constraint violations from state-action pairs.
    """

    def __init__(
        self,
        config: RewardPredictorConfig,
        num_constraints: int = 1,
    ):
        super().__init__()
        self.config = config
        self.num_constraints = num_constraints

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )

        # Per-constraint heads
        self.cost_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid(),  # Cost probability
            )
            for _ in range(num_constraints)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict constraint costs.

        Returns:
            costs: (batch_size, num_constraints) cost probabilities
        """
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)

        costs = [head(features) for head in self.cost_heads]
        return torch.cat(costs, dim=-1)

    def total_cost(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get total cost (sum of all constraints)."""
        costs = self.forward(state, action)
        return costs.sum(-1)


if __name__ == "__main__":
    config = RewardPredictorConfig(
        state_dim=768,
        action_dim=7,
        hidden_dim=512,
    )

    batch_size = 4
    state = torch.randn(batch_size, 768)
    action = torch.randn(batch_size, 7)

    # Test reward predictor
    reward_pred = RewardPredictor(config)
    reward = reward_pred(state, action)
    print(f"Reward prediction shape: {reward.shape}")

    # Test critic network
    critic = CriticNetwork(config)
    q1, q2 = critic(state, action)
    print(f"Critic Q values shape: {q1.shape}, {q2.shape}")

    # Test value predictor
    value_pred = ValuePredictor(config)
    value = value_pred(state)
    print(f"Value prediction shape: {value.shape}")

    # Test ensemble value
    ensemble = EnsembleValuePredictor(config, num_ensemble=5)
    outputs = ensemble(state)
    print(f"Ensemble value outputs:")
    print(f"  mean: {outputs['mean'].shape}")
    print(f"  std: {outputs['std'].shape}")

    # Test cost predictor
    cost_pred = CostPredictor(config, num_constraints=3)
    costs = cost_pred(state, action)
    print(f"Cost predictions shape: {costs.shape}")

    # Test target network
    target = TargetNetwork(value_pred, tau=0.005)
    target_value = target(state)
    print(f"Target value shape: {target_value.shape}")
