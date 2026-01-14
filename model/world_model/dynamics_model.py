"""
Dynamics Model Module

Predicts state transitions for model-based planning and imagination.
Supports deterministic, probabilistic, and ensemble dynamics models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class DynamicsConfig:
    """Configuration for dynamics models."""
    state_dim: int = 768
    action_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 3
    num_ensemble: int = 5
    dropout: float = 0.1
    use_layer_norm: bool = True
    output_std: bool = True  # For probabilistic models
    min_std: float = 1e-4
    max_std: float = 1.0


class DeterministicDynamics(nn.Module):
    """
    Deterministic dynamics model.

    Predicts: s_{t+1} = f(s_t, a_t)
    """

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(
            config.state_dim + config.action_dim, config.hidden_dim
        )

        # Hidden layers
        layers = []
        for _ in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))

        self.hidden = nn.Sequential(*layers)

        # Output projection (predicts state delta)
        self.output_proj = nn.Linear(config.hidden_dim, config.state_dim)

        # Residual connection
        self.use_residual = True

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)

        Returns:
            next_state: (batch_size, state_dim)
        """
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)
        x = self.hidden(x)
        delta = self.output_proj(x)

        if self.use_residual:
            return state + delta
        return delta


class ProbabilisticDynamics(nn.Module):
    """
    Probabilistic dynamics model.

    Predicts: p(s_{t+1} | s_t, a_t) as a Gaussian distribution.
    Useful for uncertainty estimation and model-based RL.
    """

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Hidden layers
        layers = []
        for _ in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
        self.hidden = nn.Sequential(*layers)

        # Mean and std outputs
        self.mean_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.std_head = nn.Linear(config.hidden_dim, config.state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state distribution.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            deterministic: If True, return mean instead of sample

        Returns:
            Dict with mean, std, sample, log_prob
        """
        x = torch.cat([state, action], dim=-1)
        x = self.encoder(x)
        x = self.hidden(x)

        # Predict mean (as delta)
        mean_delta = self.mean_head(x)
        mean = state + mean_delta

        # Predict std
        log_std = self.std_head(x)
        log_std = torch.clamp(
            log_std,
            np.log(self.config.min_std),
            np.log(self.config.max_std),
        )
        std = log_std.exp()

        # Sample
        if deterministic:
            sample = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            sample = dist.rsample()

        return {
            "mean": mean,
            "std": std,
            "sample": sample,
            "log_prob": -0.5 * (((sample - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi)).sum(-1),
        }

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood loss."""
        outputs = self.forward(state, action)
        mean = outputs["mean"]
        std = outputs["std"]

        # Gaussian NLL
        nll = 0.5 * (((next_state - mean) / std) ** 2 + 2 * std.log() + np.log(2 * np.pi))
        return nll.sum(-1).mean()


class EnsembleDynamics(nn.Module):
    """
    Ensemble of dynamics models for uncertainty quantification.

    Trains multiple dynamics models and uses disagreement as
    uncertainty estimate (epistemic uncertainty).
    """

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

        # Create ensemble
        self.models = nn.ModuleList([
            ProbabilisticDynamics(config)
            for _ in range(config.num_ensemble)
        ])

        # Ensemble aggregation
        self.aggregation = "mean"  # "mean" or "min"

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state with ensemble.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            return_all: If True, return all model predictions

        Returns:
            Dict with aggregated predictions and uncertainty
        """
        predictions = []
        means = []
        stds = []

        for model in self.models:
            outputs = model(state, action, deterministic=True)
            predictions.append(outputs["sample"])
            means.append(outputs["mean"])
            stds.append(outputs["std"])

        # Stack predictions
        all_means = torch.stack(means, dim=0)  # (ensemble, batch, state_dim)
        all_stds = torch.stack(stds, dim=0)

        # Aggregate
        if self.aggregation == "mean":
            mean = all_means.mean(0)
        else:  # min uncertainty
            uncertainties = all_stds.mean(-1)  # (ensemble, batch)
            min_idx = uncertainties.argmin(0)  # (batch,)
            batch_idx = torch.arange(state.size(0), device=state.device)
            mean = all_means[min_idx, batch_idx]

        # Epistemic uncertainty (disagreement)
        epistemic_std = all_means.std(0)

        # Aleatoric uncertainty (average model uncertainty)
        aleatoric_std = all_stds.mean(0)

        # Total uncertainty
        total_std = (epistemic_std ** 2 + aleatoric_std ** 2).sqrt()

        outputs = {
            "mean": mean,
            "std": total_std,
            "epistemic_std": epistemic_std,
            "aleatoric_std": aleatoric_std,
        }

        if return_all:
            outputs["all_means"] = all_means
            outputs["all_stds"] = all_stds

        return outputs

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute average ensemble loss."""
        total_loss = 0
        for model in self.models:
            total_loss += model.loss(state, action, next_state)
        return total_loss / len(self.models)

    def get_model_disagreement(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get model disagreement for exploration bonus.

        Higher disagreement indicates less explored regions.
        """
        outputs = self.forward(state, action, return_all=True)
        return outputs["epistemic_std"].mean(-1)


class DynamicsModel(nn.Module):
    """
    Unified dynamics model interface.

    Supports multi-step prediction and planning.
    """

    def __init__(
        self,
        config: DynamicsConfig,
        model_type: str = "ensemble",
    ):
        super().__init__()
        self.config = config

        if model_type == "deterministic":
            self.model = DeterministicDynamics(config)
        elif model_type == "probabilistic":
            self.model = ProbabilisticDynamics(config)
        elif model_type == "ensemble":
            self.model = EnsembleDynamics(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = model_type

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Single-step prediction."""
        if self.model_type == "deterministic":
            next_state = self.model(state, action)
            return {"mean": next_state, "std": torch.zeros_like(next_state)}
        else:
            return self.model(state, action)

    def multi_step_predict(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-step prediction (imagination rollout).

        Args:
            state: (batch_size, state_dim) initial state
            actions: (batch_size, horizon, action_dim) action sequence

        Returns:
            Dict with predicted states and uncertainties
        """
        batch_size, horizon, _ = actions.shape

        states = [state]
        stds = []

        current_state = state
        for t in range(horizon):
            action = actions[:, t]
            outputs = self.predict(current_state, action)
            current_state = outputs["mean"]
            states.append(current_state)
            stds.append(outputs["std"])

        return {
            "states": torch.stack(states, dim=1),  # (batch, horizon+1, state_dim)
            "stds": torch.stack(stds, dim=1),  # (batch, horizon, state_dim)
        }

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamics loss."""
        if self.model_type == "deterministic":
            pred = self.model(state, action)
            return F.mse_loss(pred, next_state)
        else:
            return self.model.loss(state, action, next_state)


if __name__ == "__main__":
    config = DynamicsConfig(
        state_dim=768,
        action_dim=7,
        hidden_dim=512,
        num_ensemble=5,
    )

    # Test deterministic dynamics
    det_model = DeterministicDynamics(config)
    state = torch.randn(4, 768)
    action = torch.randn(4, 7)
    next_state = det_model(state, action)
    print(f"Deterministic next state shape: {next_state.shape}")

    # Test probabilistic dynamics
    prob_model = ProbabilisticDynamics(config)
    outputs = prob_model(state, action)
    print(f"Probabilistic outputs:")
    print(f"  mean: {outputs['mean'].shape}")
    print(f"  std: {outputs['std'].shape}")

    # Test ensemble dynamics
    ensemble = EnsembleDynamics(config)
    outputs = ensemble(state, action)
    print(f"Ensemble outputs:")
    print(f"  mean: {outputs['mean'].shape}")
    print(f"  epistemic_std: {outputs['epistemic_std'].shape}")
    print(f"  aleatoric_std: {outputs['aleatoric_std'].shape}")

    # Test unified interface
    dynamics = DynamicsModel(config, model_type="ensemble")
    actions_seq = torch.randn(4, 10, 7)  # 10-step horizon
    outputs = dynamics.multi_step_predict(state, actions_seq)
    print(f"Multi-step prediction:")
    print(f"  states: {outputs['states'].shape}")
    print(f"  stds: {outputs['stds'].shape}")
