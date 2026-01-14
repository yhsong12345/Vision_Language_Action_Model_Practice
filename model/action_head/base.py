"""
Base Action Head

Abstract base class for action prediction heads.
Defines common interface and shared functionality.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class ActionHeadBase(nn.Module, ABC):
    """
    Abstract base class for action prediction heads.

    All action heads should inherit from this class and implement
    the required methods for consistent interface across VLA models.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Input features from VLM (batch, input_dim)
            actions: Optional ground truth actions for training

        Returns:
            Dict containing:
                - predicted_actions: (batch, [chunk_size,] action_dim)
                - loss: Optional loss if actions provided
                - Additional outputs specific to head type
        """
        pass

    @abstractmethod
    def predict(
        self,
        features: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Simple prediction interface for inference.

        Args:
            features: Input features (batch, input_dim) or (input_dim,)
            deterministic: Whether to use deterministic prediction

        Returns:
            Predicted actions
        """
        pass

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.

        Args:
            predictions: Predicted actions
            targets: Ground truth actions
            loss_type: Loss type ("mse", "l1", "smooth_l1")

        Returns:
            Loss tensor
        """
        if loss_type == "mse":
            return nn.functional.mse_loss(predictions, targets)
        elif loss_type == "l1":
            return nn.functional.l1_loss(predictions, targets)
        elif loss_type == "smooth_l1":
            return nn.functional.smooth_l1_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def get_config(self) -> Dict[str, Any]:
        """Get action head configuration."""
        return {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"action_dim={self.action_dim}, "
            f"chunk_size={self.chunk_size})"
        )
