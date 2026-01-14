"""
Checkpoint Utilities

Shared functions for model saving and loading:
- Save/load with metadata
- Version tracking
- Config preservation
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class ModelCheckpoint:
    """Container for model checkpoint with metadata."""
    state_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save checkpoint to file."""
        self.metadata["saved_at"] = datetime.now().isoformat()
        torch.save({
            "state_dict": self.state_dict,
            "config": self.config,
            "metadata": self.metadata,
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "ModelCheckpoint":
        """Load checkpoint from file."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        return cls(
            state_dict=data.get("state_dict", {}),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )


def save_checkpoint(
    model: nn.Module,
    path: str,
    config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save model checkpoint with optional training state.

    Args:
        model: PyTorch model
        path: Save path (creates directory if needed)
        config: Model/training configuration
        optimizer: Optional optimizer to save
        scheduler: Optional scheduler to save
        epoch: Current epoch number
        step: Current step number
        metrics: Training/evaluation metrics
        extra: Additional data to save

    Returns:
        Path where checkpoint was saved
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config or {},
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
        },
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if step is not None:
        checkpoint["step"] = step

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Checkpoint file path
        model: Optional model to load state into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        map_location: Device to load tensors to
        strict: Whether to strictly enforce state dict matching

    Returns:
        Checkpoint dictionary with all saved data
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """
    Get checkpoint information without loading full state dict.

    Args:
        path: Checkpoint file path

    Returns:
        Dict with checkpoint metadata
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    info = {
        "config": checkpoint.get("config", {}),
        "metadata": checkpoint.get("metadata", {}),
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "metrics": checkpoint.get("metrics", {}),
        "has_optimizer": "optimizer_state_dict" in checkpoint,
        "has_scheduler": "scheduler_state_dict" in checkpoint,
    }

    # Get state dict structure
    if "state_dict" in checkpoint:
        info["state_dict_keys"] = list(checkpoint["state_dict"].keys())[:10]  # First 10 keys
        info["num_parameters"] = sum(
            p.numel() for p in checkpoint["state_dict"].values()
        )

    return info


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save config as JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)


def load_config(path: str) -> Dict[str, Any]:
    """Load config from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Test checkpoint utilities
    class DummyModel(nn.Module):
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.fc = nn.Linear(10, hidden_dim)

    model = DummyModel(hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Test save
    path = save_checkpoint(
        model=model,
        path="./test_checkpoint.pt",
        config={"hidden_dim": 64},
        optimizer=optimizer,
        epoch=5,
        step=1000,
        metrics={"loss": 0.1, "accuracy": 0.95},
    )
    print(f"Saved checkpoint to: {path}")

    # Test load
    checkpoint = load_checkpoint(path, model)
    print(f"Loaded checkpoint:")
    print(f"  Epoch: {checkpoint.get('epoch')}")
    print(f"  Step: {checkpoint.get('step')}")
    print(f"  Metrics: {checkpoint.get('metrics')}")

    # Test info
    info = get_checkpoint_info(path)
    print(f"\nCheckpoint info:")
    print(f"  Config: {info['config']}")
    print(f"  Parameters: {info['num_parameters']:,}")

    # Cleanup
    os.remove(path)
    print("\nCheckpoint utilities test passed!")
