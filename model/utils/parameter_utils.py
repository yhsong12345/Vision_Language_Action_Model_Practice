"""
Parameter Utilities

Shared functions for managing model parameters:
- Freezing/unfreezing modules
- Counting parameters
- Getting parameter statistics
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union


def freeze_module(module: nn.Module, verbose: bool = False) -> None:
    """
    Freeze all parameters in a module.

    Args:
        module: PyTorch module to freeze
        verbose: Print info about frozen parameters
    """
    param_count = 0
    for param in module.parameters():
        param.requires_grad = False
        param_count += param.numel()

    if verbose:
        print(f"Froze {param_count:,} parameters")


def unfreeze_module(module: nn.Module, verbose: bool = False) -> None:
    """
    Unfreeze all parameters in a module.

    Args:
        module: PyTorch module to unfreeze
        verbose: Print info about unfrozen parameters
    """
    param_count = 0
    for param in module.parameters():
        param.requires_grad = True
        param_count += param.numel()

    if verbose:
        print(f"Unfroze {param_count:,} parameters")


def set_requires_grad(
    module: nn.Module,
    requires_grad: bool,
    layer_names: Optional[List[str]] = None,
) -> None:
    """
    Set requires_grad for specific layers or entire module.

    Args:
        module: PyTorch module
        requires_grad: Whether to enable gradients
        layer_names: Optional list of layer name patterns to match
    """
    if layer_names is None:
        for param in module.parameters():
            param.requires_grad = requires_grad
    else:
        for name, param in module.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = requires_grad


def count_parameters(module: nn.Module) -> int:
    """
    Count total parameters in a module.

    Args:
        module: PyTorch module

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    """
    Count trainable parameters in a module.

    Args:
        module: PyTorch module

    Returns:
        Trainable parameter count
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_parameter_stats(
    model: nn.Module,
    components: Optional[Dict[str, nn.Module]] = None,
) -> Dict[str, int]:
    """
    Get detailed parameter statistics for a model.

    Args:
        model: Main model
        components: Optional dict of named submodules to analyze

    Returns:
        Dict with parameter counts for each component
    """
    stats = {
        "total": count_parameters(model),
        "trainable": count_trainable_parameters(model),
        "frozen": count_parameters(model) - count_trainable_parameters(model),
    }

    if components:
        for name, module in components.items():
            stats[f"{name}_total"] = count_parameters(module)
            stats[f"{name}_trainable"] = count_trainable_parameters(module)

    return stats


def print_parameter_summary(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print a formatted parameter summary.

    Args:
        model: PyTorch module
        model_name: Name to display
    """
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = total - trainable

    print(f"\n{model_name} Parameter Summary:")
    print(f"  Total:     {total:>15,}")
    print(f"  Trainable: {trainable:>15,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen:    {frozen:>15,} ({100*frozen/total:.1f}%)")


if __name__ == "__main__":
    # Test utilities
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(100, 50)
            self.decoder = nn.Linear(50, 10)

    model = DummyModel()
    print(f"Total params: {count_parameters(model):,}")
    print(f"Trainable params: {count_trainable_parameters(model):,}")

    freeze_module(model.encoder, verbose=True)
    print(f"After freezing encoder - trainable: {count_trainable_parameters(model):,}")

    print_parameter_summary(model, "DummyModel")
