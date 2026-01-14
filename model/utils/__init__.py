"""
Model Utilities

Shared utilities for VLA models:
- parameter_utils: Module freezing, parameter counting
- device_utils: Device detection and management
- checkpoint_utils: Model saving and loading
"""

from .parameter_utils import (
    freeze_module,
    unfreeze_module,
    count_parameters,
    count_trainable_parameters,
    get_parameter_stats,
    set_requires_grad,
)

from .device_utils import (
    get_device,
    move_to_device,
)

from .checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    ModelCheckpoint,
)

__all__ = [
    # Parameter utilities
    "freeze_module",
    "unfreeze_module",
    "count_parameters",
    "count_trainable_parameters",
    "get_parameter_stats",
    "set_requires_grad",
    # Device utilities
    "get_device",
    "move_to_device",
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    "ModelCheckpoint",
]
