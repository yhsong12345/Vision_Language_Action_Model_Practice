"""
Device Utilities

Shared functions for device management:
- Auto device detection
- Moving tensors/modules to devices
"""

import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """
    Get PyTorch device, with auto-detection support.

    Args:
        device: Device specification:
            - "auto": Automatically detect best available device
            - "cuda": Use CUDA GPU
            - "cuda:0", "cuda:1", etc.: Use specific GPU
            - "cpu": Use CPU
            - "mps": Use Apple Metal (M1/M2)
            - torch.device object: Use directly

    Returns:
        torch.device object
    """
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device)


def move_to_device(
    data: Union[torch.Tensor, nn.Module, Dict, list, tuple],
    device: Union[str, torch.device],
) -> Union[torch.Tensor, nn.Module, Dict, list, tuple]:
    """
    Move tensors, modules, or nested data structures to device.

    Args:
        data: Data to move (tensor, module, dict, list, or tuple)
        device: Target device

    Returns:
        Data on target device
    """
    device = get_device(device)

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, nn.Module):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device) for item in data]
        return type(data)(moved)
    else:
        return data


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.

    Returns:
        Dict with device availability and info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    if info["cuda_available"]:
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(info["cuda_device_count"])
        ]
        info["cuda_memory"] = [
            torch.cuda.get_device_properties(i).total_memory / (1024**3)
            for i in range(info["cuda_device_count"])
        ]

    return info


def print_device_info() -> None:
    """Print formatted device information."""
    info = get_device_info()

    print("\nDevice Information:")
    print(f"  CUDA Available: {info['cuda_available']}")

    if info["cuda_available"]:
        print(f"  CUDA Devices: {info['cuda_device_count']}")
        for i, (name, mem) in enumerate(zip(info["cuda_devices"], info["cuda_memory"])):
            print(f"    [{i}] {name} ({mem:.1f} GB)")

    print(f"  MPS Available: {info['mps_available']}")
    print(f"  Recommended: {get_device('auto')}")


if __name__ == "__main__":
    print_device_info()

    # Test device detection
    device = get_device("auto")
    print(f"\nAuto-detected device: {device}")

    # Test moving data
    tensor = torch.randn(3, 3)
    moved = move_to_device(tensor, device)
    print(f"Tensor device: {moved.device}")

    # Test nested data
    data = {"a": torch.randn(2, 2), "b": [torch.randn(3), torch.randn(4)]}
    moved_data = move_to_device(data, device)
    print(f"Nested dict tensor device: {moved_data['a'].device}")
