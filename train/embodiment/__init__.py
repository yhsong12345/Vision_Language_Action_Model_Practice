"""
Embodiment-Specific Training Scripts

Training pipelines for different robot types:
- Autonomous Vehicle: BEV + trajectory planning training
- Humanoid Robot: Whole-body control training
"""

from .train_driving_vla import DrivingVLATrainer
from .train_humanoid_vla import HumanoidVLATrainer

__all__ = [
    "DrivingVLATrainer",
    "HumanoidVLATrainer",
]
