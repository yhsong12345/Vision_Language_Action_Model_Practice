"""
Embodiment-Specific Modules for VLA

Provides specialized components for different robot types:
- Autonomous Vehicle: Driving-specific perception and control
- Humanoid Robot: Whole-body control and locomotion
"""

from .autonomous_vehicle import (
    DrivingVLA,
    BEVEncoder,
    TrajectoryDecoder,
    MotionPlanner,
)
from .humanoid import (
    HumanoidVLA,
    WholeBodyController,
    LocomotionPolicy,
    ManipulationPolicy,
)

__all__ = [
    # Autonomous Vehicle
    "DrivingVLA",
    "BEVEncoder",
    "TrajectoryDecoder",
    "MotionPlanner",
    # Humanoid
    "HumanoidVLA",
    "WholeBodyController",
    "LocomotionPolicy",
    "ManipulationPolicy",
]
