"""
Integration Module for VLA

Provides bridges to robotics frameworks and simulators:
- ROS/ROS2 integration for real robot deployment
- Simulator bridges (Isaac Sim, MuJoCo, PyBullet, CARLA)
- Experiment management and tracking
"""

from .ros_bridge import ROSBridge, ROS2Bridge
from .simulator_bridge import (
    SimulatorBridge,
    IsaacSimBridge,
    MuJoCoBridge,
    CARLABridge,
)
from .experiment_manager import (
    ExperimentManager,
    ExperimentConfig,
    MetricsLogger,
)

__all__ = [
    "ROSBridge",
    "ROS2Bridge",
    "SimulatorBridge",
    "IsaacSimBridge",
    "MuJoCoBridge",
    "CARLABridge",
    "ExperimentManager",
    "ExperimentConfig",
    "MetricsLogger",
]
