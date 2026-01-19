"""
Imitation Learning Module for VLA

Implements imitation learning algorithms:
- BC: Behavioral Cloning (supervised learning on demonstrations)
- DAgger: Dataset Aggregation (interactive imitation learning)
- GAIL: Generative Adversarial Imitation Learning
"""

from .base_trainer import ILTrainer
from .behavioral_cloning import BehavioralCloning, VLABehavioralCloning
from .dagger import DAgger, VLADAgger
from .gail import GAIL, VLAGAIL

__all__ = [
    "ILTrainer",
    # Behavioral Cloning
    "BehavioralCloning",
    "VLABehavioralCloning",
    # DAgger
    "DAgger",
    "VLADAgger",
    # GAIL
    "GAIL",
    "VLAGAIL",
]
