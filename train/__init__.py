"""
VLA Training Module

Training pipelines for Vision-Language-Action models:

Submodules:
- pretrain: VLM pretraining (vision-language alignment, instruction tuning)
- finetune: Supervised fine-tuning on robot manipulation data
- online_rl: Online RL with environment interaction (PPO, SAC, GRPO)
- offline_rl: Offline RL from static datasets (CQL, IQL, TD3+BC, DT)
- il: Imitation learning (BC, DAgger, GAIL)
- world_model: World model training (Dreamer-style)
- embodiment: Embodiment-specific training (Driving, Humanoid)
- datasets: Dataset loaders for various training paradigms
"""

from .pretrain import VLMPretrainer
from .finetune import VLAFineTuner

# Online RL
from .online_rl import (
    OnlineRLTrainer,
    PPOTrainer,
    SACTrainer,
    GRPOTrainer,
    RolloutBuffer,
    ReplayBuffer,
)

# Offline RL
from .offline_rl import (
    OfflineRLTrainer,
    CQLTrainer,
    IQLTrainer,
    TD3BCTrainer,
    DecisionTransformerTrainer,
)

# Imitation Learning
from .il import ILTrainer, BehavioralCloning, DAgger

# World Model
from .world_model import WorldModelTrainer

# Embodiment-specific trainers
from .embodiment import DrivingVLATrainer, HumanoidVLATrainer

# Dataset loaders
from .datasets import (
    # LeRobot
    LeRobotDataset,
    PushTDataset,
    AlohaDataset,
    XArmDataset,
    create_lerobot_dataloader,
    # Open X-Embodiment
    OpenXDataset,
    BridgeDataset,
    RT1Dataset,
    create_openx_dataloader,
    # Driving
    DrivingDataset,
    NuScenesDataset,
    CarlaDataset,
    create_driving_dataloader,
    # RL
    RLDataset,
    D4RLDataset,
    RoboMimicDataset,
    create_rl_dataloader,
)

__all__ = [
    # Pretraining & Fine-tuning
    "VLMPretrainer",
    "VLAFineTuner",
    # Online RL
    "OnlineRLTrainer",
    "PPOTrainer",
    "SACTrainer",
    "GRPOTrainer",
    "RolloutBuffer",
    "ReplayBuffer",
    # Offline RL
    "OfflineRLTrainer",
    "CQLTrainer",
    "IQLTrainer",
    "TD3BCTrainer",
    "DecisionTransformerTrainer",
    # Imitation Learning
    "ILTrainer",
    "BehavioralCloning",
    "DAgger",
    # World Model
    "WorldModelTrainer",
    # Embodiment
    "DrivingVLATrainer",
    "HumanoidVLATrainer",
    # LeRobot datasets
    "LeRobotDataset",
    "PushTDataset",
    "AlohaDataset",
    "XArmDataset",
    "create_lerobot_dataloader",
    # Open X datasets
    "OpenXDataset",
    "BridgeDataset",
    "RT1Dataset",
    "create_openx_dataloader",
    # Driving datasets
    "DrivingDataset",
    "NuScenesDataset",
    "CarlaDataset",
    "create_driving_dataloader",
    # RL datasets
    "RLDataset",
    "D4RLDataset",
    "RoboMimicDataset",
    "create_rl_dataloader",
]
