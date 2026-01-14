"""
Offline Reinforcement Learning Algorithms

Offline RL algorithms that learn from static datasets without environment interaction:
- CQL: Conservative Q-Learning
- IQL: Implicit Q-Learning
- TD3+BC: TD3 with Behavioral Cloning regularization
- Decision Transformer: Sequence modeling approach to RL

These algorithms are suitable for learning from demonstration datasets.
"""

from .cql_trainer import CQLTrainer
from .iql_trainer import IQLTrainer
from .td3_bc_trainer import TD3BCTrainer
from .decision_transformer import DecisionTransformerTrainer
from .base_trainer import OfflineRLTrainer, OfflineReplayBuffer

__all__ = [
    "CQLTrainer",
    "IQLTrainer",
    "TD3BCTrainer",
    "DecisionTransformerTrainer",
    "OfflineRLTrainer",
    "OfflineReplayBuffer",
]
