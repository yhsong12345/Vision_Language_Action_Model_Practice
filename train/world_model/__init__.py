"""
World Model Training Scripts

Training pipelines for world models:
- Latent dynamics (Dreamer-style RSSM)
- Reward prediction
- Model-based planning
"""

from .train_world_model import WorldModelTrainer

__all__ = [
    "WorldModelTrainer",
]
