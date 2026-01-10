"""
SmolVLA Training Script - Lightweight VLA Model (450M parameters)
Ideal for training on consumer hardware with limited GPU memory
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from datasets import load_dataset
import numpy as np


@dataclass
class SmolVLAConfig:
    """Configuration for SmolVLA training."""
    model_name: str = field(
        default="HuggingFaceTB/SmolVLA-450M",
        metadata={"help": "SmolVLA model identifier"}
    )
    dataset_name: str = field(
        default="lerobot/pusht",
        metadata={"help": "LeRobot format dataset"}
    )
    output_dir: str = field(
        default="./smolvla_finetuned",
        metadata={"help": "Output directory"}
    )
    learning_rate: float = field(default=1e-4)
    batch_size: int = field(default=8)
    num_epochs: int = field(default=10)
    gradient_accumulation_steps: int = field(default=4)
    max_samples: Optional[int] = field(default=None)


# LeRobot datasets compatible with SmolVLA
LEROBOT_DATASETS = [
    "lerobot/pusht",
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_insertion_scripted",
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_transfer_cube_scripted",
    "lerobot/xarm_push_medium",
    "lerobot/xarm_lift_medium",
    "lerobot/unitree_g1_dexterous_manipulation",
]


def train_smolvla(config: SmolVLAConfig):
    """Train SmolVLA on a LeRobot format dataset."""

    # Import lerobot specific modules
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig as PolicyConfig
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("LeRobot not installed. Installing...")
        os.system("pip install lerobot")
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig as PolicyConfig
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Load dataset in LeRobot format
    print(f"Loading dataset: {config.dataset_name}")
    dataset = LeRobotDataset(config.dataset_name)

    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    # Initialize policy config
    policy_config = PolicyConfig(
        input_shapes={
            "observation.image": [3, 224, 224],
            "observation.state": dataset.features["observation.state"].shape,
        },
        output_shapes={
            "action": dataset.features["action"].shape,
        },
    )

    # Load model
    print("Loading SmolVLA model...")
    policy = SmolVLAPolicy(policy_config)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs * len(dataset) // config.batch_size
    )

    # Training loop
    print("Starting training...")
    policy.train()

    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = 0

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
        )

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            loss = policy.forward(batch)["loss"]

            # Backward pass with gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{config.num_epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(checkpoint_dir, "policy.pt"))

    # Save final model
    os.makedirs(config.output_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(config.output_dir, "policy_final.pt"))
    print(f"Training complete! Model saved to {config.output_dir}")


def main():
    parser = HfArgumentParser(SmolVLAConfig)
    config = parser.parse_args_into_dataclasses()[0]
    train_smolvla(config)


if __name__ == "__main__":
    main()
