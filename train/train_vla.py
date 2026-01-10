"""
Training Script for Custom VLA Model
Supports training on various robot manipulation datasets.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json
import wandb

from vla_model import VLAModel, VLAConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    vision_model: str = "google/siglip-base-patch16-224"
    llm_model: str = "Qwen/Qwen2-1.5B-Instruct"
    action_dim: int = 7
    freeze_vision: bool = True
    freeze_llm: bool = False

    # Data
    dataset_name: str = "lerobot/pusht"
    max_samples: Optional[int] = None
    num_workers: int = 4

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Logging
    output_dir: str = "./vla_output"
    logging_steps: int = 10
    save_steps: int = 500
    use_wandb: bool = False
    wandb_project: str = "vla-training"


class RobotDataset(Dataset):
    """Generic robot manipulation dataset."""

    def __init__(
        self,
        dataset_name: str,
        image_processor,
        tokenizer,
        max_samples: Optional[int] = None,
        split: str = "train",
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        # Try to load from HuggingFace datasets
        try:
            from datasets import load_dataset
            self.hf_dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            self.dataset_type = "huggingface"
        except Exception as e:
            print(f"Could not load HuggingFace dataset: {e}")
            print("Creating dummy dataset for testing...")
            self.hf_dataset = None
            self.dataset_type = "dummy"
            self._create_dummy_data(max_samples or 100)

        if max_samples and self.hf_dataset:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

    def _create_dummy_data(self, num_samples: int):
        """Create dummy data for testing."""
        self.dummy_data = []
        instructions = [
            "pick up the red block",
            "move the arm to the left",
            "place the object on the table",
            "push the cube forward",
            "grasp the handle",
        ]
        for i in range(num_samples):
            self.dummy_data.append({
                "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "instruction": instructions[i % len(instructions)],
                "action": np.random.randn(7).astype(np.float32),
            })

    def __len__(self):
        if self.dataset_type == "dummy":
            return len(self.dummy_data)
        return len(self.hf_dataset)

    def _get_image(self, item: Dict) -> Image.Image:
        """Extract image from dataset item."""
        # Handle different dataset formats
        if "image" in item:
            img = item["image"]
        elif "observation" in item:
            if isinstance(item["observation"], dict):
                img = item["observation"].get("image", item["observation"].get("rgb"))
            else:
                img = item["observation"]
        elif "pixel_values" in item:
            img = item["pixel_values"]
        else:
            # Default to random image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy().astype(np.uint8))

        return img.convert("RGB")

    def _get_instruction(self, item: Dict) -> str:
        """Extract instruction from dataset item."""
        if "instruction" in item:
            return item["instruction"]
        elif "language_instruction" in item:
            return item["language_instruction"]
        elif "task" in item:
            return item["task"]
        elif "text" in item:
            return item["text"]
        else:
            return "Perform the manipulation task."

    def _get_action(self, item: Dict) -> np.ndarray:
        """Extract action from dataset item."""
        if "action" in item:
            action = item["action"]
        elif "actions" in item:
            action = item["actions"]
        else:
            action = np.zeros(7)

        if isinstance(action, torch.Tensor):
            action = action.numpy()

        return np.array(action, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.dataset_type == "dummy":
            item = self.dummy_data[idx]
        else:
            item = self.hf_dataset[idx]

        # Get components
        image = self._get_image(item)
        instruction = self._get_instruction(item)
        action = self._get_action(item)

        # Process image
        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        # Tokenize instruction
        text_inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "action": torch.tensor(action, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "action": torch.stack([x["action"] for x in batch]),
    }


def train(config: TrainingConfig):
    """Main training function."""

    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(project=config.wandb_project, config=vars(config))

    # Create model
    print("Creating VLA model...")
    model_config = VLAConfig(
        vision_model_name=config.vision_model,
        llm_model_name=config.llm_model,
        action_dim=config.action_dim,
        freeze_vision=config.freeze_vision,
        freeze_llm=config.freeze_llm,
    )
    model = VLAModel(model_config)
    model = model.to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataset
    print(f"Loading dataset: {config.dataset_name}")
    train_dataset = RobotDataset(
        dataset_name=config.dataset_name,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        max_samples=config.max_samples,
    )
    print(f"Dataset size: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training loop
    print("Starting training...")
    global_step = 0
    best_loss = float("inf")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )
                loss = outputs["loss"] / config.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    log_dict = {
                        "loss": loss.item() * config.gradient_accumulation_steps,
                        "learning_rate": lr,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    }
                    if config.use_wandb:
                        wandb.log(log_dict)

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    checkpoint_path = os.path.join(
                        config.output_dir, f"checkpoint-{global_step}.pt"
                    )
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": global_step,
                        "config": vars(config),
                    }, checkpoint_path)
                    print(f"\nSaved checkpoint to {checkpoint_path}")

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(config.output_dir, "best_model.pt")
            model.save_pretrained(best_path)
            print(f"Saved best model to {best_path}")

    # Save final model
    final_path = os.path.join(config.output_dir, "final_model.pt")
    model.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    if config.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train VLA Model")

    # Model arguments
    parser.add_argument("--vision_model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true")

    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="lerobot/pusht")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./vla_output")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vla-training")

    args = parser.parse_args()

    config = TrainingConfig(**vars(args))
    train(config)


if __name__ == "__main__":
    main()
