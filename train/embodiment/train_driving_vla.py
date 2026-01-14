"""
Autonomous Driving VLA Trainer

Training pipeline for driving-specific VLA model:
- Multi-camera BEV encoding
- Trajectory prediction and planning
- Language-conditioned driving
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import argparse

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.embodiment.autonomous_vehicle import DrivingVLA, DrivingVLAConfig
from model.safety.safety_shield import SafetyShield, SafetyConfig
from model.safety.rule_checker import TrafficRuleChecker


@dataclass
class DrivingTrainConfig:
    """Configuration for driving VLA training."""
    # Model
    num_cameras: int = 6
    image_size: int = 224
    bev_size: int = 200
    bev_resolution: float = 0.5
    hidden_dim: int = 512
    llm_hidden_dim: int = 4096
    trajectory_length: int = 20

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000

    # Loss weights
    trajectory_loss_weight: float = 1.0
    control_loss_weight: float = 0.5
    cost_map_loss_weight: float = 0.1
    safety_loss_weight: float = 0.2

    # Data
    data_path: str = "./data/driving"
    num_workers: int = 8

    # Checkpointing
    output_dir: str = "./checkpoints/driving_vla"
    save_steps: int = 1000
    eval_steps: int = 500

    # Safety
    use_safety_shield: bool = True
    max_speed: float = 30.0  # m/s
    max_acceleration: float = 5.0  # m/s^2
    min_distance: float = 2.0  # meters


class DrivingDataset(Dataset):
    """Dataset for autonomous driving training."""

    def __init__(
        self,
        data_path: str,
        config: DrivingTrainConfig,
        split: str = "train",
    ):
        self.data_path = data_path
        self.config = config
        self.split = split

        # Load data index
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample index from data path."""
        # Placeholder - in practice, load from actual dataset
        samples = []
        index_file = os.path.join(self.data_path, f"{self.split}_index.json")

        if os.path.exists(index_file):
            import json
            with open(index_file, "r") as f:
                samples = json.load(f)
        else:
            # Generate dummy samples for testing
            for i in range(1000):
                samples.append({
                    "id": i,
                    "camera_paths": [f"cam_{j}/frame_{i:06d}.jpg" for j in range(self.config.num_cameras)],
                    "trajectory": f"trajectories/traj_{i:06d}.npy",
                    "controls": f"controls/ctrl_{i:06d}.npy",
                    "instruction": f"Drive forward safely",
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load camera images (placeholder - random for now)
        images = torch.randn(
            self.config.num_cameras, 3,
            self.config.image_size, self.config.image_size
        )

        # Load trajectory (placeholder)
        trajectory = torch.randn(self.config.trajectory_length, 2)

        # Load controls (placeholder)
        controls = torch.rand(3)  # throttle, brake, steer

        # Language features (placeholder - would come from LLM)
        language_features = torch.randn(32, self.config.llm_hidden_dim)

        return {
            "images": images,
            "trajectory": trajectory,
            "controls": controls,
            "language_features": language_features,
        }


class DrivingVLATrainer:
    """Trainer for autonomous driving VLA."""

    def __init__(self, config: DrivingTrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        model_config = DrivingVLAConfig(
            num_cameras=config.num_cameras,
            image_size=config.image_size,
            bev_size=config.bev_size,
            bev_resolution=config.bev_resolution,
            hidden_dim=config.hidden_dim,
            llm_hidden_dim=config.llm_hidden_dim,
            trajectory_length=config.trajectory_length,
        )
        self.model = DrivingVLA(model_config).to(self.device)

        # Safety components
        if config.use_safety_shield:
            safety_config = SafetyConfig(
                max_velocity=config.max_speed,
                max_acceleration=config.max_acceleration,
                min_distance=config.min_distance,
            )
            self.safety_shield = SafetyShield(safety_config)
            self.rule_checker = TrafficRuleChecker()
        else:
            self.safety_shield = None
            self.rule_checker = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )

        # Metrics
        self.global_step = 0
        self.best_loss = float("inf")

    def train(self):
        """Main training loop."""
        # Create datasets
        train_dataset = DrivingDataset(
            self.config.data_path, self.config, split="train"
        )
        val_dataset = DrivingDataset(
            self.config.data_path, self.config, split="val"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        print(f"Training on {len(train_dataset)} samples")
        print(f"Validating on {len(val_dataset)} samples")

        for epoch in range(self.config.num_epochs):
            # Training epoch
            train_loss = self._train_epoch(train_loader, epoch)

            # Validation
            if (epoch + 1) % (self.config.eval_steps // len(train_loader) + 1) == 0:
                val_loss = self._validate(val_loader)
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint("best_model.pt")

            # Update scheduler
            self.scheduler.step()

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        print("Training completed!")

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["images"].to(self.device)
            trajectory_gt = batch["trajectory"].to(self.device)
            controls_gt = batch["controls"].to(self.device)
            language_features = batch["language_features"].to(self.device)

            # Forward pass
            outputs = self.model(images, language_features)

            # Compute losses
            loss = self._compute_loss(outputs, trajectory_gt, controls_gt)

            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({"loss": total_loss / num_batches})

        return total_loss / num_batches

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        trajectory_gt: torch.Tensor,
        controls_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss."""
        # Trajectory loss (L2)
        trajectory_loss = F.mse_loss(outputs["trajectory"], trajectory_gt)

        # Control loss
        control_loss = F.mse_loss(outputs["controls"], controls_gt)

        # Cost map regularization (encourage smooth cost maps)
        cost_map = outputs["cost_map"]
        cost_smoothness = torch.mean(
            torch.abs(cost_map[:, :, 1:, :] - cost_map[:, :, :-1, :]) +
            torch.abs(cost_map[:, :, :, 1:] - cost_map[:, :, :, :-1])
        )

        # Safety loss (penalize unsafe predictions)
        safety_loss = torch.tensor(0.0, device=outputs["trajectory"].device)
        if self.safety_shield is not None:
            # Check speed constraints
            speeds = outputs.get("speeds", None)
            if speeds is not None:
                speed_violation = F.relu(speeds - self.config.max_speed).mean()
                safety_loss = safety_loss + speed_violation

        # Total loss
        total_loss = (
            self.config.trajectory_loss_weight * trajectory_loss +
            self.config.control_loss_weight * control_loss +
            self.config.cost_map_loss_weight * cost_smoothness +
            self.config.safety_loss_weight * safety_loss
        )

        return total_loss

    def _validate(self, dataloader: DataLoader) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch["images"].to(self.device)
                trajectory_gt = batch["trajectory"].to(self.device)
                controls_gt = batch["controls"].to(self.device)
                language_features = batch["language_features"].to(self.device)

                outputs = self.model(images, language_features)
                loss = self._compute_loss(outputs, trajectory_gt, controls_gt)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        print(f"Checkpoint loaded from {path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Driving VLA")

    # Model args
    parser.add_argument("--num-cameras", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--bev-size", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=512)

    # Training args
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    # Data args
    parser.add_argument("--data-path", type=str, default="./data/driving")
    parser.add_argument("--num-workers", type=int, default=8)

    # Output args
    parser.add_argument("--output-dir", type=str, default="./checkpoints/driving_vla")
    parser.add_argument("--resume", type=str, default=None)

    # Safety args
    parser.add_argument("--use-safety-shield", action="store_true", default=True)
    parser.add_argument("--max-speed", type=float, default=30.0)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = DrivingTrainConfig(
        num_cameras=args.num_cameras,
        image_size=args.image_size,
        bev_size=args.bev_size,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        data_path=args.data_path,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        use_safety_shield=args.use_safety_shield,
        max_speed=args.max_speed,
    )

    trainer = DrivingVLATrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
