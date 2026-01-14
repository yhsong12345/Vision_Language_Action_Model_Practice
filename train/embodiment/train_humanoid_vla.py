"""
Humanoid Robot VLA Trainer

Training pipeline for humanoid-specific VLA model:
- Whole-body control
- Locomotion and manipulation
- Language-conditioned behavior
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

from model.embodiment.humanoid import HumanoidVLA, HumanoidVLAConfig
from model.safety.safety_shield import SafetyShield, SafetyConfig
from model.safety.constraint_handler import ConstraintHandler


@dataclass
class HumanoidTrainConfig:
    """Configuration for humanoid VLA training."""
    # Model
    num_joints: int = 32
    joint_dim: int = 12  # pos, vel, torque for each
    image_size: int = 224
    hidden_dim: int = 512
    llm_hidden_dim: int = 4096

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 200
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000

    # Loss weights
    action_loss_weight: float = 1.0
    locomotion_loss_weight: float = 0.5
    manipulation_loss_weight: float = 0.5
    stability_loss_weight: float = 0.3
    smoothness_loss_weight: float = 0.1

    # Data
    data_path: str = "./data/humanoid"
    num_workers: int = 8
    sequence_length: int = 16

    # Checkpointing
    output_dir: str = "./checkpoints/humanoid_vla"
    save_steps: int = 1000
    eval_steps: int = 500

    # Safety
    use_safety_constraints: bool = True
    max_joint_velocity: float = 5.0  # rad/s
    max_joint_torque: float = 100.0  # Nm
    min_com_height: float = 0.3  # meters


class HumanoidDataset(Dataset):
    """Dataset for humanoid robot training."""

    def __init__(
        self,
        data_path: str,
        config: HumanoidTrainConfig,
        split: str = "train",
    ):
        self.data_path = data_path
        self.config = config
        self.split = split

        # Load data index
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample index from data path."""
        samples = []
        index_file = os.path.join(self.data_path, f"{self.split}_index.json")

        if os.path.exists(index_file):
            import json
            with open(index_file, "r") as f:
                samples = json.load(f)
        else:
            # Generate dummy samples for testing
            for i in range(2000):
                samples.append({
                    "id": i,
                    "image_path": f"images/frame_{i:06d}.jpg",
                    "proprioception_path": f"proprio/proprio_{i:06d}.npy",
                    "action_path": f"actions/action_{i:06d}.npy",
                    "instruction": "Walk forward",
                    "task_type": np.random.choice(["locomotion", "manipulation", "whole_body"]),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image (placeholder)
        image = torch.randn(3, self.config.image_size, self.config.image_size)

        # Load proprioception (placeholder)
        # Format: [joint_positions, joint_velocities, joint_torques, imu_data]
        proprioception = torch.randn(self.config.num_joints * 3 + 6)  # +6 for IMU

        # Load action sequence (placeholder)
        actions = torch.randn(self.config.sequence_length, self.config.num_joints)

        # Language features (placeholder)
        language_features = torch.randn(32, self.config.llm_hidden_dim)

        # Task type encoding
        task_types = {"locomotion": 0, "manipulation": 1, "whole_body": 2}
        task_type = torch.tensor(task_types.get(sample.get("task_type", "whole_body"), 2))

        return {
            "image": image,
            "proprioception": proprioception,
            "actions": actions,
            "language_features": language_features,
            "task_type": task_type,
        }


class HumanoidVLATrainer:
    """Trainer for humanoid robot VLA."""

    def __init__(self, config: HumanoidTrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        model_config = HumanoidVLAConfig(
            num_joints=config.num_joints,
            joint_dim=config.joint_dim,
            image_size=config.image_size,
            hidden_dim=config.hidden_dim,
            llm_hidden_dim=config.llm_hidden_dim,
        )
        self.model = HumanoidVLA(model_config).to(self.device)

        # Safety components
        if config.use_safety_constraints:
            safety_config = SafetyConfig(
                max_velocity=config.max_joint_velocity,
                max_acceleration=config.max_joint_torque,  # reusing for torque
            )
            self.safety_shield = SafetyShield(safety_config)
            self.constraint_handler = ConstraintHandler(
                action_dim=config.num_joints,
                hidden_dim=config.hidden_dim,
            )
        else:
            self.safety_shield = None
            self.constraint_handler = None

        # Optimizer with layer-wise learning rate decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Metrics
        self.global_step = 0
        self.best_loss = float("inf")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            return max(
                0.1,
                0.5 * (1.0 + np.cos(np.pi * (step - self.config.warmup_steps) /
                                    (self.config.num_epochs * 100 - self.config.warmup_steps)))
            )
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Main training loop."""
        # Create datasets
        train_dataset = HumanoidDataset(
            self.config.data_path, self.config, split="train"
        )
        val_dataset = HumanoidDataset(
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
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.num_epochs):
            # Training epoch
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation
            val_metrics = self._validate(val_loader)

            print(
                f"Epoch {epoch+1}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_action_error={val_metrics['action_error']:.4f}"
            )

            # Save best model
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self._save_checkpoint("best_model.pt")

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        print("Training completed!")

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_stability_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            image = batch["image"].to(self.device)
            proprioception = batch["proprioception"].to(self.device)
            actions_gt = batch["actions"].to(self.device)
            language_features = batch["language_features"].to(self.device)
            task_type = batch["task_type"].to(self.device)

            # Forward pass
            outputs = self.model(
                image=image,
                proprioception=proprioception,
                language_features=language_features,
            )

            # Compute losses
            losses = self._compute_loss(outputs, actions_gt, proprioception)
            loss = losses["total"]

            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += losses["total"].item()
            total_action_loss += losses["action"].item()
            total_stability_loss += losses["stability"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": total_loss / num_batches,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

        return {
            "loss": total_loss / num_batches,
            "action_loss": total_action_loss / num_batches,
            "stability_loss": total_stability_loss / num_batches,
        }

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        actions_gt: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        losses = {}

        # Action prediction loss
        # Use first timestep action from outputs
        action_pred = outputs["actions"]
        if action_pred.dim() == 2:
            # Expand to match sequence length if needed
            action_pred = action_pred.unsqueeze(1).expand(-1, actions_gt.shape[1], -1)

        action_loss = F.mse_loss(action_pred, actions_gt)
        losses["action"] = action_loss

        # Locomotion-specific loss (if locomotion outputs available)
        loco_loss = torch.tensor(0.0, device=action_pred.device)
        if "locomotion_actions" in outputs:
            loco_actions = outputs["locomotion_actions"]
            # Lower body joints (typically first half)
            num_loco_joints = self.config.num_joints // 2
            loco_loss = F.mse_loss(loco_actions[:, :num_loco_joints], actions_gt[:, 0, :num_loco_joints])
        losses["locomotion"] = loco_loss

        # Manipulation-specific loss
        manip_loss = torch.tensor(0.0, device=action_pred.device)
        if "manipulation_actions" in outputs:
            manip_actions = outputs["manipulation_actions"]
            # Upper body joints (typically second half)
            num_loco_joints = self.config.num_joints // 2
            manip_loss = F.mse_loss(
                manip_actions[:, :self.config.num_joints - num_loco_joints],
                actions_gt[:, 0, num_loco_joints:]
            )
        losses["manipulation"] = manip_loss

        # Stability loss (penalize actions that could cause instability)
        stability_loss = self._compute_stability_loss(outputs, proprioception)
        losses["stability"] = stability_loss

        # Smoothness loss (penalize jerky motions)
        smoothness_loss = torch.tensor(0.0, device=action_pred.device)
        if action_pred.shape[1] > 1:
            action_diff = action_pred[:, 1:] - action_pred[:, :-1]
            smoothness_loss = torch.mean(action_diff ** 2)
        losses["smoothness"] = smoothness_loss

        # Total loss
        total_loss = (
            self.config.action_loss_weight * action_loss +
            self.config.locomotion_loss_weight * loco_loss +
            self.config.manipulation_loss_weight * manip_loss +
            self.config.stability_loss_weight * stability_loss +
            self.config.smoothness_loss_weight * smoothness_loss
        )
        losses["total"] = total_loss

        return losses

    def _compute_stability_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stability-related loss."""
        stability_loss = torch.tensor(0.0, device=proprioception.device)

        # Penalize high joint velocities
        action_pred = outputs["actions"]
        velocity_penalty = F.relu(torch.abs(action_pred) - self.config.max_joint_velocity)
        stability_loss = stability_loss + velocity_penalty.mean()

        # COM height constraint (if available)
        if "com_height" in outputs:
            com_violation = F.relu(self.config.min_com_height - outputs["com_height"])
            stability_loss = stability_loss + com_violation.mean()

        return stability_loss

    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_action_error = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                image = batch["image"].to(self.device)
                proprioception = batch["proprioception"].to(self.device)
                actions_gt = batch["actions"].to(self.device)
                language_features = batch["language_features"].to(self.device)

                outputs = self.model(
                    image=image,
                    proprioception=proprioception,
                    language_features=language_features,
                )

                losses = self._compute_loss(outputs, actions_gt, proprioception)

                # Action error (L2 norm)
                action_pred = outputs["actions"]
                if action_pred.dim() == 2:
                    action_pred = action_pred.unsqueeze(1)
                action_error = torch.norm(action_pred[:, 0] - actions_gt[:, 0], dim=-1).mean()

                total_loss += losses["total"].item()
                total_action_error += action_error.item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "action_error": total_action_error / num_batches,
        }

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
    parser = argparse.ArgumentParser(description="Train Humanoid VLA")

    # Model args
    parser.add_argument("--num-joints", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=512)

    # Training args
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)

    # Data args
    parser.add_argument("--data-path", type=str, default="./data/humanoid")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)

    # Output args
    parser.add_argument("--output-dir", type=str, default="./checkpoints/humanoid_vla")
    parser.add_argument("--resume", type=str, default=None)

    # Safety args
    parser.add_argument("--use-safety-constraints", action="store_true", default=True)
    parser.add_argument("--max-joint-velocity", type=float, default=5.0)
    parser.add_argument("--max-joint-torque", type=float, default=100.0)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = HumanoidTrainConfig(
        num_joints=args.num_joints,
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        data_path=args.data_path,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir,
        use_safety_constraints=args.use_safety_constraints,
        max_joint_velocity=args.max_joint_velocity,
        max_joint_torque=args.max_joint_torque,
    )

    trainer = HumanoidVLATrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
