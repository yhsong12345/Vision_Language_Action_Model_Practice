"""
World Model Trainer

Training pipeline for latent world models:
- RSSM (Recurrent State-Space Model)
- Latent dynamics learning
- Reward and value prediction
- Imagination-based planning
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

from model.world_model.latent_world_model import LatentWorldModel, WorldModelConfig
from model.world_model.reward_predictor import RewardPredictor, ValuePredictor


@dataclass
class WorldModelTrainConfig:
    """Configuration for world model training."""
    # Model
    state_dim: int = 256
    action_dim: int = 7
    hidden_dim: int = 512
    latent_dim: int = 32
    num_categories: int = 32
    image_size: int = 64

    # RSSM specific
    deterministic_dim: int = 256
    stochastic_dim: int = 32

    # Training
    batch_size: int = 32
    sequence_length: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 100.0
    warmup_steps: int = 1000

    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 1.0
    kl_balance: float = 0.8  # Balance between prior and posterior
    reward_weight: float = 1.0
    value_weight: float = 0.5
    dynamics_weight: float = 1.0

    # KL annealing
    kl_free_bits: float = 1.0
    kl_anneal_steps: int = 10000

    # Data
    data_path: str = "./data/world_model"
    num_workers: int = 8

    # Checkpointing
    output_dir: str = "./checkpoints/world_model"
    save_steps: int = 5000
    eval_steps: int = 1000

    # Imagination
    imagination_horizon: int = 15
    num_imagination_samples: int = 8


class SequenceDataset(Dataset):
    """Dataset for sequence-based world model training."""

    def __init__(
        self,
        data_path: str,
        config: WorldModelTrainConfig,
        split: str = "train",
    ):
        self.data_path = data_path
        self.config = config
        self.split = split

        # Load data
        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List[Dict[str, np.ndarray]]:
        """Load episode data."""
        episodes = []
        episode_dir = os.path.join(self.data_path, self.split)

        if os.path.exists(episode_dir):
            for ep_file in os.listdir(episode_dir):
                if ep_file.endswith(".npz"):
                    ep_path = os.path.join(episode_dir, ep_file)
                    data = np.load(ep_path)
                    episodes.append({
                        "observations": data["observations"],
                        "actions": data["actions"],
                        "rewards": data["rewards"],
                        "dones": data.get("dones", np.zeros(len(data["rewards"]))),
                    })
        else:
            # Generate dummy episodes for testing
            for i in range(100):
                ep_length = np.random.randint(100, 500)
                episodes.append({
                    "observations": np.random.randn(ep_length, 3, self.config.image_size, self.config.image_size).astype(np.float32),
                    "actions": np.random.randn(ep_length, self.config.action_dim).astype(np.float32),
                    "rewards": np.random.randn(ep_length).astype(np.float32),
                    "dones": np.zeros(ep_length, dtype=np.float32),
                })

        return episodes

    def __len__(self) -> int:
        return len(self.episodes) * 10  # Sample multiple sequences per episode

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select episode
        ep_idx = idx % len(self.episodes)
        episode = self.episodes[ep_idx]

        ep_length = len(episode["rewards"])
        seq_length = min(self.config.sequence_length, ep_length)

        # Random starting point
        start_idx = np.random.randint(0, max(1, ep_length - seq_length))
        end_idx = start_idx + seq_length

        return {
            "observations": torch.from_numpy(episode["observations"][start_idx:end_idx]),
            "actions": torch.from_numpy(episode["actions"][start_idx:end_idx]),
            "rewards": torch.from_numpy(episode["rewards"][start_idx:end_idx]),
            "dones": torch.from_numpy(episode["dones"][start_idx:end_idx]),
        }


class WorldModelTrainer:
    """Trainer for latent world models."""

    def __init__(self, config: WorldModelTrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize world model
        model_config = WorldModelConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_categories=config.num_categories,
        )
        self.world_model = LatentWorldModel(model_config).to(self.device)

        # Reward and value predictors
        self.reward_predictor = RewardPredictor(
            state_dim=config.deterministic_dim + config.stochastic_dim * config.num_categories,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.value_predictor = ValuePredictor(
            state_dim=config.deterministic_dim + config.stochastic_dim * config.num_categories,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Combine all parameters
        self.all_params = list(self.world_model.parameters()) + \
                          list(self.reward_predictor.parameters()) + \
                          list(self.value_predictor.parameters())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.all_params,
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
        train_dataset = SequenceDataset(
            self.config.data_path, self.config, split="train"
        )
        val_dataset = SequenceDataset(
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

        print(f"Training world model on {len(train_dataset)} sequences")
        print(f"Model parameters: {sum(p.numel() for p in self.all_params):,}")

        for epoch in range(self.config.num_epochs):
            # Training epoch
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation
            if (epoch + 1) % 5 == 0:
                val_metrics = self._validate(val_loader)
                print(
                    f"Epoch {epoch+1}: "
                    f"train_loss={train_metrics['total_loss']:.4f}, "
                    f"val_loss={val_metrics['total_loss']:.4f}, "
                    f"recon={val_metrics['recon_loss']:.4f}, "
                    f"kl={val_metrics['kl_loss']:.4f}"
                )

                # Save best model
                if val_metrics["total_loss"] < self.best_loss:
                    self.best_loss = val_metrics["total_loss"]
                    self._save_checkpoint("best_model.pt")

            # Update scheduler
            self.scheduler.step()

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        print("Training completed!")

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.world_model.train()
        self.reward_predictor.train()
        self.value_predictor.train()

        metrics = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "reward_loss": 0.0,
            "dynamics_loss": 0.0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Move to device
            observations = batch["observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            rewards = batch["rewards"].to(self.device)

            # Forward pass through world model
            wm_outputs = self.world_model(observations, actions)

            # Compute losses
            losses = self._compute_losses(wm_outputs, observations, rewards)

            # Backward pass
            total_loss = losses["total"]
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, self.config.max_grad_norm)
            self.optimizer.step()

            self.global_step += 1

            # Update metrics
            for k, v in losses.items():
                if k in metrics:
                    metrics[k] += v.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": metrics["total_loss"] / num_batches,
                "recon": metrics["recon_loss"] / num_batches,
            })

        # Average metrics
        for k in metrics:
            metrics[k] /= num_batches

        return metrics

    def _compute_losses(
        self,
        wm_outputs: Dict[str, torch.Tensor],
        observations: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all training losses."""
        losses = {}

        # Reconstruction loss
        recon_obs = wm_outputs.get("reconstructed_obs", None)
        if recon_obs is not None:
            recon_loss = F.mse_loss(recon_obs, observations)
        else:
            recon_loss = torch.tensor(0.0, device=observations.device)
        losses["recon_loss"] = recon_loss

        # KL divergence loss with free bits
        kl_loss = wm_outputs.get("kl_loss", torch.tensor(0.0, device=observations.device))

        # Apply KL free bits
        kl_loss = torch.clamp(kl_loss, min=self.config.kl_free_bits)

        # KL annealing
        kl_weight = self.config.kl_weight * min(1.0, self.global_step / self.config.kl_anneal_steps)
        losses["kl_loss"] = kl_loss

        # Reward prediction loss
        states = wm_outputs.get("states", None)
        if states is not None:
            # Flatten batch and time dimensions
            batch_size, seq_len = states.shape[:2]
            states_flat = states.view(-1, states.shape[-1])
            rewards_flat = rewards.view(-1, 1)

            reward_pred = self.reward_predictor(states_flat)
            reward_loss = F.mse_loss(reward_pred, rewards_flat)
        else:
            reward_loss = torch.tensor(0.0, device=observations.device)
        losses["reward_loss"] = reward_loss

        # Dynamics consistency loss
        dynamics_loss = wm_outputs.get("dynamics_loss", torch.tensor(0.0, device=observations.device))
        losses["dynamics_loss"] = dynamics_loss

        # Total loss
        total_loss = (
            self.config.reconstruction_weight * recon_loss +
            kl_weight * kl_loss +
            self.config.reward_weight * reward_loss +
            self.config.dynamics_weight * dynamics_loss
        )
        losses["total"] = total_loss
        losses["total_loss"] = total_loss

        return losses

    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.world_model.eval()
        self.reward_predictor.eval()

        metrics = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "reward_loss": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                observations = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                rewards = batch["rewards"].to(self.device)

                wm_outputs = self.world_model(observations, actions)
                losses = self._compute_losses(wm_outputs, observations, rewards)

                for k, v in losses.items():
                    if k in metrics:
                        metrics[k] += v.item()
                num_batches += 1

        for k in metrics:
            metrics[k] /= num_batches

        return metrics

    def imagine(
        self,
        initial_state: torch.Tensor,
        policy: nn.Module,
        horizon: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Imagine trajectories using the world model."""
        if horizon is None:
            horizon = self.config.imagination_horizon

        self.world_model.eval()

        with torch.no_grad():
            states = [initial_state]
            actions = []
            rewards = []

            state = initial_state
            for t in range(horizon):
                # Get action from policy
                action = policy(state)
                actions.append(action)

                # Predict next state
                next_state = self.world_model.imagine_step(state, action)
                states.append(next_state)

                # Predict reward
                reward = self.reward_predictor(next_state)
                rewards.append(reward)

                state = next_state

        return {
            "states": torch.stack(states, dim=1),
            "actions": torch.stack(actions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
        }

    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, filename)

        checkpoint = {
            "world_model_state_dict": self.world_model.state_dict(),
            "reward_predictor_state_dict": self.reward_predictor.state_dict(),
            "value_predictor_state_dict": self.value_predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        self.reward_predictor.load_state_dict(checkpoint["reward_predictor_state_dict"])
        self.value_predictor.load_state_dict(checkpoint["value_predictor_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        print(f"Checkpoint loaded from {path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train World Model")

    # Model args
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)

    # Training args
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=100)

    # Loss weights
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)

    # Data args
    parser.add_argument("--data-path", type=str, default="./data/world_model")
    parser.add_argument("--num-workers", type=int, default=8)

    # Output args
    parser.add_argument("--output-dir", type=str, default="./checkpoints/world_model")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = WorldModelTrainConfig(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        kl_weight=args.kl_weight,
        reconstruction_weight=args.reconstruction_weight,
        data_path=args.data_path,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    trainer = WorldModelTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
