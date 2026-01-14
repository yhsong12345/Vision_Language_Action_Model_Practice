"""
Experiment Management Module

Provides comprehensive experiment tracking and management:
- Experiment configuration and logging
- Metrics tracking with multiple backends (WandB, TensorBoard, CSV)
- Checkpoint management
- Reproducibility tools
"""

import os
import json
import yaml
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict
import csv


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    # Identification
    name: str = "vla_experiment"
    project: str = "vla-training"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Paths
    base_dir: str = "./experiments"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Logging
    use_wandb: bool = False
    use_tensorboard: bool = True
    use_csv: bool = True
    log_freq: int = 100
    save_freq: int = 1000

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Hardware
    device: str = "auto"
    num_gpus: int = 1
    mixed_precision: str = "bf16"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class MetricsLogger:
    """
    Unified metrics logger supporting multiple backends.

    Backends:
    - WandB: Weights & Biases
    - TensorBoard: TensorFlow TensorBoard
    - CSV: Simple CSV logging
    - Console: Print to stdout
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: str,
    ):
        self.config = config
        self.run_dir = run_dir

        # Initialize backends
        self.backends = {}
        self._init_backends()

        # Metric history
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.step = 0

    def _init_backends(self):
        """Initialize logging backends."""
        # WandB
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.project,
                    name=self.config.name,
                    config=self.config.to_dict(),
                    tags=self.config.tags,
                    notes=self.config.notes,
                    dir=self.run_dir,
                )
                self.backends["wandb"] = wandb
                print("WandB initialized")
            except ImportError:
                print("WandB not available")

        # TensorBoard
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(self.run_dir, self.config.log_dir, "tensorboard")
                os.makedirs(log_dir, exist_ok=True)
                self.backends["tensorboard"] = SummaryWriter(log_dir)
                print(f"TensorBoard logging to {log_dir}")
            except ImportError:
                print("TensorBoard not available")

        # CSV
        if self.config.use_csv:
            csv_path = os.path.join(self.run_dir, self.config.log_dir, "metrics.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.csv_path = csv_path
            self.csv_file = None
            self.csv_writer = None
            self.backends["csv"] = True
            print(f"CSV logging to {csv_path}")

    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics to all backends."""
        if step is not None:
            self.step = step
        else:
            self.step += 1

        # Add timestamp
        metrics["timestamp"] = time.time()

        # Update history
        for k, v in metrics.items():
            self.history[k].append(v)

        # Log to backends
        if "wandb" in self.backends:
            self.backends["wandb"].log(metrics, step=self.step)

        if "tensorboard" in self.backends:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.backends["tensorboard"].add_scalar(k, v, self.step)

        if "csv" in self.backends:
            self._log_csv(metrics)

    def _log_csv(self, metrics: Dict[str, float]):
        """Log metrics to CSV file."""
        metrics["step"] = self.step

        # Initialize CSV on first write
        if self.csv_writer is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics.keys()))
            self.csv_writer.writeheader()

        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """Log image to backends."""
        if step is not None:
            self.step = step

        if "wandb" in self.backends:
            import wandb
            self.backends["wandb"].log({tag: wandb.Image(image)}, step=self.step)

        if "tensorboard" in self.backends:
            # Convert to CHW format if needed
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
            self.backends["tensorboard"].add_image(tag, image, self.step)

    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """Log histogram to backends."""
        if step is not None:
            self.step = step

        if "wandb" in self.backends:
            import wandb
            self.backends["wandb"].log({tag: wandb.Histogram(values)}, step=self.step)

        if "tensorboard" in self.backends:
            self.backends["tensorboard"].add_histogram(tag, values, self.step)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of logged metrics."""
        summary = {}
        for k, v in self.history.items():
            if len(v) > 0 and isinstance(v[0], (int, float)):
                summary[f"{k}_mean"] = np.mean(v)
                summary[f"{k}_std"] = np.std(v)
                summary[f"{k}_min"] = np.min(v)
                summary[f"{k}_max"] = np.max(v)
        return summary

    def close(self):
        """Close all backends."""
        if "wandb" in self.backends:
            self.backends["wandb"].finish()

        if "tensorboard" in self.backends:
            self.backends["tensorboard"].close()

        if self.csv_file:
            self.csv_file.close()


class CheckpointManager:
    """Manages model checkpoints with versioning and cleanup."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Dict[str, Any]] = []

        os.makedirs(checkpoint_dir, exist_ok=True)
        self._load_checkpoint_index()

    def _load_checkpoint_index(self):
        """Load existing checkpoint index."""
        index_path = os.path.join(self.checkpoint_dir, "checkpoints.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.checkpoints = json.load(f)

    def _save_checkpoint_index(self):
        """Save checkpoint index."""
        index_path = os.path.join(self.checkpoint_dir, "checkpoints.json")
        with open(index_path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)

    def save(
        self,
        state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint.

        Args:
            state_dict: Model state dict
            step: Training step
            metrics: Optional metrics at checkpoint
            is_best: If True, also save as best checkpoint

        Returns:
            Path to saved checkpoint
        """
        import torch

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step_{step}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        # Save checkpoint
        checkpoint = {
            "step": step,
            "timestamp": timestamp,
            "state_dict": state_dict,
            "metrics": metrics,
        }
        torch.save(checkpoint, path)

        # Update index
        self.checkpoints.append({
            "path": path,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics,
        })
        self._save_checkpoint_index()

        # Save best if needed
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

        # Cleanup old checkpoints
        self._cleanup()

        return path

    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest["path"]):
                os.remove(oldest["path"])
        self._save_checkpoint_index()

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        import torch

        if not self.checkpoints:
            return None

        latest = self.checkpoints[-1]
        return torch.load(latest["path"], map_location="cpu")

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint."""
        import torch

        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            return torch.load(best_path, map_location="cpu")
        return None

    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Load checkpoint at specific step."""
        import torch

        for ckpt in self.checkpoints:
            if ckpt["step"] == step:
                return torch.load(ckpt["path"], map_location="cpu")
        return None


class ExperimentManager:
    """
    Comprehensive experiment management.

    Combines configuration, logging, checkpointing, and reproducibility.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

        # Create run directory
        self.run_id = self._generate_run_id()
        self.run_dir = os.path.join(
            config.base_dir,
            config.project,
            config.name,
            self.run_id,
        )
        os.makedirs(self.run_dir, exist_ok=True)

        # Save config
        config.to_yaml(os.path.join(self.run_dir, "config.yaml"))

        # Initialize components
        self.logger = MetricsLogger(config, self.run_dir)
        self.checkpoint_manager = CheckpointManager(
            os.path.join(self.run_dir, config.checkpoint_dir)
        )

        # Set seeds
        self._set_seeds()

        print(f"Experiment initialized: {self.run_dir}")

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_str = hashlib.md5(f"{self.config.name}{timestamp}".encode()).hexdigest()[:6]
        return f"{timestamp}_{hash_str}"

    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        import random
        import torch

        seed = self.config.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        self.logger.log(metrics, step)

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint."""
        return self.checkpoint_manager.save(state_dict, step, metrics, is_best)

    def load_checkpoint(self, mode: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Load checkpoint.

        Args:
            mode: "latest", "best", or step number as string
        """
        if mode == "latest":
            return self.checkpoint_manager.load_latest()
        elif mode == "best":
            return self.checkpoint_manager.load_best()
        else:
            return self.checkpoint_manager.load_step(int(mode))

    def finish(self):
        """Finish experiment and cleanup."""
        # Save final summary
        summary = self.logger.get_summary()
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.close()
        print(f"Experiment finished: {self.run_dir}")

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "ExperimentManager":
        """Resume experiment from checkpoint."""
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        config = ExperimentConfig.from_yaml(config_path)

        manager = cls(config)
        manager.run_dir = checkpoint_dir

        return manager


def create_experiment(
    name: str,
    project: str = "vla-training",
    **kwargs,
) -> ExperimentManager:
    """Convenience function to create an experiment."""
    config = ExperimentConfig(
        name=name,
        project=project,
        **kwargs,
    )
    return ExperimentManager(config)


if __name__ == "__main__":
    # Test experiment manager
    config = ExperimentConfig(
        name="test_experiment",
        project="vla-testing",
        use_wandb=False,
        use_tensorboard=True,
        use_csv=True,
    )

    manager = ExperimentManager(config)

    # Log some metrics
    for step in range(100):
        metrics = {
            "loss": np.random.random(),
            "accuracy": np.random.random(),
            "learning_rate": 0.001 * (0.99 ** step),
        }
        manager.log(metrics, step)

        # Save checkpoint periodically
        if step % 20 == 0:
            manager.save_checkpoint(
                {"model": "dummy_state"},
                step,
                metrics,
                is_best=(step == 80),
            )

    # Finish experiment
    manager.finish()

    print("\nExperiment manager test complete!")
