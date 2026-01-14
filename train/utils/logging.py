"""
Logging Utilities

Shared logging and metrics tracking for training:
- MetricsTracker: Track and aggregate training metrics
- TrainingLogger: Unified logging interface
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from datetime import datetime
import numpy as np


class MetricsTracker:
    """
    Track and aggregate training metrics.

    Supports windowed averaging and logging to file.
    """

    def __init__(
        self,
        window_size: int = 100,
        log_dir: Optional[str] = None,
    ):
        self.window_size = window_size
        self.log_dir = log_dir
        self.metrics = defaultdict(list)
        self.step_metrics = {}
        self.global_step = 0

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def add(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Add single metric value."""
        self.metrics[key].append(value)

        # Keep window size
        if len(self.metrics[key]) > self.window_size:
            self.metrics[key].pop(0)

        if step is not None:
            self.global_step = step

    def add_dict(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Add multiple metrics at once."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.add(key, value, step)

    def get(self, key: str) -> float:
        """Get latest value for metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return 0.0

    def get_mean(self, key: str) -> float:
        """Get windowed mean for metric."""
        if key in self.metrics and self.metrics[key]:
            return float(np.mean(self.metrics[key]))
        return 0.0

    def get_std(self, key: str) -> float:
        """Get windowed std for metric."""
        if key in self.metrics and self.metrics[key]:
            return float(np.std(self.metrics[key]))
        return 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all tracked metrics."""
        summary = {}
        for key in self.metrics:
            summary[f"{key}_mean"] = self.get_mean(key)
            summary[f"{key}_std"] = self.get_std(key)
            summary[f"{key}_latest"] = self.get(key)
        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.global_step = 0

    def save(self, filename: str = "metrics.json") -> str:
        """Save metrics to JSON file."""
        if self.log_dir is None:
            return ""

        path = os.path.join(self.log_dir, filename)
        data = {
            "global_step": self.global_step,
            "summary": self.get_summary(),
            "raw": {k: list(v) for k, v in self.metrics.items()},
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path


class TrainingLogger:
    """
    Unified training logger.

    Handles console output, file logging, and optional tensorboard/wandb.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(output_dir, "logs")

        os.makedirs(self.log_dir, exist_ok=True)

        # Metrics tracker
        self.metrics = MetricsTracker(log_dir=self.log_dir)

        # Timing
        self.start_time = time.time()
        self.step_times = []

        # Log file
        self.log_file = os.path.join(self.log_dir, "training.log")

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                print("TensorBoard not available")

        # Wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "vla-training",
                    name=self.experiment_name,
                    config=config or {},
                    dir=output_dir,
                )
            except ImportError:
                print("Wandb not available")

        # Save config
        if config:
            self.save_config(config)

    def log(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to all backends."""
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Track in metrics tracker
        self.metrics.add_dict(metrics, step)

        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

        # Wandb
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log single scalar value."""
        self.metrics.add(key, value, step)

        if self.writer:
            self.writer.add_scalar(key, value, step)

        if self.wandb_run:
            import wandb
            wandb.log({key: value}, step=step)

    def print(
        self,
        message: str,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print formatted message to console and log file."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if step is not None:
            elapsed = time.time() - self.start_time
            header = f"[{timestamp}] Step {step} ({elapsed:.0f}s)"
        else:
            header = f"[{timestamp}]"

        if metrics:
            metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            full_message = f"{header} {message} | {metric_str}"
        else:
            full_message = f"{header} {message}"

        print(full_message)

        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(full_message + "\n")

    def print_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Print formatted metrics."""
        elapsed = time.time() - self.start_time
        steps_per_sec = step / max(elapsed, 1)

        lines = [f"Step {step} ({elapsed:.0f}s, {steps_per_sec:.1f} steps/s)"]
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.4f}")

        message = "\n".join(lines)
        print(message)

        with open(self.log_file, "a") as f:
            f.write(message + "\n\n")

    def save_config(self, config: Dict[str, Any]) -> str:
        """Save configuration to file."""
        path = os.path.join(self.output_dir, "config.json")

        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        return path

    def save_metrics(self) -> str:
        """Save current metrics."""
        return self.metrics.save()

    def close(self) -> None:
        """Close all logging backends."""
        if self.writer:
            self.writer.close()

        if self.wandb_run:
            import wandb
            wandb.finish()

        self.save_metrics()
        self.print("Training completed", metrics=self.metrics.get_summary())


if __name__ == "__main__":
    print("Testing logging utilities...")

    # Test MetricsTracker
    tracker = MetricsTracker(window_size=10)
    for i in range(20):
        tracker.add("loss", np.random.rand())
        tracker.add("accuracy", 0.5 + 0.5 * np.random.rand())

    print(f"Loss mean: {tracker.get_mean('loss'):.4f}")
    print(f"Accuracy mean: {tracker.get_mean('accuracy'):.4f}")

    # Test TrainingLogger (minimal)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(output_dir=tmpdir, experiment_name="test")
        logger.log({"loss": 0.5, "accuracy": 0.9}, step=100)
        logger.print("Test message", step=100, metrics={"loss": 0.5})
        logger.close()

    print("\nLogging utilities test passed!")
