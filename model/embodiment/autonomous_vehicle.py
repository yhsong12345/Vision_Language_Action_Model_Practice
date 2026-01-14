"""
Autonomous Vehicle Specific Components

Provides driving-specific modules for VLA:
- BEV (Bird's Eye View) encoding
- Multi-camera fusion
- Trajectory prediction and planning
- Vehicle-specific action heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class DrivingVLAConfig:
    """Configuration for driving VLA."""
    # Vision
    num_cameras: int = 6  # Typical surround-view setup
    image_size: int = 224
    bev_size: int = 200  # BEV grid size
    bev_resolution: float = 0.5  # meters per pixel

    # LLM
    llm_hidden_dim: int = 4096
    use_language_conditioning: bool = True

    # Action
    trajectory_length: int = 20  # Future waypoints
    dt: float = 0.1  # Time between waypoints

    # Architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4


class BEVEncoder(nn.Module):
    """
    Bird's Eye View (BEV) encoder.

    Transforms multi-camera images into unified BEV representation.
    Similar to LSS (Lift-Splat-Shoot) approach.
    """

    def __init__(self, config: DrivingVLAConfig):
        super().__init__()
        self.config = config

        # Image encoder (shared across cameras)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, config.hidden_dim, 3, stride=2, padding=1),
        )

        # Depth estimation head
        self.depth_head = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, 64, 1),  # 64 depth bins
            nn.Softmax(dim=1),
        )

        # BEV projection
        self.bev_conv = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
        )

        # Camera-to-BEV transformation (learned)
        self.cam_to_bev = nn.Parameter(
            torch.randn(config.num_cameras, config.hidden_dim, config.bev_size, config.bev_size) * 0.01
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-camera images to BEV.

        Args:
            images: (batch, num_cameras, 3, H, W)
            intrinsics: (batch, num_cameras, 3, 3) camera intrinsics
            extrinsics: (batch, num_cameras, 4, 4) camera extrinsics

        Returns:
            Dict with BEV features
        """
        batch_size, num_cameras = images.shape[:2]

        # Reshape for batch processing
        images_flat = images.view(-1, *images.shape[2:])

        # Encode images
        features = self.image_encoder(images_flat)  # (B*N, C, H', W')
        features = features.view(batch_size, num_cameras, *features.shape[1:])

        # Estimate depth
        depth_flat = self.depth_head(features.view(-1, *features.shape[2:]))
        depth = depth_flat.view(batch_size, num_cameras, *depth_flat.shape[1:])

        # Project to BEV (simplified - using learned projection)
        bev_features = torch.zeros(
            batch_size, self.config.hidden_dim,
            self.config.bev_size, self.config.bev_size,
            device=images.device,
        )

        for cam_idx in range(num_cameras):
            cam_features = features[:, cam_idx]  # (B, C, H', W')
            # Apply learned camera-specific projection
            proj = self.cam_to_bev[cam_idx]  # (C, bev_h, bev_w)
            bev_features += torch.einsum("bchw,cxy->bcxy", cam_features, proj.softmax(dim=0))

        # Refine BEV features
        bev_features = self.bev_conv(bev_features)

        return {
            "bev_features": bev_features,
            "camera_features": features,
            "depth": depth,
        }


class TrajectoryDecoder(nn.Module):
    """
    Decodes BEV features into future trajectory.

    Predicts waypoints in vehicle coordinate frame.
    """

    def __init__(self, config: DrivingVLAConfig):
        super().__init__()
        self.config = config

        # BEV feature aggregation
        self.bev_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(config.hidden_dim * 64, config.hidden_dim),
            nn.ReLU(),
        )

        # Trajectory prediction (autoregressive)
        self.trajectory_gru = nn.GRU(
            input_size=config.hidden_dim + 2,  # features + previous waypoint
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 2),  # (x, y) in vehicle frame
        )

        # Optional: predict additional attributes
        self.speed_head = nn.Linear(config.hidden_dim, 1)
        self.heading_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        bev_features: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode trajectory from BEV features.

        Args:
            bev_features: (batch, hidden_dim, bev_h, bev_w)
            language_features: (batch, hidden_dim) from LLM

        Returns:
            Dict with trajectory and attributes
        """
        batch_size = bev_features.shape[0]

        # Pool BEV features
        features = self.bev_pool(bev_features)

        # Add language conditioning if available
        if language_features is not None:
            features = features + language_features

        # Autoregressive trajectory prediction
        waypoints = []
        speeds = []
        headings = []

        prev_waypoint = torch.zeros(batch_size, 2, device=bev_features.device)
        hidden = None

        for t in range(self.config.trajectory_length):
            # Input: features + previous waypoint
            gru_input = torch.cat([features, prev_waypoint], dim=-1)
            gru_input = gru_input.unsqueeze(1)

            output, hidden = self.trajectory_gru(gru_input, hidden)
            output = output.squeeze(1)

            # Predict waypoint
            waypoint = self.waypoint_head(output)
            waypoints.append(waypoint)
            prev_waypoint = waypoint

            # Predict attributes
            speeds.append(self.speed_head(output))
            headings.append(self.heading_head(output))

        # Stack predictions
        trajectory = torch.stack(waypoints, dim=1)  # (batch, T, 2)
        speeds = torch.stack(speeds, dim=1).squeeze(-1)  # (batch, T)
        headings = torch.stack(headings, dim=1).squeeze(-1)  # (batch, T)

        return {
            "trajectory": trajectory,
            "speeds": speeds,
            "headings": headings,
        }


class MotionPlanner(nn.Module):
    """
    Neural motion planner for autonomous driving.

    Combines perception, prediction, and planning in an end-to-end manner.
    """

    def __init__(self, config: DrivingVLAConfig):
        super().__init__()
        self.config = config

        # Cost volume prediction
        self.cost_encoder = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, 1, 1),
        )

        # Trajectory sampler (generates candidate trajectories)
        self.num_samples = 256
        self.trajectory_decoder = TrajectoryDecoder(config)

        # Trajectory scorer
        self.trajectory_scorer = nn.Sequential(
            nn.Linear(config.trajectory_length * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        bev_features: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Plan trajectory given BEV features.

        Args:
            bev_features: (batch, hidden_dim, bev_h, bev_w)
            language_features: (batch, hidden_dim) language conditioning
            goal: (batch, 2) optional goal location

        Returns:
            Dict with planned trajectory and cost map
        """
        batch_size = bev_features.shape[0]

        # Compute cost map
        cost_map = self.cost_encoder(bev_features)  # (batch, 1, bev_h, bev_w)

        # Generate trajectory
        traj_output = self.trajectory_decoder(bev_features, language_features)
        trajectory = traj_output["trajectory"]

        # Score trajectory against cost map
        # Sample points along trajectory and lookup cost
        traj_costs = self._compute_trajectory_cost(trajectory, cost_map)

        return {
            "trajectory": trajectory,
            "cost_map": cost_map,
            "trajectory_cost": traj_costs,
            "speeds": traj_output["speeds"],
            "headings": traj_output["headings"],
        }

    def _compute_trajectory_cost(
        self,
        trajectory: torch.Tensor,
        cost_map: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cost of trajectory by sampling from cost map."""
        batch_size = trajectory.shape[0]

        # Normalize trajectory to cost map coordinates
        traj_normalized = trajectory / (self.config.bev_size * self.config.bev_resolution / 2)
        traj_normalized = traj_normalized.clamp(-1, 1)

        # Sample cost map at trajectory points
        traj_grid = traj_normalized.view(batch_size, -1, 1, 2)
        costs = F.grid_sample(cost_map, traj_grid, mode="bilinear", align_corners=True)
        costs = costs.view(batch_size, -1)

        return costs.sum(dim=1)


class DrivingVLA(nn.Module):
    """
    Complete VLA model for autonomous driving.

    Integrates:
    - Multi-camera BEV encoding
    - Language understanding
    - Trajectory prediction
    - Safety constraints
    """

    def __init__(self, config: DrivingVLAConfig):
        super().__init__()
        self.config = config

        # Vision encoding
        self.bev_encoder = BEVEncoder(config)

        # Language encoding (uses pretrained LLM features)
        self.language_projector = nn.Linear(config.llm_hidden_dim, config.hidden_dim)

        # Cross-modal fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            batch_first=True,
        )

        # Motion planning
        self.planner = MotionPlanner(config)

        # Control output
        self.control_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 3),  # throttle, brake, steer
        )

    def forward(
        self,
        images: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for driving VLA.

        Args:
            images: (batch, num_cameras, 3, H, W) surround-view images
            language_features: (batch, seq_len, llm_dim) LLM features
            instruction: Optional text instruction

        Returns:
            Dict with trajectory, controls, and intermediate outputs
        """
        # Encode BEV
        bev_output = self.bev_encoder(images)
        bev_features = bev_output["bev_features"]

        # Process language
        if language_features is not None:
            lang_proj = self.language_projector(language_features)
            lang_pooled = lang_proj.mean(dim=1)  # Pool sequence

            # Cross-attention fusion
            bev_flat = bev_features.flatten(2).permute(0, 2, 1)  # (B, HW, C)
            fused, _ = self.fusion(bev_flat, lang_proj, lang_proj)
            bev_features = fused.permute(0, 2, 1).view_as(bev_features)
        else:
            lang_pooled = None

        # Plan trajectory
        plan_output = self.planner(bev_features, lang_pooled)

        # Generate control commands
        bev_pooled = bev_features.mean(dim=[2, 3])
        if lang_pooled is not None:
            control_features = bev_pooled + lang_pooled
        else:
            control_features = bev_pooled

        controls = self.control_head(control_features)
        controls = torch.sigmoid(controls)  # Normalize to [0, 1]

        return {
            "trajectory": plan_output["trajectory"],
            "controls": controls,
            "bev_features": bev_features,
            "cost_map": plan_output["cost_map"],
            "speeds": plan_output["speeds"],
        }


if __name__ == "__main__":
    config = DrivingVLAConfig(
        num_cameras=6,
        image_size=224,
        bev_size=200,
        hidden_dim=256,
    )

    # Test BEV encoder
    bev_encoder = BEVEncoder(config)
    images = torch.randn(2, 6, 3, 224, 224)  # batch=2, 6 cameras
    bev_output = bev_encoder(images)
    print(f"BEV features shape: {bev_output['bev_features'].shape}")

    # Test trajectory decoder
    traj_decoder = TrajectoryDecoder(config)
    traj_output = traj_decoder(bev_output["bev_features"])
    print(f"Trajectory shape: {traj_output['trajectory'].shape}")

    # Test full driving VLA
    driving_vla = DrivingVLA(config)
    output = driving_vla(images)
    print(f"\nDriving VLA outputs:")
    print(f"  trajectory: {output['trajectory'].shape}")
    print(f"  controls: {output['controls'].shape}")
    print(f"  cost_map: {output['cost_map'].shape}")
