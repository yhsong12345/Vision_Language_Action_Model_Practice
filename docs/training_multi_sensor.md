# Training VLA with Multi-Sensor Fusion

This comprehensive guide covers the complete training process for Vision-Language-Action models with multi-sensor fusion capabilities, including RGB-D, LiDAR, Radar, IMU, and tactile sensor integration.

## Table of Contents

1. [Overview](#overview)
2. [Multi-Sensor Architecture](#multi-sensor-architecture)
3. [Sensor Encoders](#sensor-encoders)
4. [Stage 1: Individual Sensor Pretraining](#stage-1-individual-sensor-pretraining)
5. [Stage 2: Cross-Modal Alignment](#stage-2-cross-modal-alignment)
6. [Stage 3: Fusion Training](#stage-3-fusion-training)
7. [Stage 4: Task-Specific Fine-tuning](#stage-4-task-specific-fine-tuning)
8. [Stage 5: Robustness Training](#stage-5-robustness-training)
9. [Sensor-Specific Configurations](#sensor-specific-configurations)
10. [Advanced Topics](#advanced-topics)
11. [Deployment](#deployment)
12. [Evaluation and Benchmarks](#evaluation-and-benchmarks)

---

## Overview

### Multi-Sensor VLA Pipeline

```
+=======================================================================================+
|                        MULTI-SENSOR VLA TRAINING PIPELINE                              |
+=======================================================================================+
|                                                                                        |
|  SENSOR INPUTS                                                                         |
|  +-----------------------------------------------------------------------------------+ |
|  |  RGB Camera(s)  |  Depth Camera  |  LiDAR  |  Radar  |  IMU  |  Tactile  |  Audio | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  SENSOR ENCODERS                                                                       |
|  +-----------------------------------------------------------------------------------+ |
|  |  Vision Encoder  |  Depth Encoder  |  PointNet++  |  RadarNet  |  IMU LSTM       | |
|  |  (SigLIP/CLIP)   |  (CNN/ViT)      |  (VoxelNet)  |  (Range-Doppler)  | (MLP)    | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  CROSS-MODAL ALIGNMENT                                                                 |
|  +-----------------------------------------------------------------------------------+ |
|  |  Contrastive Learning  |  Cross-Attention  |  Shared Latent Space                 | |
|  |  (CLIP-style)          |  (Transformer)    |  (VAE/VQ-VAE)                         | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  MULTI-MODAL FUSION                                                                    |
|  +-----------------------------------------------------------------------------------+ |
|  |  Early Fusion: Concatenate sensor features                                         | |
|  |  Late Fusion: Combine predictions                                                  | |
|  |  Cross-Modal Fusion: Attention between modalities                                  | |
|  |  BEV Fusion: Project to unified BEV space                                          | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  VLM BACKBONE (Language-Conditioned)                                                   |
|  +-----------------------------------------------------------------------------------+ |
|  |  Vision-Language Model with Multi-Sensor Context                                   | |
|  |  - Processes: Natural language instructions                                        | |
|  |  - Integrates: All sensor modalities                                               | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  ACTION OUTPUT                                                                         |
|  +-----------------------------------------------------------------------------------+ |
|  |  Action Head: MLP / Gaussian / Diffusion                                           | |
|  |  - Robot actions informed by all sensors                                           | |
|  +-----------------------------------------------------------------------------------+ |
|                                                                                        |
+=======================================================================================+
```

### Sensor Comparison

| Sensor | Information Type | Range | Resolution | Conditions | Latency |
|--------|-----------------|-------|------------|------------|---------|
| **RGB Camera** | Color, texture, semantics | 0.5-100m | High | Light-dependent | Low |
| **Depth Camera** | Distance, 3D structure | 0.3-10m | Medium | Indoor, no sunlight | Low |
| **LiDAR** | 3D point cloud, distance | 1-200m | High | All weather | Medium |
| **Radar** | Velocity, distance | 1-300m | Low | All weather | Medium |
| **IMU** | Acceleration, orientation | - | High temporal | All conditions | Very low |
| **Tactile** | Contact, force, texture | Contact | Very high | Contact required | Very low |
| **Audio** | Sound, speech | 0-30m | - | Noise sensitive | Low |

---

## Multi-Sensor Architecture

### MultiSensorVLA Configuration

```python
from model.vla import MultiSensorVLA, MultiSensorVLAConfig

@dataclass
class MultiSensorVLAConfig:
    # Vision-Language Model
    vlm_backbone: str = "Qwen/Qwen2-1.5B-Instruct"
    vision_encoder: str = "google/siglip-base-patch16-224"

    # RGB Camera Configuration
    num_cameras: int = 1
    image_size: int = 224
    rgb_feature_dim: int = 768

    # Depth Camera Configuration
    use_depth: bool = True
    depth_encoder_type: str = "cnn"  # cnn, vit, dpt
    depth_feature_dim: int = 256
    depth_range: Tuple[float, float] = (0.0, 10.0)  # meters

    # LiDAR Configuration
    use_lidar: bool = False
    lidar_encoder_type: str = "pointnet++"  # pointnet++, voxelnet, pillarnet
    lidar_feature_dim: int = 512
    lidar_range: float = 50.0  # meters
    num_lidar_points: int = 100000

    # Radar Configuration
    use_radar: bool = False
    radar_encoder_type: str = "range_doppler"  # range_doppler, point_target
    radar_feature_dim: int = 256

    # IMU Configuration
    use_imu: bool = False
    imu_sequence_length: int = 10
    imu_feature_dim: int = 64

    # Tactile Configuration
    use_tactile: bool = False
    tactile_encoder_type: str = "cnn"  # cnn, transformer
    tactile_feature_dim: int = 128
    num_tactile_sensors: int = 2

    # Audio Configuration
    use_audio: bool = False
    audio_encoder_type: str = "wav2vec"  # wav2vec, mel_spectrogram
    audio_feature_dim: int = 256

    # Fusion Configuration
    fusion_type: str = "cross_modal"  # early, late, cross_modal, bev
    fusion_dim: int = 512
    num_fusion_layers: int = 4

    # Action Output
    action_dim: int = 7
    action_head_type: str = "gaussian"


class MultiSensorVLA(nn.Module):
    """
    VLA model with multi-sensor fusion.

    Supports flexible sensor configurations for different robot setups.
    """

    def __init__(self, config: MultiSensorVLAConfig):
        super().__init__()
        self.config = config

        # ====== Sensor Encoders ======

        # RGB encoder
        self.rgb_encoder = VisionEncoder(
            model_name=config.vision_encoder,
            output_dim=config.rgb_feature_dim,
        )

        # Depth encoder
        if config.use_depth:
            self.depth_encoder = DepthEncoder(
                encoder_type=config.depth_encoder_type,
                output_dim=config.depth_feature_dim,
                depth_range=config.depth_range,
            )

        # LiDAR encoder
        if config.use_lidar:
            self.lidar_encoder = LiDAREncoder(
                encoder_type=config.lidar_encoder_type,
                output_dim=config.lidar_feature_dim,
                num_points=config.num_lidar_points,
            )

        # Radar encoder
        if config.use_radar:
            self.radar_encoder = RadarEncoder(
                encoder_type=config.radar_encoder_type,
                output_dim=config.radar_feature_dim,
            )

        # IMU encoder
        if config.use_imu:
            self.imu_encoder = IMUEncoder(
                sequence_length=config.imu_sequence_length,
                output_dim=config.imu_feature_dim,
            )

        # Tactile encoder
        if config.use_tactile:
            self.tactile_encoder = TactileEncoder(
                encoder_type=config.tactile_encoder_type,
                output_dim=config.tactile_feature_dim,
                num_sensors=config.num_tactile_sensors,
            )

        # Audio encoder
        if config.use_audio:
            self.audio_encoder = AudioEncoder(
                encoder_type=config.audio_encoder_type,
                output_dim=config.audio_feature_dim,
            )

        # ====== Fusion Module ======
        self.fusion = self._build_fusion_module(config)

        # ====== VLM Backbone ======
        self.vlm = VLMModel(
            llm_model_name=config.vlm_backbone,
            vision_dim=config.fusion_dim,
        )

        # ====== Action Head ======
        self.action_head = self._build_action_head(config)

    def _build_fusion_module(self, config: MultiSensorVLAConfig) -> nn.Module:
        """Build fusion module based on configuration."""
        # Calculate total feature dimension
        total_dim = config.rgb_feature_dim
        if config.use_depth:
            total_dim += config.depth_feature_dim
        if config.use_lidar:
            total_dim += config.lidar_feature_dim
        if config.use_radar:
            total_dim += config.radar_feature_dim
        if config.use_imu:
            total_dim += config.imu_feature_dim
        if config.use_tactile:
            total_dim += config.tactile_feature_dim
        if config.use_audio:
            total_dim += config.audio_feature_dim

        if config.fusion_type == "early":
            return EarlyFusion(
                input_dim=total_dim,
                output_dim=config.fusion_dim,
            )
        elif config.fusion_type == "late":
            return LateFusion(
                modality_dims={
                    "rgb": config.rgb_feature_dim,
                    "depth": config.depth_feature_dim if config.use_depth else 0,
                    "lidar": config.lidar_feature_dim if config.use_lidar else 0,
                    "radar": config.radar_feature_dim if config.use_radar else 0,
                    "imu": config.imu_feature_dim if config.use_imu else 0,
                    "tactile": config.tactile_feature_dim if config.use_tactile else 0,
                    "audio": config.audio_feature_dim if config.use_audio else 0,
                },
                output_dim=config.fusion_dim,
            )
        elif config.fusion_type == "cross_modal":
            return CrossModalFusion(
                feature_dim=config.fusion_dim,
                num_heads=8,
                num_layers=config.num_fusion_layers,
            )
        elif config.fusion_type == "bev":
            return BEVFusion(
                feature_dim=config.fusion_dim,
                bev_size=(200, 200),
            )

    def forward(
        self,
        rgb_images: torch.Tensor,           # (B, C, H, W) or (B, N, C, H, W)
        depth_images: Optional[torch.Tensor] = None,  # (B, 1, H, W)
        lidar_points: Optional[torch.Tensor] = None,  # (B, N, 4)
        radar_data: Optional[torch.Tensor] = None,    # (B, M, 5)
        imu_data: Optional[torch.Tensor] = None,      # (B, T, 6)
        tactile_data: Optional[torch.Tensor] = None,  # (B, num_sensors, ...)
        audio_data: Optional[torch.Tensor] = None,    # (B, audio_len)
        instruction: str = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-sensor inputs.

        All sensor inputs are optional except RGB.
        """
        sensor_features = {}

        # Encode RGB
        rgb_feat = self.rgb_encoder(rgb_images)
        sensor_features["rgb"] = rgb_feat

        # Encode depth
        if self.config.use_depth and depth_images is not None:
            depth_feat = self.depth_encoder(depth_images)
            sensor_features["depth"] = depth_feat

        # Encode LiDAR
        if self.config.use_lidar and lidar_points is not None:
            lidar_feat = self.lidar_encoder(lidar_points)
            sensor_features["lidar"] = lidar_feat

        # Encode Radar
        if self.config.use_radar and radar_data is not None:
            radar_feat = self.radar_encoder(radar_data)
            sensor_features["radar"] = radar_feat

        # Encode IMU
        if self.config.use_imu and imu_data is not None:
            imu_feat = self.imu_encoder(imu_data)
            sensor_features["imu"] = imu_feat

        # Encode tactile
        if self.config.use_tactile and tactile_data is not None:
            tactile_feat = self.tactile_encoder(tactile_data)
            sensor_features["tactile"] = tactile_feat

        # Encode audio
        if self.config.use_audio and audio_data is not None:
            audio_feat = self.audio_encoder(audio_data)
            sensor_features["audio"] = audio_feat

        # Fuse sensor features
        fused_features = self.fusion(sensor_features)

        # VLM processing with instruction
        if instruction is not None:
            vlm_features = self.vlm(
                visual_features=fused_features,
                instruction=instruction,
            )
        else:
            vlm_features = fused_features

        # Action prediction
        action_output = self.action_head(vlm_features)

        return {
            "fused_features": fused_features,
            "vlm_features": vlm_features,
            "action": action_output["action"],
            "sensor_features": sensor_features,
        }
```

---

## Sensor Encoders

### Depth Encoder

```python
class DepthEncoder(nn.Module):
    """
    Encode depth images into feature representations.

    Types:
    1. CNN: Standard convolutional encoder
    2. ViT: Vision Transformer for depth
    3. DPT: Dense Prediction Transformer
    """

    def __init__(
        self,
        encoder_type: str = "cnn",
        output_dim: int = 256,
        depth_range: Tuple[float, float] = (0.0, 10.0),
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.depth_range = depth_range

        if encoder_type == "cnn":
            # ResNet-style encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                self._make_layer(64, 64, 2),
                self._make_layer(64, 128, 2, stride=2),
                self._make_layer(128, 256, 2, stride=2),
                self._make_layer(256, 512, 2, stride=2),

                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, output_dim),
            )

        elif encoder_type == "vit":
            # Vision Transformer for depth
            self.patch_embed = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=768, nhead=12),
                num_layers=12,
            )
            self.head = nn.Linear(768, output_dim)

        elif encoder_type == "dpt":
            # Dense Prediction Transformer
            from transformers import DPTModel
            self.encoder = DPTModel.from_pretrained(
                "Intel/dpt-large",
                num_channels=1,
            )
            self.head = nn.Linear(1024, output_dim)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create residual layer."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Encode depth image.

        Args:
            depth: (B, 1, H, W) depth image in meters

        Returns:
            features: (B, output_dim) encoded features
        """
        # Normalize depth
        depth = (depth - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
        depth = torch.clamp(depth, 0.0, 1.0)

        if self.encoder_type == "cnn":
            features = self.encoder(depth)

        elif self.encoder_type == "vit":
            # Patch embedding
            patches = self.patch_embed(depth)  # (B, 768, H/16, W/16)
            patches = patches.flatten(2).permute(2, 0, 1)  # (num_patches, B, 768)

            # Add CLS token
            cls_token = torch.zeros(1, patches.shape[1], 768, device=depth.device)
            patches = torch.cat([cls_token, patches], dim=0)

            # Transformer encoding
            features = self.transformer(patches)
            features = self.head(features[0])  # CLS token

        elif self.encoder_type == "dpt":
            outputs = self.encoder(depth)
            features = self.head(outputs.last_hidden_state.mean(dim=1))

        return features


class RGBDEncoder(nn.Module):
    """
    Joint RGB-D encoder for manipulation tasks.

    Combines RGB and depth through:
    1. Early fusion: Concatenate channels
    2. Late fusion: Separate encoders, combine features
    3. Cross-attention: Attention between modalities
    """

    def __init__(
        self,
        fusion_type: str = "late",
        output_dim: int = 512,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "early":
            # 4-channel input (RGB + D)
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # ... more layers
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, output_dim),
            )

        elif fusion_type == "late":
            # Separate encoders
            self.rgb_encoder = VisionEncoder(output_dim=256)
            self.depth_encoder = DepthEncoder(output_dim=256)
            self.fusion = nn.Sequential(
                nn.Linear(512, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )

        elif fusion_type == "cross_attention":
            self.rgb_encoder = VisionEncoder(output_dim=256)
            self.depth_encoder = DepthEncoder(output_dim=256)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=8,
                batch_first=True,
            )
            self.head = nn.Linear(256, output_dim)

    def forward(
        self,
        rgb: torch.Tensor,    # (B, 3, H, W)
        depth: torch.Tensor,  # (B, 1, H, W)
    ) -> torch.Tensor:
        """Encode RGB-D input."""
        if self.fusion_type == "early":
            x = torch.cat([rgb, depth], dim=1)
            return self.encoder(x)

        elif self.fusion_type == "late":
            rgb_feat = self.rgb_encoder(rgb)
            depth_feat = self.depth_encoder(depth)
            combined = torch.cat([rgb_feat, depth_feat], dim=-1)
            return self.fusion(combined)

        elif self.fusion_type == "cross_attention":
            rgb_feat = self.rgb_encoder(rgb).unsqueeze(1)
            depth_feat = self.depth_encoder(depth).unsqueeze(1)

            # RGB attends to depth
            attended, _ = self.cross_attention(rgb_feat, depth_feat, depth_feat)
            return self.head(attended.squeeze(1))
```

### LiDAR Encoder

```python
class LiDAREncoder(nn.Module):
    """
    Encode LiDAR point clouds.

    Types:
    1. PointNet++: Hierarchical point cloud learning
    2. VoxelNet: Voxel-based representation
    3. PillarNet: Pillar-based (fast for driving)
    4. PointTransformer: Attention on points
    """

    def __init__(
        self,
        encoder_type: str = "pointnet++",
        output_dim: int = 512,
        num_points: int = 100000,
        point_dim: int = 4,  # x, y, z, intensity
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "pointnet++":
            self.encoder = PointNet2Encoder(
                input_dim=point_dim,
                output_dim=output_dim,
            )

        elif encoder_type == "voxelnet":
            self.encoder = VoxelNetEncoder(
                voxel_size=[0.2, 0.2, 0.4],
                point_cloud_range=[-50, -50, -3, 50, 50, 5],
                output_dim=output_dim,
            )

        elif encoder_type == "pillarnet":
            self.encoder = PillarNetEncoder(
                pillar_size=[0.2, 0.2],
                point_cloud_range=[-50, -50, -3, 50, 50, 5],
                output_dim=output_dim,
            )

        elif encoder_type == "point_transformer":
            self.encoder = PointTransformerEncoder(
                input_dim=point_dim,
                output_dim=output_dim,
                num_heads=8,
                num_layers=4,
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud.

        Args:
            points: (B, N, 4) point cloud (x, y, z, intensity)

        Returns:
            features: (B, output_dim) encoded features
        """
        return self.encoder(points)


class PointNet2Encoder(nn.Module):
    """
    PointNet++ encoder with set abstraction layers.

    Reference: "PointNet++: Deep Hierarchical Feature Learning on Point Sets
               in a Metric Space" (Qi et al., 2017)
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 512,
        num_points: List[int] = [16384, 4096, 1024, 256],
        radii: List[float] = [0.5, 1.0, 2.0, 4.0],
        k_neighbors: int = 32,
    ):
        super().__init__()

        # Set Abstraction layers
        self.sa1 = SetAbstractionLayer(
            num_points=num_points[0],
            radius=radii[0],
            k_neighbors=k_neighbors,
            in_channels=input_dim,
            mlp_channels=[32, 32, 64],
        )

        self.sa2 = SetAbstractionLayer(
            num_points=num_points[1],
            radius=radii[1],
            k_neighbors=k_neighbors,
            in_channels=64 + 3,  # +3 for xyz
            mlp_channels=[64, 64, 128],
        )

        self.sa3 = SetAbstractionLayer(
            num_points=num_points[2],
            radius=radii[2],
            k_neighbors=k_neighbors,
            in_channels=128 + 3,
            mlp_channels=[128, 128, 256],
        )

        self.sa4 = SetAbstractionLayer(
            num_points=num_points[3],
            radius=radii[3],
            k_neighbors=k_neighbors,
            in_channels=256 + 3,
            mlp_channels=[256, 256, 512],
        )

        # Global pooling and output
        self.global_pool = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 4) input point cloud

        Returns:
            features: (B, output_dim) global features
        """
        xyz = points[:, :, :3]
        features = points

        # Hierarchical abstraction
        xyz1, features1 = self.sa1(xyz, features)
        xyz2, features2 = self.sa2(xyz1, features1)
        xyz3, features3 = self.sa3(xyz2, features2)
        xyz4, features4 = self.sa4(xyz3, features3)

        # Global max pooling
        global_feat = features4.max(dim=1)[0]  # (B, 512)

        return self.global_pool(global_feat)


class SetAbstractionLayer(nn.Module):
    """Set Abstraction layer for PointNet++."""

    def __init__(
        self,
        num_points: int,
        radius: float,
        k_neighbors: int,
        in_channels: int,
        mlp_channels: List[int],
    ):
        super().__init__()
        self.num_points = num_points
        self.radius = radius
        self.k_neighbors = k_neighbors

        # PointNet MLP
        layers = []
        prev_channels = in_channels
        for channels in mlp_channels:
            layers.extend([
                nn.Conv1d(prev_channels, channels, 1),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ])
            prev_channels = channels
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,      # (B, N, 3)
        features: torch.Tensor,  # (B, N, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            new_xyz: (B, num_points, 3) sampled points
            new_features: (B, num_points, C') aggregated features
        """
        B, N, _ = xyz.shape

        # Farthest point sampling
        new_xyz = self._farthest_point_sample(xyz, self.num_points)

        # Ball query
        idx = self._ball_query(xyz, new_xyz, self.radius, self.k_neighbors)

        # Group features
        grouped_xyz = self._index_points(xyz, idx)  # (B, num_points, k, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # Relative coordinates

        grouped_features = self._index_points(features, idx)  # (B, num_points, k, C)
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)

        # Apply MLP
        grouped_features = grouped_features.permute(0, 3, 1, 2)  # (B, C+3, num_points, k)
        grouped_features = grouped_features.reshape(B, -1, self.num_points * self.k_neighbors)
        new_features = self.mlp(grouped_features)
        new_features = new_features.reshape(B, -1, self.num_points, self.k_neighbors)

        # Max pooling over neighbors
        new_features = new_features.max(dim=-1)[0].permute(0, 2, 1)  # (B, num_points, C')

        return new_xyz, new_features

    def _farthest_point_sample(self, xyz: torch.Tensor, num_points: int) -> torch.Tensor:
        """Farthest point sampling."""
        B, N, _ = xyz.shape
        device = xyz.device

        centroids = torch.zeros(B, num_points, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10

        # Random initial point
        farthest = torch.randint(0, N, (B,), device=device)

        for i in range(num_points):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            distance = torch.min(distance, dist)
            farthest = torch.argmax(distance, dim=-1)

        return self._index_points(xyz, centroids)

    def _ball_query(
        self,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        radius: float,
        k: int,
    ) -> torch.Tensor:
        """Ball query for grouping."""
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        device = xyz.device

        # Compute pairwise distances
        dist = torch.cdist(new_xyz, xyz)  # (B, S, N)

        # Find k nearest within radius
        dist[dist > radius] = float('inf')
        _, idx = dist.topk(k, dim=-1, largest=False)

        return idx

    def _index_points(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Index points by indices."""
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points
```

### IMU Encoder

```python
class IMUEncoder(nn.Module):
    """
    Encode IMU (Inertial Measurement Unit) data.

    IMU provides:
    - Accelerometer: 3-axis linear acceleration
    - Gyroscope: 3-axis angular velocity
    - (Optional) Magnetometer: 3-axis magnetic field
    """

    def __init__(
        self,
        input_dim: int = 6,  # 3 accel + 3 gyro
        sequence_length: int = 10,
        output_dim: int = 64,
        encoder_type: str = "lstm",  # lstm, transformer, tcn
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.head = nn.Linear(256, output_dim)

        elif encoder_type == "transformer":
            self.pos_encoding = SinusoidalPositionalEncoding(input_dim, sequence_length)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_dim, nhead=2),
                num_layers=2,
            )
            self.head = nn.Linear(input_dim, output_dim)

        elif encoder_type == "tcn":
            self.encoder = TemporalConvNet(
                num_inputs=input_dim,
                num_channels=[32, 64, 128],
                kernel_size=3,
            )
            self.head = nn.Linear(128, output_dim)

        elif encoder_type == "mlp":
            # Simple MLP for current IMU reading
            self.encoder = nn.Sequential(
                nn.Linear(input_dim * sequence_length, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Encode IMU sequence.

        Args:
            imu_data: (B, T, 6) IMU readings (accel + gyro)

        Returns:
            features: (B, output_dim) encoded features
        """
        if self.encoder_type == "lstm":
            output, (h_n, c_n) = self.encoder(imu_data)
            # Use last hidden state from both directions
            features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            return self.head(features)

        elif self.encoder_type == "transformer":
            imu_data = imu_data + self.pos_encoding(imu_data)
            imu_data = imu_data.permute(1, 0, 2)  # (T, B, C)
            output = self.encoder(imu_data)
            features = output.mean(dim=0)  # Average pooling
            return self.head(features)

        elif self.encoder_type == "tcn":
            imu_data = imu_data.permute(0, 2, 1)  # (B, C, T)
            output = self.encoder(imu_data)
            features = output[:, :, -1]  # Last timestep
            return self.head(features)

        elif self.encoder_type == "mlp":
            imu_flat = imu_data.reshape(imu_data.shape[0], -1)
            return self.encoder(imu_flat)
```

### Tactile Encoder

```python
class TactileEncoder(nn.Module):
    """
    Encode tactile sensor data.

    Supports:
    1. Pressure sensors: Spatial pressure distribution
    2. BioTac: Multi-modal tactile (pressure, vibration, temperature)
    3. GelSight: Vision-based tactile (high-resolution images)
    4. Tactile arrays: Grid of pressure sensors
    """

    def __init__(
        self,
        encoder_type: str = "cnn",  # cnn, transformer, pointnet
        tactile_type: str = "pressure_grid",  # pressure_grid, biotac, gelsight
        output_dim: int = 128,
        num_sensors: int = 2,  # e.g., 2 fingers
        grid_size: Tuple[int, int] = (16, 16),  # For pressure grid
    ):
        super().__init__()
        self.tactile_type = tactile_type
        self.num_sensors = num_sensors

        if tactile_type == "pressure_grid":
            # Grid of pressure sensors
            if encoder_type == "cnn":
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.head = nn.Linear(128 * num_sensors, output_dim)

            elif encoder_type == "transformer":
                self.patch_embed = nn.Linear(grid_size[0] * grid_size[1], 128)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=128, nhead=4),
                    num_layers=2,
                )
                self.head = nn.Linear(128 * num_sensors, output_dim)

        elif tactile_type == "biotac":
            # BioTac multi-modal: 19 electrodes + pressure + temperature
            input_dim = 19 + 1 + 1  # electrodes + DC pressure + temperature
            self.encoder = nn.Sequential(
                nn.Linear(input_dim * num_sensors, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

        elif tactile_type == "gelsight":
            # GelSight vision-based tactile
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.head = nn.Linear(128 * num_sensors, output_dim)

    def forward(self, tactile_data: torch.Tensor) -> torch.Tensor:
        """
        Encode tactile data.

        Args:
            tactile_data: Depends on tactile_type
                - pressure_grid: (B, num_sensors, H, W)
                - biotac: (B, num_sensors, 21)
                - gelsight: (B, num_sensors, 3, H, W)

        Returns:
            features: (B, output_dim) encoded features
        """
        B = tactile_data.shape[0]

        if self.tactile_type == "pressure_grid":
            # Encode each sensor separately
            sensor_features = []
            for i in range(self.num_sensors):
                sensor_data = tactile_data[:, i:i+1]  # (B, 1, H, W)
                feat = self.encoder(sensor_data)
                sensor_features.append(feat)
            combined = torch.cat(sensor_features, dim=-1)
            return self.head(combined)

        elif self.tactile_type == "biotac":
            # Flatten all sensor data
            flat_data = tactile_data.reshape(B, -1)
            return self.encoder(flat_data)

        elif self.tactile_type == "gelsight":
            # Encode each sensor image separately
            sensor_features = []
            for i in range(self.num_sensors):
                sensor_img = tactile_data[:, i]  # (B, 3, H, W)
                feat = self.encoder(sensor_img)
                sensor_features.append(feat)
            combined = torch.cat(sensor_features, dim=-1)
            return self.head(combined)
```

---

## Stage 1: Individual Sensor Pretraining

### RGB Pretraining

```python
class RGBPretrainer:
    """
    Pretrain RGB encoder on robot-specific visual data.

    Methods:
    1. ImageNet pretraining (standard)
    2. Contrastive learning (SimCLR, MoCo)
    3. Masked image modeling (MAE)
    4. Robot-specific pretraining (in-domain)
    """

    def __init__(
        self,
        encoder: VisionEncoder,
        config: PretrainingConfig,
    ):
        self.encoder = encoder
        self.config = config

    def pretrain_contrastive(
        self,
        dataset: RobotImageDataset,
        num_epochs: int = 100,
    ):
        """SimCLR-style contrastive pretraining."""
        # Projection head
        projector = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(projector.parameters()),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                images = batch["images"]  # (B, 3, H, W)

                # Create two augmented views
                view1 = self.augment(images)
                view2 = self.augment(images)

                # Encode both views
                z1 = projector(self.encoder(view1))
                z2 = projector(self.encoder(view2))

                # NT-Xent loss
                loss = self._nt_xent_loss(z1, z2, temperature=0.5)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Contrastive Loss = {loss.item():.4f}")

    def _nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss."""
        B = z1.shape[0]

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity
        z = torch.cat([z1, z2], dim=0)  # (2B, dim)
        sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -float('inf'))

        # Labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


class DepthPretrainer:
    """
    Pretrain depth encoder.

    Methods:
    1. Depth prediction from RGB (self-supervised)
    2. Depth completion (predict missing depth)
    3. Contrastive depth-RGB alignment
    """

    def __init__(
        self,
        depth_encoder: DepthEncoder,
        rgb_encoder: VisionEncoder,
        config: PretrainingConfig,
    ):
        self.depth_encoder = depth_encoder
        self.rgb_encoder = rgb_encoder
        self.config = config

    def pretrain_depth_completion(
        self,
        dataset: RGBDDataset,
        mask_ratio: float = 0.5,
        num_epochs: int = 100,
    ):
        """
        Train depth encoder with depth completion task.

        Mask random regions of depth and predict them.
        """
        # Decoder for depth completion
        decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        optimizer = torch.optim.AdamW(
            list(self.depth_encoder.parameters()) + list(decoder.parameters()),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                depth = batch["depth"]  # (B, 1, H, W)

                # Create mask
                B, _, H, W = depth.shape
                mask = torch.rand(B, 1, H // 16, W // 16, device=depth.device) > mask_ratio
                mask = F.interpolate(mask.float(), size=(H, W), mode='nearest')

                # Masked depth
                masked_depth = depth * mask

                # Encode and decode
                features = self.depth_encoder(masked_depth)
                reconstructed = decoder(features.view(B, 256, 1, 1).expand(-1, -1, H // 16, W // 16))

                # Loss only on masked regions
                loss = F.mse_loss(reconstructed * (1 - mask), depth * (1 - mask))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Depth Completion Loss = {loss.item():.4f}")

    def pretrain_cross_modal(
        self,
        dataset: RGBDDataset,
        num_epochs: int = 100,
    ):
        """
        Cross-modal contrastive learning between RGB and depth.

        Align representations of corresponding RGB-D pairs.
        """
        # Freeze RGB encoder
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(
            self.depth_encoder.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                rgb = batch["rgb"]      # (B, 3, H, W)
                depth = batch["depth"]  # (B, 1, H, W)

                # Encode both
                with torch.no_grad():
                    rgb_feat = self.rgb_encoder(rgb)
                depth_feat = self.depth_encoder(depth)

                # Normalize
                rgb_feat = F.normalize(rgb_feat, dim=-1)
                depth_feat = F.normalize(depth_feat, dim=-1)

                # Contrastive loss: match RGB-depth pairs
                similarity = torch.mm(depth_feat, rgb_feat.t())
                labels = torch.arange(depth_feat.shape[0], device=depth.device)
                loss = F.cross_entropy(similarity / 0.1, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Cross-Modal Loss = {loss.item():.4f}")
```

---

## Stage 2: Cross-Modal Alignment

### Multi-Modal Contrastive Learning

```python
class MultiModalContrastiveTrainer:
    """
    Align representations across multiple sensor modalities.

    Goals:
    1. Similar scenes should have similar representations across modalities
    2. Enable cross-modal retrieval and transfer
    3. Create unified latent space for fusion
    """

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        config: AlignmentConfig,
    ):
        self.encoders = encoders
        self.config = config

        # Projection heads for each modality
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(encoder.output_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            for name, encoder in encoders.items()
        })

    def train_pairwise_alignment(
        self,
        dataset: MultiSensorDataset,
        modality_pairs: List[Tuple[str, str]],
        num_epochs: int = 100,
    ):
        """
        Train pairwise alignment between modality pairs.

        E.g., RGB-Depth, RGB-LiDAR, Depth-LiDAR
        """
        params = list(self.encoders.parameters()) + list(self.projectors.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in dataset:
                loss = 0.0

                for mod1, mod2 in modality_pairs:
                    # Get data
                    data1 = batch[mod1]
                    data2 = batch[mod2]

                    # Encode
                    feat1 = self.projectors[mod1](self.encoders[mod1](data1))
                    feat2 = self.projectors[mod2](self.encoders[mod2](data2))

                    # Contrastive loss
                    pair_loss = self._contrastive_loss(feat1, feat2)
                    loss += pair_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Alignment Loss = {total_loss/len(dataset):.4f}")

    def train_hub_alignment(
        self,
        dataset: MultiSensorDataset,
        hub_modality: str = "rgb",
        num_epochs: int = 100,
    ):
        """
        Align all modalities to a central hub (e.g., RGB).

        This creates a star-shaped alignment where all modalities
        are aligned to RGB, which is often the richest representation.
        """
        params = list(self.encoders.parameters()) + list(self.projectors.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        other_modalities = [m for m in self.encoders.keys() if m != hub_modality]

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in dataset:
                # Encode hub modality
                hub_data = batch[hub_modality]
                hub_feat = self.projectors[hub_modality](self.encoders[hub_modality](hub_data))
                hub_feat = F.normalize(hub_feat, dim=-1)

                loss = 0.0
                for modality in other_modalities:
                    if modality not in batch or batch[modality] is None:
                        continue

                    data = batch[modality]
                    feat = self.projectors[modality](self.encoders[modality](data))
                    feat = F.normalize(feat, dim=-1)

                    # Align to hub
                    pair_loss = self._contrastive_loss(feat, hub_feat)
                    loss += pair_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Hub Alignment Loss = {total_loss/len(dataset):.4f}")

    def _contrastive_loss(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        similarity = torch.mm(feat1, feat2.t()) / temperature
        labels = torch.arange(feat1.shape[0], device=feat1.device)

        loss = (
            F.cross_entropy(similarity, labels) +
            F.cross_entropy(similarity.t(), labels)
        ) / 2

        return loss
```

---

## Stage 3: Fusion Training

### Fusion Modules

```python
class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate all sensor features and process jointly.

    Pros: Simple, allows learning complex interactions
    Cons: Requires all sensors at all times
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, sensor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenate all features
        features = torch.cat(list(sensor_features.values()), dim=-1)
        return self.fusion(features)


class LateFusion(nn.Module):
    """
    Late fusion: Process each modality separately, combine predictions.

    Pros: Robust to missing sensors
    Cons: Limited cross-modal interaction
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
        fusion_method: str = "attention",  # attention, gated, average
    ):
        super().__init__()
        self.fusion_method = fusion_method

        # Per-modality heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
            if dim > 0
        })

        if fusion_method == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True,
            )
            self.query = nn.Parameter(torch.randn(1, 1, output_dim))

        elif fusion_method == "gated":
            num_modalities = len([d for d in modality_dims.values() if d > 0])
            self.gate = nn.Linear(output_dim * num_modalities, num_modalities)

    def forward(self, sensor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = list(sensor_features.values())[0].shape[0]

        # Process each modality
        modality_outputs = []
        for name, features in sensor_features.items():
            if name in self.heads:
                out = self.heads[name](features)
                modality_outputs.append(out)

        modality_outputs = torch.stack(modality_outputs, dim=1)  # (B, M, D)

        if self.fusion_method == "attention":
            query = self.query.expand(B, -1, -1)
            fused, _ = self.attention(query, modality_outputs, modality_outputs)
            return fused.squeeze(1)

        elif self.fusion_method == "gated":
            flat = modality_outputs.reshape(B, -1)
            gates = F.softmax(self.gate(flat), dim=-1)  # (B, M)
            fused = (modality_outputs * gates.unsqueeze(-1)).sum(dim=1)
            return fused

        elif self.fusion_method == "average":
            return modality_outputs.mean(dim=1)


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion: Attention between all modalities.

    Each modality attends to all others for rich interaction.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()

        # Modality-specific projections
        self.input_proj = nn.LazyLinear(feature_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(feature_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, sensor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            sensor_features: Dict of (B, D_i) features per modality

        Returns:
            fused: (B, feature_dim) fused features
        """
        # Project all to same dimension
        projected = {}
        for name, feat in sensor_features.items():
            projected[name] = self.input_proj(feat)

        # Stack into sequence
        # (B, num_modalities, feature_dim)
        x = torch.stack(list(projected.values()), dim=1)

        # Apply cross-attention layers
        for layer in self.layers:
            x = layer(x)

        # Pool across modalities
        fused = x.mean(dim=1)

        return self.output_proj(fused)


class CrossModalAttentionLayer(nn.Module):
    """Single layer of cross-modal attention."""

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention across modalities
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class BEVFusion(nn.Module):
    """
    BEV (Bird's Eye View) fusion for spatial sensor data.

    Projects all sensors to unified BEV space.
    Used primarily for autonomous driving and mobile robots.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        bev_size: Tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,  # meters per pixel
    ):
        super().__init__()
        self.bev_size = bev_size
        self.bev_resolution = bev_resolution

        # BEV encoder for cameras
        self.camera_to_bev = CameraToBEVTransform(
            bev_size=bev_size,
            feature_dim=feature_dim,
        )

        # LiDAR to BEV
        self.lidar_to_bev = LiDARToBEVTransform(
            bev_size=bev_size,
            feature_dim=feature_dim,
        )

        # Radar to BEV
        self.radar_to_bev = RadarToBEVTransform(
            bev_size=bev_size,
            feature_dim=feature_dim,
        )

        # BEV fusion CNN
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, sensor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        bev_features = []

        # Camera to BEV
        if "rgb" in sensor_features:
            camera_bev = self.camera_to_bev(
                sensor_features["rgb"],
                sensor_features.get("camera_intrinsics"),
                sensor_features.get("camera_extrinsics"),
            )
            bev_features.append(camera_bev)

        # LiDAR to BEV
        if "lidar" in sensor_features:
            lidar_bev = self.lidar_to_bev(sensor_features["lidar"])
            bev_features.append(lidar_bev)

        # Radar to BEV
        if "radar" in sensor_features:
            radar_bev = self.radar_to_bev(sensor_features["radar"])
            bev_features.append(radar_bev)

        # Concatenate and fuse
        combined_bev = torch.cat(bev_features, dim=1)
        fused_bev = self.bev_fusion(combined_bev)

        # Global pooling
        features = self.global_pool(fused_bev).flatten(1)

        return features
```

### Fusion Training

```python
class MultiSensorFusionTrainer:
    """
    Train multi-sensor fusion module.
    """

    def __init__(
        self,
        model: MultiSensorVLA,
        config: FusionTrainingConfig,
    ):
        self.model = model
        self.config = config

    def train_with_action_supervision(
        self,
        dataset: MultiSensorRobotDataset,
        num_epochs: int = 100,
    ):
        """
        Train fusion with action prediction supervision.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Forward pass with all available sensors
                output = self.model(
                    rgb_images=batch["rgb"],
                    depth_images=batch.get("depth"),
                    lidar_points=batch.get("lidar"),
                    radar_data=batch.get("radar"),
                    imu_data=batch.get("imu"),
                    tactile_data=batch.get("tactile"),
                    instruction=batch.get("instruction"),
                )

                # Action loss
                action_loss = F.mse_loss(output["action"], batch["action"])

                # Optional: auxiliary losses for sensor predictions
                aux_loss = 0.0
                if self.config.use_auxiliary_losses:
                    # Depth prediction from RGB
                    if "depth" in batch and self.model.config.use_depth:
                        depth_pred = self.model.depth_predictor(
                            output["sensor_features"]["rgb"]
                        )
                        aux_loss += F.mse_loss(depth_pred, batch["depth"])

                total_loss = action_loss + self.config.aux_weight * aux_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Action Loss = {action_loss.item():.4f}")

    def train_with_sensor_dropout(
        self,
        dataset: MultiSensorRobotDataset,
        dropout_prob: float = 0.3,
        num_epochs: int = 100,
    ):
        """
        Train with random sensor dropout for robustness.

        This teaches the model to handle missing sensors gracefully.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Randomly drop sensors
                depth = batch.get("depth")
                lidar = batch.get("lidar")
                radar = batch.get("radar")
                imu = batch.get("imu")
                tactile = batch.get("tactile")

                if depth is not None and torch.rand(1) < dropout_prob:
                    depth = None
                if lidar is not None and torch.rand(1) < dropout_prob:
                    lidar = None
                if radar is not None and torch.rand(1) < dropout_prob:
                    radar = None
                if imu is not None and torch.rand(1) < dropout_prob:
                    imu = None
                if tactile is not None and torch.rand(1) < dropout_prob:
                    tactile = None

                # Forward with potentially missing sensors
                output = self.model(
                    rgb_images=batch["rgb"],
                    depth_images=depth,
                    lidar_points=lidar,
                    radar_data=radar,
                    imu_data=imu,
                    tactile_data=tactile,
                    instruction=batch.get("instruction"),
                )

                loss = F.mse_loss(output["action"], batch["action"])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Loss with dropout = {loss.item():.4f}")
```

---

## Stage 4: Task-Specific Fine-tuning

### RGB-D Manipulation

```python
class RGBDManipulationTrainer:
    """
    Fine-tune for RGB-D manipulation tasks.

    Tasks:
    - Object grasping
    - Pick and place
    - Pushing/sliding
    - Assembly
    """

    def __init__(
        self,
        model: MultiSensorVLA,
        config: ManipulationConfig,
    ):
        self.model = model
        self.config = config

    def train_grasping(
        self,
        dataset: GraspingDataset,
        num_epochs: int = 100,
    ):
        """Train for 6-DoF grasping."""
        # Freeze backbone, train head
        for param in self.model.rgb_encoder.parameters():
            param.requires_grad = False
        for param in self.model.depth_encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                output = self.model(
                    rgb_images=batch["rgb"],
                    depth_images=batch["depth"],
                    instruction=batch["instruction"],
                )

                # Grasp pose loss (position + orientation)
                position_loss = F.mse_loss(
                    output["action"][:, :3],
                    batch["grasp_position"],
                )
                orientation_loss = F.mse_loss(
                    output["action"][:, 3:7],
                    batch["grasp_orientation"],
                )

                # Grasp success prediction (if available)
                if "grasp_success" in output:
                    success_loss = F.binary_cross_entropy_with_logits(
                        output["grasp_success"],
                        batch["grasp_success"].float(),
                    )
                else:
                    success_loss = 0.0

                total_loss = position_loss + orientation_loss + 0.1 * success_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Grasp Loss = {total_loss.item():.4f}")


class MultiSensorDrivingTrainer:
    """
    Fine-tune for autonomous driving with multi-sensor input.
    """

    def __init__(
        self,
        model: MultiSensorVLA,
        config: DrivingConfig,
    ):
        self.model = model
        self.config = config

    def train_trajectory_prediction(
        self,
        dataset: DrivingDataset,
        num_epochs: int = 100,
    ):
        """Train trajectory prediction with all sensors."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                output = self.model(
                    rgb_images=batch["images"],
                    lidar_points=batch["lidar"],
                    radar_data=batch["radar"],
                    imu_data=batch["imu"],
                    instruction=batch.get("navigation_command"),
                )

                # Trajectory loss
                trajectory_loss = F.mse_loss(
                    output["trajectory"],
                    batch["gt_trajectory"],
                )

                # Control loss
                control_loss = F.mse_loss(
                    output["control"],
                    batch["gt_control"],
                )

                total_loss = trajectory_loss + 0.5 * control_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Driving Loss = {total_loss.item():.4f}")
```

---

## Stage 5: Robustness Training

### Sensor Noise and Failure Handling

```python
class RobustnessTrainer:
    """
    Train for robustness to sensor noise and failures.
    """

    def __init__(
        self,
        model: MultiSensorVLA,
        config: RobustnessConfig,
    ):
        self.model = model
        self.config = config

    def train_with_noise_injection(
        self,
        dataset: MultiSensorDataset,
        num_epochs: int = 100,
    ):
        """
        Train with realistic sensor noise.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Add noise to sensors
                rgb = self.add_rgb_noise(batch["rgb"])
                depth = self.add_depth_noise(batch.get("depth"))
                lidar = self.add_lidar_noise(batch.get("lidar"))
                imu = self.add_imu_noise(batch.get("imu"))

                output = self.model(
                    rgb_images=rgb,
                    depth_images=depth,
                    lidar_points=lidar,
                    imu_data=imu,
                )

                loss = F.mse_loss(output["action"], batch["action"])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Noisy Loss = {loss.item():.4f}")

    def add_rgb_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Add realistic RGB noise."""
        # Gaussian noise
        noise_std = torch.rand(1).item() * 0.1
        images = images + torch.randn_like(images) * noise_std

        # Motion blur (random)
        if torch.rand(1) < 0.2:
            kernel_size = 5
            kernel = torch.ones(1, 1, kernel_size, 1, device=images.device) / kernel_size
            images = F.conv2d(
                images,
                kernel.expand(3, -1, -1, -1),
                padding=(kernel_size // 2, 0),
                groups=3,
            )

        # Exposure variation
        if torch.rand(1) < 0.3:
            exposure = torch.rand(1).item() * 0.5 + 0.75
            images = images * exposure

        return torch.clamp(images, 0, 1)

    def add_depth_noise(self, depth: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Add realistic depth sensor noise."""
        if depth is None:
            return None

        # Distance-dependent noise
        noise_std = 0.01 + depth * 0.002  # Noise increases with distance
        depth = depth + torch.randn_like(depth) * noise_std

        # Random dropouts (depth sensors often have holes)
        if torch.rand(1) < 0.3:
            dropout_mask = torch.rand_like(depth) > 0.1
            depth = depth * dropout_mask

        return torch.clamp(depth, 0, self.config.max_depth)

    def add_lidar_noise(self, points: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Add realistic LiDAR noise."""
        if points is None:
            return None

        # Position noise
        position_noise = torch.randn_like(points[:, :, :3]) * 0.02
        points[:, :, :3] = points[:, :, :3] + position_noise

        # Intensity noise
        if points.shape[-1] > 3:
            intensity_noise = torch.randn_like(points[:, :, 3:4]) * 0.1
            points[:, :, 3:4] = torch.clamp(points[:, :, 3:4] + intensity_noise, 0, 1)

        # Random point dropout
        if torch.rand(1) < 0.2:
            dropout_mask = torch.rand(points.shape[0], points.shape[1], device=points.device) > 0.1
            points = points * dropout_mask.unsqueeze(-1)

        return points

    def add_imu_noise(self, imu: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Add realistic IMU noise."""
        if imu is None:
            return None

        # Accelerometer noise
        accel_noise = torch.randn_like(imu[:, :, :3]) * 0.05
        imu[:, :, :3] = imu[:, :, :3] + accel_noise

        # Gyroscope noise
        gyro_noise = torch.randn_like(imu[:, :, 3:6]) * 0.01
        imu[:, :, 3:6] = imu[:, :, 3:6] + gyro_noise

        # Bias drift
        if torch.rand(1) < 0.3:
            bias = torch.randn(1, 1, 6, device=imu.device) * 0.02
            imu = imu + bias

        return imu

    def train_adversarial_robustness(
        self,
        dataset: MultiSensorDataset,
        num_epochs: int = 100,
        epsilon: float = 0.01,
    ):
        """
        Train with adversarial perturbations (FGSM-style).
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Enable gradients for input
                rgb = batch["rgb"].clone().requires_grad_(True)

                # Forward pass
                output = self.model(rgb_images=rgb)
                loss = F.mse_loss(output["action"], batch["action"])

                # Compute gradients w.r.t. input
                loss.backward(retain_graph=True)

                # FGSM perturbation
                perturbation = epsilon * rgb.grad.sign()
                rgb_adv = torch.clamp(rgb + perturbation, 0, 1)

                # Forward with adversarial input
                output_adv = self.model(rgb_images=rgb_adv.detach())
                loss_adv = F.mse_loss(output_adv["action"], batch["action"])

                # Combined loss
                total_loss = (loss + loss_adv) / 2

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: Adversarial Loss = {total_loss.item():.4f}")
```

---

## Advanced Topics

### Uncertainty-Aware Fusion

```python
class UncertaintyAwareFusion(nn.Module):
    """
    Fuse sensors while considering uncertainty.

    Weight sensors based on their reliability.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
    ):
        super().__init__()

        # Feature extractors
        self.feature_extractors = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Uncertainty estimators
        self.uncertainty_estimators = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus(),  # Ensure positive uncertainty
            )
            for name, dim in modality_dims.items()
        })

    def forward(self, sensor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        uncertainties = []

        for name, feat in sensor_features.items():
            # Extract features
            f = self.feature_extractors[name](feat)
            features.append(f)

            # Estimate uncertainty
            u = self.uncertainty_estimators[name](feat)
            uncertainties.append(u)

        features = torch.stack(features, dim=1)      # (B, M, D)
        uncertainties = torch.stack(uncertainties, dim=1)  # (B, M, 1)

        # Weight by inverse uncertainty
        weights = 1.0 / (uncertainties + 1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Weighted average
        fused = (features * weights).sum(dim=1)

        return fused


class SensorCalibrationModule(nn.Module):
    """
    Learn sensor-specific calibration for better fusion.
    """

    def __init__(
        self,
        num_sensors: int,
        feature_dim: int,
    ):
        super().__init__()

        # Per-sensor calibration parameters
        self.scale = nn.Parameter(torch.ones(num_sensors, feature_dim))
        self.bias = nn.Parameter(torch.zeros(num_sensors, feature_dim))

        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(num_sensors))

    def forward(self, sensor_features: List[torch.Tensor]) -> List[torch.Tensor]:
        calibrated = []

        for i, feat in enumerate(sensor_features):
            # Apply calibration
            cal_feat = (feat * self.scale[i] + self.bias[i]) / self.temperature[i]
            calibrated.append(cal_feat)

        return calibrated
```

---

## Deployment

### Multi-Sensor Inference

```python
class MultiSensorInferenceEngine:
    """
    Optimized multi-sensor inference for deployment.
    """

    def __init__(
        self,
        model_path: str,
        config: InferenceConfig,
    ):
        # Load model
        self.model = MultiSensorVLA.from_pretrained(model_path)
        self.model.eval()

        # Optimize for inference
        self._optimize_model()

        # Sensor buffers for temporal smoothing
        self.sensor_buffers = {}

    def _optimize_model(self):
        """Apply inference optimizations."""
        # TorchScript compilation
        self.model = torch.jit.script(self.model)

        # Mixed precision
        self.model = self.model.half()

        # Warmup
        self._warmup()

    def _warmup(self, num_iterations: int = 10):
        """Warmup model for consistent latency."""
        dummy_inputs = self._create_dummy_inputs()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(**dummy_inputs)

    def predict(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        lidar: Optional[np.ndarray] = None,
        imu: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference with available sensors.

        Args:
            rgb: (H, W, 3) RGB image
            depth: (H, W) depth image (optional)
            lidar: (N, 4) point cloud (optional)
            imu: (T, 6) IMU readings (optional)
            instruction: Language instruction (optional)

        Returns:
            Dict with action and other predictions
        """
        # Preprocess inputs
        inputs = self._preprocess(rgb, depth, lidar, imu)

        # Run inference
        with torch.no_grad():
            output = self.model(
                **inputs,
                instruction=instruction,
            )

        # Postprocess
        result = {
            "action": output["action"].cpu().numpy(),
        }

        return result

    def _preprocess(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        lidar: Optional[np.ndarray],
        imu: Optional[np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Preprocess sensor inputs."""
        inputs = {}

        # RGB
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_tensor = F.interpolate(
            rgb_tensor.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
        )
        inputs["rgb_images"] = rgb_tensor.half().cuda()

        # Depth
        if depth is not None:
            depth_tensor = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
            depth_tensor = F.interpolate(depth_tensor, size=(224, 224), mode='bilinear')
            inputs["depth_images"] = depth_tensor.half().cuda()

        # LiDAR
        if lidar is not None:
            lidar_tensor = torch.from_numpy(lidar).float().unsqueeze(0)
            inputs["lidar_points"] = lidar_tensor.half().cuda()

        # IMU
        if imu is not None:
            imu_tensor = torch.from_numpy(imu).float().unsqueeze(0)
            inputs["imu_data"] = imu_tensor.half().cuda()

        return inputs
```

---

## Evaluation and Benchmarks

### Multi-Sensor Evaluation

```python
class MultiSensorEvaluator:
    """
    Evaluate multi-sensor fusion performance.
    """

    def evaluate(
        self,
        model: MultiSensorVLA,
        dataset: MultiSensorDataset,
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation.
        """
        metrics = {}

        # Full sensor performance
        metrics.update(self._evaluate_full_sensors(model, dataset))

        # Per-sensor ablation
        metrics.update(self._evaluate_sensor_ablation(model, dataset))

        # Noise robustness
        metrics.update(self._evaluate_noise_robustness(model, dataset))

        # Fusion quality
        metrics.update(self._evaluate_fusion_quality(model, dataset))

        return metrics

    def _evaluate_full_sensors(
        self,
        model: MultiSensorVLA,
        dataset: MultiSensorDataset,
    ) -> Dict[str, float]:
        """Evaluate with all sensors available."""
        total_error = 0.0
        total_samples = 0

        for batch in dataset:
            output = model(
                rgb_images=batch["rgb"],
                depth_images=batch.get("depth"),
                lidar_points=batch.get("lidar"),
                imu_data=batch.get("imu"),
            )

            error = F.mse_loss(output["action"], batch["action"], reduction='sum')
            total_error += error.item()
            total_samples += batch["action"].numel()

        return {"full_sensors/mse": total_error / total_samples}

    def _evaluate_sensor_ablation(
        self,
        model: MultiSensorVLA,
        dataset: MultiSensorDataset,
    ) -> Dict[str, float]:
        """Evaluate with each sensor removed."""
        sensors = ["depth", "lidar", "imu"]
        metrics = {}

        for removed in sensors:
            total_error = 0.0
            total_samples = 0

            for batch in dataset:
                # Remove one sensor
                inputs = {
                    "rgb_images": batch["rgb"],
                    "depth_images": batch.get("depth") if removed != "depth" else None,
                    "lidar_points": batch.get("lidar") if removed != "lidar" else None,
                    "imu_data": batch.get("imu") if removed != "imu" else None,
                }

                output = model(**inputs)
                error = F.mse_loss(output["action"], batch["action"], reduction='sum')
                total_error += error.item()
                total_samples += batch["action"].numel()

            metrics[f"ablation/without_{removed}_mse"] = total_error / total_samples

        return metrics
```

### Benchmark Results

```
+====================================================================================+
|                     MULTI-SENSOR FUSION BENCHMARK RESULTS                           |
+====================================================================================+
|                                                                                     |
| RGB-D Grasping (GraspNet):                                                          |
| Sensor Config        | Grasp Success | Position Error | Orientation Error          |
| --------------------|---------------|----------------|--------------------------- |
| RGB only             | 72.3%         | 2.1 cm         | 8.5°                       |
| RGB + Depth          | 85.1%         | 1.2 cm         | 5.2°                       |
| RGB + Depth + Tactile| 91.4%         | 0.8 cm         | 3.1°                       |
|                                                                                     |
| Autonomous Driving (nuScenes):                                                      |
| Sensor Config        | mAP           | NDS            | Latency (ms)               |
| --------------------|---------------|----------------|--------------------------- |
| Camera only          | 34.2%         | 41.5%          | 45                         |
| Camera + LiDAR       | 62.8%         | 67.3%          | 78                         |
| Camera + LiDAR + Radar| 65.4%        | 70.1%          | 95                         |
| All sensors          | 67.2%         | 72.4%          | 112                        |
|                                                                                     |
| Noise Robustness (Relative performance drop):                                       |
| Fusion Type          | RGB noise     | Depth dropout  | LiDAR noise                |
| --------------------|---------------|----------------|--------------------------- |
| Early Fusion         | -15%          | -25%           | -12%                       |
| Late Fusion          | -8%           | -10%           | -6%                        |
| Cross-Modal Fusion   | -5%           | -7%            | -4%                        |
|                                                                                     |
+====================================================================================+
```

---

## Summary

This guide covered the complete training pipeline for multi-sensor VLA:

1. **Stage 1**: Individual sensor pretraining (RGB, depth, LiDAR, IMU, tactile)
2. **Stage 2**: Cross-modal alignment (contrastive learning, hub alignment)
3. **Stage 3**: Fusion training (early, late, cross-modal, BEV)
4. **Stage 4**: Task-specific fine-tuning (manipulation, driving)
5. **Stage 5**: Robustness training (noise injection, adversarial)

**Key recommendations:**
- Start with pretrained sensor encoders when possible
- Use cross-modal fusion for rich sensor interaction
- Train with sensor dropout for robustness
- Add realistic noise during training
- Validate performance with sensor ablations
- Optimize per-sensor for deployment latency

---

## Datasets Used for Each Training Step

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 1: RGB Pretraining** | ImageNet | [imagenet.org](https://www.image-net.org/) | Standard vision pretraining |
| **Stage 1: RGB Pretraining** | RoboNet | [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | In-domain visual data from robot cameras |
| **Stage 1: Depth Pretraining** | NYU Depth v2 | [sayakpaul/nyu_depth_v2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2) | Indoor RGB-D with 464 scenes |
| **Stage 1: Depth Pretraining** | ScanNet | [yuchen0187/scannet](https://huggingface.co/datasets/yuchen0187/scannet) | 3D indoor scene reconstruction (776MB) |
| **Stage 1: Depth Pretraining** | Matterport3D | [matterport.com/research](https://matterport.com/partners/research) | Large-scale indoor RGB-D |
| **Stage 1: LiDAR Pretraining** | KITTI LiDAR | [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/) | 22 sequences outdoor driving LiDAR |
| **Stage 1: LiDAR Pretraining** | nuScenes LiDAR | [nuscenes.org](https://www.nuscenes.org/) | 1000 scenes, 34,720 points per sweep |
| **Stage 1: LiDAR Pretraining** | Waymo Open LiDAR | [waymo.com/open](https://waymo.com/open/download) | 1150 scenes, top + side LiDAR |
| **Stage 1: LiDAR Pretraining** | SemanticKITTI | [semantic-kitti.org](http://www.semantic-kitti.org/) | 28 sequences with semantic labels |
| **Stage 1: Radar Pretraining** | nuScenes Radar | [nuscenes.org](https://www.nuscenes.org/) | 5 radar sensors, range-doppler data |
| **Stage 1: Radar Pretraining** | RadarScenes | [radar-scenes.com](https://radar-scenes.com/) | 4 hours radar point clouds, 7500 frames |
| **Stage 1: Radar Pretraining** | RADIal | [github.com/valeoai/RADIal](https://github.com/valeoai/RADIal) | Camera + radar + LiDAR for detection |
| **Stage 1: IMU Pretraining** | EuRoC MAV | [projects.asl.ethz.ch/datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) | IMU + stereo for visual-inertial odometry |
| **Stage 1: IMU Pretraining** | TUM-VI | [vision.in.tum.de/data/datasets](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) | Visual-inertial with ground truth |
| **Stage 1: IMU Pretraining** | Oxford Inertial Odometry | [ori.ox.ac.uk](http://ori.ox.ac.uk/datasets/oxford-inertial-odometry-dataset) | IMU data for odometry estimation |
| **Stage 1: Tactile Pretraining** | YCB-Slide | [ycb-slide.org](https://www.mmintlab.com/ycb-slide/) | Tactile sliding data for manipulation |
| **Stage 1: Tactile Pretraining** | DIGIT Tactile | [github.com/facebookresearch/digit-design](https://github.com/facebookresearch/digit-design) | High-resolution tactile sensor data |
| **Stage 1: Tactile Pretraining** | Touch and Go | [touch-and-go.github.io](https://touch-and-go.github.io/) | Vision + tactile for manipulation |
| **Stage 2: Cross-Modal Alignment** | SUN RGB-D | [rgbd.cs.princeton.edu](https://rgbd.cs.princeton.edu/) | 10K RGB-D images for cross-modal contrastive learning |
| **Stage 3: Fusion Training** | nuScenes | [nuscenes.org](https://www.nuscenes.org/) | 1000 scenes with camera, LiDAR, radar, IMU for multi-sensor fusion |
| **Stage 4a: Online RL (Driving)** | CARLA Simulator | [carla.org](https://carla.org/) | Real-time multi-sensor driving simulation for PPO/SAC |
| **Stage 4a: Online RL (Manipulation)** | Isaac Gym/Isaac Sim | [developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) | Multi-sensor manipulation for online policy learning |
| **Stage 4b: Offline RL (Driving)** | nuScenes | [OpenDriveLab/DriveLM](https://huggingface.co/datasets/OpenDriveLab/DriveLM) | 1000 scenes for CQL/IQL training |
| **Stage 4b: Offline RL (Driving)** | KITTI | [kitti-dataset.com](https://www.cvlibs.net/datasets/kitti/) | Stereo + LiDAR for Decision Transformer |
| **Stage 4b: Offline RL (Driving)** | Waymo Open Dataset | [waymo.com/open](https://waymo.com/open/download) | Large-scale driving logs for offline policy optimization |
| **Stage 4b: Offline RL (Manipulation)** | GraspNet-1Billion | [graspnet.net](https://graspnet.net/) | 88 objects, 190 scenes for offline manipulation RL |
| **Stage 5: Robustness Training** | EuRoC MAV | [projects.asl.ethz.ch/datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) | IMU + visual odometry |
| **Stage 5: Robustness Training** | TUM-VI | [vision.in.tum.de/data/datasets](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) | Visual-inertial dataset |

---

## Related Documents

- [Training Pipeline Overview](training_pipeline_overview.md)
- [Autonomous Vehicle Training](training_autonomous_vehicle.md)
- [Humanoid Robot Training](training_humanoid.md)
- [Temporal and World Model Training](training_temporal_world_model.md)
- [Architecture Guide](architecture.md)
