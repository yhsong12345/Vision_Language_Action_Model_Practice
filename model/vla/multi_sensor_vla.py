"""
Multi-Sensor Vision-Language-Action Model

A VLA model designed for autonomous vehicles and robotics that handles
multiple sensor modalities:
- Camera (RGB images)
- LiDAR (point clouds)
- Radar (range-doppler maps)
- IMU (inertial measurements)

This model is suitable for:
- Autonomous driving
- Mobile robot navigation
- Drone control
- Industrial robots with multiple sensors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from typing import Optional, Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.vlm import VisionEncoder, VisionEncoderConfig, VisionProjector
from model.sensor import PointCloudEncoder, RadarEncoder, IMUEncoder
from model.fusion import SensorFusion
from model.action_head import MLPActionHead
from model.utils import freeze_module, count_parameters, count_trainable_parameters


class MultiSensorVLA(nn.Module):
    """
    Multi-Sensor Vision-Language-Action Model.

    Combines multiple sensor modalities with language instructions
    for robot control in complex environments.

    Sensors supported:
    - Camera (RGB images via SigLIP/CLIP)
    - LiDAR (point clouds via PointNet)
    - Radar (range-doppler maps via CNN)
    - IMU (temporal sequences via Transformer)

    Use cases:
    - Autonomous vehicles
    - Mobile robots
    - Drones
    - Industrial robots
    """

    def __init__(
        self,
        vision_model_name: str = "google/siglip-base-patch16-224",
        llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        action_dim: int = 7,
        hidden_dim: int = 512,
        use_lidar: bool = True,
        use_radar: bool = True,
        use_imu: bool = True,
        freeze_vision: bool = False,
        freeze_llm: bool = False,
        action_chunk_size: int = 1,
        num_vision_tokens: int = 64,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.use_imu = use_imu

        # Load vision encoder using VLM module
        print(f"Loading vision encoder: {vision_model_name}")
        vision_config = VisionEncoderConfig(
            model_name=vision_model_name,
            freeze=freeze_vision,
        )
        self.vision_encoder = VisionEncoder(vision_config)
        self.image_processor = self.vision_encoder.image_processor
        vision_dim = self.vision_encoder.get_output_dim()

        # Load LLM
        print(f"Loading LLM: {llm_model_name}")
        self.llm = AutoModel.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dim = self.llm.config.hidden_size

        # Vision projector using VLM module
        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
        )

        # Learnable query tokens for vision compression
        self.vision_query_tokens = nn.Parameter(
            torch.randn(1, num_vision_tokens, llm_dim) * 0.02
        )

        # Sensor encoders using sensor module
        if use_lidar:
            self.lidar_encoder = PointCloudEncoder(
                input_dim=4,
                output_dim=llm_dim,
                num_tokens=32,
            )
            print("LiDAR encoder initialized")

        if use_radar:
            self.radar_encoder = RadarEncoder(
                input_channels=2,
                output_dim=llm_dim,
                num_tokens=16,
            )
            print("Radar encoder initialized")

        if use_imu:
            self.imu_encoder = IMUEncoder(
                input_dim=6,
                output_dim=llm_dim,
                num_tokens=8,
            )
            print("IMU encoder initialized")

        # Sensor fusion using fusion module
        self.sensor_fusion = SensorFusion(
            hidden_dim=llm_dim,
            num_heads=8,
            num_layers=2,
        )

        # Action head using action_head module
        self.action_head = MLPActionHead(
            input_dim=llm_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            chunk_size=action_chunk_size,
            dropout=0.1,
        )

        # Freeze LLM if specified
        if freeze_llm:
            freeze_module(self.llm, verbose=True)

    def encode_camera(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode camera images."""
        vision_features = self.vision_encoder.encode_image(pixel_values)
        projected = self.vision_projector(vision_features)

        # Attention pooling
        B = projected.shape[0]
        query = self.vision_query_tokens.expand(B, -1, -1)
        attn_weights = torch.bmm(query, projected.transpose(1, 2))
        attn_weights = F.softmax(attn_weights / (projected.shape[-1] ** 0.5), dim=-1)
        pooled = torch.bmm(attn_weights, projected)

        return pooled

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lidar_points: Optional[torch.Tensor] = None,
        radar_data: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-sensor inputs.

        Args:
            pixel_values: (batch, channels, height, width) camera image
            input_ids: (batch, seq_len) tokenized instruction
            attention_mask: (batch, seq_len)
            lidar_points: (batch, num_points, 4) point cloud
            radar_data: (batch, 2, H, W) range-doppler map
            imu_data: (batch, seq_len, 6) IMU readings
            actions: (batch, action_dim) ground truth

        Returns:
            Dict with predicted_actions and optional loss
        """
        batch_size = pixel_values.shape[0]

        # Encode all sensors
        sensor_features = {}

        # Camera
        camera_feat = self.encode_camera(pixel_values)
        sensor_features["camera"] = camera_feat

        # LiDAR
        if self.use_lidar and lidar_points is not None:
            lidar_feat = self.lidar_encoder(lidar_points)
            sensor_features["lidar"] = lidar_feat

        # Radar
        if self.use_radar and radar_data is not None:
            radar_feat = self.radar_encoder(radar_data)
            sensor_features["radar"] = radar_feat

        # IMU
        if self.use_imu and imu_data is not None:
            imu_feat = self.imu_encoder(imu_data)
            sensor_features["imu"] = imu_feat

        # Fuse sensors
        fused_sensors = self.sensor_fusion(sensor_features)
        num_sensor_tokens = fused_sensors.shape[1]

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Combine: [sensors] [text]
        combined_embeds = torch.cat([fused_sensors, text_embeds], dim=1)

        # Create attention mask
        sensor_mask = torch.ones(
            batch_size, num_sensor_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([sensor_mask, attention_mask], dim=1)

        # Forward through LLM
        llm_outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
        )

        # Get action features
        last_hidden = llm_outputs.last_hidden_state[:, -1, :]

        # Predict actions using action head
        action_outputs = self.action_head(last_hidden, actions)

        return action_outputs

    @torch.no_grad()
    def predict_action(
        self,
        image,
        instruction: str,
        lidar_points: Optional[torch.Tensor] = None,
        radar_data: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Predict action for given sensor inputs.

        Args:
            image: PIL Image or tensor
            instruction: Text instruction
            lidar_points: Optional LiDAR point cloud
            radar_data: Optional radar data
            imu_data: Optional IMU data
            device: Device for inference

        Returns:
            action: (action_dim,) tensor
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Process image
        if not isinstance(image, torch.Tensor):
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values
        else:
            pixel_values = image.unsqueeze(0) if image.dim() == 3 else image

        pixel_values = pixel_values.to(device)

        # Tokenize instruction
        text_inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Move sensor data to device if provided
        if lidar_points is not None:
            lidar_points = lidar_points.unsqueeze(0).to(device) if lidar_points.dim() == 2 else lidar_points.to(device)
        if radar_data is not None:
            radar_data = radar_data.unsqueeze(0).to(device) if radar_data.dim() == 3 else radar_data.to(device)
        if imu_data is not None:
            imu_data = imu_data.unsqueeze(0).to(device) if imu_data.dim() == 2 else imu_data.to(device)

        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lidar_points=lidar_points,
            radar_data=radar_data,
            imu_data=imu_data,
        )

        return outputs["predicted_actions"].squeeze(0)

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts."""
        counts = {
            "vision_encoder": count_parameters(self.vision_encoder),
            "llm": count_parameters(self.llm),
            "vision_projector": count_parameters(self.vision_projector),
            "sensor_fusion": count_parameters(self.sensor_fusion),
            "action_head": count_parameters(self.action_head),
        }

        if self.use_lidar:
            counts["lidar_encoder"] = count_parameters(self.lidar_encoder)
        if self.use_radar:
            counts["radar_encoder"] = count_parameters(self.radar_encoder)
        if self.use_imu:
            counts["imu_encoder"] = count_parameters(self.imu_encoder)

        counts["total"] = count_parameters(self)
        counts["trainable"] = count_trainable_parameters(self)

        return counts


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Sensor VLA Model Test")
    print("=" * 60)

    model = MultiSensorVLA(
        vision_model_name="google/siglip-base-patch16-224",
        llm_model_name="Qwen/Qwen2-1.5B-Instruct",
        action_dim=7,
        use_lidar=True,
        use_radar=True,
        use_imu=True,
        freeze_vision=True,
        freeze_llm=True,
    )

    params = model.get_param_count()
    print("\nParameter Counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_lidar = torch.randn(batch_size, 4096, 4).to(device)
    dummy_radar = torch.randn(batch_size, 2, 64, 256).to(device)
    dummy_imu = torch.randn(batch_size, 100, 6).to(device)

    dummy_text = model.tokenizer(
        ["Navigate to the parking spot", "Avoid the pedestrian ahead"],
        return_tensors="pt",
        padding=True,
    )

    with torch.amp.autocast(device_type=str(device), enabled=torch.cuda.is_available()):
        outputs = model(
            pixel_values=dummy_image,
            input_ids=dummy_text.input_ids.to(device),
            attention_mask=dummy_text.attention_mask.to(device),
            lidar_points=dummy_lidar,
            radar_data=dummy_radar,
            imu_data=dummy_imu,
            actions=torch.randn(batch_size, 7).to(device),
        )

    print(f"\nPredicted actions shape: {outputs['predicted_actions'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    print("\n" + "=" * 60)
    print("Multi-Sensor VLA test passed!")
    print("=" * 60)
