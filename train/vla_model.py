"""
Vision-Language-Action (VLA) Model
Combines a vision encoder, LLM, and action head for robot manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
    SiglipModel,
    SiglipImageProcessor,
)
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class VLAConfig:
    """Configuration for the VLA model."""
    vision_model_name: str = "google/siglip-base-patch16-224"
    llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    action_dim: int = 7  # (x, y, z, roll, pitch, yaw, gripper)
    action_chunk_size: int = 1  # Number of future actions to predict
    hidden_dim: int = 512
    dropout: float = 0.1
    freeze_vision: bool = False
    freeze_llm: bool = False
    use_action_chunking: bool = False


class VisionProjector(nn.Module):
    """Projects vision features to LLM embedding space."""

    def __init__(self, vision_dim: int, llm_dim: int, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens

        # MLP projector (similar to LLaVA)
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # Learnable query tokens for visual feature compression
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, llm_dim) * 0.02)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, num_patches, vision_dim)
        Returns:
            projected: (batch, num_tokens, llm_dim)
        """
        # Project vision features
        projected = self.projector(vision_features)  # (B, num_patches, llm_dim)

        # Average pool to fixed number of tokens
        B = projected.shape[0]
        query = self.query_tokens.expand(B, -1, -1)

        # Simple attention-based pooling
        attn_weights = torch.bmm(query, projected.transpose(1, 2))  # (B, num_tokens, num_patches)
        attn_weights = F.softmax(attn_weights / (projected.shape[-1] ** 0.5), dim=-1)
        pooled = torch.bmm(attn_weights, projected)  # (B, num_tokens, llm_dim)

        return pooled


class ActionHead(nn.Module):
    """Predicts robot actions from fused embeddings."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        chunk_size: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim * chunk_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - fused features
        Returns:
            actions: (batch, chunk_size, action_dim) or (batch, action_dim)
        """
        out = self.mlp(x)

        if self.chunk_size > 1:
            return out.view(-1, self.chunk_size, self.action_dim)
        return out


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model for Robot Manipulation.

    Architecture:
        1. Vision Encoder (SigLIP) -> Extract image features
        2. Vision Projector -> Map to LLM embedding space
        3. LLM (Qwen2) -> Process fused vision + language
        4. Action Head -> Predict robot actions
    """

    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config

        # Load vision encoder
        print(f"Loading vision encoder: {config.vision_model_name}")
        self.vision_encoder = SiglipModel.from_pretrained(config.vision_model_name)
        self.image_processor = SiglipImageProcessor.from_pretrained(config.vision_model_name)
        vision_dim = self.vision_encoder.config.vision_config.hidden_size

        # Load LLM
        print(f"Loading LLM: {config.llm_model_name}")
        self.llm = AutoModel.from_pretrained(
            config.llm_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dim = self.llm.config.hidden_size

        # Vision projector
        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            num_tokens=64,
        )

        # Action head
        self.action_head = ActionHead(
            input_dim=llm_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            chunk_size=config.action_chunk_size if config.use_action_chunking else 1,
            dropout=config.dropout,
        )

        # Freeze components if specified
        if config.freeze_vision:
            self._freeze_module(self.vision_encoder)
            print("Vision encoder frozen")

        if config.freeze_llm:
            self._freeze_module(self.llm)
            print("LLM frozen")

    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images using the vision encoder.

        Args:
            pixel_values: (batch, channels, height, width)
        Returns:
            vision_features: (batch, num_tokens, llm_dim)
        """
        # Get vision features
        vision_outputs = self.vision_encoder.vision_model(pixel_values)
        vision_features = vision_outputs.last_hidden_state  # (B, num_patches, vision_dim)

        # Project to LLM space
        projected = self.vision_projector(vision_features)

        return projected

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLA model.

        Args:
            pixel_values: (batch, channels, height, width) - preprocessed images
            input_ids: (batch, seq_len) - tokenized instructions
            attention_mask: (batch, seq_len) - attention mask for text
            actions: (batch, action_dim) - ground truth actions for training

        Returns:
            Dictionary containing:
                - predicted_actions: (batch, action_dim)
                - loss: scalar (if actions provided)
        """
        batch_size = pixel_values.shape[0]

        # Encode image
        vision_embeds = self.encode_image(pixel_values)  # (B, num_vision_tokens, llm_dim)
        num_vision_tokens = vision_embeds.shape[1]

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_dim)

        # Concatenate vision and text embeddings: [vision] [text]
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Create combined attention mask
        vision_mask = torch.ones(
            batch_size, num_vision_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Forward through LLM
        llm_outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
        )

        # Get the last hidden state at the final position
        last_hidden = llm_outputs.last_hidden_state[:, -1, :]  # (B, llm_dim)

        # Predict actions
        predicted_actions = self.action_head(last_hidden)

        outputs = {"predicted_actions": predicted_actions}

        # Compute loss if ground truth provided
        if actions is not None:
            loss = F.mse_loss(predicted_actions, actions)
            outputs["loss"] = loss

        return outputs

    def predict_action(
        self,
        image: torch.Tensor,
        instruction: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Predict action for a single image and instruction.

        Args:
            image: PIL Image or (C, H, W) tensor
            instruction: Text instruction string
            device: Device to run inference on

        Returns:
            action: (action_dim,) tensor
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
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

            # Forward pass
            outputs = self.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            return outputs["predicted_actions"].squeeze(0)

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[VLAConfig] = None):
        """Load a pretrained VLA model."""
        if config is None:
            config = VLAConfig()

        model = cls(config)
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str):
        """Save the VLA model."""
        torch.save(self.state_dict(), path)


def create_vla_model(
    vision_model: str = "google/siglip-base-patch16-224",
    llm_model: str = "Qwen/Qwen2-1.5B-Instruct",
    action_dim: int = 7,
    freeze_vision: bool = False,
    freeze_llm: bool = False,
) -> VLAModel:
    """
    Factory function to create a VLA model.

    Args:
        vision_model: HuggingFace vision model name
        llm_model: HuggingFace LLM name
        action_dim: Dimension of action space
        freeze_vision: Whether to freeze vision encoder
        freeze_llm: Whether to freeze LLM

    Returns:
        VLAModel instance
    """
    config = VLAConfig(
        vision_model_name=vision_model,
        llm_model_name=llm_model,
        action_dim=action_dim,
        freeze_vision=freeze_vision,
        freeze_llm=freeze_llm,
    )
    return VLAModel(config)


# Example usage and testing
if __name__ == "__main__":
    from PIL import Image
    import requests

    print("=" * 60)
    print("VLA Model Test")
    print("=" * 60)

    # Create model
    config = VLAConfig(
        vision_model_name="google/siglip-base-patch16-224",
        llm_model_name="Qwen/Qwen2-1.5B-Instruct",
        action_dim=7,
        freeze_vision=True,  # Freeze for faster testing
        freeze_llm=True,
    )

    print("\nCreating VLA model...")
    model = VLAModel(config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with dummy data
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy inputs
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_text = model.tokenizer(
        ["pick up the red block", "move arm to the left"],
        return_tensors="pt",
        padding=True,
    )
    dummy_input_ids = dummy_text.input_ids.to(device)
    dummy_attention_mask = dummy_text.attention_mask.to(device)
    dummy_actions = torch.randn(batch_size, 7).to(device)

    # Forward pass
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = model(
            pixel_values=dummy_image,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            actions=dummy_actions,
        )

    print(f"Predicted actions shape: {outputs['predicted_actions'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    print("\n" + "=" * 60)
    print("VLA Model test passed!")
    print("=" * 60)
