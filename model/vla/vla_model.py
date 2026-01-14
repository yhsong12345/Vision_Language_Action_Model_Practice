"""
Vision-Language-Action (VLA) Base Model

A modular VLA architecture combining:
- Vision Encoder: Extracts visual features from images
- Language Model: Processes instructions and fuses with vision
- Action Head: Predicts robot actions

Supports various vision encoders and LLMs from HuggingFace.
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

from model.vlm import VisionEncoder, VisionEncoderConfig, AttentionPoolingProjector
from model.action_head import MLPActionHead
from model.utils import freeze_module, count_parameters, count_trainable_parameters


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model for Robot Manipulation.

    Architecture:
        1. Vision Encoder (SigLIP/CLIP/DINOv2) -> Extract image features
        2. Vision Projector -> Map to LLM embedding space
        3. LLM (Qwen2/LLaMA/Phi) -> Process fused vision + language
        4. Action Head -> Predict robot actions

    Supports:
        - Multiple vision encoders
        - Multiple LLM backends
        - Action chunking for temporal consistency
        - Mixed precision training
    """

    # Supported vision encoders
    VISION_ENCODERS = {
        "siglip-base": "google/siglip-base-patch16-224",
        "siglip-large": "google/siglip-large-patch16-384",
        "clip-base": "openai/clip-vit-base-patch32",
        "clip-large": "openai/clip-vit-large-patch14",
        "dinov2-base": "facebook/dinov2-base",
        "dinov2-large": "facebook/dinov2-large",
    }

    # Supported LLMs
    LLMS = {
        "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "phi3-small": "microsoft/Phi-3-small-8k-instruct",
    }

    def __init__(
        self,
        vision_model_name: str = "google/siglip-base-patch16-224",
        llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_vision_tokens: int = 64,
        action_chunk_size: int = 1,
        dropout: float = 0.1,
        freeze_vision: bool = False,
        freeze_llm: bool = False,
        use_flash_attention: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

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
        attn_implementation = "flash_attention_2" if use_flash_attention else "sdpa"
        self.llm = AutoModel.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for LLM")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dim = self.llm.config.hidden_size

        # Vision projector using VLM module
        self.vision_projector = AttentionPoolingProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            num_tokens=num_vision_tokens,
        )

        # Action head using action_head module
        self.action_head = MLPActionHead(
            input_dim=llm_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            chunk_size=action_chunk_size,
            dropout=dropout,
        )

        # Freeze LLM if specified
        if freeze_llm:
            freeze_module(self.llm, verbose=True)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images using the vision encoder.

        Args:
            pixel_values: (batch, channels, height, width)
        Returns:
            vision_features: (batch, num_tokens, llm_dim)
        """
        vision_features = self.vision_encoder.encode_image(pixel_values)
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
            pixel_values: (batch, channels, height, width)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            actions: (batch, action_dim) - ground truth for training

        Returns:
            Dict with predicted_actions and optional loss
        """
        batch_size = pixel_values.shape[0]

        # Encode image
        vision_embeds = self.encode_image(pixel_values)
        num_vision_tokens = vision_embeds.shape[1]

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate: [vision] [text]
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

        # Get last hidden state at final position
        last_hidden = llm_outputs.last_hidden_state[:, -1, :]

        # Predict actions
        action_outputs = self.action_head(last_hidden, actions)

        return action_outputs

    @torch.no_grad()
    def predict_action(
        self,
        image,
        instruction: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Predict action for a single image and instruction.

        Args:
            image: PIL Image or tensor
            instruction: Text instruction
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

        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs["predicted_actions"].squeeze(0)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load a pretrained VLA model."""
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str):
        """Save the VLA model."""
        torch.save(self.state_dict(), path)

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            "vision_encoder": count_parameters(self.vision_encoder),
            "vision_encoder_trainable": count_trainable_parameters(self.vision_encoder),
            "llm": count_parameters(self.llm),
            "llm_trainable": count_trainable_parameters(self.llm),
            "vision_projector": count_parameters(self.vision_projector),
            "action_head": count_parameters(self.action_head),
            "total": count_parameters(self),
            "total_trainable": count_trainable_parameters(self),
        }


def create_vla_model(
    vision_encoder: str = "siglip-base",
    llm: str = "qwen2-1.5b",
    action_dim: int = 7,
    **kwargs,
) -> VLAModel:
    """
    Factory function to create a VLA model.

    Args:
        vision_encoder: Key from VLAModel.VISION_ENCODERS or full HF name
        llm: Key from VLAModel.LLMS or full HF name
        action_dim: Dimension of action space
        **kwargs: Additional arguments for VLAModel

    Returns:
        VLAModel instance
    """
    vision_model = VLAModel.VISION_ENCODERS.get(vision_encoder, vision_encoder)
    llm_model = VLAModel.LLMS.get(llm, llm)

    return VLAModel(
        vision_model_name=vision_model,
        llm_model_name=llm_model,
        action_dim=action_dim,
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("VLA Model Test")
    print("=" * 60)

    # Create model
    model = create_vla_model(
        vision_encoder="siglip-base",
        llm="qwen2-1.5b",
        action_dim=7,
        freeze_vision=True,
        freeze_llm=True,
    )

    # Print parameter counts
    params = model.get_param_count()
    print("\nParameter Counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

    with torch.amp.autocast(device_type=str(device), enabled=torch.cuda.is_available()):
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
