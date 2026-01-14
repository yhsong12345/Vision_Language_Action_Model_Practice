"""
Vision-Language Alignment Trainer

Stage 1 of VLM pretraining:
- Train vision projector to align image features with LLM embedding space
- Uses image-caption pairs
- Freezes both vision encoder and LLM
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from typing import Optional, Dict, Any
import json


class ImageCaptionDataset(Dataset):
    """Dataset for vision-language alignment using image-caption pairs."""

    def __init__(
        self,
        dataset_name: str,
        image_processor,
        tokenizer,
        max_length: int = 256,
        split: str = "train",
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        try:
            from datasets import load_dataset
            self.dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(self.dataset)} samples from {dataset_name}")
        except Exception as e:
            print(f"Could not load dataset: {e}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_data()

    def _create_dummy_data(self, num_samples: int = 1000):
        """Create dummy data for testing."""
        import numpy as np
        from PIL import Image

        self.dataset = []
        captions = [
            "A photo of a robot arm",
            "The robot is picking up an object",
            "A robotic gripper holding a cube",
            "The arm is moving to the left",
            "A manipulation task in progress",
        ]

        for i in range(num_samples):
            # Create random image
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            self.dataset.append({
                "image": img,
                "caption": captions[i % len(captions)],
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item.get("image", item.get("images", None))
        if image is not None:
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        # Process caption
        caption = item.get("caption", item.get("text", "An image"))

        text_inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
        }


class AlignmentTrainer:
    """
    Trainer for Stage 1: Vision-Language Alignment.

    Only trains the vision projector to align visual features
    with the LLM embedding space using a contrastive objective.
    """

    def __init__(
        self,
        model,
        output_dir: str = "./alignment_output",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 1000,
        mixed_precision: str = "bf16",
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps

        os.makedirs(output_dir, exist_ok=True)

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Freeze model for alignment
        self._freeze_model()

    def _freeze_model(self):
        """Freeze everything except vision projector."""
        # Freeze vision encoder
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze LLM
        for param in self.model.llm.parameters():
            param.requires_grad = False

        # Freeze action head
        for param in self.model.action_head.parameters():
            param.requires_grad = False

        # Keep vision projector trainable
        for param in self.model.vision_projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """Run alignment training."""
        print("=" * 60)
        print("Vision-Language Alignment Training")
        print("=" * 60)

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Optimizer
        optimizer = AdamW(
            [p for p in self.model.vision_projector.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.num_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare for distributed training
        self.model, train_loader, optimizer, scheduler = self.accelerator.prepare(
            self.model, train_loader, optimizer, scheduler
        )

        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0

        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    loss = self._compute_alignment_loss(batch)

                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        self.model.vision_projector.parameters(),
                        self.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1

                if global_step % self.logging_steps == 0:
                    avg_loss = total_loss / self.logging_steps
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    total_loss = 0

                if global_step % self.save_steps == 0:
                    self._save_checkpoint(f"step_{global_step}")

        # Save final model
        self._save_checkpoint("final")

    def _compute_alignment_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute alignment loss between vision and text embeddings.

        Uses a next-token prediction objective to align vision with LLM.
        """
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        batch_size = pixel_values.shape[0]

        # Encode image
        vision_embeds = self.model.encode_image(pixel_values)
        num_vision_tokens = vision_embeds.shape[1]

        # Get text embeddings
        text_embeds = self.model.llm.get_input_embeddings()(input_ids)

        # Combine: [vision] [text]
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Create attention mask
        vision_mask = torch.ones(
            batch_size, num_vision_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Create labels for language modeling
        # Vision tokens have -100 (ignored), text tokens are targets
        vision_labels = torch.full(
            (batch_size, num_vision_tokens),
            -100,
            device=input_ids.device,
            dtype=input_ids.dtype
        )
        text_labels = input_ids.clone()
        combined_labels = torch.cat([vision_labels, text_labels], dim=1)

        # Forward through LLM with language modeling objective
        outputs = self.model.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return outputs.loss

    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.output_dir, f"alignment_{name}")
            os.makedirs(save_path, exist_ok=True)

            # Save projector weights only
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(
                unwrapped_model.vision_projector.state_dict(),
                os.path.join(save_path, "projector.pt")
            )

            print(f"Saved checkpoint to {save_path}")
