"""
VLM Pretrainer

Main class for Vision-Language Model pretraining following LLaVA paradigm:
1. Stage 1 (Alignment): Train projector to align vision features with LLM
2. Stage 2 (Instruction): Fine-tune LLM on visual instruction data
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.training_config import PretrainingConfig
from model.vlm import VLMModel


class PretrainingDataset(Dataset):
    """Dataset for VLM pretraining."""

    def __init__(
        self,
        dataset_name: str,
        image_processor,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        try:
            from datasets import load_dataset

            self.dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"Could not load dataset: {e}")
            self.dataset = None

    def __len__(self):
        return len(self.dataset) if self.dataset else 0

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

        # Process text
        conversations = item.get("conversations", [])
        if conversations:
            # Format conversation
            text = ""
            for conv in conversations:
                role = conv.get("from", conv.get("role", "user"))
                content = conv.get("value", conv.get("content", ""))
                text += f"{role}: {content}\n"
        else:
            text = item.get("text", item.get("caption", ""))

        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "labels": text_inputs.input_ids.squeeze(0).clone(),
        }


class VLMPretrainer:
    """
    Vision-Language Model Pretrainer.

    Implements the two-stage pretraining paradigm:
    - Stage 1: Vision-Language Alignment (train projector only)
    - Stage 2: Visual Instruction Tuning (fine-tune LLM)
    """

    def __init__(
        self,
        model: VLMModel,
        config: PretrainingConfig,
    ):
        self.model = model
        self.config = config

        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Move model to device
        self.model = self.accelerator.prepare(self.model)

        # Initialize logging
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=vars(config),
            )

    def train_stage1_alignment(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Stage 1: Vision-Language Alignment

        Only trains the vision projector while keeping vision encoder
        and LLM frozen. Uses image-caption pairs to align visual
        features with language embeddings.
        """
        print("=" * 60)
        print("Stage 1: Vision-Language Alignment")
        print("=" * 60)

        # Freeze everything except projector
        self._freeze_for_alignment()

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=(
                self.config.num_workers if hasattr(self.config, "num_workers") else 4
            ),
            pin_memory=True,
        )

        # Optimizer (only projector parameters)
        projector_params = [
            p for p in self.model.vision_projector.parameters() if p.requires_grad
        ]
        optimizer = AdamW(
            projector_params,
            lr=self.config.alignment_lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.config.alignment_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare for distributed training
        train_loader, optimizer, scheduler = self.accelerator.prepare(
            train_loader, optimizer, scheduler
        )

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(self.config.alignment_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.alignment_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self._alignment_forward(batch)
                    loss = outputs["loss"]

                    # Backward pass
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            projector_params,
                            self.config.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log(
                            {
                                "stage1/loss": loss.item(),
                                "stage1/lr": scheduler.get_last_lr()[0],
                                "stage1/step": global_step,
                            }
                        )

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save stage 1 checkpoint
        self._save_checkpoint("stage1_alignment")

    def train_stage2_instruction(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Stage 2: Visual Instruction Tuning

        Fine-tunes the LLM on visual instruction data while keeping
        the vision encoder frozen. Trains both projector and LLM.
        """
        print("=" * 60)
        print("Stage 2: Visual Instruction Tuning")
        print("=" * 60)

        # Unfreeze LLM for instruction tuning
        self._unfreeze_for_instruction()

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=(
                self.config.num_workers if hasattr(self.config, "num_workers") else 4
            ),
            pin_memory=True,
        )

        # Optimizer (projector + LLM parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=self.config.instruction_lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.config.instruction_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare for distributed training
        train_loader, optimizer, scheduler = self.accelerator.prepare(
            train_loader, optimizer, scheduler
        )

        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float("inf")

        for epoch in range(self.config.instruction_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.instruction_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Forward pass (language modeling objective)
                    outputs = self._instruction_forward(batch)
                    loss = outputs["loss"]

                    # Backward pass
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            trainable_params,
                            self.config.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log(
                            {
                                "stage2/loss": loss.item(),
                                "stage2/lr": scheduler.get_last_lr()[0],
                                "stage2/step": global_step,
                            }
                        )

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"stage2_step_{global_step}")

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_checkpoint("stage2_best")

        # Save final checkpoint
        self._save_checkpoint("stage2_final")

    def _alignment_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for alignment training (projector only)."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Encode image
        vision_embeds = self.model.encode_image(pixel_values)
        num_vision_tokens = vision_embeds.shape[1]
        batch_size = pixel_values.shape[0]

        # Get text embeddings
        text_embeds = self.model.llm.get_input_embeddings()(input_ids)

        # Combine vision and text
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Create attention mask
        vision_mask = torch.ones(
            batch_size,
            num_vision_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Create labels (shift for language modeling)
        # Vision tokens should have -100 labels (ignored)
        vision_labels = torch.full(
            (batch_size, num_vision_tokens),
            -100,
            device=labels.device,
            dtype=labels.dtype,
        )
        combined_labels = torch.cat([vision_labels, labels], dim=1)

        # Forward through LLM
        outputs = self.model.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return {"loss": outputs.loss}

    def _instruction_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for instruction tuning."""
        # Same as alignment forward but with full model training
        return self._alignment_forward(batch)

    def _freeze_for_alignment(self):
        """Freeze model for Stage 1 (alignment)."""
        # Freeze vision encoder
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze LLM
        for param in self.model.llm.parameters():
            param.requires_grad = False

        # Keep projector trainable
        for param in self.model.vision_projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    def _unfreeze_for_instruction(self):
        """Unfreeze model for Stage 2 (instruction tuning)."""
        # Keep vision encoder frozen
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        # Unfreeze LLM
        for param in self.model.llm.parameters():
            param.requires_grad = True

        # Keep projector trainable
        for param in self.model.vision_projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.config.output_dir, name)
            os.makedirs(save_path, exist_ok=True)

            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(os.path.join(save_path, "model.pt"))

            # Save config
            with open(os.path.join(save_path, "config.json"), "w") as f:
                json.dump(vars(self.config), f, indent=2, default=str)

            print(f"Saved checkpoint to {save_path}")


def pretrain_vlm(
    vision_model: str = "google/siglip-base-patch16-224",
    llm_model: str = "Qwen/Qwen2-1.5B-Instruct",
    alignment_dataset: str = "liuhaotian/LLaVA-Pretrain",
    instruction_dataset: str = "liuhaotian/LLaVA-Instruct-150K",
    output_dir: str = "./pretrained_vlm",
    **kwargs,
):
    """
    Convenience function to run full VLM pretraining.

    Args:
        vision_model: Vision encoder name
        llm_model: LLM name
        alignment_dataset: Dataset for Stage 1
        instruction_dataset: Dataset for Stage 2
        output_dir: Output directory
        **kwargs: Additional config arguments
    """
    # Create model
    model = VLMModel(
        vision_model_name=vision_model,
        llm_model_name=llm_model,
    )

    # Create config
    config = PretrainingConfig(
        output_dir=output_dir,
        dataset_name=alignment_dataset,
        instruction_dataset=instruction_dataset,
        **kwargs,
    )

    # Create trainer
    trainer = VLMPretrainer(model, config)

    # Create datasets
    alignment_data = PretrainingDataset(
        alignment_dataset,
        model.image_processor,
        model.tokenizer,
    )
    instruction_data = PretrainingDataset(
        instruction_dataset,
        model.image_processor,
        model.tokenizer,
    )

    # Run training
    trainer.train_stage1_alignment(alignment_data)
    trainer.train_stage2_instruction(instruction_data)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VLM Pretrainer - LLaVA-style Vision-Language Model pretraining"
    )

    # Model arguments
    parser.add_argument(
        "--vision-model",
        type=str,
        default="google/siglip-base-patch16-224",
        help="Vision encoder model name (default: google/siglip-base-patch16-224)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="LLM model name (default: Qwen/Qwen2-1.5B-Instruct)",
    )

    # Dataset arguments
    parser.add_argument(
        "--alignment-dataset",
        type=str,
        default="liuhaotian/LLaVA-Pretrain",
        help="Dataset for Stage 1 alignment (default: liuhaotian/LLaVA-Pretrain)",
    )
    parser.add_argument(
        "--instruction-dataset",
        type=str,
        default="liuhaotian/LLaVA-Instruct-150K",
        help="Dataset for Stage 2 instruction tuning (default: liuhaotian/LLaVA-Instruct-150K)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./pretrained_vlm",
        help="Output directory for checkpoints (default: ./pretrained_vlm)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--alignment-epochs",
        type=int,
        default=1,
        help="Number of epochs for Stage 1 alignment (default: 1)",
    )
    parser.add_argument(
        "--instruction-epochs",
        type=int,
        default=3,
        help="Number of epochs for Stage 2 instruction tuning (default: 3)",
    )
    parser.add_argument(
        "--alignment-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Stage 1 alignment (default: 1e-3)",
    )
    parser.add_argument(
        "--instruction-lr",
        type=float,
        default=2e-5,
        help="Learning rate for Stage 2 instruction tuning (default: 2e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )

    # Hardware arguments
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision mode (default: bf16)",
    )

    # Logging arguments
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency (default: 10)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Checkpoint save frequency (default: 1000)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vlm-pretraining",
        help="W&B project name (default: vlm-pretraining)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="vlm_pretrain",
        help="Experiment name (default: vlm_pretrain)",
    )

    args = parser.parse_args()

    # Convert args to kwargs for pretrain_vlm
    kwargs = {
        "batch_size": args.batch_size,
        "alignment_epochs": args.alignment_epochs,
        "instruction_epochs": args.instruction_epochs,
        "alignment_lr": args.alignment_lr,
        "instruction_lr": args.instruction_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "experiment_name": args.experiment_name,
    }

    print("=" * 60)
    print("VLM Pretrainer")
    print("=" * 60)
    print(f"Vision Model: {args.vision_model}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Alignment Dataset: {args.alignment_dataset}")
    print(f"Instruction Dataset: {args.instruction_dataset}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    pretrain_vlm(
        vision_model=args.vision_model,
        llm_model=args.llm_model,
        alignment_dataset=args.alignment_dataset,
        instruction_dataset=args.instruction_dataset,
        output_dir=args.output_dir,
        **kwargs,
    )
