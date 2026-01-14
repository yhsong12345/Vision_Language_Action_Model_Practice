"""
Visual Instruction Tuning Trainer

Stage 2 of VLM pretraining:
- Fine-tune LLM on visual instruction data
- Keep vision encoder frozen
- Train both projector and LLM
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from typing import Optional, Dict, Any, List
import json


class InstructionDataset(Dataset):
    """Dataset for visual instruction tuning."""

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

        # System prompt template
        self.system_prompt = (
            "You are a helpful vision-language assistant. "
            "Analyze the image and respond to the user's question accurately."
        )

        try:
            from datasets import load_dataset
            self.dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(self.dataset)} samples from {dataset_name}")
        except Exception as e:
            print(f"Could not load dataset: {e}")
            self._create_dummy_data()

    def _create_dummy_data(self, num_samples: int = 1000):
        """Create dummy instruction data."""
        import numpy as np
        from PIL import Image

        conversations_templates = [
            [
                {"from": "human", "value": "What is the robot doing in this image?"},
                {"from": "assistant", "value": "The robot is picking up an object from the table."},
            ],
            [
                {"from": "human", "value": "Describe the manipulation task shown."},
                {"from": "assistant", "value": "The image shows a robotic arm performing a grasping task."},
            ],
            [
                {"from": "human", "value": "What should the robot do next?"},
                {"from": "assistant", "value": "The robot should move the object to the target location."},
            ],
        ]

        self.dataset = []
        for i in range(num_samples):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            self.dataset.append({
                "image": img,
                "conversations": conversations_templates[i % len(conversations_templates)],
            })

    def __len__(self):
        return len(self.dataset)

    def _format_conversation(self, conversations: List[Dict]) -> str:
        """Format conversation for the model."""
        formatted = f"<|system|>\n{self.system_prompt}\n"

        for conv in conversations:
            role = conv.get("from", conv.get("role", "human"))
            content = conv.get("value", conv.get("content", ""))

            if role in ["human", "user"]:
                formatted += f"<|user|>\n{content}\n"
            else:
                formatted += f"<|assistant|>\n{content}\n"

        return formatted

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

        # Format and tokenize conversation
        conversations = item.get("conversations", [])
        text = self._format_conversation(conversations)

        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Create labels (same as input_ids for causal LM)
        labels = text_inputs.input_ids.clone().squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "labels": labels,
        }


class InstructionTrainer:
    """
    Trainer for Stage 2: Visual Instruction Tuning.

    Fine-tunes the LLM on visual instruction data to enable
    following complex instructions about images.
    """

    def __init__(
        self,
        model,
        output_dir: str = "./instruction_output",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        mixed_precision: str = "bf16",
        gradient_accumulation_steps: int = 4,
        use_lora: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 32,
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
        self.eval_steps = eval_steps
        self.use_lora = use_lora

        os.makedirs(output_dir, exist_ok=True)

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Setup model for instruction tuning
        self._setup_model(lora_r, lora_alpha)

    def _setup_model(self, lora_r: int, lora_alpha: int):
        """Setup model for instruction tuning."""
        # Freeze vision encoder
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        if self.use_lora:
            # Apply LoRA to LLM
            try:
                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                )
                self.model.llm = get_peft_model(self.model.llm, lora_config)
                self.model.llm.print_trainable_parameters()
            except ImportError:
                print("PEFT not installed, training full model")
                self.use_lora = False
        else:
            # Full fine-tuning
            for param in self.model.llm.parameters():
                param.requires_grad = True

        # Keep projector trainable
        for param in self.model.vision_projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """Run instruction tuning."""
        print("=" * 60)
        print("Visual Instruction Tuning")
        print("=" * 60)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
            )

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
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
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)

        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float("inf")
        total_loss = 0

        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    loss = self._compute_instruction_loss(batch)

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            trainable_params,
                            self.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1

                if global_step % self.logging_steps == 0:
                    avg_loss = total_loss / self.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}"
                    })
                    total_loss = 0

                # Evaluation
                if val_loader is not None and global_step % self.eval_steps == 0:
                    val_loss = self._evaluate(val_loader)
                    if self.accelerator.is_main_process:
                        print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint("best")

                    self.model.train()

                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self._save_checkpoint(f"step_{global_step}")

        # Save final model
        self._save_checkpoint("final")

    def _compute_instruction_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute language modeling loss for instruction tuning."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

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

        # Create labels
        vision_labels = torch.full(
            (batch_size, num_vision_tokens),
            -100,
            device=labels.device,
            dtype=labels.dtype
        )
        combined_labels = torch.cat([vision_labels, labels], dim=1)

        # Forward through LLM
        outputs = self.model.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return outputs.loss

    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_instruction_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.output_dir, f"instruction_{name}")
            os.makedirs(save_path, exist_ok=True)

            unwrapped_model = self.accelerator.unwrap_model(self.model)

            if self.use_lora:
                # Save LoRA weights
                unwrapped_model.llm.save_pretrained(os.path.join(save_path, "lora"))
            else:
                # Save full model
                torch.save(
                    unwrapped_model.state_dict(),
                    os.path.join(save_path, "model.pt")
                )

            # Save projector
            torch.save(
                unwrapped_model.vision_projector.state_dict(),
                os.path.join(save_path, "projector.pt")
            )

            print(f"Saved checkpoint to {save_path}")
