"""
VLA Model Training Script using OpenVLA with HuggingFace
Supports fine-tuning on custom robotics datasets
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import numpy as np


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="openvla/openvla-7b",
        metadata={"help": "Path to pretrained model or model identifier from HuggingFace"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(default=32, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit precision"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit precision"}
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""
    dataset_name: str = field(
        default="berkeley-autolab/bridge_data_v2",
        metadata={"help": "Name of the dataset on HuggingFace Hub"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use (for debugging)"}
    )
    image_size: int = field(
        default=224,
        metadata={"help": "Input image size"}
    )


# Popular robotics datasets available on HuggingFace
RECOMMENDED_DATASETS = {
    # Open X-Embodiment based datasets
    "bridge": "berkeley-autolab/bridge_data_v2",
    "rt1": "google/rt-1-robot-data",
    "fractal": "google/fractal",

    # LeRobot format datasets
    "pusht": "lerobot/pusht",
    "aloha_sim": "lerobot/aloha_sim_insertion_human",
    "xarm": "lerobot/xarm_push_medium",

    # NVIDIA datasets
    "nvidia_physical_ai": "nvidia/PhysicalAI-Robotics-GR00T",
}


class VLADataCollator:
    """Custom data collator for VLA training."""

    def __init__(self, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        images = []
        texts = []
        actions = []

        for item in batch:
            # Handle different dataset formats
            if "image" in item:
                img = item["image"]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                images.append(img)
            elif "observation" in item:
                img = item["observation"].get("image", item["observation"].get("rgb"))
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                images.append(img)

            # Get instruction/language command
            if "instruction" in item:
                texts.append(item["instruction"])
            elif "language_instruction" in item:
                texts.append(item["language_instruction"])
            elif "task" in item:
                texts.append(item["task"])
            else:
                texts.append("Perform the manipulation task.")

            # Get action labels
            if "action" in item:
                actions.append(torch.tensor(item["action"], dtype=torch.float32))
            elif "actions" in item:
                actions.append(torch.tensor(item["actions"], dtype=torch.float32))

        # Process inputs
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Add action labels if available
        if actions:
            inputs["labels"] = torch.stack(actions)

        return inputs


def load_vla_model(model_args: ModelArguments):
    """Load VLA model with optional quantization and LoRA."""

    # Set up quantization config
    quantization_config = None
    if model_args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA if requested
    if model_args.use_lora:
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def load_processor(model_args: ModelArguments):
    """Load the processor/tokenizer for the VLA model."""
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    return processor


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model and processor
    print(f"Loading model: {model_args.model_name_or_path}")
    model = load_vla_model(model_args)
    processor = load_processor(model_args)

    # Load dataset
    print(f"Loading dataset: {data_args.dataset_name}")
    dataset = load_dataset(
        data_args.dataset_name,
        split=data_args.dataset_split,
        trust_remote_code=True,
    )

    if data_args.max_samples:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)}")

    # Create data collator
    data_collator = VLADataCollator(processor)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
