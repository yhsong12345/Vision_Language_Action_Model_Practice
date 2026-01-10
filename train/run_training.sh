#!/bin/bash
# VLA Model Training Script
# Usage: bash run_training.sh [openvla|smolvla|custom]

set -e

# Configuration
MODEL_TYPE=${1:-"custom"}
OUTPUT_DIR="./outputs/${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"

echo "================================"
echo "VLA Training Script"
echo "Model: ${MODEL_TYPE}"
echo "Output: ${OUTPUT_DIR}"
echo "================================"

# Install requirements
pip install -r requirements.txt

if [ "$MODEL_TYPE" == "custom" ]; then
    echo "Training Custom VLA (SigLIP + Qwen2)..."
    echo "Vision: google/siglip-base-patch16-224"
    echo "LLM: Qwen/Qwen2-1.5B-Instruct"

    python train_vla.py \
        --vision_model google/siglip-base-patch16-224 \
        --llm_model Qwen/Qwen2-1.5B-Instruct \
        --dataset_name lerobot/pusht \
        --action_dim 7 \
        --freeze_vision \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_epochs 10 \
        --gradient_accumulation_steps 4 \
        --logging_steps 10 \
        --save_steps 500

elif [ "$MODEL_TYPE" == "openvla" ]; then
    echo "Training OpenVLA (7B parameters)..."
    echo "Requires ~40GB+ VRAM for full fine-tuning"
    echo "Using LoRA for memory efficiency..."

    python train.py \
        --model_name_or_path openvla/openvla-7b \
        --dataset_name berkeley-autolab/bridge_data_v2 \
        --use_lora True \
        --lora_r 32 \
        --lora_alpha 32 \
        --load_in_4bit True \
        --output_dir ${OUTPUT_DIR} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --warmup_ratio 0.03 \
        --logging_steps 10 \
        --save_steps 500 \
        --bf16 True \
        --tf32 True \
        --dataloader_num_workers 4 \
        --report_to wandb

elif [ "$MODEL_TYPE" == "smolvla" ]; then
    echo "Training SmolVLA (450M parameters)..."
    echo "Memory efficient - works on consumer GPUs"

    python train_smolvla.py \
        --model_name HuggingFaceTB/SmolVLA-450M \
        --dataset_name lerobot/pusht \
        --output_dir ${OUTPUT_DIR} \
        --learning_rate 1e-4 \
        --batch_size 8 \
        --num_epochs 10 \
        --gradient_accumulation_steps 4

else
    echo "Unknown model type: ${MODEL_TYPE}"
    echo "Usage: bash run_training.sh [openvla|smolvla]"
    exit 1
fi

echo "================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "================================"
