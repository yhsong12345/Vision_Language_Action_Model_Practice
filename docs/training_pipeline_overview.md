# VLA Training Pipeline - Complete Overview

This document provides a comprehensive overview of the entire VLA (Vision-Language-Action) training pipeline, covering all stages from VLM pretraining to deployment.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Stage 1: VLM Pretraining](#stage-1-vlm-pretraining)
3. [Stage 2: Action Head Training](#stage-2-action-head-training)
4. [Stage 3: Policy Improvement](#stage-3-policy-improvement)
5. [Stage 4: Deployment](#stage-4-deployment)
6. [Training Components](#training-components)
7. [Hardware Requirements](#hardware-requirements)
8. [Category-Specific Training](#category-specific-training)

---

## Pipeline Overview

The VLA training follows a **4-stage pipeline** inspired by the LLaVA paradigm:

```
+====================================================================================+
|                           VLA TRAINING PIPELINE                                     |
+====================================================================================+
|                                                                                     |
|  STAGE 1: VLM PRETRAINING (Foundation)                                              |
|  +-------------------------------------------------------------------------+        |
|  |  1a. Vision-Language Alignment                                          |        |
|  |      - Train vision projector to align with LLM embeddings              |        |
|  |      - Dataset: Image-caption pairs (LLaVA-Pretrain 558K)               |        |
|  |      - Trainable: Vision Projector only                                 |        |
|  |      - Frozen: Vision Encoder + LLM                                     |        |
|  +-------------------------------------------------------------------------+        |
|                                    |                                                |
|                                    v                                                |
|  +-------------------------------------------------------------------------+        |
|  |  1b. Visual Instruction Tuning                                          |        |
|  |      - Fine-tune LLM for visual instruction following                   |        |
|  |      - Dataset: Visual QA/Instructions (LLaVA-Instruct 150K)            |        |
|  |      - Trainable: Vision Projector + LLM                                |        |
|  |      - Frozen: Vision Encoder                                           |        |
|  +-------------------------------------------------------------------------+        |
|                                    |                                                |
+=================================== | ===============================================+
                                     v
+====================================================================================+
|  STAGE 2: ACTION HEAD TRAINING (Supervised Fine-tuning)                            |
|  +-------------------------------------------------------------------------+        |
|  |  Add action prediction capability using robot demonstrations             |        |
|  |  - Dataset: Robot demonstrations (LeRobot, Open X-Embodiment)           |        |
|  |  - Methods: Behavioral Cloning (BC), basic imitation                    |        |
|  |  - Action Heads: MLP, Gaussian, Diffusion, Transformer                  |        |
|  |  - Trainable: Action Head (+ LLM with LoRA optional)                    |        |
|  |  - Frozen: Vision Encoder, Vision Projector                             |        |
|  +-------------------------------------------------------------------------+        |
|                                    |                                                |
+=================================== | ===============================================+
                                     v
+====================================================================================+
|  STAGE 3: POLICY IMPROVEMENT                                                        |
|  +-------------------------------------------------------------------------+        |
|  |  3a. IMITATION LEARNING (Expert-guided)                                  |        |
|  |      - BC: Direct state-action mapping                                  |        |
|  |      - DAgger: Interactive learning with expert corrections             |        |
|  |      - GAIL: Adversarial imitation, learns implicit reward              |        |
|  +-------------------------------------------------------------------------+        |
|  |  3b. ONLINE REINFORCEMENT LEARNING (Environment interaction)             |        |
|  |      - PPO: On-policy, stable training                                  |        |
|  |      - SAC: Off-policy, sample efficient                                |        |
|  |      - GRPO: LLM-style policy optimization                              |        |
|  +-------------------------------------------------------------------------+        |
|  |  3c. OFFLINE REINFORCEMENT LEARNING (Static dataset)                     |        |
|  |      - CQL: Conservative Q-learning, penalizes OOD actions              |        |
|  |      - IQL: Implicit Q-learning, stable offline training                |        |
|  |      - TD3+BC: TD3 with behavioral cloning regularization               |        |
|  |      - Decision Transformer: Sequence modeling approach                 |        |
|  +-------------------------------------------------------------------------+        |
|  |  3d. MODEL-BASED RL (World Model)                                        |        |
|  |      - RSSM: Recurrent State-Space Model (Dreamer-style)                |        |
|  |      - Latent dynamics learning                                         |        |
|  |      - Imagination-based planning (MPC)                                 |        |
|  +-------------------------------------------------------------------------+        |
|                                    |                                                |
+=================================== | ===============================================+
                                     v
+====================================================================================+
|  STAGE 4: DEPLOYMENT                                                                |
|  +-------------------------------------------------------------------------+        |
|  |  - Export: ONNX, TorchScript, OpenVINO, Triton                          |        |
|  |  - Optimization: INT8 quantization, mixed precision                     |        |
|  |  - Integration: ROS/ROS2, CARLA, Isaac Sim, MuJoCo                      |        |
|  |  - Safety: Rule checking, constraint handling, safety shields           |        |
|  +-------------------------------------------------------------------------+        |
+====================================================================================+
```

---

## Stage 1: VLM Pretraining

### Stage 1a: Vision-Language Alignment

**Purpose**: Create alignment between vision encoder features and LLM embedding space.

```python
from model.vlm import VLMModel
from train.pretrain import VLMPretrainer, PretrainingDataset
from config import PretrainingConfig

# Create VLM model (no action head)
model = VLMModel(
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",
    projector_type="mlp",  # Options: mlp, attention, perceiver
)

# Configure Stage 1a
config = PretrainingConfig(
    output_dir="./pretrained_vlm",
    dataset_name="liuhaotian/LLaVA-Pretrain",
    stage1_epochs=1,
    stage1_lr=1e-3,
    freeze_vision=True,   # Keep vision encoder frozen
    freeze_llm=True,      # Keep LLM frozen, only train projector
)

# Load alignment dataset
alignment_data = PretrainingDataset(
    "liuhaotian/LLaVA-Pretrain",
    model.image_processor,
    model.tokenizer,
)

# Train vision-language alignment
trainer = VLMPretrainer(model, config)
trainer.train_stage1_alignment(alignment_data)
```

**Key Parameters**:
| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| Learning Rate | 1e-3 | Higher LR for projector-only training |
| Epochs | 1 | Usually 1 epoch is sufficient |
| Batch Size | 256 | Larger batches for stability |
| Dataset | LLaVA-Pretrain | 558K image-caption pairs |

### Stage 1b: Visual Instruction Tuning

**Purpose**: Enable the LLM to follow visual instructions.

```python
# Configure Stage 1b (continuing from Stage 1a)
config = PretrainingConfig(
    output_dir="./pretrained_vlm",
    instruction_dataset="liuhaotian/LLaVA-Instruct-150K",
    stage2_epochs=1,
    stage2_lr=2e-5,
    freeze_vision=True,   # Keep vision frozen
    freeze_llm=False,     # Unfreeze LLM for instruction tuning
)

# Load instruction dataset
instruction_data = PretrainingDataset(
    "liuhaotian/LLaVA-Instruct-150K",
    model.image_processor,
    model.tokenizer,
)

# Train instruction following
trainer.train_stage2_instruction(instruction_data)

# Save pretrained VLM
model.save_pretrained("./pretrained_vlm/vlm_final.pt")
```

**Key Parameters**:
| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| Learning Rate | 2e-5 | Lower LR for LLM fine-tuning |
| Epochs | 1 | 1 epoch on instruction data |
| Batch Size | 128 | Moderate batch size |
| Dataset | LLaVA-Instruct | 150K visual QA pairs |

---

## Stage 2: Action Head Training

**Purpose**: Add action prediction capability to the VLM.

### Supported Action Heads

| Type | Description | Best For |
|------|-------------|----------|
| **MLP** | Simple deterministic prediction | Fast inference, simple tasks |
| **Gaussian** | Stochastic with learned variance | RL training, exploration |
| **Diffusion** | Denoising diffusion for actions | Multi-modal distributions, precision |
| **Transformer** | Autoregressive action generation | Action sequences, long horizon |

### Training Process

```python
from model.vla import VLAModel
from train.il import BehavioralCloning
from train.datasets import create_lerobot_dataloader

# Load pretrained VLM and add action head
vla_model = VLAModel.from_pretrained_vlm(
    vlm_path="./pretrained_vlm/vlm_final.pt",
    action_dim=7,                    # Robot action dimension
    hidden_dim=512,                  # Action head hidden size
    action_chunk_size=1,             # Actions per prediction
    action_head_type="mlp",          # Options: mlp, gaussian, diffusion, transformer

    # Model config (must match pretrained VLM)
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",

    # Fine-tuning strategy
    freeze_vision=True,              # Keep vision frozen
    freeze_llm=True,                 # Keep LLM frozen or use LoRA
    use_lora=True,                   # Enable LoRA for efficient fine-tuning
    lora_r=16,
    lora_alpha=32,
)

# Create dataloader from robot demonstrations
dataloader = create_lerobot_dataloader(
    dataset_name="lerobot/pusht",
    batch_size=32,
)

# Train with Behavioral Cloning
trainer = BehavioralCloning(
    model=vla_model,
    learning_rate=1e-4,
    use_wandb=True,
)
trainer.train(dataloader)

# Save final VLA model
vla_model.save_pretrained("./trained_vla/vla_final.pt")
```

### Action Head Selection Guide

```
+--------------------+---------------+------------------+----------------------+
|    Task Type       |  Action Head  |  Chunk Size      |  Rationale           |
+--------------------+---------------+------------------+----------------------+
| Simple 2D (PushT)  | MLP           | 1                | Fast, deterministic  |
| Pick & Place       | Gaussian      | 1-4              | Uncertainty aware    |
| Bimanual (ALOHA)   | Diffusion     | 16               | Multi-modal, precise |
| Long-horizon       | Transformer   | 32-64            | Temporal modeling    |
| Driving            | Diffusion     | 20 (waypoints)   | Trajectory planning  |
| Humanoid           | Gaussian      | 1-4              | RL compatible        |
+--------------------+---------------+------------------+----------------------+
```

---

## Stage 3: Policy Improvement

### 3a. Imitation Learning

#### Behavioral Cloning (BC)
- **When to use**: High-quality expert demonstrations available
- **Pros**: Simple, fast, no environment needed
- **Cons**: Distribution shift, limited generalization

```python
from train.il import BehavioralCloning

trainer = BehavioralCloning(
    model=vla_model,
    learning_rate=1e-4,
    batch_size=32,
)
trainer.train(dataloader)
```

#### DAgger (Dataset Aggregation)
- **When to use**: Can query expert during training
- **Pros**: Addresses distribution shift
- **Cons**: Requires interactive expert

```python
from train.il import DAgger

trainer = DAgger(
    model=vla_model,
    env=environment,
    expert_policy=expert,
    beta_decay=0.9,  # Expert mixing ratio decay
)
for iteration in range(10):
    trainer.collect_and_train(num_episodes=50)
```

#### GAIL (Generative Adversarial IL)
- **When to use**: No explicit reward function
- **Pros**: Learns implicit reward
- **Cons**: Training can be unstable

```python
from train.il import GAIL

trainer = GAIL(
    model=vla_model,
    env=environment,
    expert_demos=demonstrations,
    discriminator_lr=1e-4,
)
trainer.train()
```

### 3b. Online Reinforcement Learning

#### PPO (Proximal Policy Optimization)
- **When to use**: Stable on-policy training needed
- **Pros**: Stable, well-understood
- **Cons**: Sample inefficient

```python
from train.online_rl import PPOTrainer

trainer = PPOTrainer(
    model=vla_model,
    env=environment,
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    clip_range=0.2,
    entropy_coef=0.01,
)
trainer.train()
```

#### SAC (Soft Actor-Critic)
- **When to use**: Sample efficiency needed
- **Pros**: Off-policy, entropy regularization
- **Cons**: More hyperparameters

```python
from train.online_rl import SACTrainer

trainer = SACTrainer(
    model=vla_model,
    env=environment,
    buffer_size=1_000_000,
    learning_rate=3e-4,
    alpha=0.2,  # Entropy coefficient
)
trainer.train()
```

#### GRPO (Group Relative Policy Optimization)
- **When to use**: LLM-style optimization for VLA
- **Pros**: Compatible with language models
- **Cons**: Newer, less proven

```python
from train.online_rl import GRPOTrainer

trainer = GRPOTrainer(
    model=vla_model,
    env=environment,
    group_size=8,
    kl_coef=0.1,
)
trainer.train()
```

### 3c. Offline Reinforcement Learning

#### Algorithm Selection Guide

| Algorithm | Best For | Key Hyperparameter |
|-----------|----------|-------------------|
| **CQL** | Mixed-quality data | Conservative penalty |
| **IQL** | Suboptimal demonstrations | Expectile (0.7-0.9) |
| **TD3+BC** | Near-expert data | BC coefficient |
| **Decision Transformer** | Long-horizon tasks | Context length |

```python
# IQL Example (Recommended for most cases)
from train.offline_rl import IQLTrainer

trainer = IQLTrainer(
    model=vla_model,
    dataset=offline_dataset,
    expectile=0.7,      # Asymmetric value learning
    temperature=3.0,    # Advantage temperature
    learning_rate=3e-4,
)
trainer.train(num_epochs=1000)
```

### 3d. Model-Based RL (World Model)

**Purpose**: Learn environment dynamics for planning.

```python
from train.world_model import WorldModelTrainer
from model.world_model import RSSM

# Create world model
world_model = RSSM(
    state_dim=256,
    action_dim=7,
    hidden_dim=512,
)

# Train world model
trainer = WorldModelTrainer(
    world_model=world_model,
    dataset=trajectory_dataset,
    learning_rate=1e-4,
)
trainer.train()

# Use for Model Predictive Control (MPC)
def plan_action(world_model, observation, horizon=15):
    state = world_model.encode(observation)
    best_action = None
    best_reward = -float('inf')

    for _ in range(num_samples):
        actions = sample_action_sequence(horizon)
        states = world_model.imagine(state, actions)
        rewards = world_model.predict_rewards(states)

        if rewards.sum() > best_reward:
            best_reward = rewards.sum()
            best_action = actions[0]

    return best_action
```

---

## Stage 4: Deployment

### Export Formats

```python
from model.utils.export import (
    ONNXExporter,
    TorchScriptExporter,
    OpenVINOExporter,
    TritonExporter,
)

# Load trained model
vla_model = VLAModel.from_pretrained("./trained_vla/vla_final.pt")
vla_model.eval()

# Export to ONNX (cross-platform)
onnx_exporter = ONNXExporter(vla_model)
onnx_exporter.export("./deployed/vla_model.onnx")

# Export to TorchScript (PyTorch production)
ts_exporter = TorchScriptExporter(vla_model)
ts_exporter.export("./deployed/vla_model.pt")

# Export to OpenVINO (Intel hardware)
ov_exporter = OpenVINOExporter(vla_model)
ov_exporter.export("./deployed/vla_openvino/")

# Export to Triton (NVIDIA inference server)
triton_exporter = TritonExporter(vla_model)
triton_exporter.export("./deployed/triton_model/")
```

### Quantization

```python
from model.utils.export import QuantizationExporter

# INT8 quantization for edge deployment
quant_exporter = QuantizationExporter(vla_model)
quant_exporter.export(
    "./deployed/vla_quantized.onnx",
    quantization_type="int8",  # ~70% memory reduction, ~30% latency improvement
)
```

### Real-Time Inference

```python
from infer import VLAInferenceEngine, InferenceConfig

config = InferenceConfig(
    model_path="./checkpoints/vla.pt",
    device="cuda",
    precision="fp16",
)

engine = VLAInferenceEngine(config)

# Single inference
result = engine.predict(
    image_path="robot_view.jpg",
    instruction="Pick up the red cube"
)
print(f"Action: {result['action']}")
print(f"Latency: {result['inference_time_ms']:.2f} ms")
```

---

## Training Components

### Temporal Modeling

```python
from model.temporal import HistoryEncoder, TemporalTransformer

# History encoder for context
history_encoder = HistoryEncoder(
    obs_dim=768,
    action_dim=7,
    context_dim=256,
    max_seq_len=16,
)

# Temporal transformer for action sequences
temporal_transformer = TemporalTransformer(
    d_model=512,
    num_heads=8,
    num_layers=4,
    use_causal_mask=True,  # For autoregressive generation
)
```

### Multi-Sensor Fusion

```python
from model.sensor import DepthEncoder, LiDAREncoder, RadarEncoder, IMUEncoder
from model.fusion import CrossModalFusion

# Create sensor encoders
depth_encoder = DepthEncoder(output_dim=256)
lidar_encoder = LiDAREncoder(output_dim=512)
radar_encoder = RadarEncoder(output_dim=256)
imu_encoder = IMUEncoder(output_dim=128)

# Fuse sensor features
fusion = CrossModalFusion(feature_dim=512, num_heads=8)
fused_features = fusion(vision_features, lidar_features, radar_features)
```

### Safety Constraints

```python
from model.safety import SafetyShield, RuleChecker

# Safety shield for action filtering
safety_shield = SafetyShield(
    action_dim=7,
    max_velocity=0.5,  # m/s
    max_acceleration=2.0,  # m/s^2
    workspace_bounds=[[-0.5, 0.5], [-0.5, 0.5], [0.1, 0.8]],
)

# Rule-based safety checking
rule_checker = RuleChecker(
    collision_mesh_path="./meshes/workspace.stl",
    joint_limits=ROBOT_JOINT_LIMITS,
)

# Apply safety in control loop
if not rule_checker.is_safe(action, current_state):
    action = safety_shield.filter(action, current_state)
```

---

## Hardware Requirements

| Training Stage | GPU Memory | Recommended Hardware | Typical Time |
|---------------|------------|---------------------|--------------|
| Stage 1a (Alignment) | 16GB | RTX 4080 | 4-8 hours |
| Stage 1b (Instruction) | 24GB | RTX 4090 | 8-16 hours |
| Stage 2 (Action Head) | 8-16GB | RTX 3080+ | 1-8 hours |
| Stage 3 (Policy) | 16-32GB | A100 | 8-48 hours |
| Distributed Training | 32GB x8 | 8xA100 | 2-4 hours |

### Distributed Training (DeepSpeed ZeRO-3)

```yaml
# config/deepspeed_zero3.yaml
zero_optimization:
  stage: 3
  overlap_comm: true
  reduce_bucket_size: 5e8
  stage3_prefetch_bucket_size: 5e8

mixed_precision:
  bf16: true

gradient_accumulation_steps: 4
train_micro_batch_size_per_gpu: 8  # Effective: 8*4*32=1024

optimizer:
  type: AdamW
  lr: 1e-4
```

```bash
# Launch distributed training
accelerate launch --config_file config/deepspeed_zero3.yaml train/embodiment/train_driving_vla.py
```

---

## Category-Specific Training

For detailed category-specific training guides, see:

- [Autonomous Vehicle Training](training_autonomous_vehicle.md) - Complete guide for autonomous driving VLA
- [Humanoid Robot Training](training_humanoid.md) - Whole-body control and locomotion
- [Temporal and World Model Training](training_temporal_world_model.md) - Temporal modeling and dynamics learning
- [Multi-Sensor Fusion Training](training_multi_sensor.md) - RGB-D, LiDAR, Radar integration
- [Embodiment-Specific Training](training_embodiment.md) - Task-specific configurations

---

## Quick Reference

### Training Command Presets

```bash
# Simple manipulation (PushT)
python run.py train --preset pusht-bc

# Bimanual manipulation (ALOHA)
python run.py train --preset aloha-diffusion

# Autonomous driving (CARLA)
python run.py train --preset carla-driving

# Humanoid locomotion
python run.py train --preset humanoid-ppo

# Offline RL (D4RL)
python run.py train --preset d4rl-offline-rl

# List all presets
python run.py list
```

### Hyperparameter Quick Reference

| Model Size | Learning Rate (BC) | Learning Rate (RL) | Batch Size |
|------------|-------------------|-------------------|------------|
| Small (<500M) | 1e-4 | 3e-4 | 64-128 |
| Medium (1-3B) | 5e-5 | 1e-4 | 32-64 |
| Large (7B+) | 1e-5 | 3e-5 | 8-32 |

---

## Datasets Used for Each Training Step

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 1a: Vision-Language Alignment** | LLaVA-Pretrain | [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | 558K image-caption pairs for training vision projector |
| **Stage 1b: Visual Instruction Tuning** | LLaVA-Instruct-150K | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | 150K visual QA pairs for instruction tuning |
| **Stage 2: Action Head Training** | LeRobot | [lerobot on HuggingFace](https://huggingface.co/lerobot) | 100K+ episodes for robot manipulation tasks |
| **Stage 2: Action Head Training** | Open X-Embodiment | [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | 1M+ episodes, 22+ robot types (1.13TB) |
| **Stage 3a: Online RL** | MuJoCo/Isaac Gym | [mujoco.org](https://mujoco.org/) / [isaac-gym](https://developer.nvidia.com/isaac-gym) | Real-time simulation interaction for PPO/SAC training |
| **Stage 3a: Online RL** | CARLA Simulator | [carla.org](https://carla.org/) | Driving simulation for online policy learning |
| **Stage 3b: Offline RL** | D4RL | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) | 12 standardized offline RL benchmark tasks for CQL/IQL training |
| **Stage 3b: Offline RL** | Visual D4RL (VD4RL) | [conglu/vd4rl](https://huggingface.co/datasets/conglu/vd4rl) | Pixel-based offline RL for Decision Transformer |
| **Stage 3b: Offline RL** | RoboMimic | [amandlek/robomimic](https://huggingface.co/datasets/amandlek/robomimic) | Environment-specific trajectory datasets for world model training |

---

## Next Steps

1. Choose your target application:
   - Robot manipulation: Start with [training_recipes.md](training_recipes.md)
   - Autonomous driving: See [training_autonomous_vehicle.md](training_autonomous_vehicle.md)
   - Humanoid: See [training_humanoid.md](training_humanoid.md)

2. Select appropriate training method:
   - Have expert demonstrations? Start with BC
   - Need to improve? Add DAgger or RL fine-tuning
   - Have static offline data? Use IQL or CQL

3. Deploy your model:
   - Export to appropriate format
   - Apply quantization for edge deployment
   - Integrate with ROS/simulator
