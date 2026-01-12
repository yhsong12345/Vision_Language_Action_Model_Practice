# Vision-Language-Action (VLA) Model Training

A comprehensive framework for training Vision-Language-Action models for robotics and autonomous vehicle applications.

## Project Structure

```
Vision_Language_Model_Practice/
├── config/                   # Configuration classes
│   ├── model_config.py       # Model architecture configs (VLA, MultiSensor, OpenVLA, SmolVLA)
│   ├── dataset_config.py     # Dataset configs (LeRobot, OpenX, Driving datasets)
│   └── training_config.py    # Training configs (Pretrain, Finetune, RL, IL)
├── model/                    # Model components
│   ├── vlm/                  # Vision-Language Model
│   │   ├── vision_encoder.py # Vision encoders (SigLIP, CLIP, DINOv2)
│   │   └── vision_projector.py # Projectors (MLP, Attention, Perceiver)
│   ├── action_head/          # Action prediction heads
│   │   ├── mlp_action_head.py # MLP and Gaussian MLP heads
│   │   ├── diffusion_action_head.py # DDPM/DDIM diffusion head
│   │   └── transformer_action_head.py # Autoregressive transformer head
│   ├── sensor/               # Sensor encoders
│   │   ├── lidar_encoder.py  # PointNet/PointTransformer for LiDAR
│   │   ├── radar_encoder.py  # CNN encoder for radar
│   │   └── imu_encoder.py    # Transformer encoder for IMU
│   ├── vla/                  # Complete VLA implementations
│   │   ├── vla_model.py      # Core VLA model
│   │   ├── multi_sensor_vla.py # Multi-sensor VLA for autonomous driving
│   │   ├── openvla_wrapper.py # OpenVLA-7B wrapper
│   │   └── smolvla_wrapper.py # SmolVLA lightweight wrapper
│   └── fusion/               # Multi-modal sensor fusion
│       └── sensor_fusion.py  # Self-attention, Cross-modal, Hierarchical, Gated fusion
├── train/                    # Training modules
│   ├── pretrain/             # VLM pretraining
│   │   ├── vlm_pretrainer.py # Main pretraining orchestrator
│   │   ├── alignment_trainer.py # Stage 1: Vision-language alignment
│   │   └── instruction_trainer.py # Stage 2: Visual instruction tuning
│   ├── finetune/             # Supervised fine-tuning
│   │   ├── vla_finetuner.py  # VLA fine-tuning with LoRA support
│   │   └── dataset.py        # Robot dataset loader
│   ├── rl/                   # Reinforcement learning
│   │   ├── base_trainer.py   # Base RL trainer with rollout buffer
│   │   ├── ppo_trainer.py    # Proximal Policy Optimization
│   │   ├── sac_trainer.py    # Soft Actor-Critic
│   │   └── grpo_trainer.py   # Group Relative Policy Optimization
│   ├── il/                   # Imitation learning
│   │   ├── base_trainer.py   # Base IL trainer
│   │   ├── behavioral_cloning.py # Behavioral Cloning (BC)
│   │   ├── dagger.py         # Dataset Aggregation (DAgger)
│   │   └── gail.py           # Generative Adversarial Imitation Learning
│   └── datasets/             # Dataset loaders
│       ├── lerobot_dataset.py # LeRobot datasets (PushT, ALOHA, xArm)
│       ├── openx_dataset.py  # Open X-Embodiment datasets
│       ├── driving_dataset.py # Driving datasets (nuScenes, Waymo, KITTI)
│       ├── rl_dataset.py     # RL trajectory datasets
│       └── bc_dataset.py     # D4RL BC datasets with filtering
├── eval/                     # Evaluation scripts
│   ├── metrics.py            # Success rate, trajectory, action metrics
│   ├── evaluator.py          # Main evaluation orchestrator
│   └── benchmark.py          # Benchmark suites
├── scripts/                  # SLURM job submission scripts
│   ├── run_pretrain.sh       # VLM pretraining (multi-node)
│   ├── run_finetune.sh       # VLA fine-tuning
│   ├── run_rl.sh             # RL training (PPO/SAC/GRPO)
│   ├── run_il.sh             # IL training (BC/DAgger/GAIL)
│   └── run_eval.sh           # Model evaluation
├── requirements.txt          # Python dependencies
└── setup.py                  # Package setup
```

---

## Training Methods: Purpose and Benefits

This section explains each training paradigm, when to use it, and what benefits it provides.

### Stage 1: VLM Pretraining

#### 1a. Vision-Language Alignment

| Aspect | Description |
|--------|-------------|
| **Purpose** | Train a projector to map visual features into the LLM's embedding space |
| **What's Trained** | Only the vision projector (MLP or attention-based) |
| **Dataset** | Image-caption pairs (CC3M, LAION-400M) |
| **Benefits** | - Enables LLM to "understand" visual inputs<br>- Cheap to train (only small projector)<br>- Can use frozen pretrained vision encoder and LLM |
| **When to Use** | When building a VLM from scratch; skip if using pretrained VLM (LLaVA, Qwen-VL) |

#### 1b. Visual Instruction Tuning

| Aspect | Description |
|--------|-------------|
| **Purpose** | Fine-tune the model to follow visual instructions and answer questions about images |
| **What's Trained** | Projector + LLM (vision encoder stays frozen) |
| **Dataset** | Visual instruction data (LLaVA-Instruct-150K, ShareGPT-4V) |
| **Benefits** | - Enables instruction-following with images<br>- Improves visual reasoning and grounding<br>- Creates general-purpose multimodal assistant |
| **When to Use** | When you need a conversational VLM before adding action capabilities |

---

### Stage 2: Supervised Fine-tuning (SFT)

| Aspect | Description |
|--------|-------------|
| **Purpose** | Add action prediction capability by training an action head on robot demonstrations |
| **What's Trained** | Action head (+ optionally VLM with LoRA) |
| **Dataset** | Robot demonstration datasets (Bridge V2, RT-1, ALOHA, PushT) |
| **Benefits** | - Direct mapping from vision+language to actions<br>- Simple and stable training<br>- Works well with high-quality demonstrations<br>- Fast inference at deployment |
| **When to Use** | Core stage for any VLA model; always required before RL/IL refinement |

---

### Stage 3: Policy Improvement

#### Reinforcement Learning (RL)

##### PPO (Proximal Policy Optimization)

| Aspect | Description |
|--------|-------------|
| **Purpose** | Improve policy through environment interaction with reward signal |
| **Algorithm Type** | On-policy, actor-critic |
| **Benefits** | - Stable training with clipped objective<br>- Good sample efficiency for on-policy methods<br>- Works well with discrete and continuous actions<br>- Supports parallel environment rollouts |
| **When to Use** | When you have a simulator and want stable online RL training |
| **Limitations** | Requires environment interaction; cannot use offline data efficiently |

##### SAC (Soft Actor-Critic)

| Aspect | Description |
|--------|-------------|
| **Purpose** | Learn maximum-entropy policy for robust, exploratory behavior |
| **Algorithm Type** | Off-policy, actor-critic with entropy regularization |
| **Benefits** | - Excellent sample efficiency (reuses past experience)<br>- Automatic temperature tuning<br>- Learns diverse, robust policies<br>- Good exploration through entropy maximization |
| **When to Use** | When sample efficiency is critical; when you want robust policies |
| **Limitations** | Can be unstable with function approximation; requires replay buffer |

##### GRPO (Group Relative Policy Optimization)

| Aspect | Description |
|--------|-------------|
| **Purpose** | Optimize VLA using LLM-style preference learning |
| **Algorithm Type** | Group-based policy optimization with KL regularization |
| **Benefits** | - Designed specifically for LLM-based policies<br>- Compares multiple rollouts to determine better actions<br>- Maintains proximity to base policy via KL penalty<br>- Works well with language-conditioned tasks |
| **When to Use** | For VLA models where you want LLM-style optimization |
| **Limitations** | Requires multiple rollouts per update; needs clear reward signal |

#### Imitation Learning (IL)

##### BC (Behavioral Cloning)

| Aspect | Description |
|--------|-------------|
| **Purpose** | Learn policy by supervised learning on expert demonstrations |
| **Algorithm Type** | Offline, supervised learning |
| **Benefits** | - Simplest IL method; just supervised learning<br>- No environment interaction needed<br>- Fast training<br>- Works with any demonstration format |
| **When to Use** | When you have high-quality demonstrations; as baseline or warm-start |
| **Limitations** | Distribution shift problem (compounding errors); needs large diverse datasets |

##### DAgger (Dataset Aggregation)

| Aspect | Description |
|--------|-------------|
| **Purpose** | Address BC's distribution shift by querying expert during policy rollouts |
| **Algorithm Type** | Interactive imitation learning |
| **Benefits** | - Reduces distribution shift problem<br>- Learns to recover from mistakes<br>- Provably converges to expert performance<br>- Progressive reduction of expert involvement |
| **When to Use** | When you have access to an expert policy and environment simulator |
| **Limitations** | Requires expert availability; needs environment for rollouts |

##### GAIL (Generative Adversarial Imitation Learning)

| Aspect | Description |
|--------|-------------|
| **Purpose** | Learn reward function from demonstrations, then optimize via RL |
| **Algorithm Type** | Adversarial learning + RL |
| **Benefits** | - Learns implicit reward from demonstrations<br>- Can generalize beyond demonstrated states<br>- Doesn't require explicit reward engineering<br>- Works with limited demonstrations |
| **When to Use** | When you lack a reward function but have expert demonstrations |
| **Limitations** | Training can be unstable; requires careful tuning; needs environment |

---

### Training Method Comparison

| Method | Environment Required | Expert Required | Sample Efficiency | Stability | Best For |
|--------|---------------------|-----------------|-------------------|-----------|----------|
| **BC** | No | Yes (data only) | High (offline) | Very High | Quick start, warm-start |
| **DAgger** | Yes | Yes (online) | Medium | High | Reducing distribution shift |
| **GAIL** | Yes | Yes (data only) | Low | Medium | No reward function |
| **PPO** | Yes | No | Medium | High | Stable online RL |
| **SAC** | Yes | No | High | Medium | Sample-efficient RL |
| **GRPO** | Yes | No | Medium | High | LLM-based VLA |

---

### Action Head Comparison

| Action Head | Purpose | Benefits | Best For |
|-------------|---------|----------|----------|
| **MLP** | Simple deterministic action prediction | Fast, simple, low memory | Single-mode action distributions |
| **Gaussian MLP** | Stochastic action with uncertainty | Uncertainty estimation, RL-compatible | RL training, uncertainty-aware control |
| **Diffusion** | Multi-modal action distribution | Handles complex distributions, high quality | Multi-modal actions, precise manipulation |
| **Transformer** | Autoregressive action sequence | Temporal modeling, action chunking | Long action sequences, ACT-style training |

---

## VLA Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VLA Training Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: VLM Pretraining (Optional - use pretrained VLM)                  │
│     │                                                                       │
│     ├── 1a. Vision-Language Alignment                                       │
│     │       └── Train projector to align vision encoder with LLM           │
│     │                                                                       │
│     └── 1b. Visual Instruction Tuning                                       │
│             └── Fine-tune on multimodal instruction data                    │
│                                                                             │
│  Stage 2: VLA Supervised Fine-tuning                                        │
│     │                                                                       │
│     └── Train action head on robot demonstrations                           │
│         ├── Behavioral Cloning (BC)                                         │
│         └── Action Chunking (ACT-style)                                     │
│                                                                             │
│  Stage 3: Policy Improvement (Optional)                                     │
│     │                                                                       │
│     ├── 3a. Reinforcement Learning                                          │
│     │       ├── PPO (On-policy)                                             │
│     │       ├── SAC (Off-policy)                                            │
│     │       └── GRPO (LLM-style optimization)                               │
│     │                                                                       │
│     └── 3b. Interactive Imitation Learning                                  │
│             └── DAgger (expert-in-the-loop)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: VLM Pretraining (Optional)

> **Note**: You can skip this stage by using a pretrained VLM like LLaVA, Qwen-VL, or PaliGemma.

### 1a. Vision-Language Alignment

Train a projector to align visual features with the LLM embedding space.

```python
from model import VisionEncoder, VisionProjector
from train.pretrain import AlignmentTrainer
from config import PretrainingConfig

# Configure
config = PretrainingConfig(
    vision_encoder="openai/clip-vit-large-patch14",
    llm_model="meta-llama/Llama-3.2-3B",
    projector_type="mlp",
    freeze_vision=True,
    freeze_llm=True,
    learning_rate=1e-3,
    batch_size=256,
    num_epochs=1,
)

# Dataset: Image-caption pairs (CC3M, LAION, etc.)
# Only the projector is trained

trainer = AlignmentTrainer(config)
trainer.train()
```

**What's trained**: Only the vision projector (MLP or attention-based)
**Dataset**: Image-caption pairs (CC3M, LAION-400M)

### 1b. Visual Instruction Tuning

Fine-tune the model on multimodal instruction-following data.

```python
from train.pretrain import InstructionTrainer

config = PretrainingConfig(
    freeze_vision=True,
    freeze_llm=False,  # Unfreeze LLM
    learning_rate=2e-5,
    batch_size=128,
    num_epochs=1,
)

# Dataset: LLaVA-Instruct, ShareGPT-4V, etc.
trainer = InstructionTrainer(config)
trainer.train()
```

**What's trained**: Projector + LLM (LoRA optional)
**Dataset**: Visual instruction data (LLaVA-Instruct-150K)

---

## Stage 2: VLA Supervised Fine-tuning

This is the core stage where we add action prediction capability.

### Option A: Basic VLA Fine-tuning

```python
from model import VLAModel, MLPActionHead
from train.finetune import VLAFineTuner
from train.datasets import BridgeDataset, create_openx_dataloader
from config import FineTuningConfig

# 1. Create VLA model
model = VLAModel(
    vlm_name_or_path="llava-hf/llava-1.5-7b-hf",
    action_head=MLPActionHead(
        hidden_dim=1024,
        action_dim=7,  # 6 DoF + gripper
        num_layers=3,
    ),
    freeze_vlm=True,  # Only train action head initially
)

# 2. Load dataset
dataset = BridgeDataset(split="train", max_samples=50000)
dataloader = create_openx_dataloader(dataset, batch_size=32)

# 3. Configure training
config = FineTuningConfig(
    learning_rate=1e-4,
    num_epochs=50,
    action_loss="mse",  # or "smooth_l1"
    gradient_accumulation_steps=4,
)

# 4. Train
trainer = VLAFineTuner(model, config)
trainer.train(dataloader)
```

### Option B: Multi-Sensor VLA (for Autonomous Driving)

```python
from model import MultiSensorVLA, PointNetEncoder, RadarEncoder
from model import GatedFusion  # Learned modality weighting
from train.datasets import NuScenesDataset, create_driving_dataloader

# Create multi-sensor VLA
model = MultiSensorVLA(
    vlm_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    sensor_encoders={
        "lidar": PointNetEncoder(output_dim=512),
        "radar": RadarEncoder(output_dim=256),
    },
    action_dim=3,  # steering, throttle, brake
)

# Load driving dataset
dataset = NuScenesDataset(
    data_root="/path/to/nuscenes",
    version="v1.0-trainval",
    use_lidar=True,
    use_radar=True,
)
dataloader = create_driving_dataloader(dataset, batch_size=16)

trainer = VLAFineTuner(model, config)
trainer.train(dataloader)
```

#### Available Fusion Strategies

```python
from model import SensorFusion, CrossModalFusion, HierarchicalFusion, GatedFusion

# Simple concatenation + self-attention
fusion = SensorFusion(feature_dim=512, num_heads=8)

# Cross-attention between modalities
fusion = CrossModalFusion(query_dim=512, key_dim=256, num_heads=8)

# Hierarchical spatial-temporal fusion
fusion = HierarchicalFusion(
    feature_dim=512,
    num_spatial_layers=2,
    num_temporal_layers=2,
)

# Learned gating for modality weighting (recommended for multi-sensor)
fusion = GatedFusion(
    modality_dims=[512, 256, 128],  # LiDAR, Radar, IMU
    output_dim=512,
)
```

### Option C: Action Chunking (ACT-style)

For temporal action prediction with Transformer action head:

```python
from model import VLAModel, TransformerActionHead

model = VLAModel(
    vlm_name_or_path="HuggingFaceTB/SmolVLM-Instruct",
    action_head=TransformerActionHead(
        action_dim=7,
        chunk_size=16,  # Predict 16 future actions
        num_layers=4,
    ),
)
```

### Option D: Diffusion Action Head with DDIM Sampling

For multi-modal action distributions with fast inference:

```python
from model import VLAModel, DiffusionActionHead

model = VLAModel(
    vlm_name_or_path="llava-hf/llava-1.5-7b-hf",
    action_head=DiffusionActionHead(
        action_dim=7,
        num_diffusion_steps=100,
        hidden_dim=512,
    ),
)

# Fast inference with DDIM (5-10x faster than DDPM)
action = model.action_head.sample(
    features,
    num_inference_steps=20,  # Only 20 steps instead of 100
    use_ddim=True,           # Enable DDIM for deterministic, fast sampling
)
```

### Option E: Stochastic Actions with Gaussian MLP

For uncertainty-aware action prediction:

```python
from model import VLAModel, GaussianMLPActionHead

model = VLAModel(
    vlm_name_or_path="HuggingFaceTB/SmolVLM-Instruct",
    action_head=GaussianMLPActionHead(
        input_dim=1024,
        action_dim=7,
        hidden_dim=512,
        min_std=0.01,
        max_std=1.0,
    ),
)

# During inference, sample actions with learned uncertainty
action, log_prob = model.action_head.sample(features)
```

---

## Stage 3: Policy Improvement

After supervised fine-tuning, improve the policy with RL or interactive IL.

### Option 3a: Reinforcement Learning

#### PPO (Recommended for Online RL)

```python
from train.rl import PPOTrainer
from config import RLConfig

config = RLConfig(
    algorithm="ppo",
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    entropy_coef=0.01,
    num_envs=8,
    steps_per_update=2048,
)

trainer = PPOTrainer(model, env, config)
trainer.train(total_timesteps=1_000_000)
```

#### SAC (For Sample Efficiency)

```python
from train.rl import SACTrainer
from train.datasets import ReplayBuffer

buffer = ReplayBuffer(capacity=1_000_000)
trainer = SACTrainer(model, env, buffer, config)
trainer.train()
```

#### GRPO (For VLA with LLM)

Group Relative Policy Optimization - optimizes VLA like an LLM:

```python
from train.rl import GRPOTrainer

config = RLConfig(
    algorithm="grpo",
    group_size=4,  # Compare 4 rollouts
    kl_coef=0.1,
)

trainer = GRPOTrainer(model, env, config)
trainer.train()
```

### Option 3b: DAgger (Interactive Imitation Learning)

```python
from train.il import DAgger

config = ILConfig(
    beta_schedule="linear",  # Expert intervention decreases over time
    num_iterations=10,
    rollouts_per_iteration=100,
)

trainer = DAgger(model, env, expert_policy, config)
trainer.train()
```

---

## Recommended Training Recipes

### Recipe 1: Quick Start (Robotics)

For a simple manipulation task:

```python
# 1. Use pretrained SmolVLM (smallest, fastest)
# 2. Train on PushT dataset (simple, fast to train)
# 3. Use MLP action head

from model import create_vla_model
from train.datasets import PushTDataset
from train.il import BehavioralCloning

model = create_vla_model(
    vlm="smolvlm",
    action_head="mlp",
    action_dim=2,
)

dataset = PushTDataset(split="train")
trainer = BehavioralCloning(model, dataset)
trainer.train(num_epochs=100)
```

### Recipe 2: Production (Robotics)

For real robot deployment:

```python
# 1. Use OpenVLA or fine-tuned LLaVA
# 2. Train on Bridge V2 + RT-1 combined
# 3. Use Diffusion action head for multi-modal actions
# 4. Fine-tune with GRPO for task success

from model import OpenVLAWrapper, DiffusionActionHead
from train.datasets import BridgeDataset, RT1Dataset
from torch.utils.data import ConcatDataset

model = OpenVLAWrapper(
    action_head=DiffusionActionHead(action_dim=7),
)

dataset = ConcatDataset([
    BridgeDataset(split="train"),
    RT1Dataset(split="train[:10000]"),
])

# Stage 1: BC
bc_trainer = BehavioralCloning(model, dataset)
bc_trainer.train(num_epochs=50)

# Stage 2: RL fine-tuning
grpo_trainer = GRPOTrainer(model, env, config)
grpo_trainer.train(total_timesteps=100_000)
```

### Recipe 3: Autonomous Driving

```python
from model import MultiSensorVLA, PointTransformerEncoder
from train.datasets import NuScenesDataset

model = MultiSensorVLA(
    vlm="qwen2-vl",
    sensor_encoders={
        "lidar": PointTransformerEncoder(output_dim=512),
    },
    action_dim=3,
)

dataset = NuScenesDataset(
    use_lidar=True,
    prediction_horizon=10,  # Predict 10 future waypoints
)

# BC training (offline RL not recommended for driving)
trainer = BehavioralCloning(model, dataset)
trainer.train()
```

---

## Behavior Cloning with D4RL

The framework includes a dedicated BC dataset based on D4RL for training policies via supervised imitation learning.

### Quick Start: Behavior Cloning

```python
from train.datasets import BCDataset, create_bc_dataloader
from train.il import BehavioralCloning

# 1. Load expert demonstrations from D4RL
dataset = BCDataset(env_name="hopper-expert-v2")
dataloader = create_bc_dataloader(dataset, batch_size=256)

# 2. Train with behavioral cloning
trainer = BehavioralCloning(env, policy)
trainer.train(states=dataset.observations, actions=dataset.actions)
```

### Filtered BC (for mixed-quality data)

When using datasets with varying demonstration quality, filter to top trajectories:

```python
from train.datasets import FilteredBCDataset, create_bc_dataloader

# Automatically filters to top 10% trajectories by return
dataset = FilteredBCDataset(
    env_name="hopper-medium-v2",
    filter_top_k=0.1,  # Top 10%
)
dataloader = create_bc_dataloader(dataset, batch_size=256)
```

### Weighted BC

Use all data but prioritize higher-quality demonstrations:

```python
from train.datasets import WeightedBCDataset, create_bc_dataloader

dataset = WeightedBCDataset(
    env_name="hopper-medium-v2",
    temperature=1.0,  # Lower = more weight on high-return trajectories
)

# Use weighted sampling
dataloader = create_bc_dataloader(
    dataset,
    batch_size=256,
    use_weighted_sampling=True,
)
```

### Recommended D4RL Datasets for BC

| Dataset Type | Examples | Use Case |
|--------------|----------|----------|
| **Expert** | `hopper-expert-v2`, `walker2d-expert-v2` | Best quality, use directly |
| **Medium-Expert** | `hopper-medium-expert-v2` | Good balance, filter recommended |
| **Kitchen** | `kitchen-complete-v0` | Multi-task manipulation |
| **Adroit** | `pen-expert-v1`, `door-expert-v1` | Dexterous manipulation |

---

## Dataset Selection Guide

| Dataset | Best For | Training Method |
|---------|----------|-----------------|
| **BCDataset** (D4RL Expert) | Behavior Cloning | BC, Filtered BC |
| **LeRobot** (PushT, ALOHA) | Quick testing, Imitation Learning | BC, DAgger |
| **Open X-Embodiment** (Bridge, RT-1) | VLA fine-tuning with language | BC, VLA SFT |
| **D4RL** (MuJoCo, Antmaze) | Offline RL research | CQL, IQL, DT |
| **RoboMimic** | Robot manipulation IL | BC, Action Chunking |
| **nuScenes** | Autonomous driving | End-to-end BC |
| **CARLA** | Driving simulation + RL | PPO, SAC |

---

## Supported Datasets

### Robot Manipulation
- **LeRobot**: PushT, ALOHA (sim), xArm, Unitree G1, UCSD Kitchen
- **Open X-Embodiment**: Bridge V2, RT-1, DROID

### Autonomous Driving
- nuScenes, Waymo, KITTI, CARLA

### Offline RL (D4RL)
- MuJoCo: Hopper, Walker2D, HalfCheetah
- Adroit: Pen, Door, Hammer
- Kitchen: Multi-task manipulation

---

## Hardware Requirements

| Model Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| SmolVLM (256M) | 8GB | RTX 3080, RTX 4070 |
| LLaVA-1.5 (7B) | 24GB | RTX 3090, RTX 4090 |
| LLaVA-1.5 (13B) | 48GB | A6000, A100 |
| OpenVLA (7B) | 24GB | RTX 4090, A6000 |

**Tips**:
- Use gradient checkpointing for larger models
- Use LoRA for memory-efficient fine-tuning
- Use DeepSpeed ZeRO-3 for multi-GPU training

### Memory-Efficient Training

For limited GPU memory (8-16GB), use the built-in memory-efficient configuration:

```python
from config import get_training_config

# Pre-configured memory-efficient settings
config = get_training_config("finetune-memory-efficient")
# - Batch size: 2 with gradient accumulation (effective batch: 16)
# - LoRA enabled with r=16
# - Mixed precision (bf16)
# - Vision and LLM frozen

# Or manually enable gradient checkpointing
from model import create_vla_model

model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    use_gradient_checkpointing=True,  # Reduces memory ~30%
)
```

### Available Training Configurations

```python
from config import get_training_config

# Fine-tuning configs
config = get_training_config("finetune-quick")            # Quick testing
config = get_training_config("finetune-standard")         # Standard training
config = get_training_config("finetune-full")             # Full fine-tuning
config = get_training_config("finetune-lora")             # LoRA fine-tuning
config = get_training_config("finetune-memory-efficient") # Low memory (8-16GB)

# RL configs
config = get_training_config("rl-ppo")   # PPO
config = get_training_config("rl-sac")   # SAC
config = get_training_config("rl-grpo")  # GRPO for VLA

# IL configs
config = get_training_config("il-bc")    # Behavioral Cloning
config = get_training_config("il-gail")  # GAIL
```

---

## Installation

```bash
# Core dependencies
pip install torch torchvision transformers accelerate

# VLM specific
pip install lerobot  # For LeRobot datasets
pip install tensorflow-datasets  # For Open X-Embodiment

# RL specific
pip install gymnasium mujoco

# Driving specific
pip install nuscenes-devkit
```

---

## Running Training via SLURM

Each training method has its own SLURM submission script for multi-node distributed training with DeepSpeed ZeRO-3.

### Pretraining

```bash
# VLM Pretraining (2-stage: alignment + instruction tuning)
sbatch scripts/run_pretrain.sh --stage alignment --num_epochs 1
sbatch scripts/run_pretrain.sh --stage instruction --num_epochs 3
```

### Fine-tuning

```bash
# VLA supervised fine-tuning
sbatch scripts/run_finetune.sh --dataset bridge_v2 --num_epochs 50
```

### Reinforcement Learning

```bash
# PPO training (default)
sbatch scripts/run_rl.sh --env CartPole-v1 --total_timesteps 100000

# SAC training (change Python script in run_rl.sh to sac_trainer.py)
sbatch scripts/run_rl.sh --env Pendulum-v1 --total_timesteps 100000

# GRPO for VLA
sbatch scripts/run_rl.sh --model_path ./output/vla_model.pt --reward_type smoothness
```

### Imitation Learning

```bash
# Behavioral Cloning (default)
sbatch scripts/run_il.sh --env CartPole-v1 --bc_epochs 100

# Behavioral Cloning for VLA
sbatch scripts/run_il.sh --model_path ./output/vla_model.pt --bc_epochs 50

# DAgger (change Python script in run_il.sh to dagger.py)
sbatch scripts/run_il.sh --env CartPole-v1 --dagger_iterations 10

# GAIL (change Python script in run_il.sh to gail.py)
sbatch scripts/run_il.sh --env CartPole-v1 --total_timesteps 100000
```

### Evaluation

```bash
# Evaluate trained model
sbatch scripts/run_eval.sh --model_path ./output/policy.pt --env CartPole-v1
```

---

## CLI Arguments Reference

### Behavioral Cloning (`train/il/behavioral_cloning.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | CartPole-v1 | Gymnasium environment |
| `--expert_data` | None | Path to expert data (.npz) |
| `--num_expert_episodes` | 50 | Episodes to collect from expert |
| `--model_path` | None | VLA model path (enables VLA BC mode) |
| `--bc_epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--output_dir` | ./output/bc | Output directory |

### DAgger (`train/il/dagger.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | CartPole-v1 | Gymnasium environment |
| `--dagger_iterations` | 10 | Number of DAgger iterations |
| `--dagger_beta_schedule` | linear | Beta schedule (linear, exponential, constant) |
| `--rollouts_per_iter` | 50 | Rollouts per iteration |
| `--model_path` | None | VLA model path (enables VLA DAgger mode) |

### GAIL (`train/il/gail.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | CartPole-v1 | Gymnasium environment |
| `--gail_disc_hidden_dim` | 256 | Discriminator hidden dimension |
| `--gail_disc_updates` | 5 | Discriminator updates per policy update |
| `--gail_disc_lr` | 3e-4 | Discriminator learning rate |
| `--total_timesteps` | 100000 | Total training timesteps |
| `--model_path` | None | VLA model path (enables VLA GAIL mode) |

### PPO (`train/rl/ppo_trainer.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | CartPole-v1 | Gymnasium environment |
| `--total_timesteps` | 100000 | Total training timesteps |
| `--rollout_steps` | 2048 | Steps per rollout |
| `--ppo_epochs` | 4 | PPO epochs per update |
| `--ppo_clip_range` | 0.2 | PPO clip range |
| `--ppo_gae_lambda` | 0.95 | GAE lambda |

### SAC (`train/rl/sac_trainer.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | Pendulum-v1 | Gymnasium environment (continuous only) |
| `--total_timesteps` | 100000 | Total training timesteps |
| `--sac_tau` | 0.005 | Soft update coefficient |
| `--sac_buffer_size` | 1000000 | Replay buffer size |
| `--sac_learning_starts` | 1000 | Steps before learning |

### GRPO (`train/rl/grpo_trainer.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to VLA model |
| `--grpo_group_size` | 8 | Samples per prompt |
| `--grpo_kl_coef` | 0.1 | KL divergence coefficient |
| `--reward_type` | smoothness | Reward function (distance, smoothness) |
| `--num_epochs` | 10 | Training epochs |

---

## Quick Start Example

```python
from model import create_vla_model
from train.datasets import PushTDataset, create_lerobot_dataloader
from train.il import BehavioralCloning
from config import ILConfig

# 1. Create model
model = create_vla_model(
    vlm="smolvlm",
    action_head="mlp",
    action_dim=2,
)

# 2. Load dataset
dataset = PushTDataset(split="train[:1000]")
dataloader = create_lerobot_dataloader(dataset, batch_size=64)

# 3. Train with Behavioral Cloning
config = ILConfig(
    learning_rate=1e-4,
    num_epochs=100,
)

trainer = BehavioralCloning(model, config)
trainer.train(dataloader)

# 4. Save model
model.save_pretrained("./my_vla_model")
```

---

## File Summary

| Directory | Files | Purpose |
|-----------|-------|---------|
| `config/` | 4 | Model, dataset, and training configuration dataclasses |
| `model/vlm/` | 3 | Vision encoders (SigLIP, CLIP, DINOv2) and projectors |
| `model/action_head/` | 4 | Action prediction heads (MLP, Diffusion, Transformer) |
| `model/sensor/` | 4 | Sensor encoders for LiDAR, Radar, IMU |
| `model/vla/` | 5 | Complete VLA model implementations |
| `model/fusion/` | 2 | Multi-modal sensor fusion strategies |
| `train/pretrain/` | 4 | VLM pretraining (alignment + instruction tuning) |
| `train/finetune/` | 3 | Supervised fine-tuning for VLA |
| `train/rl/` | 5 | Reinforcement learning trainers (PPO, SAC, GRPO) |
| `train/il/` | 5 | Imitation learning trainers (BC, DAgger, GAIL) + VLA variants |
| `train/datasets/` | 6 | Dataset loaders for various robot/driving datasets |
| `eval/` | 4 | Evaluation metrics and benchmark suites |
| `scripts/` | 5 | SLURM job submission scripts |
| **Total** | **54** | Complete VLA training framework |

---

## References

- [OpenVLA](https://openvla.github.io/) - Open-source VLA model
- [RT-2](https://robotics-transformer2.github.io/) - Vision-Language-Action model from Google
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace robot learning library
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - Cross-embodiment dataset
- [D4RL](https://github.com/Farama-Foundation/D4RL) - Offline RL benchmark datasets

---

## License

MIT License
