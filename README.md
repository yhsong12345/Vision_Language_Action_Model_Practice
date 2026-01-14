# Vision-Language-Action (VLA) Model Training Framework

This repository implements a complete Vision-Language-Action training pipeline from data preparation to real-robot deployment.

A comprehensive framework for training Vision-Language-Action models for robotics and autonomous systems. This framework provides a complete pipeline from VLM pretraining to policy deployment, supporting multiple training paradigms including imitation learning, reinforcement learning, and world model-based approaches.

## Overview

VLA models combine vision understanding, language reasoning, and action prediction to enable robots to perform tasks based on visual observations and natural language instructions. This framework implements the complete training pipeline:

1. **VLM Pretraining**: Align vision encoders with language models
2. **Action Head Training**: Add action prediction capabilities via supervised fine-tuning
3. **Policy Improvement**: Refine policies using RL, IL, or world model-based methods
4. **Deployment**: Deploy to simulators or real robots via ROS integration

## Project Structure

```
Vision_Language_Model_Practice/
├── config/                      # Configuration classes
│   ├── model_config.py          # VLA, MultiSensor, OpenVLA, SmolVLA configs
│   ├── dataset_config.py        # Dataset configurations
│   └── training_config.py       # Training/RL/IL hyperparameters
├── model/                       # Model components
│   ├── vlm/                     # Vision-Language Model
│   │   ├── vision_encoder.py    # SigLIP, CLIP, DINOv2 encoders
│   │   └── vision_projector.py  # MLP, Attention, Perceiver projectors
│   ├── action_head/             # Action prediction heads
│   │   ├── mlp_action_head.py   # MLP and Gaussian MLP heads
│   │   ├── diffusion_action_head.py  # DDPM/DDIM diffusion head
│   │   └── transformer_action_head.py # Autoregressive transformer head
│   ├── vla/                     # VLA implementations
│   │   ├── vla_model.py         # Core VLA model
│   │   ├── multi_sensor_vla.py  # Multi-sensor VLA for autonomous driving
│   │   ├── openvla_wrapper.py   # OpenVLA-7B wrapper
│   │   └── smolvla_wrapper.py   # SmolVLA-450M lightweight wrapper
│   ├── sensor/                  # Sensor encoders
│   │   ├── lidar_encoder.py     # PointNet/PointTransformer for LiDAR
│   │   ├── radar_encoder.py     # CNN encoder for radar
│   │   └── imu_encoder.py       # Transformer encoder for IMU
│   ├── fusion/                  # Multi-modal fusion
│   │   └── sensor_fusion.py     # Self-attention, Cross-modal, Gated fusion
│   ├── temporal/                # Temporal modeling
│   │   ├── temporal_encoder.py  # Transformer/LSTM temporal encoding
│   │   ├── memory_buffer.py     # Episodic and working memory
│   │   └── history_encoder.py   # Action-observation history encoding
│   ├── world_model/             # World modeling
│   │   ├── dynamics_model.py    # State transition prediction
│   │   ├── latent_world_model.py # Dreamer-style RSSM
│   │   └── reward_predictor.py  # Reward function learning
│   ├── safety/                  # Safety constraints
│   │   ├── safety_shield.py     # Runtime action filtering
│   │   ├── rule_checker.py      # Rule-based constraint verification
│   │   └── constraint_handler.py # Constraint optimization
│   └── embodiment/              # Embodiment-specific models
│       ├── autonomous_vehicle.py # BEV encoder, trajectory decoder
│       └── humanoid.py          # Whole-body controller, locomotion
├── train/                       # Training pipelines
│   ├── pretrain/                # VLM pretraining
│   │   ├── vlm_pretrainer.py    # Main pretraining orchestrator
│   │   ├── alignment_trainer.py # Vision-language alignment
│   │   └── instruction_trainer.py # Visual instruction tuning
│   ├── finetune/                # Supervised fine-tuning
│   │   ├── vla_finetuner.py     # VLA fine-tuning with LoRA
│   │   └── dataset.py           # Robot dataset loader
│   ├── online_rl/               # Online RL (with environment)
│   │   ├── ppo_trainer.py       # Proximal Policy Optimization
│   │   ├── sac_trainer.py       # Soft Actor-Critic
│   │   └── grpo_trainer.py      # Group Relative Policy Optimization
│   ├── offline_rl/              # Offline RL (from static data)
│   │   ├── cql_trainer.py       # Conservative Q-Learning
│   │   ├── iql_trainer.py       # Implicit Q-Learning
│   │   ├── td3_bc_trainer.py    # TD3 + Behavioral Cloning
│   │   └── decision_transformer.py # Decision Transformer
│   ├── il/                      # Imitation learning
│   │   ├── behavioral_cloning.py # BC implementation
│   │   ├── dagger.py            # Dataset Aggregation
│   │   └── gail.py              # Generative Adversarial IL
│   ├── world_model/             # World model training
│   │   └── train_world_model.py # Dreamer-style training
│   ├── embodiment/              # Embodiment-specific training
│   │   ├── train_driving_vla.py # Autonomous driving training
│   │   └── train_humanoid_vla.py # Humanoid robot training
│   └── datasets/                # Dataset loaders
│       ├── lerobot_dataset.py   # LeRobot (PushT, ALOHA, xArm)
│       ├── openx_dataset.py     # Open X-Embodiment
│       ├── driving_dataset.py   # nuScenes, Waymo, CARLA
│       ├── rl_dataset.py        # Offline RL datasets
│       └── bc_dataset.py        # D4RL BC datasets
├── eval/                        # Evaluation
│   ├── evaluator.py             # Main evaluation orchestrator
│   ├── metrics.py               # Success rate, trajectory metrics
│   └── benchmark.py             # Benchmark suites
├── integration/                 # System integration
│   ├── ros_bridge.py            # ROS/ROS2 integration
│   ├── simulator_bridge.py      # CARLA, Isaac Sim, MuJoCo
│   └── experiment_manager.py    # Experiment tracking (W&B, MLflow)
└── scripts/                     # SLURM training scripts
    ├── run_pretrain.sh          # VLM pretraining
    ├── run_finetune.sh          # Supervised fine-tuning
    ├── run_il.sh                # Imitation learning
    ├── run_online_rl_*.sh       # Online RL (PPO, SAC, GRPO)
    ├── run_offline_rl_*.sh      # Offline RL (CQL, IQL, TD3+BC, DT)
    ├── run_world_model.sh       # World model training
    ├── run_driving_vla.sh       # Autonomous driving VLA
    ├── run_humanoid_vla.sh      # Humanoid robot VLA
    └── run_all_training.sh      # Complete pipeline orchestrator
```

---

## Training Pipeline

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
│  Stage 2: Action Head Training (Supervised Fine-tuning)                    │
│     │                                                                       │
│     └── Train action head on robot demonstrations                           │
│         ├── Temporal modeling (history window / action chunking)           │
│         ├── MLP / Gaussian MLP (simple, fast)                              │
│         ├── Diffusion Head (multi-modal actions)                           │
│         └── Transformer Head (action sequences)                            │
│                                                                             │
│  Stage 3: Policy Improvement                                                │
│     │                                                                       │
│     ├── 3a. Imitation-based                                                 │
│     │       ├── BC (Behavioral Cloning) - supervised baseline              │
│     │       ├── DAgger (Dataset Aggregation) - expert-in-the-loop          │
│     │       └── GAIL (Generative Adversarial IL) - learns implicit reward  │
│     │                                                                       │
│     ├── 3b. RL-based                                                        │
│     │       │                                                               │
│     │       ├── Online RL (requires simulator/environment)                 │
│     │       │       ├── PPO (Proximal Policy Optimization) - on-policy     │
│     │       │       ├── SAC (Soft Actor-Critic) - off-policy               │
│     │       │       └── GRPO (Group Relative PO) - LLM-style optimization  │
│     │       │                                                               │
│     │       └── Offline RL (from static datasets)                          │
│     │               ├── CQL (Conservative Q-Learning) - penalizes OOD      │
│     │               ├── IQL (Implicit Q-Learning) - stable, no max         │
│     │               ├── TD3+BC - TD3 with BC regularization                │
│     │               ├── Decision Transformer - sequence modeling           │
│     │               └── Reward modeling / preference learning (optional)   │
│     │                                                                       │
│     └── 3c. Model-based                                                     │
│             │                                                               │
│             └── World Model                                                 │
│                     ├── RSSM (Recurrent State-Space Model)                 │
│                     ├── Latent Dynamics Learning                           │
│                     ├── Imagination-based Planning                         │
│                     └── Dreamer-style Training                             │
│                                                                             │
│  Stage 4: Deployment                                                        │
│     │                                                                       │
│     ├── Simulator (CARLA, Isaac Sim, MuJoCo)                               │
│     └── Real Robot (ROS/ROS2 integration)                                  │
│             ├── Safety shield / rule-based controller                      │
│             └── Emergency stop                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Methods: Purpose and Benefits

### Stage 1: VLM Pretraining

| Stage | Purpose | What's Trained | Dataset | Benefits |
|-------|---------|----------------|---------|----------|
| **1a. Vision-Language Alignment** | Map visual features into LLM embedding space | Vision projector only | Image-caption pairs (CC3M, LAION) | Cheap to train, uses frozen encoders |
| **1b. Visual Instruction Tuning** | Enable instruction-following with images | Projector + LLM | LLaVA-Instruct, ShareGPT-4V | Improves visual reasoning |

### Stage 2: Supervised Fine-tuning

| Aspect | Description |
|--------|-------------|
| **Purpose** | Add action prediction capability |
| **What's Trained** | Action head (+ optional VLM LoRA) |
| **Dataset** | Robot demonstrations (Bridge V2, RT-1, ALOHA) |
| **Benefits** | Direct vision-to-action mapping, stable training |

### Stage 3: Policy Improvement

#### 3a. Imitation-based Methods

| Algorithm | Type | Benefits | Best For |
|-----------|------|----------|----------|
| **BC** | Supervised | Simple, fast, stable | High-quality expert demos |
| **DAgger** | Interactive | Reduces distribution shift | Expert available online |
| **GAIL** | Adversarial | Learns implicit reward function | Limited demos, no reward |

#### 3b. RL-based Methods

**Online RL** (requires simulator/environment interaction)

| Algorithm | Type | Benefits | Best For |
|-----------|------|----------|----------|
| **PPO** | On-policy | Stable training, clipped objective | Simulator-based training |
| **SAC** | Off-policy | Sample efficient, entropy regularization | Continuous control |
| **GRPO** | LLM-style | Designed for VLA, KL-constrained | Language-conditioned tasks |

**Offline RL** (from static datasets, no environment needed)

| Algorithm | Type | Benefits | Best For |
|-----------|------|----------|----------|
| **CQL** | Conservative | Penalizes OOD actions | Mixed-quality data |
| **IQL** | Implicit | No max over actions, stable | Suboptimal demonstrations |
| **TD3+BC** | Hybrid | BC regularization | Near-expert data |
| **Decision Transformer** | Sequence | No value function, return-conditioned | Long-horizon tasks |

#### 3c. Model-based Methods

| Component | Purpose | Benefits |
|-----------|---------|----------|
| **World Model (RSSM)** | Learn latent dynamics | Sample efficient, imagination-based planning |
| **Latent Dynamics** | Predict future states | Enables rollouts without real environment |
| **Reward Predictor** | Learn reward function | Model-based RL without explicit reward |
| **Dreamer Training** | End-to-end world model RL | State-of-the-art sample efficiency |

### Training Method Comparison

| Method | Category | Environment | Expert | Sample Efficiency | Stability | Best For |
|--------|----------|-------------|--------|-------------------|-----------|----------|
| **BC** | Imitation | No | Data only | High | Very High | Quick start |
| **DAgger** | Imitation | Yes | Online | Medium | High | Distribution shift |
| **GAIL** | Imitation | Yes | Data only | Low | Medium | No reward function |
| **PPO** | Online RL | Yes | No | Medium | High | Stable online RL |
| **SAC** | Online RL | Yes | No | High | Medium | Sample-efficient RL |
| **CQL** | Offline RL | No | Data only | High | High | Offline with OOD |
| **IQL** | Offline RL | No | Data only | High | Very High | Stable offline |
| **DT** | Offline RL | No | Data only | High | High | Return conditioning |
| **Dreamer** | Model-based | Yes | No | Very High | Medium | Limited interactions |

---

## Action Head Comparison

| Action Head | Type | Benefits | Best For |
|-------------|------|----------|----------|
| **MLP** | Deterministic | Fast, simple | Single-mode actions |
| **Gaussian MLP** | Stochastic | Uncertainty estimation | RL training |
| **Diffusion** | Multi-modal | Handles complex distributions | Precise manipulation |
| **Transformer** | Autoregressive | Temporal modeling | Action sequences |

---

## Datasets

### Robot Manipulation

| Dataset | Source | Description |
|---------|--------|-------------|
| **LeRobot** | HuggingFace | PushT, ALOHA, xArm, Unitree |
| **Open X-Embodiment** | Google | Bridge V2, RT-1, DROID |
| **RoboMimic** | Stanford | Multi-task manipulation |

### Autonomous Driving

| Dataset | Source | Description |
|---------|--------|-------------|
| **nuScenes** | Motional | Multi-sensor driving data |
| **Waymo** | Waymo | Large-scale driving dataset |
| **CARLA** | Simulation | Configurable driving scenarios |

### Offline RL

| Dataset | Source | Description |
|---------|--------|-------------|
| **D4RL** | UC Berkeley | MuJoCo, Antmaze, Adroit |
| **Kitchen** | Google | Multi-task kitchen manipulation |

---

## Quick Start

### Basic VLA Training

```python
from model import create_vla_model, MLPActionHead
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
config = ILConfig(learning_rate=1e-4, num_epochs=100)
trainer = BehavioralCloning(model, config)
trainer.train(dataloader)

# 4. Save model
model.save_pretrained("./my_vla_model")
```

### Multi-Sensor VLA for Autonomous Driving

```python
from model import MultiSensorVLA, PointNetEncoder, RadarEncoder
from train.datasets import NuScenesDataset

model = MultiSensorVLA(
    vlm_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    sensor_encoders={
        "lidar": PointNetEncoder(output_dim=512),
        "radar": RadarEncoder(output_dim=256),
    },
    action_dim=3,  # steering, throttle, brake
)

dataset = NuScenesDataset(
    data_root="/path/to/nuscenes",
    use_lidar=True,
    use_radar=True,
)
```

### Offline RL Training

```python
from train.offline_rl import IQLTrainer
from train.datasets import D4RLDataset
from config import OfflineRLConfig

dataset = D4RLDataset(env_name="hopper-medium-expert-v2")
config = OfflineRLConfig(
    algorithm="iql",
    expectile=0.7,
    temperature=3.0,
)

trainer = IQLTrainer(model, dataset, config)
trainer.train(num_epochs=1000)
```

### Online RL Fine-tuning

```python
from train.online_rl import PPOTrainer
from config import RLConfig
import gymnasium as gym

env = gym.make("CartPole-v1")
config = RLConfig(
    algorithm="ppo",
    total_timesteps=100000,
    ppo_clip_range=0.2,
)

trainer = PPOTrainer(model, env, config)
trainer.train()
```

---

## Complete Training and Deployment Procedure

This section provides a step-by-step guide for the entire VLA training and deployment pipeline, from initial setup to real-world deployment.

### Overview Flowchart

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE VLA TRAINING & DEPLOYMENT FLOW                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Step 1    │───>│   Step 2    │───>│   Step 3    │───>│   Step 4    │     │
│  │  Prepare    │    │  Pretrain   │    │  Fine-tune  │    │  Improve    │     │
│  │   Data      │    │    VLM      │    │  Action     │    │  Policy     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │                  │            │
│         v                  v                  v                  v            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Step 5    │<───│   Step 6    │<───│   Step 7    │<───│   Step 8    │     │
│  │   Deploy    │    │  Evaluate   │    │   Safety    │    │  Embodiment │     │
│  │  to Robot   │    │   & Test    │    │   Layer     │    │  Specific   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Environment Setup and Data Preparation

#### 1.1 Install Dependencies

```bash
# Core dependencies
pip install torch torchvision transformers accelerate peft
pip install lerobot gymnasium mujoco

# For distributed training
pip install deepspeed

# For specific embodiments
pip install nuscenes-devkit  # Autonomous driving
pip install mujoco-py        # Humanoid simulation
```

#### 1.2 Prepare Dataset

```python
from train.datasets import (
    LeRobotDataset,       # Robot manipulation
    OpenXDataset,         # Cross-embodiment
    NuScenesDataset,      # Autonomous driving
    D4RLDataset,          # Offline RL
)

# Option A: LeRobot datasets (manipulation)
dataset = LeRobotDataset(
    repo_id="lerobot/pusht",
    split="train",
)

# Option B: Open X-Embodiment (cross-robot)
dataset = OpenXDataset(
    datasets=["bridge_v2", "rt_1_robot_action"],
    split="train",
)

# Option C: Driving dataset
dataset = NuScenesDataset(
    data_root="/path/to/nuscenes",
    version="v1.0-trainval",
    use_lidar=True,
    use_radar=True,
)

# Verify data
print(f"Dataset size: {len(dataset)}")
print(f"Sample keys: {dataset[0].keys()}")
```

#### 1.3 Data Validation Checklist

| Check | Description | Command |
|-------|-------------|---------|
| Image shapes | Consistent resolution | `assert all(d['image'].shape == (3, 224, 224))` |
| Action dimensions | Match action head | `assert d['action'].shape[-1] == action_dim` |
| Language present | Instructions available | `assert 'instruction' in d or 'language' in d` |
| No NaN/Inf | Clean data | `assert not torch.isnan(d['action']).any()` |

---

### Step 2: VLM Pretraining (Optional)

Skip this step if using pretrained VLM (recommended for most cases).

#### 2.1 Vision-Language Alignment

```python
from train.pretrain import VLMPretrainer
from config import PretrainConfig

config = PretrainConfig(
    stage="alignment",
    vlm_backbone="Qwen/Qwen2.5-VL-3B",
    vision_encoder="google/siglip-so400m-patch14-384",
    freeze_vision=True,
    freeze_llm=True,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=1,
)

pretrainer = VLMPretrainer(config)
pretrainer.train(image_caption_dataset)
pretrainer.save("./checkpoints/vlm_aligned")
```

#### 2.2 Visual Instruction Tuning

```python
config = PretrainConfig(
    stage="instruction",
    vlm_path="./checkpoints/vlm_aligned",
    freeze_vision=True,
    freeze_llm=False,  # Unfreeze LLM
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-5,
    num_epochs=3,
)

pretrainer = VLMPretrainer(config)
pretrainer.train(instruction_dataset)
pretrainer.save("./checkpoints/vlm_instructed")
```

---

### Step 3: Action Head Fine-tuning (Supervised)

#### 3.1 Choose Action Head

```python
from model import (
    create_vla_model,
    MLPActionHead,           # Simple, fast
    GaussianMLPActionHead,   # For RL training
    DiffusionActionHead,     # Multi-modal actions
    TransformerActionHead,   # Action sequences
)

# For manipulation tasks (single mode)
model = create_vla_model(
    vlm="Qwen/Qwen2.5-VL-3B",
    action_head="mlp",
    action_dim=7,  # 6 DoF + gripper
)

# For precise manipulation (multi-modal)
model = create_vla_model(
    vlm="Qwen/Qwen2.5-VL-3B",
    action_head="diffusion",
    action_dim=7,
    diffusion_steps=100,
    action_horizon=16,
)
```

#### 3.2 Train with Behavioral Cloning

```python
from train.il import BehavioralCloning
from train.datasets import create_lerobot_dataloader
from config import ILConfig

config = ILConfig(
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=100,
    gradient_accumulation_steps=4,
    use_lora=True,
    freeze_vlm=True,  # Only train action head
)

dataloader = create_lerobot_dataloader(dataset, batch_size=64)
trainer = BehavioralCloning(model, config)

# Train
for epoch in range(config.num_epochs):
    metrics = trainer.train_epoch(dataloader)
    print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}")

    # Validate every 10 epochs
    if epoch % 10 == 0:
        val_metrics = trainer.validate(val_dataloader)
        print(f"  Val Success Rate: {val_metrics['success_rate']:.2%}")

# Save checkpoint
trainer.save("./checkpoints/vla_bc")
```

---

### Step 4: Policy Improvement

Choose one or more methods based on your setup:

#### 4.1 Option A: Online RL (requires simulator)

```python
from train.online_rl import PPOTrainer, SACTrainer
from config import RLConfig
import gymnasium as gym

# Setup environment
env = gym.make("FetchPickAndPlace-v2")

# Load BC-pretrained model
model = create_vla_model.from_pretrained("./checkpoints/vla_bc")

# PPO training
config = RLConfig(
    algorithm="ppo",
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    ppo_clip_range=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    n_steps=2048,
    batch_size=64,
)

trainer = PPOTrainer(model, env, config)
trainer.train()
trainer.save("./checkpoints/vla_ppo")
```

#### 4.2 Option B: Offline RL (from static data)

```python
from train.offline_rl import IQLTrainer, CQLTrainer
from train.datasets import D4RLDataset
from config import OfflineRLConfig

# Load offline dataset
dataset = D4RLDataset(env_name="hopper-medium-expert-v2")

# IQL (recommended - stable)
config = OfflineRLConfig(
    algorithm="iql",
    expectile=0.7,
    temperature=3.0,
    learning_rate=3e-4,
    num_epochs=1000,
)

trainer = IQLTrainer(model, dataset, config)
trainer.train()
trainer.save("./checkpoints/vla_iql")
```

#### 4.3 Option C: DAgger (expert available)

```python
from train.il import DAgger

trainer = DAgger(
    model=model,
    env=env,
    expert_policy=expert_model,  # Your expert policy
    config=config,
)

# Iterative training with expert
for iteration in range(10):
    # Collect data with current policy
    trajectories = trainer.collect_rollouts(num_episodes=100)

    # Query expert for corrections
    expert_actions = trainer.query_expert(trajectories)

    # Aggregate and train
    trainer.aggregate_and_train(expert_actions)

trainer.save("./checkpoints/vla_dagger")
```

#### 4.4 Option D: World Model-based RL

```python
from model.world_model import LatentWorldModel, RSSM
from train.world_model import WorldModelTrainer

# Train world model first
world_model = LatentWorldModel(
    state_dim=256,
    action_dim=7,
    hidden_dim=512,
    use_rssm=True,
)

wm_trainer = WorldModelTrainer(
    world_model=world_model,
    dataset=trajectory_dataset,
    imagination_horizon=15,
)
wm_trainer.train(num_epochs=100)

# Then use for planning/RL
from train.online_rl import DreamerTrainer

trainer = DreamerTrainer(
    policy=model,
    world_model=world_model,
    config=config,
)
trainer.train()
```

---

### Step 5: Embodiment-Specific Training

#### 5.1 Autonomous Driving VLA

```python
from model.embodiment import DrivingVLA, BEVEncoder
from train.embodiment import DrivingVLATrainer
from train.datasets import NuScenesDataset

# Create driving-specific model
model = DrivingVLA(
    vlm_backbone="Qwen/Qwen2.5-VL-3B",
    num_cameras=6,           # Surround view
    bev_size=(200, 200),     # BEV grid
    bev_resolution=0.5,      # meters per pixel
    trajectory_length=20,    # Future waypoints
    action_dim=3,            # [steering, throttle, brake]
)

# Load nuScenes
dataset = NuScenesDataset(
    data_root="/path/to/nuscenes",
    use_lidar=True,
    use_radar=True,
)

# Train
trainer = DrivingVLATrainer(
    model=model,
    dataset=dataset,
    config=DrivingTrainConfig(
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=100,
        trajectory_loss_weight=1.0,
        control_loss_weight=0.1,
        safety_loss_weight=0.5,
    ),
)
trainer.train()
trainer.save("./checkpoints/driving_vla")
```

#### 5.2 Humanoid Robot VLA

```python
from model.embodiment import HumanoidVLA, WholeBodyController
from train.embodiment import HumanoidVLATrainer

# Create humanoid-specific model
model = HumanoidVLA(
    vlm_backbone="Qwen/Qwen2.5-VL-3B",
    num_joints=32,           # DOF
    num_body_parts=15,       # Body segments
    hidden_dim=512,
)

# Train
trainer = HumanoidVLATrainer(
    model=model,
    dataset=humanoid_dataset,
    config=HumanoidTrainConfig(
        batch_size=16,
        learning_rate=3e-4,
        num_epochs=200,
        locomotion_loss_weight=1.0,
        manipulation_loss_weight=1.0,
        stability_loss_weight=0.5,
    ),
)
trainer.train()
trainer.save("./checkpoints/humanoid_vla")
```

---

### Step 6: Add Safety Layer

```python
from model.safety import SafetyShield, RuleChecker, ConstraintHandler

# Create safety components
safety_shield = SafetyShield(
    action_dim=7,
    max_velocity=1.0,          # m/s
    max_acceleration=2.0,      # m/s²
    workspace_bounds=[         # Safe workspace
        [-0.5, 0.5],           # x
        [-0.5, 0.5],           # y
        [0.0, 0.8],            # z
    ],
)

rule_checker = RuleChecker(
    collision_threshold=0.05,  # meters
    joint_limits=joint_limits,
)

constraint_handler = ConstraintHandler(
    constraints=[
        "max_velocity",
        "workspace_bounds",
        "collision_avoidance",
    ],
)

# Wrap model with safety
class SafeVLA(nn.Module):
    def __init__(self, vla, safety_shield, rule_checker):
        super().__init__()
        self.vla = vla
        self.safety = safety_shield
        self.rules = rule_checker

    def forward(self, image, instruction, state=None):
        # Get raw action from VLA
        raw_action = self.vla(image, instruction)

        # Check rules
        violations = self.rules.check(raw_action, state)
        if violations:
            print(f"Rule violations: {violations}")

        # Apply safety filter
        safe_action = self.safety.filter(raw_action, state)

        return safe_action

safe_model = SafeVLA(model, safety_shield, rule_checker)
```

---

### Step 7: Evaluation and Testing

#### 7.1 Offline Evaluation

```python
from eval import Evaluator, Metrics

evaluator = Evaluator(
    model=model,
    dataset=test_dataset,
    metrics=["mse", "success_rate", "trajectory_error"],
)

results = evaluator.evaluate()
print(f"MSE: {results['mse']:.4f}")
print(f"Success Rate: {results['success_rate']:.2%}")
print(f"Trajectory Error: {results['trajectory_error']:.4f}m")
```

#### 7.2 Simulator Evaluation

```python
from eval import SimulatorEvaluator
from integration import SimulatorBridge

# Connect to simulator
sim_bridge = SimulatorBridge(
    simulator="isaac_sim",  # or "mujoco", "carla"
    config=sim_config,
)

evaluator = SimulatorEvaluator(
    model=safe_model,
    simulator=sim_bridge,
    num_episodes=100,
)

results = evaluator.run()
print(f"Episode Success: {results['success_rate']:.2%}")
print(f"Average Return: {results['avg_return']:.2f}")
print(f"Collision Rate: {results['collision_rate']:.2%}")
```

#### 7.3 Benchmark Suite

```python
from eval import BenchmarkSuite

benchmark = BenchmarkSuite(
    tasks=["pick_place", "stack", "pour", "wipe"],
    difficulty=["easy", "medium", "hard"],
)

results = benchmark.run(model)
benchmark.save_report("./eval_results/benchmark_report.json")
```

---

### Step 8: Deployment

#### 8.1 Model Export

```python
# Export to optimized format
import torch

# Option A: TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("./deploy/vla_scripted.pt")

# Option B: ONNX
torch.onnx.export(
    model,
    (dummy_image, dummy_instruction),
    "./deploy/vla.onnx",
    opset_version=17,
)

# Option C: TensorRT (for NVIDIA)
# Use torch2trt or polygraphy
```

#### 8.2 ROS Integration

```python
from integration import ROSBridge

# Initialize ROS node
ros_bridge = ROSBridge(
    node_name="vla_controller",
    image_topic="/camera/rgb/image_raw",
    action_topic="/robot/joint_commands",
    instruction_topic="/instruction",
    control_rate=30,  # Hz
)

# Load model
model = torch.jit.load("./deploy/vla_scripted.pt")
model.eval()

# Run control loop
@ros_bridge.on_image
def control_callback(image, instruction):
    with torch.no_grad():
        action = model(image, instruction)
    return action

ros_bridge.run()
```

#### 8.3 Real Robot Deployment Checklist

| Step | Check | Status |
|------|-------|--------|
| 1 | Model exported successfully | ☐ |
| 2 | Safety limits configured | ☐ |
| 3 | Emergency stop tested | ☐ |
| 4 | Latency < control period | ☐ |
| 5 | Workspace bounds verified | ☐ |
| 6 | Collision detection enabled | ☐ |
| 7 | Manual override available | ☐ |
| 8 | Logging enabled | ☐ |

#### 8.4 Deployment Configuration

```python
from integration import DeploymentConfig

deploy_config = DeploymentConfig(
    # Model
    model_path="./deploy/vla_scripted.pt",
    device="cuda",
    precision="fp16",

    # Safety
    enable_safety_shield=True,
    max_velocity=0.5,
    emergency_stop_enabled=True,

    # Control
    control_frequency=30,  # Hz
    action_smoothing=0.8,  # EMA coefficient

    # Logging
    log_actions=True,
    log_observations=True,
    log_dir="./logs/deployment",
)
```

---

### Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE CHECKLIST                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  □ Step 1: Data Preparation                                                     │
│     □ Dataset downloaded and validated                                          │
│     □ Train/val/test splits created                                             │
│     □ Data loader tested                                                        │
│                                                                                 │
│  □ Step 2: VLM Pretraining (optional)                                          │
│     □ Vision-language alignment complete                                        │
│     □ Instruction tuning complete                                               │
│     □ OR: Use pretrained VLM (Qwen2.5-VL, LLaVA)                               │
│                                                                                 │
│  □ Step 3: Action Head Training                                                 │
│     □ Action head selected (MLP/Diffusion/Transformer)                          │
│     □ BC training complete                                                      │
│     □ Validation loss converged                                                 │
│                                                                                 │
│  □ Step 4: Policy Improvement                                                   │
│     □ RL method selected (Online/Offline/IL)                                    │
│     □ Training complete                                                         │
│     □ Policy improved over BC baseline                                          │
│                                                                                 │
│  □ Step 5: Embodiment Training (if needed)                                      │
│     □ Driving VLA trained (autonomous vehicle)                                  │
│     □ Humanoid VLA trained (humanoid robot)                                     │
│                                                                                 │
│  □ Step 6: Safety Layer                                                         │
│     □ Safety shield configured                                                  │
│     □ Rule checker tested                                                       │
│     □ Constraint handler validated                                              │
│                                                                                 │
│  □ Step 7: Evaluation                                                           │
│     □ Offline metrics computed                                                  │
│     □ Simulator evaluation passed                                               │
│     □ Benchmark suite completed                                                 │
│                                                                                 │
│  □ Step 8: Deployment                                                           │
│     □ Model exported (TorchScript/ONNX)                                        │
│     □ ROS integration tested                                                    │
│     □ Real robot deployment verified                                            │
│     □ Emergency stop confirmed working                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Quick Reference: Which Training Method to Use?

```
                    ┌─────────────────────────────────────┐
                    │     Do you have expert demos?       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │ Yes                         │ No
                    v                             v
          ┌─────────────────┐           ┌─────────────────┐
          │ Start with BC   │           │ Need RL from    │
          │ (Behavioral     │           │ scratch         │
          │  Cloning)       │           └────────┬────────┘
          └────────┬────────┘                    │
                   │                             v
                   v                   ┌─────────────────┐
          ┌─────────────────┐          │ Use PPO/SAC     │
          │ Demos optimal?  │          │ with simulator  │
          └────────┬────────┘          └─────────────────┘
                   │
          ┌────────┴────────┐
          │ Yes             │ No (suboptimal)
          v                 v
  ┌───────────────┐  ┌───────────────┐
  │ BC is enough  │  │ Improve with  │
  │ OR use DAgger │  │ Offline RL    │
  │ for dist.shift│  │ (IQL/CQL)     │
  └───────────────┘  └───────────────┘
                            │
                            v
                   ┌─────────────────┐
                   │ Have simulator? │
                   └────────┬────────┘
                            │
                   ┌────────┴────────┐
                   │ Yes             │ No
                   v                 v
          ┌───────────────┐  ┌───────────────┐
          │ Fine-tune     │  │ Deploy with   │
          │ with Online   │  │ offline model │
          │ RL (PPO/SAC)  │  └───────────────┘
          └───────────────┘
```

---

## Running Training via SLURM

### Single-Stage Training

```bash
# VLM Pretraining
sbatch scripts/run_pretrain.sh --stage alignment --num_epochs 1

# Supervised Fine-tuning
sbatch scripts/run_finetune.sh --dataset bridge_v2 --num_epochs 50

# Online RL
sbatch scripts/run_online_rl_ppo.sh --env CartPole-v1 --total_timesteps 100000
sbatch scripts/run_online_rl_sac.sh --env Pendulum-v1 --total_timesteps 100000
sbatch scripts/run_online_rl_grpo.sh --model_path ./output/vla.pt

# Offline RL
sbatch scripts/run_offline_rl_iql.sh --dataset hopper-medium-expert-v2
sbatch scripts/run_offline_rl_cql.sh --dataset hopper-medium-v2
sbatch scripts/run_offline_rl_dt.sh --dataset hopper-expert-v2

# Imitation Learning
sbatch scripts/run_il.sh --algorithm bc --num_epochs 100
sbatch scripts/run_il.sh --algorithm dagger --iterations 10

# World Model
sbatch scripts/run_world_model.sh --latent_dim 256 --num_epochs 100

# Embodiment-Specific
sbatch scripts/run_driving_vla.sh --dataset nuscenes
sbatch scripts/run_humanoid_vla.sh --env humanoid-walk
```

### Complete Pipeline

```bash
# Run all stages sequentially
./scripts/run_all_training.sh all

# Run specific stage
./scripts/run_all_training.sh pretrain
./scripts/run_all_training.sh offline_rl
./scripts/run_all_training.sh online_rl
```

---

## Hardware Requirements

| Model Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| SmolVLM (256M) | 8GB | RTX 3080, RTX 4070 |
| LLaVA-1.5 (7B) | 24GB | RTX 3090, RTX 4090 |
| LLaVA-1.5 (13B) | 48GB | A6000, A100 |
| OpenVLA (7B) | 24GB | RTX 4090, A6000 |

### Memory-Efficient Training

```python
from config import get_training_config

# Pre-configured memory-efficient settings
config = get_training_config("finetune-memory-efficient")
# - Batch size: 2 with gradient accumulation
# - LoRA enabled with r=16
# - Mixed precision (bf16)
# - Vision and LLM frozen
```

---

## Installation

```bash
# Core dependencies
pip install torch torchvision transformers accelerate

# VLM specific
pip install lerobot  # LeRobot datasets
pip install tensorflow-datasets  # Open X-Embodiment

# RL specific
pip install gymnasium mujoco d4rl

# Driving specific
pip install nuscenes-devkit

# Optional: Distributed training
pip install deepspeed
```

---

## File Summary

| Directory | Files | Purpose |
|-----------|-------|---------|
| `config/` | 4 | Model, dataset, training configuration |
| `model/vlm/` | 3 | Vision encoders and projectors |
| `model/action_head/` | 4 | Action prediction heads |
| `model/sensor/` | 4 | Sensor encoders (LiDAR, Radar, IMU) |
| `model/vla/` | 5 | VLA model implementations |
| `model/fusion/` | 2 | Multi-modal fusion |
| `model/temporal/` | 4 | Temporal encoding and memory |
| `model/world_model/` | 4 | World modeling components |
| `model/safety/` | 4 | Safety constraints |
| `model/embodiment/` | 3 | Embodiment-specific models |
| `train/pretrain/` | 4 | VLM pretraining |
| `train/finetune/` | 3 | Supervised fine-tuning |
| `train/online_rl/` | 5 | Online RL trainers |
| `train/offline_rl/` | 6 | Offline RL trainers |
| `train/il/` | 5 | Imitation learning |
| `train/world_model/` | 2 | World model training |
| `train/embodiment/` | 3 | Embodiment training |
| `train/datasets/` | 6 | Dataset loaders |
| `eval/` | 4 | Evaluation metrics and benchmarks |
| `integration/` | 4 | ROS, simulators, experiment tracking |
| `scripts/` | 15 | SLURM training scripts |
| **Total** | **~83** | Complete VLA training framework |

---

## References

- [OpenVLA](https://openvla.github.io/) - Open-source VLA model
- [RT-2](https://robotics-transformer2.github.io/) - Vision-Language-Action model from Google
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace robot learning library
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - Cross-embodiment dataset
- [D4RL](https://github.com/Farama-Foundation/D4RL) - Offline RL benchmark datasets
- [Dreamer](https://danijar.com/project/dreamer/) - World model-based RL
- [Decision Transformer](https://arxiv.org/abs/2106.01345) - Offline RL as sequence modeling

---

## License

MIT License
