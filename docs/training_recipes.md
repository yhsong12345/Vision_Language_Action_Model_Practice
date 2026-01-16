# Training Recipes

Task-specific training configurations and best practices.

## Table of Contents

1. [Robot Manipulation](#robot-manipulation)
2. [RGB-D Manipulation](#rgb-d-manipulation)
3. [Autonomous Driving](#autonomous-driving)
4. [Humanoid Control](#humanoid-control)
5. [Simulation Benchmarks](#simulation-benchmarks)
6. [Real Robot Deployment](#real-robot-deployment)

---

## Robot Manipulation

### Recipe 1: PushT (Simple 2D Manipulation)

**Task**: Push T-shaped block to target location

```python
from model import create_vla_model
from train.il import BehavioralCloning
from train.datasets import LeRobotDataset
from config import ILConfig

# Dataset
dataset = LeRobotDataset(
    repo_id="lerobot/pusht",
    split="train",
    delta_timestamps={
        "observation.image": [0],
        "action": [0],
    },
)

# Model (lightweight for 2D task)
model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-0.5b",
    action_dim=2,  # [x, y]
    action_chunk_size=1,
    freeze_vision=True,
    freeze_llm=True,
)

# Training config
config = ILConfig(
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=100,
    gradient_accumulation_steps=1,
)

# Train
trainer = BehavioralCloning(model, config)
trainer.train(dataset)
```

**Expected Results**:
- Training loss: < 0.01
- Success rate: > 90%
- Training time: ~1 hour on RTX 3080

---

### Recipe 2: ALOHA Bimanual Manipulation

**Task**: Dual-arm tasks (e.g., transfer cube, fold cloth)

```python
from model import create_vla_model
from train.il import BehavioralCloning
from train.datasets import ALOHADataset
from model.action_head import DiffusionActionHead

# Dataset
dataset = ALOHADataset(
    repo_id="lerobot/aloha_transfer_cube_human",
    split="train",
)

# Model (diffusion for precise bimanual coordination)
model = create_vla_model(
    vision_encoder="siglip-large",
    llm="qwen2-1.5b",
    action_dim=14,  # 7 DoF per arm
    action_head_type="diffusion",
    action_chunk_size=16,  # Temporal consistency
    diffusion_steps=100,
)

# Training config
config = ILConfig(
    learning_rate=1e-4,
    batch_size=8,
    gradient_accumulation_steps=8,  # Effective batch 64
    num_epochs=200,
    use_lora=True,
    lora_r=32,
    mixed_precision="bf16",
)

trainer = BehavioralCloning(model, config)
trainer.train(dataset)
```

**Expected Results**:
- Success rate: > 80%
- Training time: ~8 hours on RTX 4090

---

### Recipe 3: xArm Pick and Place

**Task**: Pick objects and place in target locations

```python
from model import create_vla_model
from train.il import DAgger
from train.datasets import XArmDataset

# Dataset (initial BC data)
dataset = XArmDataset(
    repo_id="lerobot/xarm_pick_place",
    split="train",
)

# Model
model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    action_dim=7,  # 6 DoF + gripper
    action_head_type="gaussian",  # For DAgger
)

# DAgger training (with expert corrections)
config = ILConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=50,
    dagger_iterations=10,
    dagger_beta_decay=0.9,  # Expert mixing ratio decay
)

trainer = DAgger(
    model=model,
    env=xarm_env,
    expert_policy=teleop_expert,  # Human teleoperation
    config=config,
)

for iteration in range(10):
    print(f"DAgger iteration {iteration + 1}")
    trainer.collect_and_train(num_episodes=50)
```

---

## RGB-D Manipulation

### Recipe 3.5: RGB-D Grasping (GraspNet)

**Task**: 6-DoF grasp pose prediction with depth camera

```python
from model.vla import MultiSensorVLA
from model.sensor import DepthEncoder
from train.il import BehavioralCloning
from train.finetune.dataset import RobotDataset
from config import MultiSensorVLAConfig, DatasetConfig

# Dataset with depth
dataset_config = DatasetConfig.rgbd_manipulation()
dataset = RobotDataset(
    dataset_name="graspnet",
    image_processor=image_processor,
    tokenizer=tokenizer,
    use_depth=True,
    depth_size=224,
)

# Model configuration
model_config = MultiSensorVLAConfig.rgbd_manipulation()

# Model with depth encoder
model = MultiSensorVLA(
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",
    action_dim=7,  # 6 DoF + gripper
    use_depth=True,
    use_lidar=False,
    use_radar=False,
    use_imu=False,
    freeze_vision=True,
    freeze_llm=True,
)

# Training config
config = ILConfig(
    learning_rate=1e-4,
    batch_size=16,
    gradient_accumulation_steps=4,
    num_epochs=100,
    use_lora=True,
    lora_r=32,
    mixed_precision="bf16",
)

trainer = BehavioralCloning(model, config)
trainer.train(dataset)
```

**Expected Results**:
- Grasp success rate: > 85%
- Training time: ~6 hours on RTX 4090

---

### Recipe 3.6: NYU Depth Indoor Navigation

**Task**: Indoor navigation with RGB-D

```python
from model.vla import MultiSensorVLA
from config import DatasetConfig

# Dataset with depth
dataset_config = DatasetConfig.with_depth(
    dataset_name="nyu_depth_v2",
    depth_clip_range=[0.0, 10.0],  # Indoor depth range
)

# Model with depth for navigation
model = MultiSensorVLA(
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",
    action_dim=2,  # [linear_vel, angular_vel]
    use_depth=True,
    use_imu=True,  # Also use IMU for stability
    freeze_vision=True,
    freeze_llm=True,
)
```

---

## Autonomous Driving

### Recipe 4: CARLA Urban Driving

**Task**: Navigate urban environments with traffic

```python
from model.embodiment import DrivingVLA, BEVEncoder
from train.embodiment import DrivingVLATrainer
from train.datasets import CARLADataset
from config import DrivingTrainConfig

# Multi-sensor dataset
dataset = CARLADataset(
    data_root="/path/to/carla_data",
    cameras=["front", "front_left", "front_right"],
    use_lidar=True,
    use_radar=False,
    trajectory_length=20,  # Future waypoints
)

# Model
model = DrivingVLA(
    vlm_backbone="Qwen/Qwen2.5-VL-3B",
    num_cameras=3,
    bev_size=(200, 200),
    bev_resolution=0.5,  # meters per pixel
    trajectory_length=20,
    action_dim=3,  # [steering, throttle, brake]
    use_lidar=True,
)

# Training config
config = DrivingTrainConfig(
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_epochs=100,

    # Loss weights
    trajectory_loss_weight=1.0,
    control_loss_weight=0.5,
    collision_loss_weight=2.0,  # Penalize collisions

    # Safety
    max_steering_rate=0.5,  # rad/s
    max_acceleration=3.0,    # m/s^2
)

trainer = DrivingVLATrainer(model, dataset, config)
trainer.train()
```

**Expected Results**:
- Route completion: > 85%
- Collision rate: < 5%
- Training time: ~24 hours on 4x A100

---

### Recipe 5: nuScenes Prediction

**Task**: Trajectory prediction on nuScenes

```python
from model.embodiment import DrivingVLA
from train.datasets import NuScenesDataset

# Dataset with all sensors
dataset = NuScenesDataset(
    data_root="/path/to/nuscenes",
    version="v1.0-trainval",
    cameras=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
    use_lidar=True,
    use_radar=True,
    prediction_horizon=6.0,  # seconds
)

# Model with full sensor suite
model = DrivingVLA(
    vlm_backbone="Qwen/Qwen2.5-VL-7B",
    num_cameras=6,
    bev_size=(400, 400),
    bev_resolution=0.25,
    use_lidar=True,
    use_radar=True,
    lidar_encoder_type="pointtransformer",
    radar_encoder_type="range_doppler",
)

# Config for prediction task
config = DrivingTrainConfig(
    batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    num_epochs=50,
    prediction_mode=True,  # No control, just prediction
    num_modes=5,  # Multi-modal predictions
)
```

---

## Humanoid Control

### Recipe 6: MuJoCo Humanoid Walking

**Task**: Bipedal locomotion

```python
from model.embodiment import HumanoidVLA
from train.embodiment import HumanoidVLATrainer
from train.online_rl import PPOTrainer
from config import HumanoidTrainConfig
import gymnasium as gym

# Environment
env = gym.make("Humanoid-v4")

# Model
model = HumanoidVLA(
    vlm_backbone="Qwen/Qwen2.5-VL-1.5B",
    num_joints=17,  # MuJoCo Humanoid joints
    hidden_dim=512,
    action_head_type="gaussian",  # For PPO
)

# Two-phase training

# Phase 1: Behavioral Cloning from motion capture
bc_config = HumanoidTrainConfig(
    phase="bc",
    learning_rate=1e-4,
    batch_size=256,
    num_epochs=100,
)

bc_trainer = HumanoidVLATrainer(model, mocap_dataset, bc_config)
bc_trainer.train()

# Phase 2: Online RL fine-tuning
rl_config = HumanoidTrainConfig(
    phase="rl",
    algorithm="ppo",
    total_timesteps=10_000_000,
    learning_rate=3e-4,
    ppo_clip_range=0.2,
    entropy_coef=0.01,
    locomotion_reward_weight=1.0,
    stability_reward_weight=0.5,
    energy_penalty_weight=0.1,
)

rl_trainer = PPOTrainer(model, env, rl_config)
rl_trainer.train()
```

**Expected Results**:
- Walking speed: > 1.5 m/s
- Fall rate: < 10%
- Training time: ~48 hours on 8x A100

---

### Recipe 7: Whole-Body Manipulation

**Task**: Humanoid picking up objects

```python
from model.embodiment import HumanoidVLA, WholeBodyController
from train.il import GAIL

# Model with whole-body control
model = HumanoidVLA(
    vlm_backbone="Qwen/Qwen2.5-VL-3B",
    num_joints=32,  # Full body
    num_body_parts=15,
    hidden_dim=768,
    use_whole_body_ik=True,
)

# GAIL training (adversarial imitation)
config = HumanoidTrainConfig(
    algorithm="gail",
    discriminator_lr=1e-4,
    generator_lr=3e-4,
    num_epochs=1000,
    expert_data_path="./data/human_demos.pkl",
)

trainer = GAIL(model, env, expert_demos, config)
trainer.train()
```

---

## Simulation Benchmarks

### Recipe 8: D4RL Offline RL

**Task**: Offline RL benchmarks

```python
from train.offline_rl import IQLTrainer, CQLTrainer, DecisionTransformer
from train.datasets import D4RLDataset
from config import OfflineRLConfig

# Dataset
datasets = {
    "hopper-medium-v2": D4RLDataset("hopper-medium-v2"),
    "hopper-medium-expert-v2": D4RLDataset("hopper-medium-expert-v2"),
    "walker2d-medium-expert-v2": D4RLDataset("walker2d-medium-expert-v2"),
}

# IQL (recommended for most cases)
iql_config = OfflineRLConfig(
    algorithm="iql",
    expectile=0.7,
    temperature=3.0,
    learning_rate=3e-4,
    batch_size=256,
    num_epochs=1000,
)

for name, dataset in datasets.items():
    print(f"Training IQL on {name}")
    trainer = IQLTrainer(model, dataset, iql_config)
    trainer.train()

    # Evaluate
    results = trainer.evaluate(num_episodes=100)
    print(f"  Normalized score: {results['normalized_score']:.2f}")
```

**Expected Scores** (IQL):

| Dataset | Score |
|---------|-------|
| hopper-medium-v2 | 66.3 |
| hopper-medium-expert-v2 | 91.5 |
| walker2d-medium-expert-v2 | 109.6 |

---

### Recipe 9: MetaWorld Multi-Task

**Task**: 50 manipulation tasks

```python
from train.il import BehavioralCloning
from train.datasets import MetaWorldDataset
from model import create_vla_model

# Multi-task dataset
dataset = MetaWorldDataset(
    tasks="all",  # All 50 tasks
    demos_per_task=100,
)

# Language-conditioned model
model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    action_dim=4,  # End-effector control
    use_language=True,
)

# Multi-task training
config = ILConfig(
    learning_rate=1e-4,
    batch_size=128,
    num_epochs=200,
    task_sampling="uniform",
)

trainer = BehavioralCloning(model, config)
trainer.train(dataset)

# Evaluate on held-out tasks
results = trainer.evaluate(
    tasks=["drawer-open", "reach", "push"],  # Held-out
    num_episodes=50,
)
```

---

## Real Robot Deployment

### Recipe 10: Franka Panda Deployment

**Task**: Deploy to real Franka robot

```python
from model import create_vla_model
from integration import ROSBridge, SafetyShield
from model.safety import RuleChecker, ConstraintHandler

# Load trained model
model = create_vla_model.from_pretrained("./checkpoints/franka_vla.pt")
model.eval()

# Safety layer
safety_shield = SafetyShield(
    action_dim=7,
    max_velocity=0.5,  # m/s
    max_acceleration=2.0,  # m/s^2
    workspace_bounds=[
        [-0.5, 0.5],  # x
        [-0.5, 0.5],  # y
        [0.1, 0.8],   # z (above table)
    ],
)

rule_checker = RuleChecker(
    collision_mesh_path="./meshes/workspace.stl",
    joint_limits=FRANKA_JOINT_LIMITS,
)

# ROS integration
ros = ROSBridge(
    node_name="vla_controller",
    image_topic="/camera/color/image_raw",
    joint_state_topic="/franka/joint_states",
    command_topic="/franka/joint_commands",
)

# Control loop
@ros.on_observation
def control_step(observation):
    # Get image and instruction
    image = observation["image"]
    instruction = "Pick up the red cube"

    # Predict action
    with torch.no_grad():
        action = model.get_action(image, instruction)

    # Apply safety
    if not rule_checker.is_safe(action, observation["joint_state"]):
        print("Unsafe action detected, applying correction")
        action = safety_shield.filter(action, observation)

    return action

# Run at 30 Hz
ros.run(control_rate=30)
```

---

### Recipe 11: Low-Latency Deployment

**Task**: Real-time control with <50ms latency

```python
import torch
from model import create_vla_model

# Optimize model
model = create_vla_model.from_pretrained("./checkpoints/vla.pt")

# TorchScript compilation
scripted_model = torch.jit.script(model)
scripted_model = torch.jit.freeze(scripted_model)

# Warm up (first inference is slow)
dummy_input = torch.randn(1, 3, 224, 224).cuda()
for _ in range(10):
    _ = scripted_model(dummy_input)

# Benchmark
import time
latencies = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        _ = scripted_model(dummy_input)
    torch.cuda.synchronize()
    latencies.append(time.perf_counter() - start)

print(f"Mean latency: {sum(latencies)/len(latencies)*1000:.2f}ms")
print(f"99th percentile: {sorted(latencies)[99]*1000:.2f}ms")
```

**Optimization Tips**:

| Technique | Latency Reduction |
|-----------|------------------|
| TorchScript | ~20% |
| Mixed Precision | ~30% |
| Flash Attention | ~40% |
| TensorRT | ~60% |
| Smaller Model | Variable |

---

## Hyperparameter Guidelines

### Learning Rate

| Model Size | LR (BC) | LR (RL) |
|------------|---------|---------|
| Small (<500M) | 1e-4 | 3e-4 |
| Medium (1-3B) | 5e-5 | 1e-4 |
| Large (7B+) | 1e-5 | 3e-5 |

### Batch Size

| Task | Min Batch | Recommended |
|------|-----------|-------------|
| Simple manipulation | 32 | 128 |
| Complex manipulation | 16 | 64 |
| Driving | 4 | 16 |
| Humanoid | 64 | 256 |

### Action Chunking

| Task Type | Chunk Size |
|-----------|------------|
| Point-to-point | 1 |
| Continuous motion | 8-16 |
| Precise manipulation | 16-32 |
| Long-horizon | 32-64 |

---

## Troubleshooting

### Training Diverges

```python
# Reduce learning rate
config.learning_rate = 1e-5

# Add gradient clipping
config.max_grad_norm = 1.0

# Use warmup
config.warmup_steps = 1000
```

### Poor Generalization

```python
# Add data augmentation
config.use_augmentation = True
config.augmentation_types = ["color_jitter", "random_crop", "gaussian_noise"]

# Use dropout
config.dropout = 0.1

# Increase diversity
config.action_noise = 0.01
```

### Out of Memory

```python
# Reduce batch size + increase accumulation
config.batch_size = 4
config.gradient_accumulation_steps = 16

# Enable checkpointing
config.gradient_checkpointing = True

# Use smaller model
config.llm = "qwen2-0.5b"
```

---

## Datasets Used for Each Training Step

| Training Task | Dataset | Public Source | Description |
|---------------|---------|---------------|-------------|
| **Recipe 1: PushT** | lerobot/pusht | [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) | 2D manipulation task demonstrations |
| **Recipe 2: ALOHA Bimanual** | lerobot/aloha_transfer_cube_human | [lerobot/aloha_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_transfer_cube_human) | Dual-arm bimanual manipulation |
| **Recipe 3: xArm Pick and Place** | lerobot/xarm_pick_place | [lerobot on HuggingFace](https://huggingface.co/lerobot) | Single-arm pick and place |
| **Recipe 3.5: RGB-D Grasping** | GraspNet-1Billion | [graspnet.net](https://graspnet.net/) | 88 objects, 190 scenes, 1.1B grasps |
| **Recipe 3.6: Indoor Navigation** | NYU Depth v2 | [sayakpaul/nyu_depth_v2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2) | Indoor RGB-D navigation |
| **Recipe 4: CARLA Driving** | CARLA Autopilot | [immanuelpeter/carla-autopilot-multimodal-dataset](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset) | Urban driving with multi-camera + LiDAR |
| **Recipe 5: nuScenes Prediction** | nuScenes | [OpenDriveLab/DriveLM](https://huggingface.co/datasets/OpenDriveLab/DriveLM) | 1000 scenes, 6 cameras, LiDAR, Radar |
| **Recipe 6: MuJoCo Humanoid** | D4RL MuJoCo | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) | Ant, Humanoid, HalfCheetah locomotion |
| **Recipe 6: MuJoCo Humanoid** | Motion capture (AMASS) | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | BC initialization from MoCap |
| **Recipe 7: Whole-Body Manipulation** | ALOHA | [tonyzhaozh/aloha](https://huggingface.co/datasets/tonyzhaozh/aloha) | GAIL adversarial imitation |
| **Recipe 8: D4RL Offline RL** | D4RL | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) | hopper-medium-v2, walker2d, etc. |
| **Recipe 9: MetaWorld Multi-Task** | MetaWorld MT50 | [lerobot/metaworld_mt50](https://huggingface.co/datasets/lerobot/metaworld_mt50) | 50 manipulation tasks, 2500 episodes |

---

## Next Steps

- See [Usage Guide](usage.md) for basic usage
- See [Architecture](architecture.md) for model details
- Check [examples/](../examples/) for runnable scripts
