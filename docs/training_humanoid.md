# Training VLA for Humanoid Robots

This comprehensive guide covers the complete training process for Vision-Language-Action models designed for humanoid robot control, including whole-body control, bipedal locomotion, manipulation, and human-robot interaction.

## Table of Contents

1. [Overview](#overview)
2. [Humanoid Architecture](#humanoid-architecture)
3. [Action Space Design](#action-space-design)
4. [Data Collection](#data-collection)
5. [Stage 1: VLM Foundation for Humanoids](#stage-1-vlm-foundation-for-humanoids)
6. [Stage 2: Motion Primitive Learning](#stage-2-motion-primitive-learning)
7. [Stage 3: Locomotion Training](#stage-3-locomotion-training)
8. [Stage 4: Manipulation Training](#stage-4-manipulation-training)
9. [Stage 5: Whole-Body Control](#stage-5-whole-body-control)
10. [Stage 6: Policy Improvement](#stage-6-policy-improvement)
11. [Stage 7: Human-Robot Interaction](#stage-7-human-robot-interaction)
12. [Simulation Environments](#simulation-environments)
13. [Real Robot Deployment](#real-robot-deployment)
14. [Advanced Topics](#advanced-topics)

---

## Overview

### Humanoid VLA Pipeline

```
+=======================================================================================+
|                        HUMANOID VLA TRAINING PIPELINE                                  |
+=======================================================================================+
|                                                                                        |
|  INPUT MODALITIES                                                                      |
|  +-----------------------------------------------------------------------------------+ |
|  |  RGB Cameras (head/body)  |  Depth Sensors  |  Joint Encoders  |  IMU  |  F/T    | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  PROPRIOCEPTION ENCODER                                                                |
|  +-----------------------------------------------------------------------------------+ |
|  |  Joint State Encoder  |  IMU Encoder  |  Contact Encoder  |  History Encoder     | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  VLM BACKBONE (Language-Conditioned)                                                   |
|  +-----------------------------------------------------------------------------------+ |
|  |  Vision-Language Model (Qwen2-VL / LLaMA-VL)                                      | |
|  |  - Processes: "Walk to the table and pick up the cup"                             | |
|  |  - Outputs: Task-conditioned features for control                                 | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  HIERARCHICAL ACTION HEADS                                                             |
|  +-----------------------------------------------------------------------------------+ |
|  |  High-Level:  Task Planner -> Skill Selector                                      | |
|  |  Mid-Level:   Motion Primitives -> Trajectory Generator                           | |
|  |  Low-Level:   Joint PD Controller -> Torque Commands                              | |
|  +-----------------------------------------------------------------------------------+ |
|                                                                                        |
|  OUTPUT: 32+ DoF Joint Commands (position/velocity/torque)                             |
+=======================================================================================+
```

### Key Challenges for Humanoid VLA

| Challenge | Description | Solution |
|-----------|-------------|----------|
| High-Dimensional Action Space | 32+ DoF joints | Hierarchical control, skill primitives |
| Balance and Stability | Bipedal locomotion | RL with stability rewards, model-based |
| Contact-Rich Manipulation | Grasping, object interaction | Force/torque feedback, compliance |
| Whole-Body Coordination | Locomotion + manipulation | Multi-task learning, attention |
| Real-Time Control | 200-1000 Hz control loop | Efficient inference, low-level PD |
| Sim-to-Real Gap | Physics differences | Domain randomization, system ID |

---

## Humanoid Architecture

### HumanoidVLA Model Architecture

```python
from model.embodiment.humanoid import HumanoidVLA, HumanoidVLAConfig

@dataclass
class HumanoidVLAConfig:
    # Vision-Language Model
    vlm_backbone: str = "Qwen/Qwen2-1.5B-Instruct"
    vision_encoder: str = "siglip-base"

    # Robot Configuration
    num_joints: int = 32  # Typical humanoid (e.g., Boston Dynamics Atlas, Tesla Optimus)
    joint_dim: int = 12   # pos, vel, torque per joint
    ee_dim: int = 12      # 2 hands: position (3) + orientation (3) each

    # Body Parts
    body_parts: List[str] = field(default_factory=lambda: [
        "head", "torso",
        "left_shoulder", "left_elbow", "left_wrist", "left_hand",
        "right_shoulder", "right_elbow", "right_wrist", "right_hand",
        "left_hip", "left_knee", "left_ankle", "left_foot",
        "right_hip", "right_knee", "right_ankle", "right_foot",
    ])

    # Proprioception
    use_imu: bool = True
    use_contact_sensors: bool = True
    use_force_torque: bool = True

    # Control
    control_frequency: float = 50.0  # Hz (high-level)
    motor_frequency: float = 1000.0  # Hz (low-level PD)
    action_type: str = "position"     # position, velocity, torque

    # Hierarchical Control
    use_hierarchical: bool = True
    num_skills: int = 16              # Motion primitives
    skill_duration: float = 1.0       # seconds

    # History and Temporal
    history_length: int = 10          # Past observations
    prediction_horizon: int = 5       # Future actions

    # Model Architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
```

### Model Implementation

```python
from model.embodiment import HumanoidVLA

class HumanoidVLA(nn.Module):
    """
    Vision-Language-Action model for humanoid robots.

    Architecture:
    1. Vision encoder: Process camera images
    2. Proprioception encoder: Process joint/IMU/contact
    3. VLM backbone: Fuse vision, proprioception, language
    4. Hierarchical action heads: Skill selection + motion generation
    """

    def __init__(self, config: HumanoidVLAConfig):
        super().__init__()
        self.config = config

        # Vision encoder (for head cameras)
        self.vision_encoder = VisionEncoder(
            model_name=config.vision_encoder,
            output_dim=config.hidden_dim,
        )

        # Proprioception encoder
        self.proprio_encoder = ProprioceptionEncoder(
            joint_dim=config.joint_dim * config.num_joints,
            imu_dim=6 if config.use_imu else 0,
            contact_dim=4 if config.use_contact_sensors else 0,
            ft_dim=12 if config.use_force_torque else 0,
            output_dim=config.hidden_dim,
        )

        # History encoder (temporal context)
        self.history_encoder = HistoryEncoder(
            input_dim=config.hidden_dim,
            context_dim=config.hidden_dim,
            max_len=config.history_length,
        )

        # VLM backbone
        self.vlm = VLMModel(
            llm_model_name=config.vlm_backbone,
            vision_dim=config.hidden_dim,
        )

        # Hierarchical action heads
        if config.use_hierarchical:
            self.skill_selector = SkillSelector(
                input_dim=config.hidden_dim,
                num_skills=config.num_skills,
            )
            self.motion_generator = MotionGenerator(
                input_dim=config.hidden_dim,
                num_joints=config.num_joints,
                horizon=config.prediction_horizon,
            )
        else:
            self.action_head = GaussianMLPActionHead(
                input_dim=config.hidden_dim,
                action_dim=config.num_joints,
            )

        # Low-level controller (PD)
        self.pd_controller = PDController(
            num_joints=config.num_joints,
            kp=config.default_kp,
            kd=config.default_kd,
        )

    def forward(
        self,
        images: torch.Tensor,          # (B, C, H, W) head camera
        proprioception: Dict[str, torch.Tensor],  # joint states, IMU, etc.
        instruction: str,              # Natural language instruction
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for humanoid VLA.

        Returns:
            - skill_probs: Probability over motion primitives
            - joint_targets: Target joint positions/velocities
            - ee_targets: End-effector targets (optional)
        """
        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode proprioception
        proprio_features = self.proprio_encoder(proprioception)

        # Encode history
        if history is not None:
            history_features = self.history_encoder(history)
        else:
            history_features = torch.zeros_like(proprio_features)

        # Combine features
        combined_features = torch.cat([
            vision_features,
            proprio_features,
            history_features,
        ], dim=-1)

        # Process through VLM with language instruction
        vlm_features = self.vlm(
            visual_features=combined_features,
            instruction=instruction,
        )

        # Hierarchical action generation
        if self.config.use_hierarchical:
            # Select skill/motion primitive
            skill_probs = self.skill_selector(vlm_features)

            # Generate motion trajectory
            joint_targets = self.motion_generator(
                vlm_features,
                skill_probs,
            )
        else:
            # Direct action prediction
            joint_targets, _ = self.action_head(vlm_features)

        return {
            "skill_probs": skill_probs if self.config.use_hierarchical else None,
            "joint_targets": joint_targets,
            "features": vlm_features,
        }
```

### Proprioception Encoder

```python
class ProprioceptionEncoder(nn.Module):
    """
    Encodes robot proprioceptive state.

    Inputs:
    - Joint positions, velocities, torques
    - IMU (accelerometer + gyroscope)
    - Contact sensors (feet, hands)
    - Force/torque sensors

    Output: Encoded proprioception features
    """

    def __init__(
        self,
        joint_dim: int,
        imu_dim: int = 6,
        contact_dim: int = 4,
        ft_dim: int = 12,
        output_dim: int = 256,
    ):
        super().__init__()

        total_dim = joint_dim + imu_dim + contact_dim + ft_dim

        # Joint encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # IMU encoder
        if imu_dim > 0:
            self.imu_encoder = nn.Sequential(
                nn.Linear(imu_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )

        # Contact encoder
        if contact_dim > 0:
            self.contact_encoder = nn.Sequential(
                nn.Linear(contact_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )

        # Force/torque encoder
        if ft_dim > 0:
            self.ft_encoder = nn.Sequential(
                nn.Linear(ft_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )

        # Fusion MLP
        fusion_dim = 128 + (64 if imu_dim > 0 else 0) + \
                     (32 if contact_dim > 0 else 0) + \
                     (64 if ft_dim > 0 else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, proprio: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Input dict keys:
        - joint_pos: (B, num_joints)
        - joint_vel: (B, num_joints)
        - joint_torque: (B, num_joints) optional
        - imu: (B, 6) accelerometer + gyroscope
        - contact: (B, 4) binary contact flags
        - ft: (B, 12) force/torque from wrists/ankles
        """
        # Concatenate joint info
        joint_info = torch.cat([
            proprio["joint_pos"],
            proprio["joint_vel"],
            proprio.get("joint_torque", torch.zeros_like(proprio["joint_pos"])),
        ], dim=-1)

        joint_feat = self.joint_encoder(joint_info)

        features = [joint_feat]

        if "imu" in proprio and hasattr(self, "imu_encoder"):
            features.append(self.imu_encoder(proprio["imu"]))

        if "contact" in proprio and hasattr(self, "contact_encoder"):
            features.append(self.contact_encoder(proprio["contact"]))

        if "ft" in proprio and hasattr(self, "ft_encoder"):
            features.append(self.ft_encoder(proprio["ft"]))

        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)
```

---

## Action Space Design

### Joint Configuration

```python
# Typical humanoid joint configuration (32 DoF example)

HUMANOID_JOINTS = {
    # Head (3 DoF)
    "head_pan": {"type": "revolute", "range": [-1.57, 1.57]},
    "head_tilt": {"type": "revolute", "range": [-0.78, 0.78]},
    "head_roll": {"type": "revolute", "range": [-0.52, 0.52]},

    # Torso (3 DoF)
    "torso_yaw": {"type": "revolute", "range": [-1.57, 1.57]},
    "torso_pitch": {"type": "revolute", "range": [-0.52, 0.52]},
    "torso_roll": {"type": "revolute", "range": [-0.26, 0.26]},

    # Left Arm (7 DoF)
    "l_shoulder_pitch": {"type": "revolute", "range": [-3.14, 3.14]},
    "l_shoulder_roll": {"type": "revolute", "range": [-1.57, 1.57]},
    "l_shoulder_yaw": {"type": "revolute", "range": [-3.14, 3.14]},
    "l_elbow": {"type": "revolute", "range": [0, 2.62]},
    "l_wrist_yaw": {"type": "revolute", "range": [-3.14, 3.14]},
    "l_wrist_pitch": {"type": "revolute", "range": [-1.57, 1.57]},
    "l_wrist_roll": {"type": "revolute", "range": [-3.14, 3.14]},

    # Right Arm (7 DoF) - mirrored
    "r_shoulder_pitch": {"type": "revolute", "range": [-3.14, 3.14]},
    # ... (similar to left arm)

    # Left Leg (6 DoF)
    "l_hip_yaw": {"type": "revolute", "range": [-0.52, 0.52]},
    "l_hip_roll": {"type": "revolute", "range": [-0.52, 0.52]},
    "l_hip_pitch": {"type": "revolute", "range": [-1.57, 0.78]},
    "l_knee": {"type": "revolute", "range": [0, 2.62]},
    "l_ankle_pitch": {"type": "revolute", "range": [-0.78, 0.78]},
    "l_ankle_roll": {"type": "revolute", "range": [-0.26, 0.26]},

    # Right Leg (6 DoF) - mirrored
    "r_hip_yaw": {"type": "revolute", "range": [-0.52, 0.52]},
    # ... (similar to left leg)
}

# Hand configuration (optional, per hand)
HAND_JOINTS = {
    # 5 fingers x 3 DoF = 15 DoF per hand
    "thumb_cmc": {"type": "revolute", "range": [0, 1.57]},
    "thumb_mcp": {"type": "revolute", "range": [0, 1.57]},
    "thumb_ip": {"type": "revolute", "range": [0, 1.57]},
    # ... (similar for other fingers)
}
```

### Action Representations

```python
class ActionRepresentation:
    """
    Different action representations for humanoid control.
    """

    @staticmethod
    def joint_position(target_pos: torch.Tensor) -> Dict:
        """
        Joint position control (simplest).
        PD controller tracks target positions.
        """
        return {"type": "position", "targets": target_pos}

    @staticmethod
    def joint_velocity(target_vel: torch.Tensor) -> Dict:
        """
        Joint velocity control.
        Good for smooth motion generation.
        """
        return {"type": "velocity", "targets": target_vel}

    @staticmethod
    def joint_torque(torques: torch.Tensor) -> Dict:
        """
        Direct torque control.
        Most flexible but hardest to learn.
        """
        return {"type": "torque", "targets": torques}

    @staticmethod
    def end_effector(
        left_hand_pose: torch.Tensor,
        right_hand_pose: torch.Tensor,
    ) -> Dict:
        """
        End-effector control with IK.
        Natural for manipulation tasks.
        """
        return {
            "type": "end_effector",
            "left_hand": left_hand_pose,   # (x, y, z, qw, qx, qy, qz)
            "right_hand": right_hand_pose,
        }

    @staticmethod
    def skill_primitive(
        skill_id: int,
        skill_params: torch.Tensor,
    ) -> Dict:
        """
        High-level skill selection with parameters.
        Used in hierarchical control.
        """
        return {
            "type": "skill",
            "skill_id": skill_id,
            "params": skill_params,
        }
```

---

## Data Collection

### Motion Capture Data

```python
class MoCapDataset(torch.utils.data.Dataset):
    """
    Motion capture dataset for humanoid training.

    Sources:
    - CMU Motion Capture Database
    - Human3.6M
    - AMASS (Archive of Motion Capture as Surface Shapes)
    - Custom teleoperation
    """

    def __init__(
        self,
        data_root: str,
        retarget_to_robot: bool = True,
        robot_urdf: str = None,
    ):
        self.data_root = data_root
        self.retarget_to_robot = retarget_to_robot

        if retarget_to_robot:
            self.retargeter = MoCapRetargeter(robot_urdf)

        # Load motion files
        self.motions = self._load_motions()

    def __getitem__(self, idx: int) -> Dict:
        motion = self.motions[idx]

        # Retarget human motion to robot kinematic
        if self.retarget_to_robot:
            motion = self.retargeter.retarget(motion)

        return {
            "joint_positions": motion["joint_pos"],      # (T, num_joints)
            "joint_velocities": motion["joint_vel"],     # (T, num_joints)
            "root_position": motion["root_pos"],         # (T, 3)
            "root_orientation": motion["root_quat"],     # (T, 4)
            "contact_labels": motion.get("contacts"),    # (T, 4) foot contacts
            "action_label": motion.get("label"),         # e.g., "walking", "reaching"
        }

    def _load_motions(self) -> List[Dict]:
        """Load and preprocess motion capture files."""
        motions = []

        for motion_file in glob.glob(f"{self.data_root}/**/*.bvh"):
            motion = self._load_bvh(motion_file)
            motions.append(motion)

        for motion_file in glob.glob(f"{self.data_root}/**/*.npz"):
            motion = np.load(motion_file)
            motions.append(dict(motion))

        return motions


class MoCapRetargeter:
    """
    Retarget human motion capture to robot kinematics.

    Handles:
    - Different skeleton structures
    - Different joint limits
    - Scaling differences
    """

    def __init__(self, robot_urdf: str):
        import pybullet as p

        self.robot = p.loadURDF(robot_urdf)
        self.joint_info = self._get_joint_info()

    def retarget(self, human_motion: Dict) -> Dict:
        """
        Retarget human motion to robot.

        Steps:
        1. Map joint names (human -> robot)
        2. Scale positions to robot dimensions
        3. Apply joint limits
        4. Compute inverse kinematics for end-effectors
        """
        robot_motion = {}

        # Map joints
        joint_mapping = self._get_joint_mapping()

        robot_joint_pos = []
        for t in range(len(human_motion["joint_pos"])):
            frame_pos = []
            for robot_joint, human_joint in joint_mapping.items():
                # Get human joint angle
                human_angle = human_motion["joint_pos"][t, human_joint]

                # Scale and clip to robot limits
                robot_angle = self._scale_angle(human_angle, robot_joint)
                robot_angle = np.clip(
                    robot_angle,
                    self.joint_info[robot_joint]["lower"],
                    self.joint_info[robot_joint]["upper"],
                )
                frame_pos.append(robot_angle)

            robot_joint_pos.append(frame_pos)

        robot_motion["joint_pos"] = np.array(robot_joint_pos)

        # Compute velocities
        robot_motion["joint_vel"] = np.gradient(
            robot_motion["joint_pos"],
            axis=0,
        ) * self.fps

        return robot_motion
```

### Teleoperation Data Collection

```python
class TeleoperationCollector:
    """
    Collect humanoid demonstration data via teleoperation.

    Methods:
    1. VR teleoperation (hand tracking)
    2. Motion capture suit
    3. Leader-follower (with another robot)
    4. Kinesthetic teaching
    """

    def __init__(
        self,
        robot: HumanoidRobot,
        teleop_mode: str = "vr",
    ):
        self.robot = robot

        if teleop_mode == "vr":
            self.teleop = VRTeleoperation()
        elif teleop_mode == "mocap_suit":
            self.teleop = MoCapSuitTeleoperation()
        elif teleop_mode == "leader_follower":
            self.teleop = LeaderFollowerTeleoperation()

    def collect_episode(
        self,
        task_instruction: str,
        max_length: int = 1000,
    ) -> Dict:
        """Collect a single teleoperation episode."""

        episode = {
            "observations": [],
            "actions": [],
            "proprioception": [],
            "instruction": task_instruction,
        }

        self.robot.reset()

        for step in range(max_length):
            # Get teleoperator commands
            teleop_command = self.teleop.get_command()

            # Get robot observation
            obs = self.robot.get_observation()

            # Execute command
            action = self._teleop_to_action(teleop_command)
            self.robot.step(action)

            # Store
            episode["observations"].append(obs)
            episode["actions"].append(action)
            episode["proprioception"].append(self.robot.get_proprioception())

            # Check for completion
            if self.teleop.is_done():
                break

        return episode

    def collect_dataset(
        self,
        tasks: List[str],
        episodes_per_task: int = 50,
        output_dir: str = "./teleop_data",
    ):
        """Collect large-scale teleoperation dataset."""

        os.makedirs(output_dir, exist_ok=True)

        for task in tasks:
            print(f"Collecting data for task: {task}")

            for ep in range(episodes_per_task):
                episode = self.collect_episode(task)

                # Save episode
                save_path = os.path.join(
                    output_dir,
                    f"{task.replace(' ', '_')}_{ep:03d}.pkl"
                )
                with open(save_path, "wb") as f:
                    pickle.dump(episode, f)

        print(f"Collected dataset to {output_dir}")
```

---

## Stage 1: VLM Foundation for Humanoids

### Humanoid-Specific VLM Pretraining

```python
from train.pretrain import HumanoidVLMPretrainer

class HumanoidVLMPretrainer:
    """
    Pretrain VLM on humanoid-specific visual instructions.

    Instruction types:
    1. Object manipulation: "Pick up the red cup"
    2. Navigation: "Walk to the door"
    3. Gesture understanding: "Wave hello"
    4. Human interaction: "Hand me the tool"
    5. Body awareness: "What is your left hand holding?"
    """

    def __init__(
        self,
        vlm_model: VLMModel,
        config: PretrainingConfig,
    ):
        self.model = vlm_model
        self.config = config

    def create_humanoid_instruction_dataset(self) -> Dataset:
        """Create instruction dataset for humanoid robots."""

        instructions = []

        # Object manipulation instructions
        manipulation_templates = [
            "Pick up the {object} from the {location}.",
            "Place the {object} on the {surface}.",
            "Hand the {object} to the person.",
            "Open the {container}.",
            "Pour water from the {container}.",
        ]

        # Navigation instructions
        navigation_templates = [
            "Walk to the {location}.",
            "Go around the {obstacle}.",
            "Climb the {structure}.",
            "Step over the {object}.",
            "Navigate to the {destination}.",
        ]

        # Gesture instructions
        gesture_templates = [
            "Wave your hand.",
            "Point to the {object}.",
            "Nod your head.",
            "Give a thumbs up.",
            "Shake hands with the person.",
        ]

        # Body awareness queries
        body_queries = [
            "What is in your left hand?",
            "Can you reach the {object}?",
            "What is your current posture?",
            "Where is the nearest obstacle?",
            "Is the path clear ahead?",
        ]

        # Generate diverse instructions
        for template_set in [manipulation_templates, navigation_templates,
                            gesture_templates, body_queries]:
            for template in template_set:
                # Fill in placeholders with various objects/locations
                for obj in ["cup", "ball", "tool", "book", "box"]:
                    for loc in ["table", "shelf", "floor", "counter"]:
                        instruction = template.format(
                            object=obj,
                            location=loc,
                            surface=loc,
                            container=obj,
                            obstacle=obj,
                            structure="stairs",
                            destination=loc,
                        )
                        instructions.append(instruction)

        return HumanoidInstructionDataset(instructions)

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 10,
    ):
        """Pretrain VLM on humanoid instructions."""

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Forward pass
                output = self.model(
                    images=batch["images"],
                    instruction=batch["instruction"],
                )

                # Language modeling loss
                loss = F.cross_entropy(
                    output["logits"],
                    batch["target_tokens"],
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        self.model.save_pretrained(self.config.output_dir)
```

---

## Stage 2: Motion Primitive Learning

### Learning Motion Primitives from MoCap

```python
class MotionPrimitiveTrainer:
    """
    Learn a library of motion primitives from motion capture data.

    Motion primitives:
    - Standing
    - Walking (forward, backward, sideways)
    - Turning (left, right)
    - Reaching (various directions)
    - Grasping
    - Lifting
    - Placing
    - Pushing
    - Pulling
    """

    def __init__(
        self,
        num_primitives: int = 16,
        primitive_length: int = 50,  # frames
        latent_dim: int = 64,
    ):
        self.num_primitives = num_primitives
        self.primitive_length = primitive_length

        # VAE for learning motion primitives
        self.vae = MotionVAE(
            input_dim=32 * 3,  # num_joints * (pos + vel + acc)
            latent_dim=latent_dim,
            hidden_dim=256,
        )

        # Primitive embeddings
        self.primitive_embeddings = nn.Embedding(num_primitives, latent_dim)

    def train_vae(
        self,
        mocap_dataset: MoCapDataset,
        num_epochs: int = 100,
    ):
        """
        Train VAE to learn motion primitive space.
        """
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in mocap_dataset:
                # Extract motion segments
                motion = batch["joint_positions"]  # (B, T, num_joints)

                # VAE forward
                recon, mu, logvar = self.vae(motion)

                # VAE loss
                recon_loss = F.mse_loss(recon, motion)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + 0.001 * kl_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}: Recon={recon_loss:.4f}, KL={kl_loss:.4f}")

    def cluster_primitives(
        self,
        mocap_dataset: MoCapDataset,
    ):
        """
        Cluster motions into discrete primitives using learned latents.
        """
        # Encode all motions
        latents = []
        for batch in mocap_dataset:
            with torch.no_grad():
                z = self.vae.encode(batch["joint_positions"])
                latents.append(z)

        latents = torch.cat(latents, dim=0).numpy()

        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_primitives)
        labels = kmeans.fit_predict(latents)

        # Store cluster centers as primitive embeddings
        self.primitive_embeddings.weight.data = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float32,
        )

        # Create primitive labels
        self.primitive_labels = {
            0: "stand",
            1: "walk_forward",
            2: "walk_backward",
            3: "turn_left",
            4: "turn_right",
            5: "reach_forward",
            6: "reach_up",
            7: "reach_down",
            8: "grasp",
            9: "release",
            10: "lift",
            11: "lower",
            12: "push",
            13: "pull",
            14: "wave",
            15: "point",
        }

        return labels

    def train_primitive_decoder(
        self,
        mocap_dataset: MoCapDataset,
        primitive_labels: np.ndarray,
    ):
        """
        Train decoder to generate motion from primitive + parameters.
        """
        # Primitive-conditioned decoder
        self.decoder = PrimitiveDecoder(
            num_primitives=self.num_primitives,
            primitive_dim=64,
            param_dim=16,  # Additional parameters (speed, amplitude, etc.)
            output_dim=32 * self.primitive_length,
        )

        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)

        for epoch in range(100):
            for batch_idx, batch in enumerate(mocap_dataset):
                motion = batch["joint_positions"]
                labels = torch.tensor(primitive_labels[batch_idx])

                # Get primitive embedding
                primitive_emb = self.primitive_embeddings(labels)

                # Generate motion from primitive
                generated = self.decoder(primitive_emb)

                # Reconstruction loss
                loss = F.mse_loss(generated, motion.view(len(motion), -1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def generate_primitive(
        self,
        primitive_id: int,
        params: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate motion from a primitive ID.
        """
        primitive_emb = self.primitive_embeddings(torch.tensor([primitive_id]))

        if params is not None:
            primitive_emb = torch.cat([primitive_emb, params], dim=-1)

        motion = self.decoder(primitive_emb)
        return motion.view(-1, self.primitive_length, 32)
```

---

## Stage 3: Locomotion Training

### Bipedal Locomotion with RL

```python
from train.online_rl import PPOTrainer

class LocomotionTrainer:
    """
    Train bipedal locomotion policy using reinforcement learning.

    Environment: MuJoCo / Isaac Gym / PyBullet
    Algorithm: PPO / SAC with specialized reward

    Locomotion behaviors:
    - Standing balance
    - Walking (various speeds)
    - Running
    - Turning
    - Stair climbing
    - Rough terrain
    """

    def __init__(
        self,
        env: HumanoidLocomotionEnv,
        model: HumanoidVLA,
    ):
        self.env = env
        self.model = model

        # Reward components
        self.reward_weights = {
            "forward_velocity": 1.0,
            "survival": 0.1,
            "energy_efficiency": -0.01,
            "upright_bonus": 0.5,
            "smooth_motion": -0.1,
            "foot_clearance": 0.2,
            "symmetry": 0.3,
        }

    def train_standing(
        self,
        num_steps: int = 1_000_000,
    ):
        """
        Phase 1: Learn to stand and balance.
        """
        # Simplified reward for standing
        standing_rewards = {
            "upright_bonus": 1.0,
            "survival": 0.5,
            "smooth_motion": -0.1,
            "feet_on_ground": 0.3,
        }

        self.env.set_rewards(standing_rewards)

        trainer = PPOTrainer(
            model=self.model,
            env=self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

        trainer.learn(num_steps)

        # Save standing policy
        torch.save(self.model.state_dict(), "./checkpoints/standing_policy.pt")

    def train_walking(
        self,
        num_steps: int = 5_000_000,
        curriculum: bool = True,
    ):
        """
        Phase 2: Learn to walk with curriculum learning.
        """
        # Load standing policy as initialization
        self.model.load_state_dict(torch.load("./checkpoints/standing_policy.pt"))

        if curriculum:
            # Curriculum stages
            stages = [
                {"target_velocity": 0.3, "steps": 1_000_000},
                {"target_velocity": 0.5, "steps": 1_000_000},
                {"target_velocity": 0.8, "steps": 1_000_000},
                {"target_velocity": 1.0, "steps": 1_000_000},
                {"target_velocity": 1.5, "steps": 1_000_000},
            ]

            for stage in stages:
                print(f"Training for velocity: {stage['target_velocity']} m/s")

                self.env.set_target_velocity(stage["target_velocity"])

                trainer = PPOTrainer(
                    model=self.model,
                    env=self.env,
                    learning_rate=3e-4,
                )

                trainer.learn(stage["steps"])

        else:
            # Direct training without curriculum
            self.env.set_rewards(self.reward_weights)

            trainer = PPOTrainer(
                model=self.model,
                env=self.env,
                learning_rate=3e-4,
            )

            trainer.learn(num_steps)

        torch.save(self.model.state_dict(), "./checkpoints/walking_policy.pt")

    def train_with_reference_motion(
        self,
        mocap_data: MoCapDataset,
        num_steps: int = 5_000_000,
    ):
        """
        Train locomotion using motion imitation (AMP/DeepMimic style).
        """
        # Reference motion discriminator
        discriminator = MotionDiscriminator(
            input_dim=32 * 6,  # joint pos + vel
            hidden_dim=256,
        )

        trainer = AMPTrainer(
            model=self.model,
            env=self.env,
            discriminator=discriminator,
            reference_motions=mocap_data,
            style_reward_weight=0.5,
            task_reward_weight=0.5,
        )

        trainer.learn(num_steps)


class HumanoidLocomotionEnv:
    """
    Humanoid locomotion environment with customizable rewards.
    """

    def __init__(
        self,
        robot_urdf: str,
        terrain: str = "flat",
    ):
        import mujoco
        self.model = mujoco.MjModel.from_xml_path(robot_urdf)
        self.data = mujoco.MjData(self.model)

        self.terrain = terrain
        self.target_velocity = 1.0  # m/s

    def step(self, action: np.ndarray) -> Tuple:
        """Execute action and return observation, reward, done, info."""

        # Apply action (joint position targets)
        self._apply_action(action)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        done = self._check_termination()

        info = {
            "velocity": self._get_velocity(),
            "height": self._get_height(),
            "contact": self._get_contact(),
        }

        return obs, reward, done, info

    def _compute_reward(self) -> float:
        """Compute locomotion reward."""
        reward = 0.0

        # Forward velocity reward
        velocity = self._get_velocity()
        velocity_reward = -np.abs(velocity[0] - self.target_velocity)
        reward += self.reward_weights["forward_velocity"] * velocity_reward

        # Survival reward
        reward += self.reward_weights["survival"]

        # Upright bonus
        height = self._get_height()
        upright_reward = height / self.nominal_height
        reward += self.reward_weights["upright_bonus"] * upright_reward

        # Energy efficiency (minimize torques)
        torques = self._get_torques()
        energy_penalty = np.sum(torques ** 2)
        reward += self.reward_weights["energy_efficiency"] * energy_penalty

        # Smooth motion (minimize jerk)
        if hasattr(self, "prev_action"):
            action_diff = np.abs(self.action - self.prev_action)
            smooth_penalty = np.sum(action_diff)
            reward += self.reward_weights["smooth_motion"] * smooth_penalty

        return reward

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Fall detection
        height = self._get_height()
        if height < 0.5 * self.nominal_height:
            return True

        # Excessive tilt
        orientation = self._get_orientation()
        tilt = np.abs(orientation[:2])  # roll, pitch
        if np.any(tilt > 1.0):  # radians
            return True

        return False
```

### Terrain Adaptation

```python
class TerrainAdaptationTrainer:
    """
    Train locomotion policy to adapt to various terrains.

    Terrains:
    - Flat ground
    - Stairs (up/down)
    - Slopes
    - Rough terrain
    - Stepping stones
    - Gaps
    """

    def __init__(
        self,
        base_policy_path: str,
    ):
        self.model = HumanoidVLA.from_pretrained(base_policy_path)

        # Terrain configurations
        self.terrains = {
            "flat": {"friction": 1.0, "roughness": 0.0},
            "stairs_up": {"step_height": 0.17, "step_depth": 0.28},
            "stairs_down": {"step_height": -0.17, "step_depth": 0.28},
            "slope_up": {"angle": 15},  # degrees
            "slope_down": {"angle": -15},
            "rough": {"roughness": 0.05},  # meters
            "stepping_stones": {"spacing": 0.4, "size": 0.3},
        }

    def train_domain_randomization(
        self,
        num_steps: int = 10_000_000,
    ):
        """
        Train with randomized terrain parameters.
        """
        # Create environment with terrain randomization
        env = HumanoidLocomotionEnv(
            terrain="random",
            randomize_terrain=True,
            terrain_params={
                "friction": (0.5, 1.5),
                "roughness": (0.0, 0.1),
                "slope": (-20, 20),
            },
        )

        trainer = PPOTrainer(
            model=self.model,
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)

    def train_terrain_specific(
        self,
        terrain: str,
        num_steps: int = 2_000_000,
    ):
        """
        Train on specific terrain type.
        """
        env = HumanoidLocomotionEnv(
            terrain=terrain,
            terrain_config=self.terrains[terrain],
        )

        trainer = PPOTrainer(
            model=self.model,
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)

    def train_adaptive_policy(
        self,
        num_steps: int = 10_000_000,
    ):
        """
        Train policy that adapts to terrain using exteroception.
        """
        # Add terrain encoder
        terrain_encoder = TerrainEncoder(
            depth_input_dim=64 * 64,  # depth image
            output_dim=64,
        )

        # Terrain-conditioned policy
        adaptive_model = TerrainAdaptivePolicy(
            base_model=self.model,
            terrain_encoder=terrain_encoder,
        )

        env = HumanoidLocomotionEnv(
            terrain="random",
            provide_depth=True,
        )

        trainer = PPOTrainer(
            model=adaptive_model,
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)
```

---

## Stage 4: Manipulation Training

### Reaching and Grasping

```python
class ManipulationTrainer:
    """
    Train humanoid manipulation skills.

    Skills:
    - Reaching
    - Grasping (various objects)
    - Picking up
    - Placing
    - Handover (to human)
    - Two-handed manipulation
    """

    def __init__(
        self,
        model: HumanoidVLA,
        env: HumanoidManipulationEnv,
    ):
        self.model = model
        self.env = env

    def train_reaching(
        self,
        dataset: ReachingDataset,
        num_epochs: int = 100,
    ):
        """
        Train reaching behavior from demonstrations.
        """
        trainer = BehavioralCloning(
            model=self.model,
            learning_rate=1e-4,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Target: end-effector position
                target_pos = batch["target_position"]
                current_proprio = batch["proprioception"]

                # Predict joint positions to reach target
                output = self.model(
                    images=batch["images"],
                    proprioception=current_proprio,
                    instruction=f"Reach the target at position {target_pos}",
                )

                # Loss: end-effector position error + joint smoothness
                ee_pos = self._forward_kinematics(output["joint_targets"])
                ee_loss = F.mse_loss(ee_pos, target_pos)

                # Joint smoothness
                if hasattr(batch, "prev_joints"):
                    smooth_loss = F.mse_loss(
                        output["joint_targets"],
                        batch["prev_joints"],
                    ) * 0.1
                else:
                    smooth_loss = 0.0

                loss = ee_loss + smooth_loss

                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

    def train_grasping(
        self,
        dataset: GraspingDataset,
        num_epochs: int = 100,
    ):
        """
        Train grasping behavior with grasp success reward.
        """
        # Two-stage training
        # Stage 1: Pre-grasp approach (BC)
        self._train_approach(dataset)

        # Stage 2: Grasp execution (RL with success reward)
        self._train_grasp_execution()

    def _train_approach(self, dataset: GraspingDataset):
        """Train approach behavior with BC."""
        trainer = BehavioralCloning(
            model=self.model,
            learning_rate=1e-4,
        )

        for batch in dataset:
            # Filter to pre-grasp phase
            pre_grasp = batch["phase"] == "approach"

            output = self.model(
                images=batch["images"][pre_grasp],
                proprioception=batch["proprioception"][pre_grasp],
                instruction="Approach the object for grasping",
            )

            loss = F.mse_loss(
                output["joint_targets"],
                batch["expert_joints"][pre_grasp],
            )

            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

    def _train_grasp_execution(self, num_steps: int = 1_000_000):
        """Train grasp execution with RL."""
        # Reward function for grasping
        reward_fn = GraspRewardFunction(
            success_bonus=10.0,
            approach_reward=0.1,
            gripper_reward=0.5,
        )

        self.env.set_reward_fn(reward_fn)

        trainer = SACTrainer(
            model=self.model,
            env=self.env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)

    def train_bimanual(
        self,
        dataset: BimanualDataset,
        num_epochs: int = 100,
    ):
        """
        Train two-handed manipulation tasks.
        """
        # Bimanual coordination module
        bimanual_coordinator = BimanualCoordinator(
            hidden_dim=256,
            num_joints_per_arm=7,
        )

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(bimanual_coordinator.parameters()),
            lr=1e-4,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Get features from VLM
                vlm_features = self.model.encode(
                    images=batch["images"],
                    instruction=batch["instruction"],
                )

                # Coordinate left and right arm
                left_targets, right_targets = bimanual_coordinator(
                    vlm_features,
                    batch["left_proprio"],
                    batch["right_proprio"],
                )

                # Loss: track expert trajectories
                loss = (
                    F.mse_loss(left_targets, batch["left_expert"]) +
                    F.mse_loss(right_targets, batch["right_expert"])
                )

                # Coordination loss (avoid collision)
                coord_loss = bimanual_coordinator.collision_avoidance_loss(
                    left_targets, right_targets
                )

                total_loss = loss + 0.1 * coord_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

---

## Stage 5: Whole-Body Control

### Loco-Manipulation Training

```python
class LocoManipulationTrainer:
    """
    Train combined locomotion and manipulation (loco-manipulation).

    Tasks requiring whole-body coordination:
    - Walking while carrying objects
    - Mobile manipulation
    - Dynamic reaching
    - Pushing/pulling heavy objects
    """

    def __init__(
        self,
        locomotion_policy_path: str,
        manipulation_policy_path: str,
    ):
        self.model = HumanoidVLA()

        # Initialize from pretrained policies
        self.model.locomotion_head.load_state_dict(
            torch.load(locomotion_policy_path)
        )
        self.model.manipulation_head.load_state_dict(
            torch.load(manipulation_policy_path)
        )

        # Coordination module
        self.coordinator = WholeBodyCoordinator(
            locomotion_dim=12,  # leg joints
            manipulation_dim=14,  # arm joints
            hidden_dim=256,
        )

    def train_walking_while_carrying(
        self,
        env: LocoManipEnv,
        num_steps: int = 5_000_000,
    ):
        """
        Train to walk while carrying objects of various weights.
        """
        # Reward components
        rewards = {
            "locomotion": 0.5,
            "object_stability": 0.3,
            "task_completion": 0.2,
        }

        # Curriculum: increasing object weight
        weights = [0.5, 1.0, 2.0, 5.0]  # kg

        for weight in weights:
            print(f"Training with {weight}kg object")

            env.set_object_weight(weight)

            trainer = PPOTrainer(
                model=self.model,
                env=env,
                learning_rate=3e-4,
            )

            trainer.learn(num_steps // len(weights))

    def train_mobile_manipulation(
        self,
        env: MobileManipEnv,
        num_steps: int = 10_000_000,
    ):
        """
        Train mobile manipulation tasks.
        """
        # Task phases
        phases = ["approach", "manipulate", "retreat"]

        # Multi-task reward
        reward_fn = MobileManipReward(
            approach_weight=0.3,
            manipulation_weight=0.5,
            retreat_weight=0.2,
        )

        env.set_reward_fn(reward_fn)

        trainer = PPOTrainer(
            model=self.model,
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)

    def train_whole_body_ik(
        self,
        dataset: WholeBodyIKDataset,
        num_epochs: int = 100,
    ):
        """
        Train whole-body inverse kinematics network.
        """
        # Whole-body IK network
        ik_net = WholeBodyIKNetwork(
            num_joints=32,
            ee_dim=12,  # 2 hands: pos + quat each
            com_dim=3,  # center of mass
        )

        optimizer = torch.optim.Adam(ik_net.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in dataset:
                # Inputs: target end-effector poses + COM
                target_ee = batch["target_ee"]
                target_com = batch["target_com"]

                # Predict joint angles
                joint_angles = ik_net(target_ee, target_com)

                # Forward kinematics to verify
                computed_ee = self._forward_kinematics_all(joint_angles)
                computed_com = self._compute_com(joint_angles)

                # Loss
                ee_loss = F.mse_loss(computed_ee, target_ee)
                com_loss = F.mse_loss(computed_com, target_com)

                # Joint limit loss
                limit_loss = self._joint_limit_loss(joint_angles)

                total_loss = ee_loss + 0.5 * com_loss + 0.1 * limit_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

### Hierarchical Whole-Body Control

```python
class HierarchicalWholeBodyController:
    """
    Hierarchical control for whole-body humanoid tasks.

    Hierarchy:
    1. Task level: Language instruction -> task parameters
    2. Skill level: Task parameters -> skill sequence
    3. Motion level: Skills -> joint trajectories
    4. Control level: Trajectories -> torque commands
    """

    def __init__(self, model: HumanoidVLA):
        self.model = model

        # Task planner (VLM-based)
        self.task_planner = TaskPlanner(
            vlm=model.vlm,
            num_tasks=20,
        )

        # Skill selector
        self.skill_selector = SkillSelector(
            num_skills=16,
            hidden_dim=256,
        )

        # Motion generator
        self.motion_generator = MotionGenerator(
            num_joints=32,
            horizon=50,
        )

        # Low-level controller
        self.pd_controller = PDController(
            num_joints=32,
            kp=100.0,
            kd=10.0,
        )

    def train_task_planner(
        self,
        dataset: TaskPlanningDataset,
        num_epochs: int = 50,
    ):
        """
        Train task planner to decompose instructions into subtasks.
        """
        optimizer = torch.optim.Adam(self.task_planner.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in dataset:
                # Input: image + instruction
                # Output: sequence of subtasks

                output = self.task_planner(
                    images=batch["images"],
                    instruction=batch["instruction"],
                )

                # Classification loss over subtask sequence
                loss = F.cross_entropy(
                    output["subtask_logits"],
                    batch["subtask_labels"],
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def train_skill_selector(
        self,
        dataset: SkillDataset,
        num_epochs: int = 100,
    ):
        """
        Train skill selection from subtask + state.
        """
        optimizer = torch.optim.Adam(self.skill_selector.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in dataset:
                # Input: subtask + proprioception
                # Output: skill ID + parameters

                skill_probs, skill_params = self.skill_selector(
                    subtask=batch["subtask"],
                    proprioception=batch["proprioception"],
                )

                # Classification loss
                cls_loss = F.cross_entropy(skill_probs, batch["skill_label"])

                # Parameter regression loss
                param_loss = F.mse_loss(skill_params, batch["skill_params"])

                loss = cls_loss + 0.5 * param_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def train_motion_generator(
        self,
        dataset: MotionDataset,
        num_epochs: int = 100,
    ):
        """
        Train motion generation from skills.
        """
        optimizer = torch.optim.Adam(self.motion_generator.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in dataset:
                # Input: skill embedding + initial state
                # Output: joint trajectory

                trajectory = self.motion_generator(
                    skill=batch["skill_embedding"],
                    initial_state=batch["initial_joints"],
                )

                # Trajectory loss
                traj_loss = F.mse_loss(trajectory, batch["expert_trajectory"])

                # Smoothness loss
                vel = trajectory[:, 1:] - trajectory[:, :-1]
                smooth_loss = torch.norm(vel[:, 1:] - vel[:, :-1], dim=-1).mean()

                loss = traj_loss + 0.1 * smooth_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def train_end_to_end(
        self,
        env: HumanoidEnv,
        num_steps: int = 10_000_000,
    ):
        """
        Fine-tune entire hierarchy end-to-end with RL.
        """
        # Combine all components
        params = (
            list(self.task_planner.parameters()) +
            list(self.skill_selector.parameters()) +
            list(self.motion_generator.parameters())
        )

        trainer = PPOTrainer(
            policy=self,
            env=env,
            learning_rate=1e-5,  # Small LR for fine-tuning
        )

        trainer.learn(num_steps)
```

---

## Stage 6: Policy Improvement

### Reinforcement Learning Fine-tuning

```python
class HumanoidRLTrainer:
    """
    RL training and fine-tuning for humanoid policies.

    Algorithms:
    - PPO: Stable, on-policy
    - SAC: Sample-efficient, off-policy
    - GRPO: Language model-compatible
    - AMP: Adversarial motion priors
    """

    def train_ppo(
        self,
        env: HumanoidEnv,
        num_steps: int = 10_000_000,
    ):
        """Train with PPO."""
        trainer = PPOTrainer(
            model=self.model,
            env=env,

            # PPO hyperparameters
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,

            # Humanoid-specific
            normalize_observations=True,
            normalize_rewards=True,
        )

        trainer.learn(num_steps)

    def train_sac(
        self,
        env: HumanoidEnv,
        num_steps: int = 5_000_000,
    ):
        """Train with SAC for sample efficiency."""
        trainer = SACTrainer(
            model=self.model,
            env=env,

            # SAC hyperparameters
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            alpha=0.2,  # Entropy coefficient
        )

        trainer.learn(num_steps)

    def train_amp(
        self,
        env: HumanoidEnv,
        reference_motions: MoCapDataset,
        num_steps: int = 10_000_000,
    ):
        """
        Train with Adversarial Motion Priors (AMP).

        Combines:
        - Task reward (goal-directed behavior)
        - Style reward (motion quality from discriminator)
        """
        # Motion discriminator
        discriminator = MotionDiscriminator(
            observation_dim=32 * 6,  # joints * (pos + vel)
            hidden_dim=512,
        )

        trainer = AMPTrainer(
            model=self.model,
            env=env,
            discriminator=discriminator,
            reference_motions=reference_motions,

            # AMP hyperparameters
            task_reward_weight=0.5,
            style_reward_weight=0.5,
            discriminator_lr=1e-4,
            policy_lr=3e-4,
        )

        trainer.learn(num_steps)

    def train_offline_rl(
        self,
        dataset: HumanoidDemonstrationDataset,
        num_epochs: int = 1000,
    ):
        """
        Train with offline RL from demonstrations.
        """
        # IQL for humanoid (stable offline)
        trainer = IQLTrainer(
            model=self.model,
            dataset=dataset,

            # IQL hyperparameters
            expectile=0.7,
            temperature=3.0,
            learning_rate=3e-4,
            batch_size=256,
        )

        for epoch in range(num_epochs):
            metrics = trainer.train_epoch()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: {metrics}")
```

### GAIL for Motion Imitation

```python
class HumanoidGAILTrainer:
    """
    Generative Adversarial Imitation Learning for humanoid motion.

    Learns implicit reward from expert demonstrations.
    """

    def __init__(
        self,
        model: HumanoidVLA,
        env: HumanoidEnv,
        expert_demos: MoCapDataset,
    ):
        self.model = model
        self.env = env
        self.expert_demos = expert_demos

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(32 * 6 + 32, 512),  # state + action
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def train(
        self,
        num_steps: int = 10_000_000,
        disc_updates_per_policy: int = 5,
    ):
        """Train GAIL."""
        policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        buffer = RolloutBuffer(capacity=10000)

        obs = self.env.reset()

        for step in range(num_steps):
            # Collect trajectory with current policy
            with torch.no_grad():
                action = self.model.get_action(obs)

            next_obs, _, done, info = self.env.step(action)

            # Store transition
            buffer.add(obs, action, next_obs, done)

            obs = next_obs
            if done:
                obs = self.env.reset()

            # Update discriminator
            if step % disc_updates_per_policy == 0:
                for _ in range(disc_updates_per_policy):
                    # Sample from policy buffer
                    policy_batch = buffer.sample(256)

                    # Sample from expert
                    expert_batch = self.expert_demos.sample(256)

                    # Discriminator loss
                    policy_sa = torch.cat([policy_batch["obs"], policy_batch["action"]], dim=-1)
                    expert_sa = torch.cat([expert_batch["obs"], expert_batch["action"]], dim=-1)

                    policy_pred = self.discriminator(policy_sa)
                    expert_pred = self.discriminator(expert_sa)

                    disc_loss = (
                        -torch.log(expert_pred).mean() -
                        torch.log(1 - policy_pred).mean()
                    )

                    disc_optimizer.zero_grad()
                    disc_loss.backward()
                    disc_optimizer.step()

            # Update policy with discriminator reward
            if step % 2048 == 0 and step > 0:
                # Compute discriminator rewards
                for transition in buffer:
                    sa = torch.cat([transition["obs"], transition["action"]], dim=-1)
                    reward = -torch.log(1 - self.discriminator(sa) + 1e-8)
                    transition["reward"] = reward

                # PPO update with GAIL rewards
                self._ppo_update(buffer, policy_optimizer)

                buffer.clear()
```

---

## Stage 7: Human-Robot Interaction

### Natural Language Command Following

```python
class HRITrainer:
    """
    Train humanoid for human-robot interaction.

    Capabilities:
    - Following verbal commands
    - Gesture recognition and response
    - Safe proximity behavior
    - Handover tasks
    - Collaborative manipulation
    """

    def __init__(
        self,
        model: HumanoidVLA,
    ):
        self.model = model

    def train_command_following(
        self,
        dataset: CommandDataset,
        num_epochs: int = 100,
    ):
        """
        Train to follow natural language commands.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in dataset:
                # Input: image + voice command
                output = self.model(
                    images=batch["images"],
                    proprioception=batch["proprioception"],
                    instruction=batch["command"],
                )

                # Loss: match expert behavior
                loss = F.mse_loss(
                    output["joint_targets"],
                    batch["expert_joints"],
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def train_handover(
        self,
        env: HandoverEnv,
        num_steps: int = 2_000_000,
    ):
        """
        Train safe object handover to humans.
        """
        # Handover reward components
        rewards = {
            "approach_human": 0.3,
            "offer_object": 0.3,
            "safe_release": 0.2,
            "human_comfort": 0.2,
        }

        env.set_rewards(rewards)

        trainer = PPOTrainer(
            model=self.model,
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)

    def train_collaborative_manipulation(
        self,
        env: CollaborativeEnv,
        num_steps: int = 5_000_000,
    ):
        """
        Train collaborative tasks with humans.
        """
        # Multi-agent: humanoid + human model
        # Human model simulates human behavior

        trainer = MAPPOTrainer(
            agents=[self.model, HumanModel()],
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)
```

---

## Simulation Environments

### MuJoCo Humanoid Setup

```python
class MuJoCoHumanoidEnv:
    """
    MuJoCo-based humanoid simulation environment.
    """

    def __init__(
        self,
        robot_xml: str = "humanoid.xml",
        scene_xml: str = "scene.xml",
        render: bool = False,
    ):
        import mujoco

        # Load model
        self.model = mujoco.MjModel.from_xml_path(robot_xml)
        self.data = mujoco.MjData(self.model)

        # Renderer
        if render:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Get joint info
        self.num_joints = self.model.nq
        self.action_dim = self.model.nu

    def step(self, action: np.ndarray) -> Tuple:
        """Execute action."""
        # Apply action (torques or position targets)
        self.data.ctrl[:] = action

        # Step physics
        mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        return obs, reward, done, info

    def _get_observation(self) -> Dict:
        """Get full observation."""
        return {
            "joint_pos": self.data.qpos.copy(),
            "joint_vel": self.data.qvel.copy(),
            "torso_pos": self.data.body("torso").xpos.copy(),
            "torso_quat": self.data.body("torso").xquat.copy(),
            "contact": self._get_contacts(),
        }
```

### Isaac Gym / Isaac Sim Setup

```python
class IsaacGymHumanoidEnv:
    """
    Isaac Gym environment for GPU-accelerated humanoid training.

    Features:
    - Thousands of parallel environments
    - GPU-based physics
    - Fast PPO training
    """

    def __init__(
        self,
        num_envs: int = 4096,
        robot: str = "humanoid",
        terrain: str = "flat",
    ):
        from isaacgym import gymapi, gymtorch

        self.gym = gymapi.acquire_gym()
        self.num_envs = num_envs

        # Create sim
        sim_params = self._get_sim_params()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # Create environments
        self._create_envs(robot, terrain)

        # Allocate buffers on GPU
        self._allocate_buffers()

    def step(self, actions: torch.Tensor) -> Tuple:
        """
        Step all environments in parallel.

        Args:
            actions: (num_envs, action_dim) tensor on GPU
        """
        # Apply actions
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(actions),
        )

        # Step physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # Get observations (all on GPU)
        obs = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()

        # Auto-reset done environments
        reset_ids = dones.nonzero().squeeze(-1)
        if len(reset_ids) > 0:
            self._reset_envs(reset_ids)

        return obs, rewards, dones, {}

    def _compute_rewards(self) -> torch.Tensor:
        """Compute rewards for all environments (vectorized)."""
        # Example: forward velocity reward
        forward_vel = self.root_states[:, 7]  # x velocity
        velocity_reward = torch.exp(-torch.abs(forward_vel - self.target_velocity))

        # Height reward (stay upright)
        height = self.root_states[:, 2]  # z position
        height_reward = torch.exp(-torch.abs(height - self.target_height))

        # Energy penalty
        energy = torch.sum(self.dof_vel ** 2, dim=-1)

        reward = velocity_reward + 0.5 * height_reward - 0.001 * energy
        return reward
```

---

## Real Robot Deployment

### Real Humanoid Deployment

```python
class RealHumanoidDeployment:
    """
    Deploy VLA policy to real humanoid robot.

    Safety critical considerations:
    - Torque limits
    - Joint velocity limits
    - Balance monitoring
    - Emergency stop
    """

    def __init__(
        self,
        model_path: str,
        robot_interface: HumanoidInterface,
    ):
        # Load optimized model
        self.model = self._load_optimized_model(model_path)

        # Robot interface (ROS2, SDK, etc.)
        self.robot = robot_interface

        # Safety monitors
        self.balance_monitor = BalanceMonitor()
        self.joint_limit_monitor = JointLimitMonitor(self.robot.joint_limits)
        self.collision_monitor = CollisionMonitor()

        # Emergency stop
        self.emergency_stop = EmergencyStop()

    def run(self, control_frequency: float = 50.0):
        """Main control loop."""
        rate = Rate(control_frequency)

        while self.robot.is_running():
            try:
                # Get sensor data
                images = self.robot.get_camera_images()
                proprio = self.robot.get_proprioception()

                # Safety check
                if not self._safety_check(proprio):
                    self._trigger_safe_stop()
                    continue

                # Run VLA inference
                with torch.no_grad():
                    output = self.model(
                        images=torch.tensor(images).unsqueeze(0).cuda(),
                        proprioception=self._proprio_to_tensor(proprio),
                        instruction=self.current_instruction,
                    )

                # Get joint targets
                joint_targets = output["joint_targets"].cpu().numpy()[0]

                # Apply safety filter
                safe_targets = self._apply_safety_filter(joint_targets, proprio)

                # Send to robot
                self.robot.set_joint_positions(safe_targets)

                rate.sleep()

            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                self._trigger_emergency_stop()

    def _safety_check(self, proprio: Dict) -> bool:
        """Check all safety conditions."""

        # Balance check
        if not self.balance_monitor.check(proprio):
            self.logger.warning("Balance issue detected")
            return False

        # Joint limit check
        if not self.joint_limit_monitor.check(proprio["joint_pos"]):
            self.logger.warning("Joint limit violation")
            return False

        # Collision check
        if self.collision_monitor.check(proprio):
            self.logger.warning("Collision detected")
            return False

        return True

    def _apply_safety_filter(
        self,
        target: np.ndarray,
        current: Dict,
    ) -> np.ndarray:
        """Apply safety limits to joint targets."""

        # Velocity limit
        current_pos = current["joint_pos"]
        dt = 1.0 / 50.0  # control frequency
        max_delta = self.max_joint_velocity * dt

        delta = target - current_pos
        delta = np.clip(delta, -max_delta, max_delta)
        safe_target = current_pos + delta

        # Position limits
        safe_target = np.clip(
            safe_target,
            self.robot.joint_limits["lower"],
            self.robot.joint_limits["upper"],
        )

        return safe_target

    def _trigger_safe_stop(self):
        """Trigger controlled safe stop."""
        self.logger.info("Triggering safe stop")

        # Move to safe pose gradually
        safe_pose = self.robot.get_standing_pose()
        self.robot.move_to_pose(safe_pose, duration=2.0)

    def _trigger_emergency_stop(self):
        """Trigger immediate emergency stop."""
        self.logger.error("Emergency stop triggered!")
        self.emergency_stop.activate()
        self.robot.disable_motors()
```

---

## Advanced Topics

### Sim-to-Real Transfer

```python
class Sim2RealHumanoidAdapter:
    """
    Techniques for transferring humanoid policies from simulation to real.
    """

    def train_with_domain_randomization(
        self,
        env: HumanoidEnv,
        num_steps: int = 50_000_000,
    ):
        """
        Train with extensive domain randomization.
        """
        randomization_params = {
            # Physics
            "mass": (0.8, 1.2),           # Relative to nominal
            "inertia": (0.8, 1.2),
            "friction": (0.5, 1.5),
            "damping": (0.5, 1.5),

            # Sensor noise
            "joint_pos_noise": 0.01,      # radians
            "joint_vel_noise": 0.1,       # rad/s
            "imu_noise": 0.05,

            # Actuator
            "motor_strength": (0.8, 1.2),
            "motor_latency": (0, 0.02),   # seconds

            # External forces
            "push_force": (0, 100),       # Newtons
        }

        env.set_randomization(randomization_params)

        trainer = PPOTrainer(
            model=self.model,
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)

    def train_with_system_identification(
        self,
        real_robot_data: RealRobotDataset,
    ):
        """
        Identify real robot dynamics and adapt simulation.
        """
        # System ID model
        sysid_model = SystemIdentificationModel(
            num_joints=32,
            hidden_dim=256,
        )

        # Train to predict real robot dynamics
        optimizer = torch.optim.Adam(sysid_model.parameters(), lr=1e-4)

        for batch in real_robot_data:
            # Input: state + action
            # Output: next state

            predicted_next = sysid_model(
                batch["state"],
                batch["action"],
            )

            loss = F.mse_loss(predicted_next, batch["next_state"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Update simulation parameters
        identified_params = sysid_model.get_physics_params()
        self.env.set_physics_params(identified_params)
```

### Multi-Task Learning

```python
class MultiTaskHumanoidTrainer:
    """
    Train single policy for multiple humanoid tasks.
    """

    def __init__(self, model: HumanoidVLA):
        self.model = model

        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            "locomotion": LocomotionHead(256, 12),
            "manipulation": ManipulationHead(256, 14),
            "whole_body": WholeBodyHead(256, 32),
        })

    def train_multi_task(
        self,
        envs: Dict[str, HumanoidEnv],
        num_steps_per_task: int = 2_000_000,
    ):
        """
        Train on multiple tasks with task-conditioned policy.
        """
        for epoch in range(10):
            for task_name, env in envs.items():
                print(f"Training task: {task_name}")

                trainer = PPOTrainer(
                    model=self.model,
                    env=env,
                    task_head=self.task_heads[task_name],
                    learning_rate=3e-4,
                )

                trainer.learn(num_steps_per_task // 10)

            # Evaluate all tasks
            for task_name, env in envs.items():
                success_rate = self._evaluate(env, task_name)
                print(f"  {task_name}: {success_rate:.2%}")
```

---

## Summary

This guide covered the complete training pipeline for humanoid VLA:

1. **Stage 1**: VLM foundation with humanoid-specific instructions
2. **Stage 2**: Motion primitive learning from MoCap
3. **Stage 3**: Locomotion training (standing, walking, terrain adaptation)
4. **Stage 4**: Manipulation training (reaching, grasping, bimanual)
5. **Stage 5**: Whole-body control (loco-manipulation, hierarchical)
6. **Stage 6**: Policy improvement (PPO, SAC, AMP, GAIL)
7. **Stage 7**: Human-robot interaction (commands, handover, collaboration)

**Key recommendations:**
- Start with standing/balance before locomotion
- Use curriculum learning for walking speeds
- Leverage MoCap data for natural motions
- Use AMP for motion quality
- Apply domain randomization for sim-to-real
- Always prioritize safety in real deployment

---

## Datasets Used for Each Training Step

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 1: VLM Foundation** | LLaVA-Instruct-150K | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Visual instructions for manipulation, navigation, gestures |
| **Stage 2: Motion Primitive Learning** | CMU Motion Capture Database | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) | Human motion capture for diverse activities |
| **Stage 2: Motion Primitive Learning** | Human3.6M | [vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m/) | 3.6M+ 3D human poses with actions (registration required) |
| **Stage 2: Motion Primitive Learning** | AMASS | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | 40+ hours of motion data, 300+ subjects (includes CMU) |
| **Stage 2: Motion Primitive Learning** | Habitat Humanoids (AMASS subset) | [ai-habitat/habitat_humanoids](https://huggingface.co/datasets/ai-habitat/habitat_humanoids) | Motion clips from AMASS for simulation |
| **Stage 3: Locomotion Training** | D4RL MuJoCo | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) | Ant, Humanoid, HalfCheetah locomotion data |
| **Stage 4: Manipulation Training** | DROID | [cadene/droid](https://huggingface.co/datasets/cadene/droid) | VR/MoCap suit collected teleoperation data |
| **Stage 5: Whole-Body Control** | HumanoidBench | [humanoid-bench.github.io](https://humanoid-bench.github.io/) | Combined locomotion and manipulation |
| **Stage 6a: Online RL** | MuJoCo/Isaac Gym | [mujoco.org](https://mujoco.org/) / [isaac-gym](https://developer.nvidia.com/isaac-gym) | Real-time simulation for PPO/SAC humanoid policy learning |
| **Stage 6a: Online RL (AMP)** | Reference motion clips | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | MoCap data for adversarial motion priors in simulation |
| **Stage 6b: Offline RL** | D4RL Humanoid | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) | Humanoid locomotion trajectories for CQL/IQL training |
| **Stage 6b: Offline RL** | HumanoidBench | [humanoid-bench.github.io](https://humanoid-bench.github.io/) | Offline humanoid manipulation and locomotion data |
| **Stage 7: HRI Training** | HandoverSim | [handoversim.github.io](https://handoversim.github.io/) | Handover and collaborative task data |

---

## Related Documents

- [Training Pipeline Overview](training_pipeline_overview.md)
- [Autonomous Vehicle Training](training_autonomous_vehicle.md)
- [World Model Training](training_temporal_world_model.md)
- [Architecture Guide](architecture.md)
