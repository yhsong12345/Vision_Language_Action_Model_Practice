"""
Humanoid Robot Specific Components

Provides humanoid-specific modules for VLA:
- Whole-body control
- Locomotion policy
- Manipulation policy
- Balance and stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class HumanoidConfig:
    """Configuration for humanoid VLA."""
    # Robot structure
    num_joints: int = 32  # Typical humanoid DOF
    num_body_parts: int = 15  # Head, torso, arms, legs, hands, feet

    # Observation
    proprioception_dim: int = 128  # Joint pos, vel, torque
    image_size: int = 224

    # Action
    action_dim: int = 32  # Joint position or torque commands

    # Architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4

    # Control
    control_freq: float = 100.0  # Hz
    action_type: str = "position"  # "position" or "torque"

    # LLM
    llm_hidden_dim: int = 4096


class ProprioceptionEncoder(nn.Module):
    """
    Encodes proprioceptive information (joint states, IMU, etc.).

    Provides body-aware representation for control.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config

        # Joint state encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(config.num_joints * 3, config.hidden_dim),  # pos, vel, torque
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # IMU encoder (orientation, angular velocity, acceleration)
        self.imu_encoder = nn.Sequential(
            nn.Linear(9, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )

        # Foot contact encoder
        self.contact_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_dim // 4),  # 4 contact points (2 feet x 2)
            nn.ReLU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2 + config.hidden_dim // 4, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode proprioceptive state.

        Args:
            joint_positions: (batch, num_joints)
            joint_velocities: (batch, num_joints)
            joint_torques: (batch, num_joints)
            imu_data: (batch, 9) orientation + angular vel + acceleration
            foot_contacts: (batch, 4) binary contact flags

        Returns:
            proprioception_features: (batch, hidden_dim)
        """
        batch_size = joint_positions.shape[0]

        # Encode joint state
        joint_state = torch.cat([joint_positions, joint_velocities, joint_torques], dim=-1)
        joint_features = self.joint_encoder(joint_state)

        # Encode IMU
        if imu_data is not None:
            imu_features = self.imu_encoder(imu_data)
        else:
            imu_features = torch.zeros(batch_size, self.config.hidden_dim // 2, device=joint_positions.device)

        # Encode contacts
        if foot_contacts is not None:
            contact_features = self.contact_encoder(foot_contacts)
        else:
            contact_features = torch.zeros(batch_size, self.config.hidden_dim // 4, device=joint_positions.device)

        # Fuse all features
        combined = torch.cat([joint_features, imu_features, contact_features], dim=-1)
        return self.fusion(combined)


class LocomotionPolicy(nn.Module):
    """
    Locomotion policy for humanoid walking/running.

    Generates joint commands for stable locomotion given
    velocity commands and terrain information.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config

        # Proprioception encoder
        self.proprio_encoder = ProprioceptionEncoder(config)

        # Command encoder (target velocity)
        self.command_encoder = nn.Sequential(
            nn.Linear(3, config.hidden_dim // 2),  # vx, vy, vyaw
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # Action head (joint positions/torques)
        self.action_mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(config.action_dim))

        # Value head for PPO
        self.value_head = nn.Linear(config.hidden_dim, 1)

        # Phase variable for gait timing
        self.phase_encoder = nn.Linear(2, config.hidden_dim // 4)  # sin, cos of phase

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        velocity_command: torch.Tensor,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate locomotion action.

        Args:
            joint_positions: (batch, num_joints)
            joint_velocities: (batch, num_joints)
            joint_torques: (batch, num_joints)
            velocity_command: (batch, 3) target [vx, vy, vyaw]
            imu_data: (batch, 9)
            foot_contacts: (batch, 4)
            phase: (batch, 2) gait phase [sin, cos]

        Returns:
            Dict with action, value, and distribution parameters
        """
        # Encode proprioception
        proprio_features = self.proprio_encoder(
            joint_positions, joint_velocities, joint_torques, imu_data, foot_contacts
        )

        # Encode command
        command_features = self.command_encoder(velocity_command)

        # Combine features
        features = torch.cat([proprio_features, command_features], dim=-1)

        # Add phase information
        if phase is not None:
            phase_features = self.phase_encoder(phase)
            features = features + F.pad(phase_features, (0, features.shape[-1] - phase_features.shape[-1]))

        # Policy
        policy_features = self.policy_net(features)

        # Action distribution
        action_mean = self.action_mean(policy_features)
        action_std = self.action_log_std.exp()

        # Value
        value = self.value_head(policy_features)

        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value.squeeze(-1),
            "features": policy_features,
        }

    def sample_action(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        velocity_command: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample action from policy."""
        outputs = self.forward(
            joint_positions, joint_velocities, joint_torques,
            velocity_command, **kwargs
        )

        if deterministic:
            return outputs["action_mean"]

        dist = torch.distributions.Normal(outputs["action_mean"], outputs["action_std"])
        return dist.sample()


class ManipulationPolicy(nn.Module):
    """
    Manipulation policy for humanoid arm control.

    Generates arm joint commands for object manipulation
    while maintaining whole-body balance.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config

        # Arm joints (typically 7 per arm)
        self.arm_joints = 14  # Both arms

        # Visual encoder for manipulation
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, config.hidden_dim),
        )

        # Proprioception for arms
        self.arm_proprio_encoder = nn.Sequential(
            nn.Linear(self.arm_joints * 2, config.hidden_dim // 2),  # pos, vel
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )

        # End-effector goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(6 * 2, config.hidden_dim // 2),  # 6D pose for each hand
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        self.arm_action_head = nn.Linear(config.hidden_dim, self.arm_joints)
        self.gripper_head = nn.Linear(config.hidden_dim, 2)  # Left, right gripper

    def forward(
        self,
        image: torch.Tensor,
        arm_joint_positions: torch.Tensor,
        arm_joint_velocities: torch.Tensor,
        target_poses: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate manipulation action.

        Args:
            image: (batch, 3, H, W) visual observation
            arm_joint_positions: (batch, 14) both arm joints
            arm_joint_velocities: (batch, 14)
            target_poses: (batch, 12) target 6D poses for both hands
            language_features: (batch, hidden_dim) from language instruction

        Returns:
            Dict with arm actions and gripper commands
        """
        # Visual encoding
        visual_features = self.visual_encoder(image)

        # Arm proprioception
        arm_state = torch.cat([arm_joint_positions, arm_joint_velocities], dim=-1)
        arm_features = self.arm_proprio_encoder(arm_state)

        # Goal encoding
        if target_poses is not None:
            goal_features = self.goal_encoder(target_poses)
        else:
            goal_features = torch.zeros_like(arm_features)

        # Combine features
        features = torch.cat([visual_features, arm_features, goal_features], dim=-1)

        # Add language conditioning
        if language_features is not None:
            features = features + language_features

        # Policy
        policy_features = self.policy(features)

        # Actions
        arm_actions = self.arm_action_head(policy_features)
        gripper_actions = torch.sigmoid(self.gripper_head(policy_features))

        return {
            "arm_actions": arm_actions,
            "gripper_actions": gripper_actions,
            "features": policy_features,
        }


class WholeBodyController(nn.Module):
    """
    Whole-body controller that coordinates locomotion and manipulation.

    Handles task prioritization and balance maintenance.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config

        # Locomotion policy
        self.locomotion = LocomotionPolicy(config)

        # Manipulation policy
        self.manipulation = ManipulationPolicy(config)

        # Task coordinator
        self.coordinator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Balance controller (QP-based or learned)
        self.balance_net = nn.Sequential(
            nn.Linear(config.hidden_dim + 9, config.hidden_dim // 2),  # + IMU
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.action_dim),
        )

        # Task priority weights
        self.priority_net = nn.Sequential(
            nn.Linear(config.hidden_dim, 3),  # locomotion, manipulation, balance
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        velocity_command: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate whole-body action.

        Coordinates locomotion and manipulation while maintaining balance.
        """
        batch_size = joint_positions.shape[0]

        # Get locomotion action
        loco_output = self.locomotion(
            joint_positions, joint_velocities, joint_torques,
            velocity_command, imu_data, foot_contacts,
        )

        # Get manipulation action (if visual input available)
        if image is not None:
            arm_positions = joint_positions[:, :14]  # Assume first 14 are arm joints
            arm_velocities = joint_velocities[:, :14]
            manip_output = self.manipulation(
                image, arm_positions, arm_velocities,
                target_poses, language_features,
            )
            manip_features = manip_output["features"]
        else:
            manip_output = None
            manip_features = torch.zeros(batch_size, self.config.hidden_dim, device=joint_positions.device)

        # Coordinate tasks
        combined_features = torch.cat([loco_output["features"], manip_features], dim=-1)
        coordinated_features = self.coordinator(combined_features)

        # Compute task priorities
        priorities = self.priority_net(coordinated_features)

        # Balance correction
        if imu_data is not None:
            balance_input = torch.cat([coordinated_features, imu_data], dim=-1)
        else:
            balance_input = F.pad(coordinated_features, (0, 9))
        balance_correction = self.balance_net(balance_input)

        # Combine actions with priorities
        loco_action = loco_output["action_mean"]

        if manip_output is not None:
            # Construct full action from manipulation
            manip_full = torch.zeros_like(loco_action)
            manip_full[:, :14] = manip_output["arm_actions"]
        else:
            manip_full = torch.zeros_like(loco_action)

        # Weighted combination
        final_action = (
            priorities[:, 0:1] * loco_action +
            priorities[:, 1:2] * manip_full +
            priorities[:, 2:3] * balance_correction
        )

        return {
            "action": final_action,
            "locomotion_action": loco_action,
            "manipulation_action": manip_full,
            "balance_correction": balance_correction,
            "priorities": priorities,
            "gripper_actions": manip_output["gripper_actions"] if manip_output else None,
        }


class HumanoidVLA(nn.Module):
    """
    Complete VLA model for humanoid robots.

    Integrates vision, language, and whole-body control.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(256 * 49, config.hidden_dim),
        )

        # Language projector
        self.language_projector = nn.Linear(config.llm_hidden_dim, config.hidden_dim)

        # Whole-body controller
        self.controller = WholeBodyController(config)

        # Task decoder (from language to task parameters)
        self.task_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 3 + 12),  # velocity command + target poses
        )

    def forward(
        self,
        image: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        velocity_command: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for humanoid VLA.

        Args:
            image: (batch, 3, H, W)
            joint_positions: (batch, num_joints)
            joint_velocities: (batch, num_joints)
            joint_torques: (batch, num_joints)
            language_features: (batch, seq_len, llm_dim)
            velocity_command: (batch, 3) optional override
            target_poses: (batch, 12) optional override
            imu_data: (batch, 9)
            foot_contacts: (batch, 4)

        Returns:
            Dict with actions and intermediate outputs
        """
        # Encode vision
        visual_features = self.vision_encoder(image)

        # Process language
        if language_features is not None:
            lang_proj = self.language_projector(language_features)
            lang_pooled = lang_proj.mean(dim=1)

            # Decode task from language
            task_params = self.task_decoder(lang_pooled)
            if velocity_command is None:
                velocity_command = task_params[:, :3]
            if target_poses is None:
                target_poses = task_params[:, 3:]
        else:
            lang_pooled = None
            if velocity_command is None:
                velocity_command = torch.zeros(image.shape[0], 3, device=image.device)

        # Whole-body control
        control_output = self.controller(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            velocity_command=velocity_command,
            image=image,
            target_poses=target_poses,
            imu_data=imu_data,
            foot_contacts=foot_contacts,
            language_features=lang_pooled,
        )

        return {
            "action": control_output["action"],
            "velocity_command": velocity_command,
            "target_poses": target_poses,
            "priorities": control_output["priorities"],
            "gripper_actions": control_output["gripper_actions"],
            "visual_features": visual_features,
        }


if __name__ == "__main__":
    config = HumanoidConfig(
        num_joints=32,
        action_dim=32,
        hidden_dim=256,
    )

    # Test proprioception encoder
    proprio_encoder = ProprioceptionEncoder(config)
    joint_pos = torch.randn(2, 32)
    joint_vel = torch.randn(2, 32)
    joint_torque = torch.randn(2, 32)
    imu = torch.randn(2, 9)

    proprio_features = proprio_encoder(joint_pos, joint_vel, joint_torque, imu)
    print(f"Proprioception features shape: {proprio_features.shape}")

    # Test locomotion policy
    loco_policy = LocomotionPolicy(config)
    vel_cmd = torch.randn(2, 3)
    loco_output = loco_policy(joint_pos, joint_vel, joint_torque, vel_cmd, imu)
    print(f"Locomotion action shape: {loco_output['action_mean'].shape}")

    # Test full humanoid VLA
    humanoid_vla = HumanoidVLA(config)
    image = torch.randn(2, 3, 224, 224)
    output = humanoid_vla(image, joint_pos, joint_vel, joint_torque)
    print(f"\nHumanoid VLA outputs:")
    print(f"  action: {output['action'].shape}")
    print(f"  priorities: {output['priorities'].shape}")
