"""
Rule Checker Module

Rule-based constraint verification for safety-critical systems.
Implements domain-specific rules for:
- Traffic rules (autonomous vehicles)
- Collision avoidance
- Kinematic feasibility
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod


class ViolationType(Enum):
    """Types of rule violations."""
    NONE = 0
    WARNING = 1
    VIOLATION = 2
    CRITICAL = 3


@dataclass
class RuleViolation:
    """Represents a rule violation."""
    rule_name: str
    violation_type: ViolationType
    message: str
    severity: float  # 0-1 scale
    suggested_correction: Optional[torch.Tensor] = None


@dataclass
class TrafficRuleConfig:
    """Configuration for traffic rule checking."""
    max_speed: float = 30.0  # m/s (about 108 km/h)
    max_acceleration: float = 4.0  # m/s^2
    max_deceleration: float = 8.0  # m/s^2 (emergency)
    min_following_distance: float = 2.0  # seconds (time headway)
    max_lane_deviation: float = 0.5  # meters
    stop_line_tolerance: float = 0.3  # meters


class RuleChecker(ABC):
    """Abstract base class for rule checkers."""

    @abstractmethod
    def check(self, *args, **kwargs) -> List[RuleViolation]:
        """Check for rule violations."""
        pass

    @abstractmethod
    def get_rule_names(self) -> List[str]:
        """Get list of rules being checked."""
        pass


class TrafficRuleChecker(RuleChecker, nn.Module):
    """
    Checks traffic rules for autonomous vehicles.

    Includes:
    - Speed limits
    - Following distance
    - Lane keeping
    - Traffic signal compliance
    - Right of way
    """

    def __init__(self, config: TrafficRuleConfig):
        super().__init__()
        self.config = config

        # Speed limit can be context-dependent
        self.current_speed_limit = config.max_speed

        # Traffic light state
        self.traffic_light_state = "green"

        # Rule definitions
        self.rules = {
            "speed_limit": self._check_speed_limit,
            "following_distance": self._check_following_distance,
            "lane_keeping": self._check_lane_keeping,
            "traffic_signal": self._check_traffic_signal,
            "acceleration": self._check_acceleration,
        }

    def get_rule_names(self) -> List[str]:
        return list(self.rules.keys())

    def check(
        self,
        ego_state: Dict[str, torch.Tensor],
        environment: Optional[Dict[str, Any]] = None,
    ) -> List[RuleViolation]:
        """
        Check all traffic rules.

        Args:
            ego_state: Dict with speed, position, acceleration, etc.
            environment: Dict with traffic light, lead vehicle, lanes, etc.

        Returns:
            List of violations
        """
        violations = []

        for rule_name, check_fn in self.rules.items():
            violation = check_fn(ego_state, environment)
            if violation is not None:
                violations.append(violation)

        return violations

    def _check_speed_limit(
        self,
        ego_state: Dict[str, torch.Tensor],
        environment: Optional[Dict[str, Any]],
    ) -> Optional[RuleViolation]:
        """Check speed limit compliance."""
        speed = ego_state.get("speed", torch.tensor(0.0))
        if isinstance(speed, torch.Tensor):
            speed = speed.item()

        # Get context-specific speed limit
        speed_limit = self.current_speed_limit
        if environment and "speed_limit" in environment:
            speed_limit = environment["speed_limit"]

        if speed > speed_limit:
            excess = speed - speed_limit
            severity = min(1.0, excess / 10.0)  # 10 m/s excess = max severity

            return RuleViolation(
                rule_name="speed_limit",
                violation_type=ViolationType.VIOLATION if excess > 5 else ViolationType.WARNING,
                message=f"Speed {speed:.1f} exceeds limit {speed_limit:.1f}",
                severity=severity,
            )
        return None

    def _check_following_distance(
        self,
        ego_state: Dict[str, torch.Tensor],
        environment: Optional[Dict[str, Any]],
    ) -> Optional[RuleViolation]:
        """Check safe following distance."""
        if environment is None or "lead_vehicle_distance" not in environment:
            return None

        distance = environment["lead_vehicle_distance"]
        speed = ego_state.get("speed", torch.tensor(0.0))
        if isinstance(speed, torch.Tensor):
            speed = speed.item()

        # Time headway
        if speed > 0.1:
            time_headway = distance / speed
            min_headway = self.config.min_following_distance

            if time_headway < min_headway:
                severity = min(1.0, (min_headway - time_headway) / min_headway)

                return RuleViolation(
                    rule_name="following_distance",
                    violation_type=ViolationType.WARNING if severity < 0.5 else ViolationType.VIOLATION,
                    message=f"Following distance {time_headway:.1f}s < {min_headway:.1f}s",
                    severity=severity,
                )
        return None

    def _check_lane_keeping(
        self,
        ego_state: Dict[str, torch.Tensor],
        environment: Optional[Dict[str, Any]],
    ) -> Optional[RuleViolation]:
        """Check lane keeping."""
        lane_deviation = ego_state.get("lane_deviation", torch.tensor(0.0))
        if isinstance(lane_deviation, torch.Tensor):
            lane_deviation = lane_deviation.abs().item()

        max_deviation = self.config.max_lane_deviation

        if lane_deviation > max_deviation:
            severity = min(1.0, (lane_deviation - max_deviation) / max_deviation)

            return RuleViolation(
                rule_name="lane_keeping",
                violation_type=ViolationType.WARNING if severity < 0.5 else ViolationType.VIOLATION,
                message=f"Lane deviation {lane_deviation:.2f}m exceeds {max_deviation:.2f}m",
                severity=severity,
            )
        return None

    def _check_traffic_signal(
        self,
        ego_state: Dict[str, torch.Tensor],
        environment: Optional[Dict[str, Any]],
    ) -> Optional[RuleViolation]:
        """Check traffic signal compliance."""
        if environment is None:
            return None

        light_state = environment.get("traffic_light", "green")
        distance_to_stop = environment.get("distance_to_stop_line", float("inf"))
        speed = ego_state.get("speed", torch.tensor(0.0))
        if isinstance(speed, torch.Tensor):
            speed = speed.item()

        if light_state == "red":
            if distance_to_stop < 0:  # Past stop line
                return RuleViolation(
                    rule_name="traffic_signal",
                    violation_type=ViolationType.CRITICAL,
                    message="Ran red light",
                    severity=1.0,
                )
            elif distance_to_stop < self.config.stop_line_tolerance and speed > 0.5:
                return RuleViolation(
                    rule_name="traffic_signal",
                    violation_type=ViolationType.WARNING,
                    message="Approaching red light too fast",
                    severity=0.7,
                )
        return None

    def _check_acceleration(
        self,
        ego_state: Dict[str, torch.Tensor],
        environment: Optional[Dict[str, Any]],
    ) -> Optional[RuleViolation]:
        """Check acceleration limits."""
        acceleration = ego_state.get("acceleration", torch.tensor(0.0))
        if isinstance(acceleration, torch.Tensor):
            acceleration = acceleration.item()

        if acceleration > self.config.max_acceleration:
            severity = min(1.0, (acceleration - self.config.max_acceleration) / self.config.max_acceleration)
            return RuleViolation(
                rule_name="acceleration",
                violation_type=ViolationType.WARNING,
                message=f"Acceleration {acceleration:.1f} exceeds {self.config.max_acceleration:.1f}",
                severity=severity,
            )
        elif acceleration < -self.config.max_deceleration:
            severity = min(1.0, (-acceleration - self.config.max_deceleration) / self.config.max_deceleration)
            return RuleViolation(
                rule_name="acceleration",
                violation_type=ViolationType.WARNING if severity < 0.5 else ViolationType.VIOLATION,
                message=f"Hard braking: {acceleration:.1f}",
                severity=severity,
            )
        return None

    def set_speed_limit(self, speed_limit: float):
        """Update current speed limit."""
        self.current_speed_limit = speed_limit


class CollisionChecker(RuleChecker, nn.Module):
    """
    Checks for potential collisions.

    Uses geometric and predictive methods to detect collision risks.
    """

    def __init__(
        self,
        vehicle_length: float = 4.5,
        vehicle_width: float = 2.0,
        prediction_horizon: float = 3.0,  # seconds
    ):
        super().__init__()
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.prediction_horizon = prediction_horizon

        # Safety margins
        self.front_margin = 2.0
        self.side_margin = 0.5
        self.rear_margin = 1.0

    def get_rule_names(self) -> List[str]:
        return ["collision_static", "collision_dynamic", "time_to_collision"]

    def check(
        self,
        ego_state: Dict[str, torch.Tensor],
        obstacles: List[Dict[str, Any]],
    ) -> List[RuleViolation]:
        """
        Check for collision risks.

        Args:
            ego_state: Dict with position, velocity, heading
            obstacles: List of obstacle dicts with position, velocity, size

        Returns:
            List of violations
        """
        violations = []

        for obstacle in obstacles:
            # Static collision check
            violation = self._check_static_collision(ego_state, obstacle)
            if violation:
                violations.append(violation)

            # Time to collision
            ttc_violation = self._check_time_to_collision(ego_state, obstacle)
            if ttc_violation:
                violations.append(ttc_violation)

        return violations

    def _check_static_collision(
        self,
        ego_state: Dict[str, torch.Tensor],
        obstacle: Dict[str, Any],
    ) -> Optional[RuleViolation]:
        """Check for immediate collision."""
        ego_pos = ego_state.get("position", torch.zeros(2))
        if isinstance(ego_pos, torch.Tensor):
            ego_pos = ego_pos.cpu().numpy()

        obs_pos = np.array(obstacle.get("position", [0, 0]))
        obs_size = obstacle.get("size", [1, 1])

        # Simple bounding box check
        dx = abs(ego_pos[0] - obs_pos[0])
        dy = abs(ego_pos[1] - obs_pos[1])

        min_dx = (self.vehicle_length + obs_size[0]) / 2 + self.front_margin
        min_dy = (self.vehicle_width + obs_size[1]) / 2 + self.side_margin

        if dx < min_dx and dy < min_dy:
            # Collision or very close
            severity = 1.0 - min(dx / min_dx, dy / min_dy)
            return RuleViolation(
                rule_name="collision_static",
                violation_type=ViolationType.CRITICAL if severity > 0.8 else ViolationType.WARNING,
                message=f"Collision risk with obstacle at ({obs_pos[0]:.1f}, {obs_pos[1]:.1f})",
                severity=severity,
            )
        return None

    def _check_time_to_collision(
        self,
        ego_state: Dict[str, torch.Tensor],
        obstacle: Dict[str, Any],
    ) -> Optional[RuleViolation]:
        """Compute time to collision."""
        ego_pos = ego_state.get("position", torch.zeros(2))
        ego_vel = ego_state.get("velocity", torch.zeros(2))
        if isinstance(ego_pos, torch.Tensor):
            ego_pos = ego_pos.cpu().numpy()
        if isinstance(ego_vel, torch.Tensor):
            ego_vel = ego_vel.cpu().numpy()

        obs_pos = np.array(obstacle.get("position", [0, 0]))
        obs_vel = np.array(obstacle.get("velocity", [0, 0]))

        # Relative position and velocity
        rel_pos = obs_pos - ego_pos
        rel_vel = obs_vel - ego_vel

        # Simple TTC calculation (assumes constant velocity)
        if np.linalg.norm(rel_vel) > 0.1:
            # Project relative position onto relative velocity direction
            ttc = -np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_vel) ** 2)

            if 0 < ttc < self.prediction_horizon:
                # Check if collision actually occurs
                collision_point = ego_pos + ego_vel * ttc
                obs_at_collision = obs_pos + obs_vel * ttc
                distance_at_collision = np.linalg.norm(collision_point - obs_at_collision)

                safe_distance = (self.vehicle_length + obstacle.get("size", [1, 1])[0]) / 2

                if distance_at_collision < safe_distance:
                    severity = 1.0 - ttc / self.prediction_horizon
                    return RuleViolation(
                        rule_name="time_to_collision",
                        violation_type=ViolationType.CRITICAL if ttc < 1.0 else ViolationType.WARNING,
                        message=f"TTC = {ttc:.1f}s with obstacle",
                        severity=severity,
                    )
        return None


class KinematicChecker(RuleChecker, nn.Module):
    """
    Checks kinematic feasibility of actions.

    Ensures actions are physically achievable by the robot/vehicle.
    """

    def __init__(
        self,
        robot_type: str = "vehicle",  # "vehicle" or "humanoid"
        max_joint_velocities: Optional[List[float]] = None,
        max_joint_accelerations: Optional[List[float]] = None,
    ):
        super().__init__()
        self.robot_type = robot_type

        # Default limits
        if robot_type == "vehicle":
            self.max_steering_angle = 0.5  # radians
            self.max_steering_rate = 0.3  # rad/s
            self.wheelbase = 2.7  # meters
            self.min_turn_radius = self.wheelbase / np.tan(self.max_steering_angle)
        else:  # humanoid
            self.num_joints = 20
            self.max_joint_velocities = max_joint_velocities or [2.0] * self.num_joints
            self.max_joint_accelerations = max_joint_accelerations or [10.0] * self.num_joints

    def get_rule_names(self) -> List[str]:
        if self.robot_type == "vehicle":
            return ["steering_limit", "turn_radius", "steering_rate"]
        else:
            return ["joint_velocity", "joint_acceleration", "workspace"]

    def check(
        self,
        current_state: Dict[str, torch.Tensor],
        proposed_action: torch.Tensor,
        dt: float = 0.1,
    ) -> List[RuleViolation]:
        """
        Check kinematic feasibility.

        Args:
            current_state: Current joint/vehicle state
            proposed_action: Proposed action
            dt: Time step

        Returns:
            List of violations
        """
        if self.robot_type == "vehicle":
            return self._check_vehicle_kinematics(current_state, proposed_action, dt)
        else:
            return self._check_humanoid_kinematics(current_state, proposed_action, dt)

    def _check_vehicle_kinematics(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        dt: float,
    ) -> List[RuleViolation]:
        """Check vehicle kinematic constraints."""
        violations = []

        # Assume action[0] is steering angle
        if action.dim() > 0:
            steering = action[0].item() if action.dim() == 1 else action[0, 0].item()
        else:
            steering = action.item()

        # Check steering limit
        if abs(steering) > self.max_steering_angle:
            violations.append(RuleViolation(
                rule_name="steering_limit",
                violation_type=ViolationType.WARNING,
                message=f"Steering {steering:.2f} exceeds limit {self.max_steering_angle:.2f}",
                severity=min(1.0, abs(steering) / self.max_steering_angle - 1),
            ))

        # Check steering rate
        current_steering = state.get("steering_angle", torch.tensor(0.0))
        if isinstance(current_steering, torch.Tensor):
            current_steering = current_steering.item()

        steering_rate = (steering - current_steering) / dt
        if abs(steering_rate) > self.max_steering_rate:
            violations.append(RuleViolation(
                rule_name="steering_rate",
                violation_type=ViolationType.WARNING,
                message=f"Steering rate {steering_rate:.2f} exceeds limit {self.max_steering_rate:.2f}",
                severity=min(1.0, abs(steering_rate) / self.max_steering_rate - 1),
            ))

        return violations

    def _check_humanoid_kinematics(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        dt: float,
    ) -> List[RuleViolation]:
        """Check humanoid kinematic constraints."""
        violations = []

        # Check joint velocity limits
        current_joints = state.get("joint_positions", torch.zeros(self.num_joints))
        if isinstance(action, torch.Tensor):
            proposed_joints = action[:self.num_joints] if action.size(-1) >= self.num_joints else action

        joint_velocities = (proposed_joints - current_joints) / dt

        for i, (vel, max_vel) in enumerate(zip(joint_velocities, self.max_joint_velocities)):
            if isinstance(vel, torch.Tensor):
                vel = vel.item()
            if abs(vel) > max_vel:
                violations.append(RuleViolation(
                    rule_name="joint_velocity",
                    violation_type=ViolationType.WARNING,
                    message=f"Joint {i} velocity {vel:.2f} exceeds limit {max_vel:.2f}",
                    severity=min(1.0, abs(vel) / max_vel - 1),
                ))
                break  # Only report first violation

        return violations


if __name__ == "__main__":
    # Test traffic rule checker
    config = TrafficRuleConfig()
    traffic_checker = TrafficRuleChecker(config)

    ego_state = {
        "speed": torch.tensor(35.0),  # Over limit
        "acceleration": torch.tensor(2.0),
        "lane_deviation": torch.tensor(0.3),
    }

    environment = {
        "speed_limit": 30.0,
        "lead_vehicle_distance": 15.0,
        "traffic_light": "green",
    }

    violations = traffic_checker.check(ego_state, environment)
    print(f"Traffic rule violations: {len(violations)}")
    for v in violations:
        print(f"  - {v.rule_name}: {v.message}")

    # Test collision checker
    collision_checker = CollisionChecker()

    obstacles = [
        {"position": [10, 0.5], "velocity": [0, 0], "size": [4, 2]},
    ]

    ego_state = {
        "position": torch.tensor([0.0, 0.0]),
        "velocity": torch.tensor([5.0, 0.0]),
    }

    violations = collision_checker.check(ego_state, obstacles)
    print(f"\nCollision violations: {len(violations)}")
    for v in violations:
        print(f"  - {v.rule_name}: {v.message}")

    # Test kinematic checker
    kinematic_checker = KinematicChecker(robot_type="vehicle")

    state = {"steering_angle": torch.tensor(0.0)}
    action = torch.tensor([0.6, 1.0])  # steering, throttle

    violations = kinematic_checker.check(state, action)
    print(f"\nKinematic violations: {len(violations)}")
    for v in violations:
        print(f"  - {v.rule_name}: {v.message}")
