"""
Safety Shield Module

Runtime safety layer that filters and corrects unsafe actions.
Provides the last line of defense for autonomous systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Callable
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class SafetyLevel(Enum):
    """Safety levels for different scenarios."""
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class SafetyConfig:
    """Configuration for safety modules."""
    # Action limits
    max_linear_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 1.0  # rad/s
    max_acceleration: float = 3.0  # m/s^2
    max_jerk: float = 10.0  # m/s^3

    # Safety distances
    min_obstacle_distance: float = 0.5  # meters
    emergency_stop_distance: float = 0.2  # meters

    # Humanoid specific
    max_joint_velocity: float = 2.0  # rad/s
    max_joint_torque: float = 100.0  # Nm
    min_stability_margin: float = 0.05  # meters

    # Vehicle specific
    max_steering_angle: float = 0.5  # radians
    max_steering_rate: float = 0.3  # rad/s

    # Action dimensions
    action_dim: int = 7
    state_dim: int = 768

    # Shield parameters
    shield_type: str = "hard"  # "hard", "soft", "learned"
    intervention_threshold: float = 0.8


class ActionFilter(nn.Module):
    """
    Filters actions to ensure they stay within safe bounds.

    Supports both hard clipping and soft projection methods.
    """

    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config

        # Action bounds (can be state-dependent)
        self.register_buffer(
            "action_low",
            torch.tensor([-1.0] * config.action_dim)
        )
        self.register_buffer(
            "action_high",
            torch.tensor([1.0] * config.action_dim)
        )

        # Velocity limits for smooth filtering
        self.register_buffer(
            "max_action_delta",
            torch.tensor([0.1] * config.action_dim)  # Per timestep
        )

        # Previous action for rate limiting
        self.prev_action = None

    def forward(
        self,
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Filter action to ensure safety.

        Args:
            action: Proposed action
            state: Current state (for state-dependent limits)

        Returns:
            safe_action: Filtered action
            info: Dictionary with filtering information
        """
        original_action = action.clone()

        # 1. Bound clipping
        action = torch.clamp(action, self.action_low, self.action_high)

        # 2. Rate limiting
        if self.prev_action is not None:
            delta = action - self.prev_action
            delta = torch.clamp(delta, -self.max_action_delta, self.max_action_delta)
            action = self.prev_action + delta

        self.prev_action = action.clone()

        # Compute intervention info
        intervention = (original_action - action).abs().sum(-1)
        was_modified = intervention > 1e-6

        return action, {
            "original_action": original_action,
            "intervention": intervention,
            "was_modified": was_modified,
        }

    def reset(self):
        """Reset filter state (e.g., at episode start)."""
        self.prev_action = None


class SafetyMonitor(nn.Module):
    """
    Monitors system state for safety violations.

    Tracks multiple safety metrics and triggers alerts.
    """

    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config

        # Learned safety predictor
        self.safety_net = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(SafetyLevel)),
            nn.Softmax(dim=-1),
        )

        # Safety history
        self.safety_history = []
        self.max_history = 100

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Monitor safety of state-action pair.

        Returns:
            Dict with safety level probabilities and metrics
        """
        x = torch.cat([state, action], dim=-1)
        safety_probs = self.safety_net(x)

        # Get predicted safety level
        safety_level = safety_probs.argmax(-1)

        # Compute risk score (expected safety level)
        levels = torch.arange(len(SafetyLevel), device=safety_probs.device, dtype=torch.float)
        risk_score = (safety_probs * levels).sum(-1)

        return {
            "safety_probs": safety_probs,
            "safety_level": safety_level,
            "risk_score": risk_score,
            "is_safe": safety_level <= SafetyLevel.CAUTION.value,
        }

    def update_history(self, safety_info: Dict[str, torch.Tensor]):
        """Update safety history."""
        self.safety_history.append({
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in safety_info.items()
        })

        if len(self.safety_history) > self.max_history:
            self.safety_history.pop(0)

    def get_safety_trend(self) -> Dict[str, float]:
        """Analyze recent safety trend."""
        if len(self.safety_history) < 2:
            return {"trend": 0.0, "avg_risk": 0.0}

        recent_risks = [h["risk_score"].mean().item() for h in self.safety_history[-10:]]
        avg_risk = np.mean(recent_risks)
        trend = recent_risks[-1] - recent_risks[0] if len(recent_risks) > 1 else 0

        return {"trend": trend, "avg_risk": avg_risk}


class EmergencyStop(nn.Module):
    """
    Emergency stop mechanism.

    Immediately halts the system when critical safety violations detected.
    """

    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config

        # Emergency conditions
        self.emergency_triggered = False
        self.trigger_reason = None

        # Safe stop action (all zeros or specific safe pose)
        self.register_buffer(
            "stop_action",
            torch.zeros(config.action_dim)
        )

    def check_emergency(
        self,
        state: torch.Tensor,
        obstacle_distance: Optional[float] = None,
        safety_level: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check if emergency stop should be triggered.

        Returns:
            should_stop: Whether to trigger emergency stop
            reason: Reason for stop
        """
        reasons = []

        # Check obstacle distance
        if obstacle_distance is not None:
            if obstacle_distance < self.config.emergency_stop_distance:
                reasons.append(f"Obstacle too close: {obstacle_distance:.2f}m")

        # Check safety level
        if safety_level is not None:
            if safety_level >= SafetyLevel.EMERGENCY.value:
                reasons.append(f"Safety level critical: {safety_level}")

        should_stop = len(reasons) > 0

        if should_stop:
            self.emergency_triggered = True
            self.trigger_reason = "; ".join(reasons)

        return should_stop, self.trigger_reason or "No emergency"

    def get_stop_action(self) -> torch.Tensor:
        """Get the safe stop action."""
        return self.stop_action.clone()

    def reset(self):
        """Reset emergency state."""
        self.emergency_triggered = False
        self.trigger_reason = None


class SafetyShield(nn.Module):
    """
    Complete safety shield combining all safety components.

    Provides a unified interface for runtime safety enforcement.
    """

    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config

        # Components
        self.action_filter = ActionFilter(config)
        self.safety_monitor = SafetyMonitor(config)
        self.emergency_stop = EmergencyStop(config)

        # Learned intervention policy
        if config.shield_type == "learned":
            self.intervention_policy = nn.Sequential(
                nn.Linear(config.state_dim + config.action_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, config.action_dim),
                nn.Tanh(),
            )

        # Statistics
        self.intervention_count = 0
        self.total_count = 0

    def forward(
        self,
        state: torch.Tensor,
        proposed_action: torch.Tensor,
        obstacle_distance: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply safety shield to proposed action.

        Args:
            state: Current state
            proposed_action: Action from policy
            obstacle_distance: Distance to nearest obstacle

        Returns:
            Dict with safe action and safety information
        """
        self.total_count += 1

        # 1. Check for emergency
        safety_info = self.safety_monitor(state, proposed_action)
        should_stop, stop_reason = self.emergency_stop.check_emergency(
            state,
            obstacle_distance=obstacle_distance,
            safety_level=safety_info["safety_level"].item() if safety_info["safety_level"].dim() == 0
                        else safety_info["safety_level"][0].item(),
        )

        if should_stop:
            return {
                "action": self.emergency_stop.get_stop_action().unsqueeze(0).expand(state.size(0), -1),
                "is_emergency": True,
                "reason": stop_reason,
                "safety_info": safety_info,
            }

        # 2. Filter action
        safe_action, filter_info = self.action_filter(proposed_action, state)

        # 3. Learned intervention (if enabled)
        if self.config.shield_type == "learned":
            risk_score = safety_info["risk_score"]

            if risk_score.mean() > self.config.intervention_threshold:
                # Compute intervention
                x = torch.cat([state, proposed_action, safe_action], dim=-1)
                intervention = self.intervention_policy(x)
                safe_action = safe_action + intervention * (risk_score.unsqueeze(-1) - self.config.intervention_threshold)
                safe_action = torch.clamp(safe_action, -1, 1)

        # Track interventions
        if filter_info["was_modified"].any():
            self.intervention_count += 1

        # Update monitor history
        self.safety_monitor.update_history(safety_info)

        return {
            "action": safe_action,
            "original_action": proposed_action,
            "is_emergency": False,
            "safety_info": safety_info,
            "filter_info": filter_info,
            "intervention_rate": self.intervention_count / max(1, self.total_count),
        }

    def reset(self):
        """Reset shield state."""
        self.action_filter.reset()
        self.emergency_stop.reset()
        self.intervention_count = 0
        self.total_count = 0

    def get_statistics(self) -> Dict[str, float]:
        """Get safety statistics."""
        trend = self.safety_monitor.get_safety_trend()
        return {
            "intervention_rate": self.intervention_count / max(1, self.total_count),
            "avg_risk": trend["avg_risk"],
            "risk_trend": trend["trend"],
            "emergency_triggered": self.emergency_stop.emergency_triggered,
        }


class LearnedSafetyShield(nn.Module):
    """
    Learned safety shield using control barrier functions.

    Learns to minimally modify actions to ensure safety constraints.
    """

    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config

        # Barrier function network (h(x) >= 0 means safe)
        self.barrier_net = nn.Sequential(
            nn.Linear(config.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Action correction network
        self.correction_net = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.action_dim),
            nn.Tanh(),
        )

    def barrier_function(self, state: torch.Tensor) -> torch.Tensor:
        """Compute barrier function value."""
        return self.barrier_net(state)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply learned safety correction.

        Args:
            state: Current state
            action: Proposed action

        Returns:
            safe_action: Corrected action
            info: Safety information
        """
        # Compute barrier value
        h = self.barrier_function(state)

        # Compute correction based on barrier value
        x = torch.cat([state, action, h], dim=-1)
        correction = self.correction_net(x)

        # Apply correction scaled by how unsafe we are
        # More correction when h is negative (unsafe)
        correction_scale = F.relu(-h)  # Only correct when unsafe
        safe_action = action + correction * correction_scale

        # Clamp to valid range
        safe_action = torch.clamp(safe_action, -1, 1)

        return safe_action, {
            "barrier_value": h,
            "correction": correction,
            "is_safe": h >= 0,
        }

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        safe_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training loss for learned safety shield.

        Args:
            safe_labels: 1 if safe, 0 if unsafe
        """
        h = self.barrier_function(states)

        # Barrier should be positive for safe states, negative for unsafe
        barrier_loss = F.binary_cross_entropy_with_logits(h.squeeze(), safe_labels)

        # CBF constraint: h(x') >= h(x) - alpha * h(x) when h(x) >= 0
        h_next = self.barrier_function(next_states)
        alpha = 0.1
        cbf_violation = F.relu(h - h_next - alpha * h)
        cbf_loss = cbf_violation.mean()

        return barrier_loss + cbf_loss


if __name__ == "__main__":
    config = SafetyConfig(
        action_dim=7,
        state_dim=768,
        shield_type="learned",
    )

    # Test action filter
    action_filter = ActionFilter(config)
    action = torch.randn(4, 7) * 2  # Some actions out of bounds
    safe_action, info = action_filter(action)
    print(f"Action filter - was modified: {info['was_modified']}")

    # Test safety monitor
    monitor = SafetyMonitor(config)
    state = torch.randn(4, 768)
    safety_info = monitor(state, safe_action)
    print(f"Safety monitor - risk score: {safety_info['risk_score']}")

    # Test full safety shield
    shield = SafetyShield(config)
    outputs = shield(state, action)
    print(f"Safety shield outputs:")
    print(f"  is_emergency: {outputs['is_emergency']}")
    print(f"  intervention_rate: {outputs['intervention_rate']}")

    # Test learned safety shield
    learned_shield = LearnedSafetyShield(config)
    safe_action, info = learned_shield(state, action)
    print(f"Learned shield - barrier value: {info['barrier_value'].mean():.3f}")
