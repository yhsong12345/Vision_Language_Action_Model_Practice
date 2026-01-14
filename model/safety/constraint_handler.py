"""
Constraint Handler Module

Handles safety constraints through optimization-based methods.
Key components:
- SafetyConstraint: Defines constraint functions
- ConstraintOptimizer: Projects actions to satisfy constraints
- BarrierFunction: Control Barrier Functions for safe RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class ConstraintConfig:
    """Configuration for constraint handling."""
    state_dim: int = 768
    action_dim: int = 7
    hidden_dim: int = 256
    num_constraints: int = 3
    barrier_alpha: float = 0.1  # CBF decay rate
    slack_penalty: float = 100.0  # Penalty for constraint violation
    optimization_steps: int = 10
    optimization_lr: float = 0.1


class SafetyConstraint(ABC):
    """Abstract base class for safety constraints."""

    @abstractmethod
    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate constraint function.

        Returns:
            constraint_value: Positive = satisfied, Negative = violated
        """
        pass

    @abstractmethod
    def gradient(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of constraint w.r.t. action.

        Returns:
            gradient: (action_dim,) gradient tensor
        """
        pass


class ActionBoundConstraint(SafetyConstraint):
    """Constraint for action bounds."""

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Returns minimum margin to bounds."""
        lower_margin = action - self.lower_bound
        upper_margin = self.upper_bound - action

        # Minimum margin (positive = within bounds)
        return torch.min(lower_margin.min(-1).values, upper_margin.min(-1).values)

    def gradient(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient pushes action away from violated bounds."""
        lower_violation = (self.lower_bound - action).clamp(min=0)
        upper_violation = (action - self.upper_bound).clamp(min=0)
        return lower_violation - upper_violation


class CollisionConstraint(SafetyConstraint):
    """Constraint for collision avoidance."""

    def __init__(
        self,
        obstacle_positions: torch.Tensor,
        obstacle_radii: torch.Tensor,
        robot_radius: float = 0.5,
    ):
        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        self.robot_radius = robot_radius

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate collision constraint.

        Assumes action contains position delta.
        """
        # Current position from state
        position = state[:, :2]  # Assume first 2 dims are position

        # Predicted position after action
        next_position = position + action[:, :2]

        # Distance to obstacles
        distances = torch.cdist(next_position, self.obstacle_positions)

        # Minimum clearance (distance - combined radii)
        clearance = distances - (self.obstacle_radii + self.robot_radius)

        return clearance.min(-1).values

    def gradient(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient points away from obstacles."""
        position = state[:, :2]
        next_position = position + action[:, :2]

        # Vector from each obstacle to robot
        diff = next_position.unsqueeze(1) - self.obstacle_positions.unsqueeze(0)
        distances = torch.norm(diff, dim=-1, keepdim=True)

        # Normalized direction away from obstacles
        directions = diff / (distances + 1e-6)

        # Weight by proximity
        weights = 1.0 / (distances.squeeze(-1) + 0.1)
        weights = weights / weights.sum(-1, keepdim=True)

        # Weighted average direction
        gradient = (directions * weights.unsqueeze(-1)).sum(1)

        # Pad to action dimension
        full_gradient = torch.zeros_like(action)
        full_gradient[:, :2] = gradient

        return full_gradient


class LearnedConstraint(SafetyConstraint, nn.Module):
    """Learned constraint function."""

    def __init__(self, config: ConstraintConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)

    def gradient(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action = action.requires_grad_(True)
        value = self.evaluate(state, action)
        grad = torch.autograd.grad(
            value.sum(),
            action,
            create_graph=True,
        )[0]
        return grad


class ConstraintOptimizer(nn.Module):
    """
    Optimizes actions to satisfy constraints.

    Uses projected gradient descent to find nearest feasible action.
    """

    def __init__(self, config: ConstraintConfig):
        super().__init__()
        self.config = config
        self.constraints: List[SafetyConstraint] = []

    def add_constraint(self, constraint: SafetyConstraint):
        """Add a constraint to the optimizer."""
        self.constraints.append(constraint)

    def project(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Project action to satisfy all constraints.

        Args:
            state: Current state
            action: Proposed action

        Returns:
            projected_action: Feasible action
            info: Optimization info
        """
        if len(self.constraints) == 0:
            return action, {"violations": torch.zeros(action.size(0))}

        # Optimization variable
        projected = action.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([projected], lr=self.config.optimization_lr)

        violations_history = []

        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()

            # Compute all constraint values
            constraint_values = []
            for constraint in self.constraints:
                value = constraint.evaluate(state, projected)
                constraint_values.append(value)

            constraint_values = torch.stack(constraint_values, dim=-1)

            # Compute violation (negative constraint values)
            violations = F.relu(-constraint_values).sum(-1)
            violations_history.append(violations.mean().item())

            # Loss: minimize distance to original + constraint violation penalty
            distance_loss = F.mse_loss(projected, action)
            constraint_loss = self.config.slack_penalty * violations.mean()
            total_loss = distance_loss + constraint_loss

            if violations.max() < 1e-6:
                break

            total_loss.backward()
            optimizer.step()

        return projected.detach(), {
            "violations": violations.detach(),
            "steps": step + 1,
            "violation_history": violations_history,
        }


class BarrierFunction(nn.Module):
    """
    Control Barrier Function for safe RL.

    Ensures forward invariance of safe set through barrier constraint:
    dh/dt + alpha * h(x) >= 0

    where h(x) >= 0 defines the safe set.
    """

    def __init__(self, config: ConstraintConfig):
        super().__init__()
        self.config = config

        # Learned barrier function h(x)
        self.barrier_net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Learned dynamics for Lie derivative
        self.dynamics_net = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )

    def barrier_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compute h(x). Positive = safe, Negative = unsafe."""
        return self.barrier_net(state).squeeze(-1)

    def lie_derivative(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Lie derivative of barrier along dynamics.

        L_f h = dh/dx * f(x, u)
        """
        state = state.requires_grad_(True)
        h = self.barrier_value(state)

        # Gradient of h w.r.t. state
        dh_dx = torch.autograd.grad(
            h.sum(),
            state,
            create_graph=True,
        )[0]

        # Dynamics (state derivative)
        x = torch.cat([state, action], dim=-1)
        f = self.dynamics_net(x)

        # Lie derivative
        L_f_h = (dh_dx * f).sum(-1)

        return L_f_h

    def cbf_constraint(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CBF constraint value.

        Returns: L_f h + alpha * h (should be >= 0 for safety)
        """
        h = self.barrier_value(state)
        L_f_h = self.lie_derivative(state, action)

        return L_f_h + self.config.barrier_alpha * h

    def safe_action(
        self,
        state: torch.Tensor,
        nominal_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute minimally-modified safe action using CBF-QP.

        Solves: min ||u - u_nom||^2
                s.t. L_f h + alpha * h >= 0
        """
        # Simple gradient-based solution
        action = nominal_action.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([action], lr=0.1)

        for _ in range(self.config.optimization_steps):
            optimizer.zero_grad()

            # CBF constraint
            cbf = self.cbf_constraint(state, action)
            violation = F.relu(-cbf)

            # Loss: distance to nominal + constraint violation
            loss = F.mse_loss(action, nominal_action) + 100 * violation.mean()

            if violation.max() < 1e-6:
                break

            loss.backward()
            optimizer.step()

        return action.detach(), {
            "cbf_value": cbf.detach(),
            "barrier_value": self.barrier_value(state).detach(),
        }

    def loss(
        self,
        safe_states: torch.Tensor,
        unsafe_states: torch.Tensor,
        boundary_states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training loss for CBF.

        Args:
            safe_states: States known to be safe (h should be > 0)
            unsafe_states: States known to be unsafe (h should be < 0)
            boundary_states: States on boundary (h should be ~ 0)
            actions: Actions taken
            next_states: Resulting next states
        """
        # Safe states: h > 0
        h_safe = self.barrier_value(safe_states)
        safe_loss = F.relu(-h_safe + 0.1).mean()  # Margin of 0.1

        # Unsafe states: h < 0
        h_unsafe = self.barrier_value(unsafe_states)
        unsafe_loss = F.relu(h_unsafe + 0.1).mean()

        # Decrease constraint
        h_current = self.barrier_value(boundary_states)
        h_next = self.barrier_value(next_states)
        decrease_loss = F.relu(
            -h_next + h_current - self.config.barrier_alpha * h_current
        ).mean()

        total_loss = safe_loss + unsafe_loss + decrease_loss

        return {
            "total_loss": total_loss,
            "safe_loss": safe_loss,
            "unsafe_loss": unsafe_loss,
            "decrease_loss": decrease_loss,
        }


class ConstraintHandler(nn.Module):
    """
    Complete constraint handling system.

    Integrates multiple constraint types and optimization methods.
    """

    def __init__(self, config: ConstraintConfig):
        super().__init__()
        self.config = config

        # Constraint optimizer
        self.optimizer = ConstraintOptimizer(config)

        # Control barrier function
        self.cbf = BarrierFunction(config)

        # Learned constraints
        self.learned_constraints = nn.ModuleList([
            LearnedConstraint(config)
            for _ in range(config.num_constraints)
        ])

        # Mode: "optimization" or "cbf"
        self.mode = "optimization"

    def add_bound_constraint(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ):
        """Add action bound constraint."""
        constraint = ActionBoundConstraint(lower, upper)
        self.optimizer.add_constraint(constraint)

    def add_collision_constraint(
        self,
        obstacle_positions: torch.Tensor,
        obstacle_radii: torch.Tensor,
    ):
        """Add collision avoidance constraint."""
        constraint = CollisionConstraint(obstacle_positions, obstacle_radii)
        self.optimizer.add_constraint(constraint)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply constraint handling to action.

        Returns:
            Dict with safe_action and constraint info
        """
        if self.mode == "cbf":
            safe_action, cbf_info = self.cbf.safe_action(state, action)
            return {
                "safe_action": safe_action,
                "method": "cbf",
                **cbf_info,
            }
        else:
            # Add learned constraints to optimizer
            for lc in self.learned_constraints:
                self.optimizer.add_constraint(lc)

            safe_action, opt_info = self.optimizer.project(state, action)

            return {
                "safe_action": safe_action,
                "method": "optimization",
                **opt_info,
            }

    def get_constraint_values(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get all constraint values for monitoring."""
        values = []

        for constraint in self.optimizer.constraints:
            values.append(constraint.evaluate(state, action))

        for lc in self.learned_constraints:
            values.append(lc.evaluate(state, action))

        if len(values) == 0:
            return torch.zeros(state.size(0), 1)

        return torch.stack(values, dim=-1)

    def is_feasible(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        tolerance: float = 0.0,
    ) -> torch.Tensor:
        """Check if state-action pair satisfies all constraints."""
        values = self.get_constraint_values(state, action)
        return (values >= -tolerance).all(-1)


if __name__ == "__main__":
    config = ConstraintConfig(
        state_dim=768,
        action_dim=7,
    )

    # Test bound constraint
    lower = torch.tensor([-1.0] * 7)
    upper = torch.tensor([1.0] * 7)
    bound_constraint = ActionBoundConstraint(lower, upper)

    state = torch.randn(4, 768)
    action = torch.randn(4, 7) * 2  # Some out of bounds

    value = bound_constraint.evaluate(state, action)
    print(f"Bound constraint values: {value}")

    # Test constraint optimizer
    optimizer = ConstraintOptimizer(config)
    optimizer.add_constraint(bound_constraint)

    projected, info = optimizer.project(state, action)
    print(f"Projected action range: [{projected.min():.2f}, {projected.max():.2f}]")
    print(f"Optimization steps: {info['steps']}")

    # Test CBF
    cbf = BarrierFunction(config)
    h = cbf.barrier_value(state)
    print(f"Barrier values: {h}")

    cbf_value = cbf.cbf_constraint(state, action)
    print(f"CBF constraint values: {cbf_value}")

    safe_action, cbf_info = cbf.safe_action(state, action)
    print(f"Safe action computed via CBF")

    # Test full constraint handler
    handler = ConstraintHandler(config)
    handler.add_bound_constraint(lower, upper)

    outputs = handler(state, action)
    print(f"\nConstraint handler outputs:")
    print(f"  method: {outputs['method']}")
    print(f"  safe_action range: [{outputs['safe_action'].min():.2f}, {outputs['safe_action'].max():.2f}]")
