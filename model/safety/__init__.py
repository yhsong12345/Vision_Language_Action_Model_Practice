"""
Safety and Constraint Modules for VLA

Critical for autonomous vehicles and humanoid robots.
Provides multiple layers of safety:
- SafetyShield: Runtime action filtering and correction
- RuleChecker: Rule-based constraint verification
- ConstraintHandler: Constraint satisfaction and optimization
"""

from .safety_shield import (
    SafetyShield,
    ActionFilter,
    SafetyMonitor,
    EmergencyStop,
)
from .rule_checker import (
    RuleChecker,
    TrafficRuleChecker,
    CollisionChecker,
    KinematicChecker,
)
from .constraint_handler import (
    ConstraintHandler,
    SafetyConstraint,
    ConstraintOptimizer,
    BarrierFunction,
)

__all__ = [
    "SafetyShield",
    "ActionFilter",
    "SafetyMonitor",
    "EmergencyStop",
    "RuleChecker",
    "TrafficRuleChecker",
    "CollisionChecker",
    "KinematicChecker",
    "ConstraintHandler",
    "SafetyConstraint",
    "ConstraintOptimizer",
    "BarrierFunction",
]
