"""
Model Package

VLA (Vision-Language-Action) model components:

Submodules:
- vlm/: Vision-Language Model components (vision encoders, projectors)
- vla/: Complete VLA model implementations
- action_head/: Action prediction heads (MLP, Diffusion, Transformer)
- sensor/: Sensor encoders (LiDAR, Radar, IMU)
- fusion/: Multi-modal sensor fusion modules
- temporal/: Temporal encoding and memory modules
- world_model/: World modeling for model-based RL
- safety/: Safety constraints and action filtering
- embodiment/: Embodiment-specific models (Driving, Humanoid)
- utils/: Shared utilities (parameter management, checkpointing, device handling)
"""

# VLM components
from .vlm import (
    VisionEncoder,
    VisionEncoderConfig,
    VisionProjector,
    AttentionPoolingProjector,
    PerceiverProjector,
    create_projector,
)

# VLA models
from .vla import (
    VLAModel,
    create_vla_model,
    MultiSensorVLA,
    OpenVLAWrapper,
    SmolVLAWrapper,
)

# Action heads
from .action_head import (
    ActionHeadBase,
    MLPActionHead,
    GaussianMLPActionHead,
    DiffusionActionHead,
    TransformerActionHead,
    GPTActionHead,
)

# Utilities
from .utils import (
    freeze_module,
    unfreeze_module,
    count_parameters,
    count_trainable_parameters,
    get_device,
    save_checkpoint,
    load_checkpoint,
)

# Sensor encoders
from .sensor import (
    PointCloudEncoder,
    PointNetEncoder,
    PointTransformerEncoder,
    RadarEncoder,
    RangeDopplerEncoder,
    IMUEncoder,
    TemporalIMUEncoder,
)

# Fusion modules
from .fusion import (
    SensorFusion,
    CrossModalFusion,
    HierarchicalFusion,
    GatedFusion,
)

# Temporal modules
from .temporal import (
    TemporalEncoder,
    TemporalTransformer,
    TemporalLSTM,
    MemoryBuffer,
    EpisodicMemory,
    WorkingMemory,
    HistoryEncoder,
)

# World model
from .world_model import (
    DynamicsModel,
    LatentWorldModel,
    RSSM,
    RewardPredictor,
)

# Safety
from .safety import (
    SafetyShield,
    RuleChecker,
    ConstraintHandler,
)

# Embodiment
from .embodiment import (
    DrivingVLA,
    BEVEncoder,
    TrajectoryDecoder,
    HumanoidVLA,
    WholeBodyController,
)

__all__ = [
    # VLM
    "VisionEncoder",
    "VisionEncoderConfig",
    "VisionProjector",
    "AttentionPoolingProjector",
    "PerceiverProjector",
    "create_projector",
    # VLA
    "VLAModel",
    "create_vla_model",
    "MultiSensorVLA",
    "OpenVLAWrapper",
    "SmolVLAWrapper",
    # Action heads
    "ActionHeadBase",
    "MLPActionHead",
    "GaussianMLPActionHead",
    "DiffusionActionHead",
    "TransformerActionHead",
    "GPTActionHead",
    # Utilities
    "freeze_module",
    "unfreeze_module",
    "count_parameters",
    "count_trainable_parameters",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    # Sensors
    "PointCloudEncoder",
    "PointNetEncoder",
    "PointTransformerEncoder",
    "RadarEncoder",
    "RangeDopplerEncoder",
    "IMUEncoder",
    "TemporalIMUEncoder",
    # Fusion
    "SensorFusion",
    "CrossModalFusion",
    "HierarchicalFusion",
    "GatedFusion",
    # Temporal
    "TemporalEncoder",
    "TemporalTransformer",
    "TemporalLSTM",
    "MemoryBuffer",
    "EpisodicMemory",
    "WorkingMemory",
    "HistoryEncoder",
    # World Model
    "DynamicsModel",
    "LatentWorldModel",
    "RSSM",
    "RewardPredictor",
    # Safety
    "SafetyShield",
    "RuleChecker",
    "ConstraintHandler",
    # Embodiment
    "DrivingVLA",
    "BEVEncoder",
    "TrajectoryDecoder",
    "HumanoidVLA",
    "WholeBodyController",
]
