"""
AION Distributed Training System

Production-grade distributed reinforcement learning infrastructure providing:
- Distributed RL training with parameter server pattern and leader/follower roles
- SOTA gradient synchronization with compression, ring all-reduce, and mixed precision
- Cross-node experience sharing with priority selection, deduplication, and diversity tracking
"""

from aion.distributed.training.distributed_rl import (
    DistributedRLConfig,
    DistributedRLTrainer,
)
from aion.distributed.training.experience_sharing import (
    ExperienceSharing,
    ExperienceSharingConfig,
)
from aion.distributed.training.gradient_sync import (
    GradientSyncConfig,
    GradientSynchronizer,
)

__all__ = [
    "DistributedRLConfig",
    "DistributedRLTrainer",
    "ExperienceSharing",
    "ExperienceSharingConfig",
    "GradientSyncConfig",
    "GradientSynchronizer",
]
