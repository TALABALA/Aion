"""
AION Distributed Task System

Production-grade distributed task infrastructure providing:
- Priority-based task queuing with deduplication and dead letter support
- DAG-aware task scheduling with dependency resolution
- Local task execution with handler registry and timeout enforcement
- Intelligent task routing with capability matching and load balancing
"""

from aion.distributed.tasks.executor import TaskExecutor
from aion.distributed.tasks.queue import DistributedTaskQueue
from aion.distributed.tasks.routing import TaskRouter
from aion.distributed.tasks.scheduler import TaskScheduler

__all__ = [
    "DistributedTaskQueue",
    "TaskScheduler",
    "TaskExecutor",
    "TaskRouter",
]
