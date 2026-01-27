"""
AION Distributed Memory Subsystem

Provides consistent-hash-based sharding, distributed vector search,
two-tier caching, and tuneable consistency guarantees for the AION
distributed computing infrastructure.

Exports
-------
ConsistentHash
    SHA-256 consistent hash ring with virtual nodes.
jump_consistent_hash
    Lightweight jump-hash for static cluster topologies.
MemoryShardManager
    Quorum-based shard placement, replication, and rebalancing.
DistributedVectorSearch
    FAISS-style distributed nearest-neighbour search.
DistributedCache
    Two-tier (local LRU + distributed) cache with TTL and invalidation.
ConsistencyManager
    Configurable consistency levels, read repair, and hinted handoff.
"""

from __future__ import annotations

from aion.distributed.memory.cache import DistributedCache
from aion.distributed.memory.consistency import ConsistencyManager
from aion.distributed.memory.distributed_faiss import DistributedVectorSearch
from aion.distributed.memory.sharding import (
    ConsistentHash,
    MemoryShardManager,
    jump_consistent_hash,
)

__all__ = [
    "ConsistentHash",
    "ConsistencyManager",
    "DistributedCache",
    "DistributedVectorSearch",
    "MemoryShardManager",
    "jump_consistent_hash",
]
