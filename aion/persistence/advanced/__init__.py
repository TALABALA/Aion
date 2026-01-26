"""
AION Advanced Persistence Features

True SOTA capabilities:
- Real-time streaming queries
- Temporal tables (time-travel queries)
- Multi-tenant data isolation
- Schema registry for event evolution
- Data lineage tracking
- Write-behind caching
- Materialized view management
"""

from aion.persistence.advanced.streaming import (
    StreamingQueryManager,
    Subscription,
    ChangeStream,
)
from aion.persistence.advanced.temporal import (
    TemporalTable,
    TimePoint,
    TemporalQuery,
)
from aion.persistence.advanced.multitenancy import (
    TenantManager,
    TenantContext,
    TenantIsolationLevel,
)
from aion.persistence.advanced.schema_registry import (
    SchemaRegistry,
    EventSchema,
    SchemaVersion,
    Upcaster,
)
from aion.persistence.advanced.lineage import (
    DataLineageTracker,
    LineageNode,
    LineageEdge,
)

__all__ = [
    # Streaming
    "StreamingQueryManager",
    "Subscription",
    "ChangeStream",
    # Temporal
    "TemporalTable",
    "TimePoint",
    "TemporalQuery",
    # Multi-tenancy
    "TenantManager",
    "TenantContext",
    "TenantIsolationLevel",
    # Schema Registry
    "SchemaRegistry",
    "EventSchema",
    "SchemaVersion",
    "Upcaster",
    # Lineage
    "DataLineageTracker",
    "LineageNode",
    "LineageEdge",
]
