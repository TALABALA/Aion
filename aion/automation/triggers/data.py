"""
AION Data Change Trigger Handler

Data mutation triggers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.automation.types import Trigger, TriggerType, DataOperation
from aion.automation.triggers.manager import BaseTriggerHandler

logger = structlog.get_logger(__name__)


class DataChangeTriggerHandler(BaseTriggerHandler):
    """
    Handler for data change triggers.

    Features:
    - Source filtering (memory, knowledge, state)
    - Operation filtering (create, update, delete)
    - Data pattern matching
    - Change tracking
    """

    def __init__(self, manager):
        super().__init__(manager)
        self._change_log: List[DataChange] = []
        self._max_log_size = 1000

    async def register(self, trigger: Trigger) -> None:
        """Register a data change trigger."""
        config = trigger.config

        logger.info(
            "data_trigger_registered",
            trigger_id=trigger.id,
            source=config.data_source,
            operation=config.data_operation,
            has_filter=bool(config.data_filter),
        )

    async def unregister(self, trigger: Trigger) -> None:
        """Unregister a data change trigger."""
        logger.info("data_trigger_unregistered", trigger_id=trigger.id)

    async def notify_change(
        self,
        source: str,
        operation: str,
        key: str,
        old_value: Any = None,
        new_value: Any = None,
        metadata: Dict[str, Any] = None,
    ) -> List[str]:
        """
        Notify of a data change and trigger matching workflows.

        Args:
            source: Data source (memory, knowledge, state)
            operation: Operation type (create, update, delete)
            key: Data key that changed
            old_value: Previous value (for update/delete)
            new_value: New value (for create/update)
            metadata: Additional metadata

        Returns:
            List of triggered execution IDs
        """
        change = DataChange(
            source=source,
            operation=operation,
            key=key,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {},
        )

        # Log the change
        self._change_log.append(change)
        if len(self._change_log) > self._max_log_size:
            self._change_log = self._change_log[-self._max_log_size:]

        # Prepare data for triggers
        data = {
            "source": source,
            "operation": operation,
            "key": key,
            "new_value": new_value,
            "old_value": old_value,
            "metadata": metadata,
        }

        # Trigger matching workflows
        execution_ids = await self.manager.handle_data_change(
            source=source,
            operation=operation,
            data=data,
        )

        if execution_ids:
            logger.info(
                "data_change_triggered",
                source=source,
                operation=operation,
                key=key,
                triggered=len(execution_ids),
            )

        return execution_ids

    def get_change_log(
        self,
        source: str = None,
        operation: str = None,
        key_pattern: str = None,
        limit: int = 100,
    ) -> List["DataChange"]:
        """Get change log with optional filters."""
        changes = self._change_log

        if source:
            changes = [c for c in changes if c.source == source]

        if operation:
            changes = [c for c in changes if c.operation == operation]

        if key_pattern:
            import re
            pattern = re.compile(key_pattern)
            changes = [c for c in changes if pattern.match(c.key)]

        # Return most recent first
        return list(reversed(changes[-limit:]))

    def clear_change_log(self) -> int:
        """Clear change log."""
        count = len(self._change_log)
        self._change_log.clear()
        return count


# === Data Change Model ===


from dataclasses import dataclass, field


@dataclass
class DataChange:
    """Record of a data change."""
    source: str
    operation: str
    key: str
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "operation": self.operation,
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# === Data Sources ===


class DataSources:
    """Standard data source constants."""

    # AION data sources
    MEMORY = "memory"           # Vector memory
    KNOWLEDGE = "knowledge"     # Knowledge graph
    STATE = "state"             # Persistent state
    CONTEXT = "context"         # Execution context
    CACHE = "cache"             # Cache layer

    # External sources
    DATABASE = "database"
    API = "api"
    FILE = "file"


class DataOperations:
    """Standard data operation constants."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    BATCH = "batch"
