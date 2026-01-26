"""
AION Data Action Handler

Data operations from workflows.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import structlog

from aion.automation.types import ActionConfig, DataOperation
from aion.automation.actions.executor import BaseActionHandler

if TYPE_CHECKING:
    from aion.automation.execution.context import ExecutionContext

logger = structlog.get_logger(__name__)


class DataActionHandler(BaseActionHandler):
    """
    Handler for data operations.

    Supports operations on:
    - Context (execution context)
    - Memory (vector memory)
    - Knowledge (knowledge graph)
    - State (persistent state)
    """

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a data action."""
        operation = action.data_operation
        if not operation:
            return {"error": "No data operation specified"}

        source = action.data_source or "context"
        key = context.resolve(action.data_key or "")

        if operation == "read":
            return await self._read(source, key, context, action)
        elif operation == "write":
            value = context.resolve(action.data_value)
            return await self._write(source, key, value, context, action)
        elif operation == "delete":
            return await self._delete(source, key, context, action)
        elif operation == "update":
            value = context.resolve(action.data_value)
            return await self._update(source, key, value, context, action)
        elif operation == "query":
            query = self.resolve_params(action.data_query or {}, context)
            return await self._query(source, query, context, action)
        else:
            return {"error": f"Unknown data operation: {operation}"}

    async def _read(
        self,
        source: str,
        key: str,
        context: "ExecutionContext",
        action: ActionConfig,
    ) -> Dict[str, Any]:
        """Read data from a source."""
        logger.debug("data_read", source=source, key=key)

        if source == "context":
            value = context.get(key)
            return {
                "operation": "read",
                "source": source,
                "key": key,
                "value": value,
                "found": value is not None,
            }

        if source == "memory":
            return await self._read_memory(key, context)

        if source == "knowledge":
            return await self._read_knowledge(key, context)

        if source == "state":
            return await self._read_state(key, context)

        return {"error": f"Unknown data source: {source}"}

    async def _write(
        self,
        source: str,
        key: str,
        value: Any,
        context: "ExecutionContext",
        action: ActionConfig,
    ) -> Dict[str, Any]:
        """Write data to a source."""
        logger.debug("data_write", source=source, key=key)

        if source == "context":
            context.set(key, value)
            return {
                "operation": "write",
                "source": source,
                "key": key,
                "success": True,
            }

        if source == "memory":
            return await self._write_memory(key, value, context)

        if source == "knowledge":
            return await self._write_knowledge(key, value, context)

        if source == "state":
            return await self._write_state(key, value, context)

        return {"error": f"Unknown data source: {source}"}

    async def _delete(
        self,
        source: str,
        key: str,
        context: "ExecutionContext",
        action: ActionConfig,
    ) -> Dict[str, Any]:
        """Delete data from a source."""
        logger.debug("data_delete", source=source, key=key)

        if source == "context":
            context.delete(key)
            return {
                "operation": "delete",
                "source": source,
                "key": key,
                "success": True,
            }

        if source == "memory":
            return await self._delete_memory(key, context)

        if source == "state":
            return await self._delete_state(key, context)

        return {"error": f"Unknown data source: {source}"}

    async def _update(
        self,
        source: str,
        key: str,
        value: Any,
        context: "ExecutionContext",
        action: ActionConfig,
    ) -> Dict[str, Any]:
        """Update data in a source (merge for dicts)."""
        if source == "context":
            existing = context.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
                context.set(key, existing)
            else:
                context.set(key, value)

            return {
                "operation": "update",
                "source": source,
                "key": key,
                "success": True,
            }

        # For other sources, update is same as write
        return await self._write(source, key, value, context, action)

    async def _query(
        self,
        source: str,
        query: Dict[str, Any],
        context: "ExecutionContext",
        action: ActionConfig,
    ) -> Dict[str, Any]:
        """Query data from a source."""
        logger.debug("data_query", source=source, query=query)

        if source == "memory":
            return await self._query_memory(query, context)

        if source == "knowledge":
            return await self._query_knowledge(query, context)

        return {"error": f"Query not supported for source: {source}"}

    # === Memory Operations ===

    async def _read_memory(
        self,
        key: str,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Read from vector memory."""
        try:
            from aion.systems.memory.search import MemorySearch

            memory = MemorySearch()
            result = await memory.retrieve(key)

            return {
                "operation": "read",
                "source": "memory",
                "key": key,
                "value": result,
                "found": result is not None,
            }

        except ImportError:
            return {
                "operation": "read",
                "source": "memory",
                "key": key,
                "value": None,
                "found": False,
                "simulated": True,
            }

    async def _write_memory(
        self,
        key: str,
        value: Any,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Write to vector memory."""
        try:
            from aion.systems.memory.search import MemorySearch

            memory = MemorySearch()
            await memory.store(key, value)

            return {
                "operation": "write",
                "source": "memory",
                "key": key,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "write",
                "source": "memory",
                "key": key,
                "success": True,
                "simulated": True,
            }

    async def _delete_memory(
        self,
        key: str,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Delete from vector memory."""
        try:
            from aion.systems.memory.search import MemorySearch

            memory = MemorySearch()
            await memory.delete(key)

            return {
                "operation": "delete",
                "source": "memory",
                "key": key,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "delete",
                "source": "memory",
                "key": key,
                "success": True,
                "simulated": True,
            }

    async def _query_memory(
        self,
        query: Dict[str, Any],
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Query vector memory."""
        try:
            from aion.systems.memory.search import MemorySearch

            memory = MemorySearch()
            results = await memory.search(
                query=query.get("query", ""),
                limit=query.get("limit", 10),
            )

            return {
                "operation": "query",
                "source": "memory",
                "results": results,
                "count": len(results),
            }

        except ImportError:
            return {
                "operation": "query",
                "source": "memory",
                "results": [],
                "count": 0,
                "simulated": True,
            }

    # === Knowledge Operations ===

    async def _read_knowledge(
        self,
        key: str,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Read from knowledge graph."""
        try:
            from aion.systems.knowledge.graph import KnowledgeGraph

            kg = KnowledgeGraph()
            result = await kg.get_entity(key)

            return {
                "operation": "read",
                "source": "knowledge",
                "key": key,
                "value": result,
                "found": result is not None,
            }

        except ImportError:
            return {
                "operation": "read",
                "source": "knowledge",
                "key": key,
                "value": None,
                "found": False,
                "simulated": True,
            }

    async def _write_knowledge(
        self,
        key: str,
        value: Any,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Write to knowledge graph."""
        try:
            from aion.systems.knowledge.graph import KnowledgeGraph

            kg = KnowledgeGraph()
            await kg.add_entity(key, value)

            return {
                "operation": "write",
                "source": "knowledge",
                "key": key,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "write",
                "source": "knowledge",
                "key": key,
                "success": True,
                "simulated": True,
            }

    async def _query_knowledge(
        self,
        query: Dict[str, Any],
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Query knowledge graph."""
        try:
            from aion.systems.knowledge.graph import KnowledgeGraph

            kg = KnowledgeGraph()
            results = await kg.query(query)

            return {
                "operation": "query",
                "source": "knowledge",
                "results": results,
            }

        except ImportError:
            return {
                "operation": "query",
                "source": "knowledge",
                "results": [],
                "simulated": True,
            }

    # === State Operations ===

    async def _read_state(
        self,
        key: str,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Read from persistent state."""
        try:
            from aion.state.manager import StateManager

            state = StateManager()
            value = await state.get(key)

            return {
                "operation": "read",
                "source": "state",
                "key": key,
                "value": value,
                "found": value is not None,
            }

        except ImportError:
            return {
                "operation": "read",
                "source": "state",
                "key": key,
                "value": None,
                "found": False,
                "simulated": True,
            }

    async def _write_state(
        self,
        key: str,
        value: Any,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Write to persistent state."""
        try:
            from aion.state.manager import StateManager

            state = StateManager()
            await state.set(key, value)

            return {
                "operation": "write",
                "source": "state",
                "key": key,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "write",
                "source": "state",
                "key": key,
                "success": True,
                "simulated": True,
            }

    async def _delete_state(
        self,
        key: str,
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Delete from persistent state."""
        try:
            from aion.state.manager import StateManager

            state = StateManager()
            await state.delete(key)

            return {
                "operation": "delete",
                "source": "state",
                "key": key,
                "success": True,
            }

        except ImportError:
            return {
                "operation": "delete",
                "source": "state",
                "key": key,
                "success": True,
                "simulated": True,
            }
