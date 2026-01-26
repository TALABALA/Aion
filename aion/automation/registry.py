"""
AION Workflow Registry

Storage and retrieval of workflow definitions.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import uuid

import structlog

from aion.automation.types import (
    Workflow,
    WorkflowStatus,
    WorkflowExecution,
    ExecutionStatus,
)

logger = structlog.get_logger(__name__)


class WorkflowRegistry:
    """
    Registry for workflow definitions and executions.

    Features:
    - In-memory storage with optional persistence
    - Query and filtering
    - Version management
    - Event callbacks
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        auto_persist: bool = True,
    ):
        self.persistence_path = persistence_path
        self.auto_persist = auto_persist

        # Storage
        self._workflows: Dict[str, Workflow] = {}
        self._executions: Dict[str, WorkflowExecution] = {}

        # Indices
        self._by_name: Dict[str, str] = {}  # name -> id
        self._by_status: Dict[WorkflowStatus, List[str]] = {s: [] for s in WorkflowStatus}
        self._by_tag: Dict[str, List[str]] = {}
        self._by_owner: Dict[str, List[str]] = {}

        # Event callbacks
        self._on_workflow_created: List[Callable] = []
        self._on_workflow_updated: List[Callable] = []
        self._on_workflow_deleted: List[Callable] = []
        self._on_execution_started: List[Callable] = []
        self._on_execution_completed: List[Callable] = []

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the registry."""
        if self._initialized:
            return

        # Load persisted workflows
        if self.persistence_path and self.persistence_path.exists():
            await self._load_from_disk()

        self._initialized = True
        logger.info("Workflow registry initialized", workflow_count=len(self._workflows))

    async def shutdown(self) -> None:
        """Shutdown the registry."""
        if self.persistence_path and self.auto_persist:
            await self._save_to_disk()

        self._initialized = False

    # === Workflow Operations ===

    async def save(self, workflow: Workflow) -> str:
        """Save a workflow."""
        async with self._lock:
            is_new = workflow.id not in self._workflows

            # Update timestamp
            workflow.updated_at = datetime.now()
            if is_new:
                workflow.created_at = datetime.now()

            # Validate
            errors = workflow.validate()
            if errors:
                raise ValueError(f"Invalid workflow: {errors}")

            # Store
            old_workflow = self._workflows.get(workflow.id)
            self._workflows[workflow.id] = workflow

            # Update indices
            self._update_indices(workflow, old_workflow)

            # Persist
            if self.persistence_path and self.auto_persist:
                await self._save_to_disk()

            # Fire callbacks
            if is_new:
                await self._fire_callbacks(self._on_workflow_created, workflow)
            else:
                await self._fire_callbacks(self._on_workflow_updated, workflow)

            logger.info(
                "workflow_saved",
                workflow_id=workflow.id,
                name=workflow.name,
                is_new=is_new,
            )

            return workflow.id

    async def get(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    async def get_by_name(self, name: str) -> Optional[Workflow]:
        """Get a workflow by name."""
        workflow_id = self._by_name.get(name)
        if workflow_id:
            return self._workflows.get(workflow_id)
        return None

    async def delete(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        async with self._lock:
            workflow = self._workflows.pop(workflow_id, None)
            if not workflow:
                return False

            # Update indices
            self._remove_from_indices(workflow)

            # Persist
            if self.persistence_path and self.auto_persist:
                await self._save_to_disk()

            # Fire callbacks
            await self._fire_callbacks(self._on_workflow_deleted, workflow)

            logger.info("workflow_deleted", workflow_id=workflow_id)

            return True

    async def list(
        self,
        status: Optional[WorkflowStatus] = None,
        tag: Optional[str] = None,
        owner_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Workflow]:
        """List workflows with filters."""
        workflows = list(self._workflows.values())

        # Apply filters
        if status:
            workflow_ids = set(self._by_status.get(status, []))
            workflows = [w for w in workflows if w.id in workflow_ids]

        if tag:
            workflow_ids = set(self._by_tag.get(tag, []))
            workflows = [w for w in workflows if w.id in workflow_ids]

        if owner_id:
            workflow_ids = set(self._by_owner.get(owner_id, []))
            workflows = [w for w in workflows if w.id in workflow_ids]

        if search:
            search_lower = search.lower()
            workflows = [
                w for w in workflows
                if search_lower in w.name.lower() or search_lower in w.description.lower()
            ]

        # Sort by updated_at descending
        workflows.sort(key=lambda w: w.updated_at, reverse=True)

        # Paginate
        return workflows[offset:offset + limit]

    async def count(
        self,
        status: Optional[WorkflowStatus] = None,
    ) -> int:
        """Count workflows."""
        if status:
            return len(self._by_status.get(status, []))
        return len(self._workflows)

    # === Execution Operations ===

    async def save_execution(self, execution: WorkflowExecution) -> str:
        """Save an execution."""
        async with self._lock:
            is_new = execution.id not in self._executions

            self._executions[execution.id] = execution

            # Fire callbacks
            if is_new:
                await self._fire_callbacks(self._on_execution_started, execution)
            elif execution.is_terminal():
                await self._fire_callbacks(self._on_execution_completed, execution)

            return execution.id

    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get an execution by ID."""
        return self._executions.get(execution_id)

    async def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        initiated_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WorkflowExecution]:
        """List executions with filters."""
        executions = list(self._executions.values())

        # Apply filters
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]

        if status:
            executions = [e for e in executions if e.status == status]

        if initiated_by:
            executions = [e for e in executions if e.initiated_by == initiated_by]

        # Sort by started_at descending
        executions.sort(
            key=lambda e: e.started_at or datetime.min,
            reverse=True,
        )

        # Paginate
        return executions[offset:offset + limit]

    async def count_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
    ) -> int:
        """Count executions."""
        executions = list(self._executions.values())

        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]

        if status:
            executions = [e for e in executions if e.status == status]

        return len(executions)

    async def delete_execution(self, execution_id: str) -> bool:
        """Delete an execution."""
        async with self._lock:
            execution = self._executions.pop(execution_id, None)
            return execution is not None

    async def cleanup_executions(
        self,
        older_than_hours: float = 24.0,
        keep_count: int = 100,
    ) -> int:
        """Clean up old executions."""
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=older_than_hours)

        async with self._lock:
            # Get executions to delete
            executions = sorted(
                self._executions.values(),
                key=lambda e: e.started_at or datetime.min,
                reverse=True,
            )

            deleted = 0
            for i, execution in enumerate(executions):
                # Keep recent ones
                if i < keep_count:
                    continue

                # Delete old completed ones
                if execution.is_terminal():
                    if execution.completed_at and execution.completed_at < cutoff:
                        del self._executions[execution.id]
                        deleted += 1

            logger.info("executions_cleaned", deleted=deleted)
            return deleted

    # === Indexing ===

    def _update_indices(
        self,
        workflow: Workflow,
        old_workflow: Optional[Workflow] = None,
    ) -> None:
        """Update indices for a workflow."""
        # Remove old indices
        if old_workflow:
            self._remove_from_indices(old_workflow)

        # Add new indices
        self._by_name[workflow.name] = workflow.id
        self._by_status[workflow.status].append(workflow.id)

        for tag in workflow.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(workflow.id)

        if workflow.owner_id:
            if workflow.owner_id not in self._by_owner:
                self._by_owner[workflow.owner_id] = []
            self._by_owner[workflow.owner_id].append(workflow.id)

    def _remove_from_indices(self, workflow: Workflow) -> None:
        """Remove workflow from indices."""
        # Remove from name index
        if self._by_name.get(workflow.name) == workflow.id:
            del self._by_name[workflow.name]

        # Remove from status index
        if workflow.id in self._by_status[workflow.status]:
            self._by_status[workflow.status].remove(workflow.id)

        # Remove from tag indices
        for tag in workflow.tags:
            if tag in self._by_tag and workflow.id in self._by_tag[tag]:
                self._by_tag[tag].remove(workflow.id)

        # Remove from owner index
        if workflow.owner_id and workflow.owner_id in self._by_owner:
            if workflow.id in self._by_owner[workflow.owner_id]:
                self._by_owner[workflow.owner_id].remove(workflow.id)

    # === Event Callbacks ===

    def on_workflow_created(self, callback: Callable) -> None:
        """Register callback for workflow creation."""
        self._on_workflow_created.append(callback)

    def on_workflow_updated(self, callback: Callable) -> None:
        """Register callback for workflow updates."""
        self._on_workflow_updated.append(callback)

    def on_workflow_deleted(self, callback: Callable) -> None:
        """Register callback for workflow deletion."""
        self._on_workflow_deleted.append(callback)

    def on_execution_started(self, callback: Callable) -> None:
        """Register callback for execution start."""
        self._on_execution_started.append(callback)

    def on_execution_completed(self, callback: Callable) -> None:
        """Register callback for execution completion."""
        self._on_execution_completed.append(callback)

    async def _fire_callbacks(self, callbacks: List[Callable], *args) -> None:
        """Fire callbacks."""
        for callback in callbacks:
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("callback_error", error=str(e))

    # === Persistence ===

    async def _load_from_disk(self) -> None:
        """Load workflows from disk."""
        try:
            workflows_file = self.persistence_path / "workflows.json"
            if workflows_file.exists():
                with open(workflows_file, "r") as f:
                    data = json.load(f)

                for workflow_data in data.get("workflows", []):
                    workflow = Workflow.from_dict(workflow_data)
                    self._workflows[workflow.id] = workflow
                    self._update_indices(workflow)

                logger.info("workflows_loaded", count=len(self._workflows))

        except Exception as e:
            logger.error("load_error", error=str(e))

    async def _save_to_disk(self) -> None:
        """Save workflows to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)

            workflows_file = self.persistence_path / "workflows.json"
            data = {
                "workflows": [w.to_dict() for w in self._workflows.values()],
                "saved_at": datetime.now().isoformat(),
            }

            with open(workflows_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error("save_error", error=str(e))

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        status_counts = {s.value: len(ids) for s, ids in self._by_status.items()}

        execution_status_counts = {}
        for status in ExecutionStatus:
            execution_status_counts[status.value] = len([
                e for e in self._executions.values()
                if e.status == status
            ])

        return {
            "total_workflows": len(self._workflows),
            "workflows_by_status": status_counts,
            "total_executions": len(self._executions),
            "executions_by_status": execution_status_counts,
            "total_tags": len(self._by_tag),
            "total_owners": len(self._by_owner),
        }
