"""
AION Deployment Registry - Track all deployed systems.

In-memory registry with querying capabilities for
managing deployed NLP-generated systems.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import structlog

from aion.nlp.types import (
    DeployedSystem,
    DeploymentStatus,
    SpecificationType,
)

logger = structlog.get_logger(__name__)


class DeploymentRegistry:
    """
    Registry for tracking deployed systems.

    Provides:
    - CRUD operations on deployments
    - Querying by type, status, name
    - Version history tracking
    """

    def __init__(self) -> None:
        self._systems: Dict[str, DeployedSystem] = {}
        self._name_index: Dict[str, str] = {}  # name -> id

    def register(self, system: DeployedSystem) -> None:
        """Register a deployed system."""
        self._systems[system.id] = system
        self._name_index[system.name] = system.id
        logger.debug("System registered", id=system.id, name=system.name)

    def get(self, system_id: str) -> Optional[DeployedSystem]:
        """Get a system by ID."""
        return self._systems.get(system_id)

    def find_by_name(self, name: str) -> Optional[DeployedSystem]:
        """Find a system by name."""
        system_id = self._name_index.get(name)
        if system_id:
            return self._systems.get(system_id)
        return None

    def update(self, system: DeployedSystem) -> None:
        """Update a system record."""
        system.updated_at = datetime.now()
        self._systems[system.id] = system
        self._name_index[system.name] = system.id

    def remove(self, system_id: str) -> bool:
        """Remove a system from the registry."""
        system = self._systems.pop(system_id, None)
        if system:
            self._name_index.pop(system.name, None)
            return True
        return False

    def list_all(
        self,
        system_type: Optional[SpecificationType] = None,
        status: Optional[DeploymentStatus] = None,
    ) -> List[DeployedSystem]:
        """List all deployed systems with optional filtering."""
        results = list(self._systems.values())

        if system_type:
            results = [s for s in results if s.system_type == system_type]
        if status:
            results = [s for s in results if s.status == status]

        return sorted(results, key=lambda s: s.created_at, reverse=True)

    def search(self, query: str) -> List[DeployedSystem]:
        """Search systems by name or description."""
        query_lower = query.lower()
        results = []
        for system in self._systems.values():
            if (
                query_lower in system.name.lower()
                or (system.specification and query_lower in getattr(system.specification, "description", "").lower())
            ):
                results.append(system)
        return results

    @property
    def count(self) -> int:
        return len(self._systems)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._systems.values() if s.is_active)
