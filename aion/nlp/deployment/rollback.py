"""
AION Rollback Manager - Version rollback for deployed systems.

Supports rolling back deployed systems to previous versions
with safety checks.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import DeployedSystem, DeploymentStatus

if TYPE_CHECKING:
    from aion.nlp.deployment.registry import DeploymentRegistry

logger = structlog.get_logger(__name__)


class RollbackManager:
    """
    Manages version rollback for deployed systems.

    Provides:
    - Rollback to previous version
    - Rollback to specific version
    - Safety checks before rollback
    """

    def __init__(self, registry: DeploymentRegistry) -> None:
        self._registry = registry

    async def rollback(
        self,
        system_id: str,
        target_version: Optional[int] = None,
    ) -> bool:
        """
        Rollback a deployed system.

        Args:
            system_id: System to rollback
            target_version: Specific version (None = previous)

        Returns:
            True if rollback succeeded
        """
        system = self._registry.get(system_id)
        if not system:
            logger.warning("Rollback failed: system not found", id=system_id)
            return False

        if not system.deployment_history:
            logger.warning("Rollback failed: no history", name=system.name)
            return False

        # Determine target version
        if target_version is None:
            target_version = system.version - 1

        if target_version < 1:
            logger.warning("Rollback failed: cannot rollback below v1", name=system.name)
            return False

        # Find the target deployment record
        target_record = None
        for record in system.deployment_history:
            if record.version == target_version:
                target_record = record
                break

        if not target_record:
            logger.warning(
                "Rollback failed: version not found",
                name=system.name,
                target_version=target_version,
            )
            return False

        if not target_record.rollback_safe:
            logger.warning(
                "Rollback warning: target version marked as unsafe",
                name=system.name,
                target_version=target_version,
            )

        # Perform rollback
        try:
            system.status = DeploymentStatus.ROLLED_BACK
            system.version = target_version
            self._registry.update(system)

            logger.info(
                "System rolled back",
                name=system.name,
                from_version=system.version + 1,
                to_version=target_version,
            )
            return True

        except Exception as e:
            logger.error("Rollback failed", name=system.name, error=str(e))
            return False

    def can_rollback(self, system_id: str) -> bool:
        """Check if a system can be rolled back."""
        system = self._registry.get(system_id)
        if not system:
            return False
        return system.version > 1 and len(system.deployment_history) > 1
