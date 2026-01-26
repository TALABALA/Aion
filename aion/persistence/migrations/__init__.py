"""
AION Database Migrations

Automatic schema evolution with:
- Version tracking
- Up/down migrations
- Rollback support
- Checksum validation
"""

from aion.persistence.migrations.runner import MigrationRunner

__all__ = ["MigrationRunner"]
