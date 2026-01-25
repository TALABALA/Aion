"""
AION Authorization Module

Role-based and policy-based access control.
"""

from aion.security.authorization.service import (
    AuthorizationService,
    BUILTIN_ROLES,
)

__all__ = [
    "AuthorizationService",
    "BUILTIN_ROLES",
]
