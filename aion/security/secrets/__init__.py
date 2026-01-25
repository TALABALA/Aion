"""
AION Secrets Module

Secure secret storage with encryption and rotation.
"""

from aion.security.secrets.manager import SecretManager, EncryptionKey

__all__ = [
    "SecretManager",
    "EncryptionKey",
]
