"""
AION Secrets Manager

Secure secret storage with encryption and rotation.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets as py_secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.security.types import Secret, SecretType

logger = structlog.get_logger(__name__)


@dataclass
class EncryptionKey:
    """An encryption key for secrets."""

    key_id: str
    key_data: bytes
    algorithm: str = "AES-256-GCM"
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_primary: bool = True


class SecretManager:
    """
    Secure secret management.

    Features:
    - Encrypted secret storage
    - Key rotation
    - Version control
    - Access logging
    - Automatic rotation scheduling
    """

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        auto_rotate_days: int = 90,
    ):
        # Master key for encrypting secrets
        self._master_key = master_key or self._generate_key()

        # Encryption keys (for key rotation)
        self._encryption_keys: Dict[str, EncryptionKey] = {}
        self._primary_key_id: Optional[str] = None

        # Secret storage
        self._secrets: Dict[str, Secret] = {}
        self._secrets_by_name: Dict[str, str] = {}  # name -> id

        # Auto-rotation config
        self.auto_rotate_days = auto_rotate_days

        # Access log
        self._access_log: List[Dict[str, Any]] = []

        # Rotation handlers
        self._rotation_handlers: Dict[str, Callable] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the secret manager."""
        if self._initialized:
            return

        logger.info("Initializing Secret Manager")

        # Create primary encryption key
        self._create_encryption_key(primary=True)

        self._initialized = True

    # =========================================================================
    # Secret Management
    # =========================================================================

    async def create_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.GENERIC,
        description: str = "",
        tenant_id: Optional[str] = None,
        environment: str = "production",
        rotation_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Secret:
        """Create a new secret."""
        if name in self._secrets_by_name:
            raise ValueError(f"Secret already exists: {name}")

        # Encrypt the value
        encrypted_value = self._encrypt(value.encode())

        # Calculate rotation schedule
        rotation_interval = rotation_days or self.auto_rotate_days
        next_rotation = datetime.now() + timedelta(days=rotation_interval)

        secret = Secret(
            name=name,
            description=description,
            secret_type=secret_type,
            encrypted_value=encrypted_value,
            encryption_key_id=self._primary_key_id,
            tenant_id=tenant_id,
            environment=environment,
            rotation_enabled=True,
            rotation_interval_days=rotation_interval,
            next_rotation_at=next_rotation,
            metadata=metadata or {},
        )

        self._secrets[secret.id] = secret
        self._secrets_by_name[name] = secret.id

        logger.info("Secret created", name=name, type=secret_type.value)

        return secret

    async def get_secret(
        self,
        name: str,
        version: Optional[int] = None,
        accessor: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a secret value.

        Returns the decrypted value or None if not found.
        """
        secret_id = self._secrets_by_name.get(name)
        if not secret_id:
            return None

        secret = self._secrets.get(secret_id)
        if not secret:
            return None

        # Check expiration
        if secret.expires_at and datetime.now() > secret.expires_at:
            logger.warning("Attempted access to expired secret", name=name)
            return None

        # Decrypt
        try:
            value = self._decrypt(secret.encrypted_value)

            # Log access
            self._log_access(name, accessor, "read")

            return value.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}", name=name)
            return None

    async def update_secret(
        self,
        name: str,
        new_value: str,
        updater: Optional[str] = None,
    ) -> bool:
        """Update a secret value."""
        secret_id = self._secrets_by_name.get(name)
        if not secret_id:
            return False

        secret = self._secrets.get(secret_id)
        if not secret:
            return False

        # Encrypt new value
        encrypted_value = self._encrypt(new_value.encode())

        # Update secret
        secret.encrypted_value = encrypted_value
        secret.encryption_key_id = self._primary_key_id
        secret.version += 1
        secret.updated_at = datetime.now()
        secret.last_rotated_at = datetime.now()
        secret.next_rotation_at = datetime.now() + timedelta(
            days=secret.rotation_interval_days
        )

        # Log update
        self._log_access(name, updater, "update")

        logger.info("Secret updated", name=name, version=secret.version)

        return True

    async def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        secret_id = self._secrets_by_name.pop(name, None)
        if not secret_id:
            return False

        self._secrets.pop(secret_id, None)

        logger.info("Secret deleted", name=name)

        return True

    async def list_secrets(
        self,
        tenant_id: Optional[str] = None,
        secret_type: Optional[SecretType] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List secrets (metadata only, not values)."""
        results = []

        for secret in self._secrets.values():
            if tenant_id and secret.tenant_id != tenant_id:
                continue
            if secret_type and secret.secret_type != secret_type:
                continue
            if environment and secret.environment != environment:
                continue

            results.append({
                "id": secret.id,
                "name": secret.name,
                "type": secret.secret_type.value,
                "environment": secret.environment,
                "version": secret.version,
                "created_at": secret.created_at.isoformat(),
                "next_rotation_at": (
                    secret.next_rotation_at.isoformat()
                    if secret.next_rotation_at
                    else None
                ),
            })

        return results

    # =========================================================================
    # Rotation
    # =========================================================================

    async def rotate_secret(
        self,
        name: str,
        new_value: Optional[str] = None,
    ) -> bool:
        """
        Rotate a secret.

        If new_value is not provided and a rotation handler is registered,
        the handler will be called to generate a new value.
        """
        secret_id = self._secrets_by_name.get(name)
        if not secret_id:
            return False

        secret = self._secrets.get(secret_id)
        if not secret:
            return False

        # Get new value
        if new_value is None:
            handler = self._rotation_handlers.get(secret.secret_type.value)
            if handler:
                try:
                    new_value = await handler(secret)
                except Exception as e:
                    logger.error(f"Rotation handler failed: {e}", name=name)
                    return False
            else:
                # Auto-generate for some types
                if secret.secret_type in (SecretType.API_KEY, SecretType.WEBHOOK_SECRET):
                    new_value = py_secrets.token_urlsafe(32)
                else:
                    logger.warning(
                        "No rotation handler and cannot auto-generate",
                        name=name,
                        type=secret.secret_type.value,
                    )
                    return False

        return await self.update_secret(name, new_value, updater="rotation")

    async def check_rotations(self) -> List[str]:
        """Check for secrets that need rotation."""
        now = datetime.now()
        needs_rotation = []

        for secret in self._secrets.values():
            if secret.rotation_enabled and secret.next_rotation_at:
                if now > secret.next_rotation_at:
                    needs_rotation.append(secret.name)

        return needs_rotation

    def register_rotation_handler(
        self,
        secret_type: str,
        handler: Callable,
    ) -> None:
        """Register a rotation handler for a secret type."""
        self._rotation_handlers[secret_type] = handler

    # =========================================================================
    # Encryption
    # =========================================================================

    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return os.urandom(32)

    def _create_encryption_key(self, primary: bool = False) -> EncryptionKey:
        """Create a new encryption key."""
        key = EncryptionKey(
            key_id=py_secrets.token_hex(8),
            key_data=self._generate_key(),
            is_primary=primary,
        )

        self._encryption_keys[key.key_id] = key

        if primary:
            # Demote current primary
            if self._primary_key_id:
                old_primary = self._encryption_keys.get(self._primary_key_id)
                if old_primary:
                    old_primary.is_primary = False

            self._primary_key_id = key.key_id

        return key

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using the primary key."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            key = self._encryption_keys[self._primary_key_id].key_data
            nonce = os.urandom(12)
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, data, None)

            # Format: nonce + ciphertext
            return nonce + ciphertext

        except ImportError:
            # Fallback: simple XOR (NOT SECURE, for development only)
            logger.warning("cryptography not installed, using insecure fallback")
            return self._xor_encrypt(data, self._master_key)

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            # Try current primary key first
            key = self._encryption_keys[self._primary_key_id].key_data
            nonce = data[:12]
            ciphertext = data[12:]

            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, None)

        except ImportError:
            return self._xor_decrypt(data, self._master_key)
        except Exception:
            # Try other keys (for rotation scenarios)
            for key_obj in self._encryption_keys.values():
                if key_obj.key_id == self._primary_key_id:
                    continue
                try:
                    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

                    nonce = data[:12]
                    ciphertext = data[12:]
                    aesgcm = AESGCM(key_obj.key_data)
                    return aesgcm.decrypt(nonce, ciphertext, None)
                except Exception:
                    continue
            raise

    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Insecure XOR encryption (fallback only)."""
        return bytes(a ^ key[i % len(key)] for i, a in enumerate(data))

    def _xor_decrypt(self, data: bytes, key: bytes) -> bytes:
        """Insecure XOR decryption (fallback only)."""
        return self._xor_encrypt(data, key)

    # =========================================================================
    # Key Rotation
    # =========================================================================

    async def rotate_encryption_key(self) -> str:
        """Rotate the encryption key and re-encrypt all secrets."""
        # Create new key
        new_key = self._create_encryption_key(primary=True)

        # Re-encrypt all secrets
        for secret in self._secrets.values():
            try:
                # Decrypt with old key
                value = self._decrypt(secret.encrypted_value)

                # Encrypt with new key
                secret.encrypted_value = self._encrypt(value)
                secret.encryption_key_id = new_key.key_id

            except Exception as e:
                logger.error(f"Failed to re-encrypt secret: {e}", name=secret.name)

        logger.info("Encryption key rotated", new_key_id=new_key.key_id)

        return new_key.key_id

    # =========================================================================
    # Access Logging
    # =========================================================================

    def _log_access(
        self,
        secret_name: str,
        accessor: Optional[str],
        operation: str,
    ) -> None:
        """Log secret access."""
        self._access_log.append({
            "timestamp": datetime.now().isoformat(),
            "secret_name": secret_name,
            "accessor": accessor,
            "operation": operation,
        })

        # Limit log size
        if len(self._access_log) > 10000:
            self._access_log = self._access_log[-5000:]

    def get_access_log(
        self,
        secret_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get secret access log."""
        logs = self._access_log

        if secret_name:
            logs = [l for l in logs if l["secret_name"] == secret_name]

        return logs[-limit:]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get secret manager statistics."""
        secrets_needing_rotation = sum(
            1
            for s in self._secrets.values()
            if s.rotation_enabled
            and s.next_rotation_at
            and datetime.now() > s.next_rotation_at
        )

        by_type = {}
        for s in self._secrets.values():
            t = s.secret_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_secrets": len(self._secrets),
            "by_type": by_type,
            "encryption_keys": len(self._encryption_keys),
            "secrets_needing_rotation": secrets_needing_rotation,
            "access_log_size": len(self._access_log),
        }
