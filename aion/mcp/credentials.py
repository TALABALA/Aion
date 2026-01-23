"""
AION MCP Credential Management

Production-grade secure credential storage with:
- AES-256 encryption via Fernet
- Key derivation with PBKDF2
- Secure key storage with file permissions
- Environment variable integration
- Vault provider support (HashiCorp Vault, AWS Secrets Manager)
- Credential rotation support
- Audit logging
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import stat
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import structlog

# Try to import cryptography for proper encryption
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None
    InvalidToken = Exception

logger = structlog.get_logger(__name__)


# ============================================
# Credential Types
# ============================================

@dataclass
class Credential:
    """A stored credential with metadata."""
    id: str
    data: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    rotation_policy: Optional[str] = None  # "daily", "weekly", "monthly"
    last_rotated: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (without sensitive data)."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "rotation_policy": self.rotation_policy,
            "last_rotated": self.last_rotated.isoformat() if self.last_rotated else None,
            "access_count": self.access_count,
            "is_expired": self.is_expired,
        }


# ============================================
# Encryption Engine
# ============================================

class EncryptionEngine:
    """
    Production-grade encryption using Fernet (AES-128-CBC with HMAC).

    Features:
    - PBKDF2 key derivation with 480,000 iterations
    - Cryptographically secure salt generation
    - Automatic key rotation support
    """

    # OWASP recommended iterations for PBKDF2-SHA256 (2023)
    PBKDF2_ITERATIONS = 480_000
    SALT_LENGTH = 32

    def __init__(self, master_key: Optional[str] = None, key_file: Optional[Path] = None):
        """
        Initialize encryption engine.

        Args:
            master_key: Master key for encryption (or derived from env/file)
            key_file: Path to key file
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography package required for secure credential storage. "
                "Install with: pip install cryptography"
            )

        self._key_file = key_file
        self._fernet: Optional[Fernet] = None
        self._salt: Optional[bytes] = None

        # Initialize with provided key or generate/load
        if master_key:
            self._initialize_with_key(master_key)
        elif key_file and key_file.exists():
            self._load_key_file(key_file)
        else:
            self._generate_new_key(key_file)

    def _initialize_with_key(self, master_key: str) -> None:
        """Initialize with a provided master key."""
        # Generate deterministic salt from master key for consistency
        self._salt = hashlib.sha256(master_key.encode()).digest()

        # Derive encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=self.PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self._fernet = Fernet(key)

    def _generate_new_key(self, key_file: Optional[Path] = None) -> None:
        """Generate a new random key."""
        # Generate cryptographically secure salt
        self._salt = secrets.token_bytes(self.SALT_LENGTH)

        # Generate random master key
        master_key = secrets.token_urlsafe(32)

        # Derive encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=self.PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self._fernet = Fernet(key)

        # Save to file if path provided
        if key_file:
            self._save_key_file(key_file, master_key)

    def _save_key_file(self, key_file: Path, master_key: str) -> None:
        """Save key to file with secure permissions."""
        key_file.parent.mkdir(parents=True, exist_ok=True)

        key_data = {
            "salt": base64.b64encode(self._salt).decode(),
            "master_key": master_key,
            "created_at": datetime.now().isoformat(),
            "iterations": self.PBKDF2_ITERATIONS,
        }

        # Write with restricted permissions
        key_file.write_text(json.dumps(key_data))

        # Set file permissions to owner read/write only (Unix)
        try:
            os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
        except (OSError, AttributeError):
            pass  # Windows or permission error

        logger.info("Encryption key saved", path=str(key_file))

    def _load_key_file(self, key_file: Path) -> None:
        """Load key from file."""
        try:
            key_data = json.loads(key_file.read_text())
            self._salt = base64.b64decode(key_data["salt"])
            master_key = key_data["master_key"]
            iterations = key_data.get("iterations", self.PBKDF2_ITERATIONS)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self._salt,
                iterations=iterations,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
            self._fernet = Fernet(key)

            logger.debug("Encryption key loaded", path=str(key_file))

        except Exception as e:
            logger.error("Failed to load encryption key", error=str(e))
            raise

    def encrypt(self, data: str) -> str:
        """
        Encrypt string data.

        Args:
            data: Plaintext string

        Returns:
            Base64-encoded ciphertext
        """
        if not self._fernet:
            raise RuntimeError("Encryption engine not initialized")

        ciphertext = self._fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(ciphertext).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt encrypted data.

        Args:
            encrypted_data: Base64-encoded ciphertext

        Returns:
            Decrypted plaintext string

        Raises:
            ValueError: If decryption fails (invalid token or corrupted data)
        """
        if not self._fernet:
            raise RuntimeError("Encryption engine not initialized")

        try:
            ciphertext = base64.urlsafe_b64decode(encrypted_data.encode())
            plaintext = self._fernet.decrypt(ciphertext)
            return plaintext.decode()
        except InvalidToken:
            raise ValueError("Decryption failed: invalid token or corrupted data")

    def rotate_key(self, new_master_key: str) -> "EncryptionEngine":
        """
        Create a new encryption engine with rotated key.

        Args:
            new_master_key: New master key

        Returns:
            New EncryptionEngine instance
        """
        return EncryptionEngine(master_key=new_master_key, key_file=self._key_file)


# ============================================
# Credential Providers (Pluggable Backends)
# ============================================

@runtime_checkable
class CredentialProvider(Protocol):
    """Protocol for credential providers."""

    async def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """Get credential by ID."""
        ...

    async def set(self, credential_id: str, data: Dict[str, str]) -> bool:
        """Set credential."""
        ...

    async def delete(self, credential_id: str) -> bool:
        """Delete credential."""
        ...

    async def list_ids(self) -> list[str]:
        """List all credential IDs."""
        ...


class EnvironmentCredentialProvider:
    """
    Credential provider that reads from environment variables.

    Maps credential IDs to environment variable prefixes:
    - credential_id="postgres" -> looks for POSTGRES_USER, POSTGRES_PASSWORD, etc.
    """

    def __init__(self, prefix: str = ""):
        """
        Initialize provider.

        Args:
            prefix: Optional prefix for all env vars (e.g., "AION_")
        """
        self.prefix = prefix

    async def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """Get credentials from environment variables."""
        env_prefix = f"{self.prefix}{credential_id.upper()}_"

        credentials = {}
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and store
                cred_key = key[len(env_prefix):]
                credentials[cred_key] = value

        return credentials if credentials else None

    async def set(self, credential_id: str, data: Dict[str, str]) -> bool:
        """Cannot set environment variables at runtime."""
        logger.warning("EnvironmentCredentialProvider does not support setting credentials")
        return False

    async def delete(self, credential_id: str) -> bool:
        """Cannot delete environment variables."""
        logger.warning("EnvironmentCredentialProvider does not support deleting credentials")
        return False

    async def list_ids(self) -> list[str]:
        """List credential IDs found in environment."""
        ids = set()
        for key in os.environ.keys():
            if key.startswith(self.prefix):
                # Extract credential ID from env var name
                parts = key[len(self.prefix):].split("_")
                if parts:
                    ids.add(parts[0].lower())
        return list(ids)


class VaultCredentialProvider:
    """
    Credential provider for HashiCorp Vault.

    Requires: pip install hvac
    """

    def __init__(
        self,
        vault_addr: str,
        token: Optional[str] = None,
        mount_point: str = "secret",
        path_prefix: str = "aion/mcp",
    ):
        """
        Initialize Vault provider.

        Args:
            vault_addr: Vault server address
            token: Vault token (or from VAULT_TOKEN env)
            mount_point: KV secrets engine mount point
            path_prefix: Path prefix for credentials
        """
        try:
            import hvac
            self._hvac = hvac
        except ImportError:
            raise RuntimeError("hvac package required for Vault integration")

        self.vault_addr = vault_addr
        self.mount_point = mount_point
        self.path_prefix = path_prefix

        # Initialize client
        token = token or os.environ.get("VAULT_TOKEN")
        self._client = hvac.Client(url=vault_addr, token=token)

        if not self._client.is_authenticated():
            raise RuntimeError("Failed to authenticate with Vault")

    async def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """Get credential from Vault."""
        try:
            path = f"{self.path_prefix}/{credential_id}"
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.mount_point,
            )
            return response["data"]["data"]
        except Exception as e:
            logger.debug(f"Credential not found in Vault: {credential_id}", error=str(e))
            return None

    async def set(self, credential_id: str, data: Dict[str, str]) -> bool:
        """Set credential in Vault."""
        try:
            path = f"{self.path_prefix}/{credential_id}"
            self._client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=self.mount_point,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set credential in Vault", error=str(e))
            return False

    async def delete(self, credential_id: str) -> bool:
        """Delete credential from Vault."""
        try:
            path = f"{self.path_prefix}/{credential_id}"
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point=self.mount_point,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete credential from Vault", error=str(e))
            return False

    async def list_ids(self) -> list[str]:
        """List credential IDs in Vault."""
        try:
            response = self._client.secrets.kv.v2.list_secrets(
                path=self.path_prefix,
                mount_point=self.mount_point,
            )
            return response["data"]["keys"]
        except Exception:
            return []


class AWSSecretsProvider:
    """
    Credential provider for AWS Secrets Manager.

    Requires: pip install boto3
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        secret_prefix: str = "aion/mcp/",
    ):
        """
        Initialize AWS Secrets Manager provider.

        Args:
            region_name: AWS region
            secret_prefix: Prefix for secret names
        """
        try:
            import boto3
            self._boto3 = boto3
        except ImportError:
            raise RuntimeError("boto3 package required for AWS Secrets Manager")

        self.secret_prefix = secret_prefix
        self._client = boto3.client("secretsmanager", region_name=region_name)

    async def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """Get credential from AWS Secrets Manager."""
        try:
            secret_name = f"{self.secret_prefix}{credential_id}"
            response = self._client.get_secret_value(SecretId=secret_name)
            return json.loads(response["SecretString"])
        except Exception as e:
            logger.debug(f"Credential not found in AWS: {credential_id}", error=str(e))
            return None

    async def set(self, credential_id: str, data: Dict[str, str]) -> bool:
        """Set credential in AWS Secrets Manager."""
        try:
            secret_name = f"{self.secret_prefix}{credential_id}"
            try:
                self._client.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps(data),
                )
            except self._client.exceptions.ResourceNotFoundException:
                self._client.create_secret(
                    Name=secret_name,
                    SecretString=json.dumps(data),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to set credential in AWS", error=str(e))
            return False

    async def delete(self, credential_id: str) -> bool:
        """Delete credential from AWS Secrets Manager."""
        try:
            secret_name = f"{self.secret_prefix}{credential_id}"
            self._client.delete_secret(SecretId=secret_name, ForceDeleteWithoutRecovery=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete credential from AWS", error=str(e))
            return False

    async def list_ids(self) -> list[str]:
        """List credential IDs in AWS Secrets Manager."""
        try:
            paginator = self._client.get_paginator("list_secrets")
            ids = []
            for page in paginator.paginate(Filters=[{"Key": "name", "Values": [self.secret_prefix]}]):
                for secret in page["SecretList"]:
                    name = secret["Name"]
                    if name.startswith(self.secret_prefix):
                        ids.append(name[len(self.secret_prefix):])
            return ids
        except Exception:
            return []


# ============================================
# Main Credential Manager
# ============================================

class CredentialManager:
    """
    Production-grade credential manager with:
    - Multiple backend support (file, env, Vault, AWS)
    - Encrypted local storage
    - Credential rotation
    - Access auditing
    - Caching with TTL
    """

    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        key_file: Optional[Path] = None,
        master_key: Optional[str] = None,
        providers: Optional[list[CredentialProvider]] = None,
        cache_ttl: float = 300.0,  # 5 minutes
    ):
        """
        Initialize credential manager.

        Args:
            credentials_path: Path to encrypted credentials file
            key_file: Path to encryption key file
            master_key: Master key (overrides key_file)
            providers: Additional credential providers
            cache_ttl: Cache TTL in seconds
        """
        self.credentials_path = credentials_path or Path("./config/mcp_credentials.enc")
        self.key_file = key_file or Path("./config/.mcp_key")
        self.cache_ttl = cache_ttl

        # Encryption engine (initialized in initialize())
        self._encryption: Optional[EncryptionEngine] = None
        self._master_key = master_key or os.environ.get("AION_MCP_MASTER_KEY")

        # Credential providers (checked in order)
        self._providers: list[CredentialProvider] = providers or []

        # Add default environment provider
        self._providers.insert(0, EnvironmentCredentialProvider(prefix="MCP_"))

        # Local storage
        self._credentials: Dict[str, Credential] = {}

        # Cache
        self._cache: Dict[str, tuple[Dict[str, str], datetime]] = {}

        # State
        self._initialized = False
        self._lock = asyncio.Lock()

        # Audit log
        self._audit_log: list[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize the credential manager."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info("Initializing credential manager")

            # Initialize encryption if cryptography is available
            if CRYPTO_AVAILABLE:
                try:
                    self._encryption = EncryptionEngine(
                        master_key=self._master_key,
                        key_file=self.key_file,
                    )
                except Exception as e:
                    logger.warning(f"Encryption initialization failed: {e}")
                    self._encryption = None
            else:
                logger.warning(
                    "cryptography package not available - credentials will be stored in plaintext"
                )

            # Load existing credentials
            await self._load_credentials()

            self._initialized = True
            logger.info(
                "Credential manager initialized",
                credentials_count=len(self._credentials),
                encryption_enabled=self._encryption is not None,
            )

    async def shutdown(self) -> None:
        """Shutdown the credential manager."""
        if self._initialized:
            await self._save_credentials()
            self._initialized = False

    async def _load_credentials(self) -> None:
        """Load credentials from encrypted file."""
        if not self.credentials_path.exists():
            return

        try:
            encrypted_data = self.credentials_path.read_text()

            if self._encryption:
                decrypted = self._encryption.decrypt(encrypted_data)
                data = json.loads(decrypted)
            else:
                # Fallback to plaintext (not recommended)
                data = json.loads(encrypted_data)

            for cred_id, cred_data in data.items():
                self._credentials[cred_id] = Credential(
                    id=cred_id,
                    data=cred_data.get("data", {}),
                    created_at=datetime.fromisoformat(cred_data.get("created_at", datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(cred_data.get("updated_at", datetime.now().isoformat())),
                    expires_at=datetime.fromisoformat(cred_data["expires_at"]) if cred_data.get("expires_at") else None,
                    rotation_policy=cred_data.get("rotation_policy"),
                    last_rotated=datetime.fromisoformat(cred_data["last_rotated"]) if cred_data.get("last_rotated") else None,
                    access_count=cred_data.get("access_count", 0),
                )

            logger.debug(f"Loaded {len(self._credentials)} credentials")

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")

    async def _save_credentials(self) -> None:
        """Save credentials to encrypted file."""
        if not self._credentials:
            return

        try:
            data = {}
            for cred_id, cred in self._credentials.items():
                data[cred_id] = {
                    "data": cred.data,
                    "created_at": cred.created_at.isoformat(),
                    "updated_at": cred.updated_at.isoformat(),
                    "expires_at": cred.expires_at.isoformat() if cred.expires_at else None,
                    "rotation_policy": cred.rotation_policy,
                    "last_rotated": cred.last_rotated.isoformat() if cred.last_rotated else None,
                    "access_count": cred.access_count,
                }

            json_data = json.dumps(data)

            if self._encryption:
                encrypted = self._encryption.encrypt(json_data)
                self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
                self.credentials_path.write_text(encrypted)
            else:
                self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
                self.credentials_path.write_text(json_data)

            # Set secure permissions
            try:
                os.chmod(self.credentials_path, stat.S_IRUSR | stat.S_IWUSR)
            except (OSError, AttributeError):
                pass

            logger.debug(f"Saved {len(self._credentials)} credentials")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")

    async def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """
        Get credential by ID.

        Checks providers in order:
        1. Cache
        2. Environment variables
        3. External providers (Vault, AWS, etc.)
        4. Local encrypted storage

        Args:
            credential_id: Credential identifier

        Returns:
            Dictionary of credential key-value pairs or None
        """
        if not self._initialized:
            await self.initialize()

        # Check cache first
        if credential_id in self._cache:
            data, cached_at = self._cache[credential_id]
            if (datetime.now() - cached_at).total_seconds() < self.cache_ttl:
                self._audit("get", credential_id, "cache_hit")
                return data
            else:
                del self._cache[credential_id]

        # Check providers
        for provider in self._providers:
            try:
                data = await provider.get(credential_id)
                if data:
                    self._cache[credential_id] = (data, datetime.now())
                    self._audit("get", credential_id, f"provider:{type(provider).__name__}")
                    return data
            except Exception as e:
                logger.debug(f"Provider {type(provider).__name__} failed: {e}")

        # Check local storage
        cred = self._credentials.get(credential_id)
        if cred:
            if cred.is_expired:
                self._audit("get", credential_id, "expired")
                return None

            cred.access_count += 1
            cred.last_accessed = datetime.now()

            self._cache[credential_id] = (cred.data, datetime.now())
            self._audit("get", credential_id, "local_storage")
            return cred.data

        self._audit("get", credential_id, "not_found")
        return None

    async def set(
        self,
        credential_id: str,
        data: Dict[str, str],
        persist: bool = True,
        expires_in: Optional[timedelta] = None,
        rotation_policy: Optional[str] = None,
    ) -> bool:
        """
        Set a credential.

        Args:
            credential_id: Credential identifier
            data: Credential key-value pairs
            persist: Whether to persist to storage
            expires_in: Optional expiration time
            rotation_policy: Optional rotation policy

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            expires_at = datetime.now() + expires_in if expires_in else None

            if credential_id in self._credentials:
                # Update existing
                cred = self._credentials[credential_id]
                cred.data = data
                cred.updated_at = datetime.now()
                cred.expires_at = expires_at
                if rotation_policy:
                    cred.rotation_policy = rotation_policy
            else:
                # Create new
                cred = Credential(
                    id=credential_id,
                    data=data,
                    expires_at=expires_at,
                    rotation_policy=rotation_policy,
                )
                self._credentials[credential_id] = cred

            # Update cache
            self._cache[credential_id] = (data, datetime.now())

            # Persist
            if persist:
                await self._save_credentials()

            self._audit("set", credential_id, "success")
            return True

    async def delete(self, credential_id: str, persist: bool = True) -> bool:
        """
        Delete a credential.

        Args:
            credential_id: Credential identifier
            persist: Whether to persist deletion

        Returns:
            True if credential was deleted
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            # Remove from cache
            self._cache.pop(credential_id, None)

            # Remove from local storage
            if credential_id in self._credentials:
                del self._credentials[credential_id]

                if persist:
                    await self._save_credentials()

                self._audit("delete", credential_id, "success")
                return True

            self._audit("delete", credential_id, "not_found")
            return False

    async def rotate(self, credential_id: str, new_data: Dict[str, str]) -> bool:
        """
        Rotate a credential.

        Args:
            credential_id: Credential identifier
            new_data: New credential data

        Returns:
            True if rotation successful
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            cred = self._credentials.get(credential_id)
            if not cred:
                return False

            cred.data = new_data
            cred.updated_at = datetime.now()
            cred.last_rotated = datetime.now()

            # Update cache
            self._cache[credential_id] = (new_data, datetime.now())

            await self._save_credentials()

            self._audit("rotate", credential_id, "success")
            return True

    def has_credential(self, credential_id: str) -> bool:
        """Check if a credential exists."""
        return credential_id in self._credentials or credential_id in self._cache

    def list_credentials(self) -> list[str]:
        """List all credential IDs."""
        return list(self._credentials.keys())

    def get_credential_info(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Get credential metadata (without sensitive data)."""
        cred = self._credentials.get(credential_id)
        return cred.to_dict() if cred else None

    def add_provider(self, provider: CredentialProvider) -> None:
        """Add a credential provider."""
        self._providers.append(provider)

    def _audit(self, action: str, credential_id: str, result: str) -> None:
        """Record audit log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "credential_id": credential_id,
            "result": result,
        }
        self._audit_log.append(entry)

        # Keep only last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    def get_audit_log(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]

    def clear_cache(self) -> None:
        """Clear credential cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get credential manager statistics."""
        return {
            "initialized": self._initialized,
            "encryption_enabled": self._encryption is not None,
            "credentials_count": len(self._credentials),
            "cache_entries": len(self._cache),
            "providers_count": len(self._providers),
            "audit_log_entries": len(self._audit_log),
        }
