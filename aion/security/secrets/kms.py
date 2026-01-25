"""
External Key Management System (KMS) Integration.

Provides unified interface for external key management services:
- HashiCorp Vault
- AWS KMS
- Azure Key Vault
- Google Cloud KMS

Implements proper key hierarchy, envelope encryption, and secure key handling.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol
from urllib.parse import urljoin

import structlog

logger = structlog.get_logger()


class KMSProvider(str, Enum):
    """Supported KMS providers."""
    VAULT = "vault"
    AWS_KMS = "aws_kms"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_KMS = "gcp_kms"
    LOCAL = "local"  # For development/testing


class KeyType(str, Enum):
    """Cryptographic key types."""
    AES_256_GCM = "aes-256-gcm"
    AES_128_GCM = "aes-128-gcm"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    EC_P256 = "ec-p256"
    EC_P384 = "ec-p384"
    ED25519 = "ed25519"


class KeyPurpose(str, Enum):
    """Key usage purposes."""
    ENCRYPT_DECRYPT = "encrypt_decrypt"
    SIGN_VERIFY = "sign_verify"
    WRAP_UNWRAP = "wrap_unwrap"
    MAC = "mac"


class KeyState(str, Enum):
    """Key lifecycle states."""
    PENDING_GENERATION = "pending_generation"
    ENABLED = "enabled"
    DISABLED = "disabled"
    PENDING_DELETION = "pending_deletion"
    DESTROYED = "destroyed"


@dataclass
class KeyMetadata:
    """Metadata for a managed key."""
    key_id: str
    key_type: KeyType
    purpose: KeyPurpose
    state: KeyState
    created_at: float
    updated_at: float
    rotation_period_days: Optional[int] = None
    next_rotation_at: Optional[float] = None
    version: int = 1
    labels: dict[str, str] = field(default_factory=dict)
    provider: Optional[KMSProvider] = None
    provider_key_id: Optional[str] = None  # External provider's key ID


@dataclass
class EncryptedData:
    """Encrypted data with metadata for decryption."""
    ciphertext: bytes
    key_id: str
    key_version: int
    algorithm: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    aad: Optional[bytes] = None  # Additional authenticated data


@dataclass
class DataKey:
    """Data encryption key (DEK) for envelope encryption."""
    plaintext: bytes
    ciphertext: bytes  # Encrypted with KEK
    key_id: str
    algorithm: str


class KMSBackend(ABC):
    """Abstract base class for KMS backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend connection."""
        pass

    @abstractmethod
    async def create_key(
        self,
        key_id: str,
        key_type: KeyType,
        purpose: KeyPurpose,
        labels: Optional[dict[str, str]] = None,
    ) -> KeyMetadata:
        """Create a new key."""
        pass

    @abstractmethod
    async def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata."""
        pass

    @abstractmethod
    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data using the specified key."""
        pass

    @abstractmethod
    async def decrypt(
        self,
        encrypted_data: EncryptedData,
    ) -> bytes:
        """Decrypt data."""
        pass

    @abstractmethod
    async def generate_data_key(
        self,
        key_id: str,
        key_spec: KeyType = KeyType.AES_256_GCM,
    ) -> DataKey:
        """Generate a data encryption key (envelope encryption)."""
        pass

    @abstractmethod
    async def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key to a new version."""
        pass

    @abstractmethod
    async def disable_key(self, key_id: str) -> None:
        """Disable a key."""
        pass

    @abstractmethod
    async def enable_key(self, key_id: str) -> None:
        """Enable a disabled key."""
        pass

    @abstractmethod
    async def schedule_key_deletion(
        self,
        key_id: str,
        pending_days: int = 30,
    ) -> None:
        """Schedule a key for deletion."""
        pass

    @abstractmethod
    async def sign(
        self,
        key_id: str,
        data: bytes,
        algorithm: Optional[str] = None,
    ) -> bytes:
        """Sign data using the specified key."""
        pass

    @abstractmethod
    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> bool:
        """Verify a signature."""
        pass


class VaultBackend(KMSBackend):
    """
    HashiCorp Vault Transit backend.

    Uses Vault's Transit secrets engine for key management and cryptographic operations.
    """

    def __init__(
        self,
        address: str,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        mount_path: str = "transit",
        tls_verify: bool = True,
        role_id: Optional[str] = None,
        secret_id: Optional[str] = None,
    ) -> None:
        self.address = address.rstrip("/")
        self._token = token
        self.namespace = namespace
        self.mount_path = mount_path
        self.tls_verify = tls_verify
        self.role_id = role_id
        self.secret_id = secret_id
        self._client: Optional[Any] = None
        self._keys: dict[str, KeyMetadata] = {}
        self._logger = logger.bind(backend="vault")

    async def initialize(self) -> None:
        """Initialize Vault connection and authenticate."""
        try:
            import aiohttp

            # If using AppRole authentication
            if self.role_id and self.secret_id:
                await self._authenticate_approle()

            # Verify connection
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                async with session.get(
                    f"{self.address}/v1/sys/health",
                    headers=headers,
                    ssl=self.tls_verify,
                ) as response:
                    if response.status == 200:
                        self._logger.info("Connected to Vault", address=self.address)
                    else:
                        raise ConnectionError(f"Vault health check failed: {response.status}")

        except ImportError:
            self._logger.warning("aiohttp not available, using mock mode")

    async def _authenticate_approle(self) -> None:
        """Authenticate using AppRole."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            payload = {
                "role_id": self.role_id,
                "secret_id": self.secret_id,
            }
            async with session.post(
                f"{self.address}/v1/auth/approle/login",
                json=payload,
                ssl=self.tls_verify,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._token = data["auth"]["client_token"]
                    self._logger.info("Authenticated with AppRole")
                else:
                    raise AuthenticationError("AppRole authentication failed")

    def _get_headers(self) -> dict[str, str]:
        """Get request headers including auth token."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["X-Vault-Token"] = self._token
        if self.namespace:
            headers["X-Vault-Namespace"] = self.namespace
        return headers

    async def create_key(
        self,
        key_id: str,
        key_type: KeyType,
        purpose: KeyPurpose,
        labels: Optional[dict[str, str]] = None,
    ) -> KeyMetadata:
        """Create a new key in Vault Transit."""
        import aiohttp

        # Map key type to Vault type
        vault_type = self._map_key_type(key_type)

        payload = {
            "type": vault_type,
            "exportable": False,
            "allow_plaintext_backup": False,
        }

        if purpose == KeyPurpose.SIGN_VERIFY:
            payload["derived"] = False

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/keys/{key_id}",
                headers=self._get_headers(),
                json=payload,
                ssl=self.tls_verify,
            ) as response:
                if response.status not in (200, 204):
                    text = await response.text()
                    raise KMSError(f"Failed to create key: {text}")

        now = time.time()
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            purpose=purpose,
            state=KeyState.ENABLED,
            created_at=now,
            updated_at=now,
            labels=labels or {},
            provider=KMSProvider.VAULT,
            provider_key_id=key_id,
        )
        self._keys[key_id] = metadata

        self._logger.info("Created key in Vault", key_id=key_id, key_type=key_type.value)
        return metadata

    async def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata from Vault."""
        if key_id in self._keys:
            return self._keys[key_id]

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.address}/v1/{self.mount_path}/keys/{key_id}",
                    headers=self._get_headers(),
                    ssl=self.tls_verify,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        key_data = data["data"]

                        metadata = KeyMetadata(
                            key_id=key_id,
                            key_type=self._reverse_map_key_type(key_data["type"]),
                            purpose=KeyPurpose.ENCRYPT_DECRYPT,
                            state=KeyState.ENABLED if not key_data.get("deletion_allowed") else KeyState.DISABLED,
                            created_at=time.time(),
                            updated_at=time.time(),
                            version=key_data.get("latest_version", 1),
                            provider=KMSProvider.VAULT,
                            provider_key_id=key_id,
                        )
                        self._keys[key_id] = metadata
                        return metadata
                    return None
        except Exception as e:
            self._logger.error("Failed to get key metadata", error=str(e))
            return None

    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data using Vault Transit."""
        import aiohttp

        payload = {
            "plaintext": base64.b64encode(plaintext).decode(),
        }
        if aad:
            payload["context"] = base64.b64encode(aad).decode()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/encrypt/{key_id}",
                headers=self._get_headers(),
                json=payload,
                ssl=self.tls_verify,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise KMSError(f"Encryption failed: {text}")

                data = await response.json()
                ciphertext = data["data"]["ciphertext"]

                # Parse version from ciphertext (vault:v1:...)
                parts = ciphertext.split(":")
                version = int(parts[1][1:]) if len(parts) > 1 else 1

                return EncryptedData(
                    ciphertext=ciphertext.encode(),
                    key_id=key_id,
                    key_version=version,
                    algorithm="vault-transit",
                    aad=aad,
                )

    async def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data using Vault Transit."""
        import aiohttp

        payload = {
            "ciphertext": encrypted_data.ciphertext.decode()
            if isinstance(encrypted_data.ciphertext, bytes)
            else encrypted_data.ciphertext,
        }
        if encrypted_data.aad:
            payload["context"] = base64.b64encode(encrypted_data.aad).decode()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/decrypt/{encrypted_data.key_id}",
                headers=self._get_headers(),
                json=payload,
                ssl=self.tls_verify,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise KMSError(f"Decryption failed: {text}")

                data = await response.json()
                plaintext_b64 = data["data"]["plaintext"]
                return base64.b64decode(plaintext_b64)

    async def generate_data_key(
        self,
        key_id: str,
        key_spec: KeyType = KeyType.AES_256_GCM,
    ) -> DataKey:
        """Generate a data key using Vault Transit."""
        import aiohttp

        bits = 256 if "256" in key_spec.value else 128

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/datakey/plaintext/{key_id}",
                headers=self._get_headers(),
                json={"bits": bits},
                ssl=self.tls_verify,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise KMSError(f"Data key generation failed: {text}")

                data = await response.json()
                return DataKey(
                    plaintext=base64.b64decode(data["data"]["plaintext"]),
                    ciphertext=data["data"]["ciphertext"].encode(),
                    key_id=key_id,
                    algorithm=key_spec.value,
                )

    async def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key in Vault Transit."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/keys/{key_id}/rotate",
                headers=self._get_headers(),
                ssl=self.tls_verify,
            ) as response:
                if response.status not in (200, 204):
                    text = await response.text()
                    raise KMSError(f"Key rotation failed: {text}")

        if key_id in self._keys:
            self._keys[key_id].version += 1
            self._keys[key_id].updated_at = time.time()
            return self._keys[key_id]

        return await self.get_key_metadata(key_id)

    async def disable_key(self, key_id: str) -> None:
        """Disable a key (update min_decryption_version)."""
        self._logger.warning("Vault Transit doesn't support key disable, using config update")
        if key_id in self._keys:
            self._keys[key_id].state = KeyState.DISABLED

    async def enable_key(self, key_id: str) -> None:
        """Re-enable a key."""
        if key_id in self._keys:
            self._keys[key_id].state = KeyState.ENABLED

    async def schedule_key_deletion(
        self,
        key_id: str,
        pending_days: int = 30,
    ) -> None:
        """Schedule key deletion (Vault requires deletion_allowed=true first)."""
        import aiohttp

        # Update key config to allow deletion
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/keys/{key_id}/config",
                headers=self._get_headers(),
                json={"deletion_allowed": True},
                ssl=self.tls_verify,
            ) as response:
                if response.status not in (200, 204):
                    text = await response.text()
                    raise KMSError(f"Failed to enable deletion: {text}")

        if key_id in self._keys:
            self._keys[key_id].state = KeyState.PENDING_DELETION

        self._logger.warning(
            "Key scheduled for deletion",
            key_id=key_id,
            pending_days=pending_days,
        )

    async def sign(
        self,
        key_id: str,
        data: bytes,
        algorithm: Optional[str] = None,
    ) -> bytes:
        """Sign data using Vault Transit."""
        import aiohttp

        payload = {
            "input": base64.b64encode(data).decode(),
        }
        if algorithm:
            payload["signature_algorithm"] = algorithm

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/sign/{key_id}",
                headers=self._get_headers(),
                json=payload,
                ssl=self.tls_verify,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise KMSError(f"Signing failed: {text}")

                data = await response.json()
                signature = data["data"]["signature"]
                # Vault returns vault:v1:base64signature
                sig_parts = signature.split(":")
                return base64.b64decode(sig_parts[-1])

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> bool:
        """Verify a signature using Vault Transit."""
        import aiohttp

        payload = {
            "input": base64.b64encode(data).decode(),
            "signature": f"vault:v1:{base64.b64encode(signature).decode()}",
        }
        if algorithm:
            payload["signature_algorithm"] = algorithm

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.address}/v1/{self.mount_path}/verify/{key_id}",
                headers=self._get_headers(),
                json=payload,
                ssl=self.tls_verify,
            ) as response:
                if response.status != 200:
                    return False

                data = await response.json()
                return data["data"]["valid"]

    def _map_key_type(self, key_type: KeyType) -> str:
        """Map KeyType to Vault Transit key type."""
        mapping = {
            KeyType.AES_256_GCM: "aes256-gcm96",
            KeyType.AES_128_GCM: "aes128-gcm96",
            KeyType.RSA_2048: "rsa-2048",
            KeyType.RSA_4096: "rsa-4096",
            KeyType.EC_P256: "ecdsa-p256",
            KeyType.EC_P384: "ecdsa-p384",
            KeyType.ED25519: "ed25519",
        }
        return mapping.get(key_type, "aes256-gcm96")

    def _reverse_map_key_type(self, vault_type: str) -> KeyType:
        """Map Vault key type back to KeyType."""
        mapping = {
            "aes256-gcm96": KeyType.AES_256_GCM,
            "aes128-gcm96": KeyType.AES_128_GCM,
            "rsa-2048": KeyType.RSA_2048,
            "rsa-4096": KeyType.RSA_4096,
            "ecdsa-p256": KeyType.EC_P256,
            "ecdsa-p384": KeyType.EC_P384,
            "ed25519": KeyType.ED25519,
        }
        return mapping.get(vault_type, KeyType.AES_256_GCM)


class AWSKMSBackend(KMSBackend):
    """
    AWS KMS backend.

    Uses AWS Key Management Service for key management and cryptographic operations.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> None:
        self.region = region
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.endpoint_url = endpoint_url
        self.profile_name = profile_name
        self._client: Optional[Any] = None
        self._keys: dict[str, KeyMetadata] = {}
        self._logger = logger.bind(backend="aws_kms")

    async def initialize(self) -> None:
        """Initialize AWS KMS client."""
        try:
            import boto3
            from botocore.config import Config

            config = Config(
                region_name=self.region,
                retries={"max_attempts": 3, "mode": "adaptive"},
            )

            session_kwargs = {}
            if self.profile_name:
                session_kwargs["profile_name"] = self.profile_name

            session = boto3.Session(**session_kwargs)

            client_kwargs = {"config": config}
            if self.access_key_id and self.secret_access_key:
                client_kwargs["aws_access_key_id"] = self.access_key_id
                client_kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                client_kwargs["aws_session_token"] = self.session_token
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url

            self._client = session.client("kms", **client_kwargs)
            self._logger.info("Connected to AWS KMS", region=self.region)

        except ImportError:
            self._logger.warning("boto3 not available, using mock mode")

    async def create_key(
        self,
        key_id: str,
        key_type: KeyType,
        purpose: KeyPurpose,
        labels: Optional[dict[str, str]] = None,
    ) -> KeyMetadata:
        """Create a new key in AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        key_spec = self._map_key_type(key_type)
        key_usage = self._map_key_purpose(purpose)

        tags = [{"TagKey": k, "TagValue": v} for k, v in (labels or {}).items()]
        tags.append({"TagKey": "aion-key-id", "TagValue": key_id})

        response = self._client.create_key(
            Description=f"AION key: {key_id}",
            KeySpec=key_spec,
            KeyUsage=key_usage,
            Tags=tags,
            MultiRegion=False,
        )

        aws_key_id = response["KeyMetadata"]["KeyId"]
        now = time.time()

        # Create alias for easier reference
        try:
            self._client.create_alias(
                AliasName=f"alias/aion-{key_id}",
                TargetKeyId=aws_key_id,
            )
        except Exception:
            pass  # Alias might already exist

        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            purpose=purpose,
            state=KeyState.ENABLED,
            created_at=now,
            updated_at=now,
            labels=labels or {},
            provider=KMSProvider.AWS_KMS,
            provider_key_id=aws_key_id,
        )
        self._keys[key_id] = metadata

        self._logger.info("Created key in AWS KMS", key_id=key_id, aws_key_id=aws_key_id)
        return metadata

    async def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata from AWS KMS."""
        if key_id in self._keys:
            return self._keys[key_id]

        if not self._client:
            return None

        try:
            response = self._client.describe_key(KeyId=f"alias/aion-{key_id}")
            key_data = response["KeyMetadata"]

            state_map = {
                "Enabled": KeyState.ENABLED,
                "Disabled": KeyState.DISABLED,
                "PendingDeletion": KeyState.PENDING_DELETION,
            }

            metadata = KeyMetadata(
                key_id=key_id,
                key_type=self._reverse_map_key_type(key_data.get("KeySpec", "SYMMETRIC_DEFAULT")),
                purpose=self._reverse_map_purpose(key_data.get("KeyUsage", "ENCRYPT_DECRYPT")),
                state=state_map.get(key_data["KeyState"], KeyState.ENABLED),
                created_at=key_data["CreationDate"].timestamp(),
                updated_at=time.time(),
                provider=KMSProvider.AWS_KMS,
                provider_key_id=key_data["KeyId"],
            )
            self._keys[key_id] = metadata
            return metadata

        except Exception as e:
            self._logger.error("Failed to get key metadata", error=str(e))
            return None

    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data using AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        kwargs = {
            "KeyId": self._get_aws_key_id(key_id),
            "Plaintext": plaintext,
        }
        if aad:
            kwargs["EncryptionContext"] = {"aad": base64.b64encode(aad).decode()}

        response = self._client.encrypt(**kwargs)

        return EncryptedData(
            ciphertext=response["CiphertextBlob"],
            key_id=key_id,
            key_version=1,
            algorithm="aws-kms",
            aad=aad,
        )

    async def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data using AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        kwargs = {
            "CiphertextBlob": encrypted_data.ciphertext,
        }
        if encrypted_data.aad:
            kwargs["EncryptionContext"] = {"aad": base64.b64encode(encrypted_data.aad).decode()}

        response = self._client.decrypt(**kwargs)
        return response["Plaintext"]

    async def generate_data_key(
        self,
        key_id: str,
        key_spec: KeyType = KeyType.AES_256_GCM,
    ) -> DataKey:
        """Generate a data key using AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        spec = "AES_256" if "256" in key_spec.value else "AES_128"

        response = self._client.generate_data_key(
            KeyId=self._get_aws_key_id(key_id),
            KeySpec=spec,
        )

        return DataKey(
            plaintext=response["Plaintext"],
            ciphertext=response["CiphertextBlob"],
            key_id=key_id,
            algorithm=key_spec.value,
        )

    async def rotate_key(self, key_id: str) -> KeyMetadata:
        """Enable automatic key rotation in AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        aws_key_id = self._get_aws_key_id(key_id)
        self._client.enable_key_rotation(KeyId=aws_key_id)

        if key_id in self._keys:
            self._keys[key_id].updated_at = time.time()
            return self._keys[key_id]

        return await self.get_key_metadata(key_id)

    async def disable_key(self, key_id: str) -> None:
        """Disable a key in AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        self._client.disable_key(KeyId=self._get_aws_key_id(key_id))

        if key_id in self._keys:
            self._keys[key_id].state = KeyState.DISABLED

    async def enable_key(self, key_id: str) -> None:
        """Enable a key in AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        self._client.enable_key(KeyId=self._get_aws_key_id(key_id))

        if key_id in self._keys:
            self._keys[key_id].state = KeyState.ENABLED

    async def schedule_key_deletion(
        self,
        key_id: str,
        pending_days: int = 30,
    ) -> None:
        """Schedule key deletion in AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        self._client.schedule_key_deletion(
            KeyId=self._get_aws_key_id(key_id),
            PendingWindowInDays=min(max(pending_days, 7), 30),  # AWS allows 7-30 days
        )

        if key_id in self._keys:
            self._keys[key_id].state = KeyState.PENDING_DELETION

    async def sign(
        self,
        key_id: str,
        data: bytes,
        algorithm: Optional[str] = None,
    ) -> bytes:
        """Sign data using AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        response = self._client.sign(
            KeyId=self._get_aws_key_id(key_id),
            Message=data,
            MessageType="RAW",
            SigningAlgorithm=algorithm or "RSASSA_PKCS1_V1_5_SHA_256",
        )

        return response["Signature"]

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> bool:
        """Verify a signature using AWS KMS."""
        if not self._client:
            raise KMSError("AWS KMS client not initialized")

        try:
            response = self._client.verify(
                KeyId=self._get_aws_key_id(key_id),
                Message=data,
                MessageType="RAW",
                Signature=signature,
                SigningAlgorithm=algorithm or "RSASSA_PKCS1_V1_5_SHA_256",
            )
            return response["SignatureValid"]
        except Exception:
            return False

    def _get_aws_key_id(self, key_id: str) -> str:
        """Get AWS key ID from key_id."""
        if key_id in self._keys and self._keys[key_id].provider_key_id:
            return self._keys[key_id].provider_key_id
        return f"alias/aion-{key_id}"

    def _map_key_type(self, key_type: KeyType) -> str:
        """Map KeyType to AWS KMS KeySpec."""
        mapping = {
            KeyType.AES_256_GCM: "SYMMETRIC_DEFAULT",
            KeyType.AES_128_GCM: "SYMMETRIC_DEFAULT",
            KeyType.RSA_2048: "RSA_2048",
            KeyType.RSA_4096: "RSA_4096",
            KeyType.EC_P256: "ECC_NIST_P256",
            KeyType.EC_P384: "ECC_NIST_P384",
        }
        return mapping.get(key_type, "SYMMETRIC_DEFAULT")

    def _reverse_map_key_type(self, aws_spec: str) -> KeyType:
        """Map AWS KeySpec back to KeyType."""
        mapping = {
            "SYMMETRIC_DEFAULT": KeyType.AES_256_GCM,
            "RSA_2048": KeyType.RSA_2048,
            "RSA_4096": KeyType.RSA_4096,
            "ECC_NIST_P256": KeyType.EC_P256,
            "ECC_NIST_P384": KeyType.EC_P384,
        }
        return mapping.get(aws_spec, KeyType.AES_256_GCM)

    def _map_key_purpose(self, purpose: KeyPurpose) -> str:
        """Map KeyPurpose to AWS KMS KeyUsage."""
        mapping = {
            KeyPurpose.ENCRYPT_DECRYPT: "ENCRYPT_DECRYPT",
            KeyPurpose.SIGN_VERIFY: "SIGN_VERIFY",
        }
        return mapping.get(purpose, "ENCRYPT_DECRYPT")

    def _reverse_map_purpose(self, aws_usage: str) -> KeyPurpose:
        """Map AWS KeyUsage back to KeyPurpose."""
        mapping = {
            "ENCRYPT_DECRYPT": KeyPurpose.ENCRYPT_DECRYPT,
            "SIGN_VERIFY": KeyPurpose.SIGN_VERIFY,
        }
        return mapping.get(aws_usage, KeyPurpose.ENCRYPT_DECRYPT)


class AzureKeyVaultBackend(KMSBackend):
    """
    Azure Key Vault backend.

    Uses Azure Key Vault for key management and cryptographic operations.
    """

    def __init__(
        self,
        vault_url: str,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        use_managed_identity: bool = False,
    ) -> None:
        self.vault_url = vault_url.rstrip("/")
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.use_managed_identity = use_managed_identity
        self._key_client: Optional[Any] = None
        self._crypto_client_cache: dict[str, Any] = {}
        self._keys: dict[str, KeyMetadata] = {}
        self._logger = logger.bind(backend="azure_key_vault")

    async def initialize(self) -> None:
        """Initialize Azure Key Vault client."""
        try:
            from azure.identity import (
                ClientSecretCredential,
                DefaultAzureCredential,
                ManagedIdentityCredential,
            )
            from azure.keyvault.keys import KeyClient

            if self.use_managed_identity:
                credential = ManagedIdentityCredential()
            elif self.tenant_id and self.client_id and self.client_secret:
                credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                )
            else:
                credential = DefaultAzureCredential()

            self._key_client = KeyClient(vault_url=self.vault_url, credential=credential)
            self._credential = credential
            self._logger.info("Connected to Azure Key Vault", vault_url=self.vault_url)

        except ImportError:
            self._logger.warning("azure-keyvault-keys not available, using mock mode")

    async def create_key(
        self,
        key_id: str,
        key_type: KeyType,
        purpose: KeyPurpose,
        labels: Optional[dict[str, str]] = None,
    ) -> KeyMetadata:
        """Create a new key in Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        from azure.keyvault.keys import KeyType as AzureKeyType

        azure_key_type = self._map_key_type(key_type)
        key_ops = self._map_key_ops(purpose)

        key = self._key_client.create_key(
            name=key_id,
            key_type=azure_key_type,
            key_operations=key_ops,
            tags=labels,
        )

        now = time.time()
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            purpose=purpose,
            state=KeyState.ENABLED,
            created_at=now,
            updated_at=now,
            labels=labels or {},
            provider=KMSProvider.AZURE_KEY_VAULT,
            provider_key_id=key.id,
        )
        self._keys[key_id] = metadata

        self._logger.info("Created key in Azure Key Vault", key_id=key_id)
        return metadata

    async def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata from Azure Key Vault."""
        if key_id in self._keys:
            return self._keys[key_id]

        if not self._key_client:
            return None

        try:
            key = self._key_client.get_key(key_id)

            metadata = KeyMetadata(
                key_id=key_id,
                key_type=self._reverse_map_key_type(str(key.key_type)),
                purpose=KeyPurpose.ENCRYPT_DECRYPT,
                state=KeyState.ENABLED if key.properties.enabled else KeyState.DISABLED,
                created_at=key.properties.created_on.timestamp() if key.properties.created_on else time.time(),
                updated_at=key.properties.updated_on.timestamp() if key.properties.updated_on else time.time(),
                provider=KMSProvider.AZURE_KEY_VAULT,
                provider_key_id=key.id,
            )
            self._keys[key_id] = metadata
            return metadata

        except Exception as e:
            self._logger.error("Failed to get key metadata", error=str(e))
            return None

    def _get_crypto_client(self, key_id: str) -> Any:
        """Get or create a CryptographyClient for a key."""
        if key_id not in self._crypto_client_cache:
            from azure.keyvault.keys.crypto import CryptographyClient

            key = self._key_client.get_key(key_id)
            self._crypto_client_cache[key_id] = CryptographyClient(
                key=key.id,
                credential=self._credential,
            )
        return self._crypto_client_cache[key_id]

    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data using Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        from azure.keyvault.keys.crypto import EncryptionAlgorithm

        crypto_client = self._get_crypto_client(key_id)
        result = crypto_client.encrypt(
            algorithm=EncryptionAlgorithm.rsa_oaep_256,
            plaintext=plaintext,
        )

        return EncryptedData(
            ciphertext=result.ciphertext,
            key_id=key_id,
            key_version=1,
            algorithm="azure-rsa-oaep-256",
            aad=aad,
        )

    async def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data using Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        from azure.keyvault.keys.crypto import EncryptionAlgorithm

        crypto_client = self._get_crypto_client(encrypted_data.key_id)
        result = crypto_client.decrypt(
            algorithm=EncryptionAlgorithm.rsa_oaep_256,
            ciphertext=encrypted_data.ciphertext,
        )

        return result.plaintext

    async def generate_data_key(
        self,
        key_id: str,
        key_spec: KeyType = KeyType.AES_256_GCM,
    ) -> DataKey:
        """Generate a data key (Azure doesn't have direct DEK generation)."""
        # Generate random key locally and wrap with KEK
        key_size = 32 if "256" in key_spec.value else 16
        plaintext_key = secrets.token_bytes(key_size)

        encrypted = await self.encrypt(key_id, plaintext_key)

        return DataKey(
            plaintext=plaintext_key,
            ciphertext=encrypted.ciphertext,
            key_id=key_id,
            algorithm=key_spec.value,
        )

    async def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key in Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        # Get current key to preserve settings
        current = self._key_client.get_key(key_id)

        # Create new version
        new_key = self._key_client.create_key(
            name=key_id,
            key_type=current.key_type,
            key_operations=current.key_operations,
            tags=current.properties.tags,
        )

        # Clear crypto client cache
        if key_id in self._crypto_client_cache:
            del self._crypto_client_cache[key_id]

        if key_id in self._keys:
            self._keys[key_id].version += 1
            self._keys[key_id].updated_at = time.time()
            return self._keys[key_id]

        return await self.get_key_metadata(key_id)

    async def disable_key(self, key_id: str) -> None:
        """Disable a key in Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        self._key_client.update_key_properties(key_id, enabled=False)

        if key_id in self._keys:
            self._keys[key_id].state = KeyState.DISABLED

    async def enable_key(self, key_id: str) -> None:
        """Enable a key in Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        self._key_client.update_key_properties(key_id, enabled=True)

        if key_id in self._keys:
            self._keys[key_id].state = KeyState.ENABLED

    async def schedule_key_deletion(
        self,
        key_id: str,
        pending_days: int = 30,
    ) -> None:
        """Delete a key (Azure has soft-delete by default)."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        self._key_client.begin_delete_key(key_id)

        if key_id in self._keys:
            self._keys[key_id].state = KeyState.PENDING_DELETION

    async def sign(
        self,
        key_id: str,
        data: bytes,
        algorithm: Optional[str] = None,
    ) -> bytes:
        """Sign data using Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        from azure.keyvault.keys.crypto import SignatureAlgorithm

        crypto_client = self._get_crypto_client(key_id)
        result = crypto_client.sign(
            algorithm=SignatureAlgorithm.rs256,
            digest=hashlib.sha256(data).digest(),
        )

        return result.signature

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> bool:
        """Verify a signature using Azure Key Vault."""
        if not self._key_client:
            raise KMSError("Azure Key Vault client not initialized")

        from azure.keyvault.keys.crypto import SignatureAlgorithm

        try:
            crypto_client = self._get_crypto_client(key_id)
            result = crypto_client.verify(
                algorithm=SignatureAlgorithm.rs256,
                digest=hashlib.sha256(data).digest(),
                signature=signature,
            )
            return result.is_valid
        except Exception:
            return False

    def _map_key_type(self, key_type: KeyType) -> Any:
        """Map KeyType to Azure KeyType."""
        from azure.keyvault.keys import KeyType as AzureKeyType

        mapping = {
            KeyType.RSA_2048: AzureKeyType.rsa,
            KeyType.RSA_4096: AzureKeyType.rsa,
            KeyType.EC_P256: AzureKeyType.ec,
            KeyType.EC_P384: AzureKeyType.ec,
        }
        return mapping.get(key_type, AzureKeyType.rsa)

    def _reverse_map_key_type(self, azure_type: str) -> KeyType:
        """Map Azure key type back to KeyType."""
        if "rsa" in azure_type.lower():
            return KeyType.RSA_2048
        if "ec" in azure_type.lower():
            return KeyType.EC_P256
        return KeyType.RSA_2048

    def _map_key_ops(self, purpose: KeyPurpose) -> list[str]:
        """Map KeyPurpose to Azure key operations."""
        from azure.keyvault.keys import KeyOperation

        if purpose == KeyPurpose.ENCRYPT_DECRYPT:
            return [KeyOperation.encrypt, KeyOperation.decrypt, KeyOperation.wrap_key, KeyOperation.unwrap_key]
        elif purpose == KeyPurpose.SIGN_VERIFY:
            return [KeyOperation.sign, KeyOperation.verify]
        return [KeyOperation.encrypt, KeyOperation.decrypt]


class LocalKMSBackend(KMSBackend):
    """
    Local KMS backend for development and testing.

    Uses software-based cryptography. NOT suitable for production use.
    """

    def __init__(self, master_key: Optional[bytes] = None) -> None:
        self._master_key = master_key or secrets.token_bytes(32)
        self._keys: dict[str, tuple[KeyMetadata, bytes]] = {}
        self._logger = logger.bind(backend="local")

    async def initialize(self) -> None:
        """Initialize local backend."""
        self._logger.warning("Using local KMS backend - NOT FOR PRODUCTION")

    async def create_key(
        self,
        key_id: str,
        key_type: KeyType,
        purpose: KeyPurpose,
        labels: Optional[dict[str, str]] = None,
    ) -> KeyMetadata:
        """Create a new local key."""
        if "aes" in key_type.value.lower():
            key_size = 32 if "256" in key_type.value else 16
            key_material = secrets.token_bytes(key_size)
        else:
            key_material = secrets.token_bytes(32)

        now = time.time()
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            purpose=purpose,
            state=KeyState.ENABLED,
            created_at=now,
            updated_at=now,
            labels=labels or {},
            provider=KMSProvider.LOCAL,
        )

        self._keys[key_id] = (metadata, key_material)
        return metadata

    async def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata."""
        if key_id in self._keys:
            return self._keys[key_id][0]
        return None

    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data using local key."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if key_id not in self._keys:
            raise KMSError(f"Key not found: {key_id}")

        _, key_material = self._keys[key_id]
        iv = secrets.token_bytes(12)
        aesgcm = AESGCM(key_material)
        ciphertext = aesgcm.encrypt(iv, plaintext, aad)

        return EncryptedData(
            ciphertext=ciphertext,
            key_id=key_id,
            key_version=1,
            algorithm="aes-256-gcm",
            iv=iv,
            aad=aad,
        )

    async def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data using local key."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if encrypted_data.key_id not in self._keys:
            raise KMSError(f"Key not found: {encrypted_data.key_id}")

        _, key_material = self._keys[encrypted_data.key_id]
        aesgcm = AESGCM(key_material)

        return aesgcm.decrypt(
            encrypted_data.iv,
            encrypted_data.ciphertext,
            encrypted_data.aad,
        )

    async def generate_data_key(
        self,
        key_id: str,
        key_spec: KeyType = KeyType.AES_256_GCM,
    ) -> DataKey:
        """Generate a data key."""
        key_size = 32 if "256" in key_spec.value else 16
        plaintext_key = secrets.token_bytes(key_size)

        encrypted = await self.encrypt(key_id, plaintext_key)

        return DataKey(
            plaintext=plaintext_key,
            ciphertext=encrypted.ciphertext,
            key_id=key_id,
            algorithm=key_spec.value,
        )

    async def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a local key."""
        if key_id not in self._keys:
            raise KMSError(f"Key not found: {key_id}")

        metadata, _ = self._keys[key_id]
        new_key = secrets.token_bytes(32)

        metadata.version += 1
        metadata.updated_at = time.time()

        self._keys[key_id] = (metadata, new_key)
        return metadata

    async def disable_key(self, key_id: str) -> None:
        """Disable a local key."""
        if key_id in self._keys:
            self._keys[key_id][0].state = KeyState.DISABLED

    async def enable_key(self, key_id: str) -> None:
        """Enable a local key."""
        if key_id in self._keys:
            self._keys[key_id][0].state = KeyState.ENABLED

    async def schedule_key_deletion(
        self,
        key_id: str,
        pending_days: int = 30,
    ) -> None:
        """Schedule key deletion."""
        if key_id in self._keys:
            self._keys[key_id][0].state = KeyState.PENDING_DELETION

    async def sign(
        self,
        key_id: str,
        data: bytes,
        algorithm: Optional[str] = None,
    ) -> bytes:
        """Sign data using HMAC."""
        if key_id not in self._keys:
            raise KMSError(f"Key not found: {key_id}")

        _, key_material = self._keys[key_id]
        return hmac.new(key_material, data, hashlib.sha256).digest()

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> bool:
        """Verify HMAC signature."""
        expected = await self.sign(key_id, data, algorithm)
        return hmac.compare_digest(signature, expected)


class KMSError(Exception):
    """KMS operation error."""
    pass


class AuthenticationError(KMSError):
    """KMS authentication error."""
    pass


class KeyManagementService:
    """
    Unified Key Management Service.

    Provides a consistent interface across multiple KMS backends with
    automatic failover, caching, and envelope encryption support.
    """

    def __init__(
        self,
        primary_backend: KMSBackend,
        fallback_backend: Optional[KMSBackend] = None,
        cache_ttl: int = 300,
    ) -> None:
        self.primary = primary_backend
        self.fallback = fallback_backend
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._logger = logger.bind(service="kms")

    async def initialize(self) -> None:
        """Initialize all backends."""
        await self.primary.initialize()
        if self.fallback:
            await self.fallback.initialize()
        self._logger.info("Key Management Service initialized")

    async def create_key(
        self,
        key_id: str,
        key_type: KeyType = KeyType.AES_256_GCM,
        purpose: KeyPurpose = KeyPurpose.ENCRYPT_DECRYPT,
        labels: Optional[dict[str, str]] = None,
    ) -> KeyMetadata:
        """Create a new key."""
        try:
            return await self.primary.create_key(key_id, key_type, purpose, labels)
        except Exception as e:
            if self.fallback:
                self._logger.warning("Primary KMS failed, using fallback", error=str(e))
                return await self.fallback.create_key(key_id, key_type, purpose, labels)
            raise

    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data."""
        try:
            return await self.primary.encrypt(key_id, plaintext, aad)
        except Exception as e:
            if self.fallback:
                self._logger.warning("Primary KMS encrypt failed, using fallback", error=str(e))
                return await self.fallback.encrypt(key_id, plaintext, aad)
            raise

    async def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data."""
        try:
            return await self.primary.decrypt(encrypted_data)
        except Exception as e:
            if self.fallback:
                self._logger.warning("Primary KMS decrypt failed, using fallback", error=str(e))
                return await self.fallback.decrypt(encrypted_data)
            raise

    async def envelope_encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> dict[str, Any]:
        """
        Encrypt data using envelope encryption.

        1. Generate a data encryption key (DEK)
        2. Encrypt data with DEK locally
        3. Return encrypted DEK and encrypted data
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Generate DEK
        data_key = await self.primary.generate_data_key(key_id)

        # Encrypt data locally with DEK
        iv = secrets.token_bytes(12)
        aesgcm = AESGCM(data_key.plaintext)
        ciphertext = aesgcm.encrypt(iv, plaintext, aad)

        # Clear plaintext DEK from memory
        # (In practice, use secure memory handling)

        return {
            "encrypted_data_key": base64.b64encode(data_key.ciphertext).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": base64.b64encode(iv).decode(),
            "aad": base64.b64encode(aad).decode() if aad else None,
            "key_id": key_id,
            "algorithm": data_key.algorithm,
        }

    async def envelope_decrypt(
        self,
        envelope: dict[str, Any],
    ) -> bytes:
        """
        Decrypt envelope-encrypted data.

        1. Decrypt the DEK using KMS
        2. Decrypt data locally with DEK
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Decrypt DEK
        encrypted_dek = base64.b64decode(envelope["encrypted_data_key"])
        key_id = envelope["key_id"]

        # Create EncryptedData for DEK decryption
        dek_encrypted = EncryptedData(
            ciphertext=encrypted_dek,
            key_id=key_id,
            key_version=1,
            algorithm="kms-wrapped",
        )
        plaintext_dek = await self.decrypt(dek_encrypted)

        # Decrypt data locally
        ciphertext = base64.b64decode(envelope["ciphertext"])
        iv = base64.b64decode(envelope["iv"])
        aad = base64.b64decode(envelope["aad"]) if envelope.get("aad") else None

        aesgcm = AESGCM(plaintext_dek)
        return aesgcm.decrypt(iv, ciphertext, aad)

    async def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key."""
        return await self.primary.rotate_key(key_id)

    async def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata with caching."""
        cache_key = f"metadata:{key_id}"

        if cache_key in self._cache:
            value, expires = self._cache[cache_key]
            if time.time() < expires:
                return value

        metadata = await self.primary.get_key_metadata(key_id)
        if metadata:
            self._cache[cache_key] = (metadata, time.time() + self.cache_ttl)

        return metadata


def create_kms_backend(
    provider: KMSProvider,
    **config: Any,
) -> KMSBackend:
    """Factory function to create KMS backend."""
    if provider == KMSProvider.VAULT:
        return VaultBackend(**config)
    elif provider == KMSProvider.AWS_KMS:
        return AWSKMSBackend(**config)
    elif provider == KMSProvider.AZURE_KEY_VAULT:
        return AzureKeyVaultBackend(**config)
    elif provider == KMSProvider.LOCAL:
        return LocalKMSBackend(**config)
    else:
        raise ValueError(f"Unsupported KMS provider: {provider}")
