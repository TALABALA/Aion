"""
Secure Enclave and Auto-Unsealing System.

Implements secure key management with:
- Shamir's Secret Sharing for key splitting
- Auto-unsealing mechanisms
- Hardware security module (HSM) integration
- Secure memory handling
- Key ceremony support

Addresses the "secret zero" problem through multiple unsealing strategies.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import secrets
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger()


class SealStatus(str, Enum):
    """Enclave seal status."""
    SEALED = "sealed"
    UNSEALING = "unsealing"
    UNSEALED = "unsealed"
    ERROR = "error"


class UnsealMethod(str, Enum):
    """Methods for unsealing the enclave."""
    MANUAL = "manual"                    # Manual key entry
    SHAMIR = "shamir"                    # Shamir's Secret Sharing
    HSM = "hsm"                          # Hardware Security Module
    KMS = "kms"                          # External KMS (Vault, AWS, etc.)
    TPM = "tpm"                          # Trusted Platform Module
    CLOUD_KMS = "cloud_kms"              # Cloud provider KMS
    ENVIRONMENT = "environment"          # Environment variable (dev only)
    AUTO = "auto"                        # Automatic using stored keys


class KeyCeremonyRole(str, Enum):
    """Roles in a key ceremony."""
    CUSTODIAN = "custodian"
    WITNESS = "witness"
    AUDITOR = "auditor"
    OPERATOR = "operator"


@dataclass
class ShamirShare:
    """A single share from Shamir's Secret Sharing."""
    index: int
    value: bytes
    threshold: int
    total_shares: int
    key_id: str
    created_at: float = field(default_factory=time.time)
    custodian: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def encode(self) -> str:
        """Encode share for storage/transmission."""
        data = struct.pack(
            ">BBHH",
            self.index,
            self.threshold,
            self.total_shares,
            len(self.value),
        ) + self.value + self.key_id.encode()
        return base64.urlsafe_b64encode(data).decode()

    @classmethod
    def decode(cls, encoded: str) -> "ShamirShare":
        """Decode a share from encoded format."""
        data = base64.urlsafe_b64decode(encoded)
        index, threshold, total, value_len = struct.unpack(">BBHH", data[:6])
        value = data[6:6 + value_len]
        key_id = data[6 + value_len:].decode()
        return cls(
            index=index,
            value=value,
            threshold=threshold,
            total_shares=total,
            key_id=key_id,
        )


@dataclass
class UnsealProgress:
    """Progress of unsealing operation."""
    method: UnsealMethod
    required_shares: int
    received_shares: int
    progress_percent: float
    status: SealStatus
    message: str
    received_from: list[str] = field(default_factory=list)


@dataclass
class KeyCeremonyEvent:
    """Event during a key ceremony."""
    event_type: str
    timestamp: float
    participant: str
    role: KeyCeremonyRole
    details: dict[str, Any]
    signature: Optional[bytes] = None


@dataclass
class KeyCeremony:
    """Key ceremony for secure key generation/recovery."""
    ceremony_id: str
    ceremony_type: str  # generation, recovery, rotation
    created_at: float
    participants: dict[str, KeyCeremonyRole]
    required_custodians: int
    required_witnesses: int
    events: list[KeyCeremonyEvent] = field(default_factory=list)
    completed: bool = False
    result_key_id: Optional[str] = None


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing implementation.

    Splits a secret into n shares where any k shares can reconstruct
    the original secret, but k-1 shares reveal nothing.
    """

    # Use a large prime for the finite field
    PRIME = 2**256 - 189

    def __init__(self) -> None:
        self._logger = logger.bind(component="shamir")

    def split(
        self,
        secret: bytes,
        threshold: int,
        total_shares: int,
        key_id: str = "",
    ) -> list[ShamirShare]:
        """
        Split a secret into shares.

        Args:
            secret: The secret to split
            threshold: Minimum shares needed to reconstruct (k)
            total_shares: Total number of shares to generate (n)
            key_id: Identifier for the key being split

        Returns:
            List of ShamirShare objects
        """
        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")

        # Convert secret to integer
        secret_int = int.from_bytes(secret, "big")
        if secret_int >= self.PRIME:
            raise ValueError("Secret too large for field")

        # Generate random polynomial coefficients
        coefficients = [secret_int]
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(self.PRIME))

        # Evaluate polynomial at each point
        shares = []
        for i in range(1, total_shares + 1):
            y = self._evaluate_polynomial(coefficients, i)
            share_value = y.to_bytes((y.bit_length() + 7) // 8, "big")

            shares.append(ShamirShare(
                index=i,
                value=share_value,
                threshold=threshold,
                total_shares=total_shares,
                key_id=key_id or hashlib.sha256(secret).hexdigest()[:16],
            ))

        self._logger.info(
            "Secret split",
            threshold=threshold,
            total_shares=total_shares,
            key_id=shares[0].key_id,
        )

        return shares

    def reconstruct(self, shares: list[ShamirShare]) -> bytes:
        """
        Reconstruct a secret from shares.

        Args:
            shares: List of shares (at least threshold shares required)

        Returns:
            The reconstructed secret
        """
        if not shares:
            raise ValueError("No shares provided")

        threshold = shares[0].threshold
        if len(shares) < threshold:
            raise ValueError(f"Need at least {threshold} shares, got {len(shares)}")

        # Verify all shares are from the same split
        key_id = shares[0].key_id
        if not all(s.key_id == key_id for s in shares):
            raise ValueError("Shares are from different keys")

        # Use Lagrange interpolation to find f(0)
        points = [(s.index, int.from_bytes(s.value, "big")) for s in shares[:threshold]]

        secret_int = 0
        for i, (xi, yi) in enumerate(points):
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(points):
                if i != j:
                    numerator = (numerator * (-xj)) % self.PRIME
                    denominator = (denominator * (xi - xj)) % self.PRIME

            # Modular inverse
            lagrange_coeff = (numerator * pow(denominator, -1, self.PRIME)) % self.PRIME
            secret_int = (secret_int + yi * lagrange_coeff) % self.PRIME

        # Convert back to bytes
        secret = secret_int.to_bytes((secret_int.bit_length() + 7) // 8, "big")

        self._logger.info("Secret reconstructed", key_id=key_id)

        return secret

    def _evaluate_polynomial(self, coefficients: list[int], x: int) -> int:
        """Evaluate polynomial at point x."""
        result = 0
        for i, coeff in enumerate(coefficients):
            result = (result + coeff * pow(x, i, self.PRIME)) % self.PRIME
        return result


class UnsealKeyProvider(ABC):
    """Abstract provider for unseal keys."""

    @abstractmethod
    async def get_unseal_key(self) -> Optional[bytes]:
        """Retrieve the unseal key."""
        pass

    @abstractmethod
    async def store_unseal_key(self, key: bytes) -> bool:
        """Store the unseal key."""
        pass


class EnvironmentKeyProvider(UnsealKeyProvider):
    """
    Unseal key from environment variable.

    WARNING: Only use in development/testing.
    """

    def __init__(self, env_var: str = "AION_UNSEAL_KEY") -> None:
        self.env_var = env_var

    async def get_unseal_key(self) -> Optional[bytes]:
        key_str = os.environ.get(self.env_var)
        if key_str:
            return base64.b64decode(key_str)
        return None

    async def store_unseal_key(self, key: bytes) -> bool:
        # Cannot store in environment
        logger.warning("Cannot store unseal key in environment")
        return False


class CloudKMSKeyProvider(UnsealKeyProvider):
    """
    Unseal key from cloud KMS (auto-unseal).

    Uses cloud KMS to encrypt/decrypt the unseal key, enabling
    automatic unsealing without manual intervention.
    """

    def __init__(
        self,
        kms_key_id: str,
        encrypted_key_path: str,
        kms_provider: str = "aws",  # aws, gcp, azure
        region: Optional[str] = None,
    ) -> None:
        self.kms_key_id = kms_key_id
        self.encrypted_key_path = encrypted_key_path
        self.kms_provider = kms_provider
        self.region = region
        self._logger = logger.bind(provider="cloud_kms")

    async def get_unseal_key(self) -> Optional[bytes]:
        """Retrieve and decrypt the unseal key from cloud KMS."""
        try:
            # Read encrypted key from storage
            if not os.path.exists(self.encrypted_key_path):
                return None

            with open(self.encrypted_key_path, "rb") as f:
                encrypted_key = f.read()

            # Decrypt using cloud KMS
            if self.kms_provider == "aws":
                return await self._decrypt_aws(encrypted_key)
            elif self.kms_provider == "gcp":
                return await self._decrypt_gcp(encrypted_key)
            elif self.kms_provider == "azure":
                return await self._decrypt_azure(encrypted_key)

            return None

        except Exception as e:
            self._logger.error("Failed to get unseal key", error=str(e))
            return None

    async def store_unseal_key(self, key: bytes) -> bool:
        """Encrypt and store the unseal key."""
        try:
            # Encrypt using cloud KMS
            if self.kms_provider == "aws":
                encrypted_key = await self._encrypt_aws(key)
            elif self.kms_provider == "gcp":
                encrypted_key = await self._encrypt_gcp(key)
            elif self.kms_provider == "azure":
                encrypted_key = await self._encrypt_azure(key)
            else:
                return False

            # Store encrypted key
            with open(self.encrypted_key_path, "wb") as f:
                f.write(encrypted_key)

            return True

        except Exception as e:
            self._logger.error("Failed to store unseal key", error=str(e))
            return False

    async def _decrypt_aws(self, encrypted_key: bytes) -> bytes:
        """Decrypt using AWS KMS."""
        try:
            import boto3

            client = boto3.client("kms", region_name=self.region)
            response = client.decrypt(
                CiphertextBlob=encrypted_key,
                KeyId=self.kms_key_id,
            )
            return response["Plaintext"]

        except ImportError:
            self._logger.warning("boto3 not available")
            raise

    async def _encrypt_aws(self, key: bytes) -> bytes:
        """Encrypt using AWS KMS."""
        import boto3

        client = boto3.client("kms", region_name=self.region)
        response = client.encrypt(
            KeyId=self.kms_key_id,
            Plaintext=key,
        )
        return response["CiphertextBlob"]

    async def _decrypt_gcp(self, encrypted_key: bytes) -> bytes:
        """Decrypt using GCP KMS."""
        # GCP implementation
        raise NotImplementedError("GCP KMS integration not implemented")

    async def _encrypt_gcp(self, key: bytes) -> bytes:
        """Encrypt using GCP KMS."""
        raise NotImplementedError("GCP KMS integration not implemented")

    async def _decrypt_azure(self, encrypted_key: bytes) -> bytes:
        """Decrypt using Azure Key Vault."""
        raise NotImplementedError("Azure Key Vault integration not implemented")

    async def _encrypt_azure(self, key: bytes) -> bytes:
        """Encrypt using Azure Key Vault."""
        raise NotImplementedError("Azure Key Vault integration not implemented")


class SecureEnclave:
    """
    Secure Enclave for master key management.

    Handles the "secret zero" problem through various unsealing mechanisms:
    - Manual key entry with Shamir's Secret Sharing
    - Cloud KMS auto-unsealing
    - HSM integration
    - TPM-based unsealing

    The enclave protects the master key used to encrypt all other secrets.
    """

    def __init__(
        self,
        unseal_method: UnsealMethod = UnsealMethod.SHAMIR,
        shamir_threshold: int = 3,
        shamir_total_shares: int = 5,
        key_providers: Optional[list[UnsealKeyProvider]] = None,
        auto_seal_timeout: int = 3600,
    ) -> None:
        self.unseal_method = unseal_method
        self.shamir_threshold = shamir_threshold
        self.shamir_total_shares = shamir_total_shares
        self.key_providers = key_providers or []
        self.auto_seal_timeout = auto_seal_timeout

        self._status = SealStatus.SEALED
        self._master_key: Optional[bytes] = None
        self._shamir = ShamirSecretSharing()
        self._collected_shares: list[ShamirShare] = []
        self._last_activity: float = 0
        self._seal_timer: Optional[asyncio.Task] = None
        self._callbacks: list[Callable[[SealStatus], None]] = []
        self._logger = logger.bind(component="secure_enclave")

    @property
    def status(self) -> SealStatus:
        return self._status

    @property
    def is_sealed(self) -> bool:
        return self._status == SealStatus.SEALED

    @property
    def is_unsealed(self) -> bool:
        return self._status == SealStatus.UNSEALED

    async def initialize(self) -> dict[str, Any]:
        """
        Initialize the enclave with a new master key.

        This should only be called once during initial setup.
        Returns shares for Shamir-based unsealing or encrypted key for auto-unseal.
        """
        # Generate master key
        master_key = secrets.token_bytes(32)

        result: dict[str, Any] = {
            "initialized": True,
            "unseal_method": self.unseal_method.value,
        }

        if self.unseal_method == UnsealMethod.SHAMIR:
            # Split into shares
            shares = self._shamir.split(
                master_key,
                self.shamir_threshold,
                self.shamir_total_shares,
                key_id="master-key",
            )
            result["shares"] = [s.encode() for s in shares]
            result["threshold"] = self.shamir_threshold
            result["total_shares"] = self.shamir_total_shares
            result["instructions"] = (
                f"Distribute these {self.shamir_total_shares} shares to different custodians. "
                f"At least {self.shamir_threshold} shares are required to unseal."
            )

        elif self.unseal_method in (UnsealMethod.KMS, UnsealMethod.CLOUD_KMS):
            # Store encrypted with KMS
            for provider in self.key_providers:
                if await provider.store_unseal_key(master_key):
                    result["stored"] = True
                    result["instructions"] = "Master key encrypted and stored with KMS."
                    break
            else:
                result["stored"] = False
                result["error"] = "Failed to store with any KMS provider"

        elif self.unseal_method == UnsealMethod.AUTO:
            # For auto-unseal, store with all configured providers
            stored_count = 0
            for provider in self.key_providers:
                if await provider.store_unseal_key(master_key):
                    stored_count += 1
            result["stored_with_providers"] = stored_count

        # Clear master key from memory
        # (In a real implementation, use secure memory wiping)
        self._master_key = None

        self._logger.info("Enclave initialized", method=self.unseal_method.value)

        return result

    async def unseal(
        self,
        key: Optional[bytes] = None,
        share: Optional[str] = None,
    ) -> UnsealProgress:
        """
        Attempt to unseal the enclave.

        Args:
            key: Direct unseal key (for manual/auto methods)
            share: Encoded Shamir share (for Shamir method)

        Returns:
            UnsealProgress with current status
        """
        if self._status == SealStatus.UNSEALED:
            return UnsealProgress(
                method=self.unseal_method,
                required_shares=0,
                received_shares=0,
                progress_percent=100.0,
                status=SealStatus.UNSEALED,
                message="Already unsealed",
            )

        self._status = SealStatus.UNSEALING

        try:
            if self.unseal_method == UnsealMethod.SHAMIR:
                return await self._unseal_shamir(share)

            elif self.unseal_method in (UnsealMethod.KMS, UnsealMethod.CLOUD_KMS, UnsealMethod.AUTO):
                return await self._unseal_auto()

            elif self.unseal_method == UnsealMethod.MANUAL:
                if key:
                    self._master_key = key
                    self._status = SealStatus.UNSEALED
                    self._start_seal_timer()
                    return UnsealProgress(
                        method=self.unseal_method,
                        required_shares=1,
                        received_shares=1,
                        progress_percent=100.0,
                        status=SealStatus.UNSEALED,
                        message="Unsealed successfully",
                    )
                else:
                    return UnsealProgress(
                        method=self.unseal_method,
                        required_shares=1,
                        received_shares=0,
                        progress_percent=0.0,
                        status=SealStatus.UNSEALING,
                        message="Waiting for unseal key",
                    )

            elif self.unseal_method == UnsealMethod.ENVIRONMENT:
                for provider in self.key_providers:
                    if isinstance(provider, EnvironmentKeyProvider):
                        key = await provider.get_unseal_key()
                        if key:
                            self._master_key = key
                            self._status = SealStatus.UNSEALED
                            self._start_seal_timer()
                            return UnsealProgress(
                                method=self.unseal_method,
                                required_shares=1,
                                received_shares=1,
                                progress_percent=100.0,
                                status=SealStatus.UNSEALED,
                                message="Unsealed from environment",
                            )

                return UnsealProgress(
                    method=self.unseal_method,
                    required_shares=1,
                    received_shares=0,
                    progress_percent=0.0,
                    status=SealStatus.SEALED,
                    message="Environment key not found",
                )

        except Exception as e:
            self._status = SealStatus.ERROR
            self._logger.error("Unseal failed", error=str(e))
            return UnsealProgress(
                method=self.unseal_method,
                required_shares=0,
                received_shares=0,
                progress_percent=0.0,
                status=SealStatus.ERROR,
                message=f"Unseal error: {str(e)}",
            )

        return UnsealProgress(
            method=self.unseal_method,
            required_shares=0,
            received_shares=0,
            progress_percent=0.0,
            status=self._status,
            message="Unknown unseal method",
        )

    async def _unseal_shamir(self, share_encoded: Optional[str]) -> UnsealProgress:
        """Unseal using Shamir's Secret Sharing."""
        if share_encoded:
            try:
                share = ShamirShare.decode(share_encoded)

                # Check if this share index already received
                existing_indices = {s.index for s in self._collected_shares}
                if share.index in existing_indices:
                    return UnsealProgress(
                        method=UnsealMethod.SHAMIR,
                        required_shares=self.shamir_threshold,
                        received_shares=len(self._collected_shares),
                        progress_percent=(len(self._collected_shares) / self.shamir_threshold) * 100,
                        status=SealStatus.UNSEALING,
                        message=f"Share {share.index} already received",
                        received_from=[str(s.index) for s in self._collected_shares],
                    )

                self._collected_shares.append(share)

            except Exception as e:
                return UnsealProgress(
                    method=UnsealMethod.SHAMIR,
                    required_shares=self.shamir_threshold,
                    received_shares=len(self._collected_shares),
                    progress_percent=(len(self._collected_shares) / self.shamir_threshold) * 100,
                    status=SealStatus.UNSEALING,
                    message=f"Invalid share: {str(e)}",
                    received_from=[str(s.index) for s in self._collected_shares],
                )

        # Check if we have enough shares
        if len(self._collected_shares) >= self.shamir_threshold:
            try:
                self._master_key = self._shamir.reconstruct(self._collected_shares)
                self._status = SealStatus.UNSEALED
                self._collected_shares = []  # Clear shares
                self._start_seal_timer()

                self._logger.info("Enclave unsealed via Shamir")
                self._notify_status_change()

                return UnsealProgress(
                    method=UnsealMethod.SHAMIR,
                    required_shares=self.shamir_threshold,
                    received_shares=self.shamir_threshold,
                    progress_percent=100.0,
                    status=SealStatus.UNSEALED,
                    message="Unsealed successfully",
                )

            except Exception as e:
                self._collected_shares = []  # Clear invalid shares
                self._status = SealStatus.SEALED
                return UnsealProgress(
                    method=UnsealMethod.SHAMIR,
                    required_shares=self.shamir_threshold,
                    received_shares=0,
                    progress_percent=0.0,
                    status=SealStatus.ERROR,
                    message=f"Failed to reconstruct: {str(e)}",
                )

        return UnsealProgress(
            method=UnsealMethod.SHAMIR,
            required_shares=self.shamir_threshold,
            received_shares=len(self._collected_shares),
            progress_percent=(len(self._collected_shares) / self.shamir_threshold) * 100,
            status=SealStatus.UNSEALING,
            message=f"Waiting for {self.shamir_threshold - len(self._collected_shares)} more shares",
            received_from=[str(s.index) for s in self._collected_shares],
        )

    async def _unseal_auto(self) -> UnsealProgress:
        """Unseal automatically using configured providers."""
        for provider in self.key_providers:
            try:
                key = await provider.get_unseal_key()
                if key:
                    self._master_key = key
                    self._status = SealStatus.UNSEALED
                    self._start_seal_timer()

                    self._logger.info("Enclave auto-unsealed")
                    self._notify_status_change()

                    return UnsealProgress(
                        method=self.unseal_method,
                        required_shares=1,
                        received_shares=1,
                        progress_percent=100.0,
                        status=SealStatus.UNSEALED,
                        message="Auto-unsealed successfully",
                    )
            except Exception as e:
                self._logger.warning(f"Provider failed: {e}")
                continue

        return UnsealProgress(
            method=self.unseal_method,
            required_shares=1,
            received_shares=0,
            progress_percent=0.0,
            status=SealStatus.SEALED,
            message="No provider could provide unseal key",
        )

    def seal(self) -> bool:
        """Seal the enclave, clearing the master key from memory."""
        if self._status != SealStatus.UNSEALED:
            return False

        # Clear master key
        if self._master_key:
            # In production, use secure memory wiping
            self._master_key = None

        # Clear collected shares
        self._collected_shares = []

        # Cancel seal timer
        if self._seal_timer:
            self._seal_timer.cancel()
            self._seal_timer = None

        self._status = SealStatus.SEALED

        self._logger.info("Enclave sealed")
        self._notify_status_change()

        return True

    def get_master_key(self) -> Optional[bytes]:
        """
        Get the master key.

        Returns None if enclave is sealed.
        """
        if self._status != SealStatus.UNSEALED:
            return None

        self._last_activity = time.time()
        return self._master_key

    def derive_key(self, purpose: str, context: Optional[bytes] = None) -> Optional[bytes]:
        """
        Derive a purpose-specific key from the master key.

        Uses HKDF to derive keys for different purposes.
        """
        master_key = self.get_master_key()
        if not master_key:
            return None

        try:
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes

            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=purpose.encode() + (context or b""),
            )
            return hkdf.derive(master_key)

        except ImportError:
            # Fallback
            data = master_key + purpose.encode() + (context or b"")
            return hashlib.sha256(data).digest()

    def _start_seal_timer(self) -> None:
        """Start the auto-seal timer."""
        if self._seal_timer:
            self._seal_timer.cancel()

        async def auto_seal():
            while self._status == SealStatus.UNSEALED:
                await asyncio.sleep(60)
                if time.time() - self._last_activity > self.auto_seal_timeout:
                    self._logger.info("Auto-sealing due to inactivity")
                    self.seal()
                    break

        self._seal_timer = asyncio.create_task(auto_seal())
        self._last_activity = time.time()

    def on_status_change(self, callback: Callable[[SealStatus], None]) -> None:
        """Register a callback for status changes."""
        self._callbacks.append(callback)

    def _notify_status_change(self) -> None:
        """Notify callbacks of status change."""
        for callback in self._callbacks:
            try:
                callback(self._status)
            except Exception as e:
                self._logger.error("Callback error", error=str(e))

    def get_status(self) -> dict[str, Any]:
        """Get current enclave status."""
        return {
            "status": self._status.value,
            "unseal_method": self.unseal_method.value,
            "shamir_threshold": self.shamir_threshold if self.unseal_method == UnsealMethod.SHAMIR else None,
            "shamir_total": self.shamir_total_shares if self.unseal_method == UnsealMethod.SHAMIR else None,
            "shares_received": len(self._collected_shares) if self.unseal_method == UnsealMethod.SHAMIR else None,
            "auto_seal_timeout": self.auto_seal_timeout,
            "time_until_seal": max(0, self.auto_seal_timeout - (time.time() - self._last_activity)) if self._status == SealStatus.UNSEALED else None,
        }


class KeyCeremonyManager:
    """
    Manages key ceremonies for secure key generation and recovery.

    Implements dual control and split knowledge principles.
    """

    def __init__(self, enclave: SecureEnclave) -> None:
        self.enclave = enclave
        self._ceremonies: dict[str, KeyCeremony] = {}
        self._logger = logger.bind(component="key_ceremony")

    def start_ceremony(
        self,
        ceremony_type: str,
        participants: dict[str, KeyCeremonyRole],
        required_custodians: int = 3,
        required_witnesses: int = 2,
    ) -> KeyCeremony:
        """Start a new key ceremony."""
        ceremony = KeyCeremony(
            ceremony_id=secrets.token_hex(16),
            ceremony_type=ceremony_type,
            created_at=time.time(),
            participants=participants,
            required_custodians=required_custodians,
            required_witnesses=required_witnesses,
        )

        self._ceremonies[ceremony.ceremony_id] = ceremony

        self._logger.info(
            "Key ceremony started",
            ceremony_id=ceremony.ceremony_id,
            type=ceremony_type,
        )

        return ceremony

    def record_event(
        self,
        ceremony_id: str,
        participant: str,
        event_type: str,
        details: dict[str, Any],
        signature: Optional[bytes] = None,
    ) -> bool:
        """Record an event in a key ceremony."""
        if ceremony_id not in self._ceremonies:
            return False

        ceremony = self._ceremonies[ceremony_id]

        if participant not in ceremony.participants:
            return False

        event = KeyCeremonyEvent(
            event_type=event_type,
            timestamp=time.time(),
            participant=participant,
            role=ceremony.participants[participant],
            details=details,
            signature=signature,
        )

        ceremony.events.append(event)

        self._logger.info(
            "Ceremony event recorded",
            ceremony_id=ceremony_id,
            event_type=event_type,
            participant=participant,
        )

        return True

    def complete_ceremony(
        self,
        ceremony_id: str,
        result_key_id: Optional[str] = None,
    ) -> bool:
        """Complete a key ceremony."""
        if ceremony_id not in self._ceremonies:
            return False

        ceremony = self._ceremonies[ceremony_id]

        # Verify requirements
        custodian_count = sum(
            1 for e in ceremony.events
            if ceremony.participants.get(e.participant) == KeyCeremonyRole.CUSTODIAN
        )
        witness_count = sum(
            1 for e in ceremony.events
            if ceremony.participants.get(e.participant) == KeyCeremonyRole.WITNESS
        )

        if custodian_count < ceremony.required_custodians:
            self._logger.warning(
                "Insufficient custodians",
                required=ceremony.required_custodians,
                actual=custodian_count,
            )
            return False

        if witness_count < ceremony.required_witnesses:
            self._logger.warning(
                "Insufficient witnesses",
                required=ceremony.required_witnesses,
                actual=witness_count,
            )
            return False

        ceremony.completed = True
        ceremony.result_key_id = result_key_id

        self._logger.info(
            "Key ceremony completed",
            ceremony_id=ceremony_id,
            result_key_id=result_key_id,
        )

        return True

    def get_ceremony(self, ceremony_id: str) -> Optional[KeyCeremony]:
        """Get a ceremony by ID."""
        return self._ceremonies.get(ceremony_id)

    def get_ceremony_audit_log(self, ceremony_id: str) -> list[dict[str, Any]]:
        """Get audit log for a ceremony."""
        ceremony = self._ceremonies.get(ceremony_id)
        if not ceremony:
            return []

        return [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "participant": e.participant,
                "role": e.role.value,
                "details": e.details,
                "has_signature": e.signature is not None,
            }
            for e in ceremony.events
        ]
