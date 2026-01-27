"""
Cryptographic Code Signing for Plugins

Implements ED25519 signature generation and verification for
plugin authenticity and integrity.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class SignatureError(Exception):
    """Base exception for signature errors."""
    pass


class InvalidSignatureError(SignatureError):
    """Raised when signature verification fails."""
    pass


class ExpiredSignatureError(SignatureError):
    """Raised when signature has expired."""
    pass


class MissingSignatureError(SignatureError):
    """Raised when expected signature is missing."""
    pass


@dataclass
class SigningKey:
    """ED25519 signing key (private key)."""

    key_bytes: bytes
    key_id: str = ""
    created_at: float = field(default_factory=time.time)

    @classmethod
    def generate(cls, key_id: Optional[str] = None) -> "SigningKey":
        """Generate a new signing key."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            private_key = Ed25519PrivateKey.generate()
            key_bytes = private_key.private_bytes_raw()
        except ImportError:
            # Fallback to nacl if cryptography not available
            try:
                import nacl.signing
                signing_key = nacl.signing.SigningKey.generate()
                key_bytes = bytes(signing_key)
            except ImportError:
                # Generate random bytes as placeholder (signing will fail)
                logger.warning("No cryptography library available for key generation")
                key_bytes = os.urandom(32)

        return cls(
            key_bytes=key_bytes,
            key_id=key_id or hashlib.sha256(key_bytes).hexdigest()[:16],
        )

    def get_verifying_key(self) -> "VerifyingKey":
        """Get the corresponding verifying (public) key."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            private_key = Ed25519PrivateKey.from_private_bytes(self.key_bytes)
            public_key = private_key.public_key()
            public_bytes = public_key.public_bytes_raw()
        except ImportError:
            try:
                import nacl.signing
                signing_key = nacl.signing.SigningKey(self.key_bytes)
                public_bytes = bytes(signing_key.verify_key)
            except ImportError:
                public_bytes = b""

        return VerifyingKey(
            key_bytes=public_bytes,
            key_id=self.key_id,
        )

    def sign(self, data: bytes) -> bytes:
        """Sign data and return signature."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            private_key = Ed25519PrivateKey.from_private_bytes(self.key_bytes)
            return private_key.sign(data)
        except ImportError:
            try:
                import nacl.signing
                signing_key = nacl.signing.SigningKey(self.key_bytes)
                signed = signing_key.sign(data)
                return signed.signature
            except ImportError:
                raise SignatureError("No cryptography library available")

    def to_pem(self) -> str:
        """Export key to PEM format."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
            private_key = Ed25519PrivateKey.from_private_bytes(self.key_bytes)
            return private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode()
        except ImportError:
            return base64.b64encode(self.key_bytes).decode()

    @classmethod
    def from_pem(cls, pem: str, key_id: Optional[str] = None) -> "SigningKey":
        """Load key from PEM format."""
        try:
            from cryptography.hazmat.primitives import serialization
            private_key = serialization.load_pem_private_key(
                pem.encode(),
                password=None,
            )
            key_bytes = private_key.private_bytes_raw()
        except ImportError:
            key_bytes = base64.b64decode(pem)

        return cls(
            key_bytes=key_bytes,
            key_id=key_id or hashlib.sha256(key_bytes).hexdigest()[:16],
        )


@dataclass
class VerifyingKey:
    """ED25519 verifying key (public key)."""

    key_bytes: bytes
    key_id: str = ""
    trusted: bool = False
    trust_level: str = "unknown"  # unknown, community, official, system

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify signature on data."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.exceptions import InvalidSignature
            public_key = Ed25519PublicKey.from_public_bytes(self.key_bytes)
            try:
                public_key.verify(signature, data)
                return True
            except InvalidSignature:
                return False
        except ImportError:
            try:
                import nacl.signing
                import nacl.exceptions
                verify_key = nacl.signing.VerifyKey(self.key_bytes)
                try:
                    verify_key.verify(data, signature)
                    return True
                except nacl.exceptions.BadSignature:
                    return False
            except ImportError:
                raise SignatureError("No cryptography library available")

    def to_pem(self) -> str:
        """Export key to PEM format."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.hazmat.primitives import serialization
            public_key = Ed25519PublicKey.from_public_bytes(self.key_bytes)
            return public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode()
        except ImportError:
            return base64.b64encode(self.key_bytes).decode()

    @classmethod
    def from_pem(
        cls,
        pem: str,
        key_id: Optional[str] = None,
        trusted: bool = False,
        trust_level: str = "unknown",
    ) -> "VerifyingKey":
        """Load key from PEM format."""
        try:
            from cryptography.hazmat.primitives import serialization
            public_key = serialization.load_pem_public_key(pem.encode())
            key_bytes = public_key.public_bytes_raw()
        except ImportError:
            key_bytes = base64.b64decode(pem)

        return cls(
            key_bytes=key_bytes,
            key_id=key_id or hashlib.sha256(key_bytes).hexdigest()[:16],
            trusted=trusted,
            trust_level=trust_level,
        )


@dataclass
class SignatureInfo:
    """Information about a plugin signature."""

    plugin_id: str
    version: str
    signature: bytes
    key_id: str
    timestamp: float
    expires_at: Optional[float] = None
    algorithm: str = "ed25519"
    content_hash: str = ""  # SHA-256 of plugin content
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if signature has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plugin_id": self.plugin_id,
            "version": self.version,
            "signature": base64.b64encode(self.signature).decode(),
            "key_id": self.key_id,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "algorithm": self.algorithm,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignatureInfo":
        """Create from dictionary."""
        return cls(
            plugin_id=data["plugin_id"],
            version=data["version"],
            signature=base64.b64decode(data["signature"]),
            key_id=data["key_id"],
            timestamp=data["timestamp"],
            expires_at=data.get("expires_at"),
            algorithm=data.get("algorithm", "ed25519"),
            content_hash=data.get("content_hash", ""),
            metadata=data.get("metadata", {}),
        )


class PluginSigner:
    """
    Signs plugins for distribution.

    Creates cryptographic signatures that verify plugin
    authenticity and integrity.
    """

    def __init__(self, signing_key: SigningKey):
        self.signing_key = signing_key

    def sign_plugin(
        self,
        plugin_path: Union[str, Path],
        plugin_id: str,
        version: str,
        expires_in_days: Optional[int] = 365,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SignatureInfo:
        """
        Sign a plugin directory or file.

        Args:
            plugin_path: Path to plugin
            plugin_id: Plugin identifier
            version: Plugin version
            expires_in_days: Signature validity period
            metadata: Additional metadata

        Returns:
            Signature information
        """
        plugin_path = Path(plugin_path)

        # Calculate content hash
        content_hash = self._hash_plugin_content(plugin_path)

        # Create signature payload
        timestamp = time.time()
        expires_at = (
            timestamp + (expires_in_days * 86400)
            if expires_in_days else None
        )

        payload = {
            "plugin_id": plugin_id,
            "version": version,
            "content_hash": content_hash,
            "timestamp": timestamp,
            "expires_at": expires_at,
            "key_id": self.signing_key.key_id,
        }

        # Sign the payload
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = self.signing_key.sign(payload_bytes)

        return SignatureInfo(
            plugin_id=plugin_id,
            version=version,
            signature=signature,
            key_id=self.signing_key.key_id,
            timestamp=timestamp,
            expires_at=expires_at,
            content_hash=content_hash,
            metadata=metadata or {},
        )

    def sign_manifest(
        self,
        manifest: dict[str, Any],
        expires_in_days: Optional[int] = 365,
    ) -> SignatureInfo:
        """Sign a plugin manifest."""
        plugin_id = manifest.get("id", "")
        version = manifest.get("version", "")

        # Hash manifest content
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
        content_hash = hashlib.sha256(manifest_bytes).hexdigest()

        timestamp = time.time()
        expires_at = (
            timestamp + (expires_in_days * 86400)
            if expires_in_days else None
        )

        payload = {
            "plugin_id": plugin_id,
            "version": version,
            "content_hash": content_hash,
            "timestamp": timestamp,
            "expires_at": expires_at,
            "key_id": self.signing_key.key_id,
        }

        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = self.signing_key.sign(payload_bytes)

        return SignatureInfo(
            plugin_id=plugin_id,
            version=version,
            signature=signature,
            key_id=self.signing_key.key_id,
            timestamp=timestamp,
            expires_at=expires_at,
            content_hash=content_hash,
        )

    def _hash_plugin_content(self, plugin_path: Path) -> str:
        """Calculate SHA-256 hash of plugin content."""
        hasher = hashlib.sha256()

        if plugin_path.is_file():
            with open(plugin_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
        else:
            # Hash all files in directory
            for file_path in sorted(plugin_path.rglob("*")):
                if file_path.is_file():
                    # Include relative path in hash
                    rel_path = file_path.relative_to(plugin_path)
                    hasher.update(str(rel_path).encode())

                    with open(file_path, "rb") as f:
                        while chunk := f.read(8192):
                            hasher.update(chunk)

        return hasher.hexdigest()


class PluginVerifier:
    """
    Verifies plugin signatures.

    Checks that plugins are signed by trusted keys and
    haven't been tampered with.
    """

    def __init__(self):
        self._trusted_keys: dict[str, VerifyingKey] = {}
        self._trust_levels = {
            "system": 100,
            "official": 80,
            "community": 50,
            "unknown": 0,
        }

    def add_trusted_key(
        self,
        key: VerifyingKey,
        trust_level: str = "community",
    ) -> None:
        """Add a trusted public key."""
        key.trusted = True
        key.trust_level = trust_level
        self._trusted_keys[key.key_id] = key

    def remove_trusted_key(self, key_id: str) -> None:
        """Remove a trusted key."""
        self._trusted_keys.pop(key_id, None)

    def get_trusted_key(self, key_id: str) -> Optional[VerifyingKey]:
        """Get a trusted key by ID."""
        return self._trusted_keys.get(key_id)

    def verify_signature(
        self,
        signature_info: SignatureInfo,
        plugin_path: Optional[Union[str, Path]] = None,
        manifest: Optional[dict[str, Any]] = None,
        require_trusted: bool = True,
    ) -> tuple[bool, str]:
        """
        Verify a plugin signature.

        Args:
            signature_info: Signature to verify
            plugin_path: Path to plugin (for content verification)
            manifest: Plugin manifest (alternative to path)
            require_trusted: Require key to be in trusted list

        Returns:
            Tuple of (valid, message)
        """
        # Check expiration
        if signature_info.is_expired():
            return False, "Signature has expired"

        # Get verifying key
        key = self._trusted_keys.get(signature_info.key_id)
        if key is None:
            if require_trusted:
                return False, f"Unknown signing key: {signature_info.key_id}"
            # Allow verification with untrusted key for testing
            logger.warning(
                "Verifying with untrusted key",
                key_id=signature_info.key_id,
            )

        # Verify content hash if provided
        if plugin_path:
            plugin_path = Path(plugin_path)
            signer = PluginSigner(SigningKey(b""))  # Just for hashing
            actual_hash = signer._hash_plugin_content(plugin_path)
            if actual_hash != signature_info.content_hash:
                return False, "Content hash mismatch - plugin may be tampered"

        elif manifest:
            manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
            actual_hash = hashlib.sha256(manifest_bytes).hexdigest()
            if actual_hash != signature_info.content_hash:
                return False, "Manifest hash mismatch"

        # Verify signature
        if key:
            payload = {
                "plugin_id": signature_info.plugin_id,
                "version": signature_info.version,
                "content_hash": signature_info.content_hash,
                "timestamp": signature_info.timestamp,
                "expires_at": signature_info.expires_at,
                "key_id": signature_info.key_id,
            }
            payload_bytes = json.dumps(payload, sort_keys=True).encode()

            if not key.verify(payload_bytes, signature_info.signature):
                return False, "Invalid signature"

        return True, f"Valid signature from {signature_info.key_id}"

    def verify_plugin(
        self,
        plugin_path: Union[str, Path],
        require_signature: bool = True,
        min_trust_level: str = "community",
    ) -> tuple[bool, str, Optional[SignatureInfo]]:
        """
        Verify a plugin directory.

        Args:
            plugin_path: Path to plugin
            require_signature: Fail if no signature found
            min_trust_level: Minimum required trust level

        Returns:
            Tuple of (valid, message, signature_info)
        """
        plugin_path = Path(plugin_path)
        signature_file = plugin_path / "signature.json"

        if not signature_file.exists():
            if require_signature:
                return False, "No signature file found", None
            return True, "No signature (unsigned plugin)", None

        # Load signature
        try:
            with open(signature_file) as f:
                sig_data = json.load(f)
            signature_info = SignatureInfo.from_dict(sig_data)
        except Exception as e:
            return False, f"Invalid signature file: {e}", None

        # Check trust level
        key = self._trusted_keys.get(signature_info.key_id)
        if key:
            key_trust = self._trust_levels.get(key.trust_level, 0)
            required_trust = self._trust_levels.get(min_trust_level, 0)
            if key_trust < required_trust:
                return (
                    False,
                    f"Insufficient trust level: {key.trust_level} < {min_trust_level}",
                    signature_info,
                )

        # Verify signature
        valid, message = self.verify_signature(
            signature_info,
            plugin_path=plugin_path,
        )

        return valid, message, signature_info

    def get_stats(self) -> dict[str, Any]:
        """Get verifier statistics."""
        return {
            "trusted_keys": len(self._trusted_keys),
            "keys_by_trust_level": {
                level: sum(1 for k in self._trusted_keys.values() if k.trust_level == level)
                for level in self._trust_levels
            },
        }
