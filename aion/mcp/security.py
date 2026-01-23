"""
AION MCP Security Module

Production-grade security features:
- Request signing and verification (HMAC-SHA256)
- mTLS support for Vault and external services
- Automatic credential rotation with scheduling
- Security audit logging
- Secret scanning prevention
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import ssl
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)


# ============================================
# Request Signing
# ============================================

class SignatureAlgorithm(str, Enum):
    """Supported signature algorithms."""
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"
    ED25519 = "ed25519"


@dataclass
class SignedRequest:
    """A signed request with verification metadata."""
    method: str
    path: str
    body: Optional[bytes]
    timestamp: float
    nonce: str
    signature: str
    algorithm: SignatureAlgorithm
    key_id: str
    headers: Dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        return {
            "X-Signature": self.signature,
            "X-Signature-Algorithm": self.algorithm.value,
            "X-Signature-Timestamp": str(int(self.timestamp)),
            "X-Signature-Nonce": self.nonce,
            "X-Signature-KeyId": self.key_id,
            **self.headers,
        }

    @classmethod
    def from_headers(
        cls,
        method: str,
        path: str,
        body: Optional[bytes],
        headers: Dict[str, str],
    ) -> "SignedRequest":
        """Create from HTTP headers."""
        return cls(
            method=method,
            path=path,
            body=body,
            timestamp=float(headers.get("X-Signature-Timestamp", "0")),
            nonce=headers.get("X-Signature-Nonce", ""),
            signature=headers.get("X-Signature", ""),
            algorithm=SignatureAlgorithm(
                headers.get("X-Signature-Algorithm", "hmac-sha256")
            ),
            key_id=headers.get("X-Signature-KeyId", ""),
        )


class RequestSigner:
    """
    Signs HTTP requests using HMAC or asymmetric keys.

    Implements a signing scheme similar to AWS Signature Version 4.
    """

    def __init__(
        self,
        key_id: str,
        secret_key: bytes,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
        timestamp_tolerance: float = 300.0,  # 5 minutes
    ):
        """
        Initialize request signer.

        Args:
            key_id: Key identifier
            secret_key: Secret key for signing
            algorithm: Signature algorithm
            timestamp_tolerance: Max age of valid signatures
        """
        self.key_id = key_id
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.timestamp_tolerance = timestamp_tolerance

        self._used_nonces: Set[str] = set()
        self._nonce_cleanup_time = time.time()

    def sign(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> SignedRequest:
        """
        Sign a request.

        Args:
            method: HTTP method
            path: Request path
            body: Request body
            headers: Additional headers to include in signature

        Returns:
            SignedRequest with signature
        """
        timestamp = time.time()
        nonce = secrets.token_hex(16)

        # Build canonical request
        canonical = self._build_canonical_request(
            method=method,
            path=path,
            body=body,
            timestamp=timestamp,
            nonce=nonce,
            headers=headers or {},
        )

        # Sign
        signature = self._sign_string(canonical)

        return SignedRequest(
            method=method,
            path=path,
            body=body,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            algorithm=self.algorithm,
            key_id=self.key_id,
            headers=headers or {},
        )

    def verify(self, request: SignedRequest) -> bool:
        """
        Verify a signed request.

        Args:
            request: Signed request to verify

        Returns:
            True if valid
        """
        # Check timestamp
        age = abs(time.time() - request.timestamp)
        if age > self.timestamp_tolerance:
            logger.warning(
                "Request signature expired",
                age=age,
                tolerance=self.timestamp_tolerance,
            )
            return False

        # Check nonce (replay protection)
        if request.nonce in self._used_nonces:
            logger.warning("Replay attack detected", nonce=request.nonce)
            return False

        # Cleanup old nonces periodically
        if time.time() - self._nonce_cleanup_time > self.timestamp_tolerance:
            self._used_nonces.clear()
            self._nonce_cleanup_time = time.time()

        # Build canonical request
        canonical = self._build_canonical_request(
            method=request.method,
            path=request.path,
            body=request.body,
            timestamp=request.timestamp,
            nonce=request.nonce,
            headers=request.headers,
        )

        # Verify signature
        expected = self._sign_string(canonical)

        if not hmac.compare_digest(expected, request.signature):
            logger.warning("Invalid request signature")
            return False

        # Record nonce
        self._used_nonces.add(request.nonce)

        return True

    def _build_canonical_request(
        self,
        method: str,
        path: str,
        body: Optional[bytes],
        timestamp: float,
        nonce: str,
        headers: Dict[str, str],
    ) -> str:
        """Build canonical string to sign."""
        # Body hash
        body_hash = hashlib.sha256(body or b"").hexdigest()

        # Canonical headers
        canonical_headers = "\n".join(
            f"{k.lower()}:{v}"
            for k, v in sorted(headers.items())
        )

        # Build canonical request
        parts = [
            method.upper(),
            path,
            str(int(timestamp)),
            nonce,
            body_hash,
            canonical_headers,
        ]

        return "\n".join(parts)

    def _sign_string(self, string: str) -> str:
        """Sign a string."""
        if self.algorithm == SignatureAlgorithm.HMAC_SHA256:
            signature = hmac.new(
                self.secret_key,
                string.encode(),
                hashlib.sha256,
            ).digest()
        elif self.algorithm == SignatureAlgorithm.HMAC_SHA512:
            signature = hmac.new(
                self.secret_key,
                string.encode(),
                hashlib.sha512,
            ).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        return base64.b64encode(signature).decode()


# ============================================
# mTLS Configuration
# ============================================

@dataclass
class MTLSConfig:
    """Mutual TLS configuration."""
    enabled: bool = False
    ca_cert: Optional[Path] = None
    client_cert: Optional[Path] = None
    client_key: Optional[Path] = None
    verify_hostname: bool = True
    check_hostname: bool = True
    min_version: str = "TLSv1_2"

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for mTLS."""
        # Determine minimum TLS version
        min_versions = {
            "TLSv1_2": ssl.TLSVersion.TLSv1_2,
            "TLSv1_3": ssl.TLSVersion.TLSv1_3,
        }
        min_ver = min_versions.get(self.min_version, ssl.TLSVersion.TLSv1_2)

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = min_ver
        context.check_hostname = self.check_hostname
        context.verify_mode = ssl.CERT_REQUIRED if self.verify_hostname else ssl.CERT_NONE

        # Load CA certificate
        if self.ca_cert and self.ca_cert.exists():
            context.load_verify_locations(str(self.ca_cert))

        # Load client certificate and key
        if self.client_cert and self.client_key:
            if self.client_cert.exists() and self.client_key.exists():
                context.load_cert_chain(
                    certfile=str(self.client_cert),
                    keyfile=str(self.client_key),
                )

        return context


class MTLSClientFactory:
    """
    Factory for creating mTLS-enabled HTTP clients.

    Supports:
    - HashiCorp Vault with mTLS
    - External MCP servers with mTLS
    - Certificate rotation
    """

    def __init__(self, config: MTLSConfig):
        """
        Initialize factory.

        Args:
            config: mTLS configuration
        """
        self.config = config
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._context_created_at: float = 0
        self._context_ttl: float = 3600  # Refresh context hourly

    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Get SSL context, refreshing if needed."""
        if not self.config.enabled:
            return None

        now = time.time()
        if (
            not self._ssl_context
            or now - self._context_created_at > self._context_ttl
        ):
            self._ssl_context = self.config.create_ssl_context()
            self._context_created_at = now
            logger.debug("SSL context refreshed")

        return self._ssl_context

    async def create_aiohttp_connector(self) -> Any:
        """Create aiohttp TCPConnector with mTLS."""
        try:
            import aiohttp

            ssl_context = self.get_ssl_context()
            return aiohttp.TCPConnector(ssl=ssl_context)
        except ImportError:
            raise ImportError("aiohttp required for HTTP client")


# ============================================
# Credential Rotation
# ============================================

class RotationStrategy(str, Enum):
    """Credential rotation strategies."""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    ON_DEMAND = "on_demand"
    HYBRID = "hybrid"


@dataclass
class RotationConfig:
    """Configuration for credential rotation."""
    strategy: RotationStrategy = RotationStrategy.TIME_BASED
    rotation_interval: timedelta = timedelta(hours=24)
    max_usage_count: int = 10000
    grace_period: timedelta = timedelta(minutes=5)
    notify_before: timedelta = timedelta(hours=1)
    retry_on_failure: bool = True
    max_retries: int = 3


@dataclass
class CredentialMetadata:
    """Metadata for a rotatable credential."""
    credential_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    usage_count: int = 0
    last_used: Optional[datetime] = None
    version: int = 1
    is_primary: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


class CredentialRotator:
    """
    Automatic credential rotation with scheduling.

    Features:
    - Time-based rotation
    - Usage-based rotation
    - Graceful rotation with overlap
    - Rotation notifications
    - Audit logging
    """

    def __init__(
        self,
        config: RotationConfig,
        rotation_callback: Callable[[str], Any],
        notification_callback: Optional[Callable[[str, timedelta], Any]] = None,
    ):
        """
        Initialize rotator.

        Args:
            config: Rotation configuration
            rotation_callback: Called to generate new credential
            notification_callback: Called before rotation
        """
        self.config = config
        self.rotation_callback = rotation_callback
        self.notification_callback = notification_callback

        self._credentials: Dict[str, CredentialMetadata] = {}
        self._rotation_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start rotation scheduler."""
        self._rotation_task = asyncio.create_task(self._rotation_loop())
        logger.info("Credential rotation scheduler started")

    async def stop(self) -> None:
        """Stop rotation scheduler."""
        self._shutdown.set()
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass
        logger.info("Credential rotation scheduler stopped")

    async def register_credential(
        self,
        credential_id: str,
        expires_at: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> CredentialMetadata:
        """Register a credential for rotation."""
        async with self._lock:
            metadata = CredentialMetadata(
                credential_id=credential_id,
                created_at=datetime.now(),
                expires_at=expires_at or (
                    datetime.now() + self.config.rotation_interval
                ),
                tags=tags or {},
            )
            self._credentials[credential_id] = metadata

            logger.info(
                "Credential registered for rotation",
                credential_id=credential_id,
                expires_at=metadata.expires_at,
            )

            return metadata

    async def record_usage(self, credential_id: str) -> None:
        """Record credential usage."""
        async with self._lock:
            if credential_id in self._credentials:
                meta = self._credentials[credential_id]
                meta.usage_count += 1
                meta.last_used = datetime.now()

                # Check usage-based rotation
                if (
                    self.config.strategy in (
                        RotationStrategy.USAGE_BASED,
                        RotationStrategy.HYBRID,
                    )
                    and meta.usage_count >= self.config.max_usage_count
                ):
                    asyncio.create_task(self._rotate_credential(credential_id))

    async def force_rotate(self, credential_id: str) -> bool:
        """Force immediate rotation."""
        return await self._rotate_credential(credential_id)

    async def _rotation_loop(self) -> None:
        """Background rotation loop."""
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()

                async with self._lock:
                    for cred_id, meta in list(self._credentials.items()):
                        if not meta.expires_at:
                            continue

                        time_until_expiry = meta.expires_at - now

                        # Send notification
                        if (
                            self.notification_callback
                            and time_until_expiry <= self.config.notify_before
                            and time_until_expiry > self.config.grace_period
                        ):
                            try:
                                await self._run_callback(
                                    self.notification_callback,
                                    cred_id,
                                    time_until_expiry,
                                )
                            except Exception as e:
                                logger.error(
                                    "Rotation notification failed",
                                    credential_id=cred_id,
                                    error=str(e),
                                )

                        # Rotate if expired or in grace period
                        if time_until_expiry <= self.config.grace_period:
                            await self._rotate_credential(cred_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rotation loop error: {e}")

    async def _rotate_credential(self, credential_id: str) -> bool:
        """Rotate a single credential."""
        for attempt in range(self.config.max_retries):
            try:
                # Generate new credential
                new_credential = await self._run_callback(
                    self.rotation_callback,
                    credential_id,
                )

                # Update metadata
                async with self._lock:
                    if credential_id in self._credentials:
                        old_meta = self._credentials[credential_id]
                        self._credentials[credential_id] = CredentialMetadata(
                            credential_id=credential_id,
                            created_at=datetime.now(),
                            expires_at=datetime.now() + self.config.rotation_interval,
                            version=old_meta.version + 1,
                            tags=old_meta.tags,
                        )

                logger.info(
                    "Credential rotated successfully",
                    credential_id=credential_id,
                    attempt=attempt + 1,
                )

                # Audit log
                await self._audit_rotation(credential_id, success=True)

                return True

            except Exception as e:
                logger.error(
                    "Credential rotation failed",
                    credential_id=credential_id,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if not self.config.retry_on_failure:
                    break

                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        await self._audit_rotation(credential_id, success=False)
        return False

    async def _run_callback(self, callback: Callable, *args) -> Any:
        """Run callback (async or sync)."""
        result = callback(*args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _audit_rotation(
        self,
        credential_id: str,
        success: bool,
    ) -> None:
        """Log rotation event for audit."""
        logger.info(
            "AUDIT: Credential rotation",
            event="credential_rotation",
            credential_id=credential_id,
            success=success,
            timestamp=datetime.now().isoformat(),
        )

    def get_status(self) -> Dict[str, Any]:
        """Get rotation status for all credentials."""
        now = datetime.now()
        return {
            cred_id: {
                "created_at": meta.created_at.isoformat(),
                "expires_at": meta.expires_at.isoformat() if meta.expires_at else None,
                "time_until_expiry": (
                    (meta.expires_at - now).total_seconds()
                    if meta.expires_at else None
                ),
                "usage_count": meta.usage_count,
                "version": meta.version,
            }
            for cred_id, meta in self._credentials.items()
        }


# ============================================
# Secret Scanning Prevention
# ============================================

class SecretPattern:
    """Pattern for detecting secrets."""

    def __init__(
        self,
        name: str,
        pattern: str,
        description: str,
        severity: str = "high",
    ):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.description = description
        self.severity = severity


# Common secret patterns
DEFAULT_SECRET_PATTERNS = [
    SecretPattern(
        "aws_access_key",
        r"AKIA[0-9A-Z]{16}",
        "AWS Access Key ID",
    ),
    SecretPattern(
        "aws_secret_key",
        r"[A-Za-z0-9/+=]{40}",
        "AWS Secret Access Key (potential)",
        severity="medium",
    ),
    SecretPattern(
        "github_token",
        r"gh[pousr]_[A-Za-z0-9_]{36,}",
        "GitHub Token",
    ),
    SecretPattern(
        "github_pat",
        r"github_pat_[A-Za-z0-9_]{22,}",
        "GitHub Personal Access Token",
    ),
    SecretPattern(
        "slack_token",
        r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
        "Slack Token",
    ),
    SecretPattern(
        "stripe_key",
        r"sk_live_[A-Za-z0-9]{24,}",
        "Stripe Live Key",
    ),
    SecretPattern(
        "private_key",
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "Private Key",
    ),
    SecretPattern(
        "jwt_token",
        r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*",
        "JWT Token",
        severity="medium",
    ),
    SecretPattern(
        "api_key_generic",
        r"(?:api[_-]?key|apikey)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}",
        "Generic API Key",
        severity="medium",
    ),
    SecretPattern(
        "password_assignment",
        r"(?:password|passwd|pwd)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]",
        "Password Assignment",
        severity="high",
    ),
]


@dataclass
class SecretFinding:
    """A detected secret."""
    pattern_name: str
    description: str
    severity: str
    location: str
    line_number: Optional[int] = None
    masked_match: str = ""


class SecretScanner:
    """
    Scans content for secrets to prevent accidental exposure.

    Used to:
    - Validate tool outputs before caching
    - Check logs before emission
    - Scan responses before returning
    """

    def __init__(
        self,
        patterns: Optional[List[SecretPattern]] = None,
        custom_patterns: Optional[List[SecretPattern]] = None,
        allowlist: Optional[List[str]] = None,
    ):
        """
        Initialize scanner.

        Args:
            patterns: Override default patterns
            custom_patterns: Additional patterns
            allowlist: Patterns to ignore
        """
        self.patterns = patterns or DEFAULT_SECRET_PATTERNS
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        self.allowlist = [re.compile(p) for p in (allowlist or [])]

    def scan(
        self,
        content: str,
        location: str = "unknown",
    ) -> List[SecretFinding]:
        """
        Scan content for secrets.

        Args:
            content: Content to scan
            location: Location identifier for findings

        Returns:
            List of findings
        """
        findings = []

        for line_num, line in enumerate(content.split("\n"), 1):
            for pattern in self.patterns:
                matches = pattern.pattern.finditer(line)
                for match in matches:
                    # Check allowlist
                    if any(al.search(match.group()) for al in self.allowlist):
                        continue

                    # Mask the match for safe logging
                    matched = match.group()
                    masked = matched[:4] + "*" * (len(matched) - 8) + matched[-4:]

                    findings.append(SecretFinding(
                        pattern_name=pattern.name,
                        description=pattern.description,
                        severity=pattern.severity,
                        location=location,
                        line_number=line_num,
                        masked_match=masked,
                    ))

        return findings

    def scan_dict(
        self,
        data: Dict[str, Any],
        location: str = "unknown",
    ) -> List[SecretFinding]:
        """
        Scan dictionary for secrets.

        Args:
            data: Dictionary to scan
            location: Location identifier

        Returns:
            List of findings
        """
        findings = []

        def scan_value(value: Any, path: str) -> None:
            if isinstance(value, str):
                findings.extend(self.scan(value, f"{location}:{path}"))
            elif isinstance(value, dict):
                for k, v in value.items():
                    scan_value(v, f"{path}.{k}")
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    scan_value(v, f"{path}[{i}]")

        for key, value in data.items():
            scan_value(value, key)

        return findings

    def redact(self, content: str) -> str:
        """
        Redact secrets from content.

        Args:
            content: Content to redact

        Returns:
            Redacted content
        """
        result = content

        for pattern in self.patterns:
            def replacer(match):
                matched = match.group()
                return matched[:4] + "*" * (len(matched) - 8) + matched[-4:]

            result = pattern.pattern.sub(replacer, result)

        return result


# ============================================
# Security Audit Logger
# ============================================

class AuditEventType(str, Enum):
    """Types of security audit events."""
    CREDENTIAL_ACCESS = "credential_access"
    CREDENTIAL_ROTATION = "credential_rotation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    REQUEST_SIGNED = "request_signed"
    REQUEST_VERIFIED = "request_verified"
    SECRET_DETECTED = "secret_detected"
    CONFIG_CHANGE = "config_change"


@dataclass
class AuditEvent:
    """Security audit event."""
    event_type: AuditEventType
    timestamp: datetime
    actor: str
    action: str
    resource: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    source_ip: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "success": self.success,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "source_ip": self.source_ip,
        }


class SecurityAuditLogger:
    """
    Security-focused audit logger.

    Features:
    - Structured audit events
    - Correlation ID tracking
    - Tamper-evident logging (optional)
    - Async emission
    """

    def __init__(
        self,
        emitters: Optional[List[Callable[[AuditEvent], Any]]] = None,
        include_hash: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            emitters: List of event emitters
            include_hash: Include hash chain for tamper evidence
        """
        self.emitters = emitters or []
        self.include_hash = include_hash

        self._previous_hash: Optional[str] = None
        self._lock = asyncio.Lock()

    async def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        source_ip: Optional[str] = None,
    ) -> None:
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            actor=actor,
            action=action,
            resource=resource,
            success=success,
            details=details or {},
            correlation_id=correlation_id,
            source_ip=source_ip,
        )

        # Add hash chain if enabled
        if self.include_hash:
            async with self._lock:
                event_data = json.dumps(event.to_dict(), sort_keys=True)
                chain_data = f"{self._previous_hash or 'genesis'}:{event_data}"
                event_hash = hashlib.sha256(chain_data.encode()).hexdigest()
                event.details["_hash"] = event_hash
                event.details["_previous_hash"] = self._previous_hash
                self._previous_hash = event_hash

        # Emit to all emitters
        for emitter in self.emitters:
            try:
                result = emitter(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Audit emitter error: {e}")

        # Also log via structlog
        logger.info(
            "SECURITY_AUDIT",
            **event.to_dict(),
        )

    def add_emitter(self, emitter: Callable[[AuditEvent], Any]) -> None:
        """Add an event emitter."""
        self.emitters.append(emitter)


# ============================================
# Global Instances
# ============================================

_global_audit_logger: Optional[SecurityAuditLogger] = None
_global_secret_scanner: Optional[SecretScanner] = None


def get_audit_logger() -> SecurityAuditLogger:
    """Get global audit logger."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = SecurityAuditLogger()
    return _global_audit_logger


def get_secret_scanner() -> SecretScanner:
    """Get global secret scanner."""
    global _global_secret_scanner
    if _global_secret_scanner is None:
        _global_secret_scanner = SecretScanner()
    return _global_secret_scanner
