"""
WebAuthn/FIDO2 Passwordless Authentication Provider.

Implements the Web Authentication API (WebAuthn) for passwordless authentication
using hardware security keys, platform authenticators (Touch ID, Windows Hello),
and passkeys.

References:
- W3C WebAuthn Level 3: https://www.w3.org/TR/webauthn-3/
- FIDO2 CTAP2: https://fidoalliance.org/specs/fido-v2.1-ps-20210615/fido-client-to-authenticator-protocol-v2.1-ps-20210615.html
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

from aion.security.authentication.providers.base import AuthProvider, AuthResult
from aion.security.types import AuthMethod, Credentials, User

logger = structlog.get_logger()


class AuthenticatorAttachment(str, Enum):
    """Authenticator attachment modality."""
    PLATFORM = "platform"  # Built-in (Touch ID, Windows Hello, Face ID)
    CROSS_PLATFORM = "cross-platform"  # Roaming (USB keys, NFC, BLE)


class UserVerificationRequirement(str, Enum):
    """User verification requirement."""
    REQUIRED = "required"
    PREFERRED = "preferred"
    DISCOURAGED = "discouraged"


class ResidentKeyRequirement(str, Enum):
    """Resident key (discoverable credential) requirement."""
    REQUIRED = "required"
    PREFERRED = "preferred"
    DISCOURAGED = "discouraged"


class AttestationConveyancePreference(str, Enum):
    """Attestation conveyance preference."""
    NONE = "none"
    INDIRECT = "indirect"
    DIRECT = "direct"
    ENTERPRISE = "enterprise"


class PublicKeyCredentialType(str, Enum):
    """Public key credential type."""
    PUBLIC_KEY = "public-key"


class COSEAlgorithm(int, Enum):
    """COSE algorithm identifiers."""
    ES256 = -7      # ECDSA w/ SHA-256
    ES384 = -35     # ECDSA w/ SHA-384
    ES512 = -36     # ECDSA w/ SHA-512
    RS256 = -257    # RSASSA-PKCS1-v1_5 w/ SHA-256
    RS384 = -258    # RSASSA-PKCS1-v1_5 w/ SHA-384
    RS512 = -259    # RSASSA-PKCS1-v1_5 w/ SHA-512
    PS256 = -37     # RSASSA-PSS w/ SHA-256
    PS384 = -38     # RSASSA-PSS w/ SHA-384
    PS512 = -39     # RSASSA-PSS w/ SHA-512
    EDDSA = -8      # EdDSA


class AuthenticatorTransport(str, Enum):
    """Authenticator transport mechanisms."""
    USB = "usb"
    NFC = "nfc"
    BLE = "ble"
    SMART_CARD = "smart-card"
    HYBRID = "hybrid"
    INTERNAL = "internal"


@dataclass
class PublicKeyCredentialDescriptor:
    """Descriptor for a public key credential."""
    type: PublicKeyCredentialType
    id: bytes
    transports: list[AuthenticatorTransport] = field(default_factory=list)


@dataclass
class PublicKeyCredentialUserEntity:
    """User entity for credential creation."""
    id: bytes
    name: str
    display_name: str


@dataclass
class PublicKeyCredentialRpEntity:
    """Relying party entity."""
    id: str
    name: str
    icon: Optional[str] = None


@dataclass
class PublicKeyCredentialParameters:
    """Parameters for credential creation."""
    type: PublicKeyCredentialType
    alg: COSEAlgorithm


@dataclass
class AuthenticatorSelectionCriteria:
    """Criteria for authenticator selection."""
    authenticator_attachment: Optional[AuthenticatorAttachment] = None
    resident_key: ResidentKeyRequirement = ResidentKeyRequirement.PREFERRED
    require_resident_key: bool = False
    user_verification: UserVerificationRequirement = UserVerificationRequirement.PREFERRED


@dataclass
class WebAuthnCredential:
    """Stored WebAuthn credential."""
    credential_id: bytes
    public_key: bytes
    public_key_alg: COSEAlgorithm
    sign_count: int
    user_id: str
    created_at: float
    last_used_at: Optional[float] = None
    transports: list[AuthenticatorTransport] = field(default_factory=list)
    aaguid: Optional[bytes] = None
    attestation_format: Optional[str] = None
    device_name: Optional[str] = None
    is_backup_eligible: bool = False
    is_backed_up: bool = False
    is_discoverable: bool = False


@dataclass
class RegistrationOptions:
    """WebAuthn registration options (PublicKeyCredentialCreationOptions)."""
    challenge: bytes
    rp: PublicKeyCredentialRpEntity
    user: PublicKeyCredentialUserEntity
    pub_key_cred_params: list[PublicKeyCredentialParameters]
    timeout: int = 60000  # milliseconds
    excludeCredentials: list[PublicKeyCredentialDescriptor] = field(default_factory=list)
    authenticator_selection: Optional[AuthenticatorSelectionCriteria] = None
    attestation: AttestationConveyancePreference = AttestationConveyancePreference.NONE
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary for client."""
        result = {
            "challenge": base64.urlsafe_b64encode(self.challenge).rstrip(b"=").decode(),
            "rp": {"id": self.rp.id, "name": self.rp.name},
            "user": {
                "id": base64.urlsafe_b64encode(self.user.id).rstrip(b"=").decode(),
                "name": self.user.name,
                "displayName": self.user.display_name,
            },
            "pubKeyCredParams": [
                {"type": p.type.value, "alg": p.alg.value}
                for p in self.pub_key_cred_params
            ],
            "timeout": self.timeout,
            "attestation": self.attestation.value,
        }
        if self.excludeCredentials:
            result["excludeCredentials"] = [
                {
                    "type": c.type.value,
                    "id": base64.urlsafe_b64encode(c.id).rstrip(b"=").decode(),
                    "transports": [t.value for t in c.transports],
                }
                for c in self.excludeCredentials
            ]
        if self.authenticator_selection:
            sel = {}
            if self.authenticator_selection.authenticator_attachment:
                sel["authenticatorAttachment"] = self.authenticator_selection.authenticator_attachment.value
            sel["residentKey"] = self.authenticator_selection.resident_key.value
            sel["requireResidentKey"] = self.authenticator_selection.require_resident_key
            sel["userVerification"] = self.authenticator_selection.user_verification.value
            result["authenticatorSelection"] = sel
        if self.extensions:
            result["extensions"] = self.extensions
        return result


@dataclass
class AuthenticationOptions:
    """WebAuthn authentication options (PublicKeyCredentialRequestOptions)."""
    challenge: bytes
    timeout: int = 60000
    rp_id: Optional[str] = None
    allow_credentials: list[PublicKeyCredentialDescriptor] = field(default_factory=list)
    user_verification: UserVerificationRequirement = UserVerificationRequirement.PREFERRED
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary for client."""
        result = {
            "challenge": base64.urlsafe_b64encode(self.challenge).rstrip(b"=").decode(),
            "timeout": self.timeout,
            "userVerification": self.user_verification.value,
        }
        if self.rp_id:
            result["rpId"] = self.rp_id
        if self.allow_credentials:
            result["allowCredentials"] = [
                {
                    "type": c.type.value,
                    "id": base64.urlsafe_b64encode(c.id).rstrip(b"=").decode(),
                    "transports": [t.value for t in c.transports],
                }
                for c in self.allow_credentials
            ]
        if self.extensions:
            result["extensions"] = self.extensions
        return result


@dataclass
class RegistrationResponse:
    """Client response to registration challenge."""
    id: str
    raw_id: bytes
    response: dict[str, Any]
    authenticator_attachment: Optional[AuthenticatorAttachment] = None
    client_extension_results: dict[str, Any] = field(default_factory=dict)
    type: PublicKeyCredentialType = PublicKeyCredentialType.PUBLIC_KEY


@dataclass
class AuthenticationResponse:
    """Client response to authentication challenge."""
    id: str
    raw_id: bytes
    response: dict[str, Any]
    authenticator_attachment: Optional[AuthenticatorAttachment] = None
    client_extension_results: dict[str, Any] = field(default_factory=dict)
    type: PublicKeyCredentialType = PublicKeyCredentialType.PUBLIC_KEY


@dataclass
class WebAuthnConfig:
    """WebAuthn configuration."""
    rp_id: str
    rp_name: str
    rp_icon: Optional[str] = None
    origin: str = ""
    challenge_timeout: int = 300  # seconds
    supported_algorithms: list[COSEAlgorithm] = field(default_factory=lambda: [
        COSEAlgorithm.ES256,
        COSEAlgorithm.RS256,
        COSEAlgorithm.EDDSA,
        COSEAlgorithm.PS256,
    ])
    authenticator_attachment: Optional[AuthenticatorAttachment] = None
    resident_key: ResidentKeyRequirement = ResidentKeyRequirement.PREFERRED
    user_verification: UserVerificationRequirement = UserVerificationRequirement.PREFERRED
    attestation: AttestationConveyancePreference = AttestationConveyancePreference.NONE
    allow_credential_reuse: bool = False


class WebAuthnProvider(AuthProvider):
    """
    WebAuthn/FIDO2 authentication provider.

    Implements passwordless authentication using the Web Authentication API.
    Supports hardware security keys, platform authenticators (biometrics),
    and passkeys (synced credentials).
    """

    def __init__(self, config: WebAuthnConfig) -> None:
        self.config = config
        self._credentials: dict[str, list[WebAuthnCredential]] = {}  # user_id -> credentials
        self._credential_index: dict[bytes, WebAuthnCredential] = {}  # credential_id -> credential
        self._pending_registrations: dict[bytes, tuple[str, float]] = {}  # challenge -> (user_id, expires)
        self._pending_authentications: dict[bytes, tuple[Optional[str], float]] = {}  # challenge -> (user_id, expires)
        self._logger = logger.bind(provider="webauthn")

    @property
    def method(self) -> AuthMethod:
        return AuthMethod.MFA  # WebAuthn can be primary or MFA

    async def authenticate(
        self,
        credentials: Credentials,
        context: Optional[dict[str, Any]] = None,
    ) -> AuthResult:
        """
        Authenticate using WebAuthn.

        The credentials should contain the authentication response from the client.
        """
        context = context or {}

        try:
            auth_response_data = credentials.metadata.get("webauthn_response")
            if not auth_response_data:
                return AuthResult(
                    success=False,
                    error="Missing WebAuthn authentication response",
                )

            auth_response = self._parse_authentication_response(auth_response_data)
            result = await self.verify_authentication(auth_response, context)

            if result["verified"]:
                credential = result["credential"]
                return AuthResult(
                    success=True,
                    user_id=credential.user_id,
                    metadata={
                        "credential_id": base64.urlsafe_b64encode(credential.credential_id).decode(),
                        "authenticator_attachment": auth_response.authenticator_attachment.value if auth_response.authenticator_attachment else None,
                        "sign_count": credential.sign_count,
                    },
                )
            else:
                return AuthResult(
                    success=False,
                    error=result.get("error", "Authentication failed"),
                )
        except Exception as e:
            self._logger.error("WebAuthn authentication error", error=str(e))
            return AuthResult(
                success=False,
                error=f"Authentication error: {str(e)}",
            )

    async def validate(
        self,
        credentials: Credentials,
        context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Validate WebAuthn credentials exist for user."""
        user_id = credentials.identifier
        return user_id in self._credentials and len(self._credentials[user_id]) > 0

    def generate_registration_options(
        self,
        user_id: str,
        username: str,
        display_name: Optional[str] = None,
        authenticator_attachment: Optional[AuthenticatorAttachment] = None,
        resident_key: Optional[ResidentKeyRequirement] = None,
        user_verification: Optional[UserVerificationRequirement] = None,
        exclude_existing: bool = True,
    ) -> RegistrationOptions:
        """
        Generate registration options for a new credential.

        Returns options to be passed to navigator.credentials.create() on the client.
        """
        challenge = secrets.token_bytes(32)
        user_id_bytes = user_id.encode() if isinstance(user_id, str) else user_id

        # Store pending registration
        expires = time.time() + self.config.challenge_timeout
        self._pending_registrations[challenge] = (user_id, expires)

        # Get existing credentials to exclude
        exclude_credentials: list[PublicKeyCredentialDescriptor] = []
        if exclude_existing and user_id in self._credentials:
            for cred in self._credentials[user_id]:
                exclude_credentials.append(PublicKeyCredentialDescriptor(
                    type=PublicKeyCredentialType.PUBLIC_KEY,
                    id=cred.credential_id,
                    transports=cred.transports,
                ))

        # Build credential parameters
        pub_key_cred_params = [
            PublicKeyCredentialParameters(
                type=PublicKeyCredentialType.PUBLIC_KEY,
                alg=alg,
            )
            for alg in self.config.supported_algorithms
        ]

        # Build authenticator selection
        auth_selection = AuthenticatorSelectionCriteria(
            authenticator_attachment=authenticator_attachment or self.config.authenticator_attachment,
            resident_key=resident_key or self.config.resident_key,
            require_resident_key=(resident_key or self.config.resident_key) == ResidentKeyRequirement.REQUIRED,
            user_verification=user_verification or self.config.user_verification,
        )

        options = RegistrationOptions(
            challenge=challenge,
            rp=PublicKeyCredentialRpEntity(
                id=self.config.rp_id,
                name=self.config.rp_name,
                icon=self.config.rp_icon,
            ),
            user=PublicKeyCredentialUserEntity(
                id=user_id_bytes,
                name=username,
                display_name=display_name or username,
            ),
            pub_key_cred_params=pub_key_cred_params,
            excludeCredentials=exclude_credentials,
            authenticator_selection=auth_selection,
            attestation=self.config.attestation,
            extensions={
                "credProps": True,  # Request credential properties
            },
        )

        self._logger.info(
            "Generated registration options",
            user_id=user_id,
            challenge_hash=hashlib.sha256(challenge).hexdigest()[:16],
        )

        return options

    async def verify_registration(
        self,
        response: RegistrationResponse,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Verify a registration response from the client.

        Returns verification result with the created credential.
        """
        context = context or {}

        # Parse client data
        client_data_json = base64.urlsafe_b64decode(
            response.response["clientDataJSON"] + "=="
        )
        client_data = json.loads(client_data_json)

        # Verify challenge
        challenge = base64.urlsafe_b64decode(client_data["challenge"] + "==")
        if challenge not in self._pending_registrations:
            return {"verified": False, "error": "Unknown or expired challenge"}

        user_id, expires = self._pending_registrations[challenge]
        if time.time() > expires:
            del self._pending_registrations[challenge]
            return {"verified": False, "error": "Challenge expired"}

        # Verify origin
        origin = client_data.get("origin", "")
        if self.config.origin and origin != self.config.origin:
            return {"verified": False, "error": f"Origin mismatch: {origin}"}

        # Verify type
        if client_data.get("type") != "webauthn.create":
            return {"verified": False, "error": "Invalid client data type"}

        # Parse attestation object
        attestation_object = base64.urlsafe_b64decode(
            response.response["attestationObject"] + "=="
        )
        attestation = self._parse_cbor(attestation_object)

        # Parse authenticator data
        auth_data = attestation["authData"]
        auth_data_parsed = self._parse_authenticator_data(auth_data)

        # Verify RP ID hash
        expected_rp_id_hash = hashlib.sha256(self.config.rp_id.encode()).digest()
        if auth_data_parsed["rp_id_hash"] != expected_rp_id_hash:
            return {"verified": False, "error": "RP ID hash mismatch"}

        # Verify user present flag
        if not auth_data_parsed["flags"]["up"]:
            return {"verified": False, "error": "User not present"}

        # Verify user verified if required
        if self.config.user_verification == UserVerificationRequirement.REQUIRED:
            if not auth_data_parsed["flags"]["uv"]:
                return {"verified": False, "error": "User verification required but not performed"}

        # Extract credential data
        if "attested_credential_data" not in auth_data_parsed:
            return {"verified": False, "error": "Missing attested credential data"}

        cred_data = auth_data_parsed["attested_credential_data"]
        credential_id = cred_data["credential_id"]
        public_key = cred_data["credential_public_key"]
        aaguid = cred_data["aaguid"]

        # Check for credential reuse
        if not self.config.allow_credential_reuse and credential_id in self._credential_index:
            return {"verified": False, "error": "Credential already registered"}

        # Determine algorithm from public key
        public_key_alg = self._get_cose_algorithm(public_key)

        # Verify attestation (simplified - in production, verify attestation signature)
        attestation_format = attestation.get("fmt", "none")

        # Check extension results for credential properties
        is_discoverable = False
        if response.client_extension_results.get("credProps", {}).get("rk"):
            is_discoverable = True

        # Check backup state from flags
        is_backup_eligible = auth_data_parsed["flags"].get("be", False)
        is_backed_up = auth_data_parsed["flags"].get("bs", False)

        # Determine transports
        transports = []
        if response.authenticator_attachment == AuthenticatorAttachment.PLATFORM:
            transports.append(AuthenticatorTransport.INTERNAL)
        transports_hint = response.response.get("transports", [])
        for t in transports_hint:
            try:
                transports.append(AuthenticatorTransport(t))
            except ValueError:
                pass

        # Create credential
        credential = WebAuthnCredential(
            credential_id=credential_id,
            public_key=public_key,
            public_key_alg=public_key_alg,
            sign_count=auth_data_parsed["sign_count"],
            user_id=user_id,
            created_at=time.time(),
            transports=transports,
            aaguid=aaguid,
            attestation_format=attestation_format,
            is_backup_eligible=is_backup_eligible,
            is_backed_up=is_backed_up,
            is_discoverable=is_discoverable,
        )

        # Store credential
        if user_id not in self._credentials:
            self._credentials[user_id] = []
        self._credentials[user_id].append(credential)
        self._credential_index[credential_id] = credential

        # Clean up pending registration
        del self._pending_registrations[challenge]

        self._logger.info(
            "Credential registered",
            user_id=user_id,
            credential_id=base64.urlsafe_b64encode(credential_id).decode(),
            attestation_format=attestation_format,
            is_discoverable=is_discoverable,
        )

        return {
            "verified": True,
            "credential": credential,
        }

    def generate_authentication_options(
        self,
        user_id: Optional[str] = None,
        user_verification: Optional[UserVerificationRequirement] = None,
    ) -> AuthenticationOptions:
        """
        Generate authentication options.

        If user_id is provided, only allow that user's credentials.
        If user_id is None, allow discoverable credentials (passkeys).
        """
        challenge = secrets.token_bytes(32)
        expires = time.time() + self.config.challenge_timeout
        self._pending_authentications[challenge] = (user_id, expires)

        allow_credentials: list[PublicKeyCredentialDescriptor] = []
        if user_id and user_id in self._credentials:
            for cred in self._credentials[user_id]:
                allow_credentials.append(PublicKeyCredentialDescriptor(
                    type=PublicKeyCredentialType.PUBLIC_KEY,
                    id=cred.credential_id,
                    transports=cred.transports,
                ))

        options = AuthenticationOptions(
            challenge=challenge,
            rp_id=self.config.rp_id,
            allow_credentials=allow_credentials,
            user_verification=user_verification or self.config.user_verification,
            extensions={},
        )

        self._logger.info(
            "Generated authentication options",
            user_id=user_id,
            credential_count=len(allow_credentials),
            challenge_hash=hashlib.sha256(challenge).hexdigest()[:16],
        )

        return options

    async def verify_authentication(
        self,
        response: AuthenticationResponse,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Verify an authentication response from the client.
        """
        context = context or {}

        # Parse client data
        client_data_json = base64.urlsafe_b64decode(
            response.response["clientDataJSON"] + "=="
        )
        client_data = json.loads(client_data_json)

        # Verify challenge
        challenge = base64.urlsafe_b64decode(client_data["challenge"] + "==")
        if challenge not in self._pending_authentications:
            return {"verified": False, "error": "Unknown or expired challenge"}

        expected_user_id, expires = self._pending_authentications[challenge]
        if time.time() > expires:
            del self._pending_authentications[challenge]
            return {"verified": False, "error": "Challenge expired"}

        # Verify origin
        origin = client_data.get("origin", "")
        if self.config.origin and origin != self.config.origin:
            return {"verified": False, "error": f"Origin mismatch: {origin}"}

        # Verify type
        if client_data.get("type") != "webauthn.get":
            return {"verified": False, "error": "Invalid client data type"}

        # Look up credential
        credential_id = response.raw_id
        if credential_id not in self._credential_index:
            return {"verified": False, "error": "Unknown credential"}

        credential = self._credential_index[credential_id]

        # Verify user ID if specified
        if expected_user_id and credential.user_id != expected_user_id:
            return {"verified": False, "error": "Credential does not belong to user"}

        # Parse authenticator data
        authenticator_data = base64.urlsafe_b64decode(
            response.response["authenticatorData"] + "=="
        )
        auth_data_parsed = self._parse_authenticator_data(authenticator_data)

        # Verify RP ID hash
        expected_rp_id_hash = hashlib.sha256(self.config.rp_id.encode()).digest()
        if auth_data_parsed["rp_id_hash"] != expected_rp_id_hash:
            return {"verified": False, "error": "RP ID hash mismatch"}

        # Verify user present
        if not auth_data_parsed["flags"]["up"]:
            return {"verified": False, "error": "User not present"}

        # Verify user verification if required
        if self.config.user_verification == UserVerificationRequirement.REQUIRED:
            if not auth_data_parsed["flags"]["uv"]:
                return {"verified": False, "error": "User verification required"}

        # Verify signature
        signature = base64.urlsafe_b64decode(response.response["signature"] + "==")
        client_data_hash = hashlib.sha256(client_data_json).digest()
        signed_data = authenticator_data + client_data_hash

        if not self._verify_signature(
            credential.public_key,
            credential.public_key_alg,
            signed_data,
            signature,
        ):
            return {"verified": False, "error": "Invalid signature"}

        # Verify sign count (replay protection)
        new_sign_count = auth_data_parsed["sign_count"]
        if new_sign_count != 0 and credential.sign_count != 0:
            if new_sign_count <= credential.sign_count:
                self._logger.warning(
                    "Sign count regression detected - possible cloned authenticator",
                    credential_id=base64.urlsafe_b64encode(credential_id).decode(),
                    stored_count=credential.sign_count,
                    received_count=new_sign_count,
                )
                return {"verified": False, "error": "Sign count regression - possible cloned authenticator"}

        # Update credential
        credential.sign_count = new_sign_count
        credential.last_used_at = time.time()

        # Check backup state changes
        new_backup_state = auth_data_parsed["flags"].get("bs", False)
        if credential.is_backed_up != new_backup_state:
            self._logger.info(
                "Credential backup state changed",
                credential_id=base64.urlsafe_b64encode(credential_id).decode(),
                old_state=credential.is_backed_up,
                new_state=new_backup_state,
            )
            credential.is_backed_up = new_backup_state

        # Clean up pending authentication
        del self._pending_authentications[challenge]

        self._logger.info(
            "Authentication successful",
            user_id=credential.user_id,
            credential_id=base64.urlsafe_b64encode(credential_id).decode(),
            sign_count=new_sign_count,
        )

        return {
            "verified": True,
            "credential": credential,
            "user_id": credential.user_id,
        }

    def get_user_credentials(self, user_id: str) -> list[WebAuthnCredential]:
        """Get all credentials for a user."""
        return self._credentials.get(user_id, [])

    async def remove_credential(self, credential_id: bytes) -> bool:
        """Remove a credential."""
        if credential_id not in self._credential_index:
            return False

        credential = self._credential_index[credential_id]
        user_id = credential.user_id

        if user_id in self._credentials:
            self._credentials[user_id] = [
                c for c in self._credentials[user_id]
                if c.credential_id != credential_id
            ]

        del self._credential_index[credential_id]

        self._logger.info(
            "Credential removed",
            user_id=user_id,
            credential_id=base64.urlsafe_b64encode(credential_id).decode(),
        )

        return True

    def _parse_cbor(self, data: bytes) -> dict[str, Any]:
        """Parse CBOR-encoded data (simplified implementation)."""
        # In production, use a proper CBOR library like cbor2
        # This is a simplified parser for attestation objects
        try:
            import cbor2
            return cbor2.loads(data)
        except ImportError:
            # Fallback: attempt basic parsing
            # This won't work for all attestation formats
            raise NotImplementedError(
                "CBOR parsing requires cbor2 library. Install with: pip install cbor2"
            )

    def _parse_authenticator_data(self, data: bytes) -> dict[str, Any]:
        """Parse authenticator data structure."""
        if len(data) < 37:
            raise ValueError("Authenticator data too short")

        result: dict[str, Any] = {}

        # RP ID hash (32 bytes)
        result["rp_id_hash"] = data[:32]

        # Flags (1 byte)
        flags = data[32]
        result["flags"] = {
            "up": bool(flags & 0x01),      # User Present
            "uv": bool(flags & 0x04),      # User Verified
            "be": bool(flags & 0x08),      # Backup Eligible
            "bs": bool(flags & 0x10),      # Backup State
            "at": bool(flags & 0x40),      # Attested credential data included
            "ed": bool(flags & 0x80),      # Extension data included
        }

        # Sign count (4 bytes, big-endian)
        result["sign_count"] = struct.unpack(">I", data[33:37])[0]

        offset = 37

        # Attested credential data (if present)
        if result["flags"]["at"]:
            if len(data) < offset + 18:
                raise ValueError("Attested credential data too short")

            # AAGUID (16 bytes)
            aaguid = data[offset:offset + 16]
            offset += 16

            # Credential ID length (2 bytes, big-endian)
            cred_id_len = struct.unpack(">H", data[offset:offset + 2])[0]
            offset += 2

            # Credential ID
            if len(data) < offset + cred_id_len:
                raise ValueError("Credential ID truncated")
            credential_id = data[offset:offset + cred_id_len]
            offset += cred_id_len

            # Public key (COSE format, remaining bytes minus extensions)
            # In production, properly parse CBOR to find exact length
            public_key_data = data[offset:]

            result["attested_credential_data"] = {
                "aaguid": aaguid,
                "credential_id": credential_id,
                "credential_public_key": public_key_data,
            }

        return result

    def _get_cose_algorithm(self, public_key: bytes) -> COSEAlgorithm:
        """Determine COSE algorithm from public key."""
        try:
            import cbor2
            key_data = cbor2.loads(public_key)
            alg = key_data.get(3)  # COSE key algorithm field
            if alg:
                return COSEAlgorithm(alg)
        except Exception:
            pass
        return COSEAlgorithm.ES256  # Default

    def _verify_signature(
        self,
        public_key: bytes,
        algorithm: COSEAlgorithm,
        data: bytes,
        signature: bytes,
    ) -> bool:
        """Verify a signature using the credential's public key."""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec, padding
            from cryptography.hazmat.primitives.serialization import load_der_public_key
            import cbor2

            # Parse COSE key
            key_data = cbor2.loads(public_key)
            kty = key_data.get(1)  # Key type

            if kty == 2:  # EC2 key
                crv = key_data.get(-1)  # Curve
                x = key_data.get(-2)    # X coordinate
                y = key_data.get(-3)    # Y coordinate

                if crv == 1:  # P-256
                    from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, EllipticCurvePublicNumbers
                    curve = SECP256R1()
                elif crv == 2:  # P-384
                    from cryptography.hazmat.primitives.asymmetric.ec import SECP384R1, EllipticCurvePublicNumbers
                    curve = SECP384R1()
                elif crv == 3:  # P-521
                    from cryptography.hazmat.primitives.asymmetric.ec import SECP521R1, EllipticCurvePublicNumbers
                    curve = SECP521R1()
                else:
                    return False

                x_int = int.from_bytes(x, "big")
                y_int = int.from_bytes(y, "big")

                from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicNumbers
                public_numbers = EllipticCurvePublicNumbers(x_int, y_int, curve)
                ec_public_key = public_numbers.public_key()

                # Determine hash algorithm
                if algorithm in (COSEAlgorithm.ES256,):
                    hash_alg = hashes.SHA256()
                elif algorithm in (COSEAlgorithm.ES384,):
                    hash_alg = hashes.SHA384()
                elif algorithm in (COSEAlgorithm.ES512,):
                    hash_alg = hashes.SHA512()
                else:
                    hash_alg = hashes.SHA256()

                ec_public_key.verify(signature, data, ec.ECDSA(hash_alg))
                return True

            elif kty == 3:  # RSA key
                n = key_data.get(-1)  # Modulus
                e = key_data.get(-2)  # Exponent

                from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
                n_int = int.from_bytes(n, "big")
                e_int = int.from_bytes(e, "big")

                public_numbers = RSAPublicNumbers(e_int, n_int)
                rsa_public_key = public_numbers.public_key()

                if algorithm in (COSEAlgorithm.RS256, COSEAlgorithm.PS256):
                    hash_alg = hashes.SHA256()
                elif algorithm in (COSEAlgorithm.RS384, COSEAlgorithm.PS384):
                    hash_alg = hashes.SHA384()
                elif algorithm in (COSEAlgorithm.RS512, COSEAlgorithm.PS512):
                    hash_alg = hashes.SHA512()
                else:
                    hash_alg = hashes.SHA256()

                if algorithm in (COSEAlgorithm.PS256, COSEAlgorithm.PS384, COSEAlgorithm.PS512):
                    # PSS padding
                    rsa_public_key.verify(
                        signature,
                        data,
                        padding.PSS(
                            mgf=padding.MGF1(hash_alg),
                            salt_length=padding.PSS.AUTO,
                        ),
                        hash_alg,
                    )
                else:
                    # PKCS1v15 padding
                    rsa_public_key.verify(
                        signature,
                        data,
                        padding.PKCS1v15(),
                        hash_alg,
                    )
                return True

            return False

        except Exception as e:
            self._logger.error("Signature verification failed", error=str(e))
            return False

    def _parse_authentication_response(self, data: dict[str, Any]) -> AuthenticationResponse:
        """Parse authentication response from client data."""
        raw_id = base64.urlsafe_b64decode(data["rawId"] + "==")

        attachment = None
        if "authenticatorAttachment" in data:
            attachment = AuthenticatorAttachment(data["authenticatorAttachment"])

        return AuthenticationResponse(
            id=data["id"],
            raw_id=raw_id,
            response=data["response"],
            authenticator_attachment=attachment,
            client_extension_results=data.get("clientExtensionResults", {}),
        )

    def _parse_registration_response(self, data: dict[str, Any]) -> RegistrationResponse:
        """Parse registration response from client data."""
        raw_id = base64.urlsafe_b64decode(data["rawId"] + "==")

        attachment = None
        if "authenticatorAttachment" in data:
            attachment = AuthenticatorAttachment(data["authenticatorAttachment"])

        return RegistrationResponse(
            id=data["id"],
            raw_id=raw_id,
            response=data["response"],
            authenticator_attachment=attachment,
            client_extension_results=data.get("clientExtensionResults", {}),
        )


class PasskeyManager:
    """
    High-level manager for passkey operations.

    Provides a simplified interface for implementing passkey authentication
    with automatic credential discovery and cross-device authentication.
    """

    def __init__(self, webauthn_provider: WebAuthnProvider) -> None:
        self.provider = webauthn_provider
        self._logger = logger.bind(component="passkey_manager")

    async def start_registration(
        self,
        user_id: str,
        username: str,
        display_name: Optional[str] = None,
        prefer_passkey: bool = True,
    ) -> dict[str, Any]:
        """
        Start passkey registration for a user.

        Returns options to be passed to navigator.credentials.create().
        """
        options = self.provider.generate_registration_options(
            user_id=user_id,
            username=username,
            display_name=display_name,
            resident_key=ResidentKeyRequirement.REQUIRED if prefer_passkey else ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        )

        return options.to_json()

    async def complete_registration(
        self,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Complete passkey registration.

        Returns the created credential info.
        """
        parsed_response = self.provider._parse_registration_response(response)
        result = await self.provider.verify_registration(parsed_response)

        if result["verified"]:
            cred = result["credential"]
            return {
                "success": True,
                "credential_id": base64.urlsafe_b64encode(cred.credential_id).decode(),
                "is_passkey": cred.is_discoverable,
                "is_synced": cred.is_backed_up,
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
            }

    async def start_authentication(
        self,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Start passkey authentication.

        If user_id is None, allows discoverable credential authentication
        (user selects which passkey to use).
        """
        options = self.provider.generate_authentication_options(
            user_id=user_id,
            user_verification=UserVerificationRequirement.PREFERRED,
        )

        return options.to_json()

    async def complete_authentication(
        self,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Complete passkey authentication.

        Returns the authenticated user info.
        """
        parsed_response = self.provider._parse_authentication_response(response)
        result = await self.provider.verify_authentication(parsed_response)

        if result["verified"]:
            return {
                "success": True,
                "user_id": result["user_id"],
                "credential_id": base64.urlsafe_b64encode(
                    result["credential"].credential_id
                ).decode(),
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
            }

    def get_user_passkeys(self, user_id: str) -> list[dict[str, Any]]:
        """Get all passkeys for a user."""
        credentials = self.provider.get_user_credentials(user_id)
        return [
            {
                "credential_id": base64.urlsafe_b64encode(c.credential_id).decode(),
                "created_at": c.created_at,
                "last_used_at": c.last_used_at,
                "is_passkey": c.is_discoverable,
                "is_synced": c.is_backed_up,
                "device_name": c.device_name,
                "transports": [t.value for t in c.transports],
            }
            for c in credentials
        ]

    async def remove_passkey(self, credential_id: str) -> bool:
        """Remove a passkey."""
        cred_bytes = base64.urlsafe_b64decode(credential_id + "==")
        return await self.provider.remove_credential(cred_bytes)
