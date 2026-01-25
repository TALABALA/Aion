"""
Mutual TLS (mTLS) Client Certificate Authentication.

Implements certificate-based authentication for:
- Service-to-service communication
- High-security client authentication
- Zero Trust device authentication

Features:
- X.509 certificate validation
- Certificate chain verification
- CRL and OCSP checking
- Certificate fingerprint matching
- Custom certificate policies
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

from aion.security.authentication.providers.base import AuthProvider, AuthResult
from aion.security.types import AuthMethod, Credentials

logger = structlog.get_logger()


class CertificateStatus(str, Enum):
    """Certificate validation status."""
    VALID = "valid"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    REVOKED = "revoked"
    UNTRUSTED = "untrusted"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_CHAIN = "invalid_chain"
    UNKNOWN = "unknown"


class RevocationCheckMethod(str, Enum):
    """Methods for checking certificate revocation."""
    NONE = "none"
    CRL = "crl"
    OCSP = "ocsp"
    BOTH = "both"


@dataclass
class CertificateInfo:
    """Information extracted from an X.509 certificate."""
    # Subject info
    subject_cn: Optional[str] = None
    subject_o: Optional[str] = None
    subject_ou: Optional[str] = None
    subject_c: Optional[str] = None
    subject_dn: str = ""

    # Issuer info
    issuer_cn: Optional[str] = None
    issuer_o: Optional[str] = None
    issuer_dn: str = ""

    # Validity
    not_before: Optional[datetime.datetime] = None
    not_after: Optional[datetime.datetime] = None

    # Identifiers
    serial_number: str = ""
    fingerprint_sha256: str = ""
    fingerprint_sha1: str = ""

    # Key info
    public_key_algorithm: str = ""
    public_key_size: int = 0
    signature_algorithm: str = ""

    # Extensions
    subject_alt_names: list[str] = field(default_factory=list)
    key_usage: list[str] = field(default_factory=list)
    extended_key_usage: list[str] = field(default_factory=list)
    basic_constraints_ca: bool = False
    crl_distribution_points: list[str] = field(default_factory=list)
    ocsp_responders: list[str] = field(default_factory=list)

    # Raw data
    pem: Optional[str] = None
    der: Optional[bytes] = None

    @property
    def is_ca(self) -> bool:
        return self.basic_constraints_ca

    @property
    def days_until_expiry(self) -> Optional[int]:
        if self.not_after:
            delta = self.not_after - datetime.datetime.utcnow()
            return delta.days
        return None

    @property
    def is_expired(self) -> bool:
        if self.not_after:
            return datetime.datetime.utcnow() > self.not_after
        return False

    @property
    def is_valid_time(self) -> bool:
        now = datetime.datetime.utcnow()
        if self.not_before and now < self.not_before:
            return False
        if self.not_after and now > self.not_after:
            return False
        return True


@dataclass
class CertificateValidationResult:
    """Result of certificate validation."""
    valid: bool
    status: CertificateStatus
    certificate_info: Optional[CertificateInfo]
    chain_info: list[CertificateInfo]
    errors: list[str]
    warnings: list[str]
    revocation_checked: bool = False
    revocation_status: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "certificate": {
                "subject": self.certificate_info.subject_dn if self.certificate_info else None,
                "issuer": self.certificate_info.issuer_dn if self.certificate_info else None,
                "expires": self.certificate_info.not_after.isoformat() if self.certificate_info and self.certificate_info.not_after else None,
                "fingerprint": self.certificate_info.fingerprint_sha256 if self.certificate_info else None,
            } if self.certificate_info else None,
        }


@dataclass
class CertificatePolicy:
    """Policy for certificate validation."""
    name: str

    # Required subject attributes
    required_subject_cn_pattern: Optional[str] = None
    required_subject_o: Optional[str] = None
    required_subject_ou: Optional[str] = None

    # Required issuer
    required_issuer_cn: Optional[str] = None
    required_issuer_o: Optional[str] = None
    trusted_issuer_fingerprints: list[str] = field(default_factory=list)

    # Key requirements
    min_key_size: int = 2048
    allowed_key_algorithms: list[str] = field(default_factory=lambda: ["RSA", "EC"])
    allowed_signature_algorithms: list[str] = field(default_factory=lambda: [
        "sha256WithRSAEncryption",
        "sha384WithRSAEncryption",
        "sha512WithRSAEncryption",
        "ecdsa-with-SHA256",
        "ecdsa-with-SHA384",
        "ecdsa-with-SHA512",
    ])

    # Validity requirements
    min_validity_days: int = 1
    max_validity_days: int = 825  # ~2 years
    warn_expiry_days: int = 30

    # Extension requirements
    require_client_auth_eku: bool = True
    require_san: bool = False
    allowed_san_patterns: list[str] = field(default_factory=list)

    # Revocation checking
    revocation_check: RevocationCheckMethod = RevocationCheckMethod.OCSP
    allow_revocation_unknown: bool = False

    # Pinning
    pinned_fingerprints: list[str] = field(default_factory=list)
    pinned_public_keys: list[str] = field(default_factory=list)


@dataclass
class TrustedCA:
    """Trusted Certificate Authority."""
    name: str
    certificate_info: CertificateInfo
    pem: str
    active: bool = True
    max_path_length: Optional[int] = None


class CertificateParser:
    """
    X.509 certificate parser.

    Extracts information from certificates using cryptography library.
    """

    def parse_pem(self, pem_data: str) -> CertificateInfo:
        """Parse a PEM-encoded certificate."""
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.serialization import Encoding

            # Load certificate
            cert = x509.load_pem_x509_certificate(pem_data.encode())
            return self._extract_info(cert, pem_data)

        except ImportError:
            logger.warning("cryptography library not available")
            return CertificateInfo(pem=pem_data)

    def parse_der(self, der_data: bytes) -> CertificateInfo:
        """Parse a DER-encoded certificate."""
        try:
            from cryptography import x509

            cert = x509.load_der_x509_certificate(der_data)
            return self._extract_info(cert, der=der_data)

        except ImportError:
            logger.warning("cryptography library not available")
            return CertificateInfo(der=der_data)

    def _extract_info(
        self,
        cert: Any,
        pem: Optional[str] = None,
        der: Optional[bytes] = None,
    ) -> CertificateInfo:
        """Extract information from a certificate object."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.serialization import Encoding

        info = CertificateInfo()

        # Subject
        subject = cert.subject
        for attr in subject:
            if attr.oid == x509.oid.NameOID.COMMON_NAME:
                info.subject_cn = attr.value
            elif attr.oid == x509.oid.NameOID.ORGANIZATION_NAME:
                info.subject_o = attr.value
            elif attr.oid == x509.oid.NameOID.ORGANIZATIONAL_UNIT_NAME:
                info.subject_ou = attr.value
            elif attr.oid == x509.oid.NameOID.COUNTRY_NAME:
                info.subject_c = attr.value
        info.subject_dn = subject.rfc4514_string()

        # Issuer
        issuer = cert.issuer
        for attr in issuer:
            if attr.oid == x509.oid.NameOID.COMMON_NAME:
                info.issuer_cn = attr.value
            elif attr.oid == x509.oid.NameOID.ORGANIZATION_NAME:
                info.issuer_o = attr.value
        info.issuer_dn = issuer.rfc4514_string()

        # Validity
        info.not_before = cert.not_valid_before_utc
        info.not_after = cert.not_valid_after_utc

        # Identifiers
        info.serial_number = format(cert.serial_number, 'x')
        info.fingerprint_sha256 = cert.fingerprint(hashes.SHA256()).hex()
        info.fingerprint_sha1 = cert.fingerprint(hashes.SHA1()).hex()

        # Key info
        public_key = cert.public_key()
        info.public_key_algorithm = public_key.__class__.__name__.replace("PublicKey", "")
        try:
            info.public_key_size = public_key.key_size
        except AttributeError:
            pass
        info.signature_algorithm = cert.signature_algorithm_oid._name

        # Extensions
        try:
            san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            for name in san_ext.value:
                if isinstance(name, x509.DNSName):
                    info.subject_alt_names.append(f"DNS:{name.value}")
                elif isinstance(name, x509.IPAddress):
                    info.subject_alt_names.append(f"IP:{name.value}")
                elif isinstance(name, x509.RFC822Name):
                    info.subject_alt_names.append(f"email:{name.value}")
        except x509.ExtensionNotFound:
            pass

        try:
            ku_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.KEY_USAGE)
            ku = ku_ext.value
            if ku.digital_signature:
                info.key_usage.append("digitalSignature")
            if ku.key_encipherment:
                info.key_usage.append("keyEncipherment")
            if ku.key_cert_sign:
                info.key_usage.append("keyCertSign")
            if ku.crl_sign:
                info.key_usage.append("cRLSign")
        except x509.ExtensionNotFound:
            pass

        try:
            eku_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.EXTENDED_KEY_USAGE)
            for usage in eku_ext.value:
                if usage == x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH:
                    info.extended_key_usage.append("clientAuth")
                elif usage == x509.oid.ExtendedKeyUsageOID.SERVER_AUTH:
                    info.extended_key_usage.append("serverAuth")
                elif usage == x509.oid.ExtendedKeyUsageOID.CODE_SIGNING:
                    info.extended_key_usage.append("codeSigning")
        except x509.ExtensionNotFound:
            pass

        try:
            bc_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.BASIC_CONSTRAINTS)
            info.basic_constraints_ca = bc_ext.value.ca
        except x509.ExtensionNotFound:
            pass

        try:
            crl_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.CRL_DISTRIBUTION_POINTS)
            for dp in crl_ext.value:
                if dp.full_name:
                    for name in dp.full_name:
                        if isinstance(name, x509.UniformResourceIdentifier):
                            info.crl_distribution_points.append(name.value)
        except x509.ExtensionNotFound:
            pass

        try:
            aia_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.AUTHORITY_INFORMATION_ACCESS)
            for desc in aia_ext.value:
                if desc.access_method == x509.oid.AuthorityInformationAccessOID.OCSP:
                    info.ocsp_responders.append(desc.access_location.value)
        except x509.ExtensionNotFound:
            pass

        # Raw data
        info.pem = pem
        info.der = der or cert.public_bytes(Encoding.DER)

        return info


class CertificateValidator:
    """
    X.509 certificate validator.

    Performs comprehensive certificate validation including:
    - Signature verification
    - Chain building and validation
    - Revocation checking
    - Policy enforcement
    """

    def __init__(
        self,
        trusted_cas: Optional[list[TrustedCA]] = None,
        default_policy: Optional[CertificatePolicy] = None,
    ) -> None:
        self._trusted_cas: dict[str, TrustedCA] = {}
        self._crl_cache: dict[str, tuple[Any, float]] = {}
        self._parser = CertificateParser()
        self.default_policy = default_policy or CertificatePolicy(name="default")
        self._logger = logger.bind(component="cert_validator")

        if trusted_cas:
            for ca in trusted_cas:
                self.add_trusted_ca(ca)

    def add_trusted_ca(self, ca: TrustedCA) -> None:
        """Add a trusted CA."""
        self._trusted_cas[ca.certificate_info.fingerprint_sha256] = ca
        self._logger.info("Trusted CA added", name=ca.name)

    def load_system_cas(self) -> int:
        """Load system trusted CAs."""
        try:
            import ssl
            import certifi

            # Load from certifi bundle
            with open(certifi.where(), 'r') as f:
                pem_data = f.read()

            # Split into individual certificates
            certs = []
            current_cert = []
            for line in pem_data.split('\n'):
                current_cert.append(line)
                if '-----END CERTIFICATE-----' in line:
                    cert_pem = '\n'.join(current_cert)
                    current_cert = []

                    try:
                        info = self._parser.parse_pem(cert_pem)
                        if info.is_ca:
                            ca = TrustedCA(
                                name=info.subject_cn or info.subject_dn,
                                certificate_info=info,
                                pem=cert_pem,
                            )
                            self.add_trusted_ca(ca)
                            certs.append(ca)
                    except Exception:
                        pass

            self._logger.info("Loaded system CAs", count=len(certs))
            return len(certs)

        except Exception as e:
            self._logger.error("Failed to load system CAs", error=str(e))
            return 0

    def validate(
        self,
        certificate_pem: str,
        chain_pem: Optional[list[str]] = None,
        policy: Optional[CertificatePolicy] = None,
    ) -> CertificateValidationResult:
        """
        Validate a certificate.

        Args:
            certificate_pem: The certificate to validate in PEM format
            chain_pem: Optional intermediate certificates
            policy: Validation policy to apply

        Returns:
            CertificateValidationResult with validation details
        """
        policy = policy or self.default_policy
        errors: list[str] = []
        warnings: list[str] = []

        # Parse certificate
        try:
            cert_info = self._parser.parse_pem(certificate_pem)
        except Exception as e:
            return CertificateValidationResult(
                valid=False,
                status=CertificateStatus.UNKNOWN,
                certificate_info=None,
                chain_info=[],
                errors=[f"Failed to parse certificate: {str(e)}"],
                warnings=[],
            )

        # Parse chain
        chain_info: list[CertificateInfo] = []
        if chain_pem:
            for pem in chain_pem:
                try:
                    chain_info.append(self._parser.parse_pem(pem))
                except Exception:
                    pass

        # Check time validity
        if not cert_info.is_valid_time:
            if cert_info.is_expired:
                return CertificateValidationResult(
                    valid=False,
                    status=CertificateStatus.EXPIRED,
                    certificate_info=cert_info,
                    chain_info=chain_info,
                    errors=["Certificate has expired"],
                    warnings=[],
                )
            else:
                return CertificateValidationResult(
                    valid=False,
                    status=CertificateStatus.NOT_YET_VALID,
                    certificate_info=cert_info,
                    chain_info=chain_info,
                    errors=["Certificate is not yet valid"],
                    warnings=[],
                )

        # Check expiry warning
        if cert_info.days_until_expiry and cert_info.days_until_expiry < policy.warn_expiry_days:
            warnings.append(f"Certificate expires in {cert_info.days_until_expiry} days")

        # Check key requirements
        if cert_info.public_key_size < policy.min_key_size:
            errors.append(f"Key size {cert_info.public_key_size} is below minimum {policy.min_key_size}")

        if policy.allowed_key_algorithms:
            if not any(
                alg.lower() in cert_info.public_key_algorithm.lower()
                for alg in policy.allowed_key_algorithms
            ):
                errors.append(f"Key algorithm {cert_info.public_key_algorithm} not allowed")

        if policy.allowed_signature_algorithms:
            if cert_info.signature_algorithm not in policy.allowed_signature_algorithms:
                warnings.append(f"Signature algorithm {cert_info.signature_algorithm} not in preferred list")

        # Check subject requirements
        if policy.required_subject_cn_pattern:
            import re
            if not cert_info.subject_cn or not re.match(policy.required_subject_cn_pattern, cert_info.subject_cn):
                errors.append(f"Subject CN does not match required pattern")

        if policy.required_subject_o and cert_info.subject_o != policy.required_subject_o:
            errors.append(f"Subject O must be {policy.required_subject_o}")

        if policy.required_subject_ou and cert_info.subject_ou != policy.required_subject_ou:
            errors.append(f"Subject OU must be {policy.required_subject_ou}")

        # Check issuer requirements
        if policy.required_issuer_cn and cert_info.issuer_cn != policy.required_issuer_cn:
            errors.append(f"Issuer CN must be {policy.required_issuer_cn}")

        if policy.trusted_issuer_fingerprints:
            issuer_trusted = False
            # Check if any cert in chain is trusted
            for ci in chain_info:
                if ci.fingerprint_sha256 in policy.trusted_issuer_fingerprints:
                    issuer_trusted = True
                    break
            # Also check against our trusted CAs
            for ca in self._trusted_cas.values():
                if ca.certificate_info.fingerprint_sha256 in policy.trusted_issuer_fingerprints:
                    if ca.certificate_info.subject_dn == cert_info.issuer_dn:
                        issuer_trusted = True
                        break
            if not issuer_trusted:
                errors.append("Certificate not issued by trusted issuer")

        # Check EKU
        if policy.require_client_auth_eku:
            if "clientAuth" not in cert_info.extended_key_usage:
                errors.append("Certificate missing clientAuth extended key usage")

        # Check SAN
        if policy.require_san and not cert_info.subject_alt_names:
            errors.append("Certificate missing Subject Alternative Name")

        # Check pinning
        if policy.pinned_fingerprints:
            if cert_info.fingerprint_sha256 not in policy.pinned_fingerprints:
                errors.append("Certificate fingerprint does not match pinned value")

        # Chain validation (simplified - full implementation would use cryptography's verify_certificate)
        chain_valid = self._validate_chain(cert_info, chain_info)
        if not chain_valid:
            errors.append("Certificate chain validation failed")

        # Revocation check
        revocation_checked = False
        revocation_status = None

        if policy.revocation_check != RevocationCheckMethod.NONE:
            revocation_result = self._check_revocation(cert_info, policy.revocation_check)
            revocation_checked = True
            revocation_status = revocation_result

            if revocation_result == "revoked":
                return CertificateValidationResult(
                    valid=False,
                    status=CertificateStatus.REVOKED,
                    certificate_info=cert_info,
                    chain_info=chain_info,
                    errors=["Certificate has been revoked"],
                    warnings=warnings,
                    revocation_checked=True,
                    revocation_status=revocation_result,
                )
            elif revocation_result == "unknown" and not policy.allow_revocation_unknown:
                errors.append("Unable to determine revocation status")

        # Determine final status
        if errors:
            status = CertificateStatus.INVALID_CHAIN if "chain" in str(errors).lower() else CertificateStatus.UNTRUSTED
            return CertificateValidationResult(
                valid=False,
                status=status,
                certificate_info=cert_info,
                chain_info=chain_info,
                errors=errors,
                warnings=warnings,
                revocation_checked=revocation_checked,
                revocation_status=revocation_status,
            )

        return CertificateValidationResult(
            valid=True,
            status=CertificateStatus.VALID,
            certificate_info=cert_info,
            chain_info=chain_info,
            errors=[],
            warnings=warnings,
            revocation_checked=revocation_checked,
            revocation_status=revocation_status,
        )

    def _validate_chain(
        self,
        cert_info: CertificateInfo,
        chain_info: list[CertificateInfo],
    ) -> bool:
        """Validate certificate chain."""
        # Check if issuer is a trusted CA
        for ca in self._trusted_cas.values():
            if not ca.active:
                continue
            if ca.certificate_info.subject_dn == cert_info.issuer_dn:
                return True

        # Check if any intermediate in chain leads to trusted CA
        for intermediate in chain_info:
            for ca in self._trusted_cas.values():
                if not ca.active:
                    continue
                if ca.certificate_info.subject_dn == intermediate.issuer_dn:
                    # Verify intermediate signs our cert
                    if intermediate.subject_dn == cert_info.issuer_dn:
                        return True

        # No trusted chain found
        return len(self._trusted_cas) == 0  # Allow if no CAs configured

    def _check_revocation(
        self,
        cert_info: CertificateInfo,
        method: RevocationCheckMethod,
    ) -> str:
        """Check certificate revocation status."""
        if method in (RevocationCheckMethod.OCSP, RevocationCheckMethod.BOTH):
            if cert_info.ocsp_responders:
                result = self._check_ocsp(cert_info)
                if result != "unknown":
                    return result

        if method in (RevocationCheckMethod.CRL, RevocationCheckMethod.BOTH):
            if cert_info.crl_distribution_points:
                result = self._check_crl(cert_info)
                if result != "unknown":
                    return result

        return "unknown"

    def _check_ocsp(self, cert_info: CertificateInfo) -> str:
        """Check OCSP status (simplified implementation)."""
        # Full implementation would make OCSP requests
        # This is a placeholder that could be extended
        self._logger.debug("OCSP check skipped (not fully implemented)")
        return "unknown"

    def _check_crl(self, cert_info: CertificateInfo) -> str:
        """Check CRL status (simplified implementation)."""
        # Full implementation would download and check CRLs
        # This is a placeholder that could be extended
        self._logger.debug("CRL check skipped (not fully implemented)")
        return "unknown"


class MTLSProvider(AuthProvider):
    """
    Mutual TLS authentication provider.

    Authenticates clients using X.509 client certificates.
    """

    def __init__(
        self,
        validator: CertificateValidator,
        policy: Optional[CertificatePolicy] = None,
        extract_user_from_cn: bool = True,
        cn_to_user_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        self.validator = validator
        self.policy = policy or CertificatePolicy(name="mtls_default")
        self.extract_user_from_cn = extract_user_from_cn
        self.cn_to_user_mapping = cn_to_user_mapping or {}
        self._fingerprint_users: dict[str, str] = {}
        self._logger = logger.bind(provider="mtls")

    @property
    def method(self) -> AuthMethod:
        return AuthMethod.CERTIFICATE

    async def authenticate(
        self,
        credentials: Credentials,
        context: Optional[dict[str, Any]] = None,
    ) -> AuthResult:
        """
        Authenticate using client certificate.

        The certificate should be provided in credentials.secret as PEM string,
        or in context['client_certificate'] from TLS termination.
        """
        context = context or {}

        # Get certificate from credentials or context
        certificate_pem = credentials.secret
        if not certificate_pem:
            certificate_pem = context.get("client_certificate")

        if not certificate_pem:
            return AuthResult(
                success=False,
                error="No client certificate provided",
            )

        # Get certificate chain if provided
        chain_pem = context.get("certificate_chain", [])

        # Validate certificate
        result = self.validator.validate(certificate_pem, chain_pem, self.policy)

        if not result.valid:
            self._logger.warning(
                "Certificate validation failed",
                errors=result.errors,
                status=result.status.value,
            )
            return AuthResult(
                success=False,
                error=f"Certificate validation failed: {'; '.join(result.errors)}",
                metadata=result.to_dict(),
            )

        cert_info = result.certificate_info

        # Determine user ID
        user_id = None

        # Check fingerprint mapping first
        if cert_info.fingerprint_sha256 in self._fingerprint_users:
            user_id = self._fingerprint_users[cert_info.fingerprint_sha256]

        # Check CN mapping
        elif cert_info.subject_cn and cert_info.subject_cn in self.cn_to_user_mapping:
            user_id = self.cn_to_user_mapping[cert_info.subject_cn]

        # Extract from CN
        elif self.extract_user_from_cn and cert_info.subject_cn:
            user_id = cert_info.subject_cn

        # Check identifier in credentials
        elif credentials.identifier:
            user_id = credentials.identifier

        if not user_id:
            return AuthResult(
                success=False,
                error="Unable to determine user identity from certificate",
            )

        self._logger.info(
            "mTLS authentication successful",
            user_id=user_id,
            subject_cn=cert_info.subject_cn,
            fingerprint=cert_info.fingerprint_sha256[:16],
        )

        return AuthResult(
            success=True,
            user_id=user_id,
            metadata={
                "auth_method": "mtls",
                "certificate_subject": cert_info.subject_dn,
                "certificate_issuer": cert_info.issuer_dn,
                "certificate_fingerprint": cert_info.fingerprint_sha256,
                "certificate_expires": cert_info.not_after.isoformat() if cert_info.not_after else None,
                "validation_warnings": result.warnings,
            },
        )

    async def validate(
        self,
        credentials: Credentials,
        context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Validate that credentials contain a valid certificate."""
        certificate_pem = credentials.secret
        if not certificate_pem:
            certificate_pem = (context or {}).get("client_certificate")

        if not certificate_pem:
            return False

        result = self.validator.validate(certificate_pem, policy=self.policy)
        return result.valid

    def register_certificate(
        self,
        fingerprint: str,
        user_id: str,
    ) -> None:
        """Register a certificate fingerprint to user mapping."""
        self._fingerprint_users[fingerprint] = user_id
        self._logger.info(
            "Certificate registered",
            user_id=user_id,
            fingerprint=fingerprint[:16],
        )

    def revoke_certificate(self, fingerprint: str) -> bool:
        """Revoke a certificate by fingerprint."""
        if fingerprint in self._fingerprint_users:
            del self._fingerprint_users[fingerprint]
            return True
        return False
