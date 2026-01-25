"""
Device Fingerprinting System.

Implements multi-signal device identification for:
- Fraud prevention
- Account takeover detection
- Trusted device recognition
- Session binding

Fingerprinting signals include:
- Browser/client characteristics
- Network attributes
- Hardware indicators
- Behavioral patterns
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class FingerprintComponent(str, Enum):
    """Components used in fingerprinting."""
    USER_AGENT = "user_agent"
    ACCEPT_LANGUAGE = "accept_language"
    ACCEPT_ENCODING = "accept_encoding"
    SCREEN_RESOLUTION = "screen_resolution"
    TIMEZONE = "timezone"
    TIMEZONE_OFFSET = "timezone_offset"
    PLATFORM = "platform"
    CPU_CORES = "cpu_cores"
    DEVICE_MEMORY = "device_memory"
    HARDWARE_CONCURRENCY = "hardware_concurrency"
    COLOR_DEPTH = "color_depth"
    PIXEL_RATIO = "pixel_ratio"
    TOUCH_SUPPORT = "touch_support"
    WEBGL_VENDOR = "webgl_vendor"
    WEBGL_RENDERER = "webgl_renderer"
    CANVAS_HASH = "canvas_hash"
    AUDIO_HASH = "audio_hash"
    FONTS = "fonts"
    PLUGINS = "plugins"
    DO_NOT_TRACK = "do_not_track"
    COOKIES_ENABLED = "cookies_enabled"
    LOCAL_STORAGE = "local_storage"
    SESSION_STORAGE = "session_storage"
    INDEXED_DB = "indexed_db"
    AD_BLOCKER = "ad_blocker"
    IP_ADDRESS = "ip_address"
    IP_TYPE = "ip_type"
    ASN = "asn"
    ISP = "isp"
    COUNTRY = "country"
    REGION = "region"
    CITY = "city"
    TLS_VERSION = "tls_version"
    TLS_CIPHER = "tls_cipher"
    HTTP_VERSION = "http_version"
    JA3_HASH = "ja3_hash"  # TLS fingerprint
    JA4_HASH = "ja4_hash"  # Enhanced TLS fingerprint


class DeviceTrustLevel(str, Enum):
    """Device trust levels."""
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRUSTED = "trusted"


class DeviceType(str, Enum):
    """Device types."""
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    TABLET = "tablet"
    MOBILE = "mobile"
    TV = "tv"
    CONSOLE = "console"
    WEARABLE = "wearable"
    BOT = "bot"
    UNKNOWN = "unknown"


class BrowserFamily(str, Enum):
    """Browser families."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    OPERA = "opera"
    IE = "ie"
    SAMSUNG = "samsung"
    UC = "uc"
    OTHER = "other"
    UNKNOWN = "unknown"


class OSFamily(str, Enum):
    """Operating system families."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    ANDROID = "android"
    IOS = "ios"
    CHROME_OS = "chrome_os"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class ParsedUserAgent:
    """Parsed user agent information."""
    raw: str
    browser_family: BrowserFamily = BrowserFamily.UNKNOWN
    browser_version: Optional[str] = None
    os_family: OSFamily = OSFamily.UNKNOWN
    os_version: Optional[str] = None
    device_type: DeviceType = DeviceType.UNKNOWN
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    is_bot: bool = False
    bot_name: Optional[str] = None


@dataclass
class GeoLocation:
    """Geographic location information."""
    ip_address: str
    country_code: Optional[str] = None
    country_name: Optional[str] = None
    region_code: Optional[str] = None
    region_name: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    asn: Optional[int] = None
    asn_org: Optional[str] = None
    isp: Optional[str] = None
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False
    is_datacenter: bool = False
    is_mobile: bool = False


@dataclass
class NetworkAttributes:
    """Network-related attributes."""
    ip_address: str
    ip_version: int = 4
    is_private: bool = False
    is_loopback: bool = False
    tls_version: Optional[str] = None
    tls_cipher: Optional[str] = None
    http_version: Optional[str] = None
    ja3_hash: Optional[str] = None
    ja4_hash: Optional[str] = None
    geo: Optional[GeoLocation] = None


@dataclass
class DeviceFingerprint:
    """
    Complete device fingerprint.

    Combines multiple signals to create a unique device identifier
    that persists across sessions without cookies.
    """
    fingerprint_id: str
    components: dict[str, Any]
    user_agent: ParsedUserAgent
    network: NetworkAttributes
    trust_level: DeviceTrustLevel = DeviceTrustLevel.UNKNOWN
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    seen_count: int = 1
    user_ids: set[str] = field(default_factory=set)
    risk_indicators: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def device_type(self) -> DeviceType:
        return self.user_agent.device_type

    @property
    def browser(self) -> BrowserFamily:
        return self.user_agent.browser_family

    @property
    def os(self) -> OSFamily:
        return self.user_agent.os_family

    @property
    def is_suspicious(self) -> bool:
        """Check if device has suspicious indicators."""
        return len(self.risk_indicators) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "components": self.components,
            "user_agent": {
                "raw": self.user_agent.raw,
                "browser_family": self.user_agent.browser_family.value,
                "browser_version": self.user_agent.browser_version,
                "os_family": self.user_agent.os_family.value,
                "os_version": self.user_agent.os_version,
                "device_type": self.user_agent.device_type.value,
                "is_bot": self.user_agent.is_bot,
            },
            "network": {
                "ip_address": self.network.ip_address,
                "ip_version": self.network.ip_version,
                "tls_version": self.network.tls_version,
                "ja3_hash": self.network.ja3_hash,
            },
            "trust_level": self.trust_level.value,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "seen_count": self.seen_count,
            "user_ids": list(self.user_ids),
            "risk_indicators": self.risk_indicators,
        }


class UserAgentParser:
    """
    User agent string parser.

    Extracts browser, OS, and device information from user agent strings.
    """

    # Browser patterns
    BROWSER_PATTERNS = [
        (r"(?:Chrome|CriOS)/(\d+)", BrowserFamily.CHROME),
        (r"(?:Firefox|FxiOS)/(\d+)", BrowserFamily.FIREFOX),
        (r"(?:Safari)/(\d+).*(?!Chrome|CriOS)", BrowserFamily.SAFARI),
        (r"(?:Edg|EdgA|EdgiOS)/(\d+)", BrowserFamily.EDGE),
        (r"(?:OPR|Opera)/(\d+)", BrowserFamily.OPERA),
        (r"(?:MSIE |Trident.*rv:)(\d+)", BrowserFamily.IE),
        (r"SamsungBrowser/(\d+)", BrowserFamily.SAMSUNG),
        (r"UCBrowser/(\d+)", BrowserFamily.UC),
    ]

    # OS patterns
    OS_PATTERNS = [
        (r"Windows NT (\d+\.\d+)", OSFamily.WINDOWS),
        (r"Mac OS X (\d+[._]\d+)", OSFamily.MACOS),
        (r"Android (\d+)", OSFamily.ANDROID),
        (r"(?:iPhone|iPad|iPod).*OS (\d+)", OSFamily.IOS),
        (r"Linux", OSFamily.LINUX),
        (r"CrOS", OSFamily.CHROME_OS),
    ]

    # Bot patterns
    BOT_PATTERNS = [
        r"bot",
        r"crawler",
        r"spider",
        r"scraper",
        r"Googlebot",
        r"Bingbot",
        r"Slurp",
        r"DuckDuckBot",
        r"Baiduspider",
        r"YandexBot",
        r"facebookexternalhit",
        r"Twitterbot",
        r"LinkedInBot",
        r"WhatsApp",
        r"TelegramBot",
        r"curl",
        r"wget",
        r"python-requests",
        r"axios",
        r"node-fetch",
        r"Go-http-client",
        r"Java/",
        r"okhttp",
    ]

    # Mobile patterns
    MOBILE_PATTERNS = [
        r"Mobile",
        r"Android.*Mobile",
        r"iPhone",
        r"iPod",
        r"BlackBerry",
        r"Windows Phone",
    ]

    # Tablet patterns
    TABLET_PATTERNS = [
        r"iPad",
        r"Android(?!.*Mobile)",
        r"Tablet",
    ]

    def parse(self, user_agent: str) -> ParsedUserAgent:
        """Parse a user agent string."""
        if not user_agent:
            return ParsedUserAgent(raw="")

        result = ParsedUserAgent(raw=user_agent)

        # Check for bots first
        for pattern in self.BOT_PATTERNS:
            if re.search(pattern, user_agent, re.IGNORECASE):
                result.is_bot = True
                result.device_type = DeviceType.BOT
                match = re.search(pattern, user_agent, re.IGNORECASE)
                if match:
                    result.bot_name = match.group(0)
                break

        # Parse browser
        for pattern, family in self.BROWSER_PATTERNS:
            match = re.search(pattern, user_agent)
            if match:
                result.browser_family = family
                result.browser_version = match.group(1)
                break

        # Handle Safari detection (needs special handling due to Chrome using Safari in UA)
        if "Safari" in user_agent and "Chrome" not in user_agent and "CriOS" not in user_agent:
            result.browser_family = BrowserFamily.SAFARI
            match = re.search(r"Version/(\d+)", user_agent)
            if match:
                result.browser_version = match.group(1)

        # Parse OS
        for pattern, family in self.OS_PATTERNS:
            match = re.search(pattern, user_agent)
            if match:
                result.os_family = family
                if match.lastindex:
                    result.os_version = match.group(1).replace("_", ".")
                break

        # Determine device type
        if not result.is_bot:
            if any(re.search(p, user_agent) for p in self.TABLET_PATTERNS):
                result.device_type = DeviceType.TABLET
            elif any(re.search(p, user_agent) for p in self.MOBILE_PATTERNS):
                result.device_type = DeviceType.MOBILE
            elif result.os_family in (OSFamily.WINDOWS, OSFamily.MACOS, OSFamily.LINUX):
                result.device_type = DeviceType.DESKTOP
            else:
                result.device_type = DeviceType.UNKNOWN

        return result


class DeviceFingerprintManager:
    """
    Device fingerprint manager.

    Handles fingerprint generation, storage, and matching.
    """

    def __init__(
        self,
        hmac_key: bytes,
        similarity_threshold: float = 0.85,
        max_fingerprints_per_user: int = 10,
    ) -> None:
        self.hmac_key = hmac_key
        self.similarity_threshold = similarity_threshold
        self.max_fingerprints_per_user = max_fingerprints_per_user
        self._ua_parser = UserAgentParser()
        self._fingerprints: dict[str, DeviceFingerprint] = {}
        self._user_fingerprints: dict[str, list[str]] = {}
        self._logger = logger.bind(component="device_fingerprint")

    def generate_fingerprint(
        self,
        request_data: dict[str, Any],
        client_data: Optional[dict[str, Any]] = None,
    ) -> DeviceFingerprint:
        """
        Generate a device fingerprint from request and client data.

        Args:
            request_data: Server-side request data (headers, IP, etc.)
            client_data: Client-side fingerprint data (from JavaScript)

        Returns:
            DeviceFingerprint object
        """
        components: dict[str, Any] = {}

        # Extract server-side components
        user_agent = request_data.get("user_agent", "")
        components[FingerprintComponent.USER_AGENT.value] = user_agent

        headers = request_data.get("headers", {})
        components[FingerprintComponent.ACCEPT_LANGUAGE.value] = headers.get("accept-language", "")
        components[FingerprintComponent.ACCEPT_ENCODING.value] = headers.get("accept-encoding", "")

        ip_address = request_data.get("ip_address", "0.0.0.0")
        components[FingerprintComponent.IP_ADDRESS.value] = self._anonymize_ip(ip_address)

        # TLS information
        components[FingerprintComponent.TLS_VERSION.value] = request_data.get("tls_version")
        components[FingerprintComponent.TLS_CIPHER.value] = request_data.get("tls_cipher")
        components[FingerprintComponent.JA3_HASH.value] = request_data.get("ja3_hash")
        components[FingerprintComponent.HTTP_VERSION.value] = request_data.get("http_version")

        # Extract client-side components
        if client_data:
            for key in [
                FingerprintComponent.SCREEN_RESOLUTION.value,
                FingerprintComponent.TIMEZONE.value,
                FingerprintComponent.TIMEZONE_OFFSET.value,
                FingerprintComponent.PLATFORM.value,
                FingerprintComponent.CPU_CORES.value,
                FingerprintComponent.DEVICE_MEMORY.value,
                FingerprintComponent.COLOR_DEPTH.value,
                FingerprintComponent.PIXEL_RATIO.value,
                FingerprintComponent.TOUCH_SUPPORT.value,
                FingerprintComponent.WEBGL_VENDOR.value,
                FingerprintComponent.WEBGL_RENDERER.value,
                FingerprintComponent.CANVAS_HASH.value,
                FingerprintComponent.AUDIO_HASH.value,
                FingerprintComponent.FONTS.value,
                FingerprintComponent.PLUGINS.value,
                FingerprintComponent.DO_NOT_TRACK.value,
                FingerprintComponent.COOKIES_ENABLED.value,
                FingerprintComponent.LOCAL_STORAGE.value,
                FingerprintComponent.SESSION_STORAGE.value,
                FingerprintComponent.INDEXED_DB.value,
            ]:
                if key in client_data:
                    components[key] = client_data[key]

        # Generate fingerprint ID
        fingerprint_id = self._generate_fingerprint_id(components)

        # Parse user agent
        parsed_ua = self._ua_parser.parse(user_agent)

        # Build network attributes
        network = self._build_network_attributes(request_data)

        # Detect risk indicators
        risk_indicators = self._detect_risk_indicators(components, parsed_ua, network)

        fingerprint = DeviceFingerprint(
            fingerprint_id=fingerprint_id,
            components=components,
            user_agent=parsed_ua,
            network=network,
            risk_indicators=risk_indicators,
        )

        return fingerprint

    def _generate_fingerprint_id(self, components: dict[str, Any]) -> str:
        """Generate a unique fingerprint ID from components."""
        # Select stable components for ID generation
        stable_keys = [
            FingerprintComponent.USER_AGENT.value,
            FingerprintComponent.PLATFORM.value,
            FingerprintComponent.SCREEN_RESOLUTION.value,
            FingerprintComponent.TIMEZONE_OFFSET.value,
            FingerprintComponent.CPU_CORES.value,
            FingerprintComponent.COLOR_DEPTH.value,
            FingerprintComponent.WEBGL_RENDERER.value,
            FingerprintComponent.CANVAS_HASH.value,
            FingerprintComponent.AUDIO_HASH.value,
            FingerprintComponent.FONTS.value,
        ]

        # Build stable string
        stable_parts = []
        for key in stable_keys:
            value = components.get(key, "")
            if value:
                stable_parts.append(f"{key}:{value}")

        stable_string = "|".join(sorted(stable_parts))

        # Generate HMAC-SHA256 hash
        signature = hmac.new(
            self.hmac_key,
            stable_string.encode(),
            hashlib.sha256,
        ).digest()

        return base64.urlsafe_b64encode(signature[:16]).decode().rstrip("=")

    def _anonymize_ip(self, ip_address: str) -> str:
        """Anonymize IP address for privacy (keeps network portion only)."""
        try:
            ip = ipaddress.ip_address(ip_address)
            if isinstance(ip, ipaddress.IPv4Address):
                # Keep first 3 octets
                parts = ip_address.split(".")
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
            else:
                # Keep first 48 bits for IPv6
                network = ipaddress.ip_network(f"{ip_address}/48", strict=False)
                return str(network.network_address)
        except ValueError:
            return "0.0.0.0"

    def _build_network_attributes(self, request_data: dict[str, Any]) -> NetworkAttributes:
        """Build network attributes from request data."""
        ip_address = request_data.get("ip_address", "0.0.0.0")

        try:
            ip = ipaddress.ip_address(ip_address)
            ip_version = 4 if isinstance(ip, ipaddress.IPv4Address) else 6
            is_private = ip.is_private
            is_loopback = ip.is_loopback
        except ValueError:
            ip_version = 4
            is_private = False
            is_loopback = False

        geo = None
        if "geo" in request_data:
            geo_data = request_data["geo"]
            geo = GeoLocation(
                ip_address=ip_address,
                country_code=geo_data.get("country_code"),
                country_name=geo_data.get("country_name"),
                region_code=geo_data.get("region_code"),
                region_name=geo_data.get("region_name"),
                city=geo_data.get("city"),
                latitude=geo_data.get("latitude"),
                longitude=geo_data.get("longitude"),
                timezone=geo_data.get("timezone"),
                asn=geo_data.get("asn"),
                isp=geo_data.get("isp"),
                is_vpn=geo_data.get("is_vpn", False),
                is_proxy=geo_data.get("is_proxy", False),
                is_tor=geo_data.get("is_tor", False),
                is_datacenter=geo_data.get("is_datacenter", False),
            )

        return NetworkAttributes(
            ip_address=ip_address,
            ip_version=ip_version,
            is_private=is_private,
            is_loopback=is_loopback,
            tls_version=request_data.get("tls_version"),
            tls_cipher=request_data.get("tls_cipher"),
            http_version=request_data.get("http_version"),
            ja3_hash=request_data.get("ja3_hash"),
            ja4_hash=request_data.get("ja4_hash"),
            geo=geo,
        )

    def _detect_risk_indicators(
        self,
        components: dict[str, Any],
        user_agent: ParsedUserAgent,
        network: NetworkAttributes,
    ) -> list[str]:
        """Detect risk indicators in the fingerprint."""
        indicators: list[str] = []

        # Bot detection
        if user_agent.is_bot:
            indicators.append("bot_detected")

        # Headless browser detection
        ua_lower = user_agent.raw.lower()
        if "headless" in ua_lower:
            indicators.append("headless_browser")

        if "phantomjs" in ua_lower or "selenium" in ua_lower:
            indicators.append("automation_tool")

        # WebDriver detection
        if components.get("webdriver"):
            indicators.append("webdriver_detected")

        # Missing expected components
        if not components.get(FingerprintComponent.CANVAS_HASH.value):
            indicators.append("missing_canvas")

        if not components.get(FingerprintComponent.WEBGL_RENDERER.value):
            indicators.append("missing_webgl")

        # Inconsistent data
        if user_agent.device_type == DeviceType.MOBILE:
            if components.get(FingerprintComponent.TOUCH_SUPPORT.value) is False:
                indicators.append("mobile_no_touch")

        # Privacy mode indicators
        if components.get(FingerprintComponent.DO_NOT_TRACK.value) == "1":
            indicators.append("dnt_enabled")

        # Network risk indicators
        if network.geo:
            if network.geo.is_vpn:
                indicators.append("vpn_detected")
            if network.geo.is_proxy:
                indicators.append("proxy_detected")
            if network.geo.is_tor:
                indicators.append("tor_detected")
            if network.geo.is_datacenter:
                indicators.append("datacenter_ip")

        # TLS anomalies
        if network.tls_version and network.tls_version < "TLSv1.2":
            indicators.append("outdated_tls")

        return indicators

    def calculate_similarity(
        self,
        fp1: DeviceFingerprint,
        fp2: DeviceFingerprint,
    ) -> float:
        """
        Calculate similarity score between two fingerprints.

        Returns a value between 0.0 (completely different) and 1.0 (identical).
        """
        # Component weights
        weights = {
            FingerprintComponent.CANVAS_HASH.value: 0.15,
            FingerprintComponent.WEBGL_RENDERER.value: 0.12,
            FingerprintComponent.AUDIO_HASH.value: 0.10,
            FingerprintComponent.FONTS.value: 0.10,
            FingerprintComponent.SCREEN_RESOLUTION.value: 0.08,
            FingerprintComponent.TIMEZONE_OFFSET.value: 0.08,
            FingerprintComponent.PLATFORM.value: 0.08,
            FingerprintComponent.CPU_CORES.value: 0.05,
            FingerprintComponent.COLOR_DEPTH.value: 0.05,
            FingerprintComponent.PIXEL_RATIO.value: 0.05,
            FingerprintComponent.USER_AGENT.value: 0.07,
            FingerprintComponent.ACCEPT_LANGUAGE.value: 0.04,
            FingerprintComponent.PLUGINS.value: 0.03,
        }

        total_weight = 0.0
        weighted_score = 0.0

        for component, weight in weights.items():
            v1 = fp1.components.get(component)
            v2 = fp2.components.get(component)

            if v1 is None and v2 is None:
                continue

            total_weight += weight

            if v1 == v2:
                weighted_score += weight
            elif v1 is not None and v2 is not None:
                # Partial matching for strings
                if isinstance(v1, str) and isinstance(v2, str):
                    # Calculate string similarity
                    similarity = self._string_similarity(v1, v2)
                    weighted_score += weight * similarity

        if total_weight == 0:
            return 0.0

        return weighted_score / total_weight

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings using Jaccard similarity."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Use character n-grams
        n = 3
        set1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
        set2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union

    def store_fingerprint(
        self,
        fingerprint: DeviceFingerprint,
        user_id: Optional[str] = None,
    ) -> None:
        """Store a fingerprint."""
        self._fingerprints[fingerprint.fingerprint_id] = fingerprint

        if user_id:
            fingerprint.user_ids.add(user_id)

            if user_id not in self._user_fingerprints:
                self._user_fingerprints[user_id] = []

            if fingerprint.fingerprint_id not in self._user_fingerprints[user_id]:
                self._user_fingerprints[user_id].append(fingerprint.fingerprint_id)

                # Enforce max fingerprints per user
                while len(self._user_fingerprints[user_id]) > self.max_fingerprints_per_user:
                    old_fp_id = self._user_fingerprints[user_id].pop(0)
                    if old_fp_id in self._fingerprints:
                        self._fingerprints[old_fp_id].user_ids.discard(user_id)

    def get_fingerprint(self, fingerprint_id: str) -> Optional[DeviceFingerprint]:
        """Retrieve a fingerprint by ID."""
        return self._fingerprints.get(fingerprint_id)

    def find_matching_fingerprint(
        self,
        fingerprint: DeviceFingerprint,
        user_id: Optional[str] = None,
    ) -> Optional[tuple[DeviceFingerprint, float]]:
        """
        Find a matching fingerprint in the store.

        Returns the best match and its similarity score, or None if no match found.
        """
        best_match: Optional[DeviceFingerprint] = None
        best_score = 0.0

        # Narrow search to user's fingerprints if user_id provided
        if user_id and user_id in self._user_fingerprints:
            candidates = [
                self._fingerprints[fp_id]
                for fp_id in self._user_fingerprints[user_id]
                if fp_id in self._fingerprints
            ]
        else:
            candidates = list(self._fingerprints.values())

        for candidate in candidates:
            if candidate.fingerprint_id == fingerprint.fingerprint_id:
                return (candidate, 1.0)

            score = self.calculate_similarity(fingerprint, candidate)
            if score >= self.similarity_threshold and score > best_score:
                best_match = candidate
                best_score = score

        if best_match:
            return (best_match, best_score)

        return None

    def is_known_device(
        self,
        fingerprint: DeviceFingerprint,
        user_id: str,
    ) -> bool:
        """Check if this device is known for the user."""
        if user_id not in self._user_fingerprints:
            return False

        # Check exact match first
        if fingerprint.fingerprint_id in self._user_fingerprints[user_id]:
            return True

        # Check similar fingerprints
        match = self.find_matching_fingerprint(fingerprint, user_id)
        return match is not None

    def update_trust_level(
        self,
        fingerprint_id: str,
        trust_level: DeviceTrustLevel,
    ) -> bool:
        """Update the trust level of a fingerprint."""
        if fingerprint_id not in self._fingerprints:
            return False

        self._fingerprints[fingerprint_id].trust_level = trust_level
        return True

    def record_activity(self, fingerprint_id: str) -> None:
        """Record activity for a fingerprint."""
        if fingerprint_id in self._fingerprints:
            fp = self._fingerprints[fingerprint_id]
            fp.last_seen_at = time.time()
            fp.seen_count += 1

    def get_user_devices(self, user_id: str) -> list[DeviceFingerprint]:
        """Get all known devices for a user."""
        if user_id not in self._user_fingerprints:
            return []

        return [
            self._fingerprints[fp_id]
            for fp_id in self._user_fingerprints[user_id]
            if fp_id in self._fingerprints
        ]

    def revoke_device(self, fingerprint_id: str, user_id: str) -> bool:
        """Revoke a device for a user."""
        if user_id in self._user_fingerprints:
            if fingerprint_id in self._user_fingerprints[user_id]:
                self._user_fingerprints[user_id].remove(fingerprint_id)

                if fingerprint_id in self._fingerprints:
                    self._fingerprints[fingerprint_id].user_ids.discard(user_id)

                return True

        return False


# JavaScript code to collect client-side fingerprint data
CLIENT_FINGERPRINT_JS = """
async function collectFingerprint() {
    const fp = {};

    // Screen
    fp.screen_resolution = `${screen.width}x${screen.height}`;
    fp.color_depth = screen.colorDepth;
    fp.pixel_ratio = window.devicePixelRatio || 1;

    // Timezone
    fp.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    fp.timezone_offset = new Date().getTimezoneOffset();

    // Platform
    fp.platform = navigator.platform;
    fp.cpu_cores = navigator.hardwareConcurrency || 0;
    fp.device_memory = navigator.deviceMemory || 0;

    // Touch support
    fp.touch_support = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

    // WebGL
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                fp.webgl_vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
                fp.webgl_renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            }
        }
    } catch (e) {}

    // Canvas fingerprint
    try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 200;
        canvas.height = 50;
        ctx.textBaseline = 'top';
        ctx.font = '14px Arial';
        ctx.fillStyle = '#f60';
        ctx.fillRect(125, 1, 62, 20);
        ctx.fillStyle = '#069';
        ctx.fillText('Fingerprint', 2, 15);
        ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
        ctx.fillText('Fingerprint', 4, 17);
        fp.canvas_hash = await crypto.subtle.digest('SHA-256',
            new TextEncoder().encode(canvas.toDataURL())
        ).then(h => Array.from(new Uint8Array(h)).map(b => b.toString(16).padStart(2, '0')).join(''));
    } catch (e) {}

    // Audio fingerprint
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const analyser = audioCtx.createAnalyser();
        const gain = audioCtx.createGain();
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);

        oscillator.type = 'triangle';
        oscillator.frequency.setValueAtTime(10000, audioCtx.currentTime);
        gain.gain.setValueAtTime(0, audioCtx.currentTime);

        oscillator.connect(analyser);
        analyser.connect(processor);
        processor.connect(gain);
        gain.connect(audioCtx.destination);

        const data = new Float32Array(analyser.frequencyBinCount);
        analyser.getFloatFrequencyData(data);
        fp.audio_hash = await crypto.subtle.digest('SHA-256',
            new TextEncoder().encode(data.join(','))
        ).then(h => Array.from(new Uint8Array(h)).map(b => b.toString(16).padStart(2, '0')).join(''));

        audioCtx.close();
    } catch (e) {}

    // Fonts (basic check)
    const testFonts = ['Arial', 'Times New Roman', 'Courier New', 'Georgia', 'Verdana'];
    const detectedFonts = [];
    const testString = 'mmmmmmmmmmlli';
    const testSize = '72px';
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    for (const font of testFonts) {
        ctx.font = `${testSize} ${font}`;
        const width = ctx.measureText(testString).width;
        detectedFonts.push(`${font}:${width}`);
    }
    fp.fonts = detectedFonts.join(',');

    // Plugins
    fp.plugins = Array.from(navigator.plugins || []).map(p => p.name).join(',');

    // Privacy settings
    fp.do_not_track = navigator.doNotTrack || window.doNotTrack || navigator.msDoNotTrack;
    fp.cookies_enabled = navigator.cookieEnabled;
    fp.local_storage = !!window.localStorage;
    fp.session_storage = !!window.sessionStorage;
    fp.indexed_db = !!window.indexedDB;

    return fp;
}
"""
