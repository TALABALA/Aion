"""
AION Plugin System Types

Core dataclasses and enums for plugin definitions.
Provides a comprehensive type system for plugin identity, versioning,
permissions, manifests, and runtime information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from pathlib import Path
import hashlib
import re


# === Plugin Identity ===


class PluginType(str, Enum):
    """Types of plugins supported by AION."""

    TOOL = "tool"                          # Adds new tools
    AGENT = "agent"                        # Adds agent types
    STORAGE = "storage"                    # Storage backends
    AUTH = "auth"                          # Auth providers
    WORKFLOW_TRIGGER = "workflow_trigger"  # Workflow triggers
    WORKFLOW_ACTION = "workflow_action"    # Workflow actions
    EXTRACTOR = "extractor"                # Knowledge extractors
    EXPORTER = "exporter"                  # Metrics exporters
    UI_COMPONENT = "ui_component"          # UI components
    LLM_PROVIDER = "llm_provider"          # LLM providers
    EMBEDDING = "embedding"                # Embedding providers
    MIDDLEWARE = "middleware"              # Request middleware
    HOOK = "hook"                          # System hooks
    MEMORY = "memory"                      # Memory backends
    PLANNER = "planner"                    # Custom planners
    REASONING = "reasoning"                # Reasoning strategies
    THEME = "theme"                        # UI themes
    LANGUAGE = "language"                  # Language/i18n
    INTEGRATION = "integration"            # External integrations


class PluginState(str, Enum):
    """Plugin lifecycle states."""

    DISCOVERED = "discovered"    # Found but not loaded
    VALIDATING = "validating"    # Being validated
    VALIDATED = "validated"      # Passed validation
    LOADING = "loading"          # Being loaded
    LOADED = "loaded"            # Loaded but not initialized
    INITIALIZING = "initializing"  # Being initialized
    INITIALIZED = "initialized"  # Initialized but not active
    ACTIVATING = "activating"    # Being activated
    ACTIVE = "active"            # Running and operational
    SUSPENDING = "suspending"    # Being suspended
    SUSPENDED = "suspended"      # Temporarily disabled
    STOPPING = "stopping"        # Being stopped
    STOPPED = "stopped"          # Gracefully stopped
    ERROR = "error"              # In error state
    UNLOADING = "unloading"      # Being unloaded
    UNLOADED = "unloaded"        # Removed from memory
    DISABLED = "disabled"        # Administratively disabled
    QUARANTINED = "quarantined"  # Isolated due to issues


class PluginPriority(int, Enum):
    """Plugin execution priority levels."""

    CRITICAL = 0      # System-critical plugins
    HIGH = 25         # High priority
    ABOVE_NORMAL = 40  # Above normal
    NORMAL = 50       # Default priority
    BELOW_NORMAL = 60  # Below normal
    LOW = 75          # Low priority
    BACKGROUND = 100  # Background processing


class PermissionLevel(str, Enum):
    """Permission levels for plugins."""

    MINIMAL = "minimal"      # Read-only, no network, no file system
    RESTRICTED = "restricted"  # Limited operations, sandboxed
    STANDARD = "standard"    # Basic operations with controlled access
    ELEVATED = "elevated"    # File system, network access
    PRIVILEGED = "privileged"  # System-level access
    FULL = "full"            # All permissions (trusted only)


# === Version Handling ===


@dataclass
class SemanticVersion:
    """
    Semantic versioning (SemVer 2.0.0) representation.

    Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    _SEMVER_PATTERN = re.compile(
        r'^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)'
        r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
        r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*)?)?'
        r'(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    )

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a version string into SemanticVersion."""
        if not version_str:
            return cls(0, 0, 0)

        version_str = version_str.strip().lstrip('v')

        match = cls._SEMVER_PATTERN.match(version_str)
        if match:
            groups = match.groupdict()
            return cls(
                major=int(groups['major']),
                minor=int(groups['minor']),
                patch=int(groups['patch']),
                prerelease=groups.get('prerelease'),
                build=groups.get('build'),
            )

        # Fallback for non-standard versions
        parts = version_str.replace('-', '.').replace('+', '.').split('.')
        return cls(
            major=int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0,
            minor=int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
            patch=int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0,
        )

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __repr__(self) -> str:
        return f"SemanticVersion({self})"

    def _compare_tuple(self) -> tuple:
        """Create comparison tuple (prerelease versions are lower)."""
        # Prerelease has lower precedence than release
        pre_tuple = (0, self.prerelease) if self.prerelease else (1, "")
        return (self.major, self.minor, self.patch, pre_tuple)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._compare_tuple() == other._compare_tuple()

    def __lt__(self, other: "SemanticVersion") -> bool:
        return self._compare_tuple() < other._compare_tuple()

    def __le__(self, other: "SemanticVersion") -> bool:
        return self._compare_tuple() <= other._compare_tuple()

    def __gt__(self, other: "SemanticVersion") -> bool:
        return self._compare_tuple() > other._compare_tuple()

    def __ge__(self, other: "SemanticVersion") -> bool:
        return self._compare_tuple() >= other._compare_tuple()

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def is_compatible(self, other: "SemanticVersion") -> bool:
        """Check if versions are compatible (same major version, non-zero)."""
        if self.major == 0 or other.major == 0:
            # 0.x versions are not guaranteed compatible
            return self.major == other.major and self.minor == other.minor
        return self.major == other.major

    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return self.prerelease is not None

    def bump_major(self) -> "SemanticVersion":
        """Return new version with major bumped."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        """Return new version with minor bumped."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        """Return new version with patch bumped."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


@dataclass
class VersionConstraint:
    """
    Version constraint for dependencies.

    Supports operators: ==, !=, >, >=, <, <=, ~>, ^
    """

    min_version: Optional[SemanticVersion] = None
    max_version: Optional[SemanticVersion] = None
    min_inclusive: bool = True
    max_inclusive: bool = False
    exact_version: Optional[SemanticVersion] = None
    excluded_versions: List[SemanticVersion] = field(default_factory=list)
    allow_prerelease: bool = False

    def satisfies(self, version: SemanticVersion) -> bool:
        """Check if version satisfies this constraint."""
        # Check prerelease
        if version.is_prerelease() and not self.allow_prerelease:
            # Allow if exact match or within range that includes prereleases
            if self.exact_version and version == self.exact_version:
                pass  # Allow exact match
            elif not self.min_version or not version.is_compatible(self.min_version):
                return False

        # Check exact version
        if self.exact_version:
            return version == self.exact_version

        # Check excluded versions
        if version in self.excluded_versions:
            return False

        # Check minimum
        if self.min_version:
            if self.min_inclusive:
                if version < self.min_version:
                    return False
            else:
                if version <= self.min_version:
                    return False

        # Check maximum
        if self.max_version:
            if self.max_inclusive:
                if version > self.max_version:
                    return False
            else:
                if version >= self.max_version:
                    return False

        return True

    @classmethod
    def parse(cls, constraint_str: str) -> "VersionConstraint":
        """
        Parse constraint string.

        Formats:
        - "1.0.0" or "==1.0.0" - exact version
        - ">=1.0.0" - minimum version (inclusive)
        - ">1.0.0" - minimum version (exclusive)
        - "<=2.0.0" - maximum version (inclusive)
        - "<2.0.0" - maximum version (exclusive)
        - ">=1.0.0,<2.0.0" - range
        - "~>1.0" - pessimistic (>=1.0.0, <1.1.0)
        - "^1.0.0" - compatible (>=1.0.0, <2.0.0)
        - "!=1.5.0" - exclude specific version
        - "*" - any version
        """
        constraint_str = constraint_str.strip()

        if constraint_str in ("*", "any", ""):
            return cls()

        constraint = cls()

        # Handle caret (^) - compatible with
        if constraint_str.startswith("^"):
            version = SemanticVersion.parse(constraint_str[1:])
            constraint.min_version = version
            constraint.min_inclusive = True
            if version.major == 0:
                if version.minor == 0:
                    constraint.max_version = SemanticVersion(0, 0, version.patch + 1)
                else:
                    constraint.max_version = SemanticVersion(0, version.minor + 1, 0)
            else:
                constraint.max_version = SemanticVersion(version.major + 1, 0, 0)
            constraint.max_inclusive = False
            return constraint

        # Handle tilde (~>) - pessimistic
        if constraint_str.startswith("~>") or constraint_str.startswith("~"):
            prefix = "~>" if constraint_str.startswith("~>") else "~"
            version = SemanticVersion.parse(constraint_str[len(prefix):])
            constraint.min_version = version
            constraint.min_inclusive = True
            constraint.max_version = SemanticVersion(version.major, version.minor + 1, 0)
            constraint.max_inclusive = False
            return constraint

        # Handle comma-separated constraints
        for part in constraint_str.split(","):
            part = part.strip()

            if part.startswith(">="):
                constraint.min_version = SemanticVersion.parse(part[2:])
                constraint.min_inclusive = True
            elif part.startswith(">"):
                constraint.min_version = SemanticVersion.parse(part[1:])
                constraint.min_inclusive = False
            elif part.startswith("<="):
                constraint.max_version = SemanticVersion.parse(part[2:])
                constraint.max_inclusive = True
            elif part.startswith("<"):
                constraint.max_version = SemanticVersion.parse(part[1:])
                constraint.max_inclusive = False
            elif part.startswith("=="):
                constraint.exact_version = SemanticVersion.parse(part[2:])
            elif part.startswith("!="):
                constraint.excluded_versions.append(SemanticVersion.parse(part[2:]))
            else:
                # Bare version = exact match
                constraint.exact_version = SemanticVersion.parse(part)

        return constraint

    def __str__(self) -> str:
        if self.exact_version:
            return f"=={self.exact_version}"

        parts = []
        if self.min_version:
            op = ">=" if self.min_inclusive else ">"
            parts.append(f"{op}{self.min_version}")
        if self.max_version:
            op = "<=" if self.max_inclusive else "<"
            parts.append(f"{op}{self.max_version}")
        for excluded in self.excluded_versions:
            parts.append(f"!={excluded}")

        return ",".join(parts) if parts else "*"


# === Plugin Dependencies ===


@dataclass
class PluginDependency:
    """A plugin dependency specification."""

    plugin_id: str
    version_constraint: VersionConstraint = field(default_factory=VersionConstraint)
    optional: bool = False
    features: List[str] = field(default_factory=list)  # Required features
    condition: Optional[str] = None  # Conditional dependency (e.g., "platform == 'linux'")

    def to_dict(self) -> dict:
        return {
            "plugin_id": self.plugin_id,
            "version": str(self.version_constraint),
            "optional": self.optional,
            "features": self.features,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: Union[dict, str]) -> "PluginDependency":
        """Create from dict or string specification."""
        if isinstance(data, str):
            # Parse string like "plugin-id>=1.0.0"
            for op in [">=", "<=", "!=", "==", ">", "<", "^", "~>"]:
                if op in data:
                    idx = data.index(op)
                    return cls(
                        plugin_id=data[:idx].strip(),
                        version_constraint=VersionConstraint.parse(data[idx:]),
                    )
            return cls(plugin_id=data.strip())

        return cls(
            plugin_id=data["plugin_id"],
            version_constraint=VersionConstraint.parse(data.get("version", "*")),
            optional=data.get("optional", False),
            features=data.get("features", []),
            condition=data.get("condition"),
        )


# === Permissions ===


@dataclass
class ResourceLimit:
    """Resource limits for sandboxed plugins."""

    max_memory_mb: int = 256
    max_cpu_percent: float = 25.0
    max_execution_time_seconds: float = 30.0
    max_file_size_mb: int = 10
    max_open_files: int = 10
    max_network_connections: int = 5
    max_subprocess_count: int = 0
    max_threads: int = 4


@dataclass
class PluginPermissions:
    """Permissions requested by a plugin."""

    level: PermissionLevel = PermissionLevel.MINIMAL

    # Network permissions
    network_access: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)

    # File system permissions
    file_system_access: bool = False
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    read_only_paths: List[str] = field(default_factory=list)

    # Database permissions
    database_access: bool = False
    allowed_tables: List[str] = field(default_factory=list)
    read_only_tables: List[str] = field(default_factory=list)

    # Memory/state access
    memory_access: bool = False
    allowed_memory_namespaces: List[str] = field(default_factory=list)

    # Process permissions
    subprocess_access: bool = False
    allowed_commands: List[str] = field(default_factory=list)

    # System permissions
    environment_access: bool = False
    allowed_env_vars: List[str] = field(default_factory=list)

    # AION API access
    api_access: List[str] = field(default_factory=list)  # Allowed API endpoints

    # Resource limits
    resource_limits: ResourceLimit = field(default_factory=ResourceLimit)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "network_access": self.network_access,
            "allowed_domains": self.allowed_domains,
            "file_system_access": self.file_system_access,
            "allowed_paths": self.allowed_paths,
            "database_access": self.database_access,
            "allowed_tables": self.allowed_tables,
            "memory_access": self.memory_access,
            "subprocess_access": self.subprocess_access,
            "allowed_commands": self.allowed_commands,
            "api_access": self.api_access,
            "resource_limits": {
                "max_memory_mb": self.resource_limits.max_memory_mb,
                "max_cpu_percent": self.resource_limits.max_cpu_percent,
                "max_execution_time_seconds": self.resource_limits.max_execution_time_seconds,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PluginPermissions":
        """Create from dictionary."""
        resource_data = data.get("resource_limits", {})
        return cls(
            level=PermissionLevel(data.get("level", "minimal")),
            network_access=data.get("network_access", False),
            allowed_domains=data.get("allowed_domains", []),
            blocked_domains=data.get("blocked_domains", []),
            file_system_access=data.get("file_system_access", False),
            allowed_paths=data.get("allowed_paths", []),
            blocked_paths=data.get("blocked_paths", []),
            read_only_paths=data.get("read_only_paths", []),
            database_access=data.get("database_access", False),
            allowed_tables=data.get("allowed_tables", []),
            memory_access=data.get("memory_access", False),
            allowed_memory_namespaces=data.get("allowed_memory_namespaces", []),
            subprocess_access=data.get("subprocess_access", False),
            allowed_commands=data.get("allowed_commands", []),
            environment_access=data.get("environment_access", False),
            allowed_env_vars=data.get("allowed_env_vars", []),
            api_access=data.get("api_access", []),
            resource_limits=ResourceLimit(
                max_memory_mb=resource_data.get("max_memory_mb", 256),
                max_cpu_percent=resource_data.get("max_cpu_percent", 25.0),
                max_execution_time_seconds=resource_data.get("max_execution_time_seconds", 30.0),
                max_file_size_mb=resource_data.get("max_file_size_mb", 10),
            ),
        )

    def is_allowed_domain(self, domain: str) -> bool:
        """Check if domain is allowed."""
        if not self.network_access:
            return False

        if domain in self.blocked_domains:
            return False

        if not self.allowed_domains:
            return True

        # Check wildcards
        for allowed in self.allowed_domains:
            if allowed.startswith("*."):
                if domain.endswith(allowed[1:]):
                    return True
            elif domain == allowed:
                return True

        return False

    def is_allowed_path(self, path: str) -> bool:
        """Check if file path is allowed."""
        if not self.file_system_access:
            return False

        path_obj = Path(path).resolve()

        # Check blocked paths
        for blocked in self.blocked_paths:
            if str(path_obj).startswith(str(Path(blocked).resolve())):
                return False

        # Check allowed paths
        if not self.allowed_paths:
            return True

        for allowed in self.allowed_paths:
            if str(path_obj).startswith(str(Path(allowed).resolve())):
                return True

        return False


# === Plugin Manifest ===


@dataclass
class PluginAuthor:
    """Plugin author information."""

    name: str
    email: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email, "url": self.url}

    @classmethod
    def from_dict(cls, data: Union[dict, str]) -> "PluginAuthor":
        if isinstance(data, str):
            return cls(name=data)
        return cls(
            name=data.get("name", ""),
            email=data.get("email", ""),
            url=data.get("url", ""),
        )


@dataclass
class PluginManifest:
    """
    Plugin manifest defining metadata and requirements.

    This is the core specification for a plugin, typically stored
    in manifest.json or defined programmatically.
    """

    # === Identity (Required) ===
    id: str
    name: str
    version: SemanticVersion

    # === Description ===
    description: str = ""
    long_description: str = ""
    keywords: List[str] = field(default_factory=list)

    # === Type ===
    plugin_type: PluginType = PluginType.TOOL
    categories: List[str] = field(default_factory=list)

    # === Author & Legal ===
    author: PluginAuthor = field(default_factory=lambda: PluginAuthor(name="Unknown"))
    contributors: List[PluginAuthor] = field(default_factory=list)
    homepage: str = ""
    repository: str = ""
    documentation: str = ""
    license: str = ""
    license_file: str = ""

    # === Entry Points ===
    entry_point: str = ""  # e.g., "my_plugin:MyPlugin"
    entry_points: Dict[str, str] = field(default_factory=dict)  # Multiple entry points

    # === Dependencies ===
    dependencies: List[PluginDependency] = field(default_factory=list)
    optional_dependencies: Dict[str, List[PluginDependency]] = field(default_factory=dict)
    aion_version: VersionConstraint = field(default_factory=lambda: VersionConstraint.parse(">=1.0.0"))
    python_version: str = ">=3.10"
    pip_dependencies: List[str] = field(default_factory=list)
    system_dependencies: List[str] = field(default_factory=list)

    # === Permissions ===
    permissions: PluginPermissions = field(default_factory=PluginPermissions)

    # === Configuration ===
    config_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema
    default_config: Dict[str, Any] = field(default_factory=dict)

    # === Hooks ===
    hooks: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)  # Custom hooks defined by plugin

    # === Features ===
    features: List[str] = field(default_factory=list)  # Features provided
    optional_features: Dict[str, List[str]] = field(default_factory=dict)  # Optional feature dependencies

    # === Tags & Discovery ===
    tags: List[str] = field(default_factory=list)
    icon: str = ""
    screenshots: List[str] = field(default_factory=list)

    # === Platform ===
    platforms: List[str] = field(default_factory=list)  # Supported platforms
    architectures: List[str] = field(default_factory=list)  # Supported architectures

    # === Lifecycle ===
    auto_enable: bool = False  # Auto-enable on install
    singleton: bool = False  # Only one instance allowed
    priority: PluginPriority = PluginPriority.NORMAL

    # === Security ===
    checksum: str = ""  # SHA256 of plugin package
    signature: str = ""  # Digital signature
    verified: bool = False  # Verified by AION team

    def __post_init__(self):
        """Validate manifest after creation."""
        if not self.id:
            raise ValueError("Plugin ID is required")
        if not self.name:
            raise ValueError("Plugin name is required")

    def to_dict(self) -> dict:
        """Convert manifest to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": str(self.version),
            "description": self.description,
            "type": self.plugin_type.value,
            "author": self.author.to_dict() if isinstance(self.author, PluginAuthor) else self.author,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "entry_point": self.entry_point,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "aion_version": str(self.aion_version),
            "python_version": self.python_version,
            "pip_dependencies": self.pip_dependencies,
            "permissions": self.permissions.to_dict(),
            "config_schema": self.config_schema,
            "default_config": self.default_config,
            "hooks": self.hooks,
            "provides_hooks": self.provides_hooks,
            "features": self.features,
            "tags": self.tags,
            "priority": self.priority.value,
            "auto_enable": self.auto_enable,
            "singleton": self.singleton,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PluginManifest":
        """Create manifest from dictionary."""
        # Parse author
        author_data = data.get("author", {})
        if isinstance(author_data, str):
            author = PluginAuthor(name=author_data)
        elif isinstance(author_data, dict):
            author = PluginAuthor.from_dict(author_data)
        else:
            author = PluginAuthor(name="Unknown")

        # Parse dependencies
        dependencies = []
        for dep_data in data.get("dependencies", []):
            dependencies.append(PluginDependency.from_dict(dep_data))

        return cls(
            id=data["id"],
            name=data["name"],
            version=SemanticVersion.parse(data.get("version", "0.0.0")),
            description=data.get("description", ""),
            long_description=data.get("long_description", ""),
            keywords=data.get("keywords", []),
            plugin_type=PluginType(data.get("type", "tool")),
            categories=data.get("categories", []),
            author=author,
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            documentation=data.get("documentation", ""),
            license=data.get("license", ""),
            entry_point=data.get("entry_point", ""),
            entry_points=data.get("entry_points", {}),
            dependencies=dependencies,
            aion_version=VersionConstraint.parse(data.get("aion_version", ">=1.0.0")),
            python_version=data.get("python_version", ">=3.10"),
            pip_dependencies=data.get("pip_dependencies", []),
            system_dependencies=data.get("system_dependencies", []),
            permissions=PluginPermissions.from_dict(data.get("permissions", {})),
            config_schema=data.get("config_schema", {}),
            default_config=data.get("default_config", {}),
            hooks=data.get("hooks", []),
            provides_hooks=data.get("provides_hooks", []),
            features=data.get("features", []),
            tags=data.get("tags", []),
            icon=data.get("icon", ""),
            platforms=data.get("platforms", []),
            auto_enable=data.get("auto_enable", False),
            singleton=data.get("singleton", False),
            priority=PluginPriority(data.get("priority", 50)),
            checksum=data.get("checksum", ""),
            signature=data.get("signature", ""),
            verified=data.get("verified", False),
        )

    def compute_checksum(self, content: bytes) -> str:
        """Compute SHA256 checksum."""
        return hashlib.sha256(content).hexdigest()


# === Plugin Runtime Information ===


@dataclass
class PluginMetrics:
    """Runtime metrics for a plugin."""

    invocation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    max_execution_time_ms: float = 0.0
    last_execution_time_ms: float = 0.0
    memory_usage_bytes: int = 0
    peak_memory_bytes: int = 0

    def record_execution(self, duration_ms: float, success: bool = True) -> None:
        """Record an execution."""
        self.invocation_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        self.total_execution_time_ms += duration_ms
        self.last_execution_time_ms = duration_ms
        self.avg_execution_time_ms = self.total_execution_time_ms / self.invocation_count
        self.min_execution_time_ms = min(self.min_execution_time_ms, duration_ms)
        self.max_execution_time_ms = max(self.max_execution_time_ms, duration_ms)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.invocation_count == 0:
            return 1.0
        return self.success_count / self.invocation_count

    def to_dict(self) -> dict:
        return {
            "invocations": self.invocation_count,
            "successes": self.success_count,
            "errors": self.error_count,
            "success_rate": self.success_rate,
            "total_time_ms": self.total_execution_time_ms,
            "avg_time_ms": self.avg_execution_time_ms,
            "min_time_ms": self.min_execution_time_ms if self.min_execution_time_ms != float('inf') else 0,
            "max_time_ms": self.max_execution_time_ms,
            "last_time_ms": self.last_execution_time_ms,
        }


@dataclass
class PluginInfo:
    """Runtime information about a loaded plugin."""

    manifest: PluginManifest
    state: PluginState = PluginState.DISCOVERED

    # Location
    path: Optional[Path] = None
    source: str = "local"  # local, remote, builtin, marketplace

    # Runtime
    instance: Optional[Any] = None
    module: Optional[Any] = None
    load_time_ms: float = 0.0

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    resolved_config: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    metrics: PluginMetrics = field(default_factory=PluginMetrics)

    # Errors
    last_error: Optional[str] = None
    error_trace: Optional[str] = None
    consecutive_errors: int = 0

    # Timestamps
    discovered_at: datetime = field(default_factory=datetime.now)
    loaded_at: Optional[datetime] = None
    initialized_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    last_invoked_at: Optional[datetime] = None

    # Dependencies
    resolved_dependencies: Dict[str, "PluginInfo"] = field(default_factory=dict)
    dependents: Set[str] = field(default_factory=set)

    # Hot reload
    reload_count: int = 0
    last_reloaded_at: Optional[datetime] = None

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> SemanticVersion:
        return self.manifest.version

    @property
    def plugin_type(self) -> PluginType:
        return self.manifest.plugin_type

    @property
    def is_active(self) -> bool:
        return self.state == PluginState.ACTIVE

    @property
    def is_healthy(self) -> bool:
        return self.state not in (PluginState.ERROR, PluginState.QUARANTINED)

    @property
    def uptime_seconds(self) -> float:
        """Get plugin uptime in seconds."""
        if not self.activated_at:
            return 0.0
        return (datetime.now() - self.activated_at).total_seconds()

    def record_error(self, error: str, trace: Optional[str] = None) -> None:
        """Record an error."""
        self.last_error = error
        self.error_trace = trace
        self.consecutive_errors += 1
        self.metrics.error_count += 1

    def clear_error(self) -> None:
        """Clear error state."""
        self.last_error = None
        self.error_trace = None
        self.consecutive_errors = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": str(self.version),
            "type": self.plugin_type.value,
            "state": self.state.value,
            "path": str(self.path) if self.path else None,
            "source": self.source,
            "config": self.config,
            "metrics": self.metrics.to_dict(),
            "timestamps": {
                "discovered": self.discovered_at.isoformat(),
                "loaded": self.loaded_at.isoformat() if self.loaded_at else None,
                "activated": self.activated_at.isoformat() if self.activated_at else None,
            },
            "error": self.last_error,
            "manifest": self.manifest.to_dict(),
        }


# === Hook Types ===


@dataclass
class HookDefinition:
    """Definition of a hook point in AION."""

    name: str
    description: str = ""
    parameters: Dict[str, type] = field(default_factory=dict)
    return_type: Optional[type] = None
    is_filter: bool = False  # True if hook can modify data
    is_async: bool = True    # True if hook handlers should be async
    allow_multiple: bool = True  # Allow multiple handlers
    priority_order: bool = True  # Execute in priority order
    fail_fast: bool = False  # Stop on first error

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "is_filter": self.is_filter,
            "is_async": self.is_async,
        }


@dataclass
class HookRegistration:
    """A registered hook handler."""

    hook_name: str
    plugin_id: str
    handler: Callable
    priority: int = 100  # Lower = earlier execution
    enabled: bool = True
    run_async: bool = True
    timeout_seconds: float = 30.0

    def __lt__(self, other: "HookRegistration") -> bool:
        return self.priority < other.priority

    def __hash__(self) -> int:
        return hash((self.hook_name, self.plugin_id, id(self.handler)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HookRegistration):
            return NotImplemented
        return (
            self.hook_name == other.hook_name
            and self.plugin_id == other.plugin_id
            and self.handler == other.handler
        )


# === Events ===


class PluginEvent(str, Enum):
    """Plugin lifecycle events."""

    # Discovery
    DISCOVERED = "plugin.discovered"
    SCAN_STARTED = "plugin.scan_started"
    SCAN_COMPLETED = "plugin.scan_completed"

    # Validation
    VALIDATION_STARTED = "plugin.validation_started"
    VALIDATION_PASSED = "plugin.validation_passed"
    VALIDATION_FAILED = "plugin.validation_failed"

    # Loading
    LOADING = "plugin.loading"
    LOADED = "plugin.loaded"
    LOAD_FAILED = "plugin.load_failed"

    # Initialization
    INITIALIZING = "plugin.initializing"
    INITIALIZED = "plugin.initialized"
    INIT_FAILED = "plugin.init_failed"

    # Activation
    ACTIVATING = "plugin.activating"
    ACTIVATED = "plugin.activated"
    ACTIVATION_FAILED = "plugin.activation_failed"

    # Runtime
    SUSPENDED = "plugin.suspended"
    RESUMED = "plugin.resumed"
    ERROR = "plugin.error"
    RECOVERED = "plugin.recovered"

    # Configuration
    CONFIG_CHANGED = "plugin.config_changed"
    CONFIG_VALIDATED = "plugin.config_validated"

    # Hot reload
    RELOADING = "plugin.reloading"
    RELOADED = "plugin.reloaded"
    RELOAD_FAILED = "plugin.reload_failed"

    # Unloading
    UNLOADING = "plugin.unloading"
    UNLOADED = "plugin.unloaded"

    # Security
    PERMISSION_DENIED = "plugin.permission_denied"
    QUARANTINED = "plugin.quarantined"

    # Execution
    EXECUTION_STARTED = "plugin.execution_started"
    EXECUTION_COMPLETED = "plugin.execution_completed"
    EXECUTION_FAILED = "plugin.execution_failed"


@dataclass
class PluginEventData:
    """Data for plugin events."""

    event: PluginEvent
    plugin_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    trace: Optional[str] = None
    source: str = "system"  # Who triggered the event

    def to_dict(self) -> dict:
        return {
            "event": self.event.value,
            "plugin_id": self.plugin_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "error": self.error,
            "source": self.source,
        }


# === Validation ===


@dataclass
class ValidationResult:
    """Result of plugin validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        self.info.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.valid:
            self.valid = False

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }
