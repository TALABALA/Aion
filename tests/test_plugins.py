"""
AION Plugin System Tests

Comprehensive tests for the plugin system.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from aion.plugins import (
    # Core
    PluginManager,
    PluginRegistry,
    PluginLoader,
    PluginValidator,
    LifecycleManager,

    # Types
    PluginType,
    PluginState,
    PluginPriority,
    PermissionLevel,
    PluginEvent,
    SemanticVersion,
    VersionConstraint,
    PluginDependency,
    PluginPermissions,
    PluginManifest,
    PluginInfo,
    ValidationResult,
    HookDefinition,
    HookRegistration,

    # Interfaces
    BasePlugin,
    ToolPlugin,
    Tool,
    ToolParameter,
    ToolParameterType,

    # Dependencies
    DependencyResolver,
    DependencyGraph,

    # Hooks
    HookSystem,

    # Sandbox
    PermissionChecker,
    PermissionBuilder,
)


# === Test Fixtures ===


@pytest.fixture
def temp_plugin_dir():
    """Create a temporary plugin directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_manifest():
    """Create a sample manifest."""
    return PluginManifest(
        id="test-plugin",
        name="Test Plugin",
        version=SemanticVersion(1, 0, 0),
        description="A test plugin",
        plugin_type=PluginType.TOOL,
        entry_point="plugin:TestPlugin",
        tags=["test", "example"],
    )


@pytest.fixture
def sample_plugin_info(sample_manifest):
    """Create a sample plugin info."""
    return PluginInfo(
        manifest=sample_manifest,
        state=PluginState.DISCOVERED,
        source="local",
    )


# === Semantic Version Tests ===


class TestSemanticVersion:
    """Test semantic version handling."""

    def test_parse_simple_version(self):
        """Test parsing simple version strings."""
        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_prerelease_version(self):
        """Test parsing prerelease versions."""
        v = SemanticVersion.parse("1.0.0-alpha.1")
        assert v.major == 1
        assert v.minor == 0
        assert v.patch == 0
        assert v.prerelease == "alpha.1"

    def test_parse_build_metadata(self):
        """Test parsing build metadata."""
        v = SemanticVersion.parse("1.0.0+build.123")
        assert v.build == "build.123"

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        v3 = SemanticVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 < v3

    def test_prerelease_lower_than_release(self):
        """Test that prerelease versions are lower than release."""
        release = SemanticVersion(1, 0, 0)
        prerelease = SemanticVersion(1, 0, 0, prerelease="alpha")

        assert prerelease < release

    def test_version_compatibility(self):
        """Test version compatibility check."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 5, 0)
        v3 = SemanticVersion(2, 0, 0)

        assert v1.is_compatible(v2)
        assert not v1.is_compatible(v3)

    def test_version_string_representation(self):
        """Test version string formatting."""
        v = SemanticVersion(1, 2, 3, prerelease="beta", build="abc")
        assert str(v) == "1.2.3-beta+abc"

    def test_version_bump(self):
        """Test version bumping."""
        v = SemanticVersion(1, 2, 3)
        assert v.bump_major() == SemanticVersion(2, 0, 0)
        assert v.bump_minor() == SemanticVersion(1, 3, 0)
        assert v.bump_patch() == SemanticVersion(1, 2, 4)


class TestVersionConstraint:
    """Test version constraints."""

    def test_parse_exact_version(self):
        """Test parsing exact version constraint."""
        c = VersionConstraint.parse("==1.0.0")
        assert c.exact_version == SemanticVersion(1, 0, 0)

    def test_parse_range_constraint(self):
        """Test parsing range constraints."""
        c = VersionConstraint.parse(">=1.0.0,<2.0.0")
        assert c.min_version == SemanticVersion(1, 0, 0)
        assert c.min_inclusive
        assert c.max_version == SemanticVersion(2, 0, 0)

    def test_parse_caret_constraint(self):
        """Test parsing caret constraint."""
        c = VersionConstraint.parse("^1.2.3")
        assert c.satisfies(SemanticVersion(1, 2, 3))
        assert c.satisfies(SemanticVersion(1, 9, 0))
        assert not c.satisfies(SemanticVersion(2, 0, 0))

    def test_satisfies_exact(self):
        """Test exact version satisfaction."""
        c = VersionConstraint(exact_version=SemanticVersion(1, 0, 0))
        assert c.satisfies(SemanticVersion(1, 0, 0))
        assert not c.satisfies(SemanticVersion(1, 0, 1))

    def test_satisfies_range(self):
        """Test range satisfaction."""
        c = VersionConstraint(
            min_version=SemanticVersion(1, 0, 0),
            max_version=SemanticVersion(2, 0, 0),
        )
        assert c.satisfies(SemanticVersion(1, 5, 0))
        assert not c.satisfies(SemanticVersion(0, 9, 0))
        assert not c.satisfies(SemanticVersion(2, 0, 0))

    def test_satisfies_exclusion(self):
        """Test version exclusion."""
        c = VersionConstraint(
            min_version=SemanticVersion(1, 0, 0),
            excluded_versions=[SemanticVersion(1, 5, 0)],
        )
        assert c.satisfies(SemanticVersion(1, 4, 0))
        assert not c.satisfies(SemanticVersion(1, 5, 0))


# === Plugin Manifest Tests ===


class TestPluginManifest:
    """Test plugin manifest handling."""

    def test_create_manifest(self, sample_manifest):
        """Test creating a manifest."""
        assert sample_manifest.id == "test-plugin"
        assert sample_manifest.name == "Test Plugin"
        assert sample_manifest.version == SemanticVersion(1, 0, 0)

    def test_manifest_to_dict(self, sample_manifest):
        """Test manifest serialization."""
        data = sample_manifest.to_dict()
        assert data["id"] == "test-plugin"
        assert data["version"] == "1.0.0"
        assert data["type"] == "tool"

    def test_manifest_from_dict(self):
        """Test manifest deserialization."""
        data = {
            "id": "my-plugin",
            "name": "My Plugin",
            "version": "2.0.0",
            "type": "agent",
            "author": "Test Author",
        }
        manifest = PluginManifest.from_dict(data)
        assert manifest.id == "my-plugin"
        assert manifest.version == SemanticVersion(2, 0, 0)
        assert manifest.plugin_type == PluginType.AGENT

    def test_manifest_requires_id(self):
        """Test that manifest requires an ID."""
        with pytest.raises(ValueError):
            PluginManifest(id="", name="Test", version=SemanticVersion(1, 0, 0))


# === Plugin Registry Tests ===


class TestPluginRegistry:
    """Test plugin registry."""

    def test_register_plugin(self, sample_plugin_info):
        """Test registering a plugin."""
        registry = PluginRegistry()
        assert registry.register(sample_plugin_info)
        assert registry.exists("test-plugin")

    def test_unregister_plugin(self, sample_plugin_info):
        """Test unregistering a plugin."""
        registry = PluginRegistry()
        registry.register(sample_plugin_info)
        result = registry.unregister("test-plugin")
        assert result is not None
        assert not registry.exists("test-plugin")

    def test_get_plugin(self, sample_plugin_info):
        """Test getting a plugin."""
        registry = PluginRegistry()
        registry.register(sample_plugin_info)
        plugin = registry.get("test-plugin")
        assert plugin is not None
        assert plugin.id == "test-plugin"

    def test_get_by_type(self, sample_plugin_info):
        """Test getting plugins by type."""
        registry = PluginRegistry()
        registry.register(sample_plugin_info)
        plugins = registry.get_by_type(PluginType.TOOL)
        assert len(plugins) == 1
        assert plugins[0].id == "test-plugin"

    def test_get_by_state(self, sample_plugin_info):
        """Test getting plugins by state."""
        registry = PluginRegistry()
        registry.register(sample_plugin_info)
        plugins = registry.get_by_state(PluginState.DISCOVERED)
        assert len(plugins) == 1

    def test_set_state(self, sample_plugin_info):
        """Test setting plugin state."""
        registry = PluginRegistry()
        registry.register(sample_plugin_info)
        registry.set_state("test-plugin", PluginState.LOADED)
        plugin = registry.get("test-plugin")
        assert plugin.state == PluginState.LOADED

    def test_search_by_query(self, sample_plugin_info):
        """Test searching plugins by query."""
        registry = PluginRegistry()
        registry.register(sample_plugin_info)
        results = registry.search(query="test")
        assert len(results) == 1

    def test_search_by_tags(self):
        """Test searching plugins by tags."""
        registry = PluginRegistry()
        manifest = PluginManifest(
            id="tagged-plugin",
            name="Tagged Plugin",
            version=SemanticVersion(1, 0, 0),
            tags=["special", "featured"],
        )
        registry.register(PluginInfo(manifest=manifest, state=PluginState.DISCOVERED))

        results = registry.search(tags=["special"])
        assert len(results) == 1
        assert results[0].id == "tagged-plugin"


# === Dependency Graph Tests ===


class TestDependencyGraph:
    """Test dependency graph."""

    def test_add_nodes(self):
        """Test adding nodes."""
        graph = DependencyGraph()
        graph.add_node("a")
        graph.add_node("b")
        assert "a" in graph
        assert "b" in graph
        assert len(graph) == 2

    def test_add_edges(self):
        """Test adding edges."""
        graph = DependencyGraph()
        graph.add_edge("a", "b")  # a must be loaded before b
        assert graph.has_edge("a", "b")

    def test_topological_sort(self):
        """Test topological sort."""
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("a", "c")

        order = graph.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_cycle_detection(self):
        """Test cycle detection."""
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "a")

        assert graph.has_cycle()
        cycles = graph.find_cycles()
        assert len(cycles) > 0

    def test_get_dependencies(self):
        """Test getting dependencies."""
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        deps = graph.get_dependents("a")
        assert "b" in deps
        assert "c" in deps

    def test_get_all_dependencies(self):
        """Test getting transitive dependencies."""
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        all_deps = graph.get_all_dependencies("c")
        assert "a" in all_deps
        assert "b" in all_deps


# === Dependency Resolver Tests ===


class TestDependencyResolver:
    """Test dependency resolver."""

    def test_resolve_simple(self):
        """Test simple dependency resolution."""
        resolver = DependencyResolver()

        plugins = [
            PluginInfo(
                manifest=PluginManifest(
                    id="plugin-a",
                    name="Plugin A",
                    version=SemanticVersion(1, 0, 0),
                ),
                state=PluginState.DISCOVERED,
            ),
            PluginInfo(
                manifest=PluginManifest(
                    id="plugin-b",
                    name="Plugin B",
                    version=SemanticVersion(1, 0, 0),
                    dependencies=[
                        PluginDependency(
                            plugin_id="plugin-a",
                            version_constraint=VersionConstraint.parse(">=1.0.0"),
                        ),
                    ],
                ),
                state=PluginState.DISCOVERED,
            ),
        ]

        result = resolver.resolve(plugins)
        assert result.success
        assert result.load_order.index("plugin-a") < result.load_order.index("plugin-b")

    def test_resolve_missing_dependency(self):
        """Test resolution with missing dependency."""
        resolver = DependencyResolver()

        plugins = [
            PluginInfo(
                manifest=PluginManifest(
                    id="plugin-a",
                    name="Plugin A",
                    version=SemanticVersion(1, 0, 0),
                    dependencies=[
                        PluginDependency(plugin_id="missing-plugin"),
                    ],
                ),
                state=PluginState.DISCOVERED,
            ),
        ]

        result = resolver.resolve(plugins)
        assert not result.success
        assert "plugin-a" in result.missing_dependencies


# === Hook System Tests ===


class TestHookSystem:
    """Test hook system."""

    def test_define_hook(self):
        """Test defining a hook."""
        hooks = HookSystem()
        hooks.define(HookDefinition(
            name="test.hook",
            description="A test hook",
        ))
        assert hooks.is_defined("test.hook")

    def test_register_handler(self):
        """Test registering a hook handler."""
        hooks = HookSystem()

        def handler():
            pass

        hooks.register("test.hook", "test-plugin", handler)
        handlers = hooks.get_handlers("test.hook")
        assert len(handlers) == 1
        assert handlers[0].plugin_id == "test-plugin"

    @pytest.mark.asyncio
    async def test_dispatch_action(self):
        """Test dispatching an action hook."""
        hooks = HookSystem()
        results = []

        async def handler(value):
            results.append(value)

        hooks.register("test.hook", "test-plugin", handler)
        await hooks.dispatch("test.hook", "test-value")
        assert "test-value" in results

    @pytest.mark.asyncio
    async def test_filter_hook(self):
        """Test filter hook modifies value."""
        hooks = HookSystem()

        async def handler(value):
            return value + " modified"

        hooks.register("test.filter", "test-plugin", handler)
        result = await hooks.filter("test.filter", "original")
        assert result == "original modified"

    def test_unregister_plugin_hooks(self):
        """Test unregistering all hooks for a plugin."""
        hooks = HookSystem()

        def handler():
            pass

        hooks.register("hook1", "test-plugin", handler)
        hooks.register("hook2", "test-plugin", handler)

        count = hooks.unregister_plugin("test-plugin")
        assert count == 2
        assert len(hooks.get_handlers("hook1")) == 0


# === Permission Tests ===


class TestPermissions:
    """Test permission system."""

    def test_permission_builder(self):
        """Test permission builder."""
        permissions = (
            PermissionBuilder()
            .allow_network(["api.example.com"])
            .allow_files(["/tmp"])
            .build()
        )

        assert permissions.network_access
        assert "api.example.com" in permissions.allowed_domains
        assert permissions.file_system_access

    def test_permission_checker_network(self):
        """Test network permission checking."""
        permissions = PluginPermissions(
            network_access=True,
            allowed_domains=["api.example.com", "*.trusted.com"],
        )
        checker = PermissionChecker(permissions)

        assert checker.check("network", "https://api.example.com/endpoint")
        assert checker.check("network", "https://sub.trusted.com/path")
        assert not checker.check("network", "https://evil.com/hack")

    def test_permission_checker_files(self):
        """Test file permission checking."""
        permissions = PluginPermissions(
            file_system_access=True,
            allowed_paths=["/tmp", "/home/user/data"],
            blocked_paths=["/etc"],
        )
        checker = PermissionChecker(permissions)

        assert checker.check("file_read", "/tmp/test.txt")
        assert not checker.check("file_read", "/etc/passwd")

    def test_minimal_permissions_deny_all(self):
        """Test that minimal permissions deny everything."""
        permissions = PluginPermissions(level=PermissionLevel.MINIMAL)
        checker = PermissionChecker(permissions)

        assert not checker.check("network", "https://example.com")
        assert not checker.check("file_read", "/tmp/test.txt")


# === Validation Tests ===


class TestValidation:
    """Test plugin validation."""

    def test_validate_valid_manifest(self, sample_manifest):
        """Test validating a valid manifest."""
        validator = PluginValidator()
        result = validator.validate(sample_manifest)
        assert result.valid

    def test_validate_invalid_id(self):
        """Test validation rejects invalid ID."""
        manifest = PluginManifest(
            id="INVALID_ID",  # Should be lowercase
            name="Test",
            version=SemanticVersion(1, 0, 0),
        )
        validator = PluginValidator()
        result = validator.validate(manifest)
        assert not result.valid
        assert any("ID" in e for e in result.errors)

    def test_validate_reserved_id(self):
        """Test validation rejects reserved IDs."""
        manifest = PluginManifest(
            id="aion",  # Reserved
            name="Test",
            version=SemanticVersion(1, 0, 0),
        )
        validator = PluginValidator()
        result = validator.validate(manifest)
        assert not result.valid

    def test_security_audit(self):
        """Test security audit."""
        manifest = PluginManifest(
            id="dangerous-plugin",
            name="Dangerous Plugin",
            version=SemanticVersion(1, 0, 0),
            permissions=PluginPermissions(
                level=PermissionLevel.FULL,
                subprocess_access=True,
            ),
        )
        validator = PluginValidator()
        result = validator.security_audit(manifest)
        assert len(result.warnings) > 0


# === Tool Plugin Tests ===


class SimpleTestToolPlugin(ToolPlugin):
    """Simple test tool plugin."""

    @classmethod
    def get_manifest(cls):
        return PluginManifest(
            id="test-tool",
            name="Test Tool",
            version=SemanticVersion(1, 0, 0),
            plugin_type=PluginType.TOOL,
        )

    def get_tools(self):
        return [
            Tool(
                name="test_echo",
                description="Echo the input",
                parameters=[
                    ToolParameter(
                        name="message",
                        type=ToolParameterType.STRING,
                        required=True,
                    ),
                ],
            ),
        ]

    async def execute(self, tool_name, params, context=None):
        if tool_name == "test_echo":
            return f"Echo: {params['message']}"
        raise ValueError(f"Unknown tool: {tool_name}")

    async def initialize(self, kernel, config):
        self._kernel = kernel
        self._config = config
        self._initialized = True

    async def shutdown(self):
        self._initialized = False


class TestToolPlugin:
    """Test tool plugin interface."""

    def test_get_tools(self):
        """Test getting tools from plugin."""
        plugin = SimpleTestToolPlugin()
        tools = plugin.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_echo"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool."""
        plugin = SimpleTestToolPlugin()
        await plugin.initialize(None, {})

        result = await plugin.execute("test_echo", {"message": "Hello"})
        assert result == "Echo: Hello"

    def test_validate_params(self):
        """Test parameter validation."""
        plugin = SimpleTestToolPlugin()
        valid, error = plugin.validate_params("test_echo", {"message": "test"})
        assert valid

        valid, error = plugin.validate_params("test_echo", {})
        assert not valid
        assert "message" in error


# === Plugin Loader Tests ===


class TestPluginLoader:
    """Test plugin loader."""

    def test_loader_creation(self, temp_plugin_dir):
        """Test creating a loader."""
        loader = PluginLoader([temp_plugin_dir])
        assert temp_plugin_dir in loader.plugin_dirs

    def test_discover_empty_directory(self, temp_plugin_dir):
        """Test discovery in empty directory."""
        loader = PluginLoader([temp_plugin_dir])
        manifests = loader.discover()
        assert len(manifests) == 0

    def test_discover_with_manifest(self, temp_plugin_dir):
        """Test discovery with manifest file."""
        # Create plugin directory with manifest
        plugin_dir = temp_plugin_dir / "test-plugin"
        plugin_dir.mkdir()

        manifest_data = {
            "id": "test-plugin",
            "name": "Test Plugin",
            "version": "1.0.0",
            "type": "tool",
            "entry_point": "plugin:TestPlugin",
        }
        with open(plugin_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        loader = PluginLoader([temp_plugin_dir])
        manifests = loader.discover()

        assert len(manifests) == 1
        assert manifests[0].id == "test-plugin"

    def test_get_plugin_path(self, temp_plugin_dir):
        """Test getting plugin path."""
        plugin_dir = temp_plugin_dir / "my-plugin"
        plugin_dir.mkdir()

        manifest_data = {
            "id": "my-plugin",
            "name": "My Plugin",
            "version": "1.0.0",
        }
        with open(plugin_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        loader = PluginLoader([temp_plugin_dir])
        loader.discover()

        path = loader.get_plugin_path("my-plugin")
        assert path == plugin_dir


# === Integration Tests ===


class TestPluginManagerIntegration:
    """Integration tests for plugin manager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self, temp_plugin_dir):
        """Test manager initialization."""
        manager = PluginManager(
            kernel=None,
            plugin_dirs=[temp_plugin_dir],
            enable_sandbox=False,
        )

        await manager.initialize()
        assert manager._initialized

        await manager.shutdown()
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_discover_and_load(self, temp_plugin_dir):
        """Test discovering and loading a plugin."""
        # Create test plugin
        plugin_dir = temp_plugin_dir / "test-plugin"
        plugin_dir.mkdir()

        manifest_data = {
            "id": "test-plugin",
            "name": "Test Plugin",
            "version": "1.0.0",
            "type": "tool",
            "entry_point": "plugin:TestPlugin",
        }
        with open(plugin_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        # Create plugin file
        plugin_code = '''
from aion.plugins import ToolPlugin, PluginManifest, SemanticVersion, PluginType, Tool

class TestPlugin(ToolPlugin):
    @classmethod
    def get_manifest(cls):
        return PluginManifest(
            id="test-plugin",
            name="Test Plugin",
            version=SemanticVersion(1, 0, 0),
            plugin_type=PluginType.TOOL,
        )

    def get_tools(self):
        return []

    async def execute(self, tool_name, params, context=None):
        pass

    async def initialize(self, kernel, config):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False
'''
        with open(plugin_dir / "plugin.py", "w") as f:
            f.write(plugin_code)

        manager = PluginManager(
            kernel=None,
            plugin_dirs=[temp_plugin_dir],
            enable_sandbox=False,
            auto_discover=False,
        )

        await manager.initialize()

        # Discover
        manifests = await manager.discover()
        assert len(manifests) == 1

        # Load
        success = await manager.load("test-plugin")
        assert success

        plugin = manager.get_plugin("test-plugin")
        assert plugin is not None
        assert plugin.state == PluginState.LOADED

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_lifecycle_operations(self, temp_plugin_dir, sample_plugin_info):
        """Test plugin lifecycle operations."""
        manager = PluginManager(
            kernel=None,
            plugin_dirs=[temp_plugin_dir],
            enable_sandbox=False,
        )

        # Register plugin directly
        manager.registry.register(sample_plugin_info)

        # Test state transitions
        manager.registry.set_state("test-plugin", PluginState.LOADED)
        assert manager.get_plugin("test-plugin").state == PluginState.LOADED

    def test_get_stats(self, sample_plugin_info):
        """Test getting manager stats."""
        manager = PluginManager(
            kernel=None,
            plugin_dirs=[],
            enable_sandbox=False,
        )

        manager.registry.register(sample_plugin_info)
        stats = manager.get_stats()

        assert stats["total_plugins"] == 1
        assert "by_type" in stats
        assert "by_state" in stats


# === Event Tests ===


class TestPluginEvents:
    """Test plugin events."""

    @pytest.mark.asyncio
    async def test_event_emission(self):
        """Test event emission and subscription."""
        from aion.plugins.hooks.events import PluginEventEmitter

        emitter = PluginEventEmitter()
        received_events = []

        async def handler(event_data):
            received_events.append(event_data)

        emitter.on(PluginEvent.LOADED, handler)
        await emitter.emit(PluginEvent.LOADED, "test-plugin")

        assert len(received_events) == 1
        assert received_events[0].plugin_id == "test-plugin"

    @pytest.mark.asyncio
    async def test_event_history(self):
        """Test event history tracking."""
        from aion.plugins.hooks.events import PluginEventEmitter

        emitter = PluginEventEmitter()
        await emitter.emit(PluginEvent.LOADED, "plugin-1")
        await emitter.emit(PluginEvent.ACTIVATED, "plugin-1")

        history = emitter.get_history(plugin_id="plugin-1")
        assert len(history) == 2
