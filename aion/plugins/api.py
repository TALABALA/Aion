"""
AION Plugin API Routes

REST API for plugin management.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Body, Query, Depends
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aion.plugins.manager import PluginManager

router = APIRouter(prefix="/plugins", tags=["plugins"])


# === Request/Response Models ===


class PluginConfigRequest(BaseModel):
    """Request to configure a plugin."""

    config: Dict[str, Any] = Field(default_factory=dict)


class PluginActionResponse(BaseModel):
    """Response for plugin actions."""

    status: str
    plugin_id: str
    message: Optional[str] = None


class PluginListResponse(BaseModel):
    """Response for plugin listing."""

    plugins: List[Dict[str, Any]]
    total: int


class PluginStatsResponse(BaseModel):
    """Response for plugin statistics."""

    total_plugins: int
    active_plugins: int
    by_type: Dict[str, int]
    by_state: Dict[str, int]


class PluginSearchRequest(BaseModel):
    """Request for plugin search."""

    query: Optional[str] = None
    plugin_type: Optional[str] = None
    tags: Optional[List[str]] = None
    active_only: bool = False
    limit: int = 100
    offset: int = 0


# === Route Factory ===


def create_plugin_routes(plugin_manager: "PluginManager") -> APIRouter:
    """
    Create plugin API routes with manager dependency.

    Args:
        plugin_manager: PluginManager instance

    Returns:
        Configured APIRouter
    """

    api_router = APIRouter(prefix="/plugins", tags=["plugins"])

    # === List & Search ===

    @api_router.get("", response_model=PluginListResponse)
    async def list_plugins(
        plugin_type: Optional[str] = Query(None, description="Filter by plugin type"),
        state: Optional[str] = Query(None, description="Filter by state"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
        active_only: bool = Query(False, description="Only show active plugins"),
    ) -> PluginListResponse:
        """List all plugins with optional filters."""
        plugins = plugin_manager.list_plugins()

        if plugin_type:
            plugins = [p for p in plugins if p.get("type") == plugin_type]

        if state:
            plugins = [p for p in plugins if p.get("state") == state]

        if tag:
            plugins = [
                p for p in plugins
                if tag in p.get("manifest", {}).get("tags", [])
            ]

        if active_only:
            plugins = [p for p in plugins if p.get("state") == "active"]

        return PluginListResponse(plugins=plugins, total=len(plugins))

    @api_router.post("/search", response_model=PluginListResponse)
    async def search_plugins(request: PluginSearchRequest) -> PluginListResponse:
        """Search plugins with advanced filters."""
        from aion.plugins.types import PluginType

        plugin_type = None
        if request.plugin_type:
            try:
                plugin_type = PluginType(request.plugin_type)
            except ValueError:
                pass

        results = plugin_manager.search_plugins(
            query=request.query,
            plugin_type=plugin_type,
            tags=request.tags,
            active_only=request.active_only,
        )

        plugins = [p.to_dict() for p in results]
        paginated = plugins[request.offset : request.offset + request.limit]

        return PluginListResponse(plugins=paginated, total=len(plugins))

    # === Plugin Details ===

    @api_router.get("/{plugin_id}")
    async def get_plugin(plugin_id: str) -> Dict[str, Any]:
        """Get plugin details."""
        plugin = plugin_manager.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(404, f"Plugin not found: {plugin_id}")
        return plugin.to_dict()

    @api_router.get("/{plugin_id}/health")
    async def get_plugin_health(plugin_id: str) -> Dict[str, Any]:
        """Get plugin health status."""
        health = await plugin_manager.check_health(plugin_id)
        return health

    # === Lifecycle Operations ===

    @api_router.post("/{plugin_id}/load", response_model=PluginActionResponse)
    async def load_plugin(
        plugin_id: str,
        config: PluginConfigRequest = Body(default=PluginConfigRequest()),
    ) -> PluginActionResponse:
        """Load a plugin."""
        success = await plugin_manager.load(plugin_id, config.config)
        if not success:
            plugin = plugin_manager.get_plugin(plugin_id)
            error = plugin.last_error if plugin else "Unknown error"
            raise HTTPException(500, f"Failed to load plugin: {error}")

        return PluginActionResponse(
            status="loaded",
            plugin_id=plugin_id,
            message="Plugin loaded successfully",
        )

    @api_router.post("/{plugin_id}/unload", response_model=PluginActionResponse)
    async def unload_plugin(plugin_id: str) -> PluginActionResponse:
        """Unload a plugin."""
        success = await plugin_manager.unload(plugin_id)
        if not success:
            raise HTTPException(500, f"Failed to unload plugin: {plugin_id}")

        return PluginActionResponse(
            status="unloaded",
            plugin_id=plugin_id,
            message="Plugin unloaded successfully",
        )

    @api_router.post("/{plugin_id}/reload", response_model=PluginActionResponse)
    async def reload_plugin(plugin_id: str) -> PluginActionResponse:
        """Hot-reload a plugin."""
        success = await plugin_manager.reload(plugin_id)
        if not success:
            raise HTTPException(500, f"Failed to reload plugin: {plugin_id}")

        return PluginActionResponse(
            status="reloaded",
            plugin_id=plugin_id,
            message="Plugin reloaded successfully",
        )

    @api_router.post("/{plugin_id}/activate", response_model=PluginActionResponse)
    async def activate_plugin(plugin_id: str) -> PluginActionResponse:
        """Activate a plugin."""
        success = await plugin_manager.activate(plugin_id)
        if not success:
            raise HTTPException(500, f"Failed to activate plugin: {plugin_id}")

        return PluginActionResponse(
            status="activated",
            plugin_id=plugin_id,
            message="Plugin activated successfully",
        )

    @api_router.post("/{plugin_id}/suspend", response_model=PluginActionResponse)
    async def suspend_plugin(plugin_id: str) -> PluginActionResponse:
        """Suspend a plugin."""
        success = await plugin_manager.suspend(plugin_id)
        if not success:
            raise HTTPException(500, f"Failed to suspend plugin: {plugin_id}")

        return PluginActionResponse(
            status="suspended",
            plugin_id=plugin_id,
            message="Plugin suspended successfully",
        )

    @api_router.post("/{plugin_id}/resume", response_model=PluginActionResponse)
    async def resume_plugin(plugin_id: str) -> PluginActionResponse:
        """Resume a suspended plugin."""
        success = await plugin_manager.resume(plugin_id)
        if not success:
            raise HTTPException(500, f"Failed to resume plugin: {plugin_id}")

        return PluginActionResponse(
            status="resumed",
            plugin_id=plugin_id,
            message="Plugin resumed successfully",
        )

    # === Configuration ===

    @api_router.get("/{plugin_id}/config")
    async def get_plugin_config(plugin_id: str) -> Dict[str, Any]:
        """Get plugin configuration."""
        plugin = plugin_manager.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(404, f"Plugin not found: {plugin_id}")

        return {
            "plugin_id": plugin_id,
            "config": plugin.config,
            "schema": plugin.manifest.config_schema,
        }

    @api_router.put("/{plugin_id}/config", response_model=PluginActionResponse)
    async def update_plugin_config(
        plugin_id: str,
        request: PluginConfigRequest,
    ) -> PluginActionResponse:
        """Update plugin configuration."""
        success = await plugin_manager.configure(plugin_id, request.config)
        if not success:
            raise HTTPException(500, f"Failed to configure plugin: {plugin_id}")

        return PluginActionResponse(
            status="configured",
            plugin_id=plugin_id,
            message="Plugin configuration updated",
        )

    # === Discovery ===

    @api_router.post("/discover")
    async def discover_plugins() -> Dict[str, Any]:
        """Discover new plugins."""
        manifests = await plugin_manager.discover()
        return {
            "discovered": len(manifests),
            "plugins": [m.to_dict() for m in manifests],
        }

    # === Statistics ===

    @api_router.get("/stats", response_model=PluginStatsResponse)
    async def get_stats() -> PluginStatsResponse:
        """Get plugin system statistics."""
        stats = plugin_manager.get_stats()
        return PluginStatsResponse(
            total_plugins=stats["total_plugins"],
            active_plugins=stats["active_plugins"],
            by_type=stats.get("by_type", {}),
            by_state=stats.get("by_state", {}),
        )

    # === Hooks ===

    @api_router.get("/hooks")
    async def list_hooks() -> Dict[str, Any]:
        """List all available hooks."""
        hooks = plugin_manager.hooks.list_hooks()
        active = plugin_manager.hooks.list_active_hooks()

        return {
            "hooks": hooks,
            "active": active,
            "stats": plugin_manager.hooks.get_stats(),
        }

    @api_router.get("/hooks/{hook_name}/handlers")
    async def get_hook_handlers(hook_name: str) -> Dict[str, Any]:
        """Get handlers for a specific hook."""
        handlers = plugin_manager.hooks.get_handlers(hook_name)

        return {
            "hook": hook_name,
            "handlers": [
                {
                    "plugin_id": h.plugin_id,
                    "priority": h.priority,
                    "enabled": h.enabled,
                }
                for h in handlers
            ],
        }

    # === Tools ===

    @api_router.get("/tools")
    async def list_plugin_tools() -> Dict[str, Any]:
        """List all tools from active plugins."""
        tools = plugin_manager.get_plugin_tools()

        return {
            "tools": [t.to_dict() for t in tools],
            "count": len(tools),
        }

    # === Agents ===

    @api_router.get("/agents")
    async def list_plugin_agents() -> Dict[str, Any]:
        """List all agent types from active plugins."""
        agent_types = plugin_manager.get_agent_types()

        return {
            "agent_types": agent_types,
            "count": len(agent_types),
        }

    # === Events ===

    @api_router.get("/events")
    async def get_event_history(
        plugin_id: Optional[str] = Query(None),
        limit: int = Query(100, le=1000),
    ) -> Dict[str, Any]:
        """Get plugin event history."""
        history = plugin_manager.events.get_history(plugin_id=plugin_id, limit=limit)

        return {
            "events": [e.to_dict() for e in history],
            "count": len(history),
        }

    return api_router


def setup_plugin_routes(app, plugin_manager: "PluginManager") -> None:
    """
    Setup plugin API routes on a FastAPI app.

    Args:
        app: FastAPI application
        plugin_manager: PluginManager instance
    """
    api_router = create_plugin_routes(plugin_manager)
    app.include_router(api_router, prefix="/api/v1")
