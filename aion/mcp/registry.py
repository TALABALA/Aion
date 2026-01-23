"""
AION MCP Server Registry

Manages MCP server configurations:
- Load from config files
- Runtime registration
- Enable/disable servers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog

from aion.mcp.types import ServerConfig, TransportType

logger = structlog.get_logger(__name__)


# Default MCP servers that AION can connect to
DEFAULT_SERVERS = [
    ServerConfig(
        name="filesystem",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/"],
        description="File system access via MCP",
        tags=["files", "io"],
        enabled=False,  # Disabled by default for security
    ),
    ServerConfig(
        name="postgres",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        description="PostgreSQL database access",
        tags=["database", "sql"],
        credential_id="postgres_default",
        enabled=False,
    ),
    ServerConfig(
        name="github",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        description="GitHub API access",
        tags=["git", "api", "vcs"],
        credential_id="github_token",
        enabled=False,
    ),
    ServerConfig(
        name="slack",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        description="Slack workspace access",
        tags=["chat", "communication"],
        credential_id="slack_token",
        enabled=False,
    ),
    ServerConfig(
        name="brave-search",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        description="Web search via Brave",
        tags=["search", "web"],
        credential_id="brave_api_key",
        enabled=False,
    ),
    ServerConfig(
        name="sqlite",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sqlite"],
        description="SQLite database access",
        tags=["database", "sql", "local"],
        enabled=False,
    ),
    ServerConfig(
        name="memory",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
        description="Persistent memory via MCP",
        tags=["memory", "storage"],
        enabled=False,
    ),
    ServerConfig(
        name="puppeteer",
        transport=TransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        description="Browser automation",
        tags=["browser", "automation", "scraping"],
        enabled=False,
    ),
]


class ServerRegistry:
    """
    Registry for MCP server configurations.

    Manages:
    - Loading configs from file/defaults
    - Runtime registration
    - Enable/disable status
    - Filtering by tags
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize server registry.

        Args:
            config_path: Path to server configuration file
        """
        self.config_path = config_path or Path("./config/mcp_servers.json")
        self._servers: dict[str, ServerConfig] = {}
        self._loaded = False

    async def load(self) -> None:
        """Load server configurations."""
        if self._loaded:
            return

        # Load defaults
        for config in DEFAULT_SERVERS:
            self._servers[config.name] = config

        # Load from file if exists
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)

                for server_data in data.get("servers", []):
                    config = ServerConfig.from_dict(server_data)
                    self._servers[config.name] = config

                logger.info(
                    "Loaded server configs from file",
                    count=len(data.get("servers", [])),
                    path=str(self.config_path),
                )

            except Exception as e:
                logger.error("Failed to load server config", error=str(e))

        self._loaded = True
        logger.info(
            "Registry loaded",
            total_servers=len(self._servers),
            enabled_servers=len(self.get_enabled_servers()),
        )

    async def save(self) -> None:
        """Save server configurations to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "servers": [
                config.to_dict()
                for config in self._servers.values()
            ]
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved server configs", path=str(self.config_path))

    def register(self, config: ServerConfig) -> None:
        """
        Register a new server configuration.

        Args:
            config: Server configuration
        """
        if config.name in self._servers:
            logger.info("Updating existing server config", name=config.name)
        else:
            logger.info("Registered new MCP server", name=config.name)

        self._servers[config.name] = config

    def unregister(self, name: str) -> bool:
        """
        Unregister a server configuration.

        Args:
            name: Server name

        Returns:
            True if server was unregistered
        """
        if name in self._servers:
            del self._servers[name]
            logger.info("Unregistered MCP server", name=name)
            return True
        return False

    def get_server(self, name: str) -> Optional[ServerConfig]:
        """Get a server configuration by name."""
        return self._servers.get(name)

    def get_all_servers(self) -> list[ServerConfig]:
        """Get all server configurations."""
        return list(self._servers.values())

    def get_enabled_servers(self) -> list[ServerConfig]:
        """Get only enabled server configurations."""
        return [s for s in self._servers.values() if s.enabled]

    def enable_server(self, name: str) -> bool:
        """
        Enable a server.

        Args:
            name: Server name

        Returns:
            True if server was enabled
        """
        if name in self._servers:
            self._servers[name].enabled = True
            logger.info("Enabled MCP server", name=name)
            return True
        return False

    def disable_server(self, name: str) -> bool:
        """
        Disable a server.

        Args:
            name: Server name

        Returns:
            True if server was disabled
        """
        if name in self._servers:
            self._servers[name].enabled = False
            logger.info("Disabled MCP server", name=name)
            return True
        return False

    def is_enabled(self, name: str) -> bool:
        """Check if a server is enabled."""
        server = self._servers.get(name)
        return server is not None and server.enabled

    def find_by_tag(self, tag: str) -> list[ServerConfig]:
        """
        Find servers by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching server configurations
        """
        return [s for s in self._servers.values() if tag in s.tags]

    def find_by_transport(self, transport: TransportType) -> list[ServerConfig]:
        """
        Find servers by transport type.

        Args:
            transport: Transport type

        Returns:
            List of matching server configurations
        """
        return [s for s in self._servers.values() if s.transport == transport]

    def search(self, query: str) -> list[ServerConfig]:
        """
        Search servers by name or description.

        Args:
            query: Search query

        Returns:
            List of matching server configurations
        """
        query_lower = query.lower()
        return [
            s for s in self._servers.values()
            if query_lower in s.name.lower() or
               (s.description and query_lower in s.description.lower())
        ]

    def to_dict(self) -> dict:
        """Convert registry to dictionary."""
        return {
            "servers": [s.to_dict() for s in self._servers.values()],
            "config_path": str(self.config_path),
        }
