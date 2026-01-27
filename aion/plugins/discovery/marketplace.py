"""
AION Plugin Marketplace Client

Client for the AION plugin marketplace.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import structlog

from aion.plugins.types import PluginManifest, SemanticVersion

logger = structlog.get_logger(__name__)


@dataclass
class MarketplacePlugin:
    """Plugin listing from marketplace."""

    id: str
    name: str
    description: str
    version: SemanticVersion
    author: str
    downloads: int = 0
    rating: float = 0.0
    ratings_count: int = 0
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    icon_url: str = ""
    homepage: str = ""
    repository: str = ""
    verified: bool = False
    featured: bool = False
    manifest: Optional[PluginManifest] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": str(self.version),
            "author": self.author,
            "downloads": self.downloads,
            "rating": self.rating,
            "tags": self.tags,
            "verified": self.verified,
            "featured": self.featured,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketplacePlugin":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=SemanticVersion.parse(data.get("version", "0.0.0")),
            author=data.get("author", ""),
            downloads=data.get("downloads", 0),
            rating=data.get("rating", 0.0),
            ratings_count=data.get("ratings_count", 0),
            published_at=datetime.fromisoformat(data["published_at"]) if data.get("published_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            icon_url=data.get("icon_url", ""),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            verified=data.get("verified", False),
            featured=data.get("featured", False),
        )


@dataclass
class SearchResult:
    """Search results from marketplace."""

    plugins: List[MarketplacePlugin]
    total: int
    page: int
    page_size: int
    has_more: bool


class MarketplaceClient:
    """
    Client for AION plugin marketplace.

    Features:
    - Search plugins
    - Get plugin details
    - Download plugins
    - Check for updates
    - Submit ratings/reviews
    """

    def __init__(
        self,
        base_url: str = "https://plugins.aion.ai/api/v1",
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._cache_dir = cache_dir or Path.home() / ".aion" / "marketplace_cache"
        self._session = None

    async def initialize(self) -> None:
        """Initialize the marketplace client."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
            )
        except ImportError:
            logger.warning("aiohttp not installed, marketplace features disabled")

    async def shutdown(self) -> None:
        """Shutdown the marketplace client."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AION-Plugin-System/1.0",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # === Search & Browse ===

    async def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        verified_only: bool = False,
        sort_by: str = "relevance",
        page: int = 1,
        page_size: int = 20,
    ) -> SearchResult:
        """
        Search for plugins.

        Args:
            query: Search query
            tags: Filter by tags
            category: Filter by category
            verified_only: Only show verified plugins
            sort_by: Sort field (relevance, downloads, rating, updated)
            page: Page number
            page_size: Results per page

        Returns:
            SearchResult with plugins
        """
        if not self._session:
            return SearchResult([], 0, page, page_size, False)

        params = {
            "page": page,
            "page_size": page_size,
            "sort_by": sort_by,
        }
        if query:
            params["q"] = query
        if tags:
            params["tags"] = ",".join(tags)
        if category:
            params["category"] = category
        if verified_only:
            params["verified"] = "true"

        try:
            async with self._session.get(
                f"{self._base_url}/plugins",
                params=params,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    plugins = [
                        MarketplacePlugin.from_dict(p)
                        for p in data.get("plugins", [])
                    ]
                    return SearchResult(
                        plugins=plugins,
                        total=data.get("total", len(plugins)),
                        page=page,
                        page_size=page_size,
                        has_more=data.get("has_more", False),
                    )
        except Exception as e:
            logger.error(f"Marketplace search error: {e}")

        return SearchResult([], 0, page, page_size, False)

    async def get_featured(self) -> List[MarketplacePlugin]:
        """Get featured plugins."""
        if not self._session:
            return []

        try:
            async with self._session.get(
                f"{self._base_url}/plugins/featured"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        MarketplacePlugin.from_dict(p)
                        for p in data.get("plugins", [])
                    ]
        except Exception as e:
            logger.error(f"Failed to get featured plugins: {e}")

        return []

    async def get_popular(self, limit: int = 10) -> List[MarketplacePlugin]:
        """Get popular plugins."""
        result = await self.search(sort_by="downloads", page_size=limit)
        return result.plugins

    async def get_recent(self, limit: int = 10) -> List[MarketplacePlugin]:
        """Get recently updated plugins."""
        result = await self.search(sort_by="updated", page_size=limit)
        return result.plugins

    # === Plugin Details ===

    async def get_plugin(self, plugin_id: str) -> Optional[MarketplacePlugin]:
        """Get plugin details."""
        if not self._session:
            return None

        try:
            async with self._session.get(
                f"{self._base_url}/plugins/{plugin_id}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return MarketplacePlugin.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to get plugin {plugin_id}: {e}")

        return None

    async def get_versions(self, plugin_id: str) -> List[SemanticVersion]:
        """Get available versions for a plugin."""
        if not self._session:
            return []

        try:
            async with self._session.get(
                f"{self._base_url}/plugins/{plugin_id}/versions"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        SemanticVersion.parse(v["version"])
                        for v in data.get("versions", [])
                    ]
        except Exception as e:
            logger.error(f"Failed to get versions for {plugin_id}: {e}")

        return []

    # === Download ===

    async def download(
        self,
        plugin_id: str,
        version: Optional[str] = None,
        target_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Download a plugin.

        Args:
            plugin_id: Plugin to download
            version: Specific version (or latest)
            target_dir: Directory to save to

        Returns:
            Path to downloaded plugin
        """
        if not self._session:
            return None

        target_dir = target_dir or self._cache_dir / "downloads"
        target_dir.mkdir(parents=True, exist_ok=True)

        url = f"{self._base_url}/plugins/{plugin_id}/download"
        if version:
            url += f"?version={version}"

        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    # Get filename from header or use plugin_id
                    filename = f"{plugin_id}.zip"
                    if "Content-Disposition" in response.headers:
                        cd = response.headers["Content-Disposition"]
                        if "filename=" in cd:
                            filename = cd.split("filename=")[1].strip('"')

                    target_path = target_dir / filename

                    with open(target_path, "wb") as f:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)

                    logger.info(f"Downloaded plugin: {plugin_id} to {target_path}")
                    return target_path

        except Exception as e:
            logger.error(f"Failed to download plugin {plugin_id}: {e}")

        return None

    # === Updates ===

    async def check_updates(
        self,
        installed: Dict[str, SemanticVersion],
    ) -> Dict[str, SemanticVersion]:
        """
        Check for plugin updates.

        Args:
            installed: Dict of plugin_id -> installed version

        Returns:
            Dict of plugin_id -> available version for those with updates
        """
        if not self._session:
            return {}

        updates = {}

        try:
            async with self._session.post(
                f"{self._base_url}/plugins/check-updates",
                json={"plugins": {k: str(v) for k, v in installed.items()}},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    for plugin_id, version_str in data.get("updates", {}).items():
                        updates[plugin_id] = SemanticVersion.parse(version_str)

        except Exception as e:
            logger.error(f"Failed to check updates: {e}")

        return updates

    # === Ratings ===

    async def submit_rating(
        self,
        plugin_id: str,
        rating: int,
        review: Optional[str] = None,
    ) -> bool:
        """
        Submit a rating for a plugin.

        Args:
            plugin_id: Plugin to rate
            rating: Rating (1-5)
            review: Optional review text

        Returns:
            True if successful
        """
        if not self._session or not self._api_key:
            return False

        try:
            async with self._session.post(
                f"{self._base_url}/plugins/{plugin_id}/ratings",
                json={"rating": rating, "review": review},
            ) as response:
                return response.status == 201

        except Exception as e:
            logger.error(f"Failed to submit rating: {e}")

        return False

    # === Categories ===

    async def get_categories(self) -> List[Dict[str, Any]]:
        """Get available plugin categories."""
        if not self._session:
            return []

        try:
            async with self._session.get(
                f"{self._base_url}/categories"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("categories", [])

        except Exception as e:
            logger.error(f"Failed to get categories: {e}")

        return []

    async def get_tags(self) -> List[str]:
        """Get popular tags."""
        if not self._session:
            return []

        try:
            async with self._session.get(
                f"{self._base_url}/tags"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("tags", [])

        except Exception as e:
            logger.error(f"Failed to get tags: {e}")

        return []
