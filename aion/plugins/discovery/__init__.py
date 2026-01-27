"""
AION Plugin Discovery

Plugin discovery from various sources.
"""

from aion.plugins.discovery.local import LocalDiscovery
from aion.plugins.discovery.marketplace import MarketplaceClient

__all__ = [
    "LocalDiscovery",
    "MarketplaceClient",
]
