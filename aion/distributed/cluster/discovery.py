"""
AION Node Discovery

Production-grade node discovery supporting multiple discovery methods:
- Static seed list: pre-configured ``host:port`` pairs
- DNS-based discovery: SRV / A-record resolution for dynamic environments
- Multicast discovery: UDP multicast for LAN-local auto-detection

Each method is implemented as an async strategy and selected via
configuration at startup.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import structlog

from aion.distributed.types import (
    NodeCapability,
    NodeInfo,
    NodeRole,
    NodeStatus,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_PORT = 5000
_DEFAULT_MULTICAST_GROUP = "239.1.2.3"
_DEFAULT_MULTICAST_PORT = 5002
_MULTICAST_TTL = 1
_DNS_POLL_INTERVAL = 30  # seconds
_DISCOVERY_TIMEOUT = 5.0  # seconds per attempt


class NodeDiscovery:
    """
    Discovers cluster peers using configurable strategies.

    Supported methods (set via ``config["discovery_method"]`` or
    ``config["discovery"]["method"]``):

    * ``static``    -- seed nodes given as ``["host:port", ...]``
    * ``dns``       -- resolve a hostname to multiple A records
    * ``multicast`` -- UDP multicast announcement/listen on a LAN
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # Accept both flat and nested config shapes
        disc_cfg = config.get("discovery", config)
        self._method: str = disc_cfg.get("method", "static")

        # Static seeds
        self._static_nodes: List[str] = disc_cfg.get("static_nodes", [])

        # DNS
        self._dns_name: Optional[str] = disc_cfg.get("dns_name")
        self._dns_port: int = disc_cfg.get("port", _DEFAULT_PORT)
        self._dns_poll_interval: float = float(
            disc_cfg.get("dns_poll_interval_seconds", _DNS_POLL_INTERVAL)
        )

        # Multicast
        self._multicast_group: str = disc_cfg.get(
            "multicast_group", _DEFAULT_MULTICAST_GROUP
        )
        self._multicast_port: int = disc_cfg.get(
            "multicast_port", _DEFAULT_MULTICAST_PORT
        )
        self._multicast_ttl: int = disc_cfg.get("multicast_ttl", _MULTICAST_TTL)

        # Internal state
        self._known_addresses: Set[str] = set()
        self._running = False
        self._poll_task: Optional[asyncio.Task[None]] = None

        logger.info(
            "node_discovery.init",
            method=self._method,
            static_nodes=len(self._static_nodes),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start periodic background discovery (DNS / multicast)."""
        if self._running:
            return
        self._running = True

        # For DNS we poll periodically
        if self._method == "dns" and self._dns_name:
            self._poll_task = asyncio.create_task(self._dns_poll_loop())
            logger.info(
                "node_discovery.dns_polling_started",
                hostname=self._dns_name,
                interval=self._dns_poll_interval,
            )

        # For multicast we start the listener
        if self._method == "multicast":
            self._poll_task = asyncio.create_task(self._multicast_listen_loop())
            logger.info(
                "node_discovery.multicast_listener_started",
                group=self._multicast_group,
                port=self._multicast_port,
            )

        logger.info("node_discovery.started", method=self._method)

    async def stop(self) -> None:
        """Stop background discovery."""
        if not self._running:
            return
        self._running = False

        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        logger.info("node_discovery.stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def discover(self) -> List[NodeInfo]:
        """
        Execute a single round of discovery and return discovered nodes.

        Returns a list of ``NodeInfo`` stubs.  The caller is responsible for
        determining whether these nodes are already known.
        """
        if self._method == "static":
            return await self._discover_static()
        elif self._method == "dns":
            return await self._discover_dns()
        elif self._method == "multicast":
            return await self._discover_multicast()
        else:
            logger.warning(
                "node_discovery.unknown_method",
                method=self._method,
            )
            return []

    # ------------------------------------------------------------------
    # Static discovery
    # ------------------------------------------------------------------

    async def _discover_static(self) -> List[NodeInfo]:
        """
        Parse the pre-configured static seed list.

        Each entry is expected to be ``"host:port"`` or just ``"host"``.
        Returns a ``NodeInfo`` stub for each entry.
        """
        nodes: List[NodeInfo] = []
        for entry in self._static_nodes:
            entry = entry.strip()
            if not entry:
                continue
            try:
                host, port = self._parse_address(entry)
                node = NodeInfo(
                    id=self._address_to_id(host, port),
                    name=f"node-{host}:{port}",
                    host=host,
                    port=port,
                    role=NodeRole.FOLLOWER,
                    status=NodeStatus.JOINING,
                    capabilities={
                        NodeCapability.COMPUTE.value,
                        NodeCapability.MEMORY.value,
                        NodeCapability.TOOLS.value,
                    },
                )
                nodes.append(node)
                self._known_addresses.add(f"{host}:{port}")
            except Exception:
                logger.warning(
                    "node_discovery.static_parse_failed",
                    entry=entry,
                )
        logger.debug(
            "node_discovery.static_discovered",
            count=len(nodes),
        )
        return nodes

    # ------------------------------------------------------------------
    # DNS discovery
    # ------------------------------------------------------------------

    async def _discover_dns(self) -> List[NodeInfo]:
        """
        Resolve the configured DNS hostname and create a ``NodeInfo`` stub
        for every resolved IP address.
        """
        if not self._dns_name:
            return []

        nodes: List[NodeInfo] = []
        try:
            loop = asyncio.get_running_loop()
            # Resolve hostname in executor to avoid blocking the event loop
            addr_infos = await loop.getaddrinfo(
                self._dns_name,
                self._dns_port,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
            seen: Set[str] = set()
            for family, kind, proto, canon, sockaddr in addr_infos:
                ip = sockaddr[0]
                port = sockaddr[1]
                addr_key = f"{ip}:{port}"
                if addr_key in seen:
                    continue
                seen.add(addr_key)
                node = NodeInfo(
                    id=self._address_to_id(ip, port),
                    name=f"node-{ip}:{port}",
                    host=ip,
                    port=port,
                    role=NodeRole.FOLLOWER,
                    status=NodeStatus.JOINING,
                )
                nodes.append(node)
                self._known_addresses.add(addr_key)

            logger.debug(
                "node_discovery.dns_discovered",
                hostname=self._dns_name,
                count=len(nodes),
            )
        except socket.gaierror as exc:
            logger.warning(
                "node_discovery.dns_resolution_failed",
                hostname=self._dns_name,
                error=str(exc),
            )
        except Exception:
            logger.exception(
                "node_discovery.dns_error",
                hostname=self._dns_name,
            )

        return nodes

    async def _dns_poll_loop(self) -> None:
        """Periodically re-resolve DNS and emit new peers."""
        while self._running:
            try:
                await self._discover_dns()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("node_discovery.dns_poll_error")
            await asyncio.sleep(self._dns_poll_interval)

    # ------------------------------------------------------------------
    # Multicast discovery
    # ------------------------------------------------------------------

    async def _discover_multicast(self) -> List[NodeInfo]:
        """
        Send a multicast probe and collect responses for a short window.

        Returns ``NodeInfo`` stubs for every responding peer.
        """
        nodes: List[NodeInfo] = []
        loop = asyncio.get_running_loop()

        try:
            # Create a UDP socket and join the multicast group
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                self._multicast_ttl,
            )
            sock.settimeout(_DISCOVERY_TIMEOUT)

            # Send probe
            probe = json.dumps({"type": "discover", "ts": datetime.now().isoformat()}).encode()
            sock.sendto(probe, (self._multicast_group, self._multicast_port))

            # Collect responses within the timeout window
            deadline = asyncio.get_event_loop().time() + _DISCOVERY_TIMEOUT
            while asyncio.get_event_loop().time() < deadline:
                try:
                    data, addr = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: sock.recvfrom(4096)),
                        timeout=max(0.1, deadline - asyncio.get_event_loop().time()),
                    )
                    node = self._parse_multicast_response(data, addr)
                    if node is not None:
                        nodes.append(node)
                        self._known_addresses.add(f"{node.host}:{node.port}")
                except asyncio.TimeoutError:
                    break
                except Exception:
                    break

            sock.close()
        except Exception:
            logger.exception("node_discovery.multicast_probe_error")

        logger.debug(
            "node_discovery.multicast_discovered",
            count=len(nodes),
        )
        return nodes

    async def _multicast_listen_loop(self) -> None:
        """Background listener for multicast announcements."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self._multicast_port))

            # Join multicast group
            mreq = struct.pack(
                "4sl",
                socket.inet_aton(self._multicast_group),
                socket.INADDR_ANY,
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(1.0)

            logger.info(
                "node_discovery.multicast_listening",
                group=self._multicast_group,
                port=self._multicast_port,
            )

            loop = asyncio.get_running_loop()
            while self._running:
                try:
                    data, addr = await loop.run_in_executor(
                        None, lambda: sock.recvfrom(4096)
                    )
                    node = self._parse_multicast_response(data, addr)
                    if node is not None:
                        self._known_addresses.add(f"{node.host}:{node.port}")
                except socket.timeout:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.debug("node_discovery.multicast_recv_error")

            sock.close()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("node_discovery.multicast_listener_error")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_address(entry: str) -> tuple:
        """Parse ``'host:port'`` into ``(host, port)``."""
        if ":" in entry:
            parts = entry.rsplit(":", 1)
            return parts[0], int(parts[1])
        return entry, _DEFAULT_PORT

    @staticmethod
    def _address_to_id(host: str, port: int) -> str:
        """Derive a deterministic node id from an address."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aion://{host}:{port}"))

    @staticmethod
    def _parse_multicast_response(
        data: bytes, addr: tuple
    ) -> Optional[NodeInfo]:
        """Try to parse a multicast response payload into ``NodeInfo``."""
        try:
            payload = json.loads(data.decode())
            if payload.get("type") not in ("announce", "discover_response"):
                return None
            host = payload.get("host", addr[0])
            port = int(payload.get("port", _DEFAULT_PORT))
            node = NodeInfo(
                id=payload.get("node_id", str(uuid.uuid4())),
                name=payload.get("name", f"node-{host}:{port}"),
                host=host,
                port=port,
                role=NodeRole.FOLLOWER,
                status=NodeStatus.JOINING,
                capabilities=set(payload.get("capabilities", [
                    NodeCapability.COMPUTE.value,
                ])),
            )
            return node
        except Exception:
            return None

    @property
    def known_addresses(self) -> Set[str]:
        """Return all addresses discovered so far."""
        return set(self._known_addresses)
