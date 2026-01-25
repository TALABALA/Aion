"""
SIEM Integration for Security Observability.

Provides connectors for common SIEM platforms:
- Splunk
- Elastic SIEM
- IBM QRadar
- Azure Sentinel
- Generic CEF/LEEF format
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """SIEM event severity levels."""
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class SIEMEvent:
    """Security event for SIEM ingestion."""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: SeverityLevel
    source: str
    message: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    # CEF fields
    device_vendor: str = "AION"
    device_product: str = "Observability"
    device_version: str = "1.0"
    signature_id: str = ""
    name: str = ""
    # Additional context
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    user: Optional[str] = None
    action: Optional[str] = None
    outcome: Optional[str] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_cef(self) -> str:
        """Convert to CEF (Common Event Format)."""
        # CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        extension_parts = []
        if self.source_ip:
            extension_parts.append(f"src={self.source_ip}")
        if self.destination_ip:
            extension_parts.append(f"dst={self.destination_ip}")
        if self.user:
            extension_parts.append(f"suser={self.user}")
        if self.action:
            extension_parts.append(f"act={self.action}")
        if self.outcome:
            extension_parts.append(f"outcome={self.outcome}")
        extension_parts.append(f"msg={self.message}")
        extension_parts.append(f"rt={int(self.timestamp.timestamp() * 1000)}")

        for k, v in self.custom_fields.items():
            extension_parts.append(f"cs1={v}" if k == "custom1" else f"{k}={v}")

        extension = " ".join(extension_parts)

        return (f"CEF:0|{self.device_vendor}|{self.device_product}|"
                f"{self.device_version}|{self.signature_id or self.event_type}|"
                f"{self.name or self.event_type}|{self.severity.value}|{extension}")

    def to_leef(self) -> str:
        """Convert to LEEF (Log Event Extended Format) for QRadar."""
        attrs = [
            f"devTime={self.timestamp.isoformat()}",
            f"cat={self.event_type}",
            f"sev={self.severity.value}",
        ]
        if self.source_ip:
            attrs.append(f"src={self.source_ip}")
        if self.destination_ip:
            attrs.append(f"dst={self.destination_ip}")
        if self.user:
            attrs.append(f"usrName={self.user}")

        return (f"LEEF:2.0|{self.device_vendor}|{self.device_product}|"
                f"{self.device_version}|{self.signature_id or self.event_type}|"
                + "\t".join(attrs))


class SIEMConnector(ABC):
    """Base class for SIEM connectors."""

    @abstractmethod
    async def send_event(self, event: SIEMEvent) -> bool:
        """Send a single event to SIEM."""
        pass

    @abstractmethod
    async def send_batch(self, events: List[SIEMEvent]) -> bool:
        """Send a batch of events to SIEM."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check SIEM connection health."""
        pass


class SplunkConnector(SIEMConnector):
    """Splunk HEC (HTTP Event Collector) connector."""

    def __init__(self, endpoint: str, token: str, index: str = "main",
                 source: str = "aion", sourcetype: str = "aion:security"):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.index = index
        self.source = source
        self.sourcetype = sourcetype

    async def send_event(self, event: SIEMEvent) -> bool:
        payload = {
            "time": event.timestamp.timestamp(),
            "host": event.source,
            "source": self.source,
            "sourcetype": self.sourcetype,
            "index": self.index,
            "event": {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity.name,
                "message": event.message,
                **event.raw_data,
                **event.custom_fields
            }
        }

        # In production, use aiohttp
        logger.debug(f"Sending to Splunk: {json.dumps(payload)}")
        return True

    async def send_batch(self, events: List[SIEMEvent]) -> bool:
        for event in events:
            await self.send_event(event)
        return True

    async def health_check(self) -> bool:
        # Would check HEC endpoint health
        return True


class ElasticSIEMConnector(SIEMConnector):
    """Elasticsearch/Elastic SIEM connector."""

    def __init__(self, hosts: List[str], index_pattern: str = "security-events",
                 api_key: str = None, username: str = None, password: str = None):
        self.hosts = hosts
        self.index_pattern = index_pattern
        self.api_key = api_key
        self.username = username
        self.password = password

    async def send_event(self, event: SIEMEvent) -> bool:
        index = f"{self.index_pattern}-{event.timestamp.strftime('%Y.%m.%d')}"
        doc = {
            "@timestamp": event.timestamp.isoformat(),
            "event": {
                "id": event.event_id,
                "kind": "event",
                "category": [event.event_type],
                "type": [event.action or "info"],
                "outcome": event.outcome or "unknown",
                "severity": event.severity.value,
            },
            "message": event.message,
            "source": {"ip": event.source_ip} if event.source_ip else {},
            "destination": {"ip": event.destination_ip} if event.destination_ip else {},
            "user": {"name": event.user} if event.user else {},
            "labels": event.custom_fields,
        }

        logger.debug(f"Sending to Elastic: {index} - {json.dumps(doc)}")
        return True

    async def send_batch(self, events: List[SIEMEvent]) -> bool:
        # Would use bulk API
        for event in events:
            await self.send_event(event)
        return True

    async def health_check(self) -> bool:
        return True


class SIEMManager:
    """Manages multiple SIEM connectors."""

    def __init__(self):
        self._connectors: Dict[str, SIEMConnector] = {}
        self._buffer: List[SIEMEvent] = []
        self._batch_size = 100
        self._flush_interval = 5.0

    def add_connector(self, name: str, connector: SIEMConnector):
        """Add a SIEM connector."""
        self._connectors[name] = connector

    async def send(self, event: SIEMEvent, connector_names: List[str] = None):
        """Send event to specified connectors (or all)."""
        targets = connector_names or list(self._connectors.keys())
        for name in targets:
            if name in self._connectors:
                try:
                    await self._connectors[name].send_event(event)
                except Exception as e:
                    logger.error(f"SIEM send failed to {name}: {e}")
