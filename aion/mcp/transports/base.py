"""
AION MCP Transport Base

Abstract base class for MCP transports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

import structlog

logger = structlog.get_logger(__name__)


class Transport(ABC):
    """
    Abstract base class for MCP transports.

    Transports handle the low-level communication with MCP servers:
    - Connection management
    - Message sending and receiving
    - Error handling
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish the connection.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the connection gracefully.
        """
        pass

    @abstractmethod
    async def send(self, message: str) -> None:
        """
        Send a message to the server.

        Args:
            message: JSON-encoded message string

        Raises:
            ConnectionError: If not connected or send fails
        """
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[str, None]:
        """
        Receive messages from the server.

        Yields:
            JSON-encoded message strings

        Raises:
            ConnectionError: If receive fails
        """
        pass

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Check if transport is connected."""
        pass

    @property
    def transport_type(self) -> str:
        """Get the transport type name."""
        return self.__class__.__name__


class TransportError(Exception):
    """Base exception for transport errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        self.cause = cause
        super().__init__(message)


class ConnectionError(TransportError):
    """Error establishing or maintaining connection."""
    pass


class SendError(TransportError):
    """Error sending message."""
    pass


class ReceiveError(TransportError):
    """Error receiving message."""
    pass


class TimeoutError(TransportError):
    """Operation timed out."""
    pass
