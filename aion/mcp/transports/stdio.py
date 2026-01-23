"""
AION MCP Stdio Transport

Communicates with MCP servers via subprocess stdin/stdout.
This is the most common transport for local MCP servers.
"""

from __future__ import annotations

import asyncio
import os
import signal
from typing import AsyncGenerator, Optional

import structlog

from aion.mcp.transports.base import (
    Transport,
    ConnectionError,
    SendError,
    ReceiveError,
)

logger = structlog.get_logger(__name__)


class StdioTransport(Transport):
    """
    Stdio transport for MCP.

    Spawns a subprocess and communicates via stdin/stdout using
    newline-delimited JSON messages.
    """

    def __init__(
        self,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        startup_timeout: float = 30.0,
    ):
        """
        Initialize stdio transport.

        Args:
            command: Command to execute
            args: Command arguments
            env: Additional environment variables
            cwd: Working directory for the subprocess
            startup_timeout: Timeout for subprocess startup
        """
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        self.startup_timeout = startup_timeout

        self._process: Optional[asyncio.subprocess.Process] = None
        self._connected = False
        self._stderr_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Start the subprocess."""
        if self._connected:
            return

        # Merge environment
        full_env = os.environ.copy()
        full_env.update(self.env)

        logger.info(
            "Starting MCP server process",
            command=self.command,
            args=self.args,
        )

        try:
            # Start process
            self._process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
                cwd=self.cwd,
            )

            # Start stderr reader for logging
            self._stderr_task = asyncio.create_task(self._read_stderr())

            self._connected = True

            logger.debug(
                "MCP server process started",
                command=self.command,
                pid=self._process.pid,
            )

        except FileNotFoundError:
            raise ConnectionError(f"Command not found: {self.command}")
        except PermissionError:
            raise ConnectionError(f"Permission denied: {self.command}")
        except Exception as e:
            raise ConnectionError(f"Failed to start process: {e}", cause=e)

    async def close(self) -> None:
        """Terminate the subprocess."""
        if not self._connected or not self._process:
            return

        logger.debug("Stopping MCP server process", command=self.command)

        # Cancel stderr reader
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        # Try graceful termination first
        try:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill if graceful termination fails
                logger.warning("Process did not terminate gracefully, killing")
                self._process.kill()
                await self._process.wait()

        except ProcessLookupError:
            # Process already terminated
            pass
        except Exception as e:
            logger.error("Error terminating process", error=str(e))

        self._connected = False
        self._process = None

        logger.debug("MCP server process stopped", command=self.command)

    async def send(self, message: str) -> None:
        """Send a message to the server."""
        if not self._connected or not self._process or not self._process.stdin:
            raise SendError("Transport not connected")

        try:
            # MCP uses newline-delimited JSON
            data = (message + "\n").encode("utf-8")
            self._process.stdin.write(data)
            await self._process.stdin.drain()

            logger.debug(
                "Sent message to MCP server",
                message_length=len(message),
            )

        except BrokenPipeError:
            self._connected = False
            raise SendError("Process stdin closed")
        except Exception as e:
            raise SendError(f"Failed to send message: {e}", cause=e)

    async def receive(self) -> AsyncGenerator[str, None]:
        """Receive messages from the server."""
        if not self._connected or not self._process or not self._process.stdout:
            raise ReceiveError("Transport not connected")

        while self._connected:
            try:
                line = await self._process.stdout.readline()

                if not line:
                    # EOF - process terminated
                    logger.debug("MCP server process stdout closed")
                    self._connected = False
                    break

                message = line.decode("utf-8").strip()
                if message:
                    logger.debug(
                        "Received message from MCP server",
                        message_length=len(message),
                    )
                    yield message

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error receiving message", error=str(e))
                raise ReceiveError(f"Failed to receive message: {e}", cause=e)

    async def _read_stderr(self) -> None:
        """Background task to read and log stderr."""
        if not self._process or not self._process.stderr:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break

                stderr_text = line.decode("utf-8").strip()
                if stderr_text:
                    logger.debug(
                        "MCP server stderr",
                        command=self.command,
                        output=stderr_text,
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Error reading stderr", error=str(e))

    @property
    def connected(self) -> bool:
        """Check if transport is connected."""
        if not self._connected or not self._process:
            return False

        # Check if process is still running
        if self._process.returncode is not None:
            self._connected = False
            return False

        return True

    @property
    def process_pid(self) -> Optional[int]:
        """Get the subprocess PID."""
        if self._process:
            return self._process.pid
        return None

    @property
    def return_code(self) -> Optional[int]:
        """Get the subprocess return code (if terminated)."""
        if self._process:
            return self._process.returncode
        return None
