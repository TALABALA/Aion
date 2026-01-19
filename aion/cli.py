"""
AION Command Line Interface

Provides command-line access to AION functionality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import httpx


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="aion",
        description="AION - Artificial Intelligence Operating Nexus CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the AION server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port")
    server_parser.add_argument("--reload", action="store_true", help="Auto-reload")
    server_parser.add_argument("--workers", type=int, default=1, help="Workers")

    # Query command
    query_parser = subparsers.add_parser("query", help="Send a query to AION")
    query_parser.add_argument("query", help="The query to process")
    query_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get system status")
    status_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")

    # Memory commands
    memory_parser = subparsers.add_parser("memory", help="Memory operations")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")

    memory_store = memory_sub.add_parser("store", help="Store a memory")
    memory_store.add_argument("content", help="Memory content")
    memory_store.add_argument("--importance", type=float, default=0.5)
    memory_store.add_argument("--url", default="http://localhost:8000")

    memory_search = memory_sub.add_parser("search", help="Search memories")
    memory_search.add_argument("query", help="Search query")
    memory_search.add_argument("--limit", type=int, default=10)
    memory_search.add_argument("--url", default="http://localhost:8000")

    # Tool commands
    tool_parser = subparsers.add_parser("tool", help="Tool operations")
    tool_sub = tool_parser.add_subparsers(dest="tool_command")

    tool_list = tool_sub.add_parser("list", help="List available tools")
    tool_list.add_argument("--url", default="http://localhost:8000")

    tool_exec = tool_sub.add_parser("exec", help="Execute a tool")
    tool_exec.add_argument("name", help="Tool name")
    tool_exec.add_argument("--params", type=json.loads, default={})
    tool_exec.add_argument("--url", default="http://localhost:8000")

    # Vision commands
    vision_parser = subparsers.add_parser("vision", help="Vision operations")
    vision_sub = vision_parser.add_subparsers(dest="vision_command")

    vision_analyze = vision_sub.add_parser("analyze", help="Analyze an image")
    vision_analyze.add_argument("image_url", help="Image URL or path")
    vision_analyze.add_argument("--question", help="Question about the image")
    vision_analyze.add_argument("--url", default="http://localhost:8000")

    vision_describe = vision_sub.add_parser("describe", help="Describe an image")
    vision_describe.add_argument("image_url", help="Image URL or path")
    vision_describe.add_argument("--url", default="http://localhost:8000")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "server":
        from aion.main import run_server
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
        )

    elif args.command == "query":
        asyncio.run(cmd_query(args.url, args.query))

    elif args.command == "status":
        asyncio.run(cmd_status(args.url))

    elif args.command == "memory":
        if args.memory_command == "store":
            asyncio.run(cmd_memory_store(args.url, args.content, args.importance))
        elif args.memory_command == "search":
            asyncio.run(cmd_memory_search(args.url, args.query, args.limit))
        else:
            memory_parser.print_help()

    elif args.command == "tool":
        if args.tool_command == "list":
            asyncio.run(cmd_tool_list(args.url))
        elif args.tool_command == "exec":
            asyncio.run(cmd_tool_exec(args.url, args.name, args.params))
        else:
            tool_parser.print_help()

    elif args.command == "vision":
        if args.vision_command == "analyze":
            asyncio.run(cmd_vision_analyze(args.url, args.image_url, args.question))
        elif args.vision_command == "describe":
            asyncio.run(cmd_vision_describe(args.url, args.image_url))
        else:
            vision_parser.print_help()


async def cmd_query(base_url: str, query: str) -> None:
    """Send a query to AION."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/process",
            json={"query": query},
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def cmd_status(base_url: str) -> None:
    """Get system status."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/status", timeout=10.0)

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")


async def cmd_memory_store(base_url: str, content: str, importance: float) -> None:
    """Store a memory."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/memory/store",
            json={"content": content, "importance": importance},
            timeout=10.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")


async def cmd_memory_search(base_url: str, query: str, limit: int) -> None:
    """Search memories."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/memory/search",
            json={"query": query, "limit": limit},
            timeout=10.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")


async def cmd_tool_list(base_url: str) -> None:
    """List available tools."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/tools", timeout=10.0)

        if response.status_code == 200:
            result = response.json()
            for tool in result["tools"]:
                print(f"- {tool['name']}: {tool['description']}")
        else:
            print(f"Error: {response.status_code}")


async def cmd_tool_exec(base_url: str, name: str, params: dict) -> None:
    """Execute a tool."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/tools/execute",
            json={"tool_name": name, "params": params},
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")


async def cmd_vision_analyze(
    base_url: str, image_url: str, question: Optional[str]
) -> None:
    """Analyze an image."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/vision/analyze",
            json={"image_url": image_url, "question": question},
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")


async def cmd_vision_describe(base_url: str, image_url: str) -> None:
    """Describe an image."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/vision/describe",
            json=image_url,
            timeout=30.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(result.get("description", "No description"))
        else:
            print(f"Error: {response.status_code}")


if __name__ == "__main__":
    main()
