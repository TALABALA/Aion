"""
AION Conversation CLI

Command-line interface for conversations.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from typing import Optional

import structlog

from aion.conversation.manager import ConversationManager
from aion.conversation.types import ConversationRequest, ConversationConfig

logger = structlog.get_logger(__name__)


class ConversationCLI:
    """
    Command-line interface for AION conversations.

    Provides an interactive chat experience in the terminal.
    """

    def __init__(
        self,
        manager: ConversationManager,
        config: Optional[ConversationConfig] = None,
    ):
        self.manager = manager
        self.config = config
        self._conversation_id: Optional[str] = None
        self._running = False

    async def run(self, conversation_id: Optional[str] = None):
        """Run the interactive CLI."""
        self._running = True

        print("\n" + "=" * 60)
        print("  AION - Artificial Intelligence Operating Nexus")
        print("=" * 60)
        print("\nType 'help' for commands, 'exit' to quit\n")

        if not self.manager.is_initialized:
            print("Initializing...")
            await self.manager.initialize()
            print("Ready!\n")

        if conversation_id:
            conversation = await self.manager.get_conversation(conversation_id)
            if conversation:
                self._conversation_id = conversation_id
                print(f"Resumed conversation: {conversation_id[:8]}...")
                if conversation.title:
                    print(f"Title: {conversation.title}")
                print(f"Messages: {len(conversation.messages)}\n")
            else:
                print(f"Conversation {conversation_id} not found, creating new...")
                await self._new_conversation()
        else:
            await self._new_conversation()

        while self._running:
            try:
                user_input = await self._get_input("\nYou: ")

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    await self._handle_command(user_input[1:])
                    continue

                await self._send_message(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' or Ctrl+C again to quit.")
                try:
                    await asyncio.sleep(0.5)
                except KeyboardInterrupt:
                    break

            except EOFError:
                break

            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"CLI error: {e}")

        print("\nGoodbye!")

    async def _get_input(self, prompt: str) -> str:
        """Get input from user (async-friendly)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt).strip())

    async def _send_message(self, message: str):
        """Send a message and stream the response."""
        print("\nAION: ", end="", flush=True)

        request = ConversationRequest(
            message=message,
            conversation_id=self._conversation_id,
        )

        tools_used = []
        text_buffer = ""

        try:
            async for event in self.manager.chat_stream(request):
                if event.type == "text":
                    text = event.data or ""
                    text_buffer += text
                    print(text, end="", flush=True)

                elif event.type == "thinking":
                    pass

                elif event.type == "tool_use_start":
                    tool_name = event.data
                    print(f"\n  [Using: {tool_name}]", end="", flush=True)
                    tools_used.append(tool_name)

                elif event.type == "tool_executing":
                    print("...", end="", flush=True)

                elif event.type == "tool_result":
                    data = event.data
                    if data.get("is_error"):
                        print(f" Error!", end="", flush=True)
                    else:
                        print(f" Done", end="", flush=True)

                elif event.type == "conversation_created":
                    self._conversation_id = event.data

                elif event.type == "done":
                    print()
                    data = event.data or {}

                    if tools_used:
                        print(f"  [Tools: {', '.join(tools_used)}]")

                    latency = data.get("latency_ms", 0)
                    tokens = data.get("input_tokens", 0) + data.get("output_tokens", 0)
                    if latency > 0:
                        print(f"  [{latency:.0f}ms, {tokens} tokens]")

                elif event.type == "error":
                    print(f"\nError: {event.data}")

        except Exception as e:
            print(f"\nStream error: {e}")

    async def _handle_command(self, command: str):
        """Handle a CLI command."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            "help": self._cmd_help,
            "h": self._cmd_help,
            "new": self._cmd_new,
            "n": self._cmd_new,
            "history": self._cmd_history,
            "hist": self._cmd_history,
            "clear": self._cmd_clear,
            "title": self._cmd_title,
            "info": self._cmd_info,
            "list": self._cmd_list,
            "load": self._cmd_load,
            "export": self._cmd_export,
            "stats": self._cmd_stats,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "q": self._cmd_exit,
        }

        handler = commands.get(cmd)
        if handler:
            await handler(args)
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")

    async def _cmd_help(self, args: str):
        """Show help message."""
        print("""
Available Commands:
  /help, /h          - Show this help message
  /new, /n           - Start a new conversation
  /history, /hist    - Show conversation history
  /clear             - Clear current conversation
  /title <title>     - Set conversation title
  /info              - Show conversation info
  /list              - List recent conversations
  /load <id>         - Load a conversation by ID
  /export [file]     - Export conversation
  /stats             - Show system statistics
  /exit, /quit, /q   - Exit the CLI

Tips:
  - Just type your message and press Enter to chat
  - Use Ctrl+C to interrupt a response
  - Commands start with /
        """)

    async def _cmd_new(self, args: str):
        """Start a new conversation."""
        await self._new_conversation()

    async def _new_conversation(self):
        """Create a new conversation."""
        conversation = await self.manager.create_conversation(
            config=self.config,
        )
        self._conversation_id = conversation.id
        print(f"Started new conversation: {conversation.id[:8]}...")

    async def _cmd_history(self, args: str):
        """Show conversation history."""
        if not self._conversation_id:
            print("No active conversation")
            return

        conversation = await self.manager.get_conversation(self._conversation_id)
        if not conversation:
            print("Conversation not found")
            return

        limit = 10
        if args:
            try:
                limit = int(args)
            except ValueError:
                pass

        print(f"\nConversation History (last {min(limit, len(conversation.messages))} messages)")
        print("-" * 50)

        for msg in conversation.messages[-limit:]:
            role = "You" if msg.role.value == "user" else "AION"
            text = msg.get_text()

            if len(text) > 200:
                text = text[:200] + "..."

            timestamp = msg.created_at.strftime("%H:%M")
            print(f"\n[{timestamp}] {role}:")
            print(f"  {text}")

        print()

    async def _cmd_clear(self, args: str):
        """Clear current conversation."""
        if not self._conversation_id:
            print("No active conversation")
            return

        conversation = await self.manager.get_conversation(self._conversation_id)
        if conversation:
            conversation.clear_messages()
            await self.manager.sessions.save_conversation(conversation)
            print("Conversation cleared")

    async def _cmd_title(self, args: str):
        """Set conversation title."""
        if not self._conversation_id:
            print("No active conversation")
            return

        if not args:
            print("Usage: /title <title>")
            return

        conversation = await self.manager.get_conversation(self._conversation_id)
        if conversation:
            conversation.title = args
            await self.manager.sessions.save_conversation(conversation)
            print(f"Title set to: {args}")

    async def _cmd_info(self, args: str):
        """Show conversation info."""
        if not self._conversation_id:
            print("No active conversation")
            return

        conversation = await self.manager.get_conversation(self._conversation_id)
        if not conversation:
            print("Conversation not found")
            return

        print(f"""
Conversation Info:
  ID: {conversation.id}
  Title: {conversation.title or '(untitled)'}
  Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M')}
  Updated: {conversation.updated_at.strftime('%Y-%m-%d %H:%M')}
  Messages: {conversation.message_count}
  Tokens: {conversation.total_input_tokens + conversation.total_output_tokens}
  Model: {conversation.config.model}
        """)

    async def _cmd_list(self, args: str):
        """List recent conversations."""
        limit = 10
        if args:
            try:
                limit = int(args)
            except ValueError:
                pass

        conversations = await self.manager.list_conversations(limit=limit)

        if not conversations:
            print("No conversations found")
            return

        print(f"\nRecent Conversations ({len(conversations)}):")
        print("-" * 60)

        for conv in conversations:
            title = conv.title or "(untitled)"
            if len(title) > 30:
                title = title[:27] + "..."

            updated = conv.updated_at.strftime("%Y-%m-%d %H:%M")
            active = " *" if conv.id == self._conversation_id else ""

            print(f"  {conv.id[:8]}  {title:<30}  {conv.message_count:>3} msgs  {updated}{active}")

        print()

    async def _cmd_load(self, args: str):
        """Load a conversation."""
        if not args:
            print("Usage: /load <conversation_id>")
            return

        conversation = await self.manager.get_conversation(args)
        if not conversation:
            conversations = await self.manager.list_conversations(limit=100)
            matching = [c for c in conversations if c.id.startswith(args)]

            if len(matching) == 1:
                conversation = matching[0]
            elif len(matching) > 1:
                print(f"Multiple matches for '{args}':")
                for c in matching[:5]:
                    print(f"  {c.id}")
                return
            else:
                print(f"Conversation not found: {args}")
                return

        self._conversation_id = conversation.id
        print(f"Loaded conversation: {conversation.id[:8]}...")
        if conversation.title:
            print(f"Title: {conversation.title}")
        print(f"Messages: {len(conversation.messages)}")

    async def _cmd_export(self, args: str):
        """Export conversation."""
        if not self._conversation_id:
            print("No active conversation")
            return

        conversation = await self.manager.get_conversation(self._conversation_id)
        if not conversation:
            print("Conversation not found")
            return

        filename = args or f"conversation_{self._conversation_id[:8]}.md"

        lines = [f"# {conversation.title or 'Conversation'}\n"]
        lines.append(f"ID: {conversation.id}\n")
        lines.append(f"Created: {conversation.created_at.isoformat()}\n")
        lines.append("---\n")

        for msg in conversation.messages:
            role = "**User:**" if msg.role.value == "user" else "**AION:**"
            lines.append(f"\n{role}\n\n{msg.get_text()}\n")

        try:
            with open(filename, "w") as f:
                f.write("\n".join(lines))
            print(f"Exported to: {filename}")
        except Exception as e:
            print(f"Export failed: {e}")

    async def _cmd_stats(self, args: str):
        """Show system statistics."""
        stats = self.manager.get_stats()

        print("\nSystem Statistics:")
        print("-" * 40)
        print(f"  Conversations created: {stats.get('conversations_created', 0)}")
        print(f"  Messages processed: {stats.get('messages_processed', 0)}")
        print(f"  Tool calls: {stats.get('tool_calls', 0)}")
        print(f"  Memory retrievals: {stats.get('memory_retrievals', 0)}")
        print(f"  Active sessions: {stats.get('active_sessions', 0)}")
        print(f"  Errors: {stats.get('errors', 0)}")
        print(f"  Total tokens: {stats.get('total_input_tokens', 0) + stats.get('total_output_tokens', 0)}")
        print(f"  Avg latency: {stats.get('avg_latency_ms', 0):.1f}ms")
        print()

    async def _cmd_exit(self, args: str):
        """Exit the CLI."""
        self._running = False


async def main(conversation_id: Optional[str] = None):
    """CLI entry point."""
    manager = ConversationManager()
    cli = ConversationCLI(manager)

    try:
        await cli.run(conversation_id)
    finally:
        await manager.shutdown()


def run_cli():
    """Synchronous entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="AION Conversation CLI")
    parser.add_argument(
        "--conversation", "-c",
        help="Resume an existing conversation by ID",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-20250514",
        help="Model to use",
    )

    args = parser.parse_args()

    asyncio.run(main(args.conversation))


if __name__ == "__main__":
    run_cli()
