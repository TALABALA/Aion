"""
AION Memory Integrator

Integrates AION's memory system with the conversation interface.
Handles:
- Retrieving relevant memories for context
- Storing new memories from conversations
- Memory importance scoring
- Automatic memory consolidation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
import hashlib

import structlog

from aion.conversation.types import Message, MessageRole

logger = structlog.get_logger(__name__)


class MemoryIntegrator:
    """
    Integrates memory with the conversation system.

    Features:
    - Retrieving relevant memories for context enrichment
    - Storing conversation interactions as memories
    - Automatic importance scoring
    - Deduplication of stored memories
    """

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        default_importance: float = 0.5,
        min_importance_threshold: float = 0.3,
    ):
        self._memory = memory_system
        self.default_importance = default_importance
        self.min_importance_threshold = min_importance_threshold

        self._recent_hashes: set[str] = set()
        self._max_recent_hashes = 1000

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory integrator."""
        if self._initialized:
            return

        if self._memory and hasattr(self._memory, "initialize"):
            await self._memory.initialize()

        self._initialized = True
        logger.info("Memory integrator initialized")

    async def shutdown(self) -> None:
        """Shutdown the memory integrator."""
        self._recent_hashes.clear()
        self._initialized = False

    async def retrieve_relevant(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[list[str]] = None,
        min_similarity: float = 0.5,
    ) -> list[Any]:
        """
        Retrieve memories relevant to a query.

        Args:
            query: Search query
            limit: Maximum memories to return
            memory_types: Filter by memory types (episodic, semantic, procedural)
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant memories
        """
        if not self._memory:
            return []

        try:
            if hasattr(self._memory, "search"):
                results = await self._memory.search(
                    query=query,
                    limit=limit,
                    memory_types=memory_types,
                )

                if min_similarity > 0 and results:
                    results = [
                        r for r in results
                        if getattr(r, "similarity", 1.0) >= min_similarity
                    ]

                logger.debug(
                    "Retrieved memories",
                    query=query[:50],
                    count=len(results),
                )

                return results

            elif hasattr(self._memory, "recall"):
                result = await self._memory.recall(query, limit=limit)
                return result if isinstance(result, list) else [result] if result else []

        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []

        return []

    async def store_interaction(
        self,
        user_message: Message,
        assistant_message: Message,
        importance: Optional[float] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a conversation interaction as memory.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            importance: Importance score (0-1), auto-calculated if not provided
            conversation_id: Optional conversation ID for metadata

        Returns:
            Memory ID if stored, None if skipped
        """
        if not self._memory:
            return None

        try:
            user_text = user_message.get_text()
            assistant_text = assistant_message.get_text()

            content = f"User: {user_text}\nAssistant: {assistant_text[:1000]}"

            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self._recent_hashes:
                logger.debug("Skipping duplicate memory")
                return None

            if importance is None:
                importance = self._calculate_importance(user_message, assistant_message)

            if importance < self.min_importance_threshold:
                logger.debug(
                    "Skipping low-importance memory",
                    importance=importance,
                )
                return None

            metadata = {
                "type": "conversation",
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "timestamp": datetime.now().isoformat(),
            }

            if conversation_id:
                metadata["conversation_id"] = conversation_id

            if assistant_message.has_tool_use():
                metadata["has_tool_use"] = True
                tool_names = [t.name for t in assistant_message.get_tool_uses()]
                metadata["tools_used"] = tool_names

            memory_id = await self._store_memory(
                content=content,
                memory_type="episodic",
                importance=importance,
                metadata=metadata,
            )

            if memory_id:
                self._recent_hashes.add(content_hash)
                if len(self._recent_hashes) > self._max_recent_hashes:
                    self._recent_hashes.pop()

            return memory_id

        except Exception as e:
            logger.error(f"Memory storage error: {e}")
            return None

    async def store_fact(
        self,
        fact: str,
        source: str = "conversation",
        importance: float = 0.7,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a semantic fact.

        Args:
            fact: The fact to store
            source: Source of the fact
            importance: Importance score (0-1)
            metadata: Additional metadata

        Returns:
            Memory ID if stored
        """
        if not self._memory:
            return None

        try:
            fact_metadata = {
                "source": source,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            }

            return await self._store_memory(
                content=fact,
                memory_type="semantic",
                importance=importance,
                metadata=fact_metadata,
            )

        except Exception as e:
            logger.error(f"Fact storage error: {e}")
            return None

    async def store_procedure(
        self,
        name: str,
        steps: list[str],
        description: Optional[str] = None,
        importance: float = 0.8,
    ) -> Optional[str]:
        """
        Store a procedural memory (how to do something).

        Args:
            name: Name of the procedure
            steps: List of steps
            description: Optional description
            importance: Importance score (0-1)

        Returns:
            Memory ID if stored
        """
        if not self._memory:
            return None

        try:
            steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
            content = f"Procedure: {name}\n"
            if description:
                content += f"Description: {description}\n"
            content += f"Steps:\n{steps_text}"

            return await self._store_memory(
                content=content,
                memory_type="procedural",
                importance=importance,
                metadata={
                    "procedure_name": name,
                    "step_count": len(steps),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Procedure storage error: {e}")
            return None

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> list[Any]:
        """
        Retrieve memories from a specific conversation.

        Args:
            conversation_id: The conversation ID
            limit: Maximum memories to return

        Returns:
            List of memories from the conversation
        """
        if not self._memory:
            return []

        try:
            results = await self.retrieve_relevant(
                query=f"conversation:{conversation_id}",
                limit=limit,
            )

            return [
                r for r in results
                if getattr(r, "metadata", {}).get("conversation_id") == conversation_id
            ]

        except Exception as e:
            logger.error(f"Conversation history retrieval error: {e}")
            return []

    async def forget(
        self,
        memory_id: str,
    ) -> bool:
        """
        Forget (delete) a specific memory.

        Args:
            memory_id: The memory ID to forget

        Returns:
            True if forgotten successfully
        """
        if not self._memory:
            return False

        try:
            if hasattr(self._memory, "forget"):
                await self._memory.forget(memory_id)
                return True
            elif hasattr(self._memory, "delete"):
                await self._memory.delete(memory_id)
                return True
        except Exception as e:
            logger.error(f"Memory forget error: {e}")

        return False

    async def _store_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> Optional[str]:
        """Internal method to store a memory."""
        if hasattr(self._memory, "store"):
            try:
                from aion.systems.memory.cognitive import MemoryType as CognitiveMemoryType

                type_map = {
                    "episodic": CognitiveMemoryType.EPISODIC,
                    "semantic": CognitiveMemoryType.SEMANTIC,
                    "procedural": CognitiveMemoryType.PROCEDURAL,
                    "working": CognitiveMemoryType.WORKING,
                }
                mem_type = type_map.get(memory_type, CognitiveMemoryType.EPISODIC)

                return await self._memory.store(
                    content=content,
                    memory_type=mem_type,
                    importance=importance,
                    metadata=metadata,
                )
            except ImportError:
                return await self._memory.store(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    metadata=metadata,
                )

        elif hasattr(self._memory, "add"):
            return await self._memory.add(
                content=content,
                metadata={
                    "type": memory_type,
                    "importance": importance,
                    **metadata,
                },
            )

        return None

    def _calculate_importance(
        self,
        user_message: Message,
        assistant_message: Message,
    ) -> float:
        """
        Calculate importance score for a conversation turn.

        Factors:
        - Message length (longer = more important)
        - Tool usage (indicates task completion)
        - Question marks (indicates information need)
        - Explicit importance markers
        """
        importance = self.default_importance

        user_text = user_message.get_text()
        assistant_text = assistant_message.get_text()

        if len(user_text) > 200:
            importance += 0.1
        if len(assistant_text) > 500:
            importance += 0.1

        if assistant_message.has_tool_use():
            importance += 0.15

        if "?" in user_text:
            importance += 0.05

        important_keywords = [
            "important", "remember", "critical", "key", "essential",
            "don't forget", "make sure", "always", "never",
        ]
        if any(kw in user_text.lower() for kw in important_keywords):
            importance += 0.15

        trivial_patterns = [
            "hello", "hi", "thanks", "thank you", "bye", "goodbye",
            "ok", "okay", "yes", "no", "sure",
        ]
        if user_text.lower().strip() in trivial_patterns:
            importance -= 0.2

        return max(0.0, min(1.0, importance))


class MemoryContextEnricher:
    """
    Enriches conversation context with relevant memories.
    """

    def __init__(self, integrator: MemoryIntegrator):
        self.integrator = integrator

    async def enrich_context(
        self,
        user_message: str,
        conversation_history: list[Message],
        max_memories: int = 5,
    ) -> dict[str, Any]:
        """
        Enrich context with relevant memories.

        Returns a dict with:
        - memories: list of relevant memories
        - context_summary: brief summary of memory context
        """
        memories = await self.integrator.retrieve_relevant(
            query=user_message,
            limit=max_memories,
        )

        context_summary = ""
        if memories:
            context_summary = f"Found {len(memories)} relevant memories from previous interactions."

        return {
            "memories": memories,
            "context_summary": context_summary,
            "memory_count": len(memories),
        }
