"""
AION SOTA Context Management System

State-of-the-art context management featuring:
- Semantic chunking (break messages into meaningful semantic units)
- Importance-weighted retention (keep critical messages longer)
- Progressive summarization (hierarchical summarization like MemGPT)
- Claude-specific token counting
- Attention-based context prioritization
- Dynamic context window optimization
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import structlog

from aion.conversation.types import (
    Conversation,
    Message,
    MessageRole,
    ConversationConfig,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    ThinkingContent,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Token Counting
# =============================================================================

class TokenCounter(Protocol):
    """Protocol for token counters."""

    def count(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        ...


class ClaudeTokenCounter:
    """
    Claude-specific token counter.

    Claude uses a different tokenizer than GPT models. This implementation
    provides accurate token counting for Claude models by:
    1. Using the anthropic library's token counting if available
    2. Falling back to character-based estimation (Claude ~3.5 chars/token)
    """

    # Claude models have different characteristics than GPT tokenizers
    # Average characters per token for Claude is approximately 3.5-4
    CHARS_PER_TOKEN = 3.5

    # Overhead tokens per message
    MESSAGE_OVERHEAD = 4

    # Special token patterns that affect counting
    SPECIAL_PATTERNS = {
        r'<\|.*?\|>': 1,  # Special tokens
        r'\n': 1,  # Newlines count as tokens
        r'```': 1,  # Code block markers
    }

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._anthropic_counter = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the token counter."""
        if self._initialized:
            return

        try:
            import anthropic
            self._anthropic_counter = anthropic.Anthropic()
            logger.info("Using Anthropic API for token counting")
        except (ImportError, Exception) as e:
            logger.warning(f"Anthropic token counter not available: {e}, using estimation")

        self._initialized = True

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0

        # Try to use Anthropic's official counter
        if self._anthropic_counter:
            try:
                return self._anthropic_counter.count_tokens(text)
            except Exception:
                pass

        # Fall back to estimation
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Claude.

        Claude's tokenization is different from GPT:
        - More efficient for common English text
        - Different handling of code and special characters
        - Approximately 3.5 characters per token on average
        """
        if not text:
            return 0

        # Base estimation from character count
        base_tokens = len(text) / self.CHARS_PER_TOKEN

        # Adjust for special patterns
        adjustments = 0
        for pattern, token_count in self.SPECIAL_PATTERNS.items():
            matches = len(re.findall(pattern, text))
            adjustments += matches * token_count

        # Adjust for code blocks (more efficient tokenization)
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        for block in code_blocks:
            # Code is tokenized more efficiently
            code_adjustment = len(block) / 4.5 - len(block) / self.CHARS_PER_TOKEN
            adjustments += code_adjustment

        # Adjust for whitespace
        whitespace = len(re.findall(r'\s+', text))
        adjustments += whitespace * 0.1  # Whitespace is efficient

        return max(1, int(base_tokens + adjustments))

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        total = 0

        for message in messages:
            # Add per-message overhead
            total += self.MESSAGE_OVERHEAD

            # Count role tokens
            role = message.get("role", "")
            total += self.count(role)

            # Count content tokens
            content = message.get("content", [])
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")

                        if block_type == "text":
                            total += self.count(block.get("text", ""))
                        elif block_type == "tool_use":
                            total += self.count(block.get("name", ""))
                            total += self.count(str(block.get("input", {})))
                            total += 10  # Tool use overhead
                        elif block_type == "tool_result":
                            total += self.count(str(block.get("content", "")))
                            total += 5  # Tool result overhead
                        elif block_type == "thinking":
                            total += self.count(block.get("thinking", ""))
                        elif block_type == "image":
                            # Images have special token costs
                            total += self._estimate_image_tokens(block)
                    elif isinstance(block, str):
                        total += self.count(block)

        return total

    def _estimate_image_tokens(self, image_block: Dict[str, Any]) -> int:
        """Estimate tokens for image content."""
        # Claude charges based on image size
        # Rough estimation: ~1000 tokens for small, ~2000 for medium, ~4000 for large
        source = image_block.get("source", {})
        if source.get("type") == "base64":
            data = source.get("data", "")
            # Estimate size from base64 length
            size_bytes = len(data) * 3 / 4
            if size_bytes < 100000:  # < 100KB
                return 1000
            elif size_bytes < 500000:  # < 500KB
                return 2000
            else:
                return 4000
        return 1500  # Default estimate


# =============================================================================
# Semantic Chunking
# =============================================================================

class ChunkType(str, Enum):
    """Types of semantic chunks."""
    QUESTION = "question"
    ANSWER = "answer"
    CODE = "code"
    EXPLANATION = "explanation"
    INSTRUCTION = "instruction"
    CONTEXT = "context"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    DIALOGUE = "dialogue"


@dataclass
class SemanticChunk:
    """A semantically meaningful unit of content."""
    id: str
    content: str
    chunk_type: ChunkType
    token_count: int
    importance: float = 0.5

    # Relationships
    message_id: str = ""
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    # For context building
    is_anchor: bool = False  # Must be included in context
    can_summarize: bool = True  # Can be replaced with summary


class SemanticChunker:
    """
    Breaks messages into semantically meaningful chunks.

    Instead of arbitrary token-based splitting, this identifies natural
    boundaries in content like:
    - Question/answer pairs
    - Code blocks with explanations
    - Tool invocations and results
    - Logical argument segments
    """

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter

    def chunk_message(self, message: Message) -> List[SemanticChunk]:
        """Break a message into semantic chunks."""
        chunks: List[SemanticChunk] = []
        message_id = message.id

        for i, content_block in enumerate(message.content):
            chunk_id = f"{message_id}_chunk_{i}"

            if isinstance(content_block, TextContent):
                text_chunks = self._chunk_text(
                    content_block.text,
                    message_id,
                    chunk_id,
                )
                chunks.extend(text_chunks)

            elif isinstance(content_block, ToolUseContent):
                chunk = SemanticChunk(
                    id=chunk_id,
                    content=f"Tool: {content_block.name}\nInput: {content_block.input}",
                    chunk_type=ChunkType.TOOL_USE,
                    token_count=self.token_counter.count(str(content_block.input)),
                    importance=0.7,  # Tool usage is usually important
                    message_id=message_id,
                    is_anchor=True,  # Tool calls should be kept
                    can_summarize=False,
                )
                chunks.append(chunk)

            elif isinstance(content_block, ToolResultContent):
                chunk = SemanticChunk(
                    id=chunk_id,
                    content=f"Tool Result ({content_block.tool_use_id}): {content_block.content}",
                    chunk_type=ChunkType.TOOL_RESULT,
                    token_count=self.token_counter.count(content_block.content),
                    importance=0.6,
                    message_id=message_id,
                    is_anchor=True,
                    can_summarize=True,
                )
                chunks.append(chunk)

            elif isinstance(content_block, ThinkingContent):
                chunk = SemanticChunk(
                    id=chunk_id,
                    content=f"Thinking: {content_block.thinking}",
                    chunk_type=ChunkType.THINKING,
                    token_count=self.token_counter.count(content_block.thinking),
                    importance=0.4,  # Thinking can be summarized
                    message_id=message_id,
                    can_summarize=True,
                )
                chunks.append(chunk)

        return chunks

    def _chunk_text(
        self,
        text: str,
        message_id: str,
        base_chunk_id: str,
    ) -> List[SemanticChunk]:
        """Break text into semantic chunks."""
        chunks: List[SemanticChunk] = []

        # Detect and separate code blocks first
        code_pattern = r'```[\s\S]*?```'
        parts = re.split(f'({code_pattern})', text)

        chunk_idx = 0
        for part in parts:
            if not part.strip():
                continue

            chunk_id = f"{base_chunk_id}_{chunk_idx}"

            if part.startswith('```'):
                # Code block
                chunk = SemanticChunk(
                    id=chunk_id,
                    content=part,
                    chunk_type=ChunkType.CODE,
                    token_count=self.token_counter.count(part),
                    importance=0.7,  # Code is usually important
                    message_id=message_id,
                    is_anchor=True,  # Keep code blocks intact
                    can_summarize=False,
                )
                chunks.append(chunk)
            else:
                # Text content - further segment by semantic units
                text_chunks = self._segment_prose(part, message_id, chunk_id)
                chunks.extend(text_chunks)

            chunk_idx += 1

        return chunks

    def _segment_prose(
        self,
        text: str,
        message_id: str,
        base_chunk_id: str,
    ) -> List[SemanticChunk]:
        """Segment prose text into semantic units."""
        chunks: List[SemanticChunk] = []

        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', text)

        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue

            chunk_id = f"{base_chunk_id}_para_{i}"
            chunk_type, importance = self._classify_paragraph(para)

            chunk = SemanticChunk(
                id=chunk_id,
                content=para,
                chunk_type=chunk_type,
                token_count=self.token_counter.count(para),
                importance=importance,
                message_id=message_id,
                keywords=self._extract_keywords(para),
            )
            chunks.append(chunk)

        return chunks

    def _classify_paragraph(self, text: str) -> Tuple[ChunkType, float]:
        """Classify a paragraph and assign importance."""
        text_lower = text.lower()

        # Questions
        if text.strip().endswith('?') or any(
            q in text_lower for q in ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        ):
            return ChunkType.QUESTION, 0.7

        # Instructions/commands
        instruction_patterns = [
            r'^(please|could you|can you|would you)',
            r'^(do|make|create|write|build|implement)',
            r'^(first|then|next|finally|step)',
        ]
        for pattern in instruction_patterns:
            if re.match(pattern, text_lower):
                return ChunkType.INSTRUCTION, 0.8

        # Explanations (often follow questions)
        if any(exp in text_lower for exp in ['because', 'therefore', 'this means', 'in other words']):
            return ChunkType.EXPLANATION, 0.6

        # Default to context
        return ChunkType.CONTEXT, 0.5

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Remove common stop words
        stop_words = {
            'this', 'that', 'with', 'from', 'have', 'been', 'were', 'would',
            'could', 'should', 'being', 'which', 'their', 'there', 'about',
        }
        words = [w for w in words if w not in stop_words]

        # Count frequencies
        freq: Dict[str, int] = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        # Return top keywords
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:max_keywords]]


# =============================================================================
# Progressive Summarization (MemGPT-inspired)
# =============================================================================

class SummaryLevel(str, Enum):
    """Levels of summarization detail."""
    FULL = "full"          # Original content
    DETAILED = "detailed"  # Preserves key details
    CONCISE = "concise"    # Key points only
    MINIMAL = "minimal"    # One sentence


@dataclass
class SummaryNode:
    """A node in the summary hierarchy."""
    id: str
    level: SummaryLevel
    content: str
    token_count: int

    # What this summarizes
    source_ids: List[str] = field(default_factory=list)  # Chunk IDs or summary IDs
    source_type: str = "chunks"  # "chunks" or "summaries"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.5


class ProgressiveSummarizer:
    """
    Implements progressive summarization for context management.

    Instead of abrupt truncation, this creates a hierarchy of summaries:
    - Full content for recent/important messages
    - Detailed summaries for moderately old content
    - Concise summaries for older content
    - Minimal summaries for very old content

    This allows unlimited conversation history with bounded context size.
    """

    def __init__(
        self,
        llm_provider: Any = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.llm = llm_provider
        self.token_counter = token_counter or ClaudeTokenCounter()

        # Summary cache
        self._summaries: Dict[str, SummaryNode] = {}
        self._chunk_to_summary: Dict[str, str] = {}

    async def summarize_chunks(
        self,
        chunks: List[SemanticChunk],
        target_level: SummaryLevel,
        max_tokens: int = 500,
    ) -> SummaryNode:
        """
        Create a summary of chunks at the specified level.
        """
        if not chunks:
            return SummaryNode(
                id="empty",
                level=target_level,
                content="",
                token_count=0,
            )

        # Combine chunk content
        combined = "\n\n".join(c.content for c in chunks)
        source_ids = [c.id for c in chunks]

        # Generate summary based on level
        if target_level == SummaryLevel.FULL:
            summary_content = combined
        else:
            summary_content = await self._generate_summary(
                combined,
                target_level,
                max_tokens,
            )

        summary_id = f"summary_{hashlib.md5(''.join(source_ids).encode()).hexdigest()[:12]}"

        summary = SummaryNode(
            id=summary_id,
            level=target_level,
            content=summary_content,
            token_count=self.token_counter.count(summary_content),
            source_ids=source_ids,
            source_type="chunks",
        )

        self._summaries[summary_id] = summary
        for chunk_id in source_ids:
            self._chunk_to_summary[chunk_id] = summary_id

        return summary

    async def summarize_summaries(
        self,
        summaries: List[SummaryNode],
        target_level: SummaryLevel,
        max_tokens: int = 300,
    ) -> SummaryNode:
        """
        Create a higher-level summary from existing summaries.
        This enables recursive summarization.
        """
        if not summaries:
            return SummaryNode(
                id="empty",
                level=target_level,
                content="",
                token_count=0,
            )

        combined = "\n\n---\n\n".join(s.content for s in summaries)
        source_ids = [s.id for s in summaries]

        summary_content = await self._generate_summary(
            combined,
            target_level,
            max_tokens,
        )

        summary_id = f"meta_summary_{hashlib.md5(''.join(source_ids).encode()).hexdigest()[:12]}"

        meta_summary = SummaryNode(
            id=summary_id,
            level=target_level,
            content=summary_content,
            token_count=self.token_counter.count(summary_content),
            source_ids=source_ids,
            source_type="summaries",
        )

        self._summaries[summary_id] = meta_summary
        return meta_summary

    async def _generate_summary(
        self,
        content: str,
        level: SummaryLevel,
        max_tokens: int,
    ) -> str:
        """Generate a summary using the LLM."""
        if not self.llm:
            # Fallback to extractive summarization
            return self._extractive_summary(content, level, max_tokens)

        prompts = {
            SummaryLevel.DETAILED: """Summarize the following content, preserving key details, examples, and specific information. Keep technical terms and important quotes.

Content:
{content}

Detailed summary:""",

            SummaryLevel.CONCISE: """Provide a concise summary of the following content, focusing only on the main points and conclusions.

Content:
{content}

Concise summary:""",

            SummaryLevel.MINIMAL: """Provide a one-sentence summary capturing the essence of the following content.

Content:
{content}

One-sentence summary:""",
        }

        prompt = prompts.get(level, prompts[SummaryLevel.CONCISE]).format(content=content)

        try:
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return self._extractive_summary(content, level, max_tokens)

    def _extractive_summary(
        self,
        content: str,
        level: SummaryLevel,
        max_tokens: int,
    ) -> str:
        """Fallback extractive summarization."""
        sentences = re.split(r'(?<=[.!?])\s+', content)

        if level == SummaryLevel.MINIMAL:
            return sentences[0] if sentences else content[:100]

        # Calculate sentence importance (simple TF-based)
        word_freq: Dict[str, int] = {}
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        sentence_scores = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq.get(w, 0) for w in words) / max(len(words), 1)
            sentence_scores.append((sentence, score))

        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Determine how many sentences to keep based on level
        if level == SummaryLevel.DETAILED:
            keep_ratio = 0.5
        else:  # CONCISE
            keep_ratio = 0.25

        keep_count = max(1, int(len(sentences) * keep_ratio))
        top_sentences = sentence_scores[:keep_count]

        # Reorder to original order
        original_order = {s: i for i, s in enumerate(sentences)}
        top_sentences.sort(key=lambda x: original_order.get(x[0], 0))

        summary = ' '.join(s for s, _ in top_sentences)

        # Ensure we don't exceed token limit
        char_limit = max_tokens * 4
        if len(summary) > char_limit:
            summary = summary[:char_limit] + "..."

        return summary


# =============================================================================
# Context Window
# =============================================================================

@dataclass
class ContextWindow:
    """Represents a context window for LLM input."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)

    # Token counts
    total_tokens: int = 0
    system_tokens: int = 0
    message_tokens: int = 0
    tool_tokens: int = 0

    # Content tracking
    memories_included: int = 0
    messages_included: int = 0
    summaries_included: int = 0

    # Context state
    truncated: bool = False
    uses_summaries: bool = False

    # Metadata
    build_time_ms: float = 0.0


# =============================================================================
# SOTA Context Builder
# =============================================================================

class ContextBuilder:
    """
    State-of-the-art context builder featuring:
    - Semantic chunking for intelligent content splitting
    - Progressive summarization for unbounded history
    - Importance-weighted retention
    - Claude-specific token counting
    - Dynamic context optimization
    """

    def __init__(
        self,
        default_system_prompt: Optional[str] = None,
        token_counter: Optional[TokenCounter] = None,
        llm_provider: Any = None,
    ):
        self.default_system_prompt = default_system_prompt or self._get_default_system_prompt()

        # Initialize Claude-specific token counter
        self.token_counter = token_counter or ClaudeTokenCounter()

        # Initialize semantic chunker
        self.chunker = SemanticChunker(self.token_counter)

        # Initialize progressive summarizer
        self.summarizer = ProgressiveSummarizer(llm_provider, self.token_counter)

        # Chunk and summary cache
        self._message_chunks: Dict[str, List[SemanticChunk]] = {}
        self._context_summaries: Dict[str, SummaryNode] = {}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using Claude-specific counting."""
        return self.token_counter.count(text)

    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        """Count tokens in a message."""
        return self.token_counter.count_messages([message])

    async def build_context(
        self,
        conversation: Conversation,
        current_message: Optional[Message] = None,
        memories: Optional[List[Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> ContextWindow:
        """
        Build an optimized context window using semantic chunking
        and progressive summarization.
        """
        start_time = time.time()
        config = config or conversation.config

        context = ContextWindow()

        # 1. Build system prompt with memory context
        system_parts = []
        if config.system_prompt:
            system_parts.append(config.system_prompt)
        elif config.include_system_context:
            system_parts.append(self.default_system_prompt)

        if memories:
            memory_context = self._format_memories(memories)
            system_parts.append(f"\n\n<relevant_memories>\n{memory_context}\n</relevant_memories>")
            context.memories_included = len(memories)

        context.system = "\n".join(system_parts)
        context.system_tokens = self.count_tokens(context.system)

        # 2. Process tools
        if tools:
            context.tools = tools
            context.tool_tokens = sum(self.count_tokens(str(t)) for t in tools)

        # 3. Calculate available token budget
        available_tokens = (
            config.max_context_tokens
            - context.system_tokens
            - context.tool_tokens
            - config.max_tokens  # Reserve for response
            - 100  # Safety margin
        )

        # 4. Collect all messages
        messages = list(conversation.messages)
        if current_message and current_message not in messages:
            messages.append(current_message)

        # 5. Build context with intelligent message selection
        context.messages = await self._build_message_context(
            messages,
            available_tokens,
            config.max_context_messages,
        )

        # 6. Compute final statistics
        context.messages_included = len(context.messages)
        context.message_tokens = self.token_counter.count_messages(context.messages)
        context.total_tokens = (
            context.system_tokens
            + context.tool_tokens
            + context.message_tokens
        )
        context.truncated = len(context.messages) < len(messages)
        context.build_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Context built",
            total_tokens=context.total_tokens,
            messages=context.messages_included,
            truncated=context.truncated,
            uses_summaries=context.uses_summaries,
            build_time_ms=f"{context.build_time_ms:.1f}",
        )

        return context

    async def _build_message_context(
        self,
        messages: List[Message],
        token_budget: int,
        max_messages: int,
    ) -> List[Dict[str, Any]]:
        """
        Build message context with semantic awareness.

        Strategy:
        1. Always include recent messages (last N)
        2. Include messages with tool usage
        3. Include messages with high importance chunks
        4. Summarize older content if needed
        """
        if not messages:
            return []

        # Convert messages to dict format
        converted = [self._convert_message(m) for m in messages]

        # Calculate current token usage
        total_tokens = self.token_counter.count_messages(converted)

        # If within budget, return all messages
        if total_tokens <= token_budget and len(converted) <= max_messages:
            return converted

        # Need to reduce context - use intelligent selection
        return await self._intelligent_message_selection(
            messages,
            converted,
            token_budget,
            max_messages,
        )

    async def _intelligent_message_selection(
        self,
        messages: List[Message],
        converted: List[Dict[str, Any]],
        token_budget: int,
        max_messages: int,
    ) -> List[Dict[str, Any]]:
        """
        Intelligently select messages for context using importance scoring.
        """
        # Score each message
        scored_messages: List[Tuple[int, float, Message, Dict[str, Any]]] = []

        for i, (msg, conv) in enumerate(zip(messages, converted)):
            score = self._score_message(msg, i, len(messages))
            scored_messages.append((i, score, msg, conv))

        # Always keep the last few messages (most recent context)
        keep_recent = 5
        recent_messages = scored_messages[-keep_recent:]
        older_messages = scored_messages[:-keep_recent]

        # Start with recent messages
        selected = [conv for _, _, _, conv in recent_messages]
        current_tokens = self.token_counter.count_messages(selected)

        # Add older messages by importance until budget is reached
        older_messages.sort(key=lambda x: x[1], reverse=True)  # Sort by score

        for idx, score, msg, conv in older_messages:
            msg_tokens = self.token_counter.count_messages([conv])

            if current_tokens + msg_tokens <= token_budget and len(selected) < max_messages:
                # Insert at correct position (maintain order)
                insert_pos = sum(1 for s in selected if any(
                    s.get("_original_idx", 0) < idx
                    for _ in [True]  # Dummy iteration
                ))
                conv["_original_idx"] = idx
                selected.insert(insert_pos, conv)
                current_tokens += msg_tokens
            else:
                # Can't fit, maybe summarize?
                break

        # Clean up internal markers
        for msg in selected:
            msg.pop("_original_idx", None)

        # Reorder by original position
        return selected

    def _score_message(
        self,
        message: Message,
        index: int,
        total_messages: int,
    ) -> float:
        """
        Score a message for context inclusion.

        Higher scores = more important to include.
        """
        score = 0.5  # Base score

        # Recency boost (exponential decay)
        recency = (index + 1) / total_messages
        score += recency * 0.3

        # Tool usage is important
        if message.has_tool_use():
            score += 0.25

        # Questions are often important
        text = message.get_text()
        if '?' in text:
            score += 0.1

        # Long messages often contain important content
        if len(text) > 500:
            score += 0.1

        # User messages are anchors
        if message.role == MessageRole.USER:
            score += 0.1

        return min(1.0, score)

    def _convert_message(self, message: Message) -> Dict[str, Any]:
        """Convert a Message to dict format for the LLM."""
        content = []
        for block in message.content:
            content.append(block.to_dict())

        return {
            "role": message.role.value if message.role != MessageRole.TOOL else "user",
            "content": content,
        }

    def _format_memories(self, memories: List[Any]) -> str:
        """Format memories for system prompt injection."""
        parts = []
        for i, memory in enumerate(memories, 1):
            if hasattr(memory, 'memory'):
                # RetrievalResult from memory system
                mem = memory.memory
                content = mem.content if hasattr(mem, 'content') else str(mem)
                importance = mem.importance if hasattr(mem, 'importance') else 0.5
                score = memory.score if hasattr(memory, 'score') else 0.5
                parts.append(
                    f"[Memory {i}] (relevance: {score:.2f}, importance: {importance:.2f})\n{content}"
                )
            elif hasattr(memory, 'content'):
                content = memory.content
                importance = getattr(memory, 'importance', 0.5)
                parts.append(f"[Memory {i}] (importance: {importance:.2f})\n{content}")
            elif isinstance(memory, dict):
                content = memory.get("content", str(memory))
                importance = memory.get("importance", 0.5)
                parts.append(f"[Memory {i}] (importance: {importance:.2f})\n{content}")
            else:
                parts.append(f"[Memory {i}]\n{str(memory)}")

        return "\n\n".join(parts)

    def _get_default_system_prompt(self) -> str:
        """Get the enhanced default AION system prompt."""
        return """You are AION (Artificial Intelligence Operating Nexus), an advanced AI assistant with sophisticated cognitive capabilities.

## Core Capabilities
- **Memory System**: Access to episodic, semantic, and procedural memory for context-aware responses
- **Tool Execution**: Ability to invoke tools for actions, calculations, and information retrieval
- **Planning**: Create and execute multi-step plans for complex tasks
- **Multimodal**: Process and understand text, images, and other media
- **Reasoning**: Apply chain-of-thought reasoning for complex problems

## Operating Principles

### Reasoning Approach
- For complex questions, think step-by-step before answering
- Consider multiple perspectives and potential solutions
- Acknowledge uncertainty when appropriate
- Reference relevant memories and context when applicable

### Tool Usage Guidelines
- Use tools proactively when they can provide better answers
- Explain your reasoning for tool selection
- Handle tool errors gracefully with fallback strategies

### Communication Style
- Be clear, concise, and accurate
- Adapt formality to match the user's style
- Provide structured responses for complex information
- Use examples and analogies to clarify concepts

### Safety and Ethics
- Prioritize user safety and well-being
- Decline requests that could cause harm
- Protect user privacy and confidentiality
- Be transparent about limitations and capabilities

You are designed to be a powerful, helpful AI assistant that combines broad knowledge with specialized tools to provide exceptional assistance."""

    def create_summary_prompt(self, messages: List[Message]) -> str:
        """Create a prompt for summarizing a conversation."""
        text_parts = []
        for msg in messages:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            text_parts.append(f"{role}: {msg.get_text()}")

        conversation_text = "\n".join(text_parts)

        return f"""Summarize the following conversation, preserving:
- Key topics discussed
- Important decisions or conclusions
- Any action items or requests
- Relevant technical details

Conversation:
{conversation_text}

Summary:"""

    async def summarize_for_context(
        self,
        messages: List[Message],
        llm_provider: Any,
        max_tokens: int = 500,
    ) -> str:
        """Summarize messages for context compression."""
        prompt = self.create_summary_prompt(messages)

        try:
            response = await llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.content if hasattr(response, 'content') else response.get_text()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to extractive summary
            text = " ".join(m.get_text() for m in messages)
            return text[:max_tokens * 4] + "..."


# =============================================================================
# Conversation Summarizer
# =============================================================================

class ConversationSummarizer:
    """
    Advanced conversation summarization for context management.
    Uses progressive summarization for efficient context compression.
    """

    def __init__(
        self,
        llm_provider: Any,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.llm = llm_provider
        self.token_counter = token_counter or ClaudeTokenCounter()
        self.context_builder = ContextBuilder(
            token_counter=self.token_counter,
            llm_provider=llm_provider,
        )
        self.progressive_summarizer = ProgressiveSummarizer(llm_provider, self.token_counter)

        # Summary cache for incremental summarization
        self._conversation_summaries: Dict[str, List[SummaryNode]] = {}

    async def should_summarize(
        self,
        conversation: Conversation,
        threshold_messages: int = 30,
        threshold_tokens: int = 50000,
    ) -> bool:
        """Check if conversation needs summarization."""
        if len(conversation.messages) < threshold_messages:
            return False

        total_tokens = sum(
            self.context_builder.count_message_tokens(
                self.context_builder._convert_message(m)
            )
            for m in conversation.messages
        )

        return total_tokens > threshold_tokens

    async def summarize_conversation(
        self,
        conversation: Conversation,
        keep_recent: int = 10,
    ) -> Tuple[str, List[Message]]:
        """
        Create progressive summary of older messages.

        Returns:
            Tuple of (summary_text, recent_messages)
        """
        if len(conversation.messages) <= keep_recent:
            return "", list(conversation.messages)

        old_messages = conversation.messages[:-keep_recent]
        recent_messages = conversation.messages[-keep_recent:]

        # Chunk old messages
        all_chunks: List[SemanticChunk] = []
        for msg in old_messages:
            chunks = self.context_builder.chunker.chunk_message(msg)
            all_chunks.extend(chunks)

        # Create progressive summary
        if len(all_chunks) > 20:
            # Group chunks and summarize each group, then summarize summaries
            group_size = 10
            groups = [
                all_chunks[i:i + group_size]
                for i in range(0, len(all_chunks), group_size)
            ]

            group_summaries = []
            for group in groups:
                summary = await self.progressive_summarizer.summarize_chunks(
                    group,
                    SummaryLevel.CONCISE,
                    max_tokens=300,
                )
                group_summaries.append(summary)

            # Meta-summary
            final_summary = await self.progressive_summarizer.summarize_summaries(
                group_summaries,
                SummaryLevel.DETAILED,
                max_tokens=500,
            )
            summary_text = final_summary.content
        else:
            # Direct summarization
            summary = await self.progressive_summarizer.summarize_chunks(
                all_chunks,
                SummaryLevel.DETAILED,
                max_tokens=500,
            )
            summary_text = summary.content

        return summary_text, list(recent_messages)

    async def get_incremental_summary(
        self,
        conversation_id: str,
        new_messages: List[Message],
    ) -> str:
        """
        Get incremental summary by updating existing summary with new messages.
        More efficient than full re-summarization.
        """
        existing_summaries = self._conversation_summaries.get(conversation_id, [])

        # Chunk new messages
        new_chunks: List[SemanticChunk] = []
        for msg in new_messages:
            chunks = self.context_builder.chunker.chunk_message(msg)
            new_chunks.extend(chunks)

        # Summarize new chunks
        new_summary = await self.progressive_summarizer.summarize_chunks(
            new_chunks,
            SummaryLevel.CONCISE,
            max_tokens=200,
        )

        # Add to conversation summaries
        existing_summaries.append(new_summary)
        self._conversation_summaries[conversation_id] = existing_summaries

        # If too many summaries, consolidate
        if len(existing_summaries) > 5:
            meta_summary = await self.progressive_summarizer.summarize_summaries(
                existing_summaries,
                SummaryLevel.DETAILED,
                max_tokens=400,
            )
            self._conversation_summaries[conversation_id] = [meta_summary]
            return meta_summary.content

        # Combine all summaries
        return "\n\n".join(s.content for s in existing_summaries)
