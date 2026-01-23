"""
AION Advanced Prompt Engineering

State-of-the-art prompt engineering with:
- Chain-of-Thought (CoT) prompting
- ReAct (Reasoning + Acting) patterns
- Dynamic prompt optimization
- Self-consistency prompting
- Constitutional AI patterns
"""

from aion.conversation.prompts.engineering import (
    PromptTemplate,
    PromptBuilder,
    ChainOfThoughtPrompt,
    ReActPrompt,
    ConstitutionalPrompt,
    DynamicPromptOptimizer,
)

__all__ = [
    "PromptTemplate",
    "PromptBuilder",
    "ChainOfThoughtPrompt",
    "ReActPrompt",
    "ConstitutionalPrompt",
    "DynamicPromptOptimizer",
]
