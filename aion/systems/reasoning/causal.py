"""
AION Causal Reasoning

Causal inference and reasoning with:
- Causal graph construction
- Do-calculus operations
- Counterfactual reasoning
- Causal effect estimation
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CausalNode:
    """A node in a causal graph."""
    id: str
    name: str
    description: str
    observed: bool = True
    values: list[Any] = field(default_factory=list)


@dataclass
class CausalEdge:
    """A directed edge representing causation."""
    source: str
    target: str
    mechanism: str  # Description of causal mechanism
    strength: float = 1.0  # Causal strength


@dataclass
class CausalGraph:
    """A causal graph representing causal relationships."""
    id: str
    nodes: dict[str, CausalNode] = field(default_factory=dict)
    edges: list[CausalEdge] = field(default_factory=list)

    def add_node(self, node: CausalNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: CausalEdge) -> None:
        self.edges.append(edge)

    def get_parents(self, node_id: str) -> list[str]:
        """Get direct causal parents."""
        return [e.source for e in self.edges if e.target == node_id]

    def get_children(self, node_id: str) -> list[str]:
        """Get direct causal children."""
        return [e.target for e in self.edges if e.source == node_id]

    def get_ancestors(self, node_id: str) -> set[str]:
        """Get all causal ancestors."""
        ancestors = set()
        to_visit = self.get_parents(node_id)

        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(self.get_parents(parent))

        return ancestors

    def get_descendants(self, node_id: str) -> set[str]:
        """Get all causal descendants."""
        descendants = set()
        to_visit = self.get_children(node_id)

        while to_visit:
            child = to_visit.pop()
            if child not in descendants:
                descendants.add(child)
                to_visit.extend(self.get_children(child))

        return descendants

    def is_d_separated(
        self,
        x: str,
        y: str,
        z: set[str],
    ) -> bool:
        """
        Check if X and Y are d-separated given Z.

        Simplified implementation - full d-separation is complex.
        """
        # If there's a direct path not through Z, not d-separated
        if y in self.get_children(x) and y not in z:
            return False
        if x in self.get_children(y) and x not in z:
            return False

        # Check for common ancestors (confounders)
        x_ancestors = self.get_ancestors(x)
        y_ancestors = self.get_ancestors(y)
        common = x_ancestors & y_ancestors

        # If common ancestors are not all in Z, not d-separated
        if common - z:
            return False

        return True


class CausalReasoner:
    """
    Causal reasoning engine.

    Implements:
    - Causal graph construction from observations
    - Intervention reasoning (do-calculus)
    - Counterfactual queries
    - Causal effect estimation
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Causal graphs
        self.graphs: dict[str, CausalGraph] = {}

        # Learned causal patterns
        self.causal_patterns: list[dict] = []

    async def build_graph(
        self,
        variables: list[str],
        observations: list[dict],
        domain_knowledge: Optional[str] = None,
    ) -> CausalGraph:
        """
        Build a causal graph from observations.

        Args:
            variables: Variables to include
            observations: Observational data
            domain_knowledge: Optional domain knowledge

        Returns:
            CausalGraph
        """
        from aion.core.llm import Message

        obs_sample = observations[:10] if len(observations) > 10 else observations

        prompt = f"""Build a causal graph for these variables based on observations.

Variables: {variables}

Sample observations:
{json.dumps(obs_sample, default=str)}

Domain knowledge: {domain_knowledge or 'None provided'}

Identify causal relationships. For each relationship:
1. What causes what?
2. What is the mechanism?
3. How strong is the relationship (0-1)?

Format as:
EDGE: source -> target
MECHANISM: <how source causes target>
STRENGTH: <0-1>
---
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You are a causal reasoning expert."),
                Message(role="user", content=prompt),
            ])

            graph = CausalGraph(id=str(uuid.uuid4()))

            # Add nodes
            for var in variables:
                graph.add_node(CausalNode(
                    id=var,
                    name=var,
                    description=f"Variable: {var}",
                ))

            # Parse edges
            import re
            for block in response.content.split('---'):
                edge_match = re.search(r'EDGE:\s*(\w+)\s*->\s*(\w+)', block)
                mech_match = re.search(r'MECHANISM:\s*(.+)', block)
                strength_match = re.search(r'STRENGTH:\s*([\d.]+)', block)

                if edge_match:
                    graph.add_edge(CausalEdge(
                        source=edge_match.group(1),
                        target=edge_match.group(2),
                        mechanism=mech_match.group(1).strip() if mech_match else "",
                        strength=float(strength_match.group(1)) if strength_match else 0.5,
                    ))

            self.graphs[graph.id] = graph
            return graph

        except Exception as e:
            logger.warning("Causal graph construction failed", error=str(e))
            return CausalGraph(id=str(uuid.uuid4()))

    async def do_intervention(
        self,
        graph_id: str,
        intervention: dict[str, Any],
        query_variable: str,
    ) -> dict[str, Any]:
        """
        Compute the effect of an intervention (do-operator).

        Args:
            graph_id: Causal graph to use
            intervention: Variables to intervene on {var: value}
            query_variable: Variable to query

        Returns:
            Estimated effect of intervention
        """
        if graph_id not in self.graphs:
            return {"error": "Graph not found"}

        graph = self.graphs[graph_id]

        from aion.core.llm import Message

        # Build graph description
        edges_desc = "\n".join([
            f"  {e.source} -> {e.target}: {e.mechanism}"
            for e in graph.edges
        ])

        prompt = f"""Reason about a causal intervention using do-calculus.

Causal Graph:
{edges_desc}

Intervention: do({json.dumps(intervention)})

Query: What is the effect on {query_variable}?

Steps:
1. Identify which paths are blocked by the intervention
2. Identify which paths remain active
3. Trace the causal effect through active paths
4. Estimate the effect

Provide:
EFFECT: <description of effect>
DIRECTION: increase/decrease/no_change
MAGNITUDE: small/medium/large
CONFIDENCE: 0-1
REASONING: <step by step reasoning>
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You perform causal reasoning with do-calculus."),
                Message(role="user", content=prompt),
            ])

            # Parse response
            content = response.content

            direction = "unknown"
            if "increase" in content.lower():
                direction = "increase"
            elif "decrease" in content.lower():
                direction = "decrease"
            elif "no_change" in content.lower() or "no change" in content.lower():
                direction = "no_change"

            import re
            conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)
            confidence = float(conf_match.group(1)) if conf_match else 0.5

            return {
                "intervention": intervention,
                "query_variable": query_variable,
                "effect_direction": direction,
                "confidence": confidence,
                "reasoning": content,
            }

        except Exception as e:
            logger.warning("Intervention reasoning failed", error=str(e))
            return {"error": str(e)}

    async def counterfactual_query(
        self,
        graph_id: str,
        factual: dict[str, Any],
        counterfactual_condition: dict[str, Any],
        query_variable: str,
    ) -> dict[str, Any]:
        """
        Answer a counterfactual query.

        "Given that X=x happened, what would Y have been if X had been x'?"

        Args:
            graph_id: Causal graph to use
            factual: What actually happened
            counterfactual_condition: What we're imagining changed
            query_variable: What we want to know

        Returns:
            Counterfactual answer
        """
        if graph_id not in self.graphs:
            return {"error": "Graph not found"}

        graph = self.graphs[graph_id]

        from aion.core.llm import Message

        edges_desc = "\n".join([
            f"  {e.source} -> {e.target}: {e.mechanism}"
            for e in graph.edges
        ])

        prompt = f"""Answer a counterfactual question.

Causal Graph:
{edges_desc}

Factual situation (what happened): {json.dumps(factual)}

Counterfactual condition (what if): {json.dumps(counterfactual_condition)}

Query: What would {query_variable} have been?

Use the three-step counterfactual procedure:
1. Abduction: Infer background factors from factual observations
2. Action: Modify the model according to counterfactual condition
3. Prediction: Compute the query variable in the modified model

COUNTERFACTUAL_VALUE: <value or description>
REASONING: <step by step>
CONFIDENCE: 0-1
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You perform counterfactual reasoning."),
                Message(role="user", content=prompt),
            ])

            import re
            content = response.content

            value_match = re.search(r'COUNTERFACTUAL_VALUE:\s*(.+?)(?=\n|$)', content)
            conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)

            return {
                "factual": factual,
                "counterfactual_condition": counterfactual_condition,
                "query_variable": query_variable,
                "counterfactual_value": value_match.group(1).strip() if value_match else "unknown",
                "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                "reasoning": content,
            }

        except Exception as e:
            logger.warning("Counterfactual query failed", error=str(e))
            return {"error": str(e)}

    async def estimate_causal_effect(
        self,
        graph_id: str,
        treatment: str,
        outcome: str,
        observations: list[dict],
    ) -> dict[str, Any]:
        """
        Estimate the causal effect of treatment on outcome.

        Args:
            graph_id: Causal graph to use
            treatment: Treatment variable
            outcome: Outcome variable
            observations: Observational data

        Returns:
            Estimated causal effect
        """
        if graph_id not in self.graphs:
            return {"error": "Graph not found"}

        graph = self.graphs[graph_id]

        # Identify confounders (common causes)
        treatment_ancestors = graph.get_ancestors(treatment)
        outcome_ancestors = graph.get_ancestors(outcome)
        confounders = treatment_ancestors & outcome_ancestors

        from aion.core.llm import Message

        prompt = f"""Estimate the causal effect of {treatment} on {outcome}.

Identified confounders: {list(confounders)}

Sample observations: {json.dumps(observations[:10], default=str)}

Steps:
1. Identify the causal path from {treatment} to {outcome}
2. Account for confounding through {list(confounders)}
3. Estimate the effect size

Provide:
EFFECT_SIZE: <numeric or qualitative>
EFFECT_TYPE: positive/negative/none
CONFOUNDING_BIAS: present/absent/uncertain
ADJUSTMENT_VARIABLES: <variables to control for>
CONFIDENCE: 0-1
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You estimate causal effects."),
                Message(role="user", content=prompt),
            ])

            import re
            content = response.content

            effect_match = re.search(r'EFFECT_SIZE:\s*(.+?)(?=\n|$)', content)
            type_match = re.search(r'EFFECT_TYPE:\s*(\w+)', content)
            conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)

            return {
                "treatment": treatment,
                "outcome": outcome,
                "confounders": list(confounders),
                "effect_size": effect_match.group(1).strip() if effect_match else "unknown",
                "effect_type": type_match.group(1) if type_match else "unknown",
                "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                "reasoning": content,
            }

        except Exception as e:
            logger.warning("Causal effect estimation failed", error=str(e))
            return {"error": str(e)}

    async def explain_causally(
        self,
        observation: dict[str, Any],
        target_variable: str,
    ) -> str:
        """
        Provide a causal explanation for an observation.

        Args:
            observation: The observation to explain
            target_variable: The variable to explain

        Returns:
            Causal explanation
        """
        from aion.core.llm import Message

        prompt = f"""Provide a causal explanation for this observation.

Observation: {json.dumps(observation, default=str)}

Explain causally: Why did {target_variable} have the value it did?

Identify:
1. Direct causes
2. Indirect causes (causal chains)
3. Background conditions
4. Alternative possibilities

Provide a clear causal story.
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You provide causal explanations."),
                Message(role="user", content=prompt),
            ])

            return response.content

        except:
            return "Unable to generate causal explanation."
