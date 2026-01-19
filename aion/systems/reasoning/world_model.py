"""
AION World Model

Predictive model of the world for:
- State representation and tracking
- Future state prediction
- Counterfactual reasoning
- Imagination and planning
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WorldState:
    """A state of the world."""
    id: str
    timestamp: datetime
    entities: dict[str, dict]  # entity_id -> properties
    relations: list[tuple[str, str, str]]  # (subject, predicate, object)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "entities": self.entities,
            "relations": self.relations,
            "context": self.context,
        }

    def get_entity(self, entity_id: str) -> Optional[dict]:
        return self.entities.get(entity_id)

    def get_relations_for(self, entity_id: str) -> list[tuple[str, str, str]]:
        return [r for r in self.relations if r[0] == entity_id or r[2] == entity_id]


@dataclass
class StateTransition:
    """A transition between world states."""
    from_state_id: str
    to_state_id: str
    action: str
    effects: list[str]
    probability: float = 1.0


class WorldModel:
    """
    Predictive world model for reasoning.

    Maintains an internal model of the world that can:
    - Track current state
    - Predict future states given actions
    - Reason about counterfactuals
    - Support planning through imagination
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # State history
        self.states: dict[str, WorldState] = {}
        self.current_state_id: Optional[str] = None
        self.state_history: list[str] = []

        # Learned dynamics
        self.transitions: list[StateTransition] = []
        self.action_effects: dict[str, list[str]] = {}  # action -> typical effects

    async def initialize(self, initial_context: Optional[dict] = None) -> str:
        """Initialize the world model with an initial state."""
        state = WorldState(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            entities={},
            relations=[],
            context=initial_context or {},
        )

        self.states[state.id] = state
        self.current_state_id = state.id
        self.state_history.append(state.id)

        return state.id

    def get_current_state(self) -> Optional[WorldState]:
        """Get the current world state."""
        if self.current_state_id:
            return self.states.get(self.current_state_id)
        return None

    async def update_state(
        self,
        observations: dict[str, Any],
        actions: Optional[list[str]] = None,
    ) -> WorldState:
        """
        Update the world state based on observations.

        Args:
            observations: New observations about the world
            actions: Actions that were taken

        Returns:
            Updated WorldState
        """
        current = self.get_current_state()

        # Extract entities and relations from observations
        entities, relations = await self._extract_structure(observations)

        # Merge with current state
        new_entities = current.entities.copy() if current else {}
        new_entities.update(entities)

        new_relations = list(current.relations) if current else []
        new_relations.extend(relations)

        # Create new state
        new_state = WorldState(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            entities=new_entities,
            relations=new_relations,
            context=observations,
        )

        self.states[new_state.id] = new_state
        self.state_history.append(new_state.id)

        # Record transition
        if current and actions:
            for action in actions:
                self.transitions.append(StateTransition(
                    from_state_id=current.id,
                    to_state_id=new_state.id,
                    action=action,
                    effects=self._compute_effects(current, new_state),
                ))

        self.current_state_id = new_state.id

        return new_state

    async def _extract_structure(
        self,
        observations: dict[str, Any],
    ) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
        """Extract entities and relations from observations."""
        from aion.core.llm import Message

        obs_text = json.dumps(observations, default=str)

        prompt = f"""Extract entities and relationships from these observations.

Observations: {obs_text}

Format as JSON:
{{
    "entities": {{
        "entity_id": {{"type": "...", "properties": {{...}}}}
    }},
    "relations": [
        ["subject_id", "predicate", "object_id"]
    ]
}}
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You extract structured world state from observations."),
                Message(role="user", content=prompt),
            ])

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", {})
                relations = [tuple(r) for r in data.get("relations", [])]
                return entities, relations

        except Exception as e:
            logger.warning("Structure extraction failed", error=str(e))

        return {}, []

    def _compute_effects(
        self,
        from_state: WorldState,
        to_state: WorldState,
    ) -> list[str]:
        """Compute effects of a transition."""
        effects = []

        # New entities
        new_entities = set(to_state.entities.keys()) - set(from_state.entities.keys())
        for e in new_entities:
            effects.append(f"added:{e}")

        # Removed entities
        removed = set(from_state.entities.keys()) - set(to_state.entities.keys())
        for e in removed:
            effects.append(f"removed:{e}")

        # Changed properties
        for entity_id in from_state.entities:
            if entity_id in to_state.entities:
                if from_state.entities[entity_id] != to_state.entities[entity_id]:
                    effects.append(f"changed:{entity_id}")

        return effects

    async def predict(
        self,
        action: str,
        num_steps: int = 1,
    ) -> list[WorldState]:
        """
        Predict future states given an action.

        Args:
            action: Action to take
            num_steps: Number of steps to predict

        Returns:
            List of predicted future states
        """
        from aion.core.llm import Message

        current = self.get_current_state()
        if not current:
            return []

        predictions = []
        state = current

        for step in range(num_steps):
            # Use learned dynamics if available
            learned_effects = self.action_effects.get(action, [])

            prompt = f"""Predict the next world state after taking an action.

Current state:
- Entities: {json.dumps(state.entities, default=str)}
- Relations: {state.relations}

Action: {action}

Known effects of similar actions: {learned_effects}

Predict:
1. What entities change?
2. What new relations form?
3. What is removed?

Provide the predicted next state as JSON.
"""

            try:
                response = await self.llm.complete([
                    Message(role="system", content="You predict future world states."),
                    Message(role="user", content=prompt),
                ])

                # Parse prediction
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())

                    predicted = WorldState(
                        id=f"predicted_{step}_{uuid.uuid4().hex[:8]}",
                        timestamp=datetime.now(),
                        entities=data.get("entities", state.entities),
                        relations=data.get("relations", state.relations),
                        context={"predicted": True, "action": action},
                    )

                    predictions.append(predicted)
                    state = predicted

            except:
                break

        return predictions

    async def counterfactual(
        self,
        state_id: str,
        alternative_action: str,
    ) -> Optional[WorldState]:
        """
        Reason about what would have happened with a different action.

        Args:
            state_id: State from which to branch
            alternative_action: Alternative action to consider

        Returns:
            Counterfactual state
        """
        if state_id not in self.states:
            return None

        base_state = self.states[state_id]

        # Find what actually happened
        actual_transitions = [
            t for t in self.transitions
            if t.from_state_id == state_id
        ]

        from aion.core.llm import Message

        prompt = f"""Reason about a counterfactual scenario.

Actual state: {json.dumps(base_state.entities, default=str)}

What actually happened: {[t.action for t in actual_transitions]}

Alternative action to consider: {alternative_action}

What would the world state be if we had taken the alternative action instead?
Consider:
1. How would entities be different?
2. What relations would exist?
3. What wouldn't have happened?

Provide the counterfactual state as JSON.
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You reason about counterfactual scenarios."),
                Message(role="user", content=prompt),
            ])

            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return WorldState(
                    id=f"counterfactual_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(),
                    entities=data.get("entities", {}),
                    relations=data.get("relations", []),
                    context={"counterfactual": True, "alternative_action": alternative_action},
                )

        except:
            pass

        return None

    async def imagine_scenario(
        self,
        description: str,
    ) -> WorldState:
        """
        Imagine a hypothetical scenario.

        Args:
            description: Description of the scenario

        Returns:
            Imagined WorldState
        """
        from aion.core.llm import Message

        prompt = f"""Imagine a world state based on this description.

Description: {description}

Create a detailed world state with:
1. Entities (objects, people, places)
2. Their properties
3. Relationships between them

Format as JSON with "entities" and "relations".
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You create detailed world representations."),
                Message(role="user", content=prompt),
            ])

            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return WorldState(
                    id=f"imagined_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(),
                    entities=data.get("entities", {}),
                    relations=data.get("relations", []),
                    context={"imagined": True, "description": description},
                )

        except:
            pass

        return WorldState(
            id=f"imagined_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            entities={},
            relations=[],
            context={"imagined": True, "description": description},
        )

    def get_state_history(self, limit: int = 10) -> list[WorldState]:
        """Get recent state history."""
        return [
            self.states[sid]
            for sid in self.state_history[-limit:]
            if sid in self.states
        ]
