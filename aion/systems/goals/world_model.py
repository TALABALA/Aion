"""
AION Goal System - World Model

SOTA world model for simulating outcomes before acting.
Enables lookahead planning and counterfactual reasoning.

Key capabilities:
- State representation and transition modeling
- Outcome prediction and simulation
- Counterfactual reasoning ("what if" analysis)
- Monte Carlo Tree Search for planning
- Model-based reinforcement learning
- Causal inference for understanding effects
"""

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum
import numpy as np
from copy import deepcopy

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
    GoalMetrics,
)

logger = structlog.get_logger()


class StateType(Enum):
    """Types of state components."""
    RESOURCE = "resource"
    GOAL = "goal"
    CONTEXT = "context"
    CONSTRAINT = "constraint"


@dataclass
class StateVariable:
    """A single state variable in the world model."""

    name: str
    value: Any
    var_type: StateType
    bounds: Optional[Tuple[float, float]] = None
    is_observable: bool = True
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.var_type.value,
            "bounds": self.bounds,
            "is_observable": self.is_observable,
        }


@dataclass
class WorldState:
    """Complete state of the world at a point in time."""

    variables: Dict[str, StateVariable] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    parent_state_id: Optional[str] = None
    action_taken: Optional[str] = None

    def __post_init__(self):
        self.id = f"state_{self.timestamp.timestamp()}_{random.randint(0, 9999)}"

    def get(self, name: str, default: Any = None) -> Any:
        """Get variable value."""
        if name in self.variables:
            return self.variables[name].value
        return default

    def set(self, name: str, value: Any, var_type: StateType = StateType.CONTEXT):
        """Set variable value."""
        if name in self.variables:
            self.variables[name].value = value
            self.variables[name].last_updated = datetime.now()
        else:
            self.variables[name] = StateVariable(
                name=name,
                value=value,
                var_type=var_type,
            )

    def copy(self) -> "WorldState":
        """Create a deep copy of the state."""
        new_state = WorldState(
            variables={k: deepcopy(v) for k, v in self.variables.items()},
            timestamp=datetime.now(),
            parent_state_id=self.id,
        )
        return new_state

    def to_vector(self, feature_names: List[str]) -> np.ndarray:
        """Convert state to numerical vector."""
        vector = []
        for name in feature_names:
            var = self.variables.get(name)
            if var is None:
                vector.append(0.0)
            elif isinstance(var.value, (int, float)):
                vector.append(float(var.value))
            elif isinstance(var.value, bool):
                vector.append(1.0 if var.value else 0.0)
            else:
                vector.append(hash(str(var.value)) % 1000 / 1000)
        return np.array(vector)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "timestamp": self.timestamp.isoformat(),
            "parent_state_id": self.parent_state_id,
            "action_taken": self.action_taken,
        }


@dataclass
class Action:
    """An action that can be taken in the world."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    cost: float = 1.0
    duration: float = 1.0  # seconds

    def is_applicable(self, state: WorldState) -> bool:
        """Check if action can be taken in given state."""
        for precond in self.preconditions:
            # Simple precondition checking
            if ">" in precond:
                var, val = precond.split(">")
                if state.get(var.strip(), 0) <= float(val.strip()):
                    return False
            elif "<" in precond:
                var, val = precond.split("<")
                if state.get(var.strip(), 0) >= float(val.strip()):
                    return False
            elif "=" in precond:
                var, val = precond.split("=")
                if str(state.get(var.strip())) != val.strip():
                    return False
        return True


@dataclass
class Transition:
    """A state transition from action execution."""

    from_state: WorldState
    action: Action
    to_state: WorldState
    reward: float = 0.0
    probability: float = 1.0
    actual: bool = False  # Whether this was observed vs simulated


class TransitionModel:
    """
    Learned model of state transitions.

    Predicts next state given current state and action.
    Uses a neural network approximation.
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 16,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Neural network weights (simplified)
        self._init_weights()

        # Experience buffer
        self._transitions: List[Transition] = []
        self._trained = False

    def _init_weights(self):
        """Initialize network weights."""
        # Input: state + action -> hidden -> next_state
        input_dim = self.state_dim + self.action_dim
        scale = 0.1

        self.W1 = np.random.randn(input_dim, self.hidden_dim) * scale
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.b2 = np.zeros(self.hidden_dim)
        self.W3 = np.random.randn(self.hidden_dim, self.state_dim) * scale
        self.b3 = np.zeros(self.state_dim)

    def _encode_action(self, action: Action) -> np.ndarray:
        """Encode action as vector."""
        vec = np.zeros(self.action_dim)
        # Hash-based encoding
        vec[hash(action.name) % self.action_dim] = 1.0
        for i, (k, v) in enumerate(action.parameters.items()):
            if i < self.action_dim:
                vec[i] = hash(str(v)) % 100 / 100
        return vec

    def _forward(self, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        """Forward pass through transition model."""
        x = np.concatenate([state_vec, action_vec])

        # Layer 1
        h1 = np.tanh(np.dot(x, self.W1) + self.b1)

        # Layer 2
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)

        # Output (residual prediction)
        delta = np.dot(h2, self.W3) + self.b3

        # Next state is current + delta
        next_state = state_vec + delta

        return next_state

    def predict(
        self,
        state: WorldState,
        action: Action,
        feature_names: List[str],
    ) -> WorldState:
        """Predict next state given current state and action."""
        state_vec = state.to_vector(feature_names)

        # Pad or truncate to expected dimension
        if len(state_vec) < self.state_dim:
            state_vec = np.pad(state_vec, (0, self.state_dim - len(state_vec)))
        else:
            state_vec = state_vec[:self.state_dim]

        action_vec = self._encode_action(action)
        next_state_vec = self._forward(state_vec, action_vec)

        # Convert back to WorldState
        next_state = state.copy()
        next_state.action_taken = action.name

        for i, name in enumerate(feature_names[:self.state_dim]):
            if name in next_state.variables:
                next_state.variables[name].value = float(next_state_vec[i])

        return next_state

    def add_transition(self, transition: Transition):
        """Add observed transition for learning."""
        self._transitions.append(transition)

    def train(self, learning_rate: float = 0.001, epochs: int = 100):
        """Train the transition model on observed transitions."""
        if len(self._transitions) < 10:
            return

        for epoch in range(epochs):
            total_loss = 0.0

            for trans in self._transitions[-1000:]:  # Recent transitions
                feature_names = list(trans.from_state.variables.keys())

                state_vec = trans.from_state.to_vector(feature_names)
                if len(state_vec) < self.state_dim:
                    state_vec = np.pad(state_vec, (0, self.state_dim - len(state_vec)))
                else:
                    state_vec = state_vec[:self.state_dim]

                target_vec = trans.to_state.to_vector(feature_names)
                if len(target_vec) < self.state_dim:
                    target_vec = np.pad(target_vec, (0, self.state_dim - len(target_vec)))
                else:
                    target_vec = target_vec[:self.state_dim]

                action_vec = self._encode_action(trans.action)
                pred_vec = self._forward(state_vec, action_vec)

                # MSE loss
                loss = np.mean((pred_vec - target_vec) ** 2)
                total_loss += loss

                # Simplified gradient update (just last layer)
                error = pred_vec - target_vec
                self.W3 -= learning_rate * np.outer(
                    np.tanh(np.dot(np.tanh(np.dot(
                        np.concatenate([state_vec, action_vec]), self.W1
                    ) + self.b1), self.W2) + self.b2),
                    error
                )

            if epoch % 20 == 0:
                logger.debug("transition_model_training", epoch=epoch, loss=total_loss)

        self._trained = True


class RewardModel:
    """
    Model of rewards/values for different outcomes.

    Predicts the value of reaching different states.
    """

    def __init__(self):
        self._reward_history: List[Tuple[WorldState, float]] = []
        self._goal_rewards: Dict[str, float] = {}

    def estimate_reward(self, state: WorldState, goal: Optional[Goal] = None) -> float:
        """Estimate reward for reaching a state."""
        reward = 0.0

        # Resource rewards
        for name, var in state.variables.items():
            if var.var_type == StateType.RESOURCE:
                if isinstance(var.value, (int, float)):
                    reward += var.value * 0.1

        # Goal completion rewards
        if goal:
            progress = state.get(f"goal_{goal.id}_progress", 0.0)
            reward += progress * 10.0

            if state.get(f"goal_{goal.id}_complete", False):
                reward += 100.0

        # Time cost penalty
        elapsed = state.get("elapsed_time", 0.0)
        reward -= elapsed * 0.01

        return reward

    def add_observation(self, state: WorldState, actual_reward: float):
        """Record observed reward."""
        self._reward_history.append((state, actual_reward))

    def set_goal_reward(self, goal_id: str, reward: float):
        """Set explicit reward for goal completion."""
        self._goal_rewards[goal_id] = reward


class MCTSNode:
    """Node in Monte Carlo Tree Search."""

    def __init__(
        self,
        state: WorldState,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Action] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state

        self.children: Dict[str, "MCTSNode"] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Action] = []

    def ucb_score(self, exploration: float = 1.414) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        ) if self.parent else 0

        return exploitation + exploration_term

    def best_child(self, exploration: float = 1.414) -> "MCTSNode":
        """Select best child using UCB."""
        return max(
            self.children.values(),
            key=lambda c: c.ucb_score(exploration)
        )

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.state.get("is_terminal", False)


class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search for goal planning.

    Uses simulation to find optimal action sequences.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        max_depth: int = 10,
        n_simulations: int = 100,
        exploration: float = 1.414,
    ):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.max_depth = max_depth
        self.n_simulations = n_simulations
        self.exploration = exploration

        self._available_actions: List[Action] = []
        self._feature_names: List[str] = []

    def set_available_actions(self, actions: List[Action]):
        """Set the actions available for planning."""
        self._available_actions = actions

    def set_feature_names(self, names: List[str]):
        """Set feature names for state encoding."""
        self._feature_names = names

    def search(
        self,
        initial_state: WorldState,
        goal: Optional[Goal] = None,
    ) -> Tuple[Action, float]:
        """
        Run MCTS to find best action.

        Returns best action and its expected value.
        """
        root = MCTSNode(state=initial_state)
        root.untried_actions = [
            a for a in self._available_actions
            if a.is_applicable(initial_state)
        ]

        for _ in range(self.n_simulations):
            # Selection
            node = self._select(root)

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)

            # Simulation
            reward = self._simulate(node.state, goal)

            # Backpropagation
            self._backpropagate(node, reward)

        # Return best action
        if not root.children:
            if root.untried_actions:
                return root.untried_actions[0], 0.0
            return self._available_actions[0] if self._available_actions else None, 0.0

        best_child = max(
            root.children.values(),
            key=lambda c: c.visits
        )
        return best_child.action, best_child.value / max(best_child.visits, 1)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using UCB."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            if not node.children:
                return node
            node = node.best_child(self.exploration)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action."""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()

        # Predict next state
        next_state = self.transition_model.predict(
            node.state, action, self._feature_names
        )

        # Create child node
        child = MCTSNode(state=next_state, parent=node, action=action)
        child.untried_actions = [
            a for a in self._available_actions
            if a.is_applicable(next_state)
        ]

        node.children[action.name] = child
        return child

    def _simulate(self, state: WorldState, goal: Optional[Goal]) -> float:
        """Simulate random playout from state."""
        current_state = state.copy()
        total_reward = 0.0
        discount = 0.99

        for depth in range(self.max_depth):
            # Check terminal
            if current_state.get("is_terminal", False):
                break

            # Get applicable actions
            applicable = [
                a for a in self._available_actions
                if a.is_applicable(current_state)
            ]

            if not applicable:
                break

            # Random action (could be improved with policy)
            action = random.choice(applicable)

            # Transition
            next_state = self.transition_model.predict(
                current_state, action, self._feature_names
            )

            # Reward
            reward = self.reward_model.estimate_reward(next_state, goal)
            total_reward += (discount ** depth) * reward

            current_state = next_state

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class CausalModel:
    """
    Causal model for understanding effects.

    Enables counterfactual reasoning and intervention analysis.
    """

    def __init__(self):
        # Causal graph: var -> causes
        self._causes: Dict[str, Set[str]] = defaultdict(set)
        # Effect strengths
        self._effect_strengths: Dict[Tuple[str, str], float] = {}
        # Observed correlations
        self._correlations: Dict[Tuple[str, str], float] = {}

    def add_causal_link(self, cause: str, effect: str, strength: float = 1.0):
        """Add a causal relationship."""
        self._causes[effect].add(cause)
        self._effect_strengths[(cause, effect)] = strength

    def get_causes(self, variable: str) -> Set[str]:
        """Get direct causes of a variable."""
        return self._causes.get(variable, set())

    def get_effects(self, variable: str) -> Set[str]:
        """Get direct effects of a variable."""
        effects = set()
        for effect, causes in self._causes.items():
            if variable in causes:
                effects.add(effect)
        return effects

    def intervene(
        self,
        state: WorldState,
        variable: str,
        value: Any,
    ) -> WorldState:
        """
        Perform do-calculus intervention.

        Sets variable to value, breaking incoming causal links.
        """
        new_state = state.copy()
        new_state.set(variable, value)

        # Propagate effects
        self._propagate_effects(new_state, variable)

        return new_state

    def _propagate_effects(self, state: WorldState, changed_var: str):
        """Propagate effects of a change through causal graph."""
        effects = self.get_effects(changed_var)

        for effect in effects:
            # Compute new value based on causes
            causes = self.get_causes(effect)
            new_value = 0.0

            for cause in causes:
                cause_value = state.get(cause, 0.0)
                if isinstance(cause_value, (int, float)):
                    strength = self._effect_strengths.get((cause, effect), 1.0)
                    new_value += cause_value * strength

            state.set(effect, new_value)

            # Recurse
            self._propagate_effects(state, effect)

    def counterfactual(
        self,
        factual_state: WorldState,
        intervention_var: str,
        intervention_value: Any,
        query_var: str,
    ) -> Tuple[Any, Any]:
        """
        Answer counterfactual query.

        Returns (factual_value, counterfactual_value).
        """
        factual_value = factual_state.get(query_var)

        # Create counterfactual world
        cf_state = self.intervene(factual_state, intervention_var, intervention_value)
        cf_value = cf_state.get(query_var)

        return factual_value, cf_value

    def learn_structure(self, observations: List[WorldState]):
        """Learn causal structure from observations."""
        if len(observations) < 10:
            return

        # Get all variable names
        var_names = set()
        for obs in observations:
            var_names.update(obs.variables.keys())

        var_names = list(var_names)

        # Compute correlations
        for i, var1 in enumerate(var_names):
            for var2 in var_names[i + 1:]:
                values1 = [obs.get(var1, 0) for obs in observations]
                values2 = [obs.get(var2, 0) for obs in observations]

                # Convert to numeric
                try:
                    v1 = np.array([float(v) if isinstance(v, (int, float)) else 0 for v in values1])
                    v2 = np.array([float(v) if isinstance(v, (int, float)) else 0 for v in values2])

                    if np.std(v1) > 0 and np.std(v2) > 0:
                        corr = np.corrcoef(v1, v2)[0, 1]
                        self._correlations[(var1, var2)] = corr

                        # Simple heuristic: strong correlation might indicate causation
                        if abs(corr) > 0.5:
                            # Add bidirectional for now (would need more sophisticated methods)
                            self.add_causal_link(var1, var2, corr)
                except:
                    pass


class WorldModel:
    """
    Complete world model for simulation and planning.

    Integrates transition model, reward model, MCTS, and causal reasoning.
    """

    def __init__(
        self,
        state_dim: int = 64,
        max_planning_depth: int = 10,
        n_simulations: int = 100,
    ):
        self.transition_model = TransitionModel(state_dim=state_dim)
        self.reward_model = RewardModel()
        self.causal_model = CausalModel()

        self.mcts = MonteCarloTreeSearch(
            transition_model=self.transition_model,
            reward_model=self.reward_model,
            max_depth=max_planning_depth,
            n_simulations=n_simulations,
        )

        self._current_state: Optional[WorldState] = None
        self._state_history: List[WorldState] = []
        self._available_actions: List[Action] = []

        self._initialized = False

    async def initialize(self):
        """Initialize the world model."""
        self._current_state = WorldState()
        self._initialized = True
        logger.info("world_model_initialized")

    async def shutdown(self):
        """Shutdown the world model."""
        self._initialized = False
        logger.info("world_model_shutdown")

    def set_current_state(self, state: WorldState):
        """Set the current world state."""
        self._current_state = state
        self._state_history.append(state)

    def get_current_state(self) -> WorldState:
        """Get the current world state."""
        return self._current_state

    def register_action(self, action: Action):
        """Register an available action."""
        self._available_actions.append(action)
        self.mcts.set_available_actions(self._available_actions)

    def simulate_action(
        self,
        action: Action,
        state: Optional[WorldState] = None,
    ) -> Tuple[WorldState, float]:
        """
        Simulate an action and return predicted next state and reward.
        """
        state = state or self._current_state
        feature_names = list(state.variables.keys())
        self.mcts.set_feature_names(feature_names)

        next_state = self.transition_model.predict(state, action, feature_names)
        reward = self.reward_model.estimate_reward(next_state)

        return next_state, reward

    def simulate_trajectory(
        self,
        actions: List[Action],
        state: Optional[WorldState] = None,
    ) -> List[Tuple[WorldState, float]]:
        """Simulate a sequence of actions."""
        state = state or self._current_state
        trajectory = []

        current = state.copy()
        for action in actions:
            next_state, reward = self.simulate_action(action, current)
            trajectory.append((next_state, reward))
            current = next_state

        return trajectory

    def plan_action(
        self,
        goal: Optional[Goal] = None,
        state: Optional[WorldState] = None,
    ) -> Tuple[Action, float]:
        """
        Use MCTS to plan the best action.
        """
        state = state or self._current_state
        feature_names = list(state.variables.keys())
        self.mcts.set_feature_names(feature_names)

        return self.mcts.search(state, goal)

    def plan_sequence(
        self,
        goal: Optional[Goal] = None,
        horizon: int = 5,
    ) -> List[Tuple[Action, float]]:
        """Plan a sequence of actions."""
        state = self._current_state.copy()
        plan = []

        for _ in range(horizon):
            action, value = self.plan_action(goal, state)
            if action is None:
                break

            plan.append((action, value))

            # Advance state
            state, _ = self.simulate_action(action, state)

        return plan

    def counterfactual_analysis(
        self,
        intervention: Dict[str, Any],
        query_vars: List[str],
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Analyze counterfactuals.

        What would query_vars be if we changed intervention vars?
        """
        results = {}

        for query_var in query_vars:
            for int_var, int_val in intervention.items():
                factual, counterfactual = self.causal_model.counterfactual(
                    self._current_state,
                    int_var,
                    int_val,
                    query_var,
                )
                results[f"{query_var}_given_{int_var}={int_val}"] = (factual, counterfactual)

        return results

    def record_transition(
        self,
        from_state: WorldState,
        action: Action,
        to_state: WorldState,
        reward: float,
    ):
        """Record an observed transition for learning."""
        transition = Transition(
            from_state=from_state,
            action=action,
            to_state=to_state,
            reward=reward,
            actual=True,
        )
        self.transition_model.add_transition(transition)
        self.reward_model.add_observation(to_state, reward)

    def train_models(self):
        """Train transition and reward models."""
        self.transition_model.train()
        self.causal_model.learn_structure(self._state_history)

    def get_state_prediction_uncertainty(
        self,
        action: Action,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Estimate uncertainty in state predictions.

        Uses Monte Carlo sampling with noise.
        """
        state = self._current_state
        feature_names = list(state.variables.keys())

        predictions = []
        for _ in range(n_samples):
            # Add noise to model (simplified MC dropout)
            noise_scale = 0.1
            original_W3 = self.transition_model.W3.copy()
            self.transition_model.W3 += np.random.randn(*self.transition_model.W3.shape) * noise_scale

            next_state = self.transition_model.predict(state, action, feature_names)
            predictions.append(next_state.to_vector(feature_names))

            # Restore weights
            self.transition_model.W3 = original_W3

        predictions = np.array(predictions)
        uncertainties = {}

        for i, name in enumerate(feature_names[:predictions.shape[1]]):
            uncertainties[name] = float(np.std(predictions[:, i]))

        return uncertainties

    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return {
            "state_history_size": len(self._state_history),
            "available_actions": len(self._available_actions),
            "transition_model_trained": self.transition_model._trained,
            "transitions_recorded": len(self.transition_model._transitions),
            "causal_links": sum(len(v) for v in self.causal_model._causes.values()),
            "current_state_vars": len(self._current_state.variables) if self._current_state else 0,
        }
