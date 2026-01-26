"""
AION Workflow Exporter

Converts visual workflow graphs to executable workflow definitions.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from aion.automation.types import (
    Workflow,
    WorkflowStep,
    TriggerConfig,
    ActionConfig,
    Condition,
    TriggerType,
    ActionType,
    ConditionOperator,
)

logger = structlog.get_logger(__name__)


class WorkflowExporter:
    """
    Exports visual workflow graphs to various formats.

    Supports:
    - YAML export
    - JSON export
    - Direct Workflow object creation
    """

    def __init__(self):
        # Mapping from visual node types to trigger types
        self._trigger_type_map = {
            "trigger_schedule": TriggerType.SCHEDULE,
            "trigger_webhook": TriggerType.WEBHOOK,
            "trigger_event": TriggerType.EVENT,
            "trigger_manual": TriggerType.MANUAL,
        }

        # Mapping from visual node types to action types
        self._action_type_map = {
            "action_tool": ActionType.TOOL,
            "action_agent": ActionType.AGENT,
            "action_webhook": ActionType.WEBHOOK,
            "action_notification": ActionType.NOTIFICATION,
            "action_llm": ActionType.LLM,
            "action_delay": ActionType.DELAY,
            "action_transform": ActionType.TRANSFORM,
        }

    def export_to_workflow(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        name: str,
        description: str = "",
    ) -> Workflow:
        """
        Convert visual graph to Workflow object.

        Args:
            nodes: List of visual nodes
            edges: List of visual edges
            name: Workflow name
            description: Workflow description

        Returns:
            Workflow object
        """
        workflow_id = str(uuid.uuid4())

        # Build adjacency list for traversal
        adjacency = self._build_adjacency(edges)
        reverse_adjacency = self._build_reverse_adjacency(edges)

        # Identify trigger nodes
        triggers = []
        trigger_nodes = [n for n in nodes if n["data"]["type"].startswith("trigger")]
        for trigger_node in trigger_nodes:
            trigger_config = self._convert_trigger(trigger_node)
            if trigger_config:
                triggers.append(trigger_config)

        # Build steps from non-trigger nodes
        steps = []
        processed = set()
        node_to_step_id = {}

        # Start from trigger targets and traverse
        for trigger_node in trigger_nodes:
            targets = adjacency.get(trigger_node["id"], [])
            for target in targets:
                self._traverse_and_convert(
                    target,
                    nodes,
                    edges,
                    adjacency,
                    reverse_adjacency,
                    steps,
                    processed,
                    node_to_step_id,
                )

        # Set step dependencies based on edges
        for step in steps:
            step_node_id = next(
                (nid for nid, sid in node_to_step_id.items() if sid == step.id),
                None
            )
            if step_node_id:
                predecessors = reverse_adjacency.get(step_node_id, [])
                step.depends_on = [
                    node_to_step_id[p]
                    for p in predecessors
                    if p in node_to_step_id
                ]

        return Workflow(
            id=workflow_id,
            name=name,
            description=description,
            triggers=triggers,
            steps=steps,
        )

    def _build_adjacency(self, edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adjacency: Dict[str, List[str]] = {}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(target)
        return adjacency

    def _build_reverse_adjacency(self, edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build reverse adjacency list from edges."""
        reverse: Dict[str, List[str]] = {}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if target not in reverse:
                reverse[target] = []
            reverse[target].append(source)
        return reverse

    def _find_node(self, node_id: str, nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find node by ID."""
        for node in nodes:
            if node["id"] == node_id:
                return node
        return None

    def _traverse_and_convert(
        self,
        node_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        adjacency: Dict[str, List[str]],
        reverse_adjacency: Dict[str, List[str]],
        steps: List[WorkflowStep],
        processed: set,
        node_to_step_id: Dict[str, str],
    ) -> None:
        """Traverse graph and convert nodes to steps."""
        if node_id in processed:
            return

        node = self._find_node(node_id, nodes)
        if not node:
            return

        processed.add(node_id)

        # Convert node to step
        step = self._convert_node_to_step(node, edges, adjacency)
        if step:
            steps.append(step)
            node_to_step_id[node_id] = step.id

        # Continue traversal
        targets = adjacency.get(node_id, [])
        for target in targets:
            self._traverse_and_convert(
                target,
                nodes,
                edges,
                adjacency,
                reverse_adjacency,
                steps,
                processed,
                node_to_step_id,
            )

    def _convert_trigger(self, node: Dict[str, Any]) -> Optional[TriggerConfig]:
        """Convert trigger node to TriggerConfig."""
        node_type = node["data"]["type"]
        config = node["data"].get("config", {})

        trigger_type = self._trigger_type_map.get(node_type)
        if not trigger_type:
            return None

        trigger_config = TriggerConfig(
            type=trigger_type,
            config=config,
        )

        # Add specific fields based on trigger type
        if trigger_type == TriggerType.SCHEDULE:
            trigger_config.config["cron_expression"] = config.get("cron_expression", "0 * * * *")
        elif trigger_type == TriggerType.WEBHOOK:
            trigger_config.config["path"] = config.get("path", f"/webhook/{node['id']}")
        elif trigger_type == TriggerType.EVENT:
            trigger_config.config["event_type"] = config.get("event_type", "custom.event")

        return trigger_config

    def _convert_node_to_step(
        self,
        node: Dict[str, Any],
        edges: List[Dict[str, Any]],
        adjacency: Dict[str, List[str]],
    ) -> Optional[WorkflowStep]:
        """Convert a visual node to a WorkflowStep."""
        node_type = node["data"]["type"]
        node_label = node["data"]["label"]
        config = node["data"].get("config", {})

        # Skip triggers and end nodes
        if node_type.startswith("trigger") or node_type == "end":
            return None

        step_id = f"step_{node['id']}"

        # Handle condition nodes
        if node_type == "condition":
            return WorkflowStep(
                id=step_id,
                name=node_label,
                action=ActionConfig(
                    type=ActionType.SCRIPT,
                    config={"expression": config.get("expression", "true")},
                ),
                condition=Condition(
                    field="result",
                    operator=ConditionOperator.EQUALS,
                    value=True,
                ),
            )

        # Handle approval nodes
        if node_type == "approval":
            return WorkflowStep(
                id=step_id,
                name=node_label,
                action=ActionConfig(
                    type=ActionType.TOOL,
                    config={
                        "tool_name": "approval_gate",
                        "params": {
                            "approvers": config.get("approvers", []),
                            "gate_type": config.get("gate_type", "single"),
                            "message": config.get("message", "Approval required"),
                        },
                    },
                ),
                requires_approval=True,
                approval_config={
                    "type": config.get("gate_type", "single"),
                    "approvers": config.get("approvers", []),
                    "timeout_hours": config.get("timeout_hours", 24),
                },
            )

        # Handle parallel nodes
        if node_type == "parallel":
            targets = adjacency.get(node["id"], [])
            return WorkflowStep(
                id=step_id,
                name=node_label,
                action=ActionConfig(
                    type=ActionType.WORKFLOW,
                    config={
                        "mode": "parallel",
                        "branches": targets,
                    },
                ),
                parallel=True,
            )

        # Handle loop nodes
        if node_type == "loop":
            return WorkflowStep(
                id=step_id,
                name=node_label,
                action=ActionConfig(
                    type=ActionType.SCRIPT,
                    config={
                        "expression": config.get("collection", "[]"),
                    },
                ),
                loop=config.get("collection"),
                loop_variable=config.get("item_variable", "item"),
            )

        # Handle subworkflow nodes
        if node_type == "subworkflow":
            return WorkflowStep(
                id=step_id,
                name=node_label,
                action=ActionConfig(
                    type=ActionType.WORKFLOW,
                    config={
                        "workflow_id": config.get("workflow_id"),
                        "inputs": config.get("inputs", {}),
                        "wait": config.get("wait_for_completion", True),
                    },
                ),
            )

        # Handle action nodes
        action_type = self._action_type_map.get(node_type)
        if action_type:
            return WorkflowStep(
                id=step_id,
                name=node_label,
                action=ActionConfig(
                    type=action_type,
                    config=config,
                ),
            )

        # Unknown node type
        logger.warning(f"Unknown node type: {node_type}")
        return None

    def export_to_yaml(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        name: str,
        description: str = "",
    ) -> str:
        """Export visual workflow to YAML."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML export")

        workflow = self.export_to_workflow(nodes, edges, name, description)
        workflow_dict = workflow.to_dict()

        return yaml.dump(workflow_dict, default_flow_style=False, sort_keys=False)

    def export_to_json(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        name: str,
        description: str = "",
    ) -> str:
        """Export visual workflow to JSON."""
        workflow = self.export_to_workflow(nodes, edges, name, description)
        workflow_dict = workflow.to_dict()

        return json.dumps(workflow_dict, indent=2)


def export_to_yaml(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    name: str,
    description: str = "",
) -> str:
    """Convenience function for YAML export."""
    exporter = WorkflowExporter()
    return exporter.export_to_yaml(nodes, edges, name, description)


def export_to_json(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    name: str,
    description: str = "",
) -> str:
    """Convenience function for JSON export."""
    exporter = WorkflowExporter()
    return exporter.export_to_json(nodes, edges, name, description)


def import_from_yaml(yaml_content: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Import workflow from YAML and convert to visual nodes/edges.

    Returns:
        Tuple of (nodes, edges)
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for YAML import")

    workflow_dict = yaml.safe_load(yaml_content)
    return _workflow_dict_to_visual(workflow_dict)


def import_from_json(json_content: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Import workflow from JSON and convert to visual nodes/edges.

    Returns:
        Tuple of (nodes, edges)
    """
    workflow_dict = json.loads(json_content)
    return _workflow_dict_to_visual(workflow_dict)


def _workflow_dict_to_visual(workflow_dict: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Convert workflow dictionary to visual nodes and edges."""
    nodes = []
    edges = []

    x_offset = 100
    y_offset = 100
    y_spacing = 150
    x_spacing = 250

    # Convert triggers to nodes
    for i, trigger in enumerate(workflow_dict.get("triggers", [])):
        trigger_type = trigger.get("type", "manual")
        node_type = f"trigger_{trigger_type}"

        nodes.append({
            "id": f"trigger_{i}",
            "type": "default",
            "position": {"x": x_offset, "y": y_offset + i * y_spacing},
            "data": {
                "label": f"{trigger_type.title()} Trigger",
                "type": node_type,
                "config": trigger.get("config", {}),
            },
        })

    # Convert steps to nodes
    step_positions = {}
    for i, step in enumerate(workflow_dict.get("steps", [])):
        step_id = step.get("id", f"step_{i}")

        # Determine node type from action
        action = step.get("action", {})
        action_type = action.get("type", "tool")
        node_type = f"action_{action_type}"

        if step.get("requires_approval"):
            node_type = "approval"
        elif step.get("parallel"):
            node_type = "parallel"
        elif step.get("loop"):
            node_type = "loop"

        pos_x = x_offset + x_spacing
        pos_y = y_offset + i * y_spacing
        step_positions[step_id] = {"x": pos_x, "y": pos_y}

        nodes.append({
            "id": step_id,
            "type": "default",
            "position": {"x": pos_x, "y": pos_y},
            "data": {
                "label": step.get("name", step_id),
                "type": node_type,
                "config": action.get("config", {}),
            },
        })

    # Create edges from triggers to first steps
    first_steps = [s for s in workflow_dict.get("steps", []) if not s.get("depends_on")]
    for i, _ in enumerate(workflow_dict.get("triggers", [])):
        for step in first_steps:
            edges.append({
                "id": f"edge_trigger_{i}_to_{step['id']}",
                "source": f"trigger_{i}",
                "target": step["id"],
            })

    # Create edges from step dependencies
    for step in workflow_dict.get("steps", []):
        step_id = step.get("id")
        for dep in step.get("depends_on", []):
            edges.append({
                "id": f"edge_{dep}_to_{step_id}",
                "source": dep,
                "target": step_id,
            })

    return nodes, edges
