"""
AION Visual Workflow Builder API

Backend API for the React-based workflow editor.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import structlog

try:
    from fastapi import FastAPI, HTTPException, Query, Body
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = structlog.get_logger(__name__)


# Request/Response Models

class NodePosition(BaseModel):
    """Position of a node in the canvas."""
    x: float
    y: float


class NodeData(BaseModel):
    """Data associated with a node."""
    label: str
    type: str  # trigger, action, condition, approval, subworkflow
    config: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class Node(BaseModel):
    """A node in the workflow graph."""
    id: str
    type: str
    position: NodePosition
    data: NodeData


class Edge(BaseModel):
    """An edge connecting two nodes."""
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    label: Optional[str] = None
    type: str = "default"
    data: Optional[Dict[str, Any]] = None


class WorkflowGraph(BaseModel):
    """Complete workflow graph from the visual editor."""
    id: Optional[str] = None
    name: str
    description: str = ""
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowGraphResponse(BaseModel):
    """Response containing workflow graph."""
    id: str
    name: str
    description: str
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class NodeTypeDefinition(BaseModel):
    """Definition of a node type for the palette."""
    type: str
    label: str
    category: str
    icon: Optional[str] = None
    description: Optional[str] = None
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    inputs: int = 1
    outputs: int = 1
    color: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of workflow validation."""
    valid: bool
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)


class ConversionResult(BaseModel):
    """Result of converting visual workflow to executable."""
    workflow_id: str
    success: bool
    errors: List[str] = Field(default_factory=list)


# In-memory storage for visual workflows
_visual_workflows: Dict[str, Dict[str, Any]] = {}


def setup_visual_routes(app: "FastAPI") -> None:
    """Setup routes for the visual workflow builder."""
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, visual routes not configured")
        return

    # Node type definitions for the palette
    NODE_TYPES: List[NodeTypeDefinition] = [
        # Triggers
        NodeTypeDefinition(
            type="trigger_schedule",
            label="Schedule Trigger",
            category="triggers",
            icon="clock",
            description="Trigger workflow on a schedule (cron)",
            config_schema={
                "type": "object",
                "properties": {
                    "cron_expression": {"type": "string", "title": "Cron Expression"},
                    "timezone": {"type": "string", "title": "Timezone", "default": "UTC"},
                },
                "required": ["cron_expression"],
            },
            inputs=0,
            outputs=1,
            color="#4CAF50",
        ),
        NodeTypeDefinition(
            type="trigger_webhook",
            label="Webhook Trigger",
            category="triggers",
            icon="globe",
            description="Trigger workflow via HTTP webhook",
            config_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "title": "Webhook Path"},
                    "method": {"type": "string", "enum": ["POST", "GET", "PUT"], "default": "POST"},
                    "secret": {"type": "string", "title": "Secret (optional)"},
                },
                "required": ["path"],
            },
            inputs=0,
            outputs=1,
            color="#2196F3",
        ),
        NodeTypeDefinition(
            type="trigger_event",
            label="Event Trigger",
            category="triggers",
            icon="bell",
            description="Trigger workflow on internal event",
            config_schema={
                "type": "object",
                "properties": {
                    "event_type": {"type": "string", "title": "Event Type"},
                    "filter": {"type": "object", "title": "Event Filter"},
                },
                "required": ["event_type"],
            },
            inputs=0,
            outputs=1,
            color="#9C27B0",
        ),
        NodeTypeDefinition(
            type="trigger_manual",
            label="Manual Trigger",
            category="triggers",
            icon="hand",
            description="Trigger workflow manually",
            inputs=0,
            outputs=1,
            color="#FF9800",
        ),

        # Actions
        NodeTypeDefinition(
            type="action_tool",
            label="Execute Tool",
            category="actions",
            icon="wrench",
            description="Execute an AION tool",
            config_schema={
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "title": "Tool Name"},
                    "params": {"type": "object", "title": "Parameters"},
                    "timeout": {"type": "integer", "title": "Timeout (seconds)"},
                },
                "required": ["tool_name"],
            },
            color="#E91E63",
        ),
        NodeTypeDefinition(
            type="action_agent",
            label="Spawn Agent",
            category="actions",
            icon="robot",
            description="Spawn an AION agent",
            config_schema={
                "type": "object",
                "properties": {
                    "agent_class": {"type": "string", "title": "Agent Class"},
                    "system_prompt": {"type": "string", "title": "System Prompt"},
                    "initial_goal": {"type": "string", "title": "Initial Goal"},
                },
                "required": ["agent_class"],
            },
            color="#3F51B5",
        ),
        NodeTypeDefinition(
            type="action_webhook",
            label="HTTP Request",
            category="actions",
            icon="send",
            description="Make an HTTP request",
            config_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "title": "URL"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "POST"},
                    "headers": {"type": "object", "title": "Headers"},
                    "body": {"type": "object", "title": "Body"},
                },
                "required": ["url"],
            },
            color="#009688",
        ),
        NodeTypeDefinition(
            type="action_notification",
            label="Send Notification",
            category="actions",
            icon="message",
            description="Send a notification",
            config_schema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["email", "slack", "webhook"], "title": "Channel"},
                    "recipient": {"type": "string", "title": "Recipient"},
                    "message": {"type": "string", "title": "Message"},
                },
                "required": ["channel", "message"],
            },
            color="#FF5722",
        ),
        NodeTypeDefinition(
            type="action_llm",
            label="LLM Completion",
            category="actions",
            icon="brain",
            description="Get LLM completion",
            config_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "title": "Prompt"},
                    "model": {"type": "string", "title": "Model"},
                    "max_tokens": {"type": "integer", "title": "Max Tokens"},
                },
                "required": ["prompt"],
            },
            color="#673AB7",
        ),
        NodeTypeDefinition(
            type="action_delay",
            label="Delay",
            category="actions",
            icon="timer",
            description="Wait for a duration",
            config_schema={
                "type": "object",
                "properties": {
                    "duration_seconds": {"type": "integer", "title": "Duration (seconds)"},
                },
                "required": ["duration_seconds"],
            },
            color="#795548",
        ),
        NodeTypeDefinition(
            type="action_transform",
            label="Transform Data",
            category="actions",
            icon="shuffle",
            description="Transform data with expression",
            config_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "title": "Expression"},
                    "output_key": {"type": "string", "title": "Output Variable"},
                },
                "required": ["expression"],
            },
            color="#607D8B",
        ),

        # Control flow
        NodeTypeDefinition(
            type="condition",
            label="Condition",
            category="control",
            icon="split",
            description="Branch based on condition",
            config_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "title": "Condition Expression"},
                },
                "required": ["expression"],
            },
            inputs=1,
            outputs=2,  # true/false branches
            color="#FFC107",
        ),
        NodeTypeDefinition(
            type="parallel",
            label="Parallel",
            category="control",
            icon="layers",
            description="Execute branches in parallel",
            inputs=1,
            outputs=3,  # Multiple parallel branches
            color="#00BCD4",
        ),
        NodeTypeDefinition(
            type="loop",
            label="Loop",
            category="control",
            icon="repeat",
            description="Iterate over collection",
            config_schema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "title": "Collection Expression"},
                    "item_variable": {"type": "string", "title": "Item Variable Name"},
                },
                "required": ["collection"],
            },
            inputs=1,
            outputs=1,
            color="#CDDC39",
        ),

        # Approval
        NodeTypeDefinition(
            type="approval",
            label="Approval Gate",
            category="approvals",
            icon="check-circle",
            description="Wait for human approval",
            config_schema={
                "type": "object",
                "properties": {
                    "approvers": {"type": "array", "items": {"type": "string"}, "title": "Approvers"},
                    "gate_type": {"type": "string", "enum": ["single", "multi", "quorum"], "default": "single"},
                    "timeout_hours": {"type": "integer", "title": "Timeout (hours)"},
                    "message": {"type": "string", "title": "Approval Message"},
                },
                "required": ["approvers"],
            },
            inputs=1,
            outputs=2,  # approved/denied
            color="#F44336",
        ),

        # Subworkflow
        NodeTypeDefinition(
            type="subworkflow",
            label="Sub-workflow",
            category="composition",
            icon="box",
            description="Execute another workflow",
            config_schema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "title": "Workflow ID"},
                    "inputs": {"type": "object", "title": "Inputs"},
                    "wait_for_completion": {"type": "boolean", "default": True},
                },
                "required": ["workflow_id"],
            },
            color="#8BC34A",
        ),

        # End node
        NodeTypeDefinition(
            type="end",
            label="End",
            category="control",
            icon="stop",
            description="End the workflow",
            inputs=1,
            outputs=0,
            color="#9E9E9E",
        ),
    ]

    @app.get("/api/visual/node-types", response_model=List[NodeTypeDefinition])
    async def get_node_types() -> List[NodeTypeDefinition]:
        """Get available node types for the palette."""
        return NODE_TYPES

    @app.get("/api/visual/node-types/{category}")
    async def get_node_types_by_category(category: str) -> List[NodeTypeDefinition]:
        """Get node types by category."""
        return [n for n in NODE_TYPES if n.category == category]

    @app.post("/api/visual/workflows", response_model=WorkflowGraphResponse)
    async def create_visual_workflow(
        graph: WorkflowGraph = Body(...),
    ) -> WorkflowGraphResponse:
        """Create a new visual workflow."""
        workflow_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        workflow_data = {
            "id": workflow_id,
            "name": graph.name,
            "description": graph.description,
            "nodes": [n.dict() for n in graph.nodes],
            "edges": [e.dict() for e in graph.edges],
            "metadata": graph.metadata,
            "created_at": now,
            "updated_at": now,
        }

        _visual_workflows[workflow_id] = workflow_data

        logger.info(f"Created visual workflow: {workflow_id}")
        return WorkflowGraphResponse(**workflow_data)

    @app.get("/api/visual/workflows", response_model=List[WorkflowGraphResponse])
    async def list_visual_workflows() -> List[WorkflowGraphResponse]:
        """List all visual workflows."""
        return [WorkflowGraphResponse(**w) for w in _visual_workflows.values()]

    @app.get("/api/visual/workflows/{workflow_id}", response_model=WorkflowGraphResponse)
    async def get_visual_workflow(workflow_id: str) -> WorkflowGraphResponse:
        """Get a visual workflow by ID."""
        if workflow_id not in _visual_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return WorkflowGraphResponse(**_visual_workflows[workflow_id])

    @app.put("/api/visual/workflows/{workflow_id}", response_model=WorkflowGraphResponse)
    async def update_visual_workflow(
        workflow_id: str,
        graph: WorkflowGraph = Body(...),
    ) -> WorkflowGraphResponse:
        """Update a visual workflow."""
        if workflow_id not in _visual_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        existing = _visual_workflows[workflow_id]
        now = datetime.now().isoformat()

        workflow_data = {
            "id": workflow_id,
            "name": graph.name,
            "description": graph.description,
            "nodes": [n.dict() for n in graph.nodes],
            "edges": [e.dict() for e in graph.edges],
            "metadata": graph.metadata,
            "created_at": existing["created_at"],
            "updated_at": now,
        }

        _visual_workflows[workflow_id] = workflow_data

        logger.info(f"Updated visual workflow: {workflow_id}")
        return WorkflowGraphResponse(**workflow_data)

    @app.delete("/api/visual/workflows/{workflow_id}")
    async def delete_visual_workflow(workflow_id: str) -> Dict[str, bool]:
        """Delete a visual workflow."""
        if workflow_id not in _visual_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        del _visual_workflows[workflow_id]
        logger.info(f"Deleted visual workflow: {workflow_id}")
        return {"deleted": True}

    @app.post("/api/visual/workflows/{workflow_id}/validate", response_model=ValidationResult)
    async def validate_visual_workflow(workflow_id: str) -> ValidationResult:
        """Validate a visual workflow."""
        if workflow_id not in _visual_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        workflow = _visual_workflows[workflow_id]
        errors = []
        warnings = []

        nodes = workflow["nodes"]
        edges = workflow["edges"]

        # Check for at least one trigger
        triggers = [n for n in nodes if n["data"]["type"].startswith("trigger")]
        if not triggers:
            errors.append({
                "type": "missing_trigger",
                "message": "Workflow must have at least one trigger",
            })

        # Check for at least one end node or action
        end_nodes = [n for n in nodes if n["data"]["type"] == "end"]
        action_nodes = [n for n in nodes if n["data"]["type"].startswith("action")]
        if not end_nodes and not action_nodes:
            warnings.append({
                "type": "no_actions",
                "message": "Workflow has no actions or end nodes",
            })

        # Check for disconnected nodes
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge["source"])
            connected_nodes.add(edge["target"])

        for node in nodes:
            if node["id"] not in connected_nodes and not node["data"]["type"].startswith("trigger"):
                warnings.append({
                    "type": "disconnected_node",
                    "message": f"Node '{node['data']['label']}' is not connected",
                    "node_id": node["id"],
                })

        # Check for cycles (simple DFS)
        def has_cycle(start: str, visited: set, path: set) -> bool:
            visited.add(start)
            path.add(start)

            for edge in edges:
                if edge["source"] == start:
                    target = edge["target"]
                    if target in path:
                        return True
                    if target not in visited:
                        if has_cycle(target, visited, path):
                            return True

            path.remove(start)
            return False

        visited = set()
        for node in nodes:
            if node["id"] not in visited:
                if has_cycle(node["id"], visited, set()):
                    errors.append({
                        "type": "cycle_detected",
                        "message": "Workflow contains a cycle",
                    })
                    break

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @app.post("/api/visual/workflows/{workflow_id}/convert", response_model=ConversionResult)
    async def convert_visual_workflow(workflow_id: str) -> ConversionResult:
        """Convert visual workflow to executable workflow."""
        if workflow_id not in _visual_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        workflow = _visual_workflows[workflow_id]

        try:
            from aion.automation.visual.exporter import WorkflowExporter
            exporter = WorkflowExporter()

            # Convert visual graph to workflow definition
            workflow_def = exporter.export_to_workflow(
                nodes=workflow["nodes"],
                edges=workflow["edges"],
                name=workflow["name"],
                description=workflow["description"],
            )

            # Register with workflow engine
            from aion.automation.engine import WorkflowEngine
            # Note: This would need access to the actual engine instance

            logger.info(f"Converted visual workflow: {workflow_id}")
            return ConversionResult(
                workflow_id=workflow_def.id,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to convert workflow: {e}")
            return ConversionResult(
                workflow_id=workflow_id,
                success=False,
                errors=[str(e)],
            )

    @app.get("/api/visual/workflows/{workflow_id}/export")
    async def export_visual_workflow(
        workflow_id: str,
        format: str = Query("yaml", enum=["yaml", "json"]),
    ):
        """Export visual workflow to YAML or JSON."""
        if workflow_id not in _visual_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        workflow = _visual_workflows[workflow_id]

        from aion.automation.visual.exporter import WorkflowExporter
        exporter = WorkflowExporter()

        if format == "yaml":
            content = exporter.export_to_yaml(
                nodes=workflow["nodes"],
                edges=workflow["edges"],
                name=workflow["name"],
                description=workflow["description"],
            )
            media_type = "application/x-yaml"
        else:
            content = exporter.export_to_json(
                nodes=workflow["nodes"],
                edges=workflow["edges"],
                name=workflow["name"],
                description=workflow["description"],
            )
            media_type = "application/json"

        return JSONResponse(
            content={"content": content, "format": format},
            media_type="application/json",
        )

    @app.get("/visual", response_class=HTMLResponse)
    async def visual_editor():
        """Serve the visual workflow editor."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>AION Workflow Builder</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/reactflow@11/dist/umd/index.js"></script>
    <link href="https://unpkg.com/reactflow@11/dist/style.css" rel="stylesheet" />
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; }
        .app { display: flex; height: 100vh; }
        .sidebar { width: 250px; background: #1a1a2e; color: white; padding: 16px; overflow-y: auto; }
        .sidebar h2 { font-size: 14px; color: #888; margin: 16px 0 8px; text-transform: uppercase; }
        .node-item { padding: 8px 12px; margin: 4px 0; background: #16213e; border-radius: 4px; cursor: grab; }
        .node-item:hover { background: #0f3460; }
        .canvas { flex: 1; }
        .toolbar { position: absolute; top: 10px; right: 10px; z-index: 10; }
        .toolbar button { padding: 8px 16px; margin: 0 4px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .toolbar button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div id="root"></div>
    <script>
        // Minimal React Flow editor
        const { useState, useCallback, useEffect } = React;
        const { ReactFlow, Controls, Background, addEdge, applyNodeChanges, applyEdgeChanges } = window.ReactFlow;

        function App() {
            const [nodes, setNodes] = useState([]);
            const [edges, setEdges] = useState([]);
            const [nodeTypes, setNodeTypes] = useState([]);

            useEffect(() => {
                fetch('/api/visual/node-types')
                    .then(r => r.json())
                    .then(setNodeTypes);
            }, []);

            const onNodesChange = useCallback((changes) => {
                setNodes((nds) => applyNodeChanges(changes, nds));
            }, []);

            const onEdgesChange = useCallback((changes) => {
                setEdges((eds) => applyEdgeChanges(changes, eds));
            }, []);

            const onConnect = useCallback((params) => {
                setEdges((eds) => addEdge(params, eds));
            }, []);

            const onDragStart = (e, type, label) => {
                e.dataTransfer.setData('application/reactflow', JSON.stringify({ type, label }));
            };

            const onDrop = useCallback((e) => {
                e.preventDefault();
                const data = JSON.parse(e.dataTransfer.getData('application/reactflow'));
                const position = { x: e.clientX - 250, y: e.clientY };
                const newNode = {
                    id: `node_${Date.now()}`,
                    type: 'default',
                    position,
                    data: { label: data.label, type: data.type, config: {} }
                };
                setNodes((nds) => [...nds, newNode]);
            }, []);

            const onDragOver = (e) => { e.preventDefault(); };

            const saveWorkflow = async () => {
                const workflow = {
                    name: 'My Workflow',
                    description: '',
                    nodes,
                    edges,
                    metadata: {}
                };
                const res = await fetch('/api/visual/workflows', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(workflow)
                });
                const data = await res.json();
                alert('Saved! ID: ' + data.id);
            };

            const categories = [...new Set(nodeTypes.map(n => n.category))];

            return React.createElement('div', { className: 'app' },
                React.createElement('div', { className: 'sidebar' },
                    React.createElement('h1', { style: { fontSize: '18px', marginBottom: '16px' } }, 'AION Workflow Builder'),
                    categories.map(cat => React.createElement('div', { key: cat },
                        React.createElement('h2', null, cat),
                        nodeTypes.filter(n => n.category === cat).map(node =>
                            React.createElement('div', {
                                key: node.type,
                                className: 'node-item',
                                draggable: true,
                                onDragStart: (e) => onDragStart(e, node.type, node.label),
                                style: { borderLeft: `3px solid ${node.color || '#888'}` }
                            }, node.label)
                        )
                    ))
                ),
                React.createElement('div', { className: 'canvas', onDrop, onDragOver },
                    React.createElement('div', { className: 'toolbar' },
                        React.createElement('button', { onClick: saveWorkflow }, 'Save'),
                        React.createElement('button', { onClick: () => setNodes([]) }, 'Clear')
                    ),
                    React.createElement(ReactFlow, {
                        nodes,
                        edges,
                        onNodesChange,
                        onEdgesChange,
                        onConnect,
                        fitView: true
                    },
                        React.createElement(Controls),
                        React.createElement(Background, { color: '#aaa', gap: 16 })
                    )
                )
            );
        }

        ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
    </script>
</body>
</html>
        """

    logger.info("Visual workflow builder routes configured")
