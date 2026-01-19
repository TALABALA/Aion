"""
AION Plan Visualizer

Visualization utilities for execution plans:
- Graph visualization with matplotlib
- ASCII representation for terminals
- Interactive HTML export
"""

from __future__ import annotations

import io
from typing import Any, Optional

import structlog

from aion.systems.planning.graph import (
    PlanningGraph,
    ExecutionPlan,
    PlanNode,
    NodeStatus,
    NodeType,
)

logger = structlog.get_logger(__name__)


# Status colors for visualization
STATUS_COLORS = {
    NodeStatus.PENDING: "#gray",
    NodeStatus.READY: "#blue",
    NodeStatus.RUNNING: "#yellow",
    NodeStatus.COMPLETED: "#green",
    NodeStatus.FAILED: "#red",
    NodeStatus.SKIPPED: "#orange",
    NodeStatus.CANCELLED: "#purple",
}

# Node type shapes
NODE_SHAPES = {
    NodeType.START: "ellipse",
    NodeType.END: "ellipse",
    NodeType.ACTION: "box",
    NodeType.CONDITION: "diamond",
    NodeType.PARALLEL: "parallelogram",
    NodeType.JOIN: "parallelogram",
    NodeType.SUBGRAPH: "box3d",
    NodeType.CHECKPOINT: "cylinder",
}


class PlanVisualizer:
    """Visualization utilities for execution plans."""

    def __init__(self, planning_graph: PlanningGraph):
        self.graph = planning_graph

    def to_ascii(self, plan_id: str) -> str:
        """
        Generate ASCII representation of a plan.

        Args:
            plan_id: Plan ID

        Returns:
            ASCII string representation
        """
        plan = self.graph.get_plan(plan_id)
        if not plan:
            return f"Plan not found: {plan_id}"

        lines = []
        lines.append(f"╔{'═' * 60}╗")
        lines.append(f"║ Plan: {plan.name:<52} ║")
        lines.append(f"║ Status: {plan.status.name:<50} ║")
        lines.append(f"╠{'═' * 60}╣")

        # Get execution order
        try:
            order = self.graph.get_execution_order(plan_id)
        except Exception:
            order = [[n] for n in plan.nodes.keys()]

        for level, node_ids in enumerate(order):
            level_str = f"Level {level}: "
            nodes_str = ", ".join(
                f"{plan.nodes[nid].name}[{plan.nodes[nid].status.name[:4]}]"
                for nid in node_ids
                if nid in plan.nodes
            )
            line = f"║ {level_str}{nodes_str}"
            lines.append(f"{line:<61}║")

        lines.append(f"╠{'═' * 60}╣")

        # Show edges
        lines.append("║ Edges:                                                      ║")
        for edge in plan.edges[:10]:  # Limit to first 10
            source_name = plan.nodes.get(edge.source, PlanNode(id="?", name="?", node_type=NodeType.ACTION)).name
            target_name = plan.nodes.get(edge.target, PlanNode(id="?", name="?", node_type=NodeType.ACTION)).name
            edge_str = f"  {source_name} → {target_name}"
            if edge.condition:
                edge_str += f" [{edge.condition}]"
            lines.append(f"║{edge_str:<60}║")

        lines.append(f"╚{'═' * 60}╝")

        return "\n".join(lines)

    def to_mermaid(self, plan_id: str) -> str:
        """
        Generate Mermaid diagram syntax for a plan.

        Args:
            plan_id: Plan ID

        Returns:
            Mermaid diagram string
        """
        plan = self.graph.get_plan(plan_id)
        if not plan:
            return f"Plan not found: {plan_id}"

        lines = ["graph TD"]
        lines.append(f"    subgraph {plan.name}")

        # Define nodes
        for node_id, node in plan.nodes.items():
            safe_id = node_id.replace("-", "_")
            status_icon = {
                NodeStatus.PENDING: "⏳",
                NodeStatus.READY: "🔵",
                NodeStatus.RUNNING: "🔄",
                NodeStatus.COMPLETED: "✅",
                NodeStatus.FAILED: "❌",
                NodeStatus.SKIPPED: "⏭️",
                NodeStatus.CANCELLED: "🚫",
            }.get(node.status, "")

            if node.node_type == NodeType.START:
                lines.append(f"    {safe_id}(({node.name} {status_icon}))")
            elif node.node_type == NodeType.END:
                lines.append(f"    {safe_id}(({node.name} {status_icon}))")
            elif node.node_type == NodeType.CONDITION:
                lines.append(f"    {safe_id}{{{{{node.name} {status_icon}}}}}")
            else:
                lines.append(f"    {safe_id}[{node.name} {status_icon}]")

        lines.append("    end")

        # Define edges
        for edge in plan.edges:
            source = edge.source.replace("-", "_")
            target = edge.target.replace("-", "_")
            if edge.condition:
                lines.append(f"    {source} -->|{edge.condition}| {target}")
            else:
                lines.append(f"    {source} --> {target}")

        return "\n".join(lines)

    def to_graphviz(self, plan_id: str) -> str:
        """
        Generate GraphViz DOT syntax for a plan.

        Args:
            plan_id: Plan ID

        Returns:
            DOT syntax string
        """
        plan = self.graph.get_plan(plan_id)
        if not plan:
            return f"// Plan not found: {plan_id}"

        lines = [f'digraph "{plan.name}" {{']
        lines.append("    rankdir=TB;")
        lines.append("    node [fontname=\"Helvetica\"];")
        lines.append("    edge [fontname=\"Helvetica\"];")

        # Define nodes
        for node_id, node in plan.nodes.items():
            safe_id = node_id.replace("-", "_")
            color = {
                NodeStatus.PENDING: "gray",
                NodeStatus.READY: "lightblue",
                NodeStatus.RUNNING: "yellow",
                NodeStatus.COMPLETED: "lightgreen",
                NodeStatus.FAILED: "red",
                NodeStatus.SKIPPED: "orange",
                NodeStatus.CANCELLED: "purple",
            }.get(node.status, "white")

            shape = {
                NodeType.START: "ellipse",
                NodeType.END: "ellipse",
                NodeType.ACTION: "box",
                NodeType.CONDITION: "diamond",
                NodeType.PARALLEL: "parallelogram",
                NodeType.JOIN: "parallelogram",
                NodeType.SUBGRAPH: "box3d",
                NodeType.CHECKPOINT: "cylinder",
            }.get(node.node_type, "box")

            label = f"{node.name}\\n{node.status.name}"
            lines.append(
                f'    {safe_id} [label="{label}" shape={shape} '
                f'style=filled fillcolor={color}];'
            )

        # Define edges
        for edge in plan.edges:
            source = edge.source.replace("-", "_")
            target = edge.target.replace("-", "_")
            label = edge.condition or ""
            lines.append(f'    {source} -> {target} [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def render_matplotlib(
        self,
        plan_id: str,
        output_path: Optional[str] = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> Optional[bytes]:
        """
        Render plan using matplotlib.

        Args:
            plan_id: Plan ID
            output_path: Path to save image (optional)
            figsize: Figure size

        Returns:
            PNG bytes if no output_path, else None
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.warning("matplotlib or networkx not available for visualization")
            return None

        plan = self.graph.get_plan(plan_id)
        if not plan:
            return None

        graph = self.graph._graphs.get(plan_id)
        if not graph:
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate positions
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        except Exception:
            pos = nx.spring_layout(graph)

        # Draw nodes
        colors = [
            {
                NodeStatus.PENDING: "#cccccc",
                NodeStatus.READY: "#87CEEB",
                NodeStatus.RUNNING: "#FFD700",
                NodeStatus.COMPLETED: "#90EE90",
                NodeStatus.FAILED: "#FF6B6B",
                NodeStatus.SKIPPED: "#FFA500",
                NodeStatus.CANCELLED: "#DDA0DD",
            }.get(plan.nodes[n].status, "#ffffff")
            for n in graph.nodes()
        ]

        nx.draw_networkx_nodes(
            graph, pos,
            node_color=colors,
            node_size=2000,
            ax=ax,
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edge_color="#666666",
            arrows=True,
            arrowsize=20,
            ax=ax,
        )

        # Draw labels
        labels = {n: plan.nodes[n].name for n in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels, font_size=10, ax=ax)

        ax.set_title(f"Execution Plan: {plan.name}")
        ax.axis("off")

        if output_path:
            plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()

    def to_html(self, plan_id: str) -> str:
        """
        Generate interactive HTML visualization using vis.js.

        Args:
            plan_id: Plan ID

        Returns:
            HTML string
        """
        plan = self.graph.get_plan(plan_id)
        if not plan:
            return f"<p>Plan not found: {plan_id}</p>"

        # Build nodes and edges for vis.js
        nodes_js = []
        for node_id, node in plan.nodes.items():
            color = {
                NodeStatus.PENDING: "#cccccc",
                NodeStatus.READY: "#87CEEB",
                NodeStatus.RUNNING: "#FFD700",
                NodeStatus.COMPLETED: "#90EE90",
                NodeStatus.FAILED: "#FF6B6B",
                NodeStatus.SKIPPED: "#FFA500",
                NodeStatus.CANCELLED: "#DDA0DD",
            }.get(node.status, "#ffffff")

            shape = "box"
            if node.node_type in (NodeType.START, NodeType.END):
                shape = "ellipse"
            elif node.node_type == NodeType.CONDITION:
                shape = "diamond"

            nodes_js.append({
                "id": node_id,
                "label": f"{node.name}\\n{node.status.name}",
                "color": color,
                "shape": shape,
            })

        edges_js = []
        for edge in plan.edges:
            edges_js.append({
                "from": edge.source,
                "to": edge.target,
                "label": edge.condition or "",
            })

        import json
        nodes_json = json.dumps(nodes_js)
        edges_json = json.dumps(edges_js)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AION Plan: {plan.name}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #333; }}
        #network {{ width: 100%; height: 600px; border: 1px solid #ccc; }}
        .info {{ margin-top: 20px; padding: 10px; background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>AION Execution Plan: {plan.name}</h1>
    <p>Status: {plan.status.name}</p>
    <div id="network"></div>
    <div class="info">
        <strong>Nodes:</strong> {len(plan.nodes)} |
        <strong>Edges:</strong> {len(plan.edges)}
    </div>
    <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'directed' }} }},
            edges: {{ arrows: 'to' }},
            physics: false
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
        return html
