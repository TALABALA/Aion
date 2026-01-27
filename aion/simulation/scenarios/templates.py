"""AION Scenario Templates - Built-in scenario template library.

Provides predefined scenario templates for common testing patterns:
- Customer service interactions
- Task execution workflows
- Multi-agent coordination
- System failure recovery
- Performance benchmarks
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import structlog

from aion.simulation.types import Scenario, ScenarioType

logger = structlog.get_logger(__name__)


class ScenarioTemplateLibrary:
    """Library of built-in scenario templates.

    Templates are parameterizable blueprints for common testing patterns.
    Each template produces a fully configured Scenario with sensible defaults.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._register_builtins()

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        return list(self._templates.keys())

    def list_by_tag(self, tag: str) -> List[str]:
        return [
            name
            for name, t in self._templates.items()
            if tag in t.get("tags", set())
        ]

    def register(self, name: str, template: Dict[str, Any]) -> None:
        self._templates[name] = template

    def instantiate(
        self,
        name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Scenario:
        """Create a Scenario from a template with optional overrides."""
        template = self._templates.get(name)
        if template is None:
            raise ValueError(f"Unknown template: {name}")

        config = {**template, **(overrides or {})}

        return Scenario(
            name=config.get("name", name),
            description=config.get("description", ""),
            type=ScenarioType(config.get("type", "simple")),
            initial_state=config.get("initial_state", {}),
            initial_entities=config.get("entities", []),
            scripted_events=config.get("events", []),
            simulated_users=config.get("simulated_users", []),
            goals=config.get("goals", []),
            success_criteria=config.get("success_criteria", []),
            failure_criteria=config.get("failure_criteria", []),
            max_steps=config.get("max_steps", 1000),
            max_time=config.get("max_time", 3600.0),
            config=config.get("config", {}),
            tags=set(config.get("tags", [])),
            difficulty=config.get("difficulty", 0.5),
        )

    # -- Built-in templates --

    def _register_builtins(self) -> None:
        # --- Customer Service ---
        self.register("customer_support_basic", {
            "name": "customer_support_basic",
            "description": "Basic customer support interaction with a single query",
            "type": "simple",
            "entities": [
                {"type": "user", "name": "customer", "properties": {"role": "customer", "sentiment": "neutral"}},
                {"type": "agent", "name": "support_agent"},
            ],
            "events": [
                {"time": 0, "type": "user_input", "data": {"message": "I need help with my account"}},
            ],
            "goals": [
                {"name": "issue_resolved", "condition": "customer_satisfied"},
                {"name": "response_given", "condition": "response_sent"},
            ],
            "tags": {"customer_service", "basic", "support"},
            "difficulty": 0.3,
            "max_steps": 50,
        })

        self.register("customer_support_escalation", {
            "name": "customer_support_escalation",
            "description": "Customer service with escalation path",
            "type": "sequential",
            "entities": [
                {"type": "user", "name": "angry_customer", "properties": {"role": "customer", "sentiment": "negative"}},
                {"type": "agent", "name": "tier1_agent"},
                {"type": "agent", "name": "tier2_agent"},
            ],
            "events": [
                {"time": 0, "type": "user_input", "data": {"message": "This is unacceptable!"}},
                {"time": 10, "type": "user_input", "data": {"message": "I want to speak to a manager"}},
                {"time": 20, "type": "system_event", "data": {"action": "escalate", "to": "tier2_agent"}},
            ],
            "goals": [
                {"name": "escalation_handled", "condition": "escalation_completed"},
                {"name": "customer_retained", "condition": "customer_satisfied"},
            ],
            "tags": {"customer_service", "escalation", "support"},
            "difficulty": 0.6,
            "max_steps": 200,
        })

        # --- Task Execution ---
        self.register("task_execution_simple", {
            "name": "task_execution_simple",
            "description": "Simple task assignment and completion",
            "type": "simple",
            "entities": [
                {"type": "user", "name": "requester"},
                {"type": "task", "name": "work_item", "properties": {"status": "pending", "priority": "medium"}},
            ],
            "events": [
                {"time": 0, "type": "user_input", "data": {"message": "Complete this task", "task_id": "work_item"}},
            ],
            "goals": [
                {"name": "task_completed", "condition": "task.status == completed"},
            ],
            "tags": {"task", "execution", "basic"},
            "difficulty": 0.3,
        })

        self.register("task_execution_pipeline", {
            "name": "task_execution_pipeline",
            "description": "Multi-stage task pipeline with dependencies",
            "type": "sequential",
            "entities": [
                {"type": "user", "name": "requester"},
                {"type": "task", "name": "task_1", "properties": {"status": "pending", "stage": 1}},
                {"type": "task", "name": "task_2", "properties": {"status": "blocked", "stage": 2, "depends_on": "task_1"}},
                {"type": "task", "name": "task_3", "properties": {"status": "blocked", "stage": 3, "depends_on": "task_2"}},
            ],
            "events": [
                {"time": 0, "type": "user_input", "data": {"message": "Execute the pipeline"}},
            ],
            "goals": [
                {"name": "pipeline_complete", "condition": "all_tasks_completed"},
                {"name": "correct_order", "condition": "execution_order_valid"},
            ],
            "tags": {"task", "pipeline", "dependencies"},
            "difficulty": 0.6,
            "max_steps": 500,
        })

        # --- Multi-Agent ---
        self.register("multi_agent_collaboration", {
            "name": "multi_agent_collaboration",
            "description": "Multiple agents collaborating on a shared task",
            "type": "branching",
            "entities": [
                {"type": "agent", "name": "researcher", "properties": {"role": "research"}},
                {"type": "agent", "name": "writer", "properties": {"role": "writing"}},
                {"type": "agent", "name": "reviewer", "properties": {"role": "review"}},
                {"type": "task", "name": "project", "properties": {"status": "pending"}},
            ],
            "events": [
                {"time": 0, "type": "system_event", "data": {"action": "assign_task", "task": "project"}},
                {"time": 10, "type": "system_event", "data": {"action": "coordinate", "agents": ["researcher", "writer", "reviewer"]}},
            ],
            "goals": [
                {"name": "project_completed", "condition": "project.status == completed"},
                {"name": "all_contributed", "condition": "all_agents_participated"},
            ],
            "tags": {"multi_agent", "collaboration"},
            "difficulty": 0.7,
            "max_steps": 1000,
        })

        # --- System Failure ---
        self.register("graceful_degradation", {
            "name": "graceful_degradation",
            "description": "System failure with graceful degradation expected",
            "type": "adversarial",
            "entities": [
                {"type": "user", "name": "user"},
                {"type": "system", "name": "service_a", "properties": {"status": "healthy"}},
                {"type": "system", "name": "service_b", "properties": {"status": "healthy"}},
            ],
            "events": [
                {"time": 0, "type": "user_input", "data": {"message": "Normal request"}},
                {"time": 5, "type": "system_event", "data": {"action": "inject_failure", "target": "service_a"}},
                {"time": 10, "type": "user_input", "data": {"message": "Request during failure"}},
                {"time": 20, "type": "system_event", "data": {"action": "restore", "target": "service_a"}},
                {"time": 25, "type": "user_input", "data": {"message": "Post-recovery request"}},
            ],
            "success_criteria": [
                {"name": "no_crash", "condition": "simulation_completed"},
                {"name": "degraded_response", "condition": "fallback_activated"},
                {"name": "recovery", "condition": "service_restored"},
            ],
            "tags": {"failure", "degradation", "resilience"},
            "difficulty": 0.7,
            "max_steps": 300,
        })

        # --- Performance Benchmark ---
        self.register("throughput_benchmark", {
            "name": "throughput_benchmark",
            "description": "Benchmark throughput under sustained load",
            "type": "stress",
            "entities": [
                {"type": "user", "name": f"bench_user_{i}"}
                for i in range(50)
            ],
            "events": [
                {
                    "time": r * 0.1 + u * 0.002,
                    "type": "user_input",
                    "source": f"bench_user_{u}",
                    "data": {"message": f"Benchmark request {r}"},
                }
                for u in range(50)
                for r in range(20)
            ],
            "success_criteria": [
                {"name": "all_processed", "condition": "all_requests_processed"},
                {"name": "latency_p99", "condition": "p99_latency < 2000"},
                {"name": "throughput", "condition": "throughput > 100"},
            ],
            "tags": {"benchmark", "throughput", "performance"},
            "difficulty": 0.8,
            "max_steps": 5000,
        })

        # --- Data Processing ---
        self.register("data_pipeline", {
            "name": "data_pipeline",
            "description": "Data processing pipeline with validation",
            "type": "sequential",
            "entities": [
                {"type": "resource", "name": "input_data", "properties": {"format": "csv", "rows": 1000}},
                {"type": "resource", "name": "output_data", "properties": {"format": "json"}},
            ],
            "events": [
                {"time": 0, "type": "system_event", "data": {"action": "ingest", "source": "input_data"}},
                {"time": 5, "type": "system_event", "data": {"action": "transform"}},
                {"time": 10, "type": "system_event", "data": {"action": "validate"}},
                {"time": 15, "type": "system_event", "data": {"action": "output", "target": "output_data"}},
            ],
            "goals": [
                {"name": "pipeline_success", "condition": "output_valid"},
                {"name": "no_data_loss", "condition": "output_rows >= input_rows"},
            ],
            "tags": {"data", "pipeline", "etl"},
            "difficulty": 0.5,
        })
