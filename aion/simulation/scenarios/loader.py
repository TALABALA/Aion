"""AION Scenario Loader - Load scenarios from files and configurations.

Supports loading from:
- JSON files
- YAML-like dict structures
- Python scenario definitions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog

from aion.simulation.types import Scenario, ScenarioType

logger = structlog.get_logger(__name__)


class ScenarioLoader:
    """Loads scenarios from various sources."""

    def __init__(self) -> None:
        self._cache: Dict[str, Scenario] = {}

    def load_from_dict(self, data: Dict[str, Any]) -> Scenario:
        """Load a scenario from a dictionary."""
        return Scenario(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            type=ScenarioType(data.get("type", "simple")),
            initial_state=data.get("initial_state", {}),
            initial_entities=data.get("initial_entities", data.get("entities", [])),
            scripted_events=data.get("scripted_events", data.get("events", [])),
            simulated_users=data.get("simulated_users", []),
            goals=data.get("goals", []),
            success_criteria=data.get("success_criteria", []),
            failure_criteria=data.get("failure_criteria", []),
            max_steps=data.get("max_steps", 1000),
            max_time=data.get("max_time", 3600.0),
            config=data.get("config", {}),
            tags=set(data.get("tags", [])),
            difficulty=data.get("difficulty", 0.5),
        )

    def load_from_json(self, path: str) -> Scenario:
        """Load a scenario from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(file_path) as f:
            data = json.load(f)

        scenario = self.load_from_dict(data)
        self._cache[path] = scenario
        return scenario

    def load_from_json_string(self, json_str: str) -> Scenario:
        """Load a scenario from a JSON string."""
        data = json.loads(json_str)
        return self.load_from_dict(data)

    def load_batch(self, path: str) -> List[Scenario]:
        """Load multiple scenarios from a JSON file containing a list."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario batch file not found: {path}")

        with open(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return [self.load_from_dict(d) for d in data]
        elif isinstance(data, dict) and "scenarios" in data:
            return [self.load_from_dict(d) for d in data["scenarios"]]
        else:
            return [self.load_from_dict(data)]

    def load_directory(self, directory: str) -> List[Scenario]:
        """Load all scenario JSON files from a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        scenarios: List[Scenario] = []
        for json_file in sorted(dir_path.glob("*.json")):
            try:
                scenario = self.load_from_json(str(json_file))
                scenarios.append(scenario)
            except Exception as exc:
                logger.error("scenario_load_error", file=str(json_file), error=str(exc))

        return scenarios

    def save_to_json(self, scenario: Scenario, path: str) -> None:
        """Save a scenario to a JSON file."""
        data = scenario.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_cached(self, path: str) -> Optional[Scenario]:
        return self._cache.get(path)

    def clear_cache(self) -> None:
        self._cache.clear()
