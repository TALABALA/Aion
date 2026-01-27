"""
AION Integration Synthesizer - Generate integration code.

Creates data integration pipelines with:
- Source/target connectors
- Data mapping and transformation
- Sync scheduling
- Error handling and conflict resolution
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import structlog

from aion.nlp.synthesis.base import BaseSynthesizer
from aion.nlp.types import (
    GeneratedCode,
    IntegrationSpecification,
    SpecificationType,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class IntegrationSynthesizer(BaseSynthesizer):
    """Synthesizes integration code from IntegrationSpecification."""

    async def synthesize(self, spec: IntegrationSpecification) -> GeneratedCode:
        """Generate integration code from specification."""
        connector_code = await self._generate_connectors(spec)
        mapping_code = self._generate_mapping(spec)
        sync_code = self._generate_sync_logic(spec)

        code = f'''"""
Integration: {spec.name}
Description: {spec.description}
Source: {spec.source_system}
Target: {spec.target_system}
Sync Mode: {spec.sync_mode}
Direction: {spec.sync_direction}
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
@dataclass
class IntegrationConfig:
    """Configuration for {spec.name}."""
    source_system: str = "{spec.source_system}"
    target_system: str = "{spec.target_system}"
    sync_mode: str = "{spec.sync_mode}"
    sync_direction: str = "{spec.sync_direction}"
    on_conflict: str = "{spec.on_conflict}"
    on_error: str = "{spec.on_error}"
    max_retries: int = {spec.max_retries}

config = IntegrationConfig()

# Connectors
{connector_code}

# Data Mapping
{mapping_code}

# Sync Logic
{sync_code}

# Registration
def register_integration():
    """Register this integration with AION."""
    return {{
        "name": "{spec.name}",
        "config": config,
        "sync": sync_data,
    }}
'''

        return GeneratedCode(
            language="python",
            code=code.strip(),
            filename=f"integration_{spec.name}.py",
            spec_type=SpecificationType.INTEGRATION,
            imports=[
                "from typing import Any, Dict, List, Optional",
                "from dataclasses import dataclass, field",
            ],
            docstring=spec.description,
        )

    async def _generate_connectors(self, spec: IntegrationSpecification) -> str:
        """Generate source and target connector code."""
        prompt = f"""Generate Python connector classes for this data integration:

Source system: {spec.source_system}
Source config: {spec.source_config}
Target system: {spec.target_system}
Target config: {spec.target_config}

Generate two async classes:
1. SourceConnector - connects to source, fetches data
2. TargetConnector - connects to target, writes data

Each should have: connect(), disconnect(), fetch()/write() methods.
Be practical and handle errors. Generate ONLY the classes:"""

        try:
            return await self._llm_generate(prompt)
        except Exception:
            return f"""
class SourceConnector:
    \"\"\"Connector for {spec.source_system}.\"\"\"

    async def connect(self) -> None:
        # TODO: Implement connection to {spec.source_system}
        pass

    async def disconnect(self) -> None:
        pass

    async def fetch(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        # TODO: Implement data fetching from {spec.source_system}
        return []


class TargetConnector:
    \"\"\"Connector for {spec.target_system}.\"\"\"

    async def connect(self) -> None:
        # TODO: Implement connection to {spec.target_system}
        pass

    async def disconnect(self) -> None:
        pass

    async def write(self, records: List[Dict[str, Any]]) -> int:
        # TODO: Implement data writing to {spec.target_system}
        return 0
"""

    def _generate_mapping(self, spec: IntegrationSpecification) -> str:
        """Generate data mapping code."""
        if spec.field_mapping:
            mapping_entries = "\n".join(
                f'    "{m.get("source", "")}" : "{m.get("target", "")}",'
                for m in spec.field_mapping
            )
            return f"""
FIELD_MAPPING = {{
{mapping_entries}
}}


def map_record(source_record: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Map a source record to target format.\"\"\"
    target = {{}}
    for source_field, target_field in FIELD_MAPPING.items():
        if source_field in source_record:
            target[target_field] = source_record[source_field]
    return target
"""
        else:
            return """
def map_record(source_record: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Map a source record to target format (passthrough).\"\"\"
    return source_record.copy()
"""

    def _generate_sync_logic(self, spec: IntegrationSpecification) -> str:
        """Generate sync logic."""
        return f"""
async def sync_data(
    full_sync: bool = False,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    \"\"\"Execute data sync from {spec.source_system} to {spec.target_system}.\"\"\"
    source = SourceConnector()
    target = TargetConnector()
    results = {{
        "started_at": datetime.now().isoformat(),
        "records_fetched": 0,
        "records_written": 0,
        "errors": [],
    }}

    try:
        await source.connect()
        await target.connect()

        # Fetch data
        records = await source.fetch(since=None if full_sync else since)
        results["records_fetched"] = len(records)

        # Map and write
        mapped = [map_record(r) for r in records]

        for i in range(0, len(mapped), 100):  # Batch write
            batch = mapped[i:i + 100]
            try:
                written = await target.write(batch)
                results["records_written"] += written
            except Exception as e:
                results["errors"].append(str(e))
                if config.on_error == "stop":
                    break

    except Exception as e:
        results["errors"].append(f"Sync failed: {{e}}")
    finally:
        await source.disconnect()
        await target.disconnect()

    results["completed_at"] = datetime.now().isoformat()
    results["success"] = len(results["errors"]) == 0
    return results
"""
