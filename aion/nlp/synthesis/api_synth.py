"""
AION API Synthesizer - Generate REST API implementations.

Creates FastAPI-based API definitions with:
- CRUD endpoints
- Request/response models
- Authentication
- Rate limiting
- Input validation
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import structlog

from aion.nlp.synthesis.base import BaseSynthesizer
from aion.nlp.types import (
    APISpecification,
    APIEndpointSpec,
    GeneratedCode,
    SpecificationType,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class APISynthesizer(BaseSynthesizer):
    """Synthesizes FastAPI-based API code from APISpecification."""

    async def synthesize(self, spec: APISpecification) -> GeneratedCode:
        """Generate API code from specification."""
        # Generate models
        models_code = self._generate_models(spec)

        # Generate endpoints
        endpoints_code = self._generate_endpoints(spec)

        # Generate middleware
        middleware_code = self._generate_middleware(spec)

        code = f'''"""
API: {spec.name}
Description: {spec.description}
Version: {spec.version}
Base Path: {spec.base_path}/{spec.version}
Endpoints: {len(spec.endpoints)}
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="{spec.base_path}/{spec.version}",
    tags=["{spec.name}"],
)

# Models
{models_code}

# Middleware / Dependencies
{middleware_code}

# Endpoints
{endpoints_code}

# Registration
def setup_routes(app):
    """Register API routes with the application."""
    app.include_router(router)
'''

        return GeneratedCode(
            language="python",
            code=code.strip(),
            filename=f"api_{spec.name}.py",
            spec_type=SpecificationType.API,
            imports=[
                "from fastapi import APIRouter, HTTPException, Depends, Query",
                "from pydantic import BaseModel, Field",
                "from typing import Any, Dict, List, Optional",
            ],
            dependencies=["fastapi", "pydantic"],
            docstring=spec.description,
        )

    def _generate_models(self, spec: APISpecification) -> str:
        """Generate Pydantic models."""
        if spec.models:
            models: List[str] = []
            for model in spec.models:
                name = model.get("name", "Resource")
                fields = model.get("fields", [])
                field_lines = []
                for f in fields:
                    ftype = self._python_type(f.get("type", "string"))
                    fdesc = f.get("description", "")
                    default = f.get("default", "...")
                    field_lines.append(
                        f'    {f["name"]}: {ftype} = Field({repr(default)}, description="{fdesc}")'
                    )
                models.append(f"""
class {name}(BaseModel):
    \"\"\"Model for {name}.\"\"\"
{chr(10).join(field_lines) or "    pass"}
""")
            return "\n".join(models)

        # Generate default model from API name
        resource = spec.name.replace("_api", "").replace("_", " ").title().replace(" ", "")
        return f"""
class {resource}Base(BaseModel):
    \"\"\"Base model for {resource}.\"\"\"
    name: str = Field(..., description="Name")
    description: Optional[str] = Field(None, description="Description")


class {resource}Create({resource}Base):
    \"\"\"Model for creating a {resource}.\"\"\"
    pass


class {resource}Response({resource}Base):
    \"\"\"Model for {resource} response.\"\"\"
    id: str = Field(..., description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
"""

    def _generate_endpoints(self, spec: APISpecification) -> str:
        """Generate endpoint handlers."""
        endpoints: List[str] = []
        resource = spec.name.replace("_api", "").replace("_", " ").title().replace(" ", "")

        for ep in spec.endpoints:
            handler = self._generate_endpoint_handler(ep, resource, spec)
            endpoints.append(handler)

        return "\n\n".join(endpoints)

    def _generate_endpoint_handler(
        self,
        ep: APIEndpointSpec,
        resource: str,
        spec: APISpecification,
    ) -> str:
        """Generate a single endpoint handler."""
        method = ep.method.lower()
        path = ep.path
        func_name = self._endpoint_func_name(ep)

        # Determine handler signature and body based on method
        if method == "get" and "{id}" not in path:
            return f"""
@router.get("{path}")
async def {func_name}(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    \"\"\"{ep.description}\"\"\"
    # TODO: Implement listing
    return []
"""
        elif method == "get" and "{id}" in path:
            return f"""
@router.get("{path}")
async def {func_name}(id: str) -> Dict[str, Any]:
    \"\"\"{ep.description}\"\"\"
    # TODO: Implement get by ID
    raise HTTPException(404, detail="Not found")
"""
        elif method == "post":
            return f"""
@router.post("{path}", status_code=201)
async def {func_name}(body: {resource}Create) -> Dict[str, Any]:
    \"\"\"{ep.description}\"\"\"
    # TODO: Implement creation
    import uuid
    return {{
        "id": str(uuid.uuid4()),
        **body.model_dump(),
        "created_at": datetime.now().isoformat(),
    }}
"""
        elif method == "put":
            return f"""
@router.put("{path}")
async def {func_name}(id: str, body: {resource}Create) -> Dict[str, Any]:
    \"\"\"{ep.description}\"\"\"
    # TODO: Implement update
    return {{"id": id, **body.model_dump(), "updated_at": datetime.now().isoformat()}}
"""
        elif method == "delete":
            return f"""
@router.delete("{path}", status_code=204)
async def {func_name}(id: str) -> None:
    \"\"\"{ep.description}\"\"\"
    # TODO: Implement deletion
    return None
"""
        else:
            return f"""
@router.{method}("{path}")
async def {func_name}() -> Dict[str, Any]:
    \"\"\"{ep.description}\"\"\"
    return {{"status": "ok"}}
"""

    def _generate_middleware(self, spec: APISpecification) -> str:
        """Generate middleware/dependencies."""
        parts: List[str] = []

        if spec.auth_type:
            parts.append(f"""
async def verify_auth(token: str = Depends()):
    \"\"\"Verify authentication ({spec.auth_type}).\"\"\"
    if not token:
        raise HTTPException(401, detail="Authentication required")
    return token
""")

        if spec.rate_limit:
            parts.append(f"""
# Rate limit: {spec.rate_limit} requests per minute
RATE_LIMIT = {spec.rate_limit}
""")

        return "\n".join(parts) if parts else "# No middleware configured"

    def _endpoint_func_name(self, ep: APIEndpointSpec) -> str:
        """Generate function name from endpoint."""
        method = ep.method.lower()
        path = ep.path.strip("/").replace("{", "").replace("}", "")
        parts = path.replace("/", "_").split("_")
        name = "_".join(p for p in parts if p)
        return f"{method}_{name}" if name else f"{method}_root"
