"""
AION Template Manager

Manages workflow templates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

import structlog

from aion.automation.types import Workflow, WorkflowTemplate, WorkflowStatus

logger = structlog.get_logger(__name__)


class TemplateManager:
    """
    Manages workflow templates.

    Features:
    - Template storage and retrieval
    - Template instantiation
    - Category organization
    - Usage tracking
    """

    def __init__(self):
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the template manager."""
        if self._initialized:
            return

        # Load built-in templates
        from aion.automation.templates.builtin import get_builtin_templates

        for template in get_builtin_templates():
            await self.register(template)

        self._initialized = True
        logger.info("Template manager initialized", templates=len(self._templates))

    # === Template Management ===

    async def register(self, template: WorkflowTemplate) -> str:
        """Register a template."""
        self._templates[template.id] = template

        # Index by category
        category = template.category or "uncategorized"
        if category not in self._by_category:
            self._by_category[category] = []
        self._by_category[category].append(template.id)

        logger.info(
            "template_registered",
            template_id=template.id,
            name=template.name,
            category=category,
        )

        return template.id

    async def get(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    async def get_by_name(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a template by name."""
        for template in self._templates.values():
            if template.name == name:
                return template
        return None

    async def list(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> List[WorkflowTemplate]:
        """List templates with filters."""
        if category:
            template_ids = self._by_category.get(category, [])
            templates = [self._templates[tid] for tid in template_ids if tid in self._templates]
        else:
            templates = list(self._templates.values())

        if search:
            search_lower = search.lower()
            templates = [
                t for t in templates
                if search_lower in t.name.lower() or search_lower in t.description.lower()
            ]

        # Sort by usage count
        templates.sort(key=lambda t: t.usage_count, reverse=True)

        return templates[:limit]

    async def list_categories(self) -> List[Dict[str, Any]]:
        """List all categories with counts."""
        categories = []
        for category, template_ids in self._by_category.items():
            categories.append({
                "name": category,
                "count": len(template_ids),
            })
        return sorted(categories, key=lambda c: c["count"], reverse=True)

    async def delete(self, template_id: str) -> bool:
        """Delete a template."""
        template = self._templates.pop(template_id, None)
        if not template:
            return False

        # Remove from category index
        category = template.category or "uncategorized"
        if category in self._by_category:
            self._by_category[category] = [
                tid for tid in self._by_category[category]
                if tid != template_id
            ]

        return True

    # === Template Instantiation ===

    async def instantiate(
        self,
        template_id: str,
        name: str,
        parameters: Dict[str, Any] = None,
        owner_id: str = "",
    ) -> Workflow:
        """
        Create a workflow from a template.

        Args:
            template_id: Template to instantiate
            name: Name for the new workflow
            parameters: Template parameters
            owner_id: Owner of the new workflow

        Returns:
            New workflow instance
        """
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Create workflow from template
        workflow = template.instantiate(name, parameters)
        workflow.owner_id = owner_id

        # Update template usage
        template.usage_count += 1

        logger.info(
            "template_instantiated",
            template_id=template_id,
            template_name=template.name,
            workflow_name=name,
        )

        return workflow

    async def preview(
        self,
        template_id: str,
        parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Preview a template with parameters applied.

        Returns a representation of what the workflow would look like.
        """
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Create preview
        preview_workflow = template.instantiate("preview", parameters)

        return {
            "name": preview_workflow.name,
            "description": preview_workflow.description,
            "triggers": [t.to_dict() for t in preview_workflow.triggers],
            "steps": [s.to_dict() for s in preview_workflow.steps],
            "parameters_applied": parameters or {},
            "default_parameters": template.parameters,
        }

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get template manager statistics."""
        total_usage = sum(t.usage_count for t in self._templates.values())

        return {
            "total_templates": len(self._templates),
            "categories": len(self._by_category),
            "total_usage": total_usage,
            "top_templates": sorted(
                [{"name": t.name, "usage": t.usage_count} for t in self._templates.values()],
                key=lambda x: x["usage"],
                reverse=True,
            )[:5],
        }
