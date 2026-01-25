"""
AION Workflow Templates

Pre-built workflow templates for common patterns:
- Daily reports
- Webhook handlers
- Approval workflows
- Data sync
- Alert handling
"""

from aion.automation.templates.manager import TemplateManager
from aion.automation.templates.builtin import get_builtin_templates

__all__ = [
    "TemplateManager",
    "get_builtin_templates",
]
