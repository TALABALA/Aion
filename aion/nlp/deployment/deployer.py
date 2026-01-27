"""
AION Deployment Manager - Deploy generated systems to AION.

Handles the full deployment lifecycle including:
- Code execution and registration
- Version management
- Health monitoring
- Rollback support
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.types import (
    DeployedSystem,
    DeploymentRecord,
    DeploymentStatus,
    GeneratedCode,
    SpecificationType,
)
from aion.nlp.deployment.registry import DeploymentRegistry
from aion.nlp.deployment.rollback import RollbackManager

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class DeploymentManager:
    """
    Manages the deployment lifecycle of generated systems.

    Responsibilities:
    - Deploy tools, workflows, agents, APIs
    - Track versions and history
    - Enable rollback
    - Monitor deployed system health
    """

    def __init__(self, kernel: AIONKernel):
        self.kernel = kernel
        self._registry = DeploymentRegistry()
        self._rollback = RollbackManager(self._registry)

    async def deploy(
        self,
        code: GeneratedCode,
        spec: Any,
        user_id: str = "",
    ) -> DeployedSystem:
        """
        Deploy generated code to AION.

        Args:
            code: Generated code to deploy
            spec: Specification that generated the code
            user_id: User who initiated deployment

        Returns:
            DeployedSystem record
        """
        name = getattr(spec, "name", code.filename)

        # Check for existing deployment (version update)
        existing = self._registry.find_by_name(name)
        version = existing.version + 1 if existing else 1

        # Create deployment record
        deployed = DeployedSystem(
            name=name,
            system_type=code.spec_type,
            specification=spec,
            generated_code=code,
            status=DeploymentStatus.DEPLOYING,
            version=version,
            created_by=user_id,
        )

        try:
            # Deploy based on type
            deployer = self._get_type_deployer(code.spec_type)
            await deployer(deployed)

            # Record deployment
            deployed.status = DeploymentStatus.ACTIVE
            deployed.deployment_history.append(DeploymentRecord(
                version=version,
                code_fingerprint=code.fingerprint,
                deployed_by=user_id,
                change_summary=f"Deployed {name} v{version}",
            ))

            # Register in registry
            self._registry.register(deployed)

            logger.info(
                "System deployed",
                name=name,
                type=code.spec_type.value,
                version=version,
            )

        except Exception as e:
            deployed.status = DeploymentStatus.FAILED
            logger.error("Deployment failed", name=name, error=str(e))
            raise

        return deployed

    async def undeploy(self, system_id: str) -> bool:
        """Undeploy a system."""
        deployed = self._registry.get(system_id)
        if not deployed:
            return False

        try:
            if deployed.system_type == SpecificationType.TOOL:
                if self.kernel.tools:
                    await self.kernel.tools.unregister_tool(deployed.name)
            elif deployed.system_type == SpecificationType.WORKFLOW:
                if self.kernel.automation:
                    await self.kernel.automation.remove_workflow(deployed.name)

            deployed.status = DeploymentStatus.ROLLED_BACK
            self._registry.update(deployed)
            logger.info("System undeployed", name=deployed.name)
            return True

        except Exception as e:
            logger.error("Undeploy failed", name=deployed.name, error=str(e))
            return False

    async def rollback(self, system_id: str, target_version: Optional[int] = None) -> bool:
        """Rollback a system to a previous version."""
        return await self._rollback.rollback(system_id, target_version)

    def get_deployed(self, system_id: str) -> Optional[DeployedSystem]:
        return self._registry.get(system_id)

    def list_deployed(
        self,
        system_type: Optional[SpecificationType] = None,
        status: Optional[DeploymentStatus] = None,
    ) -> List[DeployedSystem]:
        return self._registry.list_all(system_type=system_type, status=status)

    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        all_systems = self._registry.list_all()
        active = [s for s in all_systems if s.is_active]

        return {
            "total_deployed": len(all_systems),
            "active": len(active),
            "by_type": {
                t.value: len([s for s in all_systems if s.system_type == t])
                for t in SpecificationType
            },
            "total_invocations": sum(s.invocation_count for s in all_systems),
            "total_errors": sum(s.error_count for s in all_systems),
        }

    def _get_type_deployer(self, spec_type: SpecificationType):
        """Get the deployer function for a spec type."""
        deployers = {
            SpecificationType.TOOL: self._deploy_tool,
            SpecificationType.WORKFLOW: self._deploy_workflow,
            SpecificationType.AGENT: self._deploy_agent,
            SpecificationType.API: self._deploy_api,
            SpecificationType.INTEGRATION: self._deploy_integration,
            SpecificationType.FUNCTION: self._deploy_tool,
        }
        deployer = deployers.get(spec_type)
        if not deployer:
            raise ValueError(f"No deployer for type: {spec_type.value}")
        return deployer

    async def _deploy_tool(self, deployed: DeployedSystem) -> None:
        """Deploy a tool to the tool registry."""
        code = deployed.generated_code.code
        spec = deployed.specification

        # Execute code to get the function
        module_dict: Dict[str, Any] = {}
        exec(code, module_dict)

        tool_func = module_dict.get(spec.name)
        if not tool_func:
            raise ValueError(f"Tool function '{spec.name}' not found in generated code")

        # Register with AION tool system if available
        if self.kernel.tools:
            await self.kernel.tools.register_tool(
                name=spec.name,
                handler=tool_func,
                description=spec.description,
                parameters=[p.to_dict() for p in spec.parameters],
            )

    async def _deploy_workflow(self, deployed: DeployedSystem) -> None:
        """Deploy a workflow."""
        code = deployed.generated_code.code
        module_dict: Dict[str, Any] = {}
        exec(code, module_dict)

        register_func = module_dict.get("register_workflow")
        if register_func:
            workflow = register_func()
            if self.kernel.automation:
                await self.kernel.automation.register_workflow(workflow)

    async def _deploy_agent(self, deployed: DeployedSystem) -> None:
        """Deploy an agent configuration."""
        code = deployed.generated_code.code
        module_dict: Dict[str, Any] = {}
        exec(code, module_dict)

        register_func = module_dict.get("register_agent")
        if register_func:
            agent_config = register_func()
            if self.kernel.supervisor:
                await self.kernel.supervisor.register_agent_type(
                    deployed.specification.name, agent_config
                )

    async def _deploy_api(self, deployed: DeployedSystem) -> None:
        """Deploy API endpoints."""
        # API deployment would integrate with FastAPI app
        logger.info("API deployment registered", name=deployed.name)

    async def _deploy_integration(self, deployed: DeployedSystem) -> None:
        """Deploy an integration."""
        logger.info("Integration deployment registered", name=deployed.name)
