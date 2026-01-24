"""
Skill Library System

Implements skill acquisition, storage, and transfer for agents,
enabling reuse of learned behaviors across tasks.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class SkillStatus(Enum):
    """Status of a skill."""

    LEARNING = "learning"
    AVAILABLE = "available"
    DEPRECATED = "deprecated"


class SkillComplexity(Enum):
    """Complexity levels of skills."""

    PRIMITIVE = "primitive"  # Basic, atomic skills
    COMPOSITE = "composite"  # Composed of other skills
    ABSTRACT = "abstract"  # High-level, goal-oriented


@dataclass
class SkillTemplate:
    """Template for a skill that can be instantiated."""

    id: str
    name: str
    description: str
    category: str
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    prompt_template: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
        }


@dataclass
class Skill:
    """A learned skill."""

    id: str
    name: str
    description: str
    template_id: Optional[str] = None
    complexity: SkillComplexity = SkillComplexity.PRIMITIVE
    status: SkillStatus = SkillStatus.LEARNING

    # Composition
    sub_skills: list[str] = field(default_factory=list)  # For composite skills
    parameters: dict[str, Any] = field(default_factory=dict)

    # Execution
    action_sequence: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""

    # Learning
    success_count: int = 0
    failure_count: int = 0
    proficiency: float = 0.0  # 0-1
    avg_execution_time: float = 0.0
    last_used: Optional[datetime] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def update_proficiency(self, success: bool, execution_time: float) -> None:
        """Update proficiency after execution."""
        if success:
            self.success_count += 1
            # Increase proficiency
            self.proficiency = min(1.0, self.proficiency + 0.1 * (1 - self.proficiency))
        else:
            self.failure_count += 1
            # Decrease proficiency
            self.proficiency = max(0.0, self.proficiency - 0.05)

        # Update execution time
        alpha = 0.2
        self.avg_execution_time = alpha * execution_time + (1 - alpha) * self.avg_execution_time

        self.last_used = datetime.now()
        self.updated_at = datetime.now()

        # Update status based on proficiency
        if self.proficiency >= 0.8 and self.success_count >= 5:
            self.status = SkillStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template_id": self.template_id,
            "complexity": self.complexity.value,
            "status": self.status.value,
            "sub_skills": self.sub_skills,
            "parameters": self.parameters,
            "action_sequence": self.action_sequence,
            "prompt": self.prompt,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "proficiency": self.proficiency,
            "success_rate": self.success_rate,
            "avg_execution_time": self.avg_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            template_id=data.get("template_id"),
            complexity=SkillComplexity(data.get("complexity", "primitive")),
            status=SkillStatus(data.get("status", "learning")),
            sub_skills=data.get("sub_skills", []),
            parameters=data.get("parameters", {}),
            action_sequence=data.get("action_sequence", []),
            prompt=data.get("prompt", ""),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            proficiency=data.get("proficiency", 0.0),
            avg_execution_time=data.get("avg_execution_time", 0.0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass
class SkillExecution:
    """Record of a skill execution."""

    id: str
    skill_id: str
    agent_id: str
    input_context: dict[str, Any]
    output: Optional[str] = None
    success: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None
    feedback: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "agent_id": self.agent_id,
            "input_context": self.input_context,
            "output": self.output,
            "success": self.success,
            "execution_time": self.execution_time,
            "error": self.error,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat(),
        }


# Type for skill execution function
ExecuteFn = Callable[[Skill, dict[str, Any]], Awaitable[tuple[str, bool]]]


class SkillLibrary:
    """
    Skill library for agents.

    Features:
    - Skill storage and retrieval
    - Skill composition
    - Proficiency tracking
    - Skill transfer between agents
    - Template-based skill creation
    - Skill recommendation
    """

    def __init__(
        self,
        agent_id: str,
        execute_fn: Optional[ExecuteFn] = None,
    ):
        self.agent_id = agent_id
        self.execute_fn = execute_fn

        # Storage
        self._skills: dict[str, Skill] = {}
        self._templates: dict[str, SkillTemplate] = {}
        self._executions: list[SkillExecution] = []

        # Indexes
        self._category_index: dict[str, set[str]] = {}
        self._tag_index: dict[str, set[str]] = {}

        # Counters
        self._skill_counter = 0
        self._execution_counter = 0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize skill library."""
        # Add default skill templates
        self._add_default_templates()

        self._initialized = True
        logger.info("skill_library_initialized", agent_id=self.agent_id)

    async def shutdown(self) -> None:
        """Shutdown skill library."""
        self._initialized = False
        logger.info("skill_library_shutdown", agent_id=self.agent_id)

    def _add_default_templates(self) -> None:
        """Add default skill templates."""
        templates = [
            SkillTemplate(
                id="template-research",
                name="Research",
                description="Research a topic thoroughly",
                category="knowledge",
                parameters={"topic": "string", "depth": "int"},
                preconditions=["topic_defined"],
                postconditions=["research_complete"],
                prompt_template="Research the following topic: {topic}. Depth: {depth}",
            ),
            SkillTemplate(
                id="template-code",
                name="Code Writing",
                description="Write code for a task",
                category="development",
                parameters={"language": "string", "task": "string"},
                preconditions=["requirements_clear"],
                postconditions=["code_written"],
                prompt_template="Write {language} code to: {task}",
            ),
            SkillTemplate(
                id="template-analysis",
                name="Analysis",
                description="Analyze data or information",
                category="analysis",
                parameters={"subject": "string", "method": "string"},
                preconditions=["data_available"],
                postconditions=["analysis_complete"],
                prompt_template="Analyze {subject} using {method} approach",
            ),
            SkillTemplate(
                id="template-writing",
                name="Content Writing",
                description="Write content",
                category="communication",
                parameters={"type": "string", "topic": "string", "audience": "string"},
                preconditions=["topic_defined"],
                postconditions=["content_written"],
                prompt_template="Write a {type} about {topic} for {audience}",
            ),
            SkillTemplate(
                id="template-planning",
                name="Planning",
                description="Create a plan",
                category="planning",
                parameters={"goal": "string", "constraints": "list"},
                preconditions=["goal_defined"],
                postconditions=["plan_created"],
                prompt_template="Create a plan to: {goal}. Constraints: {constraints}",
            ),
        ]

        for template in templates:
            self._templates[template.id] = template

    async def learn_skill(
        self,
        name: str,
        description: str,
        action_sequence: Optional[list[dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        template_id: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Skill:
        """Learn a new skill."""
        self._skill_counter += 1
        skill_id = f"skill-{self.agent_id}-{self._skill_counter}"

        # Determine complexity
        if action_sequence and len(action_sequence) > 3:
            complexity = SkillComplexity.COMPOSITE
        else:
            complexity = SkillComplexity.PRIMITIVE

        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            template_id=template_id,
            complexity=complexity,
            action_sequence=action_sequence or [],
            prompt=prompt or "",
            parameters=parameters or {},
            tags=tags or [],
        )

        # Get category from template
        category = "general"
        if template_id and template_id in self._templates:
            category = self._templates[template_id].category

        # Store skill
        self._skills[skill_id] = skill

        # Update indexes
        if category not in self._category_index:
            self._category_index[category] = set()
        self._category_index[category].add(skill_id)

        for tag in skill.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(skill_id)

        logger.info("skill_learned", skill_id=skill_id, name=name)

        return skill

    async def execute_skill(
        self,
        skill_id: str,
        context: dict[str, Any],
    ) -> SkillExecution:
        """Execute a skill."""
        skill = self._skills.get(skill_id)
        if not skill:
            raise ValueError(f"Skill not found: {skill_id}")

        self._execution_counter += 1
        execution = SkillExecution(
            id=f"exec-{self._execution_counter}",
            skill_id=skill_id,
            agent_id=self.agent_id,
            input_context=context,
        )

        start_time = datetime.now()

        try:
            if self.execute_fn:
                output, success = await self.execute_fn(skill, context)
                execution.output = output
                execution.success = success
            else:
                # Fallback: simulate execution
                execution.output = f"Executed {skill.name} with context: {context}"
                execution.success = True

        except Exception as e:
            execution.error = str(e)
            execution.success = False
            logger.error("skill_execution_error", skill_id=skill_id, error=str(e))

        execution.execution_time = (datetime.now() - start_time).total_seconds()

        # Update skill proficiency
        skill.update_proficiency(execution.success, execution.execution_time)

        self._executions.append(execution)

        logger.debug(
            "skill_executed",
            skill_id=skill_id,
            success=execution.success,
            time=execution.execution_time,
        )

        return execution

    async def compose_skill(
        self,
        name: str,
        description: str,
        sub_skill_ids: list[str],
        tags: Optional[list[str]] = None,
    ) -> Skill:
        """Compose a new skill from existing skills."""
        # Verify sub-skills exist
        for sid in sub_skill_ids:
            if sid not in self._skills:
                raise ValueError(f"Sub-skill not found: {sid}")

        self._skill_counter += 1
        skill_id = f"skill-{self.agent_id}-{self._skill_counter}"

        # Build action sequence from sub-skills
        action_sequence = []
        for sid in sub_skill_ids:
            sub_skill = self._skills[sid]
            action_sequence.append({
                "type": "execute_skill",
                "skill_id": sid,
                "skill_name": sub_skill.name,
            })

        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            complexity=SkillComplexity.COMPOSITE,
            sub_skills=sub_skill_ids,
            action_sequence=action_sequence,
            tags=tags or [],
        )

        # Inherit proficiency from sub-skills
        if sub_skill_ids:
            avg_proficiency = sum(
                self._skills[sid].proficiency for sid in sub_skill_ids
            ) / len(sub_skill_ids)
            skill.proficiency = avg_proficiency * 0.5  # Start at half

        self._skills[skill_id] = skill

        # Update indexes
        if "composite" not in self._category_index:
            self._category_index["composite"] = set()
        self._category_index["composite"].add(skill_id)

        logger.info("skill_composed", skill_id=skill_id, sub_skills=len(sub_skill_ids))

        return skill

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def find_skills(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_proficiency: float = 0.0,
        status: Optional[SkillStatus] = None,
    ) -> list[Skill]:
        """Find skills matching criteria."""
        candidates = set(self._skills.keys())

        # Filter by category
        if category and category in self._category_index:
            candidates &= self._category_index[category]

        # Filter by tags
        if tags:
            for tag in tags:
                if tag in self._tag_index:
                    candidates &= self._tag_index[tag]

        # Get skills and filter
        results = []
        for skill_id in candidates:
            skill = self._skills[skill_id]

            # Filter by proficiency
            if skill.proficiency < min_proficiency:
                continue

            # Filter by status
            if status and skill.status != status:
                continue

            # Filter by query (simple text match)
            if query:
                query_lower = query.lower()
                if (query_lower not in skill.name.lower() and
                    query_lower not in skill.description.lower()):
                    continue

            results.append(skill)

        # Sort by proficiency
        results.sort(key=lambda s: s.proficiency, reverse=True)

        return results

    def recommend_skill(
        self,
        task_description: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[Skill]:
        """Recommend the best skill for a task."""
        # Simple keyword matching
        task_lower = task_description.lower()

        best_skill = None
        best_score = 0.0

        for skill in self._skills.values():
            if skill.status != SkillStatus.AVAILABLE:
                continue

            score = 0.0

            # Name match
            if skill.name.lower() in task_lower:
                score += 0.5

            # Description match
            desc_words = skill.description.lower().split()
            matching_words = sum(1 for w in desc_words if w in task_lower)
            score += matching_words * 0.1

            # Tag match
            for tag in skill.tags:
                if tag.lower() in task_lower:
                    score += 0.2

            # Proficiency bonus
            score += skill.proficiency * 0.3

            if score > best_score:
                best_score = score
                best_skill = skill

        return best_skill

    async def transfer_skill(
        self,
        skill_id: str,
        target_library: "SkillLibrary",
    ) -> Optional[Skill]:
        """Transfer a skill to another agent's library."""
        source_skill = self._skills.get(skill_id)
        if not source_skill:
            return None

        # Create copy with new ID
        transferred = await target_library.learn_skill(
            name=source_skill.name,
            description=source_skill.description,
            action_sequence=source_skill.action_sequence.copy(),
            prompt=source_skill.prompt,
            template_id=source_skill.template_id,
            parameters=source_skill.parameters.copy(),
            tags=source_skill.tags.copy(),
        )

        # Transfer starts with lower proficiency
        transferred.proficiency = source_skill.proficiency * 0.5

        logger.info(
            "skill_transferred",
            skill_id=skill_id,
            from_agent=self.agent_id,
            to_agent=target_library.agent_id,
        )

        return transferred

    def get_skill_from_template(
        self,
        template_id: str,
        name: str,
        parameters: dict[str, Any],
    ) -> Skill:
        """Create a skill instance from a template."""
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Fill in prompt template
        prompt = template.prompt_template.format(**parameters)

        self._skill_counter += 1
        skill_id = f"skill-{self.agent_id}-{self._skill_counter}"

        skill = Skill(
            id=skill_id,
            name=name,
            description=template.description,
            template_id=template_id,
            complexity=SkillComplexity.PRIMITIVE,
            prompt=prompt,
            parameters=parameters,
            tags=[template.category],
        )

        self._skills[skill_id] = skill

        # Update category index
        if template.category not in self._category_index:
            self._category_index[template.category] = set()
        self._category_index[template.category].add(skill_id)

        return skill

    def get_templates(self, category: Optional[str] = None) -> list[SkillTemplate]:
        """Get available skill templates."""
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def get_stats(self) -> dict[str, Any]:
        """Get skill library statistics."""
        available_skills = [s for s in self._skills.values() if s.status == SkillStatus.AVAILABLE]

        return {
            "agent_id": self.agent_id,
            "total_skills": len(self._skills),
            "available_skills": len(available_skills),
            "templates": len(self._templates),
            "total_executions": len(self._executions),
            "categories": list(self._category_index.keys()),
            "avg_proficiency": sum(s.proficiency for s in self._skills.values()) / max(1, len(self._skills)),
            "skill_summary": [
                {"name": s.name, "proficiency": s.proficiency, "uses": s.success_count + s.failure_count}
                for s in sorted(self._skills.values(), key=lambda x: x.proficiency, reverse=True)[:5]
            ],
        }
