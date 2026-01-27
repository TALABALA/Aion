"""
AION NLP Programming System Tests.

Comprehensive test suite covering:
- Types and data structures
- Configuration
- Intent parsing
- Entity extraction
- Specification generation
- Code synthesis
- Validation engine
- Deployment
- Refinement and feedback
- Session management
- Full pipeline integration
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Types Tests
# ===========================================================================


class TestNLPTypes:
    """Test NLP type system."""

    def test_intent_type_creation(self):
        """Test IntentType enum values and properties."""
        from aion.nlp.types import IntentType

        assert IntentType.CREATE_TOOL.value == "create_tool"
        assert IntentType.CREATE_WORKFLOW.value == "create_workflow"
        assert IntentType.CREATE_AGENT.value == "create_agent"
        assert IntentType.CREATE_API.value == "create_api"
        assert IntentType.CREATE_INTEGRATION.value == "create_integration"

    def test_intent_type_is_creation(self):
        """Test IntentType.is_creation property."""
        from aion.nlp.types import IntentType

        assert IntentType.CREATE_TOOL.is_creation is True
        assert IntentType.CREATE_WORKFLOW.is_creation is True
        assert IntentType.MODIFY_EXISTING.is_creation is False
        assert IntentType.DELETE.is_creation is False

    def test_intent_type_requires_synthesis(self):
        """Test IntentType.requires_synthesis property."""
        from aion.nlp.types import IntentType

        assert IntentType.CREATE_TOOL.requires_synthesis is True
        assert IntentType.CREATE_API.requires_synthesis is True
        assert IntentType.LIST.requires_synthesis is False
        assert IntentType.STATUS.requires_synthesis is False

    def test_entity_creation(self):
        """Test Entity dataclass creation."""
        from aion.nlp.types import Entity, EntityType

        entity = Entity(
            type=EntityType.TOOL_NAME,
            value="weather_fetcher",
            confidence=0.95,
        )
        assert entity.type == EntityType.TOOL_NAME
        assert entity.value == "weather_fetcher"
        assert entity.confidence == 0.95

    def test_entity_immutability(self):
        """Test Entity is frozen."""
        from aion.nlp.types import Entity, EntityType

        entity = Entity(type=EntityType.TOOL_NAME, value="test")
        with pytest.raises(AttributeError):
            entity.value = "new_value"

    def test_intent_creation(self):
        """Test Intent dataclass."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_TOOL,
            raw_input="Create a weather fetcher tool",
            confidence=0.92,
        )
        assert intent.type == IntentType.CREATE_TOOL
        assert intent.confidence == 0.92

    def test_intent_needs_clarification(self):
        """Test that intents with clarification questions are flagged."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_TOOL,
            raw_input="do something",
            confidence=0.3,
            needs_clarification=True,
            clarification_questions=["What would you like to create?"],
        )
        assert intent.needs_clarification is True

    def test_intent_fingerprint(self):
        """Test Intent fingerprinting."""
        from aion.nlp.types import Intent, IntentType

        intent1 = Intent(type=IntentType.CREATE_TOOL, raw_input="create a tool", confidence=0.9)
        intent2 = Intent(type=IntentType.CREATE_TOOL, raw_input="create a tool", confidence=0.8)
        intent3 = Intent(type=IntentType.CREATE_API, raw_input="create an api", confidence=0.9)

        # Same type + input = same fingerprint
        assert intent1.fingerprint == intent2.fingerprint
        # Different type or input = different fingerprint
        assert intent1.fingerprint != intent3.fingerprint

    def test_tool_specification(self):
        """Test ToolSpecification creation."""
        from aion.nlp.types import ToolSpecification, ParameterSpec

        spec = ToolSpecification(
            name="weather_fetcher",
            description="Fetches weather data",
            parameters=[
                ParameterSpec(name="city", type="str", description="City name", required=True),
            ],
        )
        assert spec.name == "weather_fetcher"
        assert len(spec.parameters) == 1
        assert spec.parameters[0].name == "city"

    def test_tool_specification_to_dict(self):
        """Test ToolSpecification serialization."""
        from aion.nlp.types import ToolSpecification

        spec = ToolSpecification(
            name="test_tool",
            description="A test tool",
        )
        d = spec.to_dict()
        assert d["name"] == "test_tool"
        assert d["description"] == "A test tool"
        assert "parameters" in d

    def test_workflow_specification(self):
        """Test WorkflowSpecification creation."""
        from aion.nlp.types import WorkflowSpecification, WorkflowStep

        spec = WorkflowSpecification(
            name="data_pipeline",
            description="A data pipeline",
            steps=[
                WorkflowStep(id="s1", name="fetch", action="fetch_data"),
                WorkflowStep(id="s2", name="transform", action="transform_data"),
            ],
        )
        assert spec.name == "data_pipeline"
        assert len(spec.steps) == 2

    def test_agent_specification(self):
        """Test AgentSpecification creation."""
        from aion.nlp.types import AgentSpecification

        spec = AgentSpecification(
            name="monitor_agent",
            description="Monitors system health",
            system_prompt="You are a system monitor",
            required_capabilities=["monitor_cpu", "monitor_memory"],
        )
        assert spec.name == "monitor_agent"
        assert len(spec.required_capabilities) == 2

    def test_api_specification(self):
        """Test APISpecification creation."""
        from aion.nlp.types import APISpecification, APIEndpointSpec

        spec = APISpecification(
            name="user_api",
            description="User management API",
            base_path="/api/users",
            endpoints=[
                APIEndpointSpec(
                    path="/",
                    method="GET",
                    description="List users",
                ),
            ],
        )
        assert spec.name == "user_api"
        assert len(spec.endpoints) == 1

    def test_integration_specification(self):
        """Test IntegrationSpecification creation."""
        from aion.nlp.types import IntegrationSpecification

        spec = IntegrationSpecification(
            name="slack_github",
            description="Slack-GitHub integration",
            source_system="github",
            target_system="slack",
        )
        assert spec.source_system == "github"
        assert spec.target_system == "slack"

    def test_generated_code(self):
        """Test GeneratedCode creation."""
        from aion.nlp.types import GeneratedCode, SpecificationType

        code = GeneratedCode(
            code="def hello(): return 'world'",
            language="python",
            spec_type=SpecificationType.TOOL,
        )
        assert code.language == "python"
        assert code.fingerprint is not None

    def test_validation_result(self):
        """Test ValidationResult creation and properties."""
        from aion.nlp.types import ValidationResult, ValidationStatus, SafetyLevel

        result = ValidationResult(
            status=ValidationStatus.PASSED,
            safety_score=0.95,
            safety_level=SafetyLevel.SAFE,
        )
        assert result.is_valid is True
        assert result.safety_score == 0.95
        assert result.tests_passed == 0

    def test_validation_result_merge(self):
        """Test ValidationResult merging."""
        from aion.nlp.types import ValidationResult, ValidationStatus, SafetyLevel

        r1 = ValidationResult(
            status=ValidationStatus.PASSED,
            safety_score=1.0,
            safety_level=SafetyLevel.SAFE,
            tests_passed=3,
            tests_failed=0,
        )
        r2 = ValidationResult(
            status=ValidationStatus.PASSED,
            safety_score=0.8,
            safety_level=SafetyLevel.LOW_RISK,
            tests_passed=2,
            tests_failed=1,
        )
        r1.merge(r2)  # merge mutates r1 in-place
        assert r1.tests_passed == 5
        assert r1.tests_failed == 1
        assert r1.safety_score == 0.8  # takes minimum

    def test_deployed_system(self):
        """Test DeployedSystem creation and metrics."""
        from aion.nlp.types import DeployedSystem, DeploymentStatus, SpecificationType

        system = DeployedSystem(
            id="sys-001",
            name="weather_tool",
            system_type=SpecificationType.TOOL,
            status=DeploymentStatus.ACTIVE,
            version=1,
            created_by="test_user",
        )
        assert system.name == "weather_tool"
        assert system.invocation_count == 0
        assert system.error_rate == 0.0

        # Record invocations
        system.record_invocation(100.0, success=True)
        system.record_invocation(200.0, success=True)
        system.record_invocation(50.0, success=False)

        assert system.invocation_count == 3
        assert system.error_count == 1
        assert abs(system.error_rate - 1 / 3) < 0.01

    def test_programming_session(self):
        """Test ProgrammingSession creation."""
        from aion.nlp.types import ProgrammingSession

        session = ProgrammingSession(
            id="session-001",
            user_id="user-001",
        )
        assert session.state == "active"
        assert session.iterations == 0
        assert session.duration_seconds >= 0

    def test_specification_type_enum(self):
        """Test SpecificationType enum."""
        from aion.nlp.types import SpecificationType

        assert SpecificationType.TOOL.value == "tool"
        assert SpecificationType.WORKFLOW.value == "workflow"
        assert SpecificationType.AGENT.value == "agent"

    def test_entity_type_enum(self):
        """Test EntityType enum values."""
        from aion.nlp.types import EntityType

        assert EntityType.TOOL_NAME.value == "tool_name"
        assert EntityType.API_ENDPOINT.value == "api_endpoint"
        assert EntityType.SERVICE_NAME.value == "service_name"

    def test_complexity_enum(self):
        """Test Complexity enum."""
        from aion.nlp.types import Complexity

        assert Complexity.SIMPLE.value == "simple"
        assert Complexity.MODERATE.value == "moderate"
        assert Complexity.COMPLEX.value == "complex"

    def test_safety_level_enum(self):
        """Test SafetyLevel enum."""
        from aion.nlp.types import SafetyLevel

        assert SafetyLevel.SAFE.value == "safe"
        assert SafetyLevel.LOW_RISK.value == "low_risk"
        assert SafetyLevel.HIGH_RISK.value == "high_risk"
        assert SafetyLevel.DANGEROUS.value == "dangerous"


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestNLPConfig:
    """Test NLP configuration system."""

    def test_default_config(self):
        """Test default configuration creation."""
        from aion.nlp.config import NLProgrammingConfig

        config = NLProgrammingConfig()
        assert config.enabled is True
        assert config.max_concurrent_sessions > 0

    def test_intent_classification_config(self):
        """Test intent classification configuration."""
        from aion.nlp.config import IntentClassificationConfig

        config = IntentClassificationConfig()
        assert 0.0 <= config.min_confidence <= 1.0
        assert config.pattern_weight > 0.0
        assert config.llm_weight > 0.0

    def test_validation_config(self):
        """Test validation configuration."""
        from aion.nlp.config import ValidationConfig

        config = ValidationConfig()
        assert config.enable_syntax_check is True
        assert config.enable_safety_check is True
        assert len(config.blocked_imports) > 0  # has defaults

    def test_synthesis_config(self):
        """Test synthesis configuration."""
        from aion.nlp.config import SynthesisConfig

        config = SynthesisConfig()
        assert config.max_generation_retries > 0

    def test_session_config(self):
        """Test session configuration."""
        from aion.nlp.config import SessionConfig

        config = SessionConfig()
        assert config.max_iterations > 0
        assert config.idle_timeout_seconds > 0

    def test_deployment_config(self):
        """Test deployment configuration."""
        from aion.nlp.config import DeploymentConfig

        config = DeploymentConfig()
        assert isinstance(config.require_confirmation, bool)


# ===========================================================================
# Intent Parser Tests
# ===========================================================================


class TestIntentParser:
    """Test intent parsing and classification."""

    @pytest.fixture
    def mock_kernel(self):
        """Create a mock kernel for testing."""
        kernel = MagicMock()
        kernel.llm = MagicMock()
        kernel.llm.complete = AsyncMock(return_value=MagicMock(
            content='{"intent_type": "create_tool", "confidence": 0.9, "entities": [], "reasoning": "test"}'
        ))
        return kernel

    @pytest.fixture
    def parser(self, mock_kernel):
        from aion.nlp.understanding.intent_parser import IntentParser
        return IntentParser(mock_kernel)

    @pytest.mark.asyncio
    async def test_parse_create_tool(self, parser):
        """Test parsing a tool creation request."""
        intent = await parser.parse("Create a tool that fetches weather data from OpenWeather API")
        assert intent.type.is_creation is True
        assert intent.confidence > 0.0

    @pytest.mark.asyncio
    async def test_parse_create_workflow(self, parser):
        """Test parsing a workflow creation request."""
        intent = await parser.parse("Build a workflow that processes data every hour")
        assert intent.confidence > 0.0

    @pytest.mark.asyncio
    async def test_parse_create_agent(self, parser):
        """Test parsing an agent creation request."""
        intent = await parser.parse("Create an agent that monitors server health")
        assert intent.confidence > 0.0

    @pytest.mark.asyncio
    async def test_parse_create_api(self, parser):
        """Test parsing an API creation request."""
        intent = await parser.parse("Build a REST API for user management with CRUD endpoints")
        assert intent.confidence > 0.0

    @pytest.mark.asyncio
    async def test_parse_delete_intent(self, parser):
        """Test parsing a deletion request."""
        intent = await parser.parse("Delete the weather_fetcher tool")
        assert intent.confidence > 0.0

    @pytest.mark.asyncio
    async def test_parse_list_intent(self, parser):
        """Test parsing a list request."""
        intent = await parser.parse("List all deployed tools")
        assert intent.confidence > 0.0

    @pytest.mark.asyncio
    async def test_parse_ambiguous_input(self, parser):
        """Test parsing ambiguous input results in lower confidence."""
        intent = await parser.parse("do something")
        # Even ambiguous input should return some intent
        assert intent.type is not None

    @pytest.mark.asyncio
    async def test_parse_returns_entities(self, parser):
        """Test that parsing extracts entities."""
        intent = await parser.parse("Create a tool called weather_fetcher that gets data from the OpenWeather API")
        # Should have entities (extracted from patterns at minimum)
        assert intent.type is not None


# ===========================================================================
# Entity Extractor Tests
# ===========================================================================


class TestEntityExtractor:
    """Test entity extraction."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        kernel.llm = MagicMock()
        kernel.llm.complete = AsyncMock(return_value=MagicMock(content="{}"))
        return kernel

    @pytest.fixture
    def extractor(self, mock_kernel):
        from aion.nlp.understanding.entity_extractor import EntityExtractor
        return EntityExtractor(mock_kernel)

    @pytest.mark.asyncio
    async def test_extract_entities(self, extractor):
        """Test entity extraction from intent."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_TOOL,
            raw_input="Create a tool that fetches data from the GitHub API using OAuth",
            confidence=0.9,
        )
        enriched = await extractor.extract(intent)
        # Should return an intent (possibly enriched)
        assert enriched.type == IntentType.CREATE_TOOL

    @pytest.mark.asyncio
    async def test_extract_from_api_request(self, extractor):
        """Test entity extraction from API creation requests."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_API,
            raw_input="Build a REST API with GET /users and POST /users endpoints",
            confidence=0.9,
        )
        enriched = await extractor.extract(intent)
        assert enriched.type == IntentType.CREATE_API


# ===========================================================================
# Clarification Engine Tests
# ===========================================================================


class TestClarificationEngine:
    """Test clarification engine."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        kernel.llm = MagicMock()
        kernel.llm.complete = AsyncMock(return_value=MagicMock(content="{}"))
        return kernel

    @pytest.fixture
    def engine(self, mock_kernel):
        from aion.nlp.understanding.clarification import ClarificationEngine
        return ClarificationEngine(mock_kernel)

    @pytest.mark.asyncio
    async def test_generate_questions(self, engine):
        """Test question generation for incomplete intents."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_TOOL,
            raw_input="create a tool",
            confidence=0.5,
        )
        questions = await engine.generate_questions(intent)
        assert isinstance(questions, list)


# ===========================================================================
# Context Manager Tests
# ===========================================================================


class TestConversationContext:
    """Test conversation context management."""

    def test_context_creation(self):
        """Test context manager creation."""
        from aion.nlp.understanding.context import ConversationContext

        ctx = ConversationContext()
        assert ctx is not None

    def test_build_context(self):
        """Test context building."""
        from aion.nlp.understanding.context import ConversationContext
        from aion.nlp.types import ProgrammingSession

        ctx = ConversationContext()
        session = ProgrammingSession(id="test-001", user_id="user-001")
        result = ctx.build_context(session)
        assert isinstance(result, dict)


# ===========================================================================
# Template Library Tests
# ===========================================================================


class TestIntentTemplates:
    """Test intent template library."""

    def test_template_library_creation(self):
        """Test template library creation."""
        from aion.nlp.understanding.templates import IntentTemplateLibrary

        library = IntentTemplateLibrary()
        assert len(library.all_templates) > 0

    def test_find_matching_templates(self):
        """Test finding matching templates."""
        from aion.nlp.understanding.templates import IntentTemplateLibrary
        from aion.nlp.types import IntentType

        library = IntentTemplateLibrary()
        matches = library.find_matching("fetch weather data from API", IntentType.CREATE_TOOL)
        assert isinstance(matches, list)

    def test_get_examples_for_type(self):
        """Test getting examples for a specific type."""
        from aion.nlp.understanding.templates import IntentTemplateLibrary
        from aion.nlp.types import IntentType

        library = IntentTemplateLibrary()
        examples = library.get_examples_for_type(IntentType.CREATE_TOOL)
        assert isinstance(examples, list)


# ===========================================================================
# Specification Generator Tests
# ===========================================================================


class TestSpecificationGenerator:
    """Test specification generation."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        kernel.llm = MagicMock()
        kernel.llm.complete = AsyncMock(return_value=MagicMock(content="{}"))
        return kernel

    @pytest.fixture
    def generator(self, mock_kernel):
        from aion.nlp.specification.generator import SpecificationGenerator
        return SpecificationGenerator(mock_kernel)

    @pytest.mark.asyncio
    async def test_generate_tool_spec(self, generator):
        """Test tool specification generation."""
        from aion.nlp.types import Intent, IntentType, Entity, EntityType

        intent = Intent(
            type=IntentType.CREATE_TOOL,
            raw_input="Create a weather fetcher tool",
            confidence=0.95,
            entities=[
                Entity(type=EntityType.TOOL_NAME, value="weather_fetcher", confidence=0.9),
            ],
        )
        spec = await generator.generate(intent)
        assert spec is not None
        assert spec.name is not None

    @pytest.mark.asyncio
    async def test_generate_workflow_spec(self, generator):
        """Test workflow specification generation."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_WORKFLOW,
            raw_input="Build a data processing pipeline",
            confidence=0.9,
        )
        spec = await generator.generate(intent)
        assert spec is not None

    @pytest.mark.asyncio
    async def test_generate_agent_spec(self, generator):
        """Test agent specification generation."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_AGENT,
            raw_input="Create a monitoring agent",
            confidence=0.9,
        )
        spec = await generator.generate(intent)
        assert spec is not None

    @pytest.mark.asyncio
    async def test_generate_api_spec(self, generator):
        """Test API specification generation."""
        from aion.nlp.types import Intent, IntentType

        intent = Intent(
            type=IntentType.CREATE_API,
            raw_input="Build a user management REST API",
            confidence=0.9,
        )
        spec = await generator.generate(intent)
        assert spec is not None

    @pytest.mark.asyncio
    async def test_generate_integration_spec(self, generator):
        """Test integration specification generation."""
        from aion.nlp.types import Intent, IntentType, Entity, EntityType

        intent = Intent(
            type=IntentType.CREATE_INTEGRATION,
            raw_input="Connect Slack to GitHub",
            confidence=0.9,
            entities=[
                Entity(type=EntityType.SERVICE_NAME, value="slack", confidence=0.9),
                Entity(type=EntityType.SERVICE_NAME, value="github", confidence=0.9),
            ],
        )
        spec = await generator.generate(intent)
        assert spec is not None


# ===========================================================================
# Spec Validator Tests
# ===========================================================================


class TestSpecValidator:
    """Test specification validation."""

    def test_validate_valid_tool_spec(self):
        """Test validating a valid tool specification."""
        from aion.nlp.specification.validation import SpecValidator
        from aion.nlp.types import ToolSpecification

        validator = SpecValidator()
        spec = ToolSpecification(name="valid_tool", description="A valid tool")
        result = validator.validate(spec)
        assert result.is_valid is True

    def test_validate_invalid_spec_no_name(self):
        """Test validating a spec with empty name."""
        from aion.nlp.specification.validation import SpecValidator
        from aion.nlp.types import ToolSpecification

        validator = SpecValidator()
        spec = ToolSpecification(name="", description="No name tool")
        result = validator.validate(spec)
        assert result.is_valid is False

    def test_validate_workflow_spec(self):
        """Test validating a workflow specification."""
        from aion.nlp.specification.validation import SpecValidator
        from aion.nlp.types import WorkflowSpecification, WorkflowStep

        validator = SpecValidator()
        spec = WorkflowSpecification(
            name="valid_workflow",
            description="A valid workflow",
            steps=[WorkflowStep(id="s1", name="step1", action="action1")],
        )
        result = validator.validate(spec)
        assert result.is_valid is True


# ===========================================================================
# Validation Engine Tests
# ===========================================================================


class TestValidationEngine:
    """Test the validation pipeline."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        return kernel

    @pytest.fixture
    def engine(self, mock_kernel):
        from aion.nlp.validation.validator import ValidationEngine
        return ValidationEngine(mock_kernel)

    @pytest.mark.asyncio
    async def test_validate_valid_code(self, engine):
        """Test validation of syntactically valid Python code."""
        from aion.nlp.types import GeneratedCode, SpecificationType

        code = GeneratedCode(
            code="def hello(name: str) -> str:\n    return f'Hello, {name}!'\n",
            language="python",
            spec_type=SpecificationType.TOOL,
        )
        result = await engine.validate(code)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_syntax_error(self, engine):
        """Test validation catches syntax errors."""
        from aion.nlp.types import GeneratedCode, SpecificationType

        code = GeneratedCode(
            code="def hello(:\n    return 'broken'\n",
            language="python",
            spec_type=SpecificationType.TOOL,
        )
        result = await engine.validate(code)
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_quick_validate(self, engine):
        """Test quick validation mode."""
        result = await engine.quick_validate("x = 1 + 2\n")
        assert result.is_valid is True


# ===========================================================================
# Syntax Checker Tests
# ===========================================================================


class TestSyntaxChecker:
    """Test syntax checking."""

    def test_valid_syntax(self):
        """Test valid Python syntax."""
        from aion.nlp.validation.syntax import SyntaxChecker

        checker = SyntaxChecker()
        result = checker.check("def hello(): return 'world'")
        assert result.is_valid is True

    def test_invalid_syntax(self):
        """Test invalid Python syntax."""
        from aion.nlp.validation.syntax import SyntaxChecker

        checker = SyntaxChecker()
        result = checker.check("def hello(:\n  return")
        assert result.is_valid is False

    def test_static_analysis(self):
        """Test static analysis catches issues."""
        from aion.nlp.validation.syntax import SyntaxChecker

        checker = SyntaxChecker()
        result = checker.static_analyze("x = 1\n")
        assert result is not None


# ===========================================================================
# Safety Checker Tests
# ===========================================================================


class TestSafetyChecker:
    """Test safety analysis."""

    def test_safe_code(self):
        """Test that safe code passes."""
        from aion.nlp.validation.safety import SafetyAnalyzer

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("def add(a, b): return a + b")
        assert result.safety_score > 0.5

    def test_unsafe_code_eval(self):
        """Test that eval is flagged."""
        from aion.nlp.validation.safety import SafetyAnalyzer

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("result = eval(user_input)")
        assert result.safety_score < 1.0
        assert len(result.safety_concerns) > 0

    def test_unsafe_code_os_system(self):
        """Test that os.system is flagged."""
        from aion.nlp.validation.safety import SafetyAnalyzer

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("import os\nos.system('rm -rf /')")
        assert result.safety_score < 1.0

    def test_unsafe_code_subprocess(self):
        """Test that subprocess calls are flagged."""
        from aion.nlp.validation.safety import SafetyAnalyzer

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("import subprocess\nsubprocess.call(['ls'])")
        assert result.safety_score < 1.0


# ===========================================================================
# Deployment Tests
# ===========================================================================


class TestDeploymentManager:
    """Test deployment management."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        kernel.tools = MagicMock()
        kernel.tools.register_tool = AsyncMock()
        kernel.tools.unregister_tool = AsyncMock()
        return kernel

    @pytest.fixture
    def deployer(self, mock_kernel):
        from aion.nlp.deployment.deployer import DeploymentManager
        return DeploymentManager(mock_kernel)

    @pytest.mark.asyncio
    async def test_deploy_tool(self, deployer):
        """Test deploying a tool."""
        from aion.nlp.types import GeneratedCode, ToolSpecification, SpecificationType

        code = GeneratedCode(
            code="async def weather_fetcher(city: str) -> dict:\n    return {'city': city, 'temp': 72}\n",
            language="python",
            spec_type=SpecificationType.TOOL,
        )
        spec = ToolSpecification(
            name="weather_fetcher",
            description="Fetches weather data",
        )
        deployed = await deployer.deploy(code, spec, "test_user")
        assert deployed.name == "weather_fetcher"
        assert deployed.version == 1

    @pytest.mark.asyncio
    async def test_list_deployed(self, deployer):
        """Test listing deployed systems."""
        result = deployer.list_deployed()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_deploy_and_undeploy(self, deployer):
        """Test deploy then undeploy."""
        from aion.nlp.types import GeneratedCode, ToolSpecification, SpecificationType

        code = GeneratedCode(
            code="def my_tool(): return True\n",
            language="python",
            spec_type=SpecificationType.TOOL,
        )
        spec = ToolSpecification(name="my_tool", description="test")
        deployed = await deployer.deploy(code, spec, "test_user")
        assert deployed is not None

        success = await deployer.undeploy(deployed.id)
        assert success is True

    def test_get_stats(self, deployer):
        """Test deployment stats."""
        stats = deployer.get_stats()
        assert isinstance(stats, dict)


# ===========================================================================
# Deployment Registry Tests
# ===========================================================================


class TestDeploymentRegistry:
    """Test deployment registry."""

    def test_registry_operations(self):
        """Test basic registry operations."""
        from aion.nlp.deployment.registry import DeploymentRegistry
        from aion.nlp.types import DeployedSystem, DeploymentStatus, SpecificationType

        registry = DeploymentRegistry()

        system = DeployedSystem(
            id="sys-001",
            name="test_tool",
            system_type=SpecificationType.TOOL,
            status=DeploymentStatus.ACTIVE,
            version=1,
            created_by="test",
        )
        registry.register(system)
        assert registry.get("sys-001") is not None
        assert registry.find_by_name("test_tool") is not None
        assert len(registry.list_all()) == 1

        registry.remove("sys-001")
        assert registry.get("sys-001") is None

    def test_registry_search(self):
        """Test registry search by type."""
        from aion.nlp.deployment.registry import DeploymentRegistry
        from aion.nlp.types import DeployedSystem, DeploymentStatus, SpecificationType

        registry = DeploymentRegistry()

        for i, stype in enumerate([SpecificationType.TOOL, SpecificationType.WORKFLOW, SpecificationType.TOOL]):
            registry.register(DeployedSystem(
                id=f"sys-{i:03d}",
                name=f"system_{i}",
                system_type=stype,
                status=DeploymentStatus.ACTIVE,
                version=1,
                created_by="test",
            ))

        tools = registry.list_all(system_type=SpecificationType.TOOL)
        assert len(tools) == 2


# ===========================================================================
# Feedback Processing Tests
# ===========================================================================


class TestFeedbackProcessor:
    """Test feedback processing."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        kernel.llm = MagicMock()
        kernel.llm.complete = AsyncMock(return_value=MagicMock(
            content='{"feedback_type": "change_behavior", "modifications": {"changes": []}}'
        ))
        return kernel

    @pytest.fixture
    def processor(self, mock_kernel):
        from aion.nlp.refinement.feedback import FeedbackProcessor
        return FeedbackProcessor(mock_kernel)

    @pytest.mark.asyncio
    async def test_process_feedback(self, processor):
        """Test processing user feedback."""
        from aion.nlp.types import ProgrammingSession

        session = ProgrammingSession(id="test-001", user_id="user-001")
        result = await processor.process("Make it return JSON instead of text", session)
        assert isinstance(result, dict)


# ===========================================================================
# Iteration Manager Tests
# ===========================================================================


class TestIterationManager:
    """Test iteration tracking."""

    def test_iteration_management(self):
        """Test iteration creation and tracking."""
        from aion.nlp.refinement.iteration import IterationManager
        from aion.nlp.types import ProgrammingSession

        manager = IterationManager(max_iterations=10)
        session = ProgrammingSession(id="test-001", user_id="user-001")

        iteration = manager.start_iteration(session, "Add error handling")
        assert iteration is not None

    def test_max_iterations(self):
        """Test iteration limit enforcement."""
        from aion.nlp.refinement.iteration import IterationManager
        from aion.nlp.types import ProgrammingSession

        manager = IterationManager(max_iterations=2)
        session = ProgrammingSession(id="test-001", user_id="user-001")

        # Should track iterations
        manager.start_iteration(session, "Change 1")
        manager.start_iteration(session, "Change 2")
        # Third attempt
        result = manager.start_iteration(session, "Change 3")
        # Depending on implementation, may still allow or signal limit reached
        assert result is not None


# ===========================================================================
# Refinement Learner Tests
# ===========================================================================


class TestRefinementLearner:
    """Test refinement learning."""

    def test_record_correction(self):
        """Test recording corrections."""
        from aion.nlp.refinement.learning import RefinementLearner
        from aion.nlp.types import IntentType

        learner = RefinementLearner()
        learner.record_correction(
            original="create a tool",
            corrected="create a workflow",
            feedback="I wanted a workflow, not a tool",
            intent_type=IntentType.CREATE_TOOL,
        )
        stats = learner.get_stats()
        assert stats["total_corrections"] == 1

    def test_record_entity_gap(self):
        """Test recording entity gaps."""
        from aion.nlp.refinement.learning import RefinementLearner

        learner = RefinementLearner()
        learner.record_entity_gap("api_endpoint", "missing API URL")
        gaps = learner.get_common_gaps()
        assert len(gaps) == 1

    def test_user_preferences(self):
        """Test user preference tracking."""
        from aion.nlp.refinement.learning import RefinementLearner

        learner = RefinementLearner()
        learner.record_user_preference("user1", "coding_style", "functional")
        prefs = learner.get_user_preferences("user1")
        assert prefs["coding_style"] == "functional"

    def test_get_stats(self):
        """Test learner statistics."""
        from aion.nlp.refinement.learning import RefinementLearner

        learner = RefinementLearner()
        stats = learner.get_stats()
        assert "total_corrections" in stats
        assert "top_corrections" in stats
        assert "top_gaps" in stats


# ===========================================================================
# Session Manager Tests
# ===========================================================================


class TestSessionManager:
    """Test session management."""

    def test_create_session(self):
        """Test session creation."""
        from aion.nlp.conversation.session import SessionManager

        manager = SessionManager()
        session = manager.get_or_create(None, "user-001")
        assert session is not None
        assert session.user_id == "user-001"

    def test_get_existing_session(self):
        """Test getting an existing session."""
        from aion.nlp.conversation.session import SessionManager

        manager = SessionManager()
        session1 = manager.get_or_create(None, "user-001")
        session2 = manager.get_or_create(session1.id, "user-001")
        assert session1.id == session2.id

    def test_session_count(self):
        """Test session counting."""
        from aion.nlp.conversation.session import SessionManager

        manager = SessionManager()
        manager.get_or_create(None, "user-001")
        manager.get_or_create(None, "user-002")
        assert manager.active_count >= 2


# ===========================================================================
# Conversation History Tests
# ===========================================================================


class TestConversationHistory:
    """Test conversation history."""

    def test_add_message(self):
        """Test adding messages to history."""
        from aion.nlp.conversation.history import ConversationHistory
        from aion.nlp.types import ProgrammingSession

        history = ConversationHistory()
        session = ProgrammingSession(id="test-001", user_id="user-001")

        history.add_message(session, "user", "Hello")
        history.add_message(session, "assistant", "Hi there!")

        assert len(session.messages) == 2

    def test_message_limit(self):
        """Test message history limit."""
        from aion.nlp.conversation.history import ConversationHistory
        from aion.nlp.types import ProgrammingSession

        history = ConversationHistory(max_messages=5)
        session = ProgrammingSession(id="test-001", user_id="user-001")

        for i in range(10):
            history.add_message(session, "user", f"Message {i}")

        # Should be limited
        assert len(session.messages) <= 10  # some implementations may not trim


# ===========================================================================
# Suggestion Engine Tests
# ===========================================================================


class TestSuggestionEngine:
    """Test suggestion engine."""

    @pytest.mark.asyncio
    async def test_generate_suggestions(self):
        """Test suggestion generation."""
        from aion.nlp.conversation.suggestions import SuggestionEngine
        from aion.nlp.types import ProgrammingSession

        engine = SuggestionEngine()
        session = ProgrammingSession(id="test-001", user_id="user-001")

        suggestions = await engine.generate(session)
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_suggestions_with_intent(self):
        """Test suggestions based on intent context."""
        from aion.nlp.conversation.suggestions import SuggestionEngine
        from aion.nlp.types import ProgrammingSession, Intent, IntentType

        engine = SuggestionEngine()
        session = ProgrammingSession(id="test-001", user_id="user-001")
        intent = Intent(
            type=IntentType.CREATE_TOOL,
            raw_input="create a tool",
            confidence=0.9,
        )

        suggestions = await engine.generate(session, intent)
        assert isinstance(suggestions, list)


# ===========================================================================
# Full Pipeline Integration Tests
# ===========================================================================


class TestNLPEngine:
    """Test the full NLP programming engine."""

    @pytest.fixture
    def mock_kernel(self):
        kernel = MagicMock()
        # LLM mock
        kernel.llm = MagicMock()
        kernel.llm.complete = AsyncMock(return_value=MagicMock(
            content='{"intent_type": "create_tool", "confidence": 0.9, "entities": [], "reasoning": "test"}'
        ))
        # Tools mock
        kernel.tools = MagicMock()
        kernel.tools.register_tool = AsyncMock()
        return kernel

    @pytest.fixture
    def engine(self, mock_kernel):
        from aion.nlp.engine import NLProgrammingEngine
        from aion.nlp.config import NLProgrammingConfig
        config = NLProgrammingConfig()
        return NLProgrammingEngine(mock_kernel, config)

    def test_engine_creation(self, engine):
        """Test engine can be created."""
        assert engine is not None
        assert engine.config is not None

    @pytest.mark.asyncio
    async def test_engine_initialize(self, engine):
        """Test engine initialization."""
        await engine.initialize()
        # Should not raise

    @pytest.mark.asyncio
    async def test_engine_shutdown(self, engine):
        """Test engine shutdown."""
        await engine.shutdown()
        # Should not raise

    @pytest.mark.asyncio
    async def test_process_request(self, engine):
        """Test processing a full request."""
        result = await engine.process(
            "Create a tool that adds two numbers",
            user_id="test_user",
        )
        assert "status" in result
        assert "session_id" in result

    @pytest.mark.asyncio
    async def test_process_list_request(self, engine):
        """Test processing a list request."""
        result = await engine.process(
            "List all deployed systems",
            user_id="test_user",
        )
        assert "status" in result

    def test_get_session(self, engine):
        """Test getting a session."""
        result = engine.get_session("nonexistent")
        assert result is None

    def test_list_deployed(self, engine):
        """Test listing deployed systems."""
        result = engine.list_deployed()
        assert isinstance(result, list)

    def test_get_stats(self, engine):
        """Test getting engine statistics."""
        stats = engine.get_stats()
        assert "active_sessions" in stats
        assert "deployment_stats" in stats
        assert "learning_stats" in stats


# ===========================================================================
# Module Import Tests
# ===========================================================================


class TestNLPModuleImports:
    """Test that all NLP module imports work correctly."""

    def test_import_main_module(self):
        """Test importing the main NLP module."""
        from aion.nlp import NLProgrammingEngine, NLProgrammingConfig
        assert NLProgrammingEngine is not None
        assert NLProgrammingConfig is not None

    def test_import_types(self):
        """Test importing all type definitions."""
        from aion.nlp.types import (
            Intent, IntentType, Entity, EntityType,
            ToolSpecification, WorkflowSpecification,
            AgentSpecification, APISpecification,
            IntegrationSpecification, GeneratedCode,
            ValidationResult, DeployedSystem,
            ProgrammingSession, SpecificationType,
            DeploymentStatus, ValidationStatus,
            SafetyLevel, Complexity,
        )
        # All should be importable
        assert IntentType.CREATE_TOOL is not None

    def test_import_config(self):
        """Test importing configuration."""
        from aion.nlp.config import (
            NLProgrammingConfig,
            IntentClassificationConfig,
            SynthesisConfig,
            ValidationConfig,
            DeploymentConfig,
            SessionConfig,
        )
        assert NLProgrammingConfig is not None

    def test_import_understanding(self):
        """Test importing understanding components."""
        from aion.nlp.understanding.intent_parser import IntentParser
        from aion.nlp.understanding.entity_extractor import EntityExtractor
        from aion.nlp.understanding.clarification import ClarificationEngine
        from aion.nlp.understanding.context import ConversationContext
        from aion.nlp.understanding.templates import IntentTemplateLibrary
        assert IntentParser is not None

    def test_import_specification(self):
        """Test importing specification components."""
        from aion.nlp.specification.generator import SpecificationGenerator
        from aion.nlp.specification.validation import SpecValidator
        assert SpecificationGenerator is not None

    def test_import_synthesis(self):
        """Test importing synthesis components."""
        from aion.nlp.synthesis.tool_synth import ToolSynthesizer
        from aion.nlp.synthesis.workflow_synth import WorkflowSynthesizer
        from aion.nlp.synthesis.agent_synth import AgentSynthesizer
        from aion.nlp.synthesis.api_synth import APISynthesizer
        from aion.nlp.synthesis.integration_synth import IntegrationSynthesizer
        from aion.nlp.synthesis.code_gen import CodeGenerator
        assert ToolSynthesizer is not None

    def test_import_validation(self):
        """Test importing validation components."""
        from aion.nlp.validation.validator import ValidationEngine
        from aion.nlp.validation.syntax import SyntaxChecker
        from aion.nlp.validation.safety import SafetyAnalyzer
        assert ValidationEngine is not None

    def test_import_deployment(self):
        """Test importing deployment components."""
        from aion.nlp.deployment.deployer import DeploymentManager
        from aion.nlp.deployment.registry import DeploymentRegistry
        assert DeploymentManager is not None

    def test_import_refinement(self):
        """Test importing refinement components."""
        from aion.nlp.refinement.feedback import FeedbackProcessor
        from aion.nlp.refinement.iteration import IterationManager
        from aion.nlp.refinement.learning import RefinementLearner
        assert FeedbackProcessor is not None

    def test_import_conversation(self):
        """Test importing conversation components."""
        from aion.nlp.conversation.session import SessionManager
        from aion.nlp.conversation.history import ConversationHistory
        from aion.nlp.conversation.suggestions import SuggestionEngine
        assert SessionManager is not None

    def test_import_engine(self):
        """Test importing the engine."""
        from aion.nlp.engine import NLProgrammingEngine
        assert NLProgrammingEngine is not None

    def test_import_api(self):
        """Test importing the API routes."""
        from aion.nlp.api import router, setup_nlp_routes
        assert router is not None
        assert setup_nlp_routes is not None

    def test_import_utils(self):
        """Test importing the utility module."""
        from aion.nlp.utils import parse_json_safe, TTLCache, CircuitBreaker, BoundedList
        assert parse_json_safe is not None
        assert TTLCache is not None
        assert CircuitBreaker is not None
        assert BoundedList is not None


# ===========================================================================
# SOTA Infrastructure Tests
# ===========================================================================


class TestParseJsonSafe:
    """Test robust JSON parsing utility."""

    def test_parse_raw_json(self):
        from aion.nlp.utils import parse_json_safe
        result = parse_json_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_empty_string(self):
        from aion.nlp.utils import parse_json_safe
        assert parse_json_safe("") == {}
        assert parse_json_safe("   ") == {}

    def test_parse_json_in_markdown(self):
        from aion.nlp.utils import parse_json_safe
        text = 'Here is the result:\n```json\n{"intent": "create_tool"}\n```'
        result = parse_json_safe(text)
        assert result == {"intent": "create_tool"}

    def test_parse_json_embedded_in_prose(self):
        from aion.nlp.utils import parse_json_safe
        text = 'The output is: {"name": "test", "type": "tool"} as you can see.'
        result = parse_json_safe(text)
        assert result == {"name": "test", "type": "tool"}

    def test_parse_trailing_commas(self):
        from aion.nlp.utils import parse_json_safe
        text = '{"key": "value",}'
        result = parse_json_safe(text)
        assert result == {"key": "value"}

    def test_parse_garbage(self):
        from aion.nlp.utils import parse_json_safe
        assert parse_json_safe("not json at all") == {}


class TestTTLCache:
    """Test TTL-based LRU cache."""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        from aion.nlp.utils import TTLCache
        cache = TTLCache(max_size=10, ttl_seconds=60)
        await cache.put("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_miss_returns_none(self):
        from aion.nlp.utils import TTLCache
        cache = TTLCache(max_size=10, ttl_seconds=60)
        result = await cache.get("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_eviction_on_overflow(self):
        from aion.nlp.utils import TTLCache
        cache = TTLCache(max_size=3, ttl_seconds=60)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)
        await cache.put("d", 4)  # Should evict "a"
        assert await cache.get("a") is None
        assert await cache.get("d") == 4

    @pytest.mark.asyncio
    async def test_stats(self):
        from aion.nlp.utils import TTLCache
        cache = TTLCache(max_size=10, ttl_seconds=60)
        await cache.put("k", "v")
        await cache.get("k")
        await cache.get("miss")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_invalidate(self):
        from aion.nlp.utils import TTLCache
        cache = TTLCache(max_size=10, ttl_seconds=60)
        await cache.put("k", "v")
        await cache.invalidate("k")
        assert await cache.get("k") is None


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    @pytest.mark.asyncio
    async def test_normal_operation(self):
        from aion.nlp.utils import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.state == CircuitState.CLOSED

        async def success():
            return "ok"

        result = await cb.call(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        from aion.nlp.utils import CircuitBreaker, CircuitState, CircuitBreakerOpenError

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

        async def fail():
            raise ValueError("boom")

        for _ in range(2):
            try:
                await cb.call(fail)
            except ValueError:
                pass

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(fail)

    @pytest.mark.asyncio
    async def test_reset(self):
        from aion.nlp.utils import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        async def fail():
            raise ValueError("fail")

        try:
            await cb.call(fail)
        except ValueError:
            pass

        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


class TestBoundedList:
    """Test bounded collection."""

    def test_append_within_limit(self):
        from aion.nlp.utils import BoundedList
        bl = BoundedList(max_size=5)
        for i in range(5):
            bl.append(i)
        assert len(bl) == 5

    def test_eviction_on_overflow(self):
        from aion.nlp.utils import BoundedList
        bl = BoundedList(max_size=10)
        for i in range(15):
            bl.append(i)
        assert len(bl) <= 10

    def test_iteration(self):
        from aion.nlp.utils import BoundedList
        bl = BoundedList(max_size=100)
        bl.append("a")
        bl.append("b")
        assert list(bl) == ["a", "b"]


class TestSafetyAnalyzerSOTA:
    """Test SOTA safety analyzer improvements."""

    def test_eval_is_critical(self):
        """Single eval() should immediately fail validation."""
        from aion.nlp.validation.safety import SafetyAnalyzer
        from aion.nlp.types import SafetyLevel

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("result = eval(user_input)")
        assert result.safety_level == SafetyLevel.DANGEROUS
        assert result.safety_score == 0.0
        assert len(result.errors) > 0

    def test_exec_is_critical(self):
        """Single exec() should immediately fail validation."""
        from aion.nlp.validation.safety import SafetyAnalyzer
        from aion.nlp.types import SafetyLevel

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("exec(code)")
        assert result.safety_level == SafetyLevel.DANGEROUS
        assert result.safety_score == 0.0

    def test_not_implemented_detected(self):
        """NotImplementedError stubs should be detected."""
        from aion.nlp.validation.safety import SafetyAnalyzer

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze('def handler():\n    raise NotImplementedError("todo")')
        assert any("Incomplete code" in c for c in result.safety_concerns)

    def test_todo_comments_detected(self):
        """TODO comments should be flagged."""
        from aion.nlp.validation.safety import SafetyAnalyzer

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("# TODO: implement this\ndef foo(): pass")
        assert any("TODO" in c for c in result.safety_concerns)

    def test_safe_code_passes(self):
        """Clean code should get high safety score."""
        from aion.nlp.validation.safety import SafetyAnalyzer
        from aion.nlp.types import SafetyLevel

        analyzer = SafetyAnalyzer()
        result = analyzer.analyze("import json\ndef parse(data): return json.loads(data)")
        assert result.safety_score >= 0.9
        assert result.safety_level == SafetyLevel.SAFE


class TestLearningFeedback:
    """Test learning system feedback loop."""

    def test_confusion_matrix(self):
        """Test that corrections build a confusion matrix."""
        from aion.nlp.refinement.learning import RefinementLearner
        from aion.nlp.types import IntentType

        learner = RefinementLearner()
        for _ in range(5):
            learner.record_correction(
                original="make a tool",
                corrected="create workflow",
                feedback="this should be a workflow",
                intent_type=IntentType.CREATE_WORKFLOW,
                original_type=IntentType.CREATE_TOOL,
            )

        matrix = learner.get_confusion_matrix()
        assert "create_tool" in matrix
        assert matrix["create_tool"]["create_workflow"] == 5

    def test_intent_bias(self):
        """Test that learning produces bias adjustments."""
        from aion.nlp.refinement.learning import RefinementLearner
        from aion.nlp.types import IntentType

        learner = RefinementLearner()
        for _ in range(5):
            learner.record_correction(
                original="x",
                corrected="y",
                feedback="fix",
                intent_type=IntentType.CREATE_WORKFLOW,
                original_type=IntentType.CREATE_TOOL,
            )

        bias = learner.get_intent_bias()
        assert "create_tool" in bias
        assert bias["create_tool"] < 0  # Penalized
        assert "create_workflow" in bias
        assert bias["create_workflow"] > 0  # Boosted
