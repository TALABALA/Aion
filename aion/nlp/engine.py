"""
AION Natural Language Programming Engine - Main Orchestrator.

The central coordinator that manages the full NLP programming pipeline:
1. Intent Understanding -> 2. Specification Generation -> 3. Code Synthesis
-> 4. Validation -> 5. Deployment -> 6. Iterative Refinement

SOTA features:
- Async-safe with per-session locking
- LLM call caching with TTL
- Circuit breaker for LLM resilience
- Learning feedback loop into intent classifier
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.nlp.config import NLProgrammingConfig
from aion.nlp.types import (
    DeployedSystem,
    GeneratedCode,
    Intent,
    IntentType,
    ProgrammingSession,
    Specification,
    SpecificationType,
    ValidationResult,
)
from aion.nlp.understanding.intent_parser import IntentParser
from aion.nlp.understanding.entity_extractor import EntityExtractor
from aion.nlp.understanding.clarification import ClarificationEngine
from aion.nlp.understanding.context import ConversationContext
from aion.nlp.specification.generator import SpecificationGenerator
from aion.nlp.specification.validation import SpecValidator
from aion.nlp.synthesis.tool_synth import ToolSynthesizer
from aion.nlp.synthesis.workflow_synth import WorkflowSynthesizer
from aion.nlp.synthesis.agent_synth import AgentSynthesizer
from aion.nlp.synthesis.api_synth import APISynthesizer
from aion.nlp.synthesis.integration_synth import IntegrationSynthesizer
from aion.nlp.synthesis.code_gen import CodeGenerator
from aion.nlp.validation.validator import ValidationEngine
from aion.nlp.deployment.deployer import DeploymentManager
from aion.nlp.refinement.feedback import FeedbackProcessor
from aion.nlp.refinement.iteration import IterationManager
from aion.nlp.refinement.learning import RefinementLearner
from aion.nlp.conversation.session import SessionManager
from aion.nlp.conversation.history import ConversationHistory
from aion.nlp.conversation.suggestions import SuggestionEngine
from aion.nlp.utils import TTLCache, CircuitBreaker, CircuitBreakerOpenError

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class NLProgrammingEngine:
    """
    Main engine for natural language programming.

    Orchestrates the full pipeline from user input to deployed system:
    1. Parse intent from natural language
    2. Extract entities and enrich understanding
    3. Handle clarification if needed
    4. Generate formal specification
    5. Synthesize code
    6. Validate code
    7. Deploy to AION
    8. Handle feedback and iteration

    SOTA: Includes caching, circuit breaking, async safety,
    and learning feedback loops.
    """

    def __init__(self, kernel: AIONKernel, config: Optional[NLProgrammingConfig] = None):
        self.kernel = kernel
        self.config = config or NLProgrammingConfig()

        # Understanding pipeline
        self.intent_parser = IntentParser(kernel, config)
        self.entity_extractor = EntityExtractor(kernel, config)
        self.clarification = ClarificationEngine(kernel)
        self.context = ConversationContext(
            max_context_messages=self.config.session.context_window_messages,
        )

        # Specification
        self.spec_generator = SpecificationGenerator(kernel)
        self.spec_validator = SpecValidator()

        # Synthesis (type -> synthesizer mapping)
        self.tool_synth = ToolSynthesizer(kernel, config.synthesis if config else None)
        self.workflow_synth = WorkflowSynthesizer(kernel, config.synthesis if config else None)
        self.agent_synth = AgentSynthesizer(kernel, config.synthesis if config else None)
        self.api_synth = APISynthesizer(kernel, config.synthesis if config else None)
        self.integration_synth = IntegrationSynthesizer(kernel, config.synthesis if config else None)
        self.code_gen = CodeGenerator(kernel, config.synthesis if config else None)

        # Validation
        self.validator = ValidationEngine(
            kernel,
            config.validation if config else None,
        )

        # Deployment
        self.deployer = DeploymentManager(kernel)

        # Refinement
        self.feedback_processor = FeedbackProcessor(kernel)
        self.iteration_manager = IterationManager(
            max_iterations=self.config.session.max_iterations,
        )
        self.learner = RefinementLearner()

        # Session management
        self.sessions = SessionManager(
            max_sessions=self.config.max_concurrent_sessions,
            idle_timeout_seconds=self.config.session.idle_timeout_seconds,
        )
        self.history = ConversationHistory(
            max_messages=self.config.session.max_messages,
        )
        self.suggestions = SuggestionEngine(kernel=kernel)

        # Concurrency: per-session locks to prevent races
        self._session_locks: Dict[str, asyncio.Lock] = {}

        # Caching for LLM classification results
        self._intent_cache: Optional[TTLCache] = None
        if self.config.enable_caching:
            self._intent_cache = TTLCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )

        # Circuit breaker for LLM calls
        self._llm_circuit = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_recovery_seconds,
        )

        # Lifecycle state
        self._initialized = False
        self._request_count = 0
        self._total_processing_ms = 0.0

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the NLP programming engine and warm subsystems."""
        if self._initialized:
            return

        logger.info("NLP Programming Engine initializing...")

        # Warm the intent cache with common patterns
        if self._intent_cache:
            logger.debug("Intent cache initialized", max_size=self.config.cache_max_size)

        self._initialized = True
        logger.info("NLP Programming Engine initialized")

    async def shutdown(self) -> None:
        """Gracefully shutdown the NLP programming engine."""
        logger.info("NLP Programming Engine shutting down...")

        # Clear caches
        if self._intent_cache:
            await self._intent_cache.clear()

        # Mark all active sessions as abandoned
        for sid in list(self._session_locks.keys()):
            session = self.sessions.get(sid)
            if session and session.state == "active":
                session.state = "abandoned"

        # Clear session locks
        self._session_locks.clear()

        self._initialized = False
        logger.info("NLP Programming Engine shutdown complete",
                     total_requests=self._request_count)

    # =========================================================================
    # Main API
    # =========================================================================

    async def process(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: str = "",
    ) -> Dict[str, Any]:
        """
        Process a natural language programming request.

        Thread-safe per session via async locks.
        """
        start = time.monotonic()

        # Get or create session (async-safe with lock)
        session = await self.sessions.get_or_create(session_id, user_id)

        # Acquire per-session lock for concurrency safety
        lock = self._get_session_lock(session.id)
        async with lock:
            # Record user message
            self.history.add_message(session, "user", user_input)

            try:
                # Capture previous intent type for learning feedback
                previous_intent_type = (
                    session.current_intent.type if session.current_intent else None
                )

                # Phase 1: Parse intent (with caching + circuit breaker)
                ctx = self.context.build_context(session)
                intent = await self._cached_parse_intent(user_input, ctx)

                # Phase 2: Deep entity extraction
                intent = await self.entity_extractor.extract(intent)
                session.current_intent = intent
                session.intent_history.append(intent)

                # Record type-change correction for learning feedback loop
                if (
                    previous_intent_type is not None
                    and previous_intent_type != intent.type
                ):
                    self.learner.record_correction(
                        original=user_input,
                        corrected=user_input,
                        feedback="Intent type changed on follow-up",
                        intent_type=intent.type,
                        original_type=previous_intent_type,
                    )

                # Phase 3: Check if clarification needed
                if intent.needs_clarification and self.config.deployment.require_confirmation:
                    return self._respond_clarification(session, intent)

                # Phase 4: Route based on intent type
                if intent.type.requires_synthesis:
                    return await self._handle_creation(session, intent)
                elif intent.type == IntentType.MODIFY_EXISTING:
                    return await self._handle_modification(session, intent)
                elif intent.type == IntentType.DELETE:
                    return await self._handle_deletion(session, intent)
                elif intent.type == IntentType.DEPLOY:
                    return await self.confirm_deploy(session.id)
                elif intent.type in (IntentType.LIST, IntentType.STATUS):
                    return self._handle_query(session, intent)
                elif intent.type in (IntentType.EXPLAIN, IntentType.DEBUG):
                    return await self._handle_explain(session, intent)
                elif intent.type == IntentType.TEST:
                    return await self._handle_test(session, intent)
                elif intent.type == IntentType.ROLLBACK:
                    return await self._handle_rollback(session, intent)
                else:
                    return await self._handle_creation(session, intent)

            except Exception as e:
                logger.error("NLP processing failed", error=str(e), session_id=session.id)
                return self._respond_error(session, f"Processing failed: {e}")

            finally:
                elapsed_ms = (time.monotonic() - start) * 1000
                self._request_count += 1
                self._total_processing_ms += elapsed_ms
                logger.info(
                    "NLP request processed",
                    session_id=session.id,
                    elapsed_ms=round(elapsed_ms, 1),
                    request_count=self._request_count,
                )

    async def confirm_deploy(
        self,
        session_id: str,
        confirmed: bool = True,
    ) -> Dict[str, Any]:
        """Confirm or cancel deployment."""
        session = self.sessions.get(session_id)
        if not session:
            return {"status": "error", "error": "Session not found"}

        if not confirmed:
            self.history.add_message(
                session, "assistant",
                "Deployment cancelled. Would you like to make changes?",
            )
            return {
                "status": "cancelled",
                "session_id": session.id,
                "message": "Deployment cancelled. Would you like to make changes?",
            }

        if not session.current_code or not session.current_spec:
            return {
                "status": "error",
                "session_id": session.id,
                "error": "No code to deploy. Please describe what you want to build.",
            }

        try:
            deployed = await self.deployer.deploy(
                session.current_code,
                session.current_spec,
                session.user_id,
            )
            session.referenced_systems.append(deployed.id)

            msg = f"Successfully deployed '{deployed.name}' (v{deployed.version})!"
            self.history.add_message(session, "assistant", msg)

            suggestions = await self.suggestions.generate(session)

            return {
                "status": "deployed",
                "session_id": session.id,
                "system_id": deployed.id,
                "name": deployed.name,
                "type": deployed.system_type.value,
                "version": deployed.version,
                "message": msg,
                "suggestions": suggestions,
            }

        except Exception as e:
            return self._respond_error(session, f"Deployment failed: {e}")

    async def refine(
        self,
        session_id: str,
        feedback: str,
    ) -> Dict[str, Any]:
        """Refine the current system based on user feedback."""
        session = self.sessions.get(session_id)
        if not session:
            return {"status": "error", "error": "Session not found"}

        # Start iteration
        iteration = self.iteration_manager.start_iteration(session, feedback)
        self.history.add_message(session, "user", feedback, iteration=iteration)

        # Process feedback
        result = await self.feedback_processor.process(feedback, session)

        if result.get("requires_regeneration", True) and session.current_spec:
            # Apply modifications to spec
            session.current_spec = await self.feedback_processor.apply_to_spec(
                session.current_spec, result["modifications"]
            )

            # Re-synthesize
            code = await self._synthesize(session.current_intent.type, session.current_spec)
            session.current_code = code

            # Re-validate
            validation = await self.validator.validate(code)
            session.current_validation = validation

            # Record iteration result
            self.iteration_manager.record_result(
                session, validation,
                changes=[c.get("description", "") for c in result["modifications"].get("changes", [])],
            )

            # Learn from correction with confusion tracking
            if session.current_intent:
                self.learner.record_correction(
                    original=session.current_intent.raw_input,
                    corrected=feedback,
                    feedback=feedback,
                    intent_type=session.current_intent.type,
                    original_type=session.current_intent.type,
                )

            return await self._respond_ready_to_deploy(
                session, session.current_spec, code, validation,
                intent=session.current_intent,
            )

        return await self.process(feedback, session_id=session_id, user_id=session.user_id)

    # =========================================================================
    # Caching and Circuit Breaker
    # =========================================================================

    async def _cached_parse_intent(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]],
    ) -> Intent:
        """Parse intent with caching support."""
        if self._intent_cache:
            cache_key = hashlib.sha256(
                f"{user_input}:{str(context)}".encode()
            ).hexdigest()

            cached = await self._intent_cache.get(cache_key)
            if cached is not None:
                logger.debug("Intent cache hit", key=cache_key[:8])
                return cached

        # Parse with circuit breaker protection for LLM resilience
        try:
            intent = await self._llm_circuit.call(
                self.intent_parser.parse, user_input, context=context
            )
        except CircuitBreakerOpenError:
            logger.warning("Circuit breaker open, parsing without LLM protection")
            intent = await self.intent_parser.parse(user_input, context=context)

        # Apply learned bias adjustments
        bias = self.learner.get_intent_bias()
        if bias and intent.type.value in bias:
            adjustment = bias[intent.type.value]
            intent.confidence = max(0.0, min(1.0, intent.confidence + adjustment))

        if self._intent_cache:
            await self._intent_cache.put(cache_key, intent)

        return intent

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session async lock.

        Bounded: evicts oldest locks when exceeding 2x max sessions
        to prevent unbounded growth.
        """
        if session_id not in self._session_locks:
            # Evict stale locks if dict grows too large
            max_locks = self.config.max_concurrent_sessions * 2
            if len(self._session_locks) > max_locks:
                # Remove locks for sessions that no longer exist
                active_ids = {
                    s.id for s in self.sessions._sessions.values()
                    if s.state == "active"
                }
                stale = [
                    k for k in self._session_locks
                    if k not in active_ids
                ]
                for k in stale:
                    del self._session_locks[k]
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    # =========================================================================
    # Internal Handlers
    # =========================================================================

    async def _handle_creation(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Handle system creation requests."""
        # Generate specification
        try:
            spec = await self.spec_generator.generate(intent)
            session.current_spec = spec
        except Exception as e:
            return self._respond_error(session, f"Failed to generate specification: {e}")

        # Validate specification
        spec_validation = self.spec_validator.validate(spec)
        if not spec_validation.is_valid:
            return self._respond_error(
                session,
                f"Specification validation failed: {'; '.join(spec_validation.errors)}",
            )

        # Synthesize code
        try:
            code = await self._synthesize(intent.type, spec)
            session.current_code = code
        except Exception as e:
            return self._respond_error(session, f"Failed to generate code: {e}")

        # Validate generated code
        validation = await self.validator.validate(code)
        session.current_validation = validation

        if not validation.is_valid:
            return self._respond_validation_failed(session, validation)

        # Auto-deploy if configured, otherwise ask for confirmation
        if self.config.deployment.auto_deploy_on_pass and validation.is_valid:
            return await self.confirm_deploy(session.id)

        return await self._respond_ready_to_deploy(session, spec, code, validation, intent=intent)

    async def _handle_modification(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Handle modification of existing systems."""
        if session.current_spec:
            return await self.refine(session.id, intent.raw_input)

        return self._respond_error(
            session,
            "No active system to modify. Please create one first or specify which system to modify.",
        )

    async def _handle_deletion(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Handle system deletion."""
        name = intent.name
        if not name:
            return self._respond_error(session, "Please specify which system to delete.")

        deployed_systems = self.deployer.list_deployed()
        target = None
        for system in deployed_systems:
            if system.name == name:
                target = system
                break

        if not target:
            return self._respond_error(session, f"System '{name}' not found.")

        success = await self.deployer.undeploy(target.id)
        if success:
            msg = f"System '{name}' has been undeployed."
            self.history.add_message(session, "assistant", msg)
            return {"status": "deleted", "session_id": session.id, "name": name, "message": msg}
        else:
            return self._respond_error(session, f"Failed to undeploy '{name}'.")

    def _handle_query(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Handle list/status queries."""
        deployed = self.deployer.list_deployed()
        systems = [
            {
                "id": d.id,
                "name": d.name,
                "type": d.system_type.value,
                "status": d.status.value,
                "version": d.version,
                "invocations": d.invocation_count,
                "error_rate": round(d.error_rate, 4),
                "created_at": d.created_at.isoformat(),
            }
            for d in deployed
        ]

        msg = f"Found {len(systems)} deployed system(s)."
        self.history.add_message(session, "assistant", msg)

        return {
            "status": "query_result",
            "session_id": session.id,
            "systems": systems,
            "message": msg,
        }

    async def _handle_explain(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Handle explain/debug requests."""
        if session.current_code:
            code_preview = session.current_code.code[:1000]
            spec_info = ""
            if session.current_spec and hasattr(session.current_spec, "to_dict"):
                spec_info = str(session.current_spec.to_dict())

            prompt = f"""Explain this generated system:

Code:
```python
{code_preview}
```

Specification: {spec_info[:500]}

User question: {intent.raw_input}

Provide a clear, concise explanation."""

            try:
                response = await self.kernel.llm.complete(
                    [{"role": "user", "content": prompt}]
                )
                explanation = response.content if hasattr(response, "content") else str(response)
            except Exception:
                explanation = "Unable to generate explanation at this time."

            self.history.add_message(session, "assistant", explanation)
            return {
                "status": "explanation",
                "session_id": session.id,
                "message": explanation,
            }

        return self._respond_error(
            session,
            "No system to explain. Please create one first.",
        )

    async def _handle_test(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Handle test execution requests."""
        if not session.current_code:
            return self._respond_error(session, "No code to test. Create a system first.")

        validation = await self.validator.validate(session.current_code)
        session.current_validation = validation

        msg = f"Validation: {validation.status.value}. {validation.tests_passed} tests passed, {validation.tests_failed} failed."
        self.history.add_message(session, "assistant", msg)

        return {
            "status": "test_result",
            "session_id": session.id,
            "validation": {
                "status": validation.status.value,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "tests_passed": validation.tests_passed,
                "tests_failed": validation.tests_failed,
                "safety_score": validation.safety_score,
                "safety_level": validation.safety_level.value,
            },
            "message": msg,
        }

    async def _handle_rollback(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Handle rollback requests."""
        name = intent.name
        if not name:
            return self._respond_error(session, "Please specify which system to rollback.")

        deployed = self.deployer.list_deployed()
        target = None
        for d in deployed:
            if d.name == name:
                target = d
                break

        if not target:
            return self._respond_error(session, f"System '{name}' not found.")

        success = await self.deployer.rollback(target.id)
        if success:
            msg = f"System '{name}' has been rolled back."
            self.history.add_message(session, "assistant", msg)
            return {"status": "rolled_back", "session_id": session.id, "name": name, "message": msg}

        return self._respond_error(session, f"Failed to rollback '{name}'.")

    # =========================================================================
    # Synthesis Router
    # =========================================================================

    async def _synthesize(self, intent_type: IntentType, spec: Any) -> GeneratedCode:
        """Route to appropriate synthesizer."""
        from aion.nlp.types import (
            ToolSpecification, WorkflowSpecification,
            AgentSpecification, APISpecification,
            IntegrationSpecification,
        )

        if isinstance(spec, ToolSpecification):
            return await self.tool_synth.synthesize(spec)
        elif isinstance(spec, WorkflowSpecification):
            return await self.workflow_synth.synthesize(spec)
        elif isinstance(spec, AgentSpecification):
            return await self.agent_synth.synthesize(spec)
        elif isinstance(spec, APISpecification):
            return await self.api_synth.synthesize(spec)
        elif isinstance(spec, IntegrationSpecification):
            return await self.integration_synth.synthesize(spec)
        elif isinstance(spec, dict):
            return await self.code_gen.synthesize(spec)
        else:
            raise ValueError(f"No synthesizer for spec type: {type(spec).__name__}")

    # =========================================================================
    # Response Builders
    # =========================================================================

    def _respond_clarification(
        self,
        session: ProgrammingSession,
        intent: Intent,
    ) -> Dict[str, Any]:
        """Build clarification response."""
        questions = intent.clarification_questions

        msg = f"I need some clarification: {questions[0]}" if questions else "Could you provide more details?"
        self.history.add_message(session, "assistant", msg)

        return {
            "status": "clarification_needed",
            "session_id": session.id,
            "questions": questions,
            "current_understanding": {
                "type": intent.type.value,
                "confidence": round(intent.confidence, 3),
                "entities": [
                    {"type": e.type.value, "value": e.value}
                    for e in intent.entities[:10]
                ],
            },
            "message": msg,
        }

    async def _respond_ready_to_deploy(
        self,
        session: ProgrammingSession,
        spec: Any,
        code: GeneratedCode,
        validation: ValidationResult,
        intent: Optional[Intent] = None,
    ) -> Dict[str, Any]:
        """Build ready-to-deploy response with contextual suggestions."""
        spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else {}
        code_preview = code.code[:500] + "..." if len(code.code) > 500 else code.code

        msg = "Code generated and validated. Would you like to deploy?"
        if validation.warnings:
            msg = f"Code generated with {len(validation.warnings)} warning(s). Would you like to deploy?"

        self.history.add_message(session, "assistant", msg)

        # Generate contextual suggestions with intent awareness
        suggestions = await self.suggestions.generate(session, intent)

        return {
            "status": "ready_to_deploy",
            "session_id": session.id,
            "specification": spec_dict,
            "code_preview": code_preview,
            "code_lines": len(code.code.splitlines()),
            "validation": {
                "status": validation.status.value,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "safety_score": validation.safety_score,
                "safety_level": validation.safety_level.value,
                "tests_passed": validation.tests_passed,
                "tests_failed": validation.tests_failed,
            },
            "suggestions": suggestions,
            "message": msg,
        }

    def _respond_validation_failed(
        self,
        session: ProgrammingSession,
        validation: ValidationResult,
    ) -> Dict[str, Any]:
        """Build validation failure response."""
        msg = f"Validation failed with {len(validation.errors)} error(s). Please review."
        self.history.add_message(session, "assistant", msg)

        return {
            "status": "validation_failed",
            "session_id": session.id,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions,
            "safety_concerns": validation.safety_concerns,
            "message": msg,
        }

    def _respond_error(
        self,
        session: ProgrammingSession,
        error: str,
    ) -> Dict[str, Any]:
        """Build error response."""
        self.history.add_message(session, "assistant", f"Error: {error}")
        return {
            "status": "error",
            "session_id": session.id,
            "error": error,
        }

    # =========================================================================
    # Public Accessors
    # =========================================================================

    def get_session(self, session_id: str) -> Optional[ProgrammingSession]:
        return self.sessions.get(session_id)

    def list_deployed(self) -> List[DeployedSystem]:
        return self.deployer.list_deployed()

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats: Dict[str, Any] = {
            "active_sessions": self.sessions.active_count,
            "total_sessions": self.sessions.total_count,
            "total_requests": self._request_count,
            "avg_processing_ms": (
                round(self._total_processing_ms / self._request_count, 1)
                if self._request_count > 0 else 0.0
            ),
            "deployment_stats": self.deployer.get_stats(),
            "learning_stats": self.learner.get_stats(),
            "circuit_breaker": self._llm_circuit.stats,
        }
        if self._intent_cache:
            stats["cache"] = self._intent_cache.stats
        return stats
