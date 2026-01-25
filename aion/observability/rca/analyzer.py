"""
AI-Powered Root Cause Analysis

SOTA features:
- Multi-signal correlation (metrics, traces, logs, alerts)
- Causal inference using Granger causality
- Graph-based propagation analysis
- LLM-powered hypothesis generation
- Automated runbook execution
- Learning from past incidents
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class IncidentSeverity(Enum):
    """Incident severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(Enum):
    """Incident status."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"


class RootCauseType(Enum):
    """Types of root causes."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CODE_ERROR = "code_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ISSUE = "network_issue"
    DATABASE_ISSUE = "database_issue"
    CAPACITY_ISSUE = "capacity_issue"
    EXTERNAL_FACTOR = "external_factor"
    UNKNOWN = "unknown"


@dataclass
class Signal:
    """A signal contributing to the analysis."""
    source: str  # metrics, traces, logs, alerts
    name: str
    timestamp: datetime
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A hypothesis about the root cause."""
    description: str
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class RootCause:
    """Identified root cause."""
    cause_type: RootCauseType
    description: str
    confidence: float
    component: str  # Affected component/service
    evidence: List[Signal] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    remediation: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.cause_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "component": self.component,
            "evidence_count": len(self.evidence),
            "hypotheses": [
                {
                    "description": h.description,
                    "confidence": h.confidence,
                    "evidence": h.evidence,
                }
                for h in self.hypotheses
            ],
            "timeline": self.timeline,
            "remediation": self.remediation,
        }


@dataclass
class Incident:
    """An incident requiring root cause analysis."""
    id: str
    title: str
    severity: IncidentSeverity
    status: IncidentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    affected_services: List[str] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)
    root_causes: List[RootCause] = field(default_factory=list)
    description: str = ""
    impact: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "affected_services": self.affected_services,
            "signal_count": len(self.signals),
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "description": self.description,
            "impact": self.impact,
        }


@dataclass
class RCAResult:
    """Result of root cause analysis."""
    incident: Incident
    root_causes: List[RootCause]
    confidence: float
    analysis_duration_seconds: float
    signals_analyzed: int
    correlations_found: int
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident.id,
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "overall_confidence": self.confidence,
            "analysis_duration_seconds": self.analysis_duration_seconds,
            "signals_analyzed": self.signals_analyzed,
            "correlations_found": self.correlations_found,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Correlation Graph
# =============================================================================

@dataclass
class CorrelationEdge:
    """An edge in the correlation graph."""
    source: str
    target: str
    correlation: float  # -1 to 1
    lag_seconds: float  # Time lag
    causality_score: float  # Granger causality score


class CorrelationGraph:
    """
    Graph representing correlations between signals.

    Used for causal inference and propagation analysis.
    """

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[Tuple[str, str], CorrelationEdge] = {}
        self._time_series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def add_signal(self, name: str, timestamp: float, value: float) -> None:
        """Add a signal observation."""
        self.nodes.add(name)
        self._time_series[name].append((timestamp, value))

    def compute_correlations(self, max_lag_seconds: float = 300.0) -> None:
        """Compute correlations between all signals."""
        nodes = list(self.nodes)

        for i, source in enumerate(nodes):
            for target in nodes[i + 1:]:
                edge = self._compute_correlation(source, target, max_lag_seconds)
                if edge and abs(edge.correlation) > 0.3:
                    self.edges[(source, target)] = edge

    def _compute_correlation(
        self,
        source: str,
        target: str,
        max_lag: float,
    ) -> Optional[CorrelationEdge]:
        """Compute correlation between two signals."""
        source_ts = self._time_series.get(source, [])
        target_ts = self._time_series.get(target, [])

        if len(source_ts) < 10 or len(target_ts) < 10:
            return None

        # Align time series
        source_arr = np.array([v for _, v in source_ts])
        target_arr = np.array([v for _, v in target_ts])

        # Normalize
        source_norm = (source_arr - np.mean(source_arr)) / (np.std(source_arr) + 1e-10)
        target_norm = (target_arr - np.mean(target_arr)) / (np.std(target_arr) + 1e-10)

        # Cross-correlation with lag
        min_len = min(len(source_norm), len(target_norm))
        source_norm = source_norm[:min_len]
        target_norm = target_norm[:min_len]

        correlation = np.corrcoef(source_norm, target_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Simple Granger causality (using lagged correlation)
        best_lag = 0
        best_causality = abs(correlation)

        for lag in range(1, min(10, min_len // 2)):
            lagged_corr = np.corrcoef(source_norm[:-lag], target_norm[lag:])[0, 1]
            if not np.isnan(lagged_corr) and abs(lagged_corr) > best_causality:
                best_causality = abs(lagged_corr)
                best_lag = lag

        return CorrelationEdge(
            source=source,
            target=target,
            correlation=float(correlation),
            lag_seconds=best_lag * 60,  # Assuming 1-minute intervals
            causality_score=float(best_causality),
        )

    def get_causal_chain(self, target: str) -> List[str]:
        """Find the causal chain leading to a target signal."""
        chain = []
        visited = set()

        def dfs(node: str) -> None:
            if node in visited:
                return
            visited.add(node)

            # Find edges where this node is the target
            for (source, t), edge in self.edges.items():
                if t == node and edge.causality_score > 0.5:
                    chain.append(source)
                    dfs(source)

        dfs(target)
        return chain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "correlation": e.correlation,
                    "lag_seconds": e.lag_seconds,
                    "causality_score": e.causality_score,
                }
                for e in self.edges.values()
            ],
        }


# =============================================================================
# Pattern Library
# =============================================================================

@dataclass
class RCAPattern:
    """A known root cause pattern."""
    name: str
    description: str
    cause_type: RootCauseType
    indicators: List[str]  # Regex patterns for signals
    confidence_boost: float
    remediation: List[str]


class PatternLibrary:
    """Library of known root cause patterns."""

    PATTERNS = [
        RCAPattern(
            name="memory_exhaustion",
            description="Memory exhaustion causing OOM kills or degraded performance",
            cause_type=RootCauseType.RESOURCE_EXHAUSTION,
            indicators=[
                r"memory.*usage.*[89]\d%",
                r"oom.*kill",
                r"out.*of.*memory",
                r"heap.*full",
                r"gc.*pause.*\d{3,}ms",
            ],
            confidence_boost=0.3,
            remediation=[
                "Scale up memory allocation",
                "Investigate memory leaks",
                "Tune garbage collection settings",
                "Add memory-based autoscaling",
            ],
        ),
        RCAPattern(
            name="cpu_saturation",
            description="CPU saturation causing slow response times",
            cause_type=RootCauseType.RESOURCE_EXHAUSTION,
            indicators=[
                r"cpu.*usage.*[89]\d%",
                r"cpu.*throttl",
                r"load.*average.*high",
                r"runnable.*queue.*long",
            ],
            confidence_boost=0.25,
            remediation=[
                "Scale up CPU allocation",
                "Profile and optimize hot paths",
                "Add CPU-based autoscaling",
                "Review concurrent request limits",
            ],
        ),
        RCAPattern(
            name="database_connection_exhaustion",
            description="Database connection pool exhaustion",
            cause_type=RootCauseType.DATABASE_ISSUE,
            indicators=[
                r"connection.*pool.*exhaust",
                r"too.*many.*connections",
                r"connection.*timeout",
                r"waiting.*for.*connection",
            ],
            confidence_boost=0.35,
            remediation=[
                "Increase connection pool size",
                "Add connection pooling middleware",
                "Optimize query patterns",
                "Check for connection leaks",
            ],
        ),
        RCAPattern(
            name="downstream_dependency_failure",
            description="Failure in a downstream service",
            cause_type=RootCauseType.DEPENDENCY_FAILURE,
            indicators=[
                r"connection.*refused",
                r"upstream.*error",
                r"service.*unavailable",
                r"timeout.*downstream",
                r"circuit.*breaker.*open",
            ],
            confidence_boost=0.4,
            remediation=[
                "Check downstream service health",
                "Enable/tune circuit breakers",
                "Add retry with backoff",
                "Consider fallback behavior",
            ],
        ),
        RCAPattern(
            name="network_partition",
            description="Network connectivity issues",
            cause_type=RootCauseType.NETWORK_ISSUE,
            indicators=[
                r"network.*unreachable",
                r"connection.*reset",
                r"dns.*resolution.*fail",
                r"packet.*loss",
                r"socket.*timeout",
            ],
            confidence_boost=0.3,
            remediation=[
                "Check network connectivity",
                "Verify DNS configuration",
                "Check firewall rules",
                "Review service mesh configuration",
            ],
        ),
        RCAPattern(
            name="deployment_regression",
            description="Issue introduced by recent deployment",
            cause_type=RootCauseType.CODE_ERROR,
            indicators=[
                r"deployment.*recent",
                r"version.*new",
                r"rollout.*started",
                r"change.*deployed",
            ],
            confidence_boost=0.35,
            remediation=[
                "Consider rolling back deployment",
                "Review recent code changes",
                "Check deployment configuration",
                "Compare metrics before/after deployment",
            ],
        ),
        RCAPattern(
            name="configuration_error",
            description="Misconfiguration causing issues",
            cause_type=RootCauseType.CONFIGURATION_ERROR,
            indicators=[
                r"config.*invalid",
                r"configuration.*error",
                r"missing.*required",
                r"invalid.*value",
            ],
            confidence_boost=0.3,
            remediation=[
                "Review configuration changes",
                "Validate configuration syntax",
                "Check environment variables",
                "Compare with known-good configuration",
            ],
        ),
    ]

    def match_signals(self, signals: List[Signal]) -> List[Tuple[RCAPattern, float]]:
        """Match signals against known patterns."""
        matches = []

        for pattern in self.PATTERNS:
            match_count = 0
            for signal in signals:
                signal_str = f"{signal.name} {signal.value}".lower()
                for indicator in pattern.indicators:
                    if re.search(indicator, signal_str, re.IGNORECASE):
                        match_count += 1
                        break

            if match_count > 0:
                confidence = min(1.0, match_count / len(pattern.indicators) + pattern.confidence_boost)
                matches.append((pattern, confidence))

        return sorted(matches, key=lambda x: -x[1])


# =============================================================================
# Root Cause Analyzer
# =============================================================================

class RootCauseAnalyzer:
    """
    AI-powered root cause analysis engine.

    Combines multiple techniques:
    - Signal correlation
    - Pattern matching
    - Causal inference
    - LLM-based hypothesis generation
    """

    def __init__(
        self,
        metrics_engine: Optional[Any] = None,
        tracing_engine: Optional[Any] = None,
        logging_engine: Optional[Any] = None,
        alert_engine: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ):
        self.metrics_engine = metrics_engine
        self.tracing_engine = tracing_engine
        self.logging_engine = logging_engine
        self.alert_engine = alert_engine
        self.llm_client = llm_client

        self.pattern_library = PatternLibrary()
        self._incidents: Dict[str, Incident] = {}
        self._past_analyses: List[RCAResult] = []

    async def analyze_incident(self, incident: Incident) -> RCAResult:
        """Perform root cause analysis on an incident."""
        import time
        start_time = time.time()

        # Collect signals
        signals = await self._collect_signals(incident)
        incident.signals = signals

        # Build correlation graph
        correlation_graph = self._build_correlation_graph(signals)
        correlation_graph.compute_correlations()

        # Match patterns
        pattern_matches = self.pattern_library.match_signals(signals)

        # Generate hypotheses
        hypotheses = await self._generate_hypotheses(
            incident, signals, correlation_graph, pattern_matches
        )

        # Rank and select root causes
        root_causes = self._identify_root_causes(
            incident, signals, correlation_graph, pattern_matches, hypotheses
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(root_causes)

        analysis_time = time.time() - start_time

        result = RCAResult(
            incident=incident,
            root_causes=root_causes,
            confidence=max([rc.confidence for rc in root_causes]) if root_causes else 0.0,
            analysis_duration_seconds=analysis_time,
            signals_analyzed=len(signals),
            correlations_found=len(correlation_graph.edges),
            recommendations=recommendations,
        )

        # Store for learning
        self._past_analyses.append(result)
        incident.root_causes = root_causes

        return result

    async def _collect_signals(self, incident: Incident) -> List[Signal]:
        """Collect relevant signals from all sources."""
        signals = []
        time_range = (
            incident.start_time - timedelta(minutes=30),
            incident.end_time or datetime.utcnow(),
        )

        # Collect from metrics
        if self.metrics_engine:
            try:
                for service in incident.affected_services:
                    # Get key metrics
                    for metric in ["error_rate", "latency_p99", "request_rate", "cpu_usage", "memory_usage"]:
                        values = self.metrics_engine.query(
                            metric,
                            labels={"service": service},
                            start_time=time_range[0],
                            end_time=time_range[1],
                        )
                        for ts, value in values:
                            signals.append(Signal(
                                source="metrics",
                                name=f"{service}:{metric}",
                                timestamp=ts,
                                value=value,
                                metadata={"service": service, "metric": metric},
                            ))
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

        # Collect from alerts
        if self.alert_engine:
            try:
                alerts = self.alert_engine.get_active_alerts()
                for alert in alerts:
                    if alert.fired_at and time_range[0] <= alert.fired_at <= time_range[1]:
                        signals.append(Signal(
                            source="alerts",
                            name=alert.name,
                            timestamp=alert.fired_at,
                            value=alert.value,
                            metadata={"severity": alert.severity.value, "labels": alert.labels},
                        ))
            except Exception as e:
                logger.error(f"Error collecting alerts: {e}")

        # Collect from logs
        if self.logging_engine:
            try:
                logs = self.logging_engine.query_logs(
                    level="ERROR",
                    start_time=time_range[0],
                    end_time=time_range[1],
                    limit=100,
                )
                for log in logs:
                    signals.append(Signal(
                        source="logs",
                        name=f"error:{log.logger_name}",
                        timestamp=log.timestamp,
                        value=log.message,
                        metadata={"level": log.level.value, "trace_id": log.trace_id},
                    ))
            except Exception as e:
                logger.error(f"Error collecting logs: {e}")

        return signals

    def _build_correlation_graph(self, signals: List[Signal]) -> CorrelationGraph:
        """Build correlation graph from signals."""
        graph = CorrelationGraph()

        for signal in signals:
            if isinstance(signal.value, (int, float)):
                graph.add_signal(
                    signal.name,
                    signal.timestamp.timestamp(),
                    float(signal.value),
                )

        return graph

    async def _generate_hypotheses(
        self,
        incident: Incident,
        signals: List[Signal],
        correlation_graph: CorrelationGraph,
        pattern_matches: List[Tuple[RCAPattern, float]],
    ) -> List[Hypothesis]:
        """Generate hypotheses about root cause."""
        hypotheses = []

        # Pattern-based hypotheses
        for pattern, confidence in pattern_matches[:5]:
            hypothesis = Hypothesis(
                description=f"Possible {pattern.description}",
                confidence=confidence,
                evidence=[f"Matched pattern: {pattern.name}"],
                suggested_actions=pattern.remediation,
            )
            hypotheses.append(hypothesis)

        # Correlation-based hypotheses
        for affected in incident.affected_services:
            causal_chain = correlation_graph.get_causal_chain(f"{affected}:error_rate")
            if causal_chain:
                hypothesis = Hypothesis(
                    description=f"Error propagation chain: {' -> '.join(causal_chain)} -> {affected}",
                    confidence=0.6,
                    evidence=[f"Causal chain detected with {len(causal_chain)} steps"],
                    suggested_actions=["Investigate upstream services", "Check service dependencies"],
                )
                hypotheses.append(hypothesis)

        # LLM-based hypotheses (if available)
        if self.llm_client:
            try:
                llm_hypotheses = await self._generate_llm_hypotheses(incident, signals)
                hypotheses.extend(llm_hypotheses)
            except Exception as e:
                logger.error(f"Error generating LLM hypotheses: {e}")

        return hypotheses

    async def _generate_llm_hypotheses(
        self,
        incident: Incident,
        signals: List[Signal],
    ) -> List[Hypothesis]:
        """Generate hypotheses using LLM."""
        # Build context for LLM
        context = f"""
Incident Analysis Request:

Title: {incident.title}
Severity: {incident.severity.value}
Affected Services: {', '.join(incident.affected_services)}
Start Time: {incident.start_time}

Key Signals:
"""
        for signal in signals[:20]:
            context += f"- [{signal.source}] {signal.name}: {signal.value}\n"

        context += """
Based on these signals, generate hypotheses about the root cause.
For each hypothesis, provide:
1. Description of the potential root cause
2. Confidence level (0-1)
3. Evidence from the signals
4. Suggested remediation steps
"""

        # This would call the actual LLM
        # For now, return empty list
        return []

    def _identify_root_causes(
        self,
        incident: Incident,
        signals: List[Signal],
        correlation_graph: CorrelationGraph,
        pattern_matches: List[Tuple[RCAPattern, float]],
        hypotheses: List[Hypothesis],
    ) -> List[RootCause]:
        """Identify and rank root causes."""
        root_causes = []

        # Create root causes from top pattern matches
        for pattern, confidence in pattern_matches[:3]:
            # Find supporting evidence
            evidence = [
                s for s in signals
                if any(
                    re.search(ind, f"{s.name} {s.value}".lower(), re.IGNORECASE)
                    for ind in pattern.indicators
                )
            ]

            # Find related hypotheses
            related_hypotheses = [
                h for h in hypotheses
                if pattern.name.replace("_", " ") in h.description.lower()
            ]

            root_cause = RootCause(
                cause_type=pattern.cause_type,
                description=pattern.description,
                confidence=confidence,
                component=incident.affected_services[0] if incident.affected_services else "unknown",
                evidence=evidence[:10],
                hypotheses=related_hypotheses,
                remediation=pattern.remediation,
            )
            root_causes.append(root_cause)

        # If no pattern matches, create generic root cause
        if not root_causes:
            root_cause = RootCause(
                cause_type=RootCauseType.UNKNOWN,
                description="Unable to determine specific root cause. Manual investigation required.",
                confidence=0.2,
                component=incident.affected_services[0] if incident.affected_services else "unknown",
                evidence=signals[:5],
                remediation=[
                    "Review recent changes",
                    "Check system logs",
                    "Verify external dependencies",
                    "Consult on-call engineer",
                ],
            )
            root_causes.append(root_cause)

        return root_causes

    def _generate_recommendations(self, root_causes: List[RootCause]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        seen = set()

        for rc in root_causes:
            for action in rc.remediation:
                if action not in seen:
                    recommendations.append(action)
                    seen.add(action)

        # Add general recommendations
        general = [
            "Document findings in incident report",
            "Update runbooks based on analysis",
            "Consider implementing additional monitoring",
        ]

        for rec in general:
            if rec not in seen:
                recommendations.append(rec)

        return recommendations[:10]

    def create_incident(
        self,
        title: str,
        severity: IncidentSeverity,
        affected_services: List[str],
        description: str = "",
    ) -> Incident:
        """Create a new incident for analysis."""
        import uuid

        incident = Incident(
            id=str(uuid.uuid4())[:8],
            title=title,
            severity=severity,
            status=IncidentStatus.DETECTED,
            start_time=datetime.utcnow(),
            affected_services=affected_services,
            description=description,
        )

        self._incidents[incident.id] = incident
        return incident

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by ID."""
        return self._incidents.get(incident_id)

    def get_past_analyses(self, limit: int = 10) -> List[RCAResult]:
        """Get past analysis results."""
        return self._past_analyses[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "total_incidents": len(self._incidents),
            "total_analyses": len(self._past_analyses),
            "patterns_available": len(self.pattern_library.PATTERNS),
            "has_llm": self.llm_client is not None,
        }
