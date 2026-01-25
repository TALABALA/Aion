"""
Compliance Monitoring for Security Observability.

Provides compliance monitoring capabilities for:
- SOC 2, HIPAA, PCI-DSS, GDPR
- Custom compliance policies
- Audit logging
- Compliance reporting
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Standard compliance frameworks."""
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    ISO27001 = "iso27001"
    NIST = "nist"
    CUSTOM = "custom"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class RuleSeverity(Enum):
    """Compliance rule severity."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ComplianceRule:
    """A compliance rule definition."""
    rule_id: str
    name: str
    description: str
    framework: ComplianceFramework
    severity: RuleSeverity
    check_function: Optional[Callable[[Dict], bool]] = None
    # Pattern-based checking
    required_fields: List[str] = field(default_factory=list)
    forbidden_patterns: Dict[str, str] = field(default_factory=dict)
    required_patterns: Dict[str, str] = field(default_factory=dict)
    # Metadata
    control_id: str = ""  # e.g., "CC6.1" for SOC2
    remediation: str = ""
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class ComplianceViolation:
    """A compliance violation instance."""
    violation_id: str
    rule: ComplianceRule
    timestamp: datetime
    resource: str
    details: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ComplianceReport:
    """Compliance assessment report."""
    report_id: str
    generated_at: datetime
    framework: ComplianceFramework
    period_start: datetime
    period_end: datetime
    total_rules: int
    compliant_rules: int
    non_compliant_rules: int
    partial_rules: int
    overall_status: ComplianceStatus
    violations: List[ComplianceViolation]
    rule_results: Dict[str, ComplianceStatus]
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    """Audit log event."""
    event_id: str
    timestamp: datetime
    actor: str
    action: str
    resource: str
    resource_type: str
    outcome: str  # success, failure
    source_ip: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict] = None
    after_state: Optional[Dict] = None


class AuditLogger:
    """Audit logging for compliance."""

    def __init__(self, storage_backend: Any = None):
        self._events: List[AuditEvent] = []
        self._storage = storage_backend
        self._handlers: List[Callable[[AuditEvent], None]] = []

    def add_handler(self, handler: Callable[[AuditEvent], None]):
        """Add audit event handler."""
        self._handlers.append(handler)

    def log(self, actor: str, action: str, resource: str, resource_type: str,
            outcome: str = "success", **kwargs) -> AuditEvent:
        """Log an audit event."""
        import uuid
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            actor=actor,
            action=action,
            resource=resource,
            resource_type=resource_type,
            outcome=outcome,
            source_ip=kwargs.get("source_ip"),
            details=kwargs.get("details", {}),
            before_state=kwargs.get("before_state"),
            after_state=kwargs.get("after_state"),
        )

        self._events.append(event)

        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Audit handler error: {e}")

        return event

    def query(self, actor: str = None, action: str = None, resource_type: str = None,
              since: datetime = None, until: datetime = None) -> List[AuditEvent]:
        """Query audit events."""
        results = self._events

        if actor:
            results = [e for e in results if e.actor == actor]
        if action:
            results = [e for e in results if e.action == action]
        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]
        if since:
            results = [e for e in results if e.timestamp >= since]
        if until:
            results = [e for e in results if e.timestamp <= until]

        return results

    def export(self, format: str = "json") -> str:
        """Export audit log."""
        if format == "json":
            return json.dumps([
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "actor": e.actor,
                    "action": e.action,
                    "resource": e.resource,
                    "resource_type": e.resource_type,
                    "outcome": e.outcome,
                    "source_ip": e.source_ip,
                    "details": e.details,
                }
                for e in self._events
            ], indent=2)
        return ""


class ComplianceMonitor:
    """
    Main compliance monitoring engine.

    Continuously monitors for compliance violations and generates reports.
    """

    def __init__(self):
        self._rules: Dict[str, ComplianceRule] = {}
        self._violations: List[ComplianceViolation] = []
        self.audit_logger = AuditLogger()
        self._violation_handlers: List[Callable[[ComplianceViolation], None]] = []

        # Register built-in rules
        self._register_builtin_rules()

    def _register_builtin_rules(self):
        """Register built-in compliance rules."""
        rules = [
            # SOC 2 rules
            ComplianceRule(
                rule_id="soc2_encryption_at_rest",
                name="Data Encryption at Rest",
                description="Sensitive data must be encrypted at rest",
                framework=ComplianceFramework.SOC2,
                severity=RuleSeverity.HIGH,
                control_id="CC6.1",
                required_patterns={"encryption": r"(AES|RSA|encrypted)"},
                remediation="Enable encryption for data storage"
            ),
            ComplianceRule(
                rule_id="soc2_access_logging",
                name="Access Logging",
                description="All system access must be logged",
                framework=ComplianceFramework.SOC2,
                severity=RuleSeverity.HIGH,
                control_id="CC6.2",
                required_fields=["user", "action", "timestamp", "resource"],
                remediation="Enable comprehensive access logging"
            ),
            # HIPAA rules
            ComplianceRule(
                rule_id="hipaa_phi_access",
                name="PHI Access Controls",
                description="PHI access must be logged and authorized",
                framework=ComplianceFramework.HIPAA,
                severity=RuleSeverity.CRITICAL,
                required_fields=["user", "authorization", "phi_type"],
                remediation="Implement PHI access controls"
            ),
            # PCI-DSS rules
            ComplianceRule(
                rule_id="pci_card_data",
                name="Card Data Protection",
                description="Card data must not be stored in plain text",
                framework=ComplianceFramework.PCI_DSS,
                severity=RuleSeverity.CRITICAL,
                forbidden_patterns={"data": r"\b\d{13,16}\b"},  # Card numbers
                remediation="Tokenize or encrypt card data"
            ),
            # GDPR rules
            ComplianceRule(
                rule_id="gdpr_consent",
                name="Consent Tracking",
                description="User consent must be tracked for data processing",
                framework=ComplianceFramework.GDPR,
                severity=RuleSeverity.HIGH,
                required_fields=["consent", "purpose", "timestamp"],
                remediation="Implement consent management"
            ),
        ]

        for rule in rules:
            self.register_rule(rule)

    def register_rule(self, rule: ComplianceRule):
        """Register a compliance rule."""
        self._rules[rule.rule_id] = rule

    def add_violation_handler(self, handler: Callable[[ComplianceViolation], None]):
        """Add violation handler."""
        self._violation_handlers.append(handler)

    def check(self, data: Dict[str, Any], resource: str = "unknown",
              frameworks: List[ComplianceFramework] = None) -> List[ComplianceViolation]:
        """Check data for compliance violations."""
        violations = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if frameworks and rule.framework not in frameworks:
                continue

            violation = self._check_rule(rule, data, resource)
            if violation:
                violations.append(violation)
                self._violations.append(violation)

                for handler in self._violation_handlers:
                    try:
                        handler(violation)
                    except Exception as e:
                        logger.error(f"Violation handler error: {e}")

        return violations

    def _check_rule(self, rule: ComplianceRule, data: Dict[str, Any],
                    resource: str) -> Optional[ComplianceViolation]:
        """Check a single rule against data."""
        import uuid
        import re

        # Custom check function
        if rule.check_function:
            try:
                if not rule.check_function(data):
                    return ComplianceViolation(
                        violation_id=str(uuid.uuid4())[:8],
                        rule=rule,
                        timestamp=datetime.now(),
                        resource=resource,
                        details="Custom check failed",
                        evidence=data
                    )
            except Exception as e:
                logger.error(f"Rule check error: {e}")
            return None

        # Required fields check
        for field in rule.required_fields:
            if field not in data or data[field] is None:
                return ComplianceViolation(
                    violation_id=str(uuid.uuid4())[:8],
                    rule=rule,
                    timestamp=datetime.now(),
                    resource=resource,
                    details=f"Missing required field: {field}",
                    evidence={"missing_field": field}
                )

        # Forbidden patterns check
        for field, pattern in rule.forbidden_patterns.items():
            value = str(data.get(field, ""))
            if re.search(pattern, value):
                return ComplianceViolation(
                    violation_id=str(uuid.uuid4())[:8],
                    rule=rule,
                    timestamp=datetime.now(),
                    resource=resource,
                    details=f"Forbidden pattern found in {field}",
                    evidence={"field": field, "pattern": pattern}
                )

        # Required patterns check
        for field, pattern in rule.required_patterns.items():
            value = str(data.get(field, ""))
            if not re.search(pattern, value):
                return ComplianceViolation(
                    violation_id=str(uuid.uuid4())[:8],
                    rule=rule,
                    timestamp=datetime.now(),
                    resource=resource,
                    details=f"Required pattern not found in {field}",
                    evidence={"field": field, "pattern": pattern}
                )

        return None

    def generate_report(self, framework: ComplianceFramework,
                        period_start: datetime, period_end: datetime) -> ComplianceReport:
        """Generate compliance report."""
        import uuid

        # Get relevant rules and violations
        rules = [r for r in self._rules.values() if r.framework == framework and r.enabled]
        violations = [v for v in self._violations
                     if v.rule.framework == framework
                     and period_start <= v.timestamp <= period_end]

        # Calculate status per rule
        rule_results = {}
        violated_rules = {v.rule.rule_id for v in violations if not v.resolved}

        for rule in rules:
            if rule.rule_id in violated_rules:
                rule_results[rule.rule_id] = ComplianceStatus.NON_COMPLIANT
            else:
                rule_results[rule.rule_id] = ComplianceStatus.COMPLIANT

        # Calculate overall status
        compliant = sum(1 for s in rule_results.values() if s == ComplianceStatus.COMPLIANT)
        non_compliant = sum(1 for s in rule_results.values() if s == ComplianceStatus.NON_COMPLIANT)
        partial = sum(1 for s in rule_results.values() if s == ComplianceStatus.PARTIAL)

        if non_compliant > 0:
            overall = ComplianceStatus.NON_COMPLIANT
        elif partial > 0:
            overall = ComplianceStatus.PARTIAL
        else:
            overall = ComplianceStatus.COMPLIANT

        # Generate recommendations
        recommendations = []
        for v in violations:
            if not v.resolved and v.rule.remediation:
                recommendations.append(f"{v.rule.name}: {v.rule.remediation}")

        return ComplianceReport(
            report_id=str(uuid.uuid4())[:8],
            generated_at=datetime.now(),
            framework=framework,
            period_start=period_start,
            period_end=period_end,
            total_rules=len(rules),
            compliant_rules=compliant,
            non_compliant_rules=non_compliant,
            partial_rules=partial,
            overall_status=overall,
            violations=violations,
            rule_results=rule_results,
            recommendations=list(set(recommendations))
        )

    def get_violations(self, framework: ComplianceFramework = None,
                       resolved: bool = None, since: datetime = None) -> List[ComplianceViolation]:
        """Get violations with filters."""
        results = self._violations

        if framework:
            results = [v for v in results if v.rule.framework == framework]
        if resolved is not None:
            results = [v for v in results if v.resolved == resolved]
        if since:
            results = [v for v in results if v.timestamp >= since]

        return results
