"""
Microsegmentation for Zero Trust.

Implements network and resource microsegmentation:
- Logical resource grouping
- Fine-grained access control between segments
- Dynamic segment assignment
- Cross-segment communication policies
"""

from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class SegmentType(str, Enum):
    """Types of segments."""
    NETWORK = "network"
    APPLICATION = "application"
    DATA = "data"
    USER_GROUP = "user_group"
    WORKLOAD = "workload"


class CommunicationDirection(str, Enum):
    """Direction of allowed communication."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


class SegmentAction(str, Enum):
    """Actions for segment policies."""
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_INSPECTION = "allow_with_inspection"
    QUARANTINE = "quarantine"


@dataclass
class Microsegment:
    """Defines a microsegment."""
    id: str
    name: str
    segment_type: SegmentType
    description: str = ""

    # Membership criteria
    ip_ranges: list[str] = field(default_factory=list)
    hostnames: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    resource_patterns: list[str] = field(default_factory=list)
    user_groups: list[str] = field(default_factory=list)

    # Security classification
    sensitivity_level: str = "normal"  # low, normal, high, critical
    trust_level: str = "standard"      # untrusted, standard, trusted, highly_trusted

    # Access defaults
    default_inbound_action: SegmentAction = SegmentAction.DENY
    default_outbound_action: SegmentAction = SegmentAction.ALLOW

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def contains_ip(self, ip: str) -> bool:
        """Check if IP belongs to this segment."""
        try:
            ip_addr = ipaddress.ip_address(ip)
            for range_str in self.ip_ranges:
                try:
                    network = ipaddress.ip_network(range_str, strict=False)
                    if ip_addr in network:
                        return True
                except ValueError:
                    continue
        except ValueError:
            pass
        return False

    def contains_resource(self, resource: str) -> bool:
        """Check if resource belongs to this segment."""
        for pattern in self.resource_patterns:
            if self._pattern_matches(pattern, resource):
                return True
        return False

    def matches_labels(self, labels: dict[str, str]) -> bool:
        """Check if labels match this segment's criteria."""
        for key, value in self.labels.items():
            if labels.get(key) != value:
                return False
        return True

    def _pattern_matches(self, pattern: str, value: str) -> bool:
        """Simple pattern matching."""
        if pattern == "*":
            return True
        if pattern.endswith("/*"):
            return value.startswith(pattern[:-2])
        if pattern.endswith("/**"):
            return value.startswith(pattern[:-3])
        return pattern == value


@dataclass
class MicrosegmentPolicy:
    """Policy for communication between segments."""
    id: str
    name: str

    # Source and destination
    source_segment_id: str
    destination_segment_id: str

    # Action
    action: SegmentAction
    direction: CommunicationDirection = CommunicationDirection.BIDIRECTIONAL

    # Port/protocol restrictions
    allowed_protocols: list[str] = field(default_factory=lambda: ["*"])
    allowed_ports: list[int] = field(default_factory=list)  # Empty = all ports

    # Conditions
    require_encryption: bool = False
    require_authentication: bool = True
    max_risk_score: float = 70.0

    # Time restrictions
    active_hours: Optional[list[int]] = None
    active_days: Optional[list[int]] = None

    # Priority (higher = evaluated first)
    priority: int = 100

    # Enabled
    enabled: bool = True

    # Metadata
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass
class SegmentMembership:
    """Tracks entity membership in segments."""
    entity_id: str
    entity_type: str  # ip, hostname, resource, user
    segment_id: str
    assigned_at: float = field(default_factory=time.time)
    assigned_by: str = "system"
    dynamic: bool = True  # Dynamically assigned vs static
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationRequest:
    """Request to communicate between segments."""
    source_entity: str
    source_segment_id: Optional[str]
    destination_entity: str
    destination_segment_id: Optional[str]
    protocol: str
    port: Optional[int]
    user_id: Optional[str] = None
    risk_score: float = 0.0
    encrypted: bool = False
    authenticated: bool = False


@dataclass
class CommunicationDecision:
    """Decision for a communication request."""
    allowed: bool
    action: SegmentAction
    reason: str
    policy_id: Optional[str]
    requires_inspection: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SegmentationManager:
    """
    Manages microsegmentation policies and decisions.

    Implements fine-grained network and resource segmentation
    with dynamic policy enforcement.
    """

    def __init__(self) -> None:
        self._segments: dict[str, Microsegment] = {}
        self._policies: list[MicrosegmentPolicy] = []
        self._memberships: dict[str, list[SegmentMembership]] = {}  # entity_id -> memberships
        self._logger = logger.bind(component="segmentation_manager")

        # Initialize default segments
        self._init_default_segments()

    def _init_default_segments(self) -> None:
        """Initialize default segments."""
        # Internet segment (untrusted)
        self.add_segment(Microsegment(
            id="internet",
            name="Internet",
            segment_type=SegmentType.NETWORK,
            description="External internet traffic",
            ip_ranges=["0.0.0.0/0"],
            sensitivity_level="low",
            trust_level="untrusted",
            default_inbound_action=SegmentAction.DENY,
            default_outbound_action=SegmentAction.ALLOW_WITH_INSPECTION,
        ))

        # DMZ segment
        self.add_segment(Microsegment(
            id="dmz",
            name="DMZ",
            segment_type=SegmentType.NETWORK,
            description="Demilitarized zone for public-facing services",
            sensitivity_level="normal",
            trust_level="standard",
        ))

        # Internal segment
        self.add_segment(Microsegment(
            id="internal",
            name="Internal Network",
            segment_type=SegmentType.NETWORK,
            description="Internal corporate network",
            sensitivity_level="normal",
            trust_level="trusted",
        ))

        # Sensitive data segment
        self.add_segment(Microsegment(
            id="sensitive-data",
            name="Sensitive Data",
            segment_type=SegmentType.DATA,
            description="Segment containing sensitive data",
            sensitivity_level="high",
            trust_level="highly_trusted",
            default_inbound_action=SegmentAction.DENY,
        ))

        # Admin segment
        self.add_segment(Microsegment(
            id="admin",
            name="Administration",
            segment_type=SegmentType.USER_GROUP,
            description="Administrative access segment",
            sensitivity_level="critical",
            trust_level="highly_trusted",
        ))

    def add_segment(self, segment: Microsegment) -> None:
        """Add a microsegment."""
        self._segments[segment.id] = segment
        self._logger.info("Segment added", segment_id=segment.id, name=segment.name)

    def remove_segment(self, segment_id: str) -> bool:
        """Remove a microsegment."""
        if segment_id in self._segments:
            del self._segments[segment_id]
            # Remove associated policies
            self._policies = [
                p for p in self._policies
                if p.source_segment_id != segment_id and p.destination_segment_id != segment_id
            ]
            # Remove memberships
            for entity_id in list(self._memberships.keys()):
                self._memberships[entity_id] = [
                    m for m in self._memberships[entity_id]
                    if m.segment_id != segment_id
                ]
            return True
        return False

    def get_segment(self, segment_id: str) -> Optional[Microsegment]:
        """Get a segment by ID."""
        return self._segments.get(segment_id)

    def add_policy(self, policy: MicrosegmentPolicy) -> None:
        """Add a segmentation policy."""
        self._policies.append(policy)
        # Sort by priority (higher first)
        self._policies.sort(key=lambda p: p.priority, reverse=True)
        self._logger.info(
            "Policy added",
            policy_id=policy.id,
            source=policy.source_segment_id,
            destination=policy.destination_segment_id,
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy by ID."""
        initial_count = len(self._policies)
        self._policies = [p for p in self._policies if p.id != policy_id]
        return len(self._policies) < initial_count

    def assign_to_segment(
        self,
        entity_id: str,
        entity_type: str,
        segment_id: str,
        assigned_by: str = "system",
        dynamic: bool = True,
    ) -> SegmentMembership:
        """Assign an entity to a segment."""
        if segment_id not in self._segments:
            raise ValueError(f"Segment not found: {segment_id}")

        membership = SegmentMembership(
            entity_id=entity_id,
            entity_type=entity_type,
            segment_id=segment_id,
            assigned_by=assigned_by,
            dynamic=dynamic,
        )

        if entity_id not in self._memberships:
            self._memberships[entity_id] = []

        # Remove existing membership to same segment
        self._memberships[entity_id] = [
            m for m in self._memberships[entity_id]
            if m.segment_id != segment_id
        ]
        self._memberships[entity_id].append(membership)

        return membership

    def get_entity_segments(self, entity_id: str) -> list[Microsegment]:
        """Get all segments an entity belongs to."""
        memberships = self._memberships.get(entity_id, [])
        return [
            self._segments[m.segment_id]
            for m in memberships
            if m.segment_id in self._segments
        ]

    def determine_segment(
        self,
        entity_id: str,
        entity_type: str,
        ip_address: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        resource_path: Optional[str] = None,
    ) -> list[Microsegment]:
        """
        Dynamically determine which segments an entity belongs to.

        Uses IP ranges, labels, resource patterns, and explicit assignments.
        """
        matching_segments: list[Microsegment] = []

        # Check explicit memberships first
        if entity_id in self._memberships:
            for membership in self._memberships[entity_id]:
                if membership.segment_id in self._segments:
                    matching_segments.append(self._segments[membership.segment_id])

        # Check dynamic matching
        for segment in self._segments.values():
            if segment in matching_segments:
                continue

            # Check IP ranges
            if ip_address and segment.contains_ip(ip_address):
                matching_segments.append(segment)
                continue

            # Check labels
            if labels and segment.labels and segment.matches_labels(labels):
                matching_segments.append(segment)
                continue

            # Check resource patterns
            if resource_path and segment.contains_resource(resource_path):
                matching_segments.append(segment)
                continue

        return matching_segments

    def check_communication(
        self,
        request: CommunicationRequest,
    ) -> CommunicationDecision:
        """
        Check if communication between entities is allowed.

        Evaluates all applicable policies and returns a decision.
        """
        # Determine segments if not provided
        source_segments = []
        if request.source_segment_id:
            if request.source_segment_id in self._segments:
                source_segments = [self._segments[request.source_segment_id]]
        else:
            source_segments = self.determine_segment(
                request.source_entity,
                "unknown",
                ip_address=request.source_entity if "." in request.source_entity else None,
            )

        dest_segments = []
        if request.destination_segment_id:
            if request.destination_segment_id in self._segments:
                dest_segments = [self._segments[request.destination_segment_id]]
        else:
            dest_segments = self.determine_segment(
                request.destination_entity,
                "unknown",
                ip_address=request.destination_entity if "." in request.destination_entity else None,
            )

        # Default to internet segment if no segment found
        if not source_segments and "internet" in self._segments:
            source_segments = [self._segments["internet"]]
        if not dest_segments and "internet" in self._segments:
            dest_segments = [self._segments["internet"]]

        # Find applicable policies
        now = time.time()
        applicable_policies: list[MicrosegmentPolicy] = []

        for policy in self._policies:
            if not policy.enabled:
                continue

            if policy.expires_at and now > policy.expires_at:
                continue

            # Check if policy applies to source/destination pair
            source_match = any(
                s.id == policy.source_segment_id or policy.source_segment_id == "*"
                for s in source_segments
            )
            dest_match = any(
                s.id == policy.destination_segment_id or policy.destination_segment_id == "*"
                for s in dest_segments
            )

            if source_match and dest_match:
                # Check direction
                if policy.direction == CommunicationDirection.BIDIRECTIONAL:
                    applicable_policies.append(policy)
                elif policy.direction == CommunicationDirection.OUTBOUND:
                    applicable_policies.append(policy)
                # For inbound, would need to reverse source/dest check

        # Evaluate policies in priority order
        for policy in applicable_policies:
            decision = self._evaluate_policy(policy, request)
            if decision.action != SegmentAction.DENY or not self._has_allow_policy(applicable_policies):
                return decision

        # No matching policy - use segment defaults
        if dest_segments:
            dest_segment = dest_segments[0]
            return CommunicationDecision(
                allowed=dest_segment.default_inbound_action == SegmentAction.ALLOW,
                action=dest_segment.default_inbound_action,
                reason=f"Default action for segment {dest_segment.id}",
                policy_id=None,
            )

        # Ultimate default: deny
        return CommunicationDecision(
            allowed=False,
            action=SegmentAction.DENY,
            reason="No policy matched, default deny",
            policy_id=None,
        )

    def _evaluate_policy(
        self,
        policy: MicrosegmentPolicy,
        request: CommunicationRequest,
    ) -> CommunicationDecision:
        """Evaluate a single policy against a request."""
        # Check protocol
        if "*" not in policy.allowed_protocols:
            if request.protocol not in policy.allowed_protocols:
                return CommunicationDecision(
                    allowed=False,
                    action=SegmentAction.DENY,
                    reason=f"Protocol {request.protocol} not allowed",
                    policy_id=policy.id,
                )

        # Check port
        if policy.allowed_ports and request.port:
            if request.port not in policy.allowed_ports:
                return CommunicationDecision(
                    allowed=False,
                    action=SegmentAction.DENY,
                    reason=f"Port {request.port} not allowed",
                    policy_id=policy.id,
                )

        # Check encryption requirement
        if policy.require_encryption and not request.encrypted:
            return CommunicationDecision(
                allowed=False,
                action=SegmentAction.DENY,
                reason="Encryption required",
                policy_id=policy.id,
            )

        # Check authentication requirement
        if policy.require_authentication and not request.authenticated:
            return CommunicationDecision(
                allowed=False,
                action=SegmentAction.DENY,
                reason="Authentication required",
                policy_id=policy.id,
            )

        # Check risk score
        if request.risk_score > policy.max_risk_score:
            return CommunicationDecision(
                allowed=False,
                action=SegmentAction.DENY,
                reason=f"Risk score {request.risk_score} exceeds maximum {policy.max_risk_score}",
                policy_id=policy.id,
            )

        # Check time restrictions
        if policy.active_hours or policy.active_days:
            current_time = time.localtime()
            if policy.active_hours and current_time.tm_hour not in policy.active_hours:
                return CommunicationDecision(
                    allowed=False,
                    action=SegmentAction.DENY,
                    reason="Communication not allowed at this time",
                    policy_id=policy.id,
                )
            if policy.active_days and current_time.tm_wday not in policy.active_days:
                return CommunicationDecision(
                    allowed=False,
                    action=SegmentAction.DENY,
                    reason="Communication not allowed on this day",
                    policy_id=policy.id,
                )

        # Policy allows
        return CommunicationDecision(
            allowed=policy.action in (SegmentAction.ALLOW, SegmentAction.ALLOW_WITH_INSPECTION),
            action=policy.action,
            reason=f"Allowed by policy {policy.name}",
            policy_id=policy.id,
            requires_inspection=policy.action == SegmentAction.ALLOW_WITH_INSPECTION,
        )

    def _has_allow_policy(self, policies: list[MicrosegmentPolicy]) -> bool:
        """Check if there's any allow policy in the list."""
        return any(
            p.action in (SegmentAction.ALLOW, SegmentAction.ALLOW_WITH_INSPECTION)
            for p in policies
        )

    def get_segment_stats(self) -> dict[str, Any]:
        """Get statistics about segments and policies."""
        return {
            "total_segments": len(self._segments),
            "total_policies": len(self._policies),
            "segments_by_type": {
                st.value: sum(1 for s in self._segments.values() if s.segment_type == st)
                for st in SegmentType
            },
            "policies_by_action": {
                sa.value: sum(1 for p in self._policies if p.action == sa)
                for sa in SegmentAction
            },
            "total_memberships": sum(len(m) for m in self._memberships.values()),
        }
