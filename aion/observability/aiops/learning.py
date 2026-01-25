"""
Incident Learning for AIOps.

Implements learning from incident resolutions:
- Pattern extraction from resolved incidents
- Similarity matching for new incidents
- Knowledge base management
- Feedback collection and model improvement
"""

import math
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class Incident:
    """Representation of an incident."""
    incident_id: str
    title: str
    description: str
    severity: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    duration: Optional[timedelta] = None
    affected_services: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Resolution:
    """Resolution details for an incident."""
    resolution_id: str
    incident_id: str
    root_cause: str
    resolution_steps: List[str]
    actions_taken: List[str]
    time_to_resolve: timedelta
    prevented_future: bool = False
    effectiveness_rating: float = 0.0
    feedback: Optional[str] = None


@dataclass
class ResolutionPattern:
    """Learned pattern from incident resolutions."""
    pattern_id: str
    name: str
    description: str
    symptom_patterns: List[str]
    metric_conditions: Dict[str, Tuple[str, float]]  # metric -> (operator, threshold)
    service_patterns: List[str]
    recommended_actions: List[str]
    root_cause_template: str
    confidence: float
    success_count: int
    total_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class SimilarIncident:
    """Result of similarity search."""
    incident: Incident
    resolution: Optional[Resolution]
    similarity_score: float
    matching_features: List[str]


# =============================================================================
# Feature Extraction
# =============================================================================

class FeatureExtractor:
    """Extract features from incidents for similarity matching."""

    def __init__(self):
        # TF-IDF like weights for terms
        self._term_frequencies: Dict[str, int] = defaultdict(int)
        self._document_count = 0

    def extract_features(self, incident: Incident) -> Dict[str, Any]:
        """Extract features from an incident."""
        features = {
            'severity': incident.severity,
            'services': set(incident.affected_services),
            'symptoms': set(incident.symptoms),
            'metric_keys': set(incident.metrics.keys()),
            'labels': dict(incident.labels),
        }

        # Text features from title and description
        text = f"{incident.title} {incident.description}"
        features['terms'] = self._extract_terms(text)

        # Metric-based features
        features['metric_anomalies'] = self._detect_metric_anomalies(incident.metrics)

        # Temporal features
        if incident.created_at:
            features['hour_of_day'] = incident.created_at.hour
            features['day_of_week'] = incident.created_at.weekday()

        return features

    def _extract_terms(self, text: str) -> Set[str]:
        """Extract significant terms from text."""
        # Simple tokenization and normalization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or'
        }

        terms = {t for t in tokens if t not in stop_words and len(t) > 2}

        return terms

    def _detect_metric_anomalies(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Detect anomalous metric patterns."""
        anomalies = {}

        for metric, value in metrics.items():
            # Simple heuristics for anomaly detection
            if 'error' in metric.lower() and value > 0:
                anomalies[metric] = 'elevated'
            elif 'latency' in metric.lower() and value > 1000:
                anomalies[metric] = 'high'
            elif 'cpu' in metric.lower() and value > 80:
                anomalies[metric] = 'high'
            elif 'memory' in metric.lower() and value > 85:
                anomalies[metric] = 'high'
            elif 'disk' in metric.lower() and value > 90:
                anomalies[metric] = 'critical'

        return anomalies

    def update_statistics(self, incident: Incident):
        """Update term frequency statistics."""
        self._document_count += 1
        features = self.extract_features(incident)

        for term in features.get('terms', set()):
            self._term_frequencies[term] += 1


# =============================================================================
# Similarity Matching
# =============================================================================

class SimilarityMatcher:
    """Match incidents based on similarity."""

    def __init__(self, feature_extractor: FeatureExtractor = None):
        self.feature_extractor = feature_extractor or FeatureExtractor()

        # Weights for different feature types
        self.weights = {
            'severity': 0.15,
            'services': 0.25,
            'symptoms': 0.20,
            'terms': 0.20,
            'metrics': 0.15,
            'temporal': 0.05
        }

    def compute_similarity(
        self,
        incident1: Incident,
        incident2: Incident
    ) -> Tuple[float, List[str]]:
        """
        Compute similarity between two incidents.

        Returns:
            (similarity_score, list of matching features)
        """
        f1 = self.feature_extractor.extract_features(incident1)
        f2 = self.feature_extractor.extract_features(incident2)

        total_score = 0.0
        matching_features = []

        # Severity match
        if f1['severity'] == f2['severity']:
            total_score += self.weights['severity']
            matching_features.append(f"severity:{f1['severity']}")

        # Service overlap (Jaccard similarity)
        s1, s2 = f1['services'], f2['services']
        if s1 or s2:
            service_sim = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
            total_score += self.weights['services'] * service_sim
            if service_sim > 0.5:
                matching_features.append(f"services:{list(s1 & s2)}")

        # Symptom overlap
        sym1, sym2 = f1['symptoms'], f2['symptoms']
        if sym1 or sym2:
            symptom_sim = len(sym1 & sym2) / len(sym1 | sym2) if (sym1 | sym2) else 0
            total_score += self.weights['symptoms'] * symptom_sim
            if symptom_sim > 0.5:
                matching_features.append(f"symptoms:{list(sym1 & sym2)}")

        # Term overlap (cosine-like similarity)
        t1, t2 = f1['terms'], f2['terms']
        if t1 or t2:
            common_terms = t1 & t2
            term_sim = len(common_terms) / math.sqrt(len(t1) * len(t2)) if t1 and t2 else 0
            total_score += self.weights['terms'] * term_sim
            if term_sim > 0.3:
                matching_features.append(f"terms:{list(common_terms)[:5]}")

        # Metric key overlap
        m1, m2 = f1['metric_keys'], f2['metric_keys']
        if m1 or m2:
            metric_sim = len(m1 & m2) / len(m1 | m2) if (m1 | m2) else 0
            total_score += self.weights['metrics'] * metric_sim

        # Temporal similarity (same hour or day of week)
        temporal_score = 0
        if 'hour_of_day' in f1 and 'hour_of_day' in f2:
            hour_diff = abs(f1['hour_of_day'] - f2['hour_of_day'])
            temporal_score += (1 - hour_diff / 12) * 0.5

        if 'day_of_week' in f1 and 'day_of_week' in f2:
            if f1['day_of_week'] == f2['day_of_week']:
                temporal_score += 0.5
                matching_features.append("same_day_of_week")

        total_score += self.weights['temporal'] * temporal_score

        return (total_score, matching_features)

    def find_similar(
        self,
        incident: Incident,
        candidates: List[Tuple[Incident, Optional[Resolution]]],
        threshold: float = 0.3,
        top_k: int = 5
    ) -> List[SimilarIncident]:
        """Find similar incidents from candidates."""
        results = []

        for candidate_incident, resolution in candidates:
            if candidate_incident.incident_id == incident.incident_id:
                continue

            score, matching = self.compute_similarity(incident, candidate_incident)

            if score >= threshold:
                results.append(SimilarIncident(
                    incident=candidate_incident,
                    resolution=resolution,
                    similarity_score=score,
                    matching_features=matching
                ))

        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        return results[:top_k]


# =============================================================================
# Pattern Learning
# =============================================================================

class PatternLearner:
    """Learn patterns from resolved incidents."""

    def __init__(self):
        self._patterns: Dict[str, ResolutionPattern] = {}
        self._incident_clusters: Dict[str, List[str]] = defaultdict(list)

    def learn_from_resolution(
        self,
        incident: Incident,
        resolution: Resolution
    ) -> Optional[ResolutionPattern]:
        """Learn a pattern from a resolved incident."""
        # Extract key characteristics
        symptom_patterns = list(incident.symptoms)
        service_patterns = list(incident.affected_services)

        # Create metric conditions
        metric_conditions = {}
        for metric, value in incident.metrics.items():
            if 'error' in metric.lower() and value > 0:
                metric_conditions[metric] = ('>', 0)
            elif 'latency' in metric.lower() and value > 500:
                metric_conditions[metric] = ('>', 500)
            elif 'cpu' in metric.lower() and value > 70:
                metric_conditions[metric] = ('>', 70)
            elif 'memory' in metric.lower() and value > 80:
                metric_conditions[metric] = ('>', 80)

        # Create pattern ID
        pattern_key = self._create_pattern_key(
            symptom_patterns, service_patterns, metric_conditions
        )

        if pattern_key in self._patterns:
            # Update existing pattern
            pattern = self._patterns[pattern_key]
            pattern.total_count += 1
            if resolution.effectiveness_rating > 0.7:
                pattern.success_count += 1

            # Update confidence
            pattern.confidence = pattern.success_count / pattern.total_count
            pattern.updated_at = datetime.now()

            # Merge recommended actions
            for action in resolution.actions_taken:
                if action not in pattern.recommended_actions:
                    pattern.recommended_actions.append(action)

            return pattern

        else:
            # Create new pattern
            pattern = ResolutionPattern(
                pattern_id=pattern_key,
                name=f"Pattern from incident {incident.incident_id}",
                description=f"Learned from: {incident.title}",
                symptom_patterns=symptom_patterns,
                metric_conditions=metric_conditions,
                service_patterns=service_patterns,
                recommended_actions=resolution.actions_taken,
                root_cause_template=resolution.root_cause,
                confidence=1.0 if resolution.effectiveness_rating > 0.7 else 0.5,
                success_count=1 if resolution.effectiveness_rating > 0.7 else 0,
                total_count=1,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            self._patterns[pattern_key] = pattern
            return pattern

    def _create_pattern_key(
        self,
        symptoms: List[str],
        services: List[str],
        metrics: Dict[str, Tuple[str, float]]
    ) -> str:
        """Create a unique key for a pattern."""
        components = [
            sorted(symptoms),
            sorted(services),
            sorted(metrics.keys())
        ]
        key_str = json.dumps(components, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def match_patterns(
        self,
        incident: Incident,
        min_confidence: float = 0.5
    ) -> List[Tuple[ResolutionPattern, float]]:
        """Find matching patterns for an incident."""
        matches = []

        for pattern in self._patterns.values():
            if pattern.confidence < min_confidence:
                continue

            score = self._compute_pattern_match(incident, pattern)
            if score > 0.5:
                matches.append((pattern, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _compute_pattern_match(
        self,
        incident: Incident,
        pattern: ResolutionPattern
    ) -> float:
        """Compute match score between incident and pattern."""
        score = 0.0
        weight_sum = 0.0

        # Symptom match
        if pattern.symptom_patterns:
            symptom_set = set(incident.symptoms)
            pattern_symptoms = set(pattern.symptom_patterns)
            if pattern_symptoms:
                symptom_score = len(symptom_set & pattern_symptoms) / len(pattern_symptoms)
                score += symptom_score * 0.3
            weight_sum += 0.3

        # Service match
        if pattern.service_patterns:
            service_set = set(incident.affected_services)
            pattern_services = set(pattern.service_patterns)
            if pattern_services:
                service_score = len(service_set & pattern_services) / len(pattern_services)
                score += service_score * 0.3
            weight_sum += 0.3

        # Metric condition match
        if pattern.metric_conditions:
            metric_matches = 0
            for metric, (op, threshold) in pattern.metric_conditions.items():
                if metric in incident.metrics:
                    value = incident.metrics[metric]
                    if op == '>' and value > threshold:
                        metric_matches += 1
                    elif op == '<' and value < threshold:
                        metric_matches += 1
                    elif op == '=' and abs(value - threshold) < 0.01:
                        metric_matches += 1

            metric_score = metric_matches / len(pattern.metric_conditions)
            score += metric_score * 0.4
            weight_sum += 0.4

        return score / weight_sum if weight_sum > 0 else 0.0


# =============================================================================
# Knowledge Base
# =============================================================================

class KnowledgeBase:
    """
    Knowledge base for storing and retrieving incident learnings.
    """

    def __init__(self):
        self._incidents: Dict[str, Incident] = {}
        self._resolutions: Dict[str, Resolution] = {}
        self._patterns: Dict[str, ResolutionPattern] = {}

        self.feature_extractor = FeatureExtractor()
        self.similarity_matcher = SimilarityMatcher(self.feature_extractor)
        self.pattern_learner = PatternLearner()

    def add_incident(self, incident: Incident):
        """Add an incident to the knowledge base."""
        self._incidents[incident.incident_id] = incident
        self.feature_extractor.update_statistics(incident)

    def add_resolution(self, resolution: Resolution):
        """Add a resolution and learn from it."""
        self._resolutions[resolution.incident_id] = resolution

        # Get the incident
        incident = self._incidents.get(resolution.incident_id)
        if incident:
            # Learn pattern
            pattern = self.pattern_learner.learn_from_resolution(incident, resolution)
            if pattern:
                self._patterns[pattern.pattern_id] = pattern

    def find_similar_incidents(
        self,
        incident: Incident,
        threshold: float = 0.3,
        top_k: int = 5
    ) -> List[SimilarIncident]:
        """Find similar incidents with their resolutions."""
        candidates = [
            (inc, self._resolutions.get(inc.incident_id))
            for inc in self._incidents.values()
        ]

        return self.similarity_matcher.find_similar(
            incident, candidates, threshold, top_k
        )

    def get_recommended_actions(
        self,
        incident: Incident
    ) -> List[Tuple[str, float]]:
        """Get recommended actions for an incident."""
        recommendations = []

        # From similar incidents
        similar = self.find_similar_incidents(incident)
        for sim in similar:
            if sim.resolution:
                for action in sim.resolution.actions_taken:
                    recommendations.append((
                        action,
                        sim.similarity_score * sim.resolution.effectiveness_rating
                    ))

        # From patterns
        patterns = self.pattern_learner.match_patterns(incident)
        for pattern, match_score in patterns:
            for action in pattern.recommended_actions:
                recommendations.append((
                    action,
                    match_score * pattern.confidence
                ))

        # Aggregate and sort
        action_scores: Dict[str, float] = defaultdict(float)
        for action, score in recommendations:
            action_scores[action] = max(action_scores[action], score)

        sorted_actions = sorted(
            action_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_actions

    def get_root_cause_suggestions(
        self,
        incident: Incident
    ) -> List[Tuple[str, float]]:
        """Get root cause suggestions for an incident."""
        suggestions = []

        # From similar incidents
        similar = self.find_similar_incidents(incident)
        for sim in similar:
            if sim.resolution:
                suggestions.append((
                    sim.resolution.root_cause,
                    sim.similarity_score
                ))

        # From patterns
        patterns = self.pattern_learner.match_patterns(incident)
        for pattern, match_score in patterns:
            suggestions.append((
                pattern.root_cause_template,
                match_score * pattern.confidence
            ))

        # Deduplicate and sort
        cause_scores: Dict[str, float] = {}
        for cause, score in suggestions:
            if cause not in cause_scores or score > cause_scores[cause]:
                cause_scores[cause] = score

        return sorted(cause_scores.items(), key=lambda x: x[1], reverse=True)

    def export_knowledge(self) -> Dict[str, Any]:
        """Export knowledge base."""
        return {
            'incidents': {
                k: {
                    'incident_id': v.incident_id,
                    'title': v.title,
                    'severity': v.severity,
                    'affected_services': v.affected_services,
                    'symptoms': v.symptoms
                }
                for k, v in self._incidents.items()
            },
            'patterns': {
                k: {
                    'pattern_id': v.pattern_id,
                    'name': v.name,
                    'symptom_patterns': v.symptom_patterns,
                    'recommended_actions': v.recommended_actions,
                    'confidence': v.confidence,
                    'success_count': v.success_count,
                    'total_count': v.total_count
                }
                for k, v in self._patterns.items()
            }
        }


# =============================================================================
# Feedback Collection
# =============================================================================

@dataclass
class Feedback:
    """User feedback on remediation or suggestion."""
    feedback_id: str
    incident_id: str
    feedback_type: str  # 'resolution', 'suggestion', 'action'
    target_id: str  # ID of the thing being rated
    rating: float  # 0-1
    comment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class FeedbackCollector:
    """Collect and process user feedback for continuous improvement."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self._feedback: List[Feedback] = []
        self._action_ratings: Dict[str, List[float]] = defaultdict(list)
        self._pattern_ratings: Dict[str, List[float]] = defaultdict(list)

    def record_feedback(self, feedback: Feedback):
        """Record user feedback."""
        self._feedback.append(feedback)

        # Update ratings
        if feedback.feedback_type == 'action':
            self._action_ratings[feedback.target_id].append(feedback.rating)

        elif feedback.feedback_type == 'pattern':
            self._pattern_ratings[feedback.target_id].append(feedback.rating)

        # Update pattern confidence if needed
        self._update_pattern_confidence(feedback)

    def _update_pattern_confidence(self, feedback: Feedback):
        """Update pattern confidence based on feedback."""
        if feedback.feedback_type != 'pattern':
            return

        pattern = self.knowledge_base._patterns.get(feedback.target_id)
        if not pattern:
            return

        # Exponential moving average of ratings
        ratings = self._pattern_ratings[feedback.target_id]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            # Blend with existing confidence
            pattern.confidence = 0.7 * pattern.confidence + 0.3 * avg_rating
            pattern.updated_at = datetime.now()

    def get_action_effectiveness(self, action: str) -> Optional[float]:
        """Get effectiveness rating for an action."""
        ratings = self._action_ratings.get(action, [])
        if not ratings:
            return None
        return sum(ratings) / len(ratings)

    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improving the knowledge base."""
        suggestions = []

        # Low-rated patterns
        for pattern_id, ratings in self._pattern_ratings.items():
            avg = sum(ratings) / len(ratings) if ratings else 0
            if avg < 0.5:
                pattern = self.knowledge_base._patterns.get(pattern_id)
                if pattern:
                    suggestions.append(
                        f"Review pattern '{pattern.name}' - low user ratings ({avg:.2f})"
                    )

        # Low-rated actions
        for action, ratings in self._action_ratings.items():
            avg = sum(ratings) / len(ratings) if ratings else 0
            if avg < 0.5:
                suggestions.append(
                    f"Review action '{action}' - low effectiveness ({avg:.2f})"
                )

        return suggestions


# =============================================================================
# Incident Learner
# =============================================================================

class IncidentLearner:
    """
    Main incident learning system.

    Coordinates knowledge base, pattern learning, and feedback.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.feedback_collector = FeedbackCollector(self.knowledge_base)

    def process_incident(self, incident: Incident):
        """Process a new incident."""
        self.knowledge_base.add_incident(incident)

    def process_resolution(self, resolution: Resolution):
        """Process an incident resolution."""
        self.knowledge_base.add_resolution(resolution)

    def get_suggestions(self, incident: Incident) -> Dict[str, Any]:
        """Get suggestions for handling an incident."""
        similar = self.knowledge_base.find_similar_incidents(incident)
        actions = self.knowledge_base.get_recommended_actions(incident)
        root_causes = self.knowledge_base.get_root_cause_suggestions(incident)

        return {
            'similar_incidents': [
                {
                    'incident_id': s.incident.incident_id,
                    'title': s.incident.title,
                    'similarity': s.similarity_score,
                    'matching_features': s.matching_features,
                    'resolution': {
                        'root_cause': s.resolution.root_cause,
                        'actions': s.resolution.actions_taken,
                        'effectiveness': s.resolution.effectiveness_rating
                    } if s.resolution else None
                }
                for s in similar
            ],
            'recommended_actions': [
                {'action': a, 'confidence': c}
                for a, c in actions[:5]
            ],
            'root_cause_suggestions': [
                {'cause': c, 'confidence': conf}
                for c, conf in root_causes[:3]
            ]
        }

    def record_feedback(
        self,
        incident_id: str,
        feedback_type: str,
        target_id: str,
        rating: float,
        comment: str = None
    ):
        """Record feedback for continuous improvement."""
        import uuid

        feedback = Feedback(
            feedback_id=str(uuid.uuid4())[:8],
            incident_id=incident_id,
            feedback_type=feedback_type,
            target_id=target_id,
            rating=rating,
            comment=comment
        )

        self.feedback_collector.record_feedback(feedback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'total_incidents': len(self.knowledge_base._incidents),
            'total_resolutions': len(self.knowledge_base._resolutions),
            'total_patterns': len(self.knowledge_base._patterns),
            'total_feedback': len(self.feedback_collector._feedback),
            'improvement_suggestions': self.feedback_collector.get_improvement_suggestions()
        }
