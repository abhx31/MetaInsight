"""
Risk Scoring Algorithm for Risk Detection Agent.
Implements the 4-factor scoring system: Impact + Probability + Dependencies + Timing
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .schemas import Severity, ScoreBreakdown


class RiskScorer:
    """
    Risk Scoring Engine implementing the formula:
    TOTAL_SCORE = IMPACT(0-40) + PROBABILITY(0-30) + DEPENDENCIES(0-20) + TIMING(0-10)
    """
    
    # ========================================================================
    # IMPACT KEYWORDS AND SCORES
    # ========================================================================
    
    IMPACT_KEYWORDS = {
        "critical": {
            "keywords": [
                "critical", "blocker", "showstopper", "catastrophic", "fatal",
                "project failure", "cancellation", "legal", "compliance violation",
                "safety", "security breach", "data loss"
            ],
            "score_range": (35, 40),
            "description": "Threatens entire project success"
        },
        "high": {
            "keywords": [
                "major", "significant", "substantial", "severe",
                "key deliverable", "critical path", "major delay"
            ],
            "score_range": (25, 34),
            "description": "Major delay >4 weeks or key deliverable at risk"
        },
        "medium": {
            "keywords": [
                "moderate", "notable", "considerable",
                "workaround available", "partial"
            ],
            "score_range": (15, 24),
            "description": "Moderate delay 1-4 weeks, workarounds exist"
        },
        "low": {
            "keywords": [
                "minor", "small", "negligible", "trivial",
                "inconvenience", "easy fix"
            ],
            "score_range": (0, 14),
            "description": "Minor inconvenience, easy workarounds"
        }
    }
    
    # ========================================================================
    # PROBABILITY KEYWORDS AND SCORES
    # ========================================================================
    
    PROBABILITY_KEYWORDS = {
        "very_likely": {
            "keywords": [
                "pending", "unclear", "not decided", "tbd", "to be determined",
                "unknown", "unconfirmed", "uncertain", "not sure",
                "waiting for", "awaiting", "blocked"
            ],
            "score_range": (25, 30),
            "percentage_range": (70, 100),
            "description": "70-100% probability"
        },
        "likely": {
            "keywords": [
                "tentative", "preliminary", "assuming", "assumed",
                "provisional", "draft", "initial", "expected"
            ],
            "score_range": (18, 24),
            "percentage_range": (40, 69),
            "description": "40-69% probability"
        },
        "possible": {
            "keywords": [
                "might", "could", "maybe", "perhaps", "possibly",
                "potential", "may"
            ],
            "score_range": (10, 17),
            "percentage_range": (20, 39),
            "description": "20-39% probability"
        },
        "unlikely": {
            "keywords": [
                "unlikely", "remote", "edge case", "rare",
                "improbable", "doubtful"
            ],
            "score_range": (0, 9),
            "percentage_range": (0, 19),
            "description": "<20% probability"
        }
    }
    
    # ========================================================================
    # UNCERTAINTY MARKERS FOR DETECTION
    # ========================================================================
    
    HIGH_UNCERTAINTY_MARKERS = [
        "maybe", "perhaps", "possibly", "might", "could",
        "unclear", "tbd", "to be determined", "pending",
        "not sure", "uncertain", "unknown", "unconfirmed",
        "tentative", "provisional", "subject to change",
        "assuming", "hopefully", "ideally", "potentially"
    ]
    
    MEDIUM_UNCERTAINTY_MARKERS = [
        "likely", "probably", "should", "expected",
        "preliminary", "draft", "initial", "estimated",
        "approximately", "around", "roughly", "about"
    ]
    
    ASSUMPTION_MARKERS = [
        "assuming that", "provided that", "given that",
        "if ", "once ", "when ", "as long as",
        "depends on", "contingent upon", "subject to",
        "assuming approval", "assuming budget", "assuming resources"
    ]
    
    QUESTION_MARKERS = [
        "?",
        "need to confirm", "need to decide", "need to determine",
        "waiting for", "pending decision", "under review",
        "to be discussed", "to be finalized", "tba"
    ]
    
    MISSING_INFO_MARKERS = [
        "details missing", "not specified", "no timeline",
        "unassigned", "owner tbd", "timeline unclear",
        "budget not finalized", "scope undefined",
        "[blank]", "n/a", "tba", "not available"
    ]
    
    def __init__(self):
        """Initialize the scorer."""
        pass
    
    # ========================================================================
    # MAIN SCORING METHODS
    # ========================================================================
    
    def calculate_total_score(
        self,
        impact_score: int,
        probability_score: int,
        dependency_score: int,
        timing_score: int
    ) -> int:
        """
        Calculate total risk score from components.
        
        Args:
            impact_score: 0-40
            probability_score: 0-30
            dependency_score: 0-20
            timing_score: 0-10
            
        Returns:
            Total score 0-100
        """
        return min(100, max(0,
            impact_score + probability_score + dependency_score + timing_score
        ))
    
    def calculate_impact_score(
        self,
        description: str,
        affected_deliverables: List[str] = None,
        budget_impact_percentage: float = None,
        has_legal_compliance_risk: bool = False,
        has_safety_risk: bool = False
    ) -> Tuple[int, str]:
        """
        Calculate impact score (0-40).
        
        Returns:
            Tuple of (score, reasoning)
        """
        description_lower = description.lower()
        affected_deliverables = affected_deliverables or []
        
        # Check for critical indicators first
        if has_safety_risk or has_legal_compliance_risk:
            return (40, "Safety or legal/compliance risk - maximum impact")
        
        if budget_impact_percentage and budget_impact_percentage > 20:
            return (38, f"Budget impact >{budget_impact_percentage}% - critical financial exposure")
        
        # Check keywords
        for level, config in self.IMPACT_KEYWORDS.items():
            for keyword in config["keywords"]:
                if keyword in description_lower:
                    min_score, max_score = config["score_range"]
                    # Adjust within range based on deliverables affected
                    deliverable_bonus = min(5, len(affected_deliverables))
                    score = min(max_score, min_score + deliverable_bonus)
                    return (score, f"{config['description']} - keyword '{keyword}' detected")
        
        # Default based on deliverables
        if len(affected_deliverables) >= 3:
            return (25, "Multiple deliverables affected")
        elif len(affected_deliverables) >= 1:
            return (18, "Some deliverables affected")
        else:
            return (12, "Limited direct impact identified")
    
    def calculate_probability_score(
        self,
        description: str,
        uncertainty_markers_found: List[str] = None,
        explicit_uncertainty: bool = False,
        historical_occurrence_rate: float = None
    ) -> Tuple[int, str]:
        """
        Calculate probability score (0-30).
        
        Returns:
            Tuple of (score, reasoning)
        """
        description_lower = description.lower()
        uncertainty_markers_found = uncertainty_markers_found or []
        
        # Explicit uncertainty is highest weight
        if explicit_uncertainty or any(
            marker in description_lower 
            for marker in ["pending", "tbd", "unclear", "not decided"]
        ):
            marker_count = len(uncertainty_markers_found)
            score = min(30, 25 + marker_count)
            return (score, f"Explicit uncertainty stated - {marker_count} uncertainty markers found")
        
        # Check keywords
        for level, config in self.PROBABILITY_KEYWORDS.items():
            for keyword in config["keywords"]:
                if keyword in description_lower:
                    min_score, max_score = config["score_range"]
                    score = (min_score + max_score) // 2
                    return (score, f"{config['description']} - keyword '{keyword}' detected")
        
        # Historical data override
        if historical_occurrence_rate is not None:
            if historical_occurrence_rate >= 0.7:
                return (28, f"Historical rate {historical_occurrence_rate:.0%} indicates high probability")
            elif historical_occurrence_rate >= 0.4:
                return (20, f"Historical rate {historical_occurrence_rate:.0%} indicates likely")
            elif historical_occurrence_rate >= 0.2:
                return (12, f"Historical rate {historical_occurrence_rate:.0%} indicates possible")
            else:
                return (5, f"Historical rate {historical_occurrence_rate:.0%} indicates unlikely")
        
        # Default medium probability
        return (15, "No strong probability indicators - assuming moderate likelihood")
    
    def calculate_dependency_score(
        self,
        blocked_tasks: List[str] = None,
        blocked_decisions: List[str] = None,
        on_critical_path: bool = False,
        is_single_point_of_failure: bool = False
    ) -> Tuple[int, str]:
        """
        Calculate dependency score (0-20).
        
        Formula:
        base_score = blocked_items_count * 3
        if on_critical_path: base_score += 5
        if single_point_of_failure: base_score += 3
        cap at 20
        
        Returns:
            Tuple of (score, reasoning)
        """
        blocked_tasks = blocked_tasks or []
        blocked_decisions = blocked_decisions or []
        
        total_blocked = len(blocked_tasks) + len(blocked_decisions)
        
        # Base calculation
        base_score = total_blocked * 3
        
        # Bonuses
        if on_critical_path:
            base_score += 5
        if is_single_point_of_failure:
            base_score += 3
        
        # Cap at 20
        score = min(20, base_score)
        
        # Generate reasoning
        reasons = []
        if total_blocked > 0:
            reasons.append(f"Blocks {total_blocked} items ({len(blocked_tasks)} tasks, {len(blocked_decisions)} decisions)")
        if on_critical_path:
            reasons.append("On critical path")
        if is_single_point_of_failure:
            reasons.append("Single point of failure")
        
        if not reasons:
            reasons.append("Minimal dependency impact")
        
        return (score, "; ".join(reasons))
    
    def calculate_timing_score(
        self,
        deadline: Optional[str] = None,
        days_until_critical: Optional[int] = None,
        is_already_delayed: bool = False,
        urgency_keywords: List[str] = None
    ) -> Tuple[int, str]:
        """
        Calculate timing score (0-10).
        
        Formula:
        if days_until_critical <= 2: timing_score = 10
        elif days_until_critical <= 7: timing_score = 7
        elif days_until_critical <= 14: timing_score = 5
        elif days_until_critical <= 30: timing_score = 3
        else: timing_score = 1
        
        Returns:
            Tuple of (score, reasoning)
        """
        urgency_keywords = urgency_keywords or []
        
        # Already delayed is highest urgency
        if is_already_delayed:
            return (10, "Already causing delays - immediate action required")
        
        # Check urgency keywords
        urgent_markers = ["urgent", "asap", "immediately", "critical", "emergency"]
        if any(marker in (kw.lower() for kw in urgency_keywords) for marker in urgent_markers):
            return (9, "Urgency markers detected in document")
        
        # Calculate from days
        if days_until_critical is not None:
            if days_until_critical <= 2:
                return (10, f"Only {days_until_critical} days until critical - immediate")
            elif days_until_critical <= 7:
                return (7, f"{days_until_critical} days until critical - near-term")
            elif days_until_critical <= 14:
                return (5, f"{days_until_critical} days until critical - short-term")
            elif days_until_critical <= 30:
                return (3, f"{days_until_critical} days until critical - medium-term")
            else:
                return (1, f"{days_until_critical} days until critical - long-term")
        
        # Parse deadline string if provided
        if deadline:
            try:
                deadline_date = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                today = datetime.now(deadline_date.tzinfo) if deadline_date.tzinfo else datetime.now()
                days = (deadline_date - today).days
                return self.calculate_timing_score(days_until_critical=days)
            except (ValueError, TypeError):
                pass
        
        # Default medium urgency
        return (4, "No specific deadline - assuming medium-term")
    
    # ========================================================================
    # SEVERITY CLASSIFICATION
    # ========================================================================
    
    def classify_severity(self, total_score: int) -> Severity:
        """
        Classify risk severity based on total score.
        
        - CRITICAL: 80-100
        - HIGH: 60-79
        - MEDIUM: 40-59
        - LOW: 0-39
        """
        if total_score >= 80:
            return Severity.CRITICAL
        elif total_score >= 60:
            return Severity.HIGH
        elif total_score >= 40:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    # ========================================================================
    # FULL RISK SCORING
    # ========================================================================
    
    def score_risk(
        self,
        description: str,
        affected_deliverables: List[str] = None,
        blocked_tasks: List[str] = None,
        blocked_decisions: List[str] = None,
        deadline: Optional[str] = None,
        on_critical_path: bool = False,
        is_single_point_of_failure: bool = False,
        explicit_uncertainty: bool = False,
        uncertainty_markers: List[str] = None,
        budget_impact_percentage: float = None,
        has_legal_compliance_risk: bool = False,
        has_safety_risk: bool = False,
        is_already_delayed: bool = False,
        days_until_critical: Optional[int] = None
    ) -> Tuple[int, Severity, ScoreBreakdown]:
        """
        Calculate complete risk score with breakdown.
        
        Returns:
            Tuple of (total_score, severity, score_breakdown)
        """
        # Calculate each component
        impact, impact_reasoning = self.calculate_impact_score(
            description=description,
            affected_deliverables=affected_deliverables,
            budget_impact_percentage=budget_impact_percentage,
            has_legal_compliance_risk=has_legal_compliance_risk,
            has_safety_risk=has_safety_risk
        )
        
        probability, probability_reasoning = self.calculate_probability_score(
            description=description,
            uncertainty_markers_found=uncertainty_markers,
            explicit_uncertainty=explicit_uncertainty
        )
        
        dependencies, dependencies_reasoning = self.calculate_dependency_score(
            blocked_tasks=blocked_tasks,
            blocked_decisions=blocked_decisions,
            on_critical_path=on_critical_path,
            is_single_point_of_failure=is_single_point_of_failure
        )
        
        timing, timing_reasoning = self.calculate_timing_score(
            deadline=deadline,
            days_until_critical=days_until_critical,
            is_already_delayed=is_already_delayed
        )
        
        # Calculate total
        total_score = self.calculate_total_score(impact, probability, dependencies, timing)
        severity = self.classify_severity(total_score)
        
        # Build breakdown
        breakdown = ScoreBreakdown(
            impact=impact,
            impact_reasoning=impact_reasoning,
            probability=probability,
            probability_reasoning=probability_reasoning,
            dependencies=dependencies,
            dependencies_reasoning=dependencies_reasoning,
            timing=timing,
            timing_reasoning=timing_reasoning
        )
        
        return (total_score, severity, breakdown)
    
    # ========================================================================
    # TEXT ANALYSIS HELPERS
    # ========================================================================
    
    def detect_uncertainty_markers(self, text: str) -> Dict[str, List[str]]:
        """
        Detect all uncertainty markers in text.
        
        Returns:
            Dict with keys: high, medium, assumptions, questions, missing_info
        """
        text_lower = text.lower()
        
        return {
            "high": [m for m in self.HIGH_UNCERTAINTY_MARKERS if m in text_lower],
            "medium": [m for m in self.MEDIUM_UNCERTAINTY_MARKERS if m in text_lower],
            "assumptions": [m for m in self.ASSUMPTION_MARKERS if m in text_lower],
            "questions": [m for m in self.QUESTION_MARKERS if m in text_lower],
            "missing_info": [m for m in self.MISSING_INFO_MARKERS if m in text_lower]
        }
    
    def calculate_document_uncertainty_level(self, text: str) -> Tuple[float, str]:
        """
        Calculate overall uncertainty level of a document.
        
        Returns:
            Tuple of (uncertainty_score 0-1, description)
        """
        markers = self.detect_uncertainty_markers(text)
        
        # Weight different marker types
        high_count = len(markers["high"]) * 3
        medium_count = len(markers["medium"]) * 2
        assumption_count = len(markers["assumptions"]) * 2
        question_count = len(markers["questions"]) * 2
        missing_count = len(markers["missing_info"]) * 3
        
        total_weight = high_count + medium_count + assumption_count + question_count + missing_count
        
        # Normalize to 0-1 (assume 50 weighted markers = max uncertainty)
        uncertainty_score = min(1.0, total_weight / 50)
        
        if uncertainty_score >= 0.7:
            description = "High uncertainty - document contains many unresolved items"
        elif uncertainty_score >= 0.4:
            description = "Moderate uncertainty - several items need clarification"
        elif uncertainty_score >= 0.2:
            description = "Low uncertainty - some minor items to resolve"
        else:
            description = "Very low uncertainty - document is well-defined"
        
        return (uncertainty_score, description)
