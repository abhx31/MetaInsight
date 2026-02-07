"""
Main Risk Detection Agent (Agent 3)
Orchestrates all 3 phases of risk detection and produces comprehensive risk analysis.
"""

from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from .schemas import (
    RiskDetectorInput,
    RiskDetectorOutput,
    Risk,
    Assumption,
    OpenQuestion,
    MissingInformation,
    Conflict,
    RiskSummary,
    AnalysisMetadata,
    SeverityCount,
    CategoryCount,
    Evidence,
    RelatedItems,
    ImpactAnalysis,
    ProbabilityAssessment,
    Severity,
    RiskCategory,
    AssumptionType,
    AssumptionImpact,
    AssumptionValidation,
    QuestionBlocks,
    HowToObtain,
)
from .scoring import RiskScorer
from .detectors import LinguisticDetector, ContextualDetector, EnrichmentDetector

logger = logging.getLogger(__name__)


class RiskDetectorAgent:
    """
    Agent 3: Risk Detection and Analysis
    
    Performs 3-phase analysis:
    - Phase 1: Linguistic Analysis (keyword detection)
    - Phase 2: Contextual Analysis (deep reasoning)
    - Phase 3: Cross-Agent Enrichment (integration with other agents)
    """
    
    def __init__(self):
        """Initialize the risk detector agent."""
        self.scorer = RiskScorer()
        self.linguistic_detector = LinguisticDetector()
        self.contextual_detector = ContextualDetector()
        self.enrichment_detector = EnrichmentDetector()
        self.risk_counter = 0
        self.assumption_counter = 0
        self.question_counter = 0
        self.missing_info_counter = 0
        self.conflict_counter = 0
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID for risks, assumptions, etc."""
        if prefix == "R":
            self.risk_counter += 1
            return f"R{self.risk_counter:03d}"
        elif prefix == "A":
            self.assumption_counter += 1
            return f"A{self.assumption_counter:03d}"
        elif prefix == "Q":
            self.question_counter += 1
            return f"Q{self.question_counter:03d}"
        elif prefix == "MI":
            self.missing_info_counter += 1
            return f"MI{self.missing_info_counter:03d}"
        elif prefix == "C":
            self.conflict_counter += 1
            return f"C{self.conflict_counter:03d}"
        return f"{prefix}001"
    
    def analyze(self, input_data: RiskDetectorInput) -> RiskDetectorOutput:
        """
        Main entry point for risk analysis.
        
        Args:
            input_data: Complete input including document and optional agent outputs
            
        Returns:
            Comprehensive risk analysis output
        """
        logger.info("Starting Risk Detection Agent analysis")
        
        # Reset counters
        self.risk_counter = 0
        self.assumption_counter = 0
        self.question_counter = 0
        self.missing_info_counter = 0
        self.conflict_counter = 0
        
        document_content = input_data.document.content
        
        # Create metadata
        metadata = AnalysisMetadata(
            document_analyzed=input_data.document.metadata.title if input_data.document.metadata else None,
            analysis_depth=input_data.request.depth if input_data.request else "comprehensive",
            cross_agent_data_used={
                "summarizer": input_data.summarizer_output is not None,
                "taskmaster": input_data.taskmaster_output is not None
            }
        )
        
        # ====================================================================
        # PHASE 1: LINGUISTIC ANALYSIS
        # ====================================================================
        logger.info("Phase 1: Linguistic Analysis")
        phase1_results = self.linguistic_detector.analyze(document_content)
        
        # Extract assumptions from linguistic markers
        assumptions = self._extract_assumptions_from_markers(
            document_content,
            phase1_results["uncertainty_markers"]
        )
        
        # Extract open questions
        open_questions = self._extract_questions_from_markers(
            document_content,
            phase1_results["uncertainty_markers"]
        )
        
        # Extract missing information
        missing_info = self._extract_missing_info_from_markers(
            document_content,
            phase1_results["uncertainty_markers"]
        )
        
        # ====================================================================
        # PHASE 2: CONTEXTUAL ANALYSIS
        # ====================================================================
        logger.info("Phase 2: Contextual Analysis")
        phase2_results = self.contextual_detector.analyze(
            document_content,
            tasks=[t.dict() for t in input_data.taskmaster_output.tasks] if input_data.taskmaster_output else None,
            dependency_graph=input_data.taskmaster_output.dependency_graph if input_data.taskmaster_output else None
        )
        
        # Convert contextual risks to Risk objects
        risks = self._convert_contextual_risks_to_objects(
            phase2_results["contextual_risks"],
            document_content
        )
        
        # ====================================================================
        # PHASE 3: CROSS-AGENT ENRICHMENT
        # ====================================================================
        logger.info("Phase 3: Cross-Agent Enrichment")
        if input_data.taskmaster_output or input_data.summarizer_output:
            # Prepare existing risks as dicts
            risk_dicts = [self._risk_to_dict(r) for r in risks]
            
            phase3_results = self.enrichment_detector.analyze(
                identified_risks=risk_dicts,
                taskmaster_output=input_data.taskmaster_output,
                summarizer_output=input_data.summarizer_output
            )
            
            # Update risks with enrichment data
            risks = self._enrich_risks_with_phase3(risks, phase3_results)
            
            # Add new risks discovered from cross-referencing
            new_risks = self._convert_phase3_risks_to_objects(
                phase3_results["new_risks"],
                document_content
            )
            risks.extend(new_risks)
            
            # Add conflicts
            conflicts = self._convert_conflicts_to_objects(phase3_results["conflicts"])
        else:
            conflicts = []
        
        # ====================================================================
        # GENERATE SUMMARY
        # ====================================================================
        summary = self._generate_summary(
            risks, assumptions, open_questions, missing_info, conflicts,
            metadata.cross_agent_data_used
        )
        
        # ====================================================================
        # CONSTRUCT FINAL OUTPUT
        # ====================================================================
        output = RiskDetectorOutput(
            analysis_metadata=metadata,
            risks=risks,
            assumptions=assumptions,
            open_questions=open_questions,
            missing_information=missing_info,
            conflict_detection=conflicts,
            risk_patterns_detected=[],  # Could be enhanced with pattern matching
            summary=summary
        )
        
        logger.info(f"Risk analysis complete: {len(risks)} risks, {len(assumptions)} assumptions, {len(open_questions)} questions")
        return output
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_assumptions_from_markers(
        self,
        document_content: str,
        uncertainty_markers: Dict
    ) -> List[Assumption]:
        """Extract assumptions from linguistic markers."""
        assumptions = []
        assumption_markers = uncertainty_markers.get("assumptions", [])
        
        for marker, context in assumption_markers[:5]:  # Limit to top 5
            assumption_id = self._generate_id("A")
            
            # Determine type (explicit since we found markers)
            assumption_type = AssumptionType.EXPLICIT
            
            # Extract evidence
            evidence = Evidence(
                direct_quotes=[context],
                uncertainty_markers=[marker]
            )
            
            # Create assumption object
            assumption = Assumption(
                id=assumption_id,
                assumption=context,
                type=assumption_type,
                confidence=0.6,  # Medium confidence for marker-based
                confidence_reasoning=f"Assumption indicator '{marker}' found in text",
                evidence=evidence,
                impact_if_false=AssumptionImpact(
                    severity=Severity.MEDIUM,
                    description="Requires validation to assess full impact",
                    affected_risks=[],
                    affected_tasks=[]
                ),
                validation=AssumptionValidation(
                    method="Confirm with stakeholders mentioned in context",
                    status="not_started"
                )
            )
            
            assumptions.append(assumption)
        
        return assumptions
    
    def _extract_questions_from_markers(
        self,
        document_content: str,
        uncertainty_markers: Dict
    ) -> List[OpenQuestion]:
        """Extract open questions from linguistic markers."""
        questions = []
        question_markers = uncertainty_markers.get("questions", [])
        
        for marker, context in question_markers[:5]:  # Limit to top 5
            question_id = self._generate_id("Q")
            
            question = OpenQuestion(
                id=question_id,
                question=context if context.endswith("?") else f"Resolution needed: {context}",
                context=context,
                criticality=Severity.MEDIUM,
                criticality_reasoning="Unresolved question creates uncertainty",
                blocks=QuestionBlocks(),
                consequence_if_unanswered="May cause delays or incorrect assumptions",
                who_can_answer=["Project stakeholders"],
                suggested_resolution="Schedule clarification meeting"
            )
            
            questions.append(question)
        
        return questions
    
    def _extract_missing_info_from_markers(
        self,
        document_content: str,
        uncertainty_markers: Dict
    ) -> List[MissingInformation]:
        """Extract missing information from linguistic markers."""
        missing_list = []
        missing_markers = uncertainty_markers.get("missing_info", [])
        
        for marker, context in missing_markers[:5]:  # Limit to top 5
            missing_id = self._generate_id("MI")
            
            missing = MissingInformation(
                id=missing_id,
                item=context,
                importance=Severity.MEDIUM,
                reasoning=f"Missing information indicator '{marker}' detected",
                impacts=["Creates uncertainty in planning"],
                current_gap=context,
                how_to_obtain=HowToObtain(
                    source="Document owner or stakeholders",
                    method="Request clarification"
                )
            )
            
            missing_list.append(missing)
        
        return missing_list
    
    def _convert_contextual_risks_to_objects(
        self,
        contextual_risks: List[Dict],
        document_content: str
    ) -> List[Risk]:
        """Convert contextual risk dicts to Risk objects with scoring."""
        risks = []
        
        for risk_dict in contextual_risks:
            risk_id = self._generate_id("R")
            
            # Determine category
            category_str = risk_dict.get("category", "resource")
            try:
                category = RiskCategory(category_str)
            except ValueError:
                category = RiskCategory.RESOURCE
            
            # Score the risk
            total_score, severity, score_breakdown = self.scorer.score_risk(
                description=risk_dict.get("description", ""),
                explicit_uncertainty=True,
                uncertainty_markers=[risk_dict.get("type", "")],
            )
            
            # Create evidence
            evidence = Evidence(
                direct_quotes=[risk_dict.get("evidence", "")[:200]],
                uncertainty_markers=[risk_dict.get("type", "")]
            )
            
            # Create risk object
            risk = Risk(
                id=risk_id,
                title=risk_dict.get("description", "Untitled risk")[:100],
                description=risk_dict.get("description", "No description"),
                category=category,
                severity=severity,
                risk_score=total_score,
                score_breakdown=score_breakdown,
                evidence=evidence,
                related_items=RelatedItems(),
                impact_analysis=ImpactAnalysis(
                    if_materializes="Requires mitigation",
                    affected_deliverables=[],
                    affected_stakeholders=[],
                    cascade_effects=[]
                ),
                probability_assessment=ProbabilityAssessment(
                    likelihood="medium",
                    likelihood_percentage=50,
                    reasoning="Based on contextual analysis",
                    contributing_factors=[]
                )
            )
            
            risks.append(risk)
        
        return risks
    
    def _convert_phase3_risks_to_objects(
        self,
        new_risks: List[Dict],
        document_content: str
    ) -> List[Risk]:
        """Convert Phase 3 new risks to Risk objects."""
        risks = []
        
        for risk_dict in new_risks:
            # Use existing risk_id or generate new one
            risk_id = risk_dict.get("risk_id", self._generate_id("R"))
            
            # Get category
            category_str = risk_dict.get("category", RiskCategory.RESOURCE)
            if isinstance(category_str, str):
                try:
                    category = RiskCategory(category_str.lower())
                except ValueError:
                    category = RiskCategory.RESOURCE
            else:
                category = category_str
            
            # Get severity hint or calculate
            severity_hint = risk_dict.get("severity_hint", Severity.MEDIUM)
            if isinstance(severity_hint, str):
                try:
                    severity = Severity(severity_hint.lower())
                except ValueError:
                    severity = Severity.MEDIUM
            else:
                severity = severity_hint
            
            # Calculate score
            total_score = 50  # Default medium score
            if severity == Severity.CRITICAL:
                total_score = 85
            elif severity == Severity.HIGH:
                total_score = 70
            elif severity == Severity.LOW:
                total_score = 30
            
            # Create basic score breakdown
            impact_score = int(total_score * 0.4)
            prob_score = int(total_score * 0.3)
            dep_score = int(total_score * 0.2)
            timing_score = int(total_score * 0.1)
            
            from .schemas import ScoreBreakdown
            score_breakdown = ScoreBreakdown(
                impact=impact_score,
                impact_reasoning="Generated from phase 3 analysis",
                probability=prob_score,
                probability_reasoning="Based on cross-agent data",
                dependencies=dep_score,
                dependencies_reasoning="Calculated from task dependencies",
                timing=timing_score,
                timing_reasoning="Estimated from context"
            )
            
            evidence = Evidence(
                direct_quotes=[risk_dict.get("evidence", "")[:200]],
                uncertainty_markers=[]
            )
            
            related_items = RelatedItems(
                tasks=risk_dict.get("related_tasks", []),
                decisions=risk_dict.get("related_decisions", []),
                assumptions=[]
            )
            
            risk = Risk(
                id=risk_id,
                title=risk_dict.get("title", "Untitled risk"),
                description=risk_dict.get("description", "No description"),
                category=category,
                severity=severity,
                risk_score=total_score,
                score_breakdown=score_breakdown,
                evidence=evidence,
                related_items=related_items,
                impact_analysis=ImpactAnalysis(
                    if_materializes="Requires attention",
                    affected_deliverables=[],
                    affected_stakeholders=[],
                    cascade_effects=[]
                ),
                probability_assessment=ProbabilityAssessment(
                    likelihood=severity.value,
                    likelihood_percentage=60,
                    reasoning="Identified through cross-agent analysis",
                    contributing_factors=[]
                )
            )
            
            risks.append(risk)
        
        return risks
    
    def _enrich_risks_with_phase3(
        self,
        risks: List[Risk],
        phase3_results: Dict
    ) -> List[Risk]:
        """Enrich existing risks with Phase 3 data."""
        enriched_risks_data = phase3_results.get("enriched_risks", [])
        
        # Create a mapping
        for i, risk in enumerate(risks):
            if i < len(enriched_risks_data):
                enriched_data = enriched_risks_data[i]
                
                # Update related items
                risk.related_items.tasks = enriched_data.get("related_tasks", [])
                risk.related_items.decisions = enriched_data.get("related_decisions", [])
        
        return risks
    
    def _convert_conflicts_to_objects(self, conflicts: List[Dict]) -> List[Conflict]:
        """Convert conflict dicts to Conflict objects."""
        from .schemas import ConflictingItems, ConflictRecommendation, ConflictItem, ConflictType
        
        conflict_objects = []
        
        for conflict_dict in conflicts:
            conflict_id = conflict_dict.get("conflict_id", self._generate_id("C"))
            
            # Determine conflict type
            conflict_type_str = conflict_dict.get("type", "timeline_conflict")
            try:
                conflict_type = ConflictType(conflict_type_str)
            except ValueError:
                conflict_type = ConflictType.TIMELINE_CONFLICT
            
            # Get severity
            severity = conflict_dict.get("severity", Severity.MEDIUM)
            if isinstance(severity, str):
                try:
                    severity = Severity(severity.lower())
                except ValueError:
                    severity = Severity.MEDIUM
            
            # Create conflicting items
            item_a_data = conflict_dict.get("item_a", {})
            item_b_data = conflict_dict.get("item_b", {})
            
            conflicting_items = ConflictingItems(
                item_a=ConflictItem(**item_a_data) if item_a_data else None,
                item_b=ConflictItem(**item_b_data) if item_b_data else None
            )
            
            # Create recommendations (placeholder)
            recommendation = ConflictRecommendation(
                option_a="Adjust timeline to resolve conflict",
                option_b="Reallocate resources",
                option_c="Escalate for decision"
            )
            
            conflict = Conflict(
                id=conflict_id,
                type=conflict_type,
                severity=severity,
                description=conflict_dict.get("description", "Conflict detected"),
                conflicting_items=conflicting_items,
                impact=conflict_dict.get("description", "May cause delays"),
                recommendation=recommendation
            )
            
            conflict_objects.append(conflict)
        
        return conflict_objects
    
    def _generate_summary(
        self,
        risks: List[Risk],
        assumptions: List[Assumption],
        questions: List[OpenQuestion],
        missing_info: List[MissingInformation],
        conflicts: List[Conflict],
        cross_agent_data_used: Dict[str, bool]
    ) -> RiskSummary:
        """Generate summary statistics."""
        
        # Count by severity
        severity_count = SeverityCount()
        for risk in risks:
            if risk.severity == Severity.CRITICAL:
                severity_count.critical += 1
            elif risk.severity == Severity.HIGH:
                severity_count.high += 1
            elif risk.severity == Severity.MEDIUM:
                severity_count.medium += 1
            else:
                severity_count.low += 1
        
        # Count by category
        category_count = CategoryCount()
        for risk in risks:
            cat = risk.category.value
            if hasattr(category_count, cat):
                setattr(category_count, cat, getattr(category_count, cat) + 1)
        
        # Determine overall risk level
        if severity_count.critical > 0:
            overall_risk_level = Severity.CRITICAL
            overall_reasoning = f"{severity_count.critical} critical risks require immediate attention"
        elif severity_count.high > 0:
            overall_risk_level = Severity.HIGH
            overall_reasoning = f"{severity_count.high} high-severity risks need action this week"
        elif severity_count.medium > 0:
            overall_risk_level = Severity.MEDIUM
            overall_reasoning = f"{severity_count.medium} medium risks should be monitored"
        else:
            overall_risk_level = Severity.LOW
            overall_reasoning = "Low overall risk profile"
        
        # Key concerns (top 3 critical/high risks)
        key_concerns = [
            risk.title for risk in sorted(risks, key=lambda r: r.risk_score, reverse=True)[:3]
        ]
        
        # Confidence factors
        confidence_factors = ["Full document analyzed"]
        if cross_agent_data_used.get("taskmaster"):
            confidence_factors.append("Task Master data integrated")
        if cross_agent_data_used.get("summarizer"):
            confidence_factors.append("Summarizer data integrated")
        
        confidence = 0.7 + (0.075 * len([v for v in cross_agent_data_used.values() if v]))
        
        summary = RiskSummary(
            total_risks=len(risks),
            by_severity=severity_count,
            by_category=category_count,
            total_assumptions=len(assumptions),
            assumptions_needing_validation=len([a for a in assumptions if a.confidence < 0.7]),
            open_questions=len(questions),
            critical_questions=len([q for q in questions if q.criticality == Severity.CRITICAL]),
            missing_information_items=len(missing_info),
            critical_missing=len([m for m in missing_info if m.importance == Severity.CRITICAL]),
            conflicts_detected=len(conflicts),
            overall_risk_level=overall_risk_level,
            overall_risk_reasoning=overall_reasoning,
            key_concerns=key_concerns,
            immediate_actions_required=[],
            confidence_in_analysis=min(0.95, confidence),
            confidence_factors=confidence_factors
        )
        
        return summary
    
    def _risk_to_dict(self, risk: Risk) -> Dict:
        """Convert Risk object to dict for Phase 3."""
        return {
            "risk_id": risk.id,
            "title": risk.title,
            "description": risk.description,
            "category": risk.category.value,
            "severity": risk.severity.value,
            "risk_score": risk.risk_score
        }


# ============================================================================
# STANDALONE FUNCTION FOR ORCHESTRATOR
# ============================================================================

def run_risk_agent(context: str) -> Dict[str, Any]:
    """
    Standalone function to run risk detection agent.
    Compatible with existing orchestrator interface.
    
    Args:
        context: Retrieved document context
        
    Returns:
        Risk analysis results as dict
    """
    from .schemas import DocumentInput, RiskDetectorInput
    
    agent = RiskDetectorAgent()
    
    # Create input
    input_data = RiskDetectorInput(
        document=DocumentInput(content=context)
    )
    
    # Run analysis
    output = agent.analyze(input_data)
    
    # Convert to dict for orchestrator
    return output.dict()
