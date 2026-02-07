"""
Pydantic schemas for Risk Detection Agent inputs and outputs.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class RiskCategory(str, Enum):
    TIMELINE = "timeline"
    RESOURCE = "resource"
    TECHNICAL = "technical"
    DEPENDENCY = "dependency"
    STAKEHOLDER = "stakeholder"
    EXTERNAL = "external"
    BUDGET = "budget"
    SCOPE = "scope"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AssumptionType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class ConflictType(str, Enum):
    TIMELINE_CONFLICT = "timeline_conflict"
    RESOURCE_CONFLICT = "resource_conflict"
    SCOPE_CONFLICT = "scope_conflict"
    INTERNAL_CONTRADICTION = "internal_contradiction"


# ============================================================================
# INPUT SCHEMAS (From Orchestrator)
# ============================================================================

class DocumentMetadata(BaseModel):
    """Metadata about the document being analyzed."""
    title: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    format: Optional[str] = None


class DocumentInput(BaseModel):
    """Document content and metadata."""
    content: str = Field(..., description="Raw text content of the document")
    metadata: Optional[DocumentMetadata] = None


class TaskFromTaskMaster(BaseModel):
    """Task structure from Task Master Agent."""
    id: str
    title: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


class TaskMasterOutput(BaseModel):
    """Output from Task Master Agent."""
    tasks: List[TaskFromTaskMaster] = Field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    critical_path: List[str] = Field(default_factory=list)
    owner_workload: Dict[str, List[str]] = Field(default_factory=dict)


class DecisionFromSummarizer(BaseModel):
    """Decision structure from Summarizer Agent."""
    id: str
    decision: str
    decision_maker: Optional[str] = None
    status: Optional[str] = None  # confirmed, tentative, pending
    date: Optional[str] = None


class ConstraintFromSummarizer(BaseModel):
    """Constraint structure from Summarizer Agent."""
    type: str
    description: str
    flexibility: Optional[str] = None  # low, medium, high


class SummarizerOutput(BaseModel):
    """Output from Summarizer Agent."""
    key_decisions: List[DecisionFromSummarizer] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    constraints: List[ConstraintFromSummarizer] = Field(default_factory=list)
    stakeholders: List[Dict[str, Any]] = Field(default_factory=list)
    important_dates: List[Dict[str, str]] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    """Optional analysis configuration."""
    focus_areas: List[str] = Field(default_factory=list)
    depth: Literal["comprehensive", "focused", "quick"] = "comprehensive"
    include_mitigations: bool = True
    severity_threshold: Literal["all", "medium_and_above", "high_and_above"] = "all"


class RiskDetectorInput(BaseModel):
    """Complete input to Risk Detection Agent."""
    document: DocumentInput
    summarizer_output: Optional[SummarizerOutput] = None
    taskmaster_output: Optional[TaskMasterOutput] = None
    request: Optional[AnalysisRequest] = None


# ============================================================================
# OUTPUT SCHEMAS (Risk Analysis Results)
# ============================================================================

class ScoreBreakdown(BaseModel):
    """Breakdown of risk score components."""
    impact: int = Field(..., ge=0, le=40)
    impact_reasoning: str
    probability: int = Field(..., ge=0, le=30)
    probability_reasoning: str
    dependencies: int = Field(..., ge=0, le=20)
    dependencies_reasoning: str
    timing: int = Field(..., ge=0, le=10)
    timing_reasoning: str


class Evidence(BaseModel):
    """Evidence supporting a risk identification."""
    direct_quotes: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    uncertainty_markers: List[str] = Field(default_factory=list)


class RelatedItems(BaseModel):
    """Items related to a risk."""
    tasks: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class CascadeEffect(BaseModel):
    """Cascade effect description."""
    sequence: List[str] = Field(default_factory=list)


class FinancialImpact(BaseModel):
    """Financial impact assessment."""
    estimated_cost_if_delayed: Optional[str] = None
    reasoning: Optional[str] = None


class ImpactAnalysis(BaseModel):
    """Analysis of risk impact if materialized."""
    if_materializes: str
    affected_deliverables: List[str] = Field(default_factory=list)
    affected_stakeholders: List[str] = Field(default_factory=list)
    cascade_effects: List[str] = Field(default_factory=list)
    financial_impact: Optional[FinancialImpact] = None


class ProbabilityAssessment(BaseModel):
    """Assessment of risk probability."""
    likelihood: str
    likelihood_percentage: int = Field(..., ge=0, le=100)
    reasoning: str
    historical_pattern: Optional[str] = None
    contributing_factors: List[str] = Field(default_factory=list)


class MitigationRecommendation(BaseModel):
    """Recommendation to mitigate a risk."""
    id: str
    action: str
    description: str
    timeline: str
    owner_suggestion: Optional[str] = None
    effort: Optional[str] = None
    cost: Optional[str] = None
    success_probability: float = Field(..., ge=0, le=1)
    risk_reduction: int = Field(..., ge=0, le=100)
    new_score_if_successful: Optional[int] = None


class ContingencyPlan(BaseModel):
    """Contingency plan if risk materializes."""
    trigger_condition: str
    response_plan: List[str] = Field(default_factory=list)
    estimated_delay: Optional[str] = None
    scope_impact: Optional[str] = None
    alternative_funding: Optional[str] = None


class RiskMonitoring(BaseModel):
    """Monitoring configuration for a risk."""
    status: str = "active"
    next_check_date: Optional[str] = None
    escalation_trigger: Optional[str] = None
    owner: Optional[str] = None


class Risk(BaseModel):
    """Complete risk object."""
    id: str
    title: str
    description: str
    category: RiskCategory
    subcategory: Optional[str] = None
    
    severity: Severity
    risk_score: int = Field(..., ge=0, le=100)
    score_breakdown: ScoreBreakdown
    
    evidence: Evidence
    related_items: RelatedItems
    
    impact_analysis: ImpactAnalysis
    probability_assessment: ProbabilityAssessment
    
    mitigation_recommendations: List[MitigationRecommendation] = Field(default_factory=list)
    contingency_plan: Optional[ContingencyPlan] = None
    monitoring: Optional[RiskMonitoring] = None


class AssumptionValidation(BaseModel):
    """Validation details for an assumption."""
    method: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    status: str = "not_started"


class AssumptionImpact(BaseModel):
    """Impact if assumption proves false."""
    severity: Severity
    description: str
    affected_risks: List[str] = Field(default_factory=list)
    affected_tasks: List[str] = Field(default_factory=list)


class Assumption(BaseModel):
    """Assumption identified in the document."""
    id: str
    assumption: str
    type: AssumptionType
    confidence: float = Field(..., ge=0, le=1)
    confidence_reasoning: str
    
    evidence: Evidence
    impact_if_false: AssumptionImpact
    validation: AssumptionValidation
    fallback_if_invalid: Optional[str] = None


class QuestionBlocks(BaseModel):
    """Items blocked by an open question."""
    tasks: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    """Open question requiring resolution."""
    id: str
    question: str
    context: str
    criticality: Severity
    criticality_reasoning: str
    
    blocks: QuestionBlocks
    needs_answer_by: Optional[str] = None
    consequence_if_unanswered: str
    
    who_can_answer: List[str] = Field(default_factory=list)
    suggested_resolution: Optional[str] = None


class HowToObtain(BaseModel):
    """How to obtain missing information."""
    source: str
    method: str
    effort: Optional[str] = None


class MissingInformation(BaseModel):
    """Missing information identified in document."""
    id: str
    item: str
    importance: Severity
    reasoning: str
    
    impacts: List[str] = Field(default_factory=list)
    current_gap: str
    how_to_obtain: HowToObtain
    default_assumption_if_unavailable: Optional[str] = None


class ConflictItem(BaseModel):
    """Item involved in a conflict."""
    id: Optional[str] = None
    description: str
    date: Optional[str] = None


class ConflictingItems(BaseModel):
    """Items that are in conflict."""
    item_a: Optional[ConflictItem] = None
    item_b: Optional[ConflictItem] = None
    resource: Optional[str] = None
    tasks: List[str] = Field(default_factory=list)
    period: Optional[str] = None
    total_estimated_effort: Optional[str] = None


class ConflictRecommendation(BaseModel):
    """Recommendations to resolve a conflict."""
    option_a: Optional[str] = None
    option_b: Optional[str] = None
    option_c: Optional[str] = None


class Conflict(BaseModel):
    """Conflict detected in document."""
    id: str
    type: ConflictType
    severity: Severity
    description: str
    
    conflicting_items: ConflictingItems
    impact: str
    recommendation: ConflictRecommendation


class RiskPattern(BaseModel):
    """Known risk pattern detected."""
    pattern_id: str
    pattern_name: str
    description: str
    matching_elements: List[str] = Field(default_factory=list)
    historical_outcome: str
    recommendation: str


class ImmediateAction(BaseModel):
    """Immediate action required."""
    action: str
    deadline: str
    owner: Optional[str] = None


class SeverityCount(BaseModel):
    """Count of risks by severity."""
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0


class CategoryCount(BaseModel):
    """Count of risks by category."""
    timeline: int = 0
    resource: int = 0
    technical: int = 0
    dependency: int = 0
    stakeholder: int = 0
    external: int = 0
    budget: int = 0
    scope: int = 0


class RiskSummary(BaseModel):
    """Summary of risk analysis."""
    total_risks: int = 0
    by_severity: SeverityCount = Field(default_factory=SeverityCount)
    by_category: CategoryCount = Field(default_factory=CategoryCount)
    
    total_assumptions: int = 0
    assumptions_needing_validation: int = 0
    
    open_questions: int = 0
    critical_questions: int = 0
    
    missing_information_items: int = 0
    critical_missing: int = 0
    
    conflicts_detected: int = 0
    
    overall_risk_level: Severity
    overall_risk_reasoning: str
    
    key_concerns: List[str] = Field(default_factory=list)
    immediate_actions_required: List[ImmediateAction] = Field(default_factory=list)
    
    confidence_in_analysis: float = Field(..., ge=0, le=1)
    confidence_factors: List[str] = Field(default_factory=list)
    
    recommended_review_date: Optional[str] = None
    next_analysis_trigger: Optional[str] = None


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis."""
    agent: str = "risk_detector"
    version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    document_analyzed: Optional[str] = None
    analysis_depth: str = "comprehensive"
    confidence_score: float = Field(default=0.85, ge=0, le=1)
    cross_agent_data_used: Dict[str, bool] = Field(default_factory=lambda: {
        "summarizer": False,
        "taskmaster": False
    })


class RiskDetectorOutput(BaseModel):
    """Complete output from Risk Detection Agent."""
    analysis_metadata: AnalysisMetadata
    
    risks: List[Risk] = Field(default_factory=list)
    assumptions: List[Assumption] = Field(default_factory=list)
    open_questions: List[OpenQuestion] = Field(default_factory=list)
    missing_information: List[MissingInformation] = Field(default_factory=list)
    conflict_detection: List[Conflict] = Field(default_factory=list)
    risk_patterns_detected: List[RiskPattern] = Field(default_factory=list)
    
    summary: RiskSummary
