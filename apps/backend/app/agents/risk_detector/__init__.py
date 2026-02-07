"""
Risk Detection Agent - Agent 3
Identifies risks, uncertainties, assumptions, and gaps in documents.
"""

from .agent import RiskDetectorAgent, run_risk_agent
from .schemas import (
    RiskDetectorInput,
    RiskDetectorOutput,
    Risk,
    Assumption,
    OpenQuestion,
    MissingInformation,
    Conflict,
    RiskSummary
)
from .scoring import RiskScorer

__all__ = [
    "RiskDetectorAgent",
    "run_risk_agent",
    "RiskDetectorInput",
    "RiskDetectorOutput",
    "Risk",
    "Assumption",
    "OpenQuestion",
    "MissingInformation",
    "Conflict",
    "RiskSummary",
    "RiskScorer"
]
