"""
Agents Package
Multi-agent document intelligence system.

Available Agents:
- Summary Agent: Generates document summaries with importance classification
- Action Agent (Task Master): Extracts actionable tasks
- Risk Agent: Detects risks, assumptions, and uncertainties
"""

from .summary_agent import run_summary_agent
from .action_agent import run_action_agent
from .risk_agent import run_risk_agent
from .risk_detector import RiskDetectorAgent, RiskDetectorInput, RiskDetectorOutput
from .summary_detector import (
    run_agent1,
    run_agent1_iterative,
    validate_agent1_output,
)

__all__ = [
    "run_summary_agent",
    "run_action_agent",
    "run_risk_agent",
    "RiskDetectorAgent",
    "RiskDetectorInput",
    "RiskDetectorOutput",
    "run_agent1",
    "run_agent1_iterative",
    "validate_agent1_output",
]
