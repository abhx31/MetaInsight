"""
Risk Agent Wrapper
Provides the run_risk_agent function expected by orchestrator.py
"""

from app.agents.risk_detector import run_risk_agent

__all__ = ["run_risk_agent"]
