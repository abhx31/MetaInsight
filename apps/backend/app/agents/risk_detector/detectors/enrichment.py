"""
Phase 3: Cross-Agent Enrichment Detector
Enriches risk analysis using data from Summarizer and Task Master agents.
"""

from typing import List, Dict, Optional
from ..schemas import (
    TaskMasterOutput, 
    SummarizerOutput,
    RiskCategory,
    Severity
)


class EnrichmentDetector:
    """
    Phase 3 detector that enriches risks using cross-agent data.
    """
    
    def __init__(self):
        """Initialize the enrichment detector."""
        pass
    
    def enrich_with_taskmaster(
        self,
        identified_risks: List[Dict],
        taskmaster_output: TaskMasterOutput
    ) -> Dict:
        """
        Enrich risks with Task Master data.
        
        Args:
            identified_risks: Risks from Phase 1 & 2
            taskmaster_output: Output from Task Master agent
            
        Returns:
            Dict with enriched risks and new risks from task analysis
        """
        enriched_risks = []
        new_risks = []
        
        # Create quick lookup maps
        task_map = {task.id: task for task in taskmaster_output.tasks}
        critical_path_set = set(taskmaster_output.critical_path)
        
        # Enrich existing risks
        for risk in identified_risks:
            enriched_risk = risk.copy()
            
            # Try to link risk to tasks
            related_tasks = self._find_related_tasks(
                risk.get("description", ""), 
                task_map
            )
            enriched_risk["related_tasks"] = related_tasks
            
            # Check if affects critical path
            affects_critical_path = any(
                task_id in critical_path_set 
                for task_id in related_tasks
            )
            enriched_risk["on_critical_path"] = affects_critical_path
            
            # Count blocking impact
            enriched_risk["blocks_count"] = len(related_tasks)
            
            enriched_risks.append(enriched_risk)
        
        # Generate new risks from task analysis
        new_risks.extend(self._detect_unassigned_tasks(taskmaster_output))
        new_risks.extend(self._detect_missing_deadlines(taskmaster_output))
        new_risks.extend(self._detect_owner_overload(taskmaster_output))
        new_risks.extend(self._detect_critical_path_risks(taskmaster_output))
        
        return {
            "enriched_risks": enriched_risks,
            "new_risks_from_taskmaster": new_risks,
            "total_tasks_analyzed": len(taskmaster_output.tasks)
        }
    
    def enrich_with_summarizer(
        self,
        identified_risks: List[Dict],
        summarizer_output: SummarizerOutput
    ) -> Dict:
        """
        Enrich risks with Summarizer data.
        
        Args:
            identified_risks: Risks from Phase 1 & 2
            summarizer_output: Output from Summarizer agent
            
        Returns:
            Dict with enriched risks and new risks from summary analysis
        """
        enriched_risks = []
        new_risks = []
        
        # Create lookup maps
        decision_map = {d.id: d for d in summarizer_output.key_decisions}
        
        # Enrich existing risks
        for risk in identified_risks:
            enriched_risk = risk.copy()
            
            # Try to link risk to decisions
            related_decisions = self._find_related_decisions(
                risk.get("description", ""),
                decision_map
            )
            enriched_risk["related_decisions"] = related_decisions
            
            enriched_risks.append(enriched_risk)
        
        # Generate new risks from summary analysis
        new_risks.extend(self._detect_tentative_decisions(summarizer_output))
        new_risks.extend(self._detect_constraint_conflicts(summarizer_output))
        
        return {
            "enriched_risks": enriched_risks,
            "new_risks_from_summarizer": new_risks,
            "total_decisions_analyzed": len(summarizer_output.key_decisions)
        }
    
    def detect_conflicts(
        self,
        taskmaster_output: Optional[TaskMasterOutput] = None,
        summarizer_output: Optional[SummarizerOutput] = None
    ) -> List[Dict]:
        """
        Detect conflicts between tasks, timelines, resources.
        
        Returns:
            List of conflict dicts
        """
        conflicts = []
        
        if taskmaster_output:
            conflicts.extend(self._detect_timeline_conflicts(taskmaster_output))
            conflicts.extend(self._detect_resource_conflicts(taskmaster_output))
        
        return conflicts
    
    # ========================================================================
    # TASK MASTER ANALYSIS
    # ========================================================================
    
    def _find_related_tasks(
        self, 
        risk_description: str, 
        task_map: Dict
    ) -> List[str]:
        """Find tasks related to a risk by keyword matching."""
        related = []
        risk_lower = risk_description.lower()
        
        for task_id, task in task_map.items():
            task_title = task.title.lower()
            # Simple keyword matching - could be enhanced with embeddings
            if any(word in task_title for word in risk_lower.split() if len(word) > 3):
                related.append(task_id)
        
        return related
    
    def _detect_unassigned_tasks(self, taskmaster_output: TaskMasterOutput) -> List[Dict]:
        """Detect tasks without owners."""
        risks = []
        
        for task in taskmaster_output.tasks:
            if not task.owner or task.owner.lower() in ["tbd", "unassigned", "none", ""]:
                risks.append({
                    "risk_id": f"TM_UNASSIGNED_{task.id}",
                    "source": "taskmaster",
                    "title": f"Task {task.id} has no owner assigned",
                    "description": f"Task '{task.title}' (ID: {task.id}) has no assigned owner, creating accountability gap",
                    "evidence": f"Owner field: {task.owner or 'None'}",
                    "category": RiskCategory.RESOURCE,
                    "related_tasks": [task.id],
                    "severity_hint": Severity.HIGH if task.priority == "high" else Severity.MEDIUM
                })
        
        return risks
    
    def _detect_missing_deadlines(self, taskmaster_output: TaskMasterOutput) -> List[Dict]:
        """Detect tasks without deadlines."""
        risks = []
        
        for task in taskmaster_output.tasks:
            if not task.deadline or task.deadline.lower() in ["tbd", "none", ""]:
                risks.append({
                    "risk_id": f"TM_NO_DEADLINE_{task.id}",
                    "source": "taskmaster",
                    "title": f"Task {task.id} has no deadline",
                    "description": f"Task '{task.title}' (ID: {task.id}) has no defined deadline, risking timeline slippage",
                    "evidence": f"Deadline field: {task.deadline or 'None'}",
                    "category": RiskCategory.TIMELINE,
                    "related_tasks": [task.id],
                    "severity_hint": Severity.MEDIUM
                })
        
        return risks
    
    def _detect_owner_overload(self, taskmaster_output: TaskMasterOutput) -> List[Dict]:
        """Detect owners with too many tasks."""
        risks = []
        
        for owner, task_ids in taskmaster_output.owner_workload.items():
            if len(task_ids) >= 4:
                severity = Severity.CRITICAL if len(task_ids) >= 6 else Severity.HIGH
                risks.append({
                    "risk_id": f"TM_OVERLOAD_{owner.replace(' ', '_')}",
                    "source": "taskmaster",
                    "title": f"Owner '{owner}' is over-allocated",
                    "description": f"{owner} is assigned to {len(task_ids)} tasks, creating single point of failure and quality risk",
                    "evidence": f"Tasks assigned: {task_ids}",
                    "category": RiskCategory.RESOURCE,
                    "related_tasks": task_ids,
                    "severity_hint": severity
                })
        
        return risks
    
    def _detect_critical_path_risks(self, taskmaster_output: TaskMasterOutput) -> List[Dict]:
        """Detect risks on critical path."""
        risks = []
        
        if len(taskmaster_output.critical_path) > 5:
            risks.append({
                "risk_id": "TM_LONG_CRITICAL_PATH",
                "source": "taskmaster",
                "title": "Critical path is long and vulnerable",
                "description": f"Critical path contains {len(taskmaster_output.critical_path)} tasks. Any delay cascades through entire project.",
                "evidence": f"Critical path: {taskmaster_output.critical_path}",
                "category": RiskCategory.DEPENDENCY,
                "related_tasks": taskmaster_output.critical_path,
                "severity_hint": Severity.HIGH
            })
        
        return risks
    
    # ========================================================================
    # SUMMARIZER ANALYSIS
    # ========================================================================
    
    def _find_related_decisions(
        self, 
        risk_description: str, 
        decision_map: Dict
    ) -> List[str]:
        """Find decisions related to a risk by keyword matching."""
        related = []
        risk_lower = risk_description.lower()
        
        for decision_id, decision in decision_map.items():
            decision_text = decision.decision.lower()
            # Simple keyword matching
            if any(word in decision_text for word in risk_lower.split() if len(word) > 3):
                related.append(decision_id)
        
        return related
    
    def _detect_tentative_decisions(self, summarizer_output: SummarizerOutput) -> List[Dict]:
        """Detect decisions that are tentative or pending."""
        risks = []
        
        for decision in summarizer_output.key_decisions:
            if decision.status and decision.status.lower() in ["tentative", "pending", "draft"]:
                risks.append({
                    "risk_id": f"SUM_TENTATIVE_{decision.id}",
                    "source": "summarizer",
                    "title": f"Decision {decision.id} is not finalized",
                    "description": f"Decision '{decision.decision}' has status '{decision.status}', creating uncertainty",
                    "evidence": f"Decision status: {decision.status}",
                    "category": RiskCategory.STAKEHOLDER,
                    "related_decisions": [decision.id],
                    "severity_hint": Severity.HIGH
                })
        
        return risks
    
    def _detect_constraint_conflicts(self, summarizer_output: SummarizerOutput) -> List[Dict]:
        """Detect constraints with low flexibility."""
        risks = []
        
        for constraint in summarizer_output.constraints:
            if constraint.flexibility and constraint.flexibility.lower() == "low":
                risks.append({
                    "risk_id": f"SUM_CONSTRAINT_{constraint.type.upper()}",
                    "source": "summarizer",
                    "title": f"Inflexible {constraint.type} constraint",
                    "description": f"Constraint '{constraint.description}' has low flexibility - creates rigidity risk",
                    "evidence": f"Constraint: {constraint.description}, Flexibility: {constraint.flexibility}",
                    "category": RiskCategory.SCOPE,
                    "severity_hint": Severity.MEDIUM
                })
        
        return risks
    
    # ========================================================================
    # CONFLICT DETECTION
    # ========================================================================
    
    def _detect_timeline_conflicts(self, taskmaster_output: TaskMasterOutput) -> List[Dict]:
        """Detect timeline conflicts between tasks."""
        conflicts = []
        
        # Check for dependency inversions (task B depends on A, but A deadline after B)
        task_map = {task.id: task for task in taskmaster_output.tasks}
        
        for task_id, dependencies in taskmaster_output.dependency_graph.items():
            task = task_map.get(task_id)
            if not task or not task.deadline:
                continue
            
            for dep_id in dependencies:
                dep_task = task_map.get(dep_id)
                if not dep_task or not dep_task.deadline:
                    continue
                
                # Compare deadlines (simple string comparison - could parse dates)
                if task.deadline < dep_task.deadline:
                    conflicts.append({
                        "conflict_id": f"TIMELINE_{task_id}_{dep_id}",
                        "type": "timeline_conflict",
                        "severity": Severity.HIGH,
                        "description": f"Task {task_id} deadline ({task.deadline}) is before dependency {dep_id} ({dep_task.deadline})",
                        "item_a": {"id": task_id, "description": task.title, "date": task.deadline},
                        "item_b": {"id": dep_id, "description": dep_task.title, "date": dep_task.deadline}
                    })
        
        return conflicts
    
    def _detect_resource_conflicts(self, taskmaster_output: TaskMasterOutput) -> List[Dict]:
        """Detect resource conflicts (over-allocation)."""
        conflicts = []
        
        # Already handled in owner_overload, but could add more sophisticated analysis
        # e.g., check for overlapping task timelines for same owner
        
        return conflicts
    
    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================
    
    def analyze(
        self,
        identified_risks: List[Dict],
        taskmaster_output: Optional[TaskMasterOutput] = None,
        summarizer_output: Optional[SummarizerOutput] = None
    ) -> Dict:
        """
        Perform full Phase 3 enrichment analysis.
        
        Args:
            identified_risks: Risks from Phase 1 & 2
            taskmaster_output: Optional Task Master data
            summarizer_output: Optional Summarizer data
            
        Returns:
            Dict with enriched risks, new risks, and conflicts
        """
        result = {
            "enriched_risks": identified_risks.copy(),
            "new_risks": [],
            "conflicts": []
        }
        
        # Enrich with Task Master data
        if taskmaster_output:
            tm_result = self.enrich_with_taskmaster(identified_risks, taskmaster_output)
            result["enriched_risks"] = tm_result["enriched_risks"]
            result["new_risks"].extend(tm_result["new_risks_from_taskmaster"])
        
        # Enrich with Summarizer data
        if summarizer_output:
            sum_result = self.enrich_with_summarizer(
                result["enriched_risks"], 
                summarizer_output
            )
            result["enriched_risks"] = sum_result["enriched_risks"]
            result["new_risks"].extend(sum_result["new_risks_from_summarizer"])
        
        # Detect conflicts
        conflicts = self.detect_conflicts(taskmaster_output, summarizer_output)
        result["conflicts"] = conflicts
        
        # Add metadata
        result["cross_agent_data_used"] = {
            "taskmaster": taskmaster_output is not None,
            "summarizer": summarizer_output is not None
        }
        
        result["total_risks"] = len(result["enriched_risks"]) + len(result["new_risks"])
        
        return result
