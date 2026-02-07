"""
Phase 2: Contextual Analysis Detector
Analyzes document context for implicit risks beyond keywords.
"""

import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class ContextualDetector:
    """
    Phase 2 detector that performs deep contextual analysis.
    """
    
    def __init__(self):
        """Initialize the contextual detector."""
        pass
    
    def detect_timeline_risks(
        self, 
        document_content: str,
        tasks: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Detect timeline-related risks.
        
        Returns:
            List of timeline risk dicts
        """
        risks = []
        
        # Pattern 1: Aggressive timeline language
        aggressive_patterns = [
            r"(?:asap|immediately|urgent|rush|fast.?track)",
            r"(?:tight|aggressive|ambitious) (?:timeline|deadline|schedule)",
            r"(?:no time|limited time|short timeframe)"
        ]
        
        for pattern in aggressive_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context (100 chars before and after)
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "aggressive_timeline",
                    "description": f"Aggressive timeline detected: '{match.group()}'",
                    "evidence": context,
                    "category": "timeline"
                })
        
        # Pattern 2: Missing timeline information
        missing_timeline_patterns = [
            r"no (?:timeline|deadline|schedule|timeframe)",
            r"timeline (?:unclear|undefined|tbd|pending)",
            r"deadline (?:not set|missing|unknown)"
        ]
        
        for pattern in missing_timeline_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "missing_timeline",
                    "description": f"Missing timeline information: '{match.group()}'",
                    "evidence": context,
                    "category": "timeline"
                })
        
        # Pattern 3: Sequential dependencies without buffer
        if tasks:
            dependency_risks = self._analyze_task_dependencies(tasks)
            risks.extend(dependency_risks)
        
        return risks
    
    def detect_resource_risks(self, document_content: str) -> List[Dict]:
        """
        Detect resource-related risks.
        
        Returns:
            List of resource risk dicts
        """
        risks = []
        
        # Pattern 1: Resource constraints
        constraint_patterns = [
            r"(?:limited|insufficient|lack of|not enough) (?:resources|people|staff|budget)",
            r"(?:under.?staffed|short.?staffed|resource.?constrained)",
            r"(?:budget|funding) (?:issue|problem|constraint|limitation)"
        ]
        
        for pattern in constraint_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "resource_constraint",
                    "description": f"Resource constraint identified: '{match.group()}'",
                    "evidence": context,
                    "category": "resource"
                })
        
        # Pattern 2: Key person dependency
        single_point_patterns = [
            r"only \w+ (?:can|will|is able to)",
            r"\w+ is the only (?:person|one|resource)",
            r"depends entirely on \w+"
        ]
        
        for pattern in single_point_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "single_point_of_failure",
                    "description": f"Key person dependency: '{match.group()}'",
                    "evidence": context,
                    "category": "resource"
                })
        
        # Pattern 3: Unassigned work
        unassigned_patterns = [
            r"(?:owner|assignee|responsible party) (?:tbd|unknown|not assigned|unclear)",
            r"no (?:owner|assignee) (?:identified|assigned)",
            r"(?:unassigned|needs owner)"
        ]
        
        for pattern in unassigned_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "unassigned_work",
                    "description": f"Unassigned work item: '{match.group()}'",
                    "evidence": context,
                    "category": "resource"
                })
        
        return risks
    
    def detect_dependency_risks(
        self, 
        document_content: str,
        dependency_graph: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict]:
        """
        Detect dependency-related risks.
        
        Returns:
            List of dependency risk dicts
        """
        risks = []
        
        # Pattern 1: Explicit dependencies mentioned
        dependency_patterns = [
            r"(?:depends on|dependent on|requires|needs|blocked by|waiting on) \w+",
            r"(?:cannot start until|must wait for|contingent on)",
            r"(?:prerequisite|precondition|dependency)"
        ]
        
        for pattern in dependency_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "dependency_identified",
                    "description": f"Dependency detected: '{match.group()}'",
                    "evidence": context,
                    "category": "dependency"
                })
        
        # Pattern 2: Analyze dependency graph for complex chains
        if dependency_graph:
            chain_risks = self._analyze_dependency_chains(dependency_graph)
            risks.extend(chain_risks)
        
        # Pattern 3: External dependencies
        external_patterns = [
            r"(?:third.?party|external|vendor|partner) (?:dependency|approval|integration)",
            r"(?:regulatory|legal|compliance) (?:approval|review|clearance)",
            r"(?:depends on|requires) (?:external|third.?party|vendor)"
        ]
        
        for pattern in external_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "external_dependency",
                    "description": f"External dependency: '{match.group()}'",
                    "evidence": context,
                    "category": "external"
                })
        
        return risks
    
    def detect_technical_risks(self, document_content: str) -> List[Dict]:
        """
        Detect technical risks.
        
        Returns:
            List of technical risk dicts
        """
        risks = []
        
        # Pattern 1: Unproven technology
        unproven_patterns = [
            r"(?:new|untested|unproven|experimental) (?:technology|approach|framework|platform)",
            r"(?:never used|first time|no experience with)",
            r"(?:prototype|proof of concept|poc|beta)"
        ]
        
        for pattern in unproven_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "unproven_technology",
                    "description": f"Unproven technology: '{match.group()}'",
                    "evidence": context,
                    "category": "technical"
                })
        
        # Pattern 2: Integration complexity
        integration_patterns = [
            r"(?:complex|complicated|challenging) (?:integration|migration)",
            r"(?:multiple|many) (?:integrations|systems|platforms)",
            r"(?:legacy system|technical debt|outdated)"
        ]
        
        for pattern in integration_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "integration_complexity",
                    "description": f"Integration complexity: '{match.group()}'",
                    "evidence": context,
                    "category": "technical"
                })
        
        # Pattern 3: Missing technical specs
        missing_spec_patterns = [
            r"(?:technical spec|architecture|design) (?:not defined|missing|unclear|tbd)",
            r"no (?:technical|architecture|design) (?:document|specification|plan)",
            r"(?:stack|technology|platform) (?:not decided|tbd|unclear)"
        ]
        
        for pattern in missing_spec_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "missing_technical_specs",
                    "description": f"Missing technical specification: '{match.group()}'",
                    "evidence": context,
                    "category": "technical"
                })
        
        return risks
    
    def detect_stakeholder_risks(self, document_content: str) -> List[Dict]:
        """
        Detect stakeholder-related risks.
        
        Returns:
            List of stakeholder risk dicts
        """
        risks = []
        
        # Pattern 1: Missing decision makers
        decision_patterns = [
            r"(?:decision maker|approver|authority) (?:not identified|unclear|tbd)",
            r"no (?:decision maker|approver|authority|owner)",
            r"(?:approval|decision) (?:process unclear|not defined)"
        ]
        
        for pattern in decision_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "missing_decision_maker",
                    "description": f"Decision maker unclear: '{match.group()}'",
                    "evidence": context,
                    "category": "stakeholder"
                })
        
        # Pattern 2: Multiple approvers (potential conflict)
        multiple_approver_patterns = [
            r"(?:multiple|several|many) (?:approvers|stakeholders|decision makers)",
            r"(?:requires approval from|needs sign.?off from) [\w\s,]+and[\w\s,]+"
        ]
        
        for pattern in multiple_approver_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(document_content), match.end() + 100)
                context = document_content[start:end].strip()
                
                risks.append({
                    "type": "multiple_approvers",
                    "description": f"Multiple approvers may create delays: '{match.group()}'",
                    "evidence": context,
                    "category": "stakeholder"
                })
        
        return risks
    
    def _analyze_task_dependencies(self, tasks: List[Dict]) -> List[Dict]:
        """Analyze task list for dependency risks."""
        risks = []
        
        for task in tasks:
            task_id = task.get("id", "unknown")
            dependencies = task.get("dependencies", [])
            
            # Check for tasks with many dependencies
            if len(dependencies) > 3:
                risks.append({
                    "type": "complex_dependencies",
                    "description": f"Task {task_id} has {len(dependencies)} dependencies - high coupling risk",
                    "evidence": f"Task: {task.get('title', task_id)}, Dependencies: {dependencies}",
                    "category": "dependency"
                })
        
        return risks
    
    def _analyze_dependency_chains(self, dependency_graph: Dict[str, List[str]]) -> List[Dict]:
        """Analyze dependency graph for long chains and cycles."""
        risks = []
        
        # Detect long chains (depth > 3)
        def get_chain_depth(task_id: str, visited: set = None) -> int:
            if visited is None:
                visited = set()
            if task_id in visited:
                return 0  # Cycle detected
            visited.add(task_id)
            
            deps = dependency_graph.get(task_id, [])
            if not deps:
                return 1
            
            return 1 + max(get_chain_depth(dep, visited.copy()) for dep in deps)
        
        for task_id in dependency_graph:
            depth = get_chain_depth(task_id)
            if depth > 3:
                risks.append({
                    "type": "long_dependency_chain",
                    "description": f"Task {task_id} has dependency chain depth of {depth} - high cascade risk",
                    "evidence": f"Dependency chain: {task_id} -> ... ({depth} levels deep)",
                    "category": "dependency"
                })
        
        # Detect cycles (circular dependencies)
        def has_cycle(task_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep in dependency_graph.get(task_id, []):
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        visited = set()
        for task_id in dependency_graph:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    risks.append({
                        "type": "circular_dependency",
                        "description": f"Circular dependency detected involving {task_id}",
                        "evidence": "Dependency cycle found in task graph",
                        "category": "dependency"
                    })
                    break  # Only report once
        
        return risks
    
    def analyze(
        self, 
        document_content: str,
        tasks: Optional[List[Dict]] = None,
        dependency_graph: Optional[Dict[str, List[str]]] = None
    ) -> Dict:
        """
        Perform full Phase 2 contextual analysis.
        
        Args:
            document_content: Raw document text
            tasks: Optional task list from Task Master
            dependency_graph: Optional dependency graph from Task Master
            
        Returns:
            Dict with all contextual risks
        """
        all_risks = []
        
        all_risks.extend(self.detect_timeline_risks(document_content, tasks))
        all_risks.extend(self.detect_resource_risks(document_content))
        all_risks.extend(self.detect_dependency_risks(document_content, dependency_graph))
        all_risks.extend(self.detect_technical_risks(document_content))
        all_risks.extend(self.detect_stakeholder_risks(document_content))
        
        # Group by category
        by_category = {}
        for risk in all_risks:
            category = risk.get("category", "other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(risk)
        
        return {
            "contextual_risks": all_risks,
            "risks_by_category": by_category,
            "total_contextual_risks": len(all_risks)
        }
