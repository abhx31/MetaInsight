"""
Phase 1: Linguistic Analysis Detector
Scans document for uncertainty markers and explicit risk signals.
"""

import re
from typing import List, Dict, Tuple
from ..schemas import Evidence


class LinguisticDetector:
    """
    Phase 1 detector that performs keyword-based linguistic analysis.
    """
    
    # Uncertainty markers
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
        "depends on", "contingent upon", "subject to",
        "assuming approval", "assuming budget", "assuming resources",
        "once ", "when ", "as long as"
    ]
    
    QUESTION_MARKERS = [
        "need to confirm", "need to decide", "need to determine",
        "waiting for", "pending decision", "under review",
        "to be discussed", "to be finalized", "tba", "to be announced"
    ]
    
    MISSING_INFO_MARKERS = [
        "details missing", "not specified", "no timeline",
        "unassigned", "owner tbd", "timeline unclear",
        "budget not finalized", "scope undefined",
        "not available", "n/a"
    ]
    
    RISK_KEYWORDS = {
        "blocker": ["blocker", "blocked", "blocking", "showstopper"],
        "delay": ["delay", "delayed", "postpone", "postponed", "behind schedule"],
        "issue": ["issue", "problem", "concern", "challenge", "obstacle"],
        "dependency": ["depends", "dependent", "dependency", "relies on", "requires"],
        "approval": ["approval needed", "pending approval", "awaiting approval"],
        "resource": ["resource constraint", "insufficient resources", "lack of"],
    }
    
    def __init__(self):
        """Initialize the linguistic detector."""
        pass
    
    def detect_uncertainty_markers(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Detect all uncertainty markers in text.
        
        Returns:
            Dict with marker type -> list of (marker, context) tuples
        """
        text_lower = text.lower()
        results = {
            "high_uncertainty": [],
            "medium_uncertainty": [],
            "assumptions": [],
            "questions": [],
            "missing_info": []
        }
        
        # Split into sentences for context
        sentences = re.split(r'[.!?]+', text)
        
        # High uncertainty
        for marker in self.HIGH_UNCERTAINTY_MARKERS:
            for sentence in sentences:
                if marker in sentence.lower():
                    results["high_uncertainty"].append((marker, sentence.strip()))
        
        # Medium uncertainty
        for marker in self.MEDIUM_UNCERTAINTY_MARKERS:
            for sentence in sentences:
                if marker in sentence.lower():
                    results["medium_uncertainty"].append((marker, sentence.strip()))
        
        # Assumptions
        for marker in self.ASSUMPTION_MARKERS:
            for sentence in sentences:
                if marker in sentence.lower():
                    results["assumptions"].append((marker, sentence.strip()))
        
        # Questions
        for sentence in sentences:
            sentence_stripped = sentence.strip()
            if sentence_stripped.endswith("?"):
                results["questions"].append(("?", sentence_stripped))
            else:
                for marker in self.QUESTION_MARKERS:
                    if marker in sentence.lower():
                        results["questions"].append((marker, sentence_stripped))
                        break
        
        # Missing info
        for marker in self.MISSING_INFO_MARKERS:
            for sentence in sentences:
                if marker in sentence.lower():
                    results["missing_info"].append((marker, sentence.strip()))
        
        return results
    
    def detect_risk_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        Detect risk-related keywords.
        
        Returns:
            Dict with risk category -> list of found contexts
        """
        results = {}
        sentences = re.split(r'[.!?]+', text)
        
        for category, keywords in self.RISK_KEYWORDS.items():
            contexts = []
            for keyword in keywords:
                for sentence in sentences:
                    if keyword in sentence.lower():
                        contexts.append(sentence.strip())
            if contexts:
                results[category] = contexts
        
        return results
    
    def extract_evidence(
        self, 
        text: str, 
        markers: List[str], 
        max_quotes: int = 3
    ) -> Evidence:
        """
        Extract evidence from text for given markers.
        
        Args:
            text: Full document text
            markers: List of markers to find
            max_quotes: Maximum number of quotes to extract
            
        Returns:
            Evidence object with quotes and markers
        """
        sentences = re.split(r'[.!?]+', text)
        quotes = []
        found_markers = []
        
        for sentence in sentences:
            if len(quotes) >= max_quotes:
                break
            
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue
            
            for marker in markers:
                if marker.lower() in sentence_stripped.lower():
                    quotes.append(sentence_stripped)
                    found_markers.append(marker)
                    break
        
        return Evidence(
            direct_quotes=quotes[:max_quotes],
            uncertainty_markers=list(set(found_markers))
        )
    
    def calculate_uncertainty_level(self, markers: Dict[str, List[Tuple[str, str]]]) -> str:
        """
        Calculate overall uncertainty level from markers.
        
        Returns:
            "low", "medium", "high", or "critical"
        """
        high_count = len(markers.get("high_uncertainty", []))
        medium_count = len(markers.get("medium_uncertainty", []))
        assumption_count = len(markers.get("assumptions", []))
        question_count = len(markers.get("questions", []))
        missing_count = len(markers.get("missing_info", []))
        
        # Weight different types
        weighted_score = (
            high_count * 3 +
            medium_count * 2 +
            assumption_count * 2 +
            question_count * 2 +
            missing_count * 3
        )
        
        if weighted_score >= 30:
            return "critical"
        elif weighted_score >= 15:
            return "high"
        elif weighted_score >= 5:
            return "medium"
        else:
            return "low"
    
    def analyze(self, document_content: str) -> Dict:
        """
        Perform full Phase 1 linguistic analysis.
        
        Args:
            document_content: Raw document text
            
        Returns:
            Dict with uncertainty_markers, risk_keywords, uncertainty_level
        """
        uncertainty_markers = self.detect_uncertainty_markers(document_content)
        risk_keywords = self.detect_risk_keywords(document_content)
        uncertainty_level = self.calculate_uncertainty_level(uncertainty_markers)
        
        # Flatten markers for summary
        all_markers = []
        for marker_type, marker_list in uncertainty_markers.items():
            all_markers.extend([m[0] for m in marker_list])
        
        return {
            "uncertainty_markers": uncertainty_markers,
            "risk_keywords": risk_keywords,
            "uncertainty_level": uncertainty_level,
            "total_markers_found": len(all_markers),
            "unique_markers": list(set(all_markers))
        }
