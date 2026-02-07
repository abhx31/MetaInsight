"""
Summary Agent (Agent 1)
Context-Aware Summary Agent that classifies chunks by importance,
extracts insights, resolves conflicts, and produces validated summaries.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv


Importance = Literal["HIGH", "MEDIUM", "LOW"]
SummaryLength = Literal["short", "long", "both"]


@dataclass
class ChunkClassification:
    text: str
    importance: Importance
    reason: str


@dataclass
class ChunkInsightHigh:
    importance: Literal["HIGH"]
    intent: str
    decisions: List[str]
    constraints: List[str]


@dataclass
class ChunkInsightMedium:
    importance: Literal["MEDIUM"]
    intent: str


@dataclass
class ChunkInsightLow:
    importance: Literal["LOW"]
    intent: str


def _get_client() -> OpenAI:
    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))


def _chat_json(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    return _extract_json_object(content)


def _classify_chunk(
    client: OpenAI,
    *,
    chunk: str,
    model: str,
    temperature: float,
) -> ChunkClassification:
    system = (
        "You are a document analysis agent. "
        "Classify chunk importance as HIGH, MEDIUM, or LOW based on the rules. "
        "Return JSON only."
    )
    user = (
        "Classify the importance of the following text.\n\n"
        "Rules:\n"
        "- HIGH: contains decisions, approvals, deadlines, constraints\n"
        "- MEDIUM: supports understanding but no decisions\n"
        "- LOW: background or discussion only\n\n"
        "Respond strictly in JSON:\n"
        "{\n"
        '  "importance": "HIGH | MEDIUM | LOW",\n'
        '  "reason": "one sentence"\n'
        "}\n\n"
        f"Text:\n{chunk}"
    )

    obj = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
    )

    importance = str(obj.get("importance", "")).strip().upper()
    reason = str(obj.get("reason", "")).strip()

    if importance not in {"HIGH", "MEDIUM", "LOW"}:
        importance = "LOW"
        reason = reason or "Invalid importance label; defaulted to LOW."

    if not reason:
        reason = "No reason provided."

    return ChunkClassification(text=chunk, importance=importance, reason=reason)  # type: ignore[arg-type]


def _extract_insight(
    client: OpenAI,
    *,
    classified: ChunkClassification,
    model: str,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    if classified.importance == "HIGH":
        system = "You extract structured meaning from text. Return JSON only."
        user = (
            "Extract from the text:\n"
            "1. Intent\n"
            "2. Decisions\n"
            "3. Constraints or deadlines\n\n"
            "Return JSON only with keys: intent (string), decisions (array of strings), constraints (array of strings).\n\n"
            f"Text:\n{classified.text}"
        )
        obj = _chat_json(
            client,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
        )
        return {
            "importance": "HIGH",
            "intent": str(obj.get("intent", "")).strip(),
            "decisions": [str(x).strip() for x in (obj.get("decisions") or []) if str(x).strip()],
            "constraints": [str(x).strip() for x in (obj.get("constraints") or []) if str(x).strip()],
        }

    if classified.importance == "MEDIUM":
        system = "You summarize intent. Return JSON only."
        user = (
            "Extract the main intent in one sentence.\n\n"
            "Return JSON only with keys: intent (string).\n\n"
            f"Text:\n{classified.text}"
        )
        obj = _chat_json(
            client,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
        )
        return {
            "importance": "MEDIUM",
            "intent": str(obj.get("intent", "")).strip(),
        }

    # LOW: compress to 1 line
    system = "You compress background text. Return JSON only."
    user = (
        "Compress this to one short line capturing only the gist.\n\n"
        "Return JSON only with keys: intent (string).\n\n"
        f"Text:\n{classified.text}"
    )
    obj = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
    )
    return {
        "importance": "LOW",
        "intent": str(obj.get("intent", "")).strip(),
    }


def _resolve_conflicts(
    client: OpenAI,
    *,
    insights: List[Dict[str, Any]],
    model: str,
    temperature: float,
) -> Tuple[List[str], List[str]]:
    """Returns (resolved_decisions, timeline)."""
    high = [i for i in insights if i.get("importance") == "HIGH"]
    decisions: List[str] = []
    for item in high:
        for d in item.get("decisions") or []:
            ds = str(d).strip()
            if ds:
                decisions.append(ds)

    # quick dedupe while preserving order
    seen = set()
    decisions = [d for d in decisions if not (d in seen or seen.add(d))]

    # timeline: use intents from all insights, prefer earlier order as given
    timeline: List[str] = []
    for item in insights:
        intent = str(item.get("intent", "")).strip()
        if intent:
            timeline.append(intent)

    # If there are many decisions, ask the LLM once to resolve conflicts/overrides.
    if len(decisions) <= 1:
        return decisions, timeline

    system = "You detect conflicts and overrides in decisions. Return JSON only."
    user = (
        "You are given a list of decisions extracted from different chunks. "
        "If any decisions conflict, resolve by prioritizing the most recent or most explicit one.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "resolved_decisions": ["..."],\n'
        '  "override_notes": ["..."]\n'
        "}\n\n"
        f"Decisions:\n{json.dumps(decisions, ensure_ascii=False)}"
    )

    obj = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
    )

    resolved = [str(x).strip() for x in (obj.get("resolved_decisions") or []) if str(x).strip()]
    override_notes = [str(x).strip() for x in (obj.get("override_notes") or []) if str(x).strip()]

    if not resolved:
        resolved = decisions

    # Append override notes into timeline so orchestrator can surface them if desired.
    for note in override_notes:
        timeline.append(note)

    return resolved, timeline


def _final_synthesis(
    client: OpenAI,
    *,
    insights: List[Dict[str, Any]],
    resolved_decisions: List[str],
    model: str,
    temperature: float,
    summary_length: SummaryLength,
) -> Dict[str, Any]:
    system = "You are a Context-Aware Summary Agent. Return JSON only."

    if summary_length == "short":
        summary_instruction = "executive_summary must be 2-4 sentences."
    elif summary_length == "long":
        summary_instruction = "executive_summary must be detailed (8-15 sentences) and cover context + decisions + constraints."
    else:
        summary_instruction = (
            "Return both a short and long executive summary. "
            "executive_summary must be 2-4 sentences. "
            "executive_summary_long must be detailed (8-15 sentences)."
        )

    extra_keys = "- executive_summary_long (string)\n" if summary_length == "both" else ""

    user = (
        "Using the extracted intents, decisions, and constraints below:\n"
        "- Prioritize HIGH importance items\n"
        "- Include MEDIUM only if they add clarity\n"
        "- Ignore LOW background\n\n"
        "Produce JSON with keys:\n"
        f"- executive_summary (string) [{summary_instruction}]\n"
        f"{extra_keys}"
        "- key_decisions (array of strings)\n"
        "- constraints (array of strings)\n"
        "- assumptions (array of strings)\n\n"
        f"Resolved decisions (already conflict-handled):\n{json.dumps(resolved_decisions, ensure_ascii=False)}\n\n"
        f"Structured insights:\n{json.dumps(insights, ensure_ascii=False)}"
    )

    obj = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
    )

    executive_summary = str(obj.get("executive_summary", "")).strip()
    executive_summary_long = str(obj.get("executive_summary_long", "")).strip()
    key_decisions = [str(x).strip() for x in (obj.get("key_decisions") or []) if str(x).strip()]
    constraints = [str(x).strip() for x in (obj.get("constraints") or []) if str(x).strip()]
    assumptions = [str(x).strip() for x in (obj.get("assumptions") or []) if str(x).strip()]

    # Ensure resolved decisions are reflected.
    if resolved_decisions:
        merged = []
        seen = set()
        for d in resolved_decisions + key_decisions:
            if d not in seen:
                merged.append(d)
                seen.add(d)
        key_decisions = merged

    out: Dict[str, Any] = {
        "executive_summary": executive_summary,
        "key_decisions": key_decisions,
        "constraints": constraints,
        "assumptions": assumptions,
    }

    if summary_length == "both":
        out["executive_summary_long"] = executive_summary_long

    return out


def validate_agent1_output(
    chunks: List[str],
    output: Dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Validate Agent-1 output against the provided chunks.

    Returns JSON:
    {
      "is_valid": bool,
      "issues": ["..."],
      "missing_critical_points": ["..."],
      "unsupported_claims": ["..."],
      "suggested_fixes": ["..."],
      "corrected_output": { ... } | null
    }
    """
    client = _get_client()
    cleaned = [c.strip() for c in chunks if isinstance(c, str) and c.strip()]

    system = "You are a strict QA validator for summaries. Return JSON only."
    user = (
        "Validate the summary output against the source chunks.\n\n"
        "Rules:\n"
        "- Flag any unsupported claims that are not clearly grounded in the chunks.\n"
        "- Flag missing critical decisions/constraints/deadlines if present in chunks.\n"
        "- If issues exist, suggest fixes.\n"
        "- Optionally provide corrected_output that follows the same schema as the given output.\n\n"
        "Respond in JSON only:\n"
        "{\n"
        '  "is_valid": true|false,\n'
        '  "issues": ["..."],\n'
        '  "missing_critical_points": ["..."],\n'
        '  "unsupported_claims": ["..."],\n'
        '  "suggested_fixes": ["..."],\n'
        '  "corrected_output": { ... } | null\n'
        "}\n\n"
        f"CHUNKS:\n{json.dumps(cleaned, ensure_ascii=False)}\n\n"
        f"OUTPUT_TO_VALIDATE:\n{json.dumps(output, ensure_ascii=False)}"
    )

    obj = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
    )

    # Normalize
    is_valid = bool(obj.get("is_valid", False))
    issues = [str(x).strip() for x in (obj.get("issues") or []) if str(x).strip()]
    missing = [str(x).strip() for x in (obj.get("missing_critical_points") or []) if str(x).strip()]
    unsupported = [str(x).strip() for x in (obj.get("unsupported_claims") or []) if str(x).strip()]
    fixes = [str(x).strip() for x in (obj.get("suggested_fixes") or []) if str(x).strip()]
    corrected = obj.get("corrected_output")
    if corrected is not None and not isinstance(corrected, dict):
        corrected = None

    return {
        "is_valid": is_valid,
        "issues": issues,
        "missing_critical_points": missing,
        "unsupported_claims": unsupported,
        "suggested_fixes": fixes,
        "corrected_output": corrected,
    }


def run_agent1_iterative(
    chunks: List[str],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    summary_length: SummaryLength = "short",
    max_iters: int = 2,
) -> Dict[str, Any]:
    """Run Agent-1, validate, and (optionally) refine once or twice.

    Output includes the normal Agent-1 contract plus:
      - validation: { ... }  (latest validation report)
    """
    result = run_agent1(
        chunks,
        model=model,
        temperature=temperature,
        summary_length=summary_length,
    )

    validation: Dict[str, Any] = {}
    for _ in range(max(1, max_iters)):
        validation = validate_agent1_output(
            chunks,
            result,
            model=model,
            temperature=0.0,
        )

        if validation.get("is_valid") is True:
            break

        corrected = validation.get("corrected_output")
        if isinstance(corrected, dict) and corrected:
            # Keep contract keys from corrected when present.
            for k, v in corrected.items():
                result[k] = v
            break

        # If no corrected output was provided, stop after first validation report.
        break

    result["validation"] = validation
    return result


def run_agent1(
    chunks: List[str],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    summary_length: SummaryLength = "short",
) -> Dict[str, Any]:
    """Agent-1 contract.

    Input:
        chunks: List[str]  # filtered text chunks

    Output:
        {
          "executive_summary": str,
          "key_decisions": List[str],
          "constraints": List[str],
          "assumptions": List[str],
          "decision_timeline": List[str],
          "importance_distribution": {"high": int, "medium": int, "low": int}
        }
    """
    client = _get_client()

    cleaned = [c.strip() for c in chunks if isinstance(c, str) and c.strip()]
    if not cleaned:
        return {
            "executive_summary": "",
            "key_decisions": [],
            "constraints": [],
            "assumptions": [],
            "decision_timeline": [],
            "importance_distribution": {"high": 0, "medium": 0, "low": 0},
        }

    classifications: List[ChunkClassification] = []
    for chunk in cleaned:
        classifications.append(
            _classify_chunk(
                client,
                chunk=chunk,
                model=model,
                temperature=temperature,
            )
        )

    dist = {
        "high": sum(1 for c in classifications if c.importance == "HIGH"),
        "medium": sum(1 for c in classifications if c.importance == "MEDIUM"),
        "low": sum(1 for c in classifications if c.importance == "LOW"),
    }

    insights: List[Dict[str, Any]] = []
    for c in classifications:
        insight = _extract_insight(
            client,
            classified=c,
            model=model,
            temperature=temperature,
        )
        if insight is not None:
            insights.append(insight)

    resolved_decisions, timeline = _resolve_conflicts(
        client,
        insights=insights,
        model=model,
        temperature=temperature,
    )

    final = _final_synthesis(
        client,
        insights=insights,
        resolved_decisions=resolved_decisions,
        model=model,
        temperature=temperature,
        summary_length=summary_length,
    )

    out: Dict[str, Any] = {
        "executive_summary": final.get("executive_summary", ""),
        "key_decisions": final.get("key_decisions", []),
        "constraints": final.get("constraints", []),
        "assumptions": final.get("assumptions", []),
        "decision_timeline": timeline,
        "importance_distribution": dist,
    }

    if summary_length == "both" and "executive_summary_long" in final:
        out["executive_summary_long"] = final.get("executive_summary_long", "")

    return out


def run_summary_agent(context: str) -> Dict[str, Any]:
    """
    Orchestrator-compatible wrapper for the summary agent.
    
    Splits the combined context string into individual chunks using the 
    standard separator used by the orchestrator.
    
    Args:
        context: Retrieved document context (concatenated chunks)
        
    Returns:
        Summary output dict with executive_summary, key_decisions, etc.
    """
    if not context or not context.strip():
        return {
            "executive_summary": "No context provided for summary.",
            "key_decisions": [],
            "constraints": [],
            "assumptions": [],
            "decision_timeline": [],
            "importance_distribution": {"high": 0, "medium": 0, "low": 0},
        }
    
    # Split by the orchestrator's separator if present
    if "[Source Chunk" in context:
        # Regex to split on "[Source Chunk X]: " or similar patterns while keeping the text
        parts = re.split(r"\[Source Chunk \d+\]: ", context)
        chunks = [p.strip() for p in parts if p.strip()]
    else:
        # Fallback to splitting by newlines or treating as one chunk
        chunks = [context.strip()]
        
    return run_agent1_iterative(chunks)
