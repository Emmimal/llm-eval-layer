"""
LLM-as-Judge
============
Fallback scorer that calls an LLM when heuristic confidence is too low.

Design principles:
  - Only called when heuristic scores fall below the confidence gate
  - Results are blended with heuristic scores, not replacing them
  - Prompts are versioned in prompts.py for auditability
  - Provider-agnostic: works with any OpenAI-compatible API

Typical cost: ~0.001 USD per judge call with gpt-4o-mini.
Typical latency: 300–800ms (network dependent).
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional

from llm_judge.prompts import (
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    HALLUCINATION_PROMPT,
)


@dataclass
class JudgeResult:
    score: float
    reason: str
    used_llm: bool = True


@dataclass
class HallucinationJudgeResult:
    hallucination_detected: bool
    unsupported_claims: list
    confidence: float
    used_llm: bool = True


def _call_llm(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Call an OpenAI-compatible chat completion endpoint.
    Set OPENAI_API_KEY in your environment or pass api_key directly.
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package required for LLM judge. "
            "Install with: pip install openai"
        )

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY environment variable."
        )

    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",          # cheap + fast; swap for gpt-4o if needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                # deterministic scoring
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def judge_faithfulness(
    context: str, response: str, api_key: Optional[str] = None
) -> JudgeResult:
    prompt = FAITHFULNESS_PROMPT.format(context=context, response=response)
    raw = _call_llm(prompt, api_key)
    data = json.loads(raw)
    return JudgeResult(score=float(data["score"]), reason=data["reason"])


def judge_relevance(
    query: str, response: str, api_key: Optional[str] = None
) -> JudgeResult:
    prompt = RELEVANCE_PROMPT.format(query=query, response=response)
    raw = _call_llm(prompt, api_key)
    data = json.loads(raw)
    return JudgeResult(score=float(data["score"]), reason=data["reason"])


def judge_hallucination(
    context: str, response: str, api_key: Optional[str] = None
) -> HallucinationJudgeResult:
    prompt = HALLUCINATION_PROMPT.format(context=context, response=response)
    raw = _call_llm(prompt, api_key)
    data = json.loads(raw)
    return HallucinationJudgeResult(
        hallucination_detected=data["hallucination_detected"],
        unsupported_claims=data.get("unsupported_claims", []),
        confidence=float(data["confidence"]),
    )
