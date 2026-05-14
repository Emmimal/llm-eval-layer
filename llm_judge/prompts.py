"""
LLM-as-Judge prompt templates.

These prompts are used only when heuristic confidence is low.
Keeping prompts in one place makes them easy to version and audit.
"""

FAITHFULNESS_PROMPT = """You are an evaluation assistant. Your job is to score
how faithfully an LLM response is grounded in the provided context.

Context:
{context}

Response:
{response}

Score the response on a scale from 0.0 to 1.0:
- 1.0 = every claim in the response is directly supported by the context
- 0.5 = some claims are supported, others are not
- 0.0 = the response contradicts or ignores the context entirely

Return ONLY a JSON object in this exact format:
{{"score": <float>, "reason": "<one sentence>"}}"""


RELEVANCE_PROMPT = """You are an evaluation assistant. Your job is to score
how directly an LLM response answers the original query.

Query:
{query}

Response:
{response}

Score the response on a scale from 0.0 to 1.0:
- 1.0 = directly and completely answers the query
- 0.5 = partially answers but misses key aspects
- 0.0 = does not answer the query at all

Return ONLY a JSON object in this exact format:
{{"score": <float>, "reason": "<one sentence>"}}"""


HALLUCINATION_PROMPT = """You are an evaluation assistant. Your job is to
detect hallucinations in an LLM response given the source context.

Context:
{context}

Response:
{response}

Identify any claims in the response that are NOT supported by the context.

Return ONLY a JSON object in this exact format:
{{"hallucination_detected": <true|false>,
  "unsupported_claims": ["<claim1>", "<claim2>"],
  "confidence": <float 0.0-1.0>}}"""
