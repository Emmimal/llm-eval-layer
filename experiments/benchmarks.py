"""
Benchmark Runner
================
Measures scoring accuracy and latency across the full test set.
Produces the benchmark numbers for the article.

Run:
    python experiments/benchmarks.py
"""

import sys
import os
import json
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_engine.pipeline import EvalPipeline
from eval_engine.regression import RegressionSuite


def main():
    print("\n" + "="*60)
    print("  LLM Eval Layer — Benchmark Runner")
    print("="*60)

    with open(os.path.join(os.path.dirname(__file__), "../data/sample_queries.json")) as f:
        cases = json.load(f)

    pipeline = EvalPipeline(use_llm_judge=False)

    # ── 1. Accuracy benchmark ────────────────────────────────────────────────
    print("\n📊 Accuracy Benchmark\n")
    print("Classifying responses by label — does the scorer agree?\n")

    good_scores, bad_scores = [], []
    hallucination_hits = 0
    hallucination_total = 0

    for case in cases:
        result = pipeline.evaluate(
            query=case["query"],
            context_text=case["context"],
            response=case["response"],
        )

        if case["label"] == "good_response":
            good_scores.append(result.final_score)
        else:
            bad_scores.append(result.final_score)

        if case["label"] in ("hallucinated_response", "off_context_response"):
            hallucination_total += 1
            if result.hallucination_status in ("suspected", "confirmed"):
                hallucination_hits += 1

    print(f"  Good responses  → mean score: {statistics.mean(good_scores):.3f}")
    print(f"  Bad responses   → mean score: {statistics.mean(bad_scores):.3f}")
    print(f"  Score separation: {statistics.mean(good_scores) - statistics.mean(bad_scores):.3f}")
    print(f"\n  Hallucination detection rate: {hallucination_hits}/{hallucination_total} "
          f"({100*hallucination_hits/max(hallucination_total,1):.0f}%)")

    # ── 2. Latency benchmark ─────────────────────────────────────────────────
    print("\n\n⚡ Latency Benchmark (10 runs each)\n")

    case = cases[0]
    latencies = []

    for _ in range(10):
        result = pipeline.evaluate(
            query=case["query"],
            context_text=case["context"],
            response=case["response"],
        )
        latencies.append(result.latency_ms)

    print(f"  Mean latency  : {statistics.mean(latencies):.1f}ms")
    print(f"  Median        : {statistics.median(latencies):.1f}ms")
    print(f"  Min / Max     : {min(latencies):.1f}ms / {max(latencies):.1f}ms")
    print(f"  Stdev         : {statistics.stdev(latencies):.1f}ms")

    # ── 3. Regression suite demo ─────────────────────────────────────────────
    print("\n\n🔁 Regression Suite Demo\n")

    suite = RegressionSuite(store_path="/tmp/eval_baselines.json")

    # Record baselines
    print("Recording baselines...")
    for case in cases[:3]:
        result = pipeline.evaluate(
            query=case["query"],
            context_text=case["context"],
            response=case["response"],
        )
        suite.record_baseline(
            case_id=case["id"],
            query=case["query"],
            context=case["context"],
            response=case["response"],
            result=result,
        )

    # Simulate a regression — inject fabricated claims to degrade responses
    print("\nSimulating prompt change (degraded responses)...")
    degraded_cases = []
    for case in cases[:3]:
        degraded_response = (
            "This is unrelated to the question. "
            "The system was invented in 1850 and has nothing to do with AI or language models. "
            "It can be useful in various scenarios."
        )
        degraded_cases.append((case["id"], case["query"], case["context"], degraded_response))

    suite.run_regression(pipeline, degraded_cases)

    print("\n✅ Benchmarks complete.\n")


if __name__ == "__main__":
    main()
