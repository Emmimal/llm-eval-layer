"""
Regression Test System
======================
Stores evaluation baselines and detects quality regressions
when prompts, models, or retrieval logic changes.

This is CI/CD for LLMs.

We don't just track scores — we fail builds when quality drops
beyond a threshold. A score drop of > 0.05 (configurable) triggers
a regression failure, exactly like a failing unit test in a CI pipeline.

Usage:
    from eval_engine.regression import RegressionSuite

    suite = RegressionSuite("data/baselines.json")

    # Record a baseline after validating your system
    suite.record_baseline("q_001", query, context, response, result)

    # After changing your prompt, model, or retrieval logic:
    report = suite.run_regression(pipeline, test_cases)

    # Treat failures like CI failures — don't ship if report.failed > 0
    if report.failed > 0:
        raise SystemExit("Quality regression detected. Deployment blocked.")
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from eval_engine.pipeline import EvalPipeline, EvalResult


@dataclass
class RegressionCase:
    id: str
    query: str
    context: str
    response: str
    baseline_score: float
    baseline_passed: bool
    recorded_at: str


@dataclass
class RegressionReport:
    total: int
    passed: int
    failed: int
    regressions: List[dict]
    improvements: List[dict]
    mean_score_delta: float
    run_at: str
    threshold: float = 0.05   # the regression_threshold used in this run

    def __str__(self) -> str:
        ci_status = "✅ ALL CHECKS PASSED" if self.failed == 0 else f"🚫 {self.failed} REGRESSION(S) DETECTED — DEPLOYMENT BLOCKED"
        lines = [
            f"\n{'═'*52}",
            f"  Regression Report  —  CI/CD Quality Gate",
            f"{'═'*52}",
            f"  {ci_status}",
            f"{'─'*52}",
            f"  Total cases   : {self.total}",
            f"  Passed        : {self.passed}",
            f"  Failed        : {self.failed}",
            f"  Mean Δ score  : {self.mean_score_delta:+.4f}",
            f"  Threshold     : ±{self.threshold:.2f} (regression_threshold)",
        ]
        if self.regressions:
            lines.append(f"\n  ⚠️  Regressions — score dropped beyond threshold:")
            for r in self.regressions:
                lines.append(
                    f"    [{r['id']}] {r['baseline_score']:.3f} → {r['current_score']:.3f} "
                    f"(Δ {r['delta']:+.3f})"
                )
        if self.improvements:
            lines.append(f"\n  ✅ Improvements ({len(self.improvements)}):")
            for r in self.improvements:
                lines.append(
                    f"    [{r['id']}] {r['baseline_score']:.3f} → {r['current_score']:.3f} "
                    f"(Δ {r['delta']:+.3f})"
                )
        lines.append(f"{'═'*52}\n")
        return "\n".join(lines)


class RegressionSuite:
    """
    Persistent regression test suite backed by a JSON file.

    Parameters
    ----------
    store_path : str
        Path to JSON file where baselines are persisted.
        File is created if it doesn't exist.
    regression_threshold : float
        Score drop greater than this triggers a regression flag.
        Default: 0.05 (5-point drop)
    """

    def __init__(
        self,
        store_path: str = "data/baselines.json",
        regression_threshold: float = 0.05,
    ):
        self.store_path = store_path
        self.threshold = regression_threshold
        self._baselines: Dict[str, RegressionCase] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                raw = json.load(f)
            self._baselines = {
                k: RegressionCase(**v) for k, v in raw.items()
            }

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(
                {k: asdict(v) for k, v in self._baselines.items()},
                f, indent=2
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def record_baseline(
        self,
        case_id: str,
        query: str,
        context: str,
        response: str,
        result: EvalResult,
    ) -> None:
        """Store an evaluation result as the baseline for *case_id*."""
        self._baselines[case_id] = RegressionCase(
            id=case_id,
            query=query,
            context=context,
            response=response,
            baseline_score=result.final_score,
            baseline_passed=result.passed,
            recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._save()
        print(f"  ✓ Baseline recorded [{case_id}] score={result.final_score:.4f}")

    def run_regression(
        self,
        pipeline: EvalPipeline,
        test_cases: Optional[List[Tuple[str, str, str, str]]] = None,
    ) -> RegressionReport:
        """
        Run regression tests against all stored baselines.

        Parameters
        ----------
        pipeline   : EvalPipeline instance (with your updated prompt/model)
        test_cases : optional list of (id, query, context, response) tuples.
                     If None, uses stored query/context/response from baselines.
        """
        if not self._baselines:
            print("No baselines recorded. Run record_baseline() first.")
            return RegressionReport(0, 0, 0, [], [], 0.0, time.strftime("%Y-%m-%dT%H:%M:%S"))

        cases = {}
        if test_cases:
            for cid, q, ctx, resp in test_cases:
                cases[cid] = (q, ctx, resp)
        else:
            cases = {
                cid: (b.query, b.context, b.response)
                for cid, b in self._baselines.items()
            }

        regressions, improvements = [], []
        deltas = []

        for cid, (q, ctx, resp) in cases.items():
            if cid not in self._baselines:
                continue
            baseline = self._baselines[cid]
            current = pipeline.evaluate(q, ctx, resp)
            delta = current.final_score - baseline.baseline_score
            deltas.append(delta)

            entry = {
                "id": cid,
                "baseline_score": baseline.baseline_score,
                "current_score": current.final_score,
                "delta": round(delta, 4),
            }

            if delta < -self.threshold:
                regressions.append(entry)
            elif delta > self.threshold:
                improvements.append(entry)

        passed = len(cases) - len(regressions)
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

        report = RegressionReport(
            total=len(cases),
            passed=passed,
            failed=len(regressions),
            regressions=regressions,
            improvements=improvements,
            mean_score_delta=round(mean_delta, 4),
            run_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            threshold=self.threshold,
        )
        print(report)
        return report

    @property
    def baseline_count(self) -> int:
        return len(self._baselines)
