"""
Unit tests for the LLM Eval Layer.

Run:
    python -m pytest tests/ -v
or:
    python tests/test_eval.py
"""

import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scorers import faithfulness, relevance, context
from eval_engine.aggregator import aggregate
from eval_engine.pipeline import EvalPipeline

GOOD_CONTEXT = (
    "Context engineering is the architectural layer between retrieval and "
    "generation. It controls what information flows into the LLM context window. "
    "It includes memory management, compression, re-ranking, and token budget enforcement."
)
GOOD_RESPONSE = (
    "Context engineering controls what enters the context window. "
    "It manages memory, compresses context, and enforces token budgets "
    "to keep the model grounded in relevant information."
)
HALLUCINATED_RESPONSE = (
    "Context engineering was invented at MIT in 1987. "
    "It is primarily a hardware technique used in CPU cache design."
)
OFF_TOPIC_RESPONSE = (
    "Photosynthesis is the process by which plants use sunlight, water and "
    "carbon dioxide to produce oxygen and energy in the form of sugar."
)
QUERY = "What is context engineering?"


class TestFaithfulness(unittest.TestCase):

    def test_good_response_scores_high(self):
        r = faithfulness.score(GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertGreater(r.score, 0.40)

    def test_attribution_and_specificity_present(self):
        r = faithfulness.score(GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertGreaterEqual(r.attribution_score, 0.0)
        self.assertLessEqual(r.attribution_score, 1.0)
        self.assertGreaterEqual(r.specificity_score, 0.0)
        self.assertLessEqual(r.specificity_score, 1.0)

    def test_hallucinated_response_detected(self):
        r = faithfulness.score(GOOD_CONTEXT, HALLUCINATED_RESPONSE)
        self.assertTrue(r.hallucination_detected)

    def test_good_response_not_flagged(self):
        r = faithfulness.score(GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertFalse(r.hallucination_detected,
            f"Good response should not be flagged, attribution={r.attribution_score}")

    def test_empty_context_returns_zero(self):
        r = faithfulness.score("", GOOD_RESPONSE)
        self.assertEqual(r.score, 0.0)

    def test_attribution_lower_than_good_for_hallucination(self):
        good = faithfulness.score(GOOD_CONTEXT, GOOD_RESPONSE)
        bad  = faithfulness.score(GOOD_CONTEXT, HALLUCINATED_RESPONSE)
        self.assertGreater(good.attribution_score, bad.attribution_score)


class TestRelevance(unittest.TestCase):

    def test_relevant_response_scores_high(self):
        r = relevance.score(QUERY, GOOD_RESPONSE)
        self.assertGreater(r.score, 0.30)

    def test_off_topic_scores_lower(self):
        on  = relevance.score(QUERY, GOOD_RESPONSE)
        off = relevance.score(QUERY, OFF_TOPIC_RESPONSE)
        self.assertLess(off.score, on.score)

    def test_empty_query_returns_zero(self):
        r = relevance.score("", GOOD_RESPONSE)
        self.assertEqual(r.score, 0.0)


class TestContextScorer(unittest.TestCase):

    def test_aligned_scores_higher(self):
        aligned    = context.score(GOOD_CONTEXT, GOOD_RESPONSE)
        misaligned = context.score(GOOD_CONTEXT, OFF_TOPIC_RESPONSE)
        self.assertGreater(aligned.score, misaligned.score)

    def test_f1_is_harmonic_mean(self):
        r = context.score(GOOD_CONTEXT, GOOD_RESPONSE)
        if r.precision_score + r.recall_score > 0:
            expected = 2 * r.precision_score * r.recall_score / (r.precision_score + r.recall_score)
            self.assertAlmostEqual(r.f1_score, expected, places=3)


class TestAggregator(unittest.TestCase):

    def test_produces_valid_score(self):
        f = faithfulness.score(GOOD_CONTEXT, GOOD_RESPONSE)
        r = relevance.score(QUERY, GOOD_RESPONSE)
        c = context.score(GOOD_CONTEXT, GOOD_RESPONSE)
        agg = aggregate(f, r, c)
        self.assertGreaterEqual(agg.final_score, 0.0)
        self.assertLessEqual(agg.final_score, 1.0)

    def test_bad_scores_lower_than_good(self):
        gf = faithfulness.score(GOOD_CONTEXT, GOOD_RESPONSE)
        gr = relevance.score(QUERY, GOOD_RESPONSE)
        gc = context.score(GOOD_CONTEXT, GOOD_RESPONSE)
        bf = faithfulness.score(GOOD_CONTEXT, HALLUCINATED_RESPONSE)
        br = relevance.score(QUERY, HALLUCINATED_RESPONSE)
        bc = context.score(GOOD_CONTEXT, HALLUCINATED_RESPONSE)
        self.assertGreater(aggregate(gf, gr, gc).final_score,
                           aggregate(bf, br, bc).final_score)


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = EvalPipeline(use_llm_judge=False)

    def test_returns_result_with_all_fields(self):
        r = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertIn(r.decision, ["ACCEPT", "REVIEW", "REJECT"])
        self.assertIn(r.failure_type, ["none", "weak_grounding", "hallucination",
                                       "poor_retrieval", "off_topic", "uncertain"])
        self.assertIn(r.hallucination_status, ["none", "suspected", "confirmed"])
        self.assertGreaterEqual(r.confidence_pct, 0)
        self.assertLessEqual(r.confidence_pct, 100)
        self.assertGreaterEqual(r.attribution_score, 0.0)
        self.assertGreaterEqual(r.specificity_score, 0.0)
        self.assertGreaterEqual(r.disagreement, 0.0)

    def test_good_response_not_rejected(self):
        r = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertNotEqual(r.decision, "REJECT")

    def test_hallucinated_response_flagged(self):
        r = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, HALLUCINATED_RESPONSE)
        self.assertIn(r.hallucination_status, ["suspected", "confirmed"])

    def test_hallucinated_scores_lower_than_good(self):
        good = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, GOOD_RESPONSE)
        bad  = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, HALLUCINATED_RESPONSE)
        self.assertGreater(good.final_score, bad.final_score)

    def test_hallucination_failure_type(self):
        r = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, HALLUCINATED_RESPONSE)
        # hallucinated response should have worse attribution than good response
        good = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertLess(r.attribution_score, good.attribution_score)

    def test_to_dict_serialisable(self):
        r = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, GOOD_RESPONSE)
        d = r.to_dict()
        self.assertIn("decision", d)
        self.assertIn("scores", d)
        self.assertIn("attribution", d["scores"])
        self.assertIn("specificity", d["scores"])
        self.assertIn("disagreement", d["scores"])

    def test_latency_reasonable(self):
        r = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertLess(r.latency_ms, 5000)

    def test_disagreement_is_float(self):
        r = self.pipeline.evaluate(QUERY, GOOD_CONTEXT, GOOD_RESPONSE)
        self.assertIsInstance(r.disagreement, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
