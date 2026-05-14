"""
LLM Eval Layer — Main Entry Point
==================================
Responses are not just scored — they are decided:
  ✅ ACCEPT  → serve the response
  🔍 REVIEW  → retry or human review
  🚫 REJECT  → block the response

The biggest mistake in LLM systems is treating all failures the same.
A hallucination is not the same as a weak answer — and this system
knows the difference.

Run:
    python main.py
    python experiments/rag_eval_demo.py
    python experiments/benchmarks.py
"""

from collections import Counter
from eval_engine.pipeline import EvalPipeline

CONTEXT = (
    "Context engineering is the architectural layer between retrieval and "
    "generation. It controls what information flows into the LLM context window "
    "— managing memory, compression, re-ranking, and token budget enforcement."
)

EXAMPLES = [
    {
        "label": "Well-grounded response → expect ACCEPT",
        "query": "What is context engineering?",
        "response": (
            "Context engineering is the layer between retrieval and generation "
            "that controls what the model actually sees. It manages memory, "
            "compresses context when space is tight, and enforces token budgets "
            "to keep responses grounded in relevant information."
        ),
    },
    {
        "label": "Hallucinated response → expect REJECT",
        "query": "What is context engineering?",
        "response": (
            "Context engineering was invented at MIT in 1987 and is primarily "
            "used for hardware cache optimization in CPUs. It has nothing to do "
            "with language models."
        ),
    },
    {
        "label": "Vague response → expect REVIEW",
        "query": "What is context engineering?",
        "response": (
            "It is a technique used in AI systems to help manage information. "
            "It can be useful in various scenarios."
        ),
    },
    {
        "label": "Off-topic response → expect REJECT",
        "query": "How does token budget enforcement work?",
        "response": (
            "The French Revolution was a period of major political and societal "
            "change in France that began with the Estates General of 1789. "
            "Marie Antoinette was Queen of France at the time."
        ),
    },
]


def print_distribution(decisions: list[str]) -> None:
    total = len(decisions)
    counts = Counter(decisions)
    icons = {"ACCEPT": "✅", "REVIEW": "🔍", "REJECT": "🚫"}
    print("\n" + "═"*52)
    print("  Decision Distribution")
    print("═"*52)
    for d in ["ACCEPT", "REVIEW", "REJECT"]:
        n = counts.get(d, 0)
        pct = 100 * n / total
        bar = "█" * int(pct / 5)
        print(f"  {icons[d]} {d:<8} {bar:<20} {n}/{total}  ({pct:.0f}%)")
    print("═"*52 + "\n")


def main():
    print("\n" + "="*60)
    print("  LLM Eval Layer — Decision System Demo")
    print("="*60)
    print("\n  A hallucination is not the same as a weak answer.")
    print("  This system knows the difference.\n")

    pipeline = EvalPipeline(use_llm_judge=False)
    decisions = []

    for i, ex in enumerate(EXAMPLES, 1):
        print("─"*60)
        print(f"  [Example {i}] {ex['label']}")
        result = pipeline.evaluate(
            query=ex["query"],
            context_text=CONTEXT,
            response=ex["response"],
        )
        print(result)
        decisions.append(result.decision)

    print_distribution(decisions)
    print("  Run experiments/benchmarks.py for regression suite demo.\n")


if __name__ == "__main__":
    main()
