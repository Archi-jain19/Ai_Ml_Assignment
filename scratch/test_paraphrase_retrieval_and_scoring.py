import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_relevant_facets, load_enriched_facets
from src.scoring import score_facets_batch

def run_paraphrase_audit():
    df = load_enriched_facets()
    tests = [
        ("TEST A", "I turned in my assignment two days before it was due."),
        ("TEST B", "I finished my coursework early and handed it in ahead of the due date."),
        ("TEST C", "I got my work in before the cutoff."),
        ("TEST D", "I handed my project in late."),
        ("TEST E", "I didn't turn in my assignment."),
    ]

    print("=" * 95)
    print("PARAPHRASE RETRIEVAL AND SCORING AUDIT (TESTS A - E)")
    print("=" * 95)

    for label, conv in tests:
        print("\n" + "=" * 95)
        print(f"[{label}] CONVERSATION: \"{conv}\"")
        print("=" * 95)

        # 1. Top Candidates Retrieved BEFORE Scoring
        candidates = retrieve_relevant_facets(conv, df, top_k=6)
        print("\n--- RETRIEVAL CANDIDATES (Top-6 BEFORE Scoring) ---")
        for rank, c in enumerate(candidates, 1):
            name = c["normalized_facet"]
            sim = c.get("similarity_score", 0.0)
            defn = c.get("scoring_definition", "")
            if len(defn) > 75:
                defn = defn[:75] + "..."
            print(f"  #{rank:<2} {name:<35} | Sim: {sim:.4f} | Def: {defn}")

        # 2. Final Scored Results
        scored = score_facets_batch(conv, candidates, client=None)
        print("\n--- FINAL EVALUATION RESULTS ---")
        for s in scored:
            score_str = f"{s['score']}/5" if s['score'] is not None else "null"
            print(f"  {s['facet']:<35} -> {s['status'].upper():<22} (Score: {score_str})")
            print(f"    Reason: {s['reason']}")

if __name__ == "__main__":
    run_paraphrase_audit()
