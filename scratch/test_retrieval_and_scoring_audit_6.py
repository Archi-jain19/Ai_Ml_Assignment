import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_relevant_facets, load_enriched_facets
from src.scoring import score_facets_batch

def run_audit():
    df = load_enriched_facets()
    test_cases = [
        ("TEST 1", "I submitted my assignment yesterday."),
        ("TEST 2", "The assignment was due Monday, but I submitted it on Sunday."),
        ("TEST 3", "I disagreed with my manager, but I still followed the instructions and completed the task exactly as requested."),
        ("TEST 4", "I failed my exam twice, but I kept studying every day and passed on my third attempt."),
        ("TEST 5", "I realized my presentations were weak, so I asked my professor for feedback, practiced every weekend, recorded myself speaking, and changed my approach based on the feedback."),
        ("TEST 6", "I consume 300 mg of caffeine every day."),
    ]

    for label, conv in test_cases:
        print("\n" + "=" * 90)
        print(f"[{label}] CONVERSATION: \"{conv}\"")
        print("=" * 90)
        
        # 1. Retrieval Candidates BEFORE scoring
        candidates = retrieve_relevant_facets(conv, df, top_k=8)
        print("\n--- RETRIEVAL CANDIDATES (Top-8 BEFORE Scoring) ---")
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
    run_audit()
