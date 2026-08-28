import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_relevant_facets, load_enriched_facets
from src.scoring import score_facets_batch
import pandas as pd

def inspect_submission():
    enriched_df = load_enriched_facets()
    
    # Check any facets with words like submit, assign, task, deadline, time, work, duty, etc.
    print("Searching facet catalogue for relevant keywords:")
    kws = ["deadline", "assign", "submit", "complet", "punctual", "time", "conscientious", "procrastin", "duty", "reliab", "work"]
    for kw in kws:
        matches = enriched_df[enriched_df["normalized_facet"].str.lower().str.contains(kw) | enriched_df["raw_facet"].str.lower().str.contains(kw)]
        print(f"  Keyword '{kw}': {len(matches)} facets -> {list(matches['normalized_facet'].values[:5])}")

    conv1 = "I submitted my assignment yesterday."
    conv2 = "I submitted my assignment two days before the deadline."

    print("\n" + "=" * 80)
    print(f"CONVERSATION 1: '{conv1}'")
    print("=" * 80)
    ret1 = retrieve_relevant_facets(conv1, enriched_df, top_k=20)
    scored1 = score_facets_batch(conv1, ret1, client=None)
    for rank, (r, s) in enumerate(zip(ret1, scored1), 1):
        print(f"{rank:<3} {s['facet']:<45} Sim: {r.get('similarity', 0.0):.4f} | Status: {s['status']:<22} | Score: {str(s.get('score')):<5} | Reason: {s['reason']}")

    print("\n" + "=" * 80)
    print(f"CONVERSATION 2: '{conv2}'")
    print("=" * 80)
    ret2 = retrieve_relevant_facets(conv2, enriched_df, top_k=20)
    scored2 = score_facets_batch(conv2, ret2, client=None)
    for rank, (r, s) in enumerate(zip(ret2, scored2), 1):
        print(f"{rank:<3} {s['facet']:<45} Sim: {r.get('similarity', 0.0):.4f} | Status: {s['status']:<22} | Score: {str(s.get('score')):<5} | Reason: {s['reason']}")

if __name__ == "__main__":
    inspect_submission()
