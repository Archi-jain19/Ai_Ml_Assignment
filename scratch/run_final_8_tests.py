import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_relevant_facets, load_enriched_facets
from src.scoring import score_facets_batch

enriched_df = load_enriched_facets()

test_cases = [
    ("TEST 1", "I work eight hours every day."),
    ("TEST 2", "I never give up when things get difficult."),
    ("TEST 3", "I always submit my assignments before the deadline."),
    ("TEST 4", "I practice yoga for five hours every week."),
    ("TEST 5", "I drink three cups of coffee every morning."),
    ("TEST 6", "I commute for two hours every day."),
    ("TEST 7", "I failed the test twice but kept studying until I passed."),
    ("TEST 8", "My brother works twelve hours every day."),
    ("TEST 9", "I consume 300 mg of caffeine every day."),
    ("TEST 10", "I spend a lot of time outdoors."),
]

print("=" * 90)
print("COMPREHENSIVE 8-TEST SUITE: RETRIEVAL & QUANTITATIVE/BEHAVIORAL SCORING")
print("=" * 90)

for label, conv in test_cases:
    retrieved = retrieve_relevant_facets(conv, enriched_df, top_k=5)
    scored = score_facets_batch(conv, retrieved, client=None)
    
    print(f"\n[{label}] \"{conv}\"")
    print("Retrieved & Evaluated Top Facets:")
    for rank, (r, s) in enumerate(zip(retrieved, scored), 1):
        status_str = f"[{s['status'].upper()}]"
        score_str = f"Score: {s['score']}/5" if s['score'] is not None else "Score: null"
        print(f"  {rank}. {s['facet']:<45} {status_str:<24} {score_str:<12} (Conf: {s['confidence']:.2f})")
        print(f"     Reason: {s['reason']}")
