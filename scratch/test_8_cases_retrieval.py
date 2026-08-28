import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_relevant_facets, load_enriched_facets
from src.scoring import score_facets_batch

enriched_df = load_enriched_facets()

test_conversations = [
    ("TEST 1", "I work eight hours every day."),
    ("TEST 2", "I never give up when things get difficult."),
    ("TEST 3", "I always submit my assignments before the deadline."),
    ("TEST 4", "I practice yoga for five hours every week."),
    ("TEST 5", "I drink three cups of coffee every morning."),
    ("TEST 6", "I commute for two hours every day."),
    ("TEST 7", "I failed the test twice but kept studying until I passed."),
    ("TEST 8", "My brother works twelve hours every day."),
    ("TEST 9 (Vague outdoor)", "I spend a lot of time outside."),
    ("TEST 10 (Adversarial Serotonin)", "I've been feeling pretty fatigued and low on energy lately, probably because of the gloomy winter weather and back-to-back work shifts."),
]

print("=" * 80)
print("TESTING RETRIEVAL FOR ALL TEST CASES")
print("=" * 80)

for label, conv in test_conversations:
    retrieved = retrieve_relevant_facets(conv, enriched_df, top_k=6)
    print(f"\n{label}: '{conv}'")
    for r in retrieved[:4]:
        print(f"  - [{r['similarity_score']:.4f}] {r['normalized_facet']} ({r['facet_type']})")
