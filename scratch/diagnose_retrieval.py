import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import load_enriched_facets, retrieve_relevant_facets

df = load_enriched_facets()

test_queries = [
    "I never give up when things get difficult.",
    "I work eight hours every day.",
    "I practice yoga for five hours every week.",
    "I volunteer at a local food bank every Saturday.",
    "I drink three cups of coffee every morning.",
    "I failed the test twice, but I changed how I studied and passed on my third attempt.",
]

for q in test_queries:
    print(f"\n==================================================")
    print(f"QUERY: '{q}'")
    print(f"==================================================")
    results = retrieve_relevant_facets(q, df, top_k=10)
    for i, r in enumerate(results, 1):
        print(f"  {i:2d}. [{r['similarity_score']:.4f}] {r['normalized_facet']} ({r['facet_type']})")
        print(f"      Defn: {r['scoring_definition'][:90]}...")
