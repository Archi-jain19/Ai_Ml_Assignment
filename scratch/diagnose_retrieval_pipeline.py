import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_relevant_facets, load_enriched_facets
from src.scoring import score_facets_batch

def diagnose():
    enriched_df = load_enriched_facets()
    conv = "I enjoy learning new things. Whenever I don't understand a topic, I spend extra time studying it until I understand it."
    
    retrieved = retrieve_relevant_facets(conv, enriched_df, top_k=20)
    scored = score_facets_batch(conv, retrieved, client=None)
    
    print(f"Conversation: '{conv}'\n")
    print(f"{'Rank':<5} {'Facet':<45} {'Sim':<8} {'Status':<22} {'Score':<8} {'Reason'}")
    print("-" * 120)
    for rank, (r, s) in enumerate(zip(retrieved, scored), 1):
        sim = r.get("similarity", 0.0)
        score_str = str(s.get("score"))
        status_str = s.get("status")
        print(f"{rank:<5} {s['facet']:<45} {sim:<8.4f} {status_str:<22} {score_str:<8} {s['reason']}")

if __name__ == "__main__":
    diagnose()
