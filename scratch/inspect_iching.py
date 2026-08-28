import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import load_enriched_facets, get_scoreable_facets, build_facet_semantic_text, _get_bm25_index
from src.embeddings import load_faiss_index, embed_texts
import numpy as np

df = load_enriched_facets()
scoreable = get_scoreable_facets(df)
index, facet_ids = load_faiss_index()

conv = "I failed the test twice, but I changed how I studied and passed on my third attempt."
q_emb = embed_texts([conv])[0]

# Dense similarity
dists, indices = index.search(np.array([q_emb], dtype=np.float32), len(scoreable))
dense_scores = np.zeros(len(scoreable))
for d, i in zip(dists[0], indices[0]):
    dense_scores[i] = d

# BM25
bm25 = _get_bm25_index(scoreable)
bm25_raw = bm25.score(conv)

for i, row in scoreable.iterrows():
    if "I Ching" in row["normalized_facet"]:
        print(f"{row['normalized_facet']}: Dense={dense_scores[i]:.4f}, BM25={bm25_raw[i]:.4f}")
        print("  Indexed text:", build_facet_semantic_text(row))
        break

for i, row in scoreable.iterrows():
    if row["normalized_facet"] == "Perseverance":
        print(f"{row['normalized_facet']}: Dense={dense_scores[i]:.4f}, BM25={bm25_raw[i]:.4f}")
        print("  Indexed text:", build_facet_semantic_text(row))
        break

for i, row in scoreable.iterrows():
    if row["normalized_facet"] == "Self-improvement":
        print(f"{row['normalized_facet']}: Dense={dense_scores[i]:.4f}, BM25={bm25_raw[i]:.4f}")
        print("  Indexed text:", build_facet_semantic_text(row))
        break
