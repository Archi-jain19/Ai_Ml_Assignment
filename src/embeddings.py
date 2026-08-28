"""
Embedding generation and FAISS index management.

Uses sentence-transformers (all-MiniLM-L6-v2) to embed facet names,
and FAISS for efficient similarity search.

Caching Strategy
----------------
- Embeddings are saved to disk as .npy files
- FAISS index is saved/loaded from disk
- Facet ID mapping is saved as JSON
- Cache is invalidated when the enriched CSV changes (by row count check)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    FACET_IDS_PATH,
)

logger = logging.getLogger(__name__)

# Lazy-load the model to avoid import-time download
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Parameters
    ----------
    texts : list[str]
        Texts to embed.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    np.ndarray
        Array of shape (len(texts), EMBEDDING_DIM).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(
    facet_names: list[str],
    facet_ids: list[int],
    save_dir: Optional[Path] = None,
    texts: Optional[list[str]] = None,
) -> faiss.IndexFlatIP:
    """
    Build a FAISS index from facet embeddings.

    Uses IndexFlatIP (inner product) since embeddings are L2-normalized,
    making IP equivalent to cosine similarity.

    Parameters
    ----------
    facet_names : list[str]
        Normalized facet names (used for logging / backward-compat).
    facet_ids : list[int]
        Row indices into the enriched CSV for each facet.
    save_dir : Path, optional
        Directory to save index files.
    texts : list[str], optional
        Alternative texts to embed instead of facet_names.
        When provided (e.g. "Name: Definition"), retrieval quality improves
        because the definition anchors the semantic meaning beyond the name alone.

    Returns
    -------
    faiss.IndexFlatIP
        The built index.
    """
    save_dir = save_dir or FAISS_INDEX_PATH
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    embed_input = texts if texts is not None else facet_names
    logger.info(f"Generating embeddings for {len(embed_input)} facets (using {'enriched text' if texts else 'name only'})...")
    embeddings = embed_texts(embed_input)

    logger.info(f"Building FAISS index (dim={EMBEDDING_DIM})...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # Save index
    index_path = save_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    # Save embeddings
    np.save(str(EMBEDDINGS_PATH), embeddings)

    # Save facet ID mapping
    with open(FACET_IDS_PATH, "w") as f:
        json.dump(facet_ids, f)

    logger.info(f"Saved FAISS index ({index.ntotal} vectors) to {save_dir}")
    return index


def load_faiss_index(
    index_dir: Optional[Path] = None,
) -> tuple[faiss.IndexFlatIP, list[int]]:
    """
    Load a previously saved FAISS index and facet ID mapping.

    Returns
    -------
    tuple[faiss.IndexFlatIP, list[int]]
        The index and the list of facet row IDs.
    """
    index_dir = Path(index_dir or FAISS_INDEX_PATH)
    index_path = index_dir / "index.faiss"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/build_index.py first."
        )

    index = faiss.read_index(str(index_path))

    with open(FACET_IDS_PATH, "r") as f:
        facet_ids = json.load(f)

    logger.info(f"Loaded FAISS index ({index.ntotal} vectors) from {index_dir}")
    return index, facet_ids


def search_similar(
    query_text: str,
    index: faiss.IndexFlatIP,
    facet_ids: list[int],
    top_k: int = 20,
) -> list[tuple[int, float]]:
    """
    Search the FAISS index for facets similar to the query.

    Parameters
    ----------
    query_text : str
        The conversation text to search against.
    index : faiss.IndexFlatIP
        The FAISS index.
    facet_ids : list[int]
        Mapping from index position to enriched CSV row index.
    top_k : int
        Number of results to return.

    Returns
    -------
    list[tuple[int, float]]
        List of (facet_row_id, similarity_score) tuples.
    """
    query_embedding = embed_texts([query_text])
    distances, indices = index.search(query_embedding, min(top_k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0:  # FAISS returns -1 for padding
            results.append((facet_ids[idx], float(dist)))

    return results
