"""
Script to generate embeddings and build the FAISS index for retrieval.

Usage:
    python scripts/build_index.py
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ENRICHED_CSV_PATH, FAISS_INDEX_PATH
from src.retrieval import load_enriched_facets, build_retrieval_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not ENRICHED_CSV_PATH.exists():
        logger.error(f"Enriched facets file not found at {ENRICHED_CSV_PATH}. Run scripts/preprocess.py first.")
        sys.exit(1)

    logger.info(f"Loading enriched facets from {ENRICHED_CSV_PATH}")
    df = load_enriched_facets(ENRICHED_CSV_PATH)

    logger.info(f"Building retrieval index at {FAISS_INDEX_PATH}...")
    build_retrieval_index(df, FAISS_INDEX_PATH)
    logger.info("Retrieval index successfully built and saved.")

if __name__ == "__main__":
    main()
