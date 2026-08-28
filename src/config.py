"""
Central configuration for the facet-scoring pipeline.

All tuneable parameters are defined here. Environment variables override defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_CSV_PATH = RAW_DIR / "Facets Assignment.csv"
ENRICHED_CSV_PATH = PROCESSED_DIR / "enriched_facets.csv"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_index"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
FACET_IDS_PATH = PROCESSED_DIR / "facet_ids.json"

# ── LLM / Inference ───────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Model: Llama 3.1 8B Instruct (Meta Llama 3.1 Community License, open-weight 8B parameters <= 16B, via Groq)
SCORING_MODEL = os.getenv("SCORING_MODEL", "llama-3.1-8b-instant")

# ── Embedding ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "20"))

# ── Scoring ────────────────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
SCORE_MIN = 1
SCORE_MAX = 5
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2.0

# ── Valid output statuses ──────────────────────────────────────────────────
VALID_STATUSES = {"scored", "insufficient_evidence", "not_observable", "unsuitable"}

# ── Taxonomy categories ───────────────────────────────────────────────────
FACET_TYPES = {
    "conversation_observable",  # Can be inferred from conversation
    "external_evidence",        # Requires external data (counts, metrics)
    "medical_health",           # Lab values, diagnoses, biological markers
    "biographical",             # Personal history, demographics
    "ambiguous",                # Might be partially observable but risky
    "malformed_header",         # Section headers, invalid entries
}

# ── Sensitivity levels ─────────────────────────────────────────────────────
SENSITIVITY_LEVELS = {"low", "medium", "high", "critical"}
