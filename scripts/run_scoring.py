"""
Script to score benchmark conversations through the full pipeline.

Usage:
    python scripts/run_scoring.py
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BENCHMARK_DIR, ENRICHED_CSV_PATH, OUTPUT_DIR
from src.pipeline import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    conversations_path = BENCHMARK_DIR / "conversations.jsonl"
    reference_labels_path = BENCHMARK_DIR / "reference_labels.jsonl"
    if not conversations_path.exists():
        logger.error(f"Benchmark conversations file not found at {conversations_path}")
        sys.exit(1)

    logger.info(f"Running benchmark scoring for {conversations_path}...")
    outputs = run_benchmark(
        conversations_path=conversations_path,
        reference_labels_path=reference_labels_path,
        enriched_csv_path=ENRICHED_CSV_PATH,
        output_dir=OUTPUT_DIR,
    )

    logger.info(f"Completed scoring for {len(outputs)} conversations. Results saved to {OUTPUT_DIR / 'scoring_results.jsonl'}")

if __name__ == "__main__":
    main()
