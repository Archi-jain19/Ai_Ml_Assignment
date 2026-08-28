"""
Script to run the facet audit and preprocessing pipeline.

Usage:
    python scripts/preprocess.py
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from src.config import RAW_CSV_PATH, ENRICHED_CSV_PATH
from src.preprocessing import load_raw_facets, preprocess_facets, generate_audit_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Loading raw facets from {RAW_CSV_PATH}")
    raw_facets = load_raw_facets(RAW_CSV_PATH)
    logger.info(f"Loaded {len(raw_facets)} raw facet rows.")

    logger.info(f"Preprocessing and enriching facets...")
    df = preprocess_facets(raw_facets, ENRICHED_CSV_PATH)
    logger.info(f"Successfully generated enriched facet dataset at {ENRICHED_CSV_PATH}")

    # Generate and print summary audit report
    report = generate_audit_report(df)
    print("\n" + report)

    # Save report to text file
    report_path = ENRICHED_CSV_PATH.parent / "audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Audit report written to {report_path}")

if __name__ == "__main__":
    main()
