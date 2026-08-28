"""
Script to evaluate scoring outputs against human reference labels.

Usage:
    python scripts/evaluate.py
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from src.config import BENCHMARK_DIR, OUTPUT_DIR
from src.evaluation import load_jsonl, evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    ref_path = BENCHMARK_DIR / "reference_labels.jsonl"
    pred_path = OUTPUT_DIR / "scoring_results.jsonl"

    if not ref_path.exists():
        logger.error(f"Reference labels not found at {ref_path}")
        sys.exit(1)

    if not pred_path.exists():
        logger.error(f"Predictions not found at {pred_path}. Run scripts/run_scoring.py first.")
        sys.exit(1)

    logger.info(f"Loading reference labels from {ref_path}...")
    references = load_jsonl(ref_path)

    logger.info(f"Loading predictions from {pred_path}...")
    predictions = load_jsonl(pred_path)

    logger.info(f"Evaluating {len(references)} reference cases...")
    report = evaluate_predictions(references, predictions)

    # Print summary
    print("\n" + "=" * 65)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 65)
    for k, v in report["summary"].items():
        print(f"{k.replace('_', ' ').title():<35}: {v}")
    print("=" * 65)

    if report["failure_cases"]:
        print(f"\n--- Failure Cases ({len(report['failure_cases'])}) ---")
        for idx, fc in enumerate(report["failure_cases"], 1):
            print(f"[{idx}] Conv: {fc['conversation_id']} | Facet: {fc['facet']}")
            print(f"    Failure Type: {fc['failure_type']}")
            print(f"    Expected: status={fc['expected_status']}, score={fc['expected_score']}")
            print(f"    Predicted: status={fc['predicted_status']}, score={fc['predicted_score']}")
            print(f"    Human Reasoning: {fc['human_reasoning']}")
            print(f"    Model Reason: {fc['predicted_reason']}")
            print("-" * 50)

    # Save report
    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Full evaluation report written to {report_path}")

if __name__ == "__main__":
    main()
