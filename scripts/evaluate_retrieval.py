"""
Retrieval evaluation module.

Evaluates candidate facet retrieval performance across benchmark conversations:
- Recall@10 and Recall@20 for reference target facets
- Mean Reciprocal Rank (MRR)
- Retrieval ranking distribution
- Retrieval latency and candidate relevance inspection
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.config import (
    ENRICHED_CSV_PATH,
    BENCHMARK_DIR,
    OUTPUT_DIR,
    TOP_K,
)
from src.retrieval import load_enriched_facets, retrieve_relevant_facets

BENCHMARK_CONVERSATIONS_PATH = BENCHMARK_DIR / "conversations.jsonl"
BENCHMARK_REFERENCE_PATH = BENCHMARK_DIR / "reference_labels.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_retrieval(
    conversations_path: Path = BENCHMARK_CONVERSATIONS_PATH,
    reference_path: Path = BENCHMARK_REFERENCE_PATH,
    enriched_csv_path: Path = ENRICHED_CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run retrieval evaluation on all benchmark conversations against reference labels."""
    enriched_df = load_enriched_facets(enriched_csv_path)

    # Load conversations
    conversations = []
    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    # Load reference labels grouped by conversation_id
    ref_labels_by_conv: dict[str, list[dict]] = {}
    with open(reference_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cid = item["conversation_id"]
                ref_labels_by_conv.setdefault(cid, []).append(item)

    conv_results = []
    total_observable_targets = 0
    hits_at_10 = 0
    hits_at_20 = 0
    reciprocal_ranks = []
    latencies_ms = []

    logger.info("=" * 65)
    logger.info("EVALUATING RETRIEVAL PERFORMANCE ON BENCHMARK SUITE")
    logger.info("=" * 65)

    for conv in conversations:
        cid = conv["conversation_id"]
        text = conv["text"]
        ref_items = ref_labels_by_conv.get(cid, [])

        t0 = time.perf_counter()
        retrieved = retrieve_relevant_facets(text, enriched_df, top_k=20)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(t_elapsed_ms)

        retrieved_names_ordered = [r["normalized_facet"].strip().rstrip(":").lower() for r in retrieved]
        retrieved_top10 = set(retrieved_names_ordered[:10])
        retrieved_top20 = set(retrieved_names_ordered[:20])

        conv_eval = {
            "conversation_id": cid,
            "latency_ms": round(t_elapsed_ms, 2),
            "targets": [],
        }

        for ref in ref_items:
            facet_name = ref["facet"].strip().rstrip(":").lower()
            expected_status = ref.get("expected_status", "scored")

            # We focus retrieval recall on observable facets that have conversational signals
            if expected_status in ["scored", "insufficient_evidence"]:
                total_observable_targets += 1
                in_top10 = facet_name in retrieved_top10
                in_top20 = facet_name in retrieved_top20

                rank = None
                if facet_name in retrieved_names_ordered:
                    rank = retrieved_names_ordered.index(facet_name) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

                if in_top10:
                    hits_at_10 += 1
                if in_top20:
                    hits_at_20 += 1

                conv_eval["targets"].append({
                    "facet": ref["facet"],
                    "expected_status": expected_status,
                    "rank": rank,
                    "in_top10": in_top10,
                    "in_top20": in_top20,
                })

        conv_results.append(conv_eval)

    recall_at_10 = hits_at_10 / total_observable_targets if total_observable_targets else 0.0
    recall_at_20 = hits_at_20 / total_observable_targets if total_observable_targets else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    report = {
        "num_conversations": len(conversations),
        "total_evaluated_targets": total_observable_targets,
        "recall_at_10": round(recall_at_10, 4),
        "recall_at_20": round(recall_at_20, 4),
        "mean_reciprocal_rank": round(mrr, 4),
        "avg_retrieval_latency_ms": round(avg_latency_ms, 2),
        "conversations": conv_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "retrieval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Recall@10: {recall_at_10:.1%} ({hits_at_10}/{total_observable_targets})")
    logger.info(f"Recall@20: {recall_at_20:.1%} ({hits_at_20}/{total_observable_targets})")
    logger.info(f"MRR:       {mrr:.4f}")
    logger.info(f"Avg Latency: {avg_latency_ms:.2f} ms")
    logger.info(f"Saved retrieval report to {report_path}")

    return report


if __name__ == "__main__":
    evaluate_retrieval()
