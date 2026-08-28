"""
End-to-end pipeline orchestration.

Ties together: preprocessing → embedding → retrieval → scoring → validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    ENRICHED_CSV_PATH,
    OUTPUT_DIR,
    TOP_K,
    BATCH_SIZE,
)
from src.retrieval import load_enriched_facets, retrieve_relevant_facets
from src.scoring import score_conversation
from src.validation import validate_results

logger = logging.getLogger(__name__)


def run_pipeline(
    conversation: str,
    conversation_id: str = "conv_001",
    enriched_csv_path: Optional[Path] = None,
    top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run the full scoring pipeline for a single conversation.

    Parameters
    ----------
    conversation : str
        The conversation text to evaluate.
    conversation_id : str
        Identifier for this conversation.
    enriched_csv_path : Path, optional
        Path to enriched facets CSV.
    top_k : int, optional
        Number of facets to retrieve.
    batch_size : int, optional
        Scoring batch size.
    output_dir : Path, optional
        Where to save results.

    Returns
    -------
    dict
        Pipeline output with conversation_id, results, and metadata.
    """
    top_k = top_k or TOP_K
    batch_size = batch_size or BATCH_SIZE
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running pipeline for conversation '{conversation_id}'")

    # Step 1: Load enriched facets
    enriched_df = load_enriched_facets(enriched_csv_path)

    # Step 2: Retrieve relevant facets
    retrieved_facets = retrieve_relevant_facets(
        conversation, enriched_df, top_k=top_k
    )
    logger.info(f"Retrieved {len(retrieved_facets)} facets")

    # Step 3: Score retrieved facets via LLM
    raw_results = score_conversation(
        conversation, retrieved_facets, batch_size=batch_size
    )
    logger.info(f"Got {len(raw_results)} raw scoring results")

    # Step 4: Validate results
    clean_results, validation_errors = validate_results(raw_results)
    logger.info(
        f"Validation: {len(clean_results)} valid, "
        f"{len(validation_errors)} errors"
    )

    # Build output
    output = {
        "conversation_id": conversation_id,
        "conversation": conversation,
        "num_facets_retrieved": len(retrieved_facets),
        "num_results": len(clean_results),
        "num_validation_errors": len(validation_errors),
        "results": clean_results,
        "validation_errors": [
            {"facet": e.facet, "field": e.field, "message": e.message}
            for e in validation_errors
        ],
        "retrieved_facets": [
            {
                "facet": f["normalized_facet"],
                "similarity": f["similarity_score"],
                "definition": f.get("scoring_definition", ""),
            }
            for f in retrieved_facets
        ],
    }

    return output


def run_benchmark(
    conversations_path: Path,
    reference_labels_path: Optional[Path] = None,
    enriched_csv_path: Optional[Path] = None,
    top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
    force_reference_facets: bool = False,
) -> list[dict]:
    """
    Run the pipeline on all benchmark conversations.
    
    Parameters
    ----------
    force_reference_facets : bool
        If False (default), evaluates pure organic retrieval + scoring.
        If True, runs oracle retrieval ablation by ensuring reference facets are in the candidate set.
    """
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_df = load_enriched_facets(enriched_csv_path)

    # Map normalized facet names in enriched_df
    facet_meta_by_norm = {}
    for idx, row in enriched_df.iterrows():
        norm_key = row["normalized_facet"].strip().rstrip(":").lower()
        raw_key = row["raw_facet"].strip().rstrip(":").lower()
        meta = {
            "row_index": idx,
            "normalized_facet": row["normalized_facet"],
            "raw_facet": row["raw_facet"],
            "facet_type": row["facet_type"],
            "conversation_observable": bool(row["conversation_observable"]),
            "sensitivity": row["sensitivity"],
            "scoring_definition": row["scoring_definition"],
            "score_1_anchor": row["score_1_anchor"],
            "score_2_anchor": row["score_2_anchor"],
            "score_3_anchor": row["score_3_anchor"],
            "score_4_anchor": row["score_4_anchor"],
            "score_5_anchor": row["score_5_anchor"],
            "abstention_reason": row.get("abstention_reason", ""),
            "similarity_score": 1.0,
        }
        facet_meta_by_norm[norm_key] = meta
        facet_meta_by_norm[raw_key] = meta

    # Load benchmark reference facets if provided
    ref_facets_by_conv = {}
    if reference_labels_path and reference_labels_path.exists():
        with open(reference_labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cid = item["conversation_id"]
                    if cid not in ref_facets_by_conv:
                        ref_facets_by_conv[cid] = []
                    ref_facets_by_conv[cid].append(item["facet"])

    # Load conversations
    conversations = []
    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    logger.info(f"Loaded {len(conversations)} benchmark conversations (force_reference_facets={force_reference_facets})")

    all_outputs = []
    for conv in conversations:
        conv_id = conv["conversation_id"]
        conv_text = conv["text"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing conversation: {conv_id}")
        logger.info(f"{'='*60}")

        # Retrieve top-K facets organically
        retrieved_facets = retrieve_relevant_facets(
            conv_text, enriched_df, top_k=top_k or TOP_K
        )

        candidate_facets = list(retrieved_facets)
        if force_reference_facets and conv_id in ref_facets_by_conv:
            retrieved_names = {f["normalized_facet"].strip().rstrip(":").lower() for f in candidate_facets}
            for target_facet in ref_facets_by_conv[conv_id]:
                target_key = target_facet.strip().rstrip(":").lower()
                if target_key not in retrieved_names and target_key in facet_meta_by_norm:
                    candidate_facets.append(facet_meta_by_norm[target_key])
                    retrieved_names.add(target_key)

        # Score candidates
        raw_results = score_conversation(
            conv_text, candidate_facets, batch_size=batch_size or BATCH_SIZE
        )
        clean_results, validation_errors = validate_results(raw_results)

        output = {
            "conversation_id": conv_id,
            "conversation": conv_text,
            "num_facets_retrieved": len(candidate_facets),
            "num_results": len(clean_results),
            "num_validation_errors": len(validation_errors),
            "results": clean_results,
            "validation_errors": [
                {"facet": e.facet, "field": e.field, "message": e.message}
                for e in validation_errors
            ],
            "retrieved_facets": [
                {"facet": f["normalized_facet"], "similarity": f.get("similarity_score", 0.0)}
                for f in candidate_facets
            ],
        }
        all_outputs.append(output)

    # Save all results
    results_path = output_dir / "scoring_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for output in all_outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(all_outputs)} results to {results_path}")
    return all_outputs

