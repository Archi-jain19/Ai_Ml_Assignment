"""
Evaluation module for the facet scoring benchmark.

Compares model predictions against human-reviewed reference labels:
- Score agreement (exact match, mean absolute error)
- Abstention metrics (precision, recall, F1 for abstention status)
- Status agreement (scored vs insufficient_evidence vs not_observable vs unsuitable)
- Failure mode categorization and analysis
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_predictions(
    reference_labels: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate system predictions against reference labels.

    Parameters
    ----------
    reference_labels : List[Dict]
        Human reference labels with fields: conversation_id, facet, expected_status,
        expected_score (optional), human_reasoning.
    predictions : List[Dict]
        Pipeline predictions with conversation_id and results array.

    Returns
    -------
    Dict[str, Any]
        Comprehensive evaluation report with metrics and failure modes.
    """
    # Index predictions by (conversation_id, facet_normalized)
    pred_map = {}
    for pred in predictions:
        cid = pred["conversation_id"]
        for res in pred.get("results", []):
            f_norm = res["facet"].strip().rstrip(":").lower()
            pred_map[(cid, f_norm)] = res


    # Metrics accumulators
    total_refs = len(reference_labels)
    matched_refs = 0
    missing_refs = 0

    status_true = []
    status_pred = []

    score_errors = []
    exact_score_matches = 0
    scored_comparisons = 0

    correct_abstentions = 0
    incorrect_abstentions = 0
    false_positives_scored = 0  # Scored when should have abstained
    false_negatives_abstained = 0  # Abstained when should have been scored

    failure_cases = []
    agreements = []

    for ref in reference_labels:
        cid = ref["conversation_id"]
        facet = ref["facet"]
        f_norm = facet.strip().rstrip(":").lower()
        exp_status = ref["expected_status"]

        exp_score = ref.get("expected_score")
        human_reason = ref.get("human_reasoning", "")

        key = (cid, f_norm)
        if key not in pred_map:
            missing_refs += 1
            failure_cases.append({
                "conversation_id": cid,
                "facet": facet,
                "failure_type": "facet_not_retrieved_or_evaluated",
                "expected_status": exp_status,
                "expected_score": exp_score,
                "human_reasoning": human_reason,
                "predicted_status": None,
                "predicted_score": None,
                "predicted_reason": None,
            })
            continue

        matched_refs += 1
        pred_res = pred_map[key]
        act_status = pred_res.get("status")
        act_score = pred_res.get("score")
        act_reason = pred_res.get("reason", "")
        act_conf = pred_res.get("confidence", 0.0)

        status_true.append(exp_status)
        status_pred.append(act_status)

        is_exp_abstain = exp_status in {"insufficient_evidence", "not_observable", "unsuitable"}
        is_act_abstain = act_status in {"insufficient_evidence", "not_observable", "unsuitable"}

        # Abstention breakdown
        if is_exp_abstain and is_act_abstain:
            correct_abstentions += 1
            agreements.append({
                "conversation_id": cid,
                "facet": facet,
                "type": "correct_abstention",
                "expected_status": exp_status,
                "actual_status": act_status,
                "reason": act_reason,
            })
        elif is_exp_abstain and not is_act_abstain:
            false_positives_scored += 1
            failure_cases.append({
                "conversation_id": cid,
                "facet": facet,
                "failure_type": "hallucination_false_positive_score",
                "expected_status": exp_status,
                "expected_score": exp_score,
                "human_reasoning": human_reason,
                "predicted_status": act_status,
                "predicted_score": act_score,
                "predicted_reason": act_reason,
            })
        elif not is_exp_abstain and is_act_abstain:
            false_negatives_abstained += 1
            incorrect_abstentions += 1
            failure_cases.append({
                "conversation_id": cid,
                "facet": facet,
                "failure_type": "missed_evidence_false_abstention",
                "expected_status": exp_status,
                "expected_score": exp_score,
                "human_reasoning": human_reason,
                "predicted_status": act_status,
                "predicted_score": act_score,
                "predicted_reason": act_reason,
            })
        else:
            # Both scored
            if exp_score is not None and act_score is not None:
                scored_comparisons += 1
                diff = abs(int(exp_score) - int(act_score))
                score_errors.append(diff)
                if diff == 0:
                    exact_score_matches += 1
                    agreements.append({
                        "conversation_id": cid,
                        "facet": facet,
                        "type": "exact_score_match",
                        "score": act_score,
                        "reason": act_reason,
                    })
                else:
                    failure_cases.append({
                        "conversation_id": cid,
                        "facet": facet,
                        "failure_type": f"score_discrepancy_delta_{diff}",
                        "expected_status": exp_status,
                        "expected_score": exp_score,
                        "human_reasoning": human_reason,
                        "predicted_status": act_status,
                        "predicted_score": act_score,
                        "predicted_reason": act_reason,
                    })

    status_accuracy = sum(1 for t, p in zip(status_true, status_pred) if t == p) / len(status_true) if status_true else 0.0
    exact_score_acc = exact_score_matches / scored_comparisons if scored_comparisons > 0 else 0.0
    mean_abs_error = float(np.mean(score_errors)) if score_errors else 0.0

    report = {
        "summary": {
            "total_reference_cases": total_refs,
            "evaluated_reference_cases": matched_refs,
            "missing_from_predictions": missing_refs,
            "status_exact_accuracy": round(status_accuracy, 4),
            "scored_comparisons": scored_comparisons,
            "exact_score_matches": exact_score_matches,
            "exact_score_accuracy": round(exact_score_acc, 4),
            "score_mae": round(mean_abs_error, 4),
            "correct_abstentions": correct_abstentions,
            "false_positives_scored": false_positives_scored,
            "false_negatives_abstained": false_negatives_abstained,
        },
        "agreements": agreements,
        "failure_cases": failure_cases,
    }

    return report
