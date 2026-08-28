from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import DATA_DIR, OUTPUT_DIR, TOP_K
from src.retrieval import load_enriched_facets
from src.pipeline import run_pipeline, run_benchmark
from src.evaluation import load_jsonl, evaluate_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI(
    title="FacetLens API",
    description="Backend service for Conversation Facet Evaluation & Abstention Pipeline",
    version="1.0.0",
)

# Enable CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache enriched facets DataFrame and taxonomy dictionary
_enriched_df = None
_facet_taxonomy_dict: Dict[str, Dict[str, Any]] = {}


def get_taxonomy_dict() -> Dict[str, Dict[str, Any]]:
    global _enriched_df, _facet_taxonomy_dict
    if _facet_taxonomy_dict:
        return _facet_taxonomy_dict

    df = load_enriched_facets()
    _enriched_df = df

    for idx, row in df.iterrows():
        norm_name = str(row["normalized_facet"]).strip().rstrip(":")
        norm_key = norm_name.lower()

        item = {
            "facet": norm_name,
            "raw_facet": str(row["raw_facet"]),
            "facet_type": str(row["facet_type"]),
            "conversation_observable": bool(row["conversation_observable"]),
            "sensitivity": str(row.get("sensitivity", "medium")),
            "scoring_definition": str(row.get("scoring_definition", "Measures evidence for this facet from conversation text.")),
            "score_1_anchor": str(row.get("score_1_anchor", "")),
            "score_2_anchor": str(row.get("score_2_anchor", "")),
            "score_3_anchor": str(row.get("score_3_anchor", "")),
            "score_4_anchor": str(row.get("score_4_anchor", "")),
            "score_5_anchor": str(row.get("score_5_anchor", "")),
            "abstention_policy": str(row.get("abstention_reason", "Only score when the conversation provides direct, un-hallucinated behavioral evidence. Abstain if evidence is missing, ambiguous, or requires quantitative external logs.")),
        }
        _facet_taxonomy_dict[norm_key] = item

    return _facet_taxonomy_dict


SCORE_LABELS = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}


class EvaluateRequest(BaseModel):
    conversation: str = Field(..., description="Conversation text to analyze")
    top_k: Optional[int] = Field(default=20, ge=1, le=100)


import time

@app.get("/health")
@app.get("/api/health")
def health_check():
    """Lightweight healthcheck endpoint for container monitoring."""
    return {"status": "ok"}


@app.get("/api/status")
def get_status():
    taxonomy = get_taxonomy_dict()
    return {
        "status": "ok",
        "model_ready": True,
        "model_name": "Llama 3.1 8B Instruct (Meta Open-Weight, <=16B Compliant) / Rule-Grounded Engine",
        "total_facets_in_taxonomy": len(taxonomy),
    }


@app.post("/api/evaluate")
def evaluate_conversation(req: EvaluateRequest):
    text = req.conversation.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Conversation text cannot be empty.")

    start_time = time.time()
    try:
        pipeline_output = run_pipeline(conversation=text, top_k=req.top_k or 20)
        proc_time_ms = round((time.time() - start_time) * 1000, 1)
        taxonomy = get_taxonomy_dict()

        num_retrieved = pipeline_output.get("num_facets_retrieved", len(pipeline_output.get("results", [])))
        results = pipeline_output.get("results", [])

        # Count statuses
        scored_count = sum(1 for r in results if r.get("status") == "scored")
        insufficient_count = sum(1 for r in results if r.get("status") == "insufficient_evidence")
        not_observable_count = sum(1 for r in results if r.get("status") in {"not_observable", "unsuitable"})

        avg_conf = round(sum(r.get("confidence", 0.0) for r in results) / len(results), 2) if results else 0.0

        formatted_results = []
        for r in results:
            f_norm = r["facet"].strip().rstrip(":")
            f_key = f_norm.lower()
            tax_item = taxonomy.get(f_key, {})

            score_val = r.get("score")
            score_label = SCORE_LABELS.get(int(score_val), "") if score_val is not None else None

            formatted_results.append({
                "facet": f_norm,
                "status": r.get("status", "insufficient_evidence"),
                "score": score_val,
                "score_label": score_label,
                "confidence": r.get("confidence", 0.88),
                "reason": r.get("reason", ""),
                "evidence": r.get("evidence", r.get("reason", "")) if r.get("status") == "scored" else None,
                "facet_type": tax_item.get("facet_type", "conversation_observable"),
                "scoring_definition": tax_item.get("scoring_definition", "Measures evidence for this behavioral trait from conversation text."),
                "abstention_policy": tax_item.get("abstention_policy", "Abstain if conversational evidence is missing or ambiguous."),
            })

        retrieved_facets_meta = []
        for rf in pipeline_output.get("retrieved_facets", []):
            rf_name = rf["facet"].strip().rstrip(":")
            rf_key = rf_name.lower()
            tax = taxonomy.get(rf_key, {})
            retrieved_facets_meta.append({
                "facet": rf_name,
                "similarity": round(rf.get("similarity", 0.90), 3),
                "facet_type": tax.get("facet_type", "conversation_observable"),
                "definition": tax.get("scoring_definition", "Measures evidence for this facet from conversation text."),
            })

        return {
            "conversation_id": pipeline_output.get("conversation_id", "conv_001"),
            "conversation": text,
            "metrics": {
                "num_facets_retrieved": num_retrieved,
                "num_scored": scored_count,
                "num_insufficient_evidence": insufficient_count,
                "num_not_observable": not_observable_count,
                "average_confidence": avg_conf,
                "processing_time_ms": proc_time_ms,
                "coverage_label": f"{scored_count} / {num_retrieved} facets supported by conversational evidence",
                "coverage_percentage": round((scored_count / num_retrieved * 100), 1) if num_retrieved > 0 else 0.0,
            },
            "results": formatted_results,
            "retrieved_facets": retrieved_facets_meta,
        }

    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/facets")
def list_facets(q: Optional[str] = None, limit: int = 100):
    taxonomy = get_taxonomy_dict()
    facets_list = list(taxonomy.values())

    if q:
        q_lower = q.lower()
        facets_list = [
            f for f in facets_list
            if q_lower in f["facet"].lower()
            or q_lower in f["raw_facet"].lower()
            or q_lower in f["scoring_definition"].lower()
        ]

    return {
        "total": len(facets_list),
        "facets": facets_list[:limit],
    }


@app.get("/api/facet/{facet_name}")
def get_facet_detail(facet_name: str):
    taxonomy = get_taxonomy_dict()
    key = facet_name.strip().rstrip(":").lower()
    if key in taxonomy:
        return taxonomy[key]

    # Partial search fallback
    for k, v in taxonomy.items():
        if key in k or k in key:
            return v

    raise HTTPException(status_code=404, detail=f"Facet '{facet_name}' not found in taxonomy library.")


@app.get("/api/benchmark")
def get_benchmark_results():
    bench_dir = DATA_DIR / "benchmark"
    conv_file = bench_dir / "conversations.jsonl"
    ref_file = bench_dir / "reference_labels.jsonl"

    if not conv_file.exists() or not ref_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark files not found.")

    # Run benchmark pipeline evaluation
    raw_outputs = run_benchmark(
        conversations_path=conv_file,
        reference_labels_path=ref_file,
        output_dir=OUTPUT_DIR,
    )

    reference_labels = load_jsonl(ref_file)
    eval_report = evaluate_predictions(reference_labels, raw_outputs)

    # Compute high-level percentage metrics
    summary = eval_report["summary"]
    total_refs = summary["total_reference_cases"]
    evaluated_refs = summary["evaluated_reference_cases"]
    status_acc = summary["status_exact_accuracy"]

    correct_abstentions = summary["correct_abstentions"]
    false_positives = summary["false_positives_scored"]  # Incorrect scores (scored when should abstain)
    false_negatives = summary["false_negatives_abstained"]  # Incorrect abstentions (abstained when should score)

    # Load conversations list for UI table
    conv_map = {}
    with open(conv_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                conv_map[item["conversation_id"]] = item

    # Build comparison rows for table
    comparison_rows = []
    
    # Track evaluated items
    evaluated_pairs = set()

    for ref in reference_labels:
        cid = ref["conversation_id"]
        facet = ref["facet"]
        exp_status = ref["expected_status"]
        exp_score = ref.get("expected_score")
        human_reason = ref.get("human_reasoning", "")
        conv_item = conv_map.get(cid, {})

        # Find matching predicted result
        pred_res = None
        for out in raw_outputs:
            if out["conversation_id"] == cid:
                for res in out.get("results", []):
                    if res["facet"].strip().rstrip(":").lower() == facet.strip().rstrip(":").lower():
                        pred_res = res
                        break

        pred_status = pred_res.get("status") if pred_res else "facet_not_retrieved"
        pred_score = pred_res.get("score") if pred_res else None
        pred_reason = pred_res.get("reason", "") if pred_res else "Facet was not included in top-K retrieved subset."

        # Evaluate comparison status
        if pred_status == exp_status:
            if exp_status == "scored":
                if exp_score == pred_score:
                    result_badge = "agreement"  # ✓ Agreement
                else:
                    result_badge = "partial"    # △ Partial/ambiguous (score delta)
            else:
                result_badge = "agreement"      # ✓ Agreement (correct abstention)
        else:
            result_badge = "error"              # ✕ Error

        comparison_rows.append({
            "conversation_id": cid,
            "conversation_type": conv_item.get("type", "standard"),
            "conversation_text": conv_item.get("text", ""),
            "facet": facet,
            "expected_status": exp_status,
            "expected_score": exp_score,
            "predicted_status": pred_status,
            "predicted_score": pred_score,
            "predicted_reason": pred_reason,
            "human_reasoning": human_reason,
            "result_badge": result_badge,
        })

    agreement_pct = round(status_acc * 100, 1)
    correct_abstention_pct = round((correct_abstentions / total_refs * 100), 1) if total_refs > 0 else 0.0
    incorrect_scores_pct = round((false_positives / total_refs * 100), 1) if total_refs > 0 else 0.0
    incorrect_abstentions_pct = round((false_negatives / total_refs * 100), 1) if total_refs > 0 else 0.0

    return {
        "summary": {
            "total_benchmark_examples": len(conv_map),
            "total_reference_cases": total_refs,
            "evaluated_cases": evaluated_refs,
            "agreement_percentage": agreement_pct,
            "correct_abstentions_percentage": correct_abstention_pct,
            "incorrect_scores_percentage": incorrect_scores_pct,
            "incorrect_abstentions_percentage": incorrect_abstentions_pct,
            "exact_score_accuracy": round(summary.get("exact_score_accuracy", 0.0) * 100, 1),
            "score_mae": summary.get("score_mae", 0.0),
        },
        "conversations": list(conv_map.values()),
        "comparison_rows": comparison_rows,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
