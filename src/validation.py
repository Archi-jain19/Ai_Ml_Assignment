"""
Output validation module.

Validates structured scoring results against the expected schema.
Malformed outputs are logged but do NOT crash the pipeline.

Validation Rules
----------------
1. Each result must have: facet, status, score, confidence, reason
2. status ∈ {scored, insufficient_evidence, not_observable, unsuitable}
3. When status == "scored": score must be integer 1-5
4. When status != "scored": score must be null/None
5. confidence must be numeric, 0.0 <= confidence <= 1.0
6. reason must be a non-empty string
7. No duplicate facets in results (unless explicitly expected)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import VALID_STATUSES, SCORE_MIN, SCORE_MAX

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a single validation error."""

    def __init__(self, facet: str, field: str, message: str):
        self.facet = facet
        self.field = field
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError(facet='{self.facet}', field='{self.field}', message='{self.message}')"


def validate_single_result(result: dict) -> list[ValidationError]:
    """
    Validate a single scoring result dict.

    Returns a list of validation errors (empty if valid).
    """
    errors = []
    facet_name = result.get("facet", "<missing>")

    # ── Required fields ──────────────────────────────────────────────
    required = ["facet", "status", "score", "confidence", "reason"]
    for field in required:
        if field not in result:
            errors.append(ValidationError(facet_name, field, f"Missing required field: {field}"))

    if errors:
        # Can't validate further without required fields
        return errors

    # ── Status validation ────────────────────────────────────────────
    status = result["status"]
    if status not in VALID_STATUSES:
        errors.append(ValidationError(
            facet_name, "status",
            f"Invalid status '{status}'. Must be one of: {VALID_STATUSES}"
        ))

    # ── Score validation ─────────────────────────────────────────────
    score = result["score"]
    if status == "scored":
        if score is None:
            errors.append(ValidationError(
                facet_name, "score",
                "Score must not be null when status is 'scored'"
            ))
        elif not isinstance(score, (int, float)):
            errors.append(ValidationError(
                facet_name, "score",
                f"Score must be an integer, got {type(score).__name__}: {score}"
            ))
        else:
            score_int = int(score)
            if score_int != score:
                errors.append(ValidationError(
                    facet_name, "score",
                    f"Score must be an integer, got float: {score}"
                ))
            elif score_int < SCORE_MIN or score_int > SCORE_MAX:
                errors.append(ValidationError(
                    facet_name, "score",
                    f"Score {score_int} out of range [{SCORE_MIN}, {SCORE_MAX}]"
                ))
    else:
        if score is not None:
            errors.append(ValidationError(
                facet_name, "score",
                f"Score must be null when status is '{status}', got {score}"
            ))

    # ── Confidence validation ────────────────────────────────────────
    confidence = result["confidence"]
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            errors.append(ValidationError(
                facet_name, "confidence",
                f"Confidence must be numeric, got {type(confidence).__name__}"
            ))
        elif confidence < 0.0 or confidence > 1.0:
            errors.append(ValidationError(
                facet_name, "confidence",
                f"Confidence {confidence} out of range [0.0, 1.0]"
            ))
    else:
        errors.append(ValidationError(
            facet_name, "confidence",
            "Confidence must not be null"
        ))

    # ── Reason validation ────────────────────────────────────────────
    reason = result["reason"]
    if not reason or not isinstance(reason, str) or not reason.strip():
        errors.append(ValidationError(
            facet_name, "reason",
            "Reason must be a non-empty string"
        ))

    return errors


def validate_results(results: list[dict]) -> tuple[list[dict], list[ValidationError]]:
    """
    Validate a list of scoring results.

    Returns
    -------
    tuple[list[dict], list[ValidationError]]
        - Clean results (validated and possibly coerced)
        - All validation errors encountered
    """
    all_errors = []
    clean_results = []
    seen_facets = set()

    for result in results:
        errors = validate_single_result(result)

        # Check for duplicates
        facet_name = result.get("facet", "")
        if facet_name in seen_facets:
            errors.append(ValidationError(
                facet_name, "facet",
                f"Duplicate facet result: '{facet_name}'"
            ))
        seen_facets.add(facet_name)

        if errors:
            all_errors.extend(errors)
            # Try to salvage what we can
            coerced = _try_coerce(result)
            if coerced:
                clean_results.append(coerced)
                logger.warning(
                    f"Coerced result for '{facet_name}': "
                    f"{[str(e) for e in errors]}"
                )
            else:
                logger.error(
                    f"Discarded result for '{facet_name}': "
                    f"{[str(e) for e in errors]}"
                )
        else:
            # Ensure score is int when scored
            if result.get("status") == "scored" and result.get("score") is not None:
                result["score"] = int(result["score"])
            clean_results.append(result)

    if all_errors:
        logger.warning(f"Validation found {len(all_errors)} errors across {len(results)} results")

    return clean_results, all_errors


def _try_coerce(result: dict) -> Optional[dict]:
    """
    Attempt to coerce a partially-valid result into a valid one.

    - If score is a float, round to int
    - If confidence is out of range, clamp
    - If status is invalid, fall back to insufficient_evidence
    """
    try:
        coerced = dict(result)

        # Ensure required fields exist
        if "facet" not in coerced:
            return None
        if "status" not in coerced:
            coerced["status"] = "insufficient_evidence"
        if "reason" not in coerced or not coerced["reason"]:
            coerced["reason"] = "No reason provided by model."

        # Coerce status
        if coerced["status"] not in VALID_STATUSES:
            coerced["status"] = "insufficient_evidence"
            coerced["score"] = None

        # Coerce score
        if coerced["status"] == "scored":
            if coerced.get("score") is not None:
                try:
                    s = int(round(float(coerced["score"])))
                    s = max(SCORE_MIN, min(SCORE_MAX, s))
                    coerced["score"] = s
                except (ValueError, TypeError):
                    coerced["status"] = "insufficient_evidence"
                    coerced["score"] = None
            else:
                coerced["status"] = "insufficient_evidence"
        else:
            coerced["score"] = None

        # Coerce confidence
        if coerced.get("confidence") is not None:
            try:
                c = float(coerced["confidence"])
                coerced["confidence"] = max(0.0, min(1.0, c))
            except (ValueError, TypeError):
                coerced["confidence"] = 0.0
        else:
            coerced["confidence"] = 0.0

        return coerced
    except Exception:
        return None
