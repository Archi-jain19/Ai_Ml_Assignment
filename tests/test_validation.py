"""
Tests for structured output schema validation and error coercion.
"""

import pytest
from src.validation import validate_single_result, validate_results, _try_coerce


def test_valid_scored_result():
    res = {
        "facet": "Persistence",
        "status": "scored",
        "score": 5,
        "confidence": 0.95,
        "reason": "Clear evidence of systematic debugging across three days.",
    }
    errors = validate_single_result(res)
    assert len(errors) == 0


def test_valid_abstained_result():
    res = {
        "facet": "Fsh Level",
        "status": "not_observable",
        "score": None,
        "confidence": 0.99,
        "reason": "Cannot infer lab tests from conversation.",
    }
    errors = validate_single_result(res)
    assert len(errors) == 0


def test_invalid_score_when_abstained():
    res = {
        "facet": "Fsh Level",
        "status": "not_observable",
        "score": 3,  # Invalid: score must be null
        "confidence": 0.90,
        "reason": "Test reason",
    }
    errors = validate_single_result(res)
    assert any(e.field == "score" for e in errors)


def test_invalid_score_out_of_range():
    res = {
        "facet": "Persistence",
        "status": "scored",
        "score": 6,  # Invalid: out of [1, 5]
        "confidence": 0.90,
        "reason": "Test reason",
    }
    errors = validate_single_result(res)
    assert any(e.field == "score" for e in errors)


def test_invalid_confidence_out_of_range():
    res = {
        "facet": "Persistence",
        "status": "scored",
        "score": 4,
        "confidence": 1.5,  # Invalid: > 1.0
        "reason": "Test reason",
    }
    errors = validate_single_result(res)
    assert any(e.field == "confidence" for e in errors)


def test_missing_required_field():
    res = {
        "facet": "Persistence",
        "status": "scored",
        "score": 4,
        # missing confidence and reason
    }
    errors = validate_single_result(res)
    assert len(errors) >= 2


def test_coercion_fixes_float_score():
    res = {
        "facet": "Persistence",
        "status": "scored",
        "score": 4.2,
        "confidence": 0.85,
        "reason": "Model returned float score",
    }
    coerced = _try_coerce(res)
    assert coerced is not None
    assert coerced["score"] == 4
    assert isinstance(coerced["score"], int)


def test_validate_results_detects_duplicates():
    results = [
        {"facet": "Persistence", "status": "scored", "score": 5, "confidence": 0.9, "reason": "R1"},
        {"facet": "Persistence", "status": "scored", "score": 4, "confidence": 0.8, "reason": "R2"},
    ]
    clean, errors = validate_results(results)
    assert any("Duplicate" in e.message for e in errors)
