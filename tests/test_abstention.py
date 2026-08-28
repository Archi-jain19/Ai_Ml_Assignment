"""
Tests for abstention logic, adversarial hallucination traps, and special cases.
"""

import pytest
from src.scoring import _heuristic_offline_score_batch


def test_adversarial_medical_serotonin_abstains():
    conv = "I've been feeling pretty fatigued and low on energy lately, probably because of the gloomy winter weather."
    facets = [{
        "normalized_facet": "Serotonin Transporter Availability",
        "facet_type": "medical_health",
        "conversation_observable": False,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] == "not_observable"
    assert res[0]["score"] is None


def test_adversarial_external_commute_abstains():
    conv = "Traffic in the city has been getting worse every month. People are constantly complaining about gridlock."
    facets = [{
        "normalized_facet": "Commute Time/Day",
        "facet_type": "external_evidence",
        "conversation_observable": False,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] == "not_observable"
    assert res[0]["score"] is None


def test_adversarial_biographical_nationality_abstains():
    conv = "I really enjoy cooking authentic Italian pasta and watching French cinema on weekends."
    facets = [{
        "normalized_facet": "Nationality",
        "facet_type": "biographical",
        "conversation_observable": False,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] == "insufficient_evidence"
    assert res[0]["score"] is None


def test_quoted_speech_hostility_separation():
    conv = "My manager walked in and screamed, 'You are all completely incompetent!' I just kept my voice calm."
    facets = [{
        "normalized_facet": "Hostility",
        "facet_type": "conversation_observable",
        "conversation_observable": True,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] == "scored"
    assert res[0]["score"] == 1  # Speaker was calm, manager was hostile


def test_sarcasm_recognition_happiness():
    conv = "Oh, wonderful! Another unexpected 7 AM production outage. Truly the highlight of my week."
    facets = [{
        "normalized_facet": "Happiness",
        "facet_type": "conversation_observable",
        "conversation_observable": True,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] == "scored"
    assert res[0]["score"] == 1  # Sarcasm indicates low happiness


def test_hallucination_trap_doctor_cholesterol_abstains():
    """Example 1: 'My doctor said my cholesterol is fine.' Must NOT invent a cholesterol score/value."""
    conv = "My doctor said my cholesterol is fine."
    facets = [{
        "normalized_facet": "Cholesterol Level (mg/dL)",
        "facet_type": "medical_health",
        "conversation_observable": False,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] == "not_observable"
    assert res[0]["score"] is None


def test_hallucination_trap_usually_wake_up_consistency_abstains():
    """Example 2: 'I usually wake up at 6 AM.' Wake-time Consistency must abstain on single unverified mention."""
    conv = "I usually wake up at 6 AM."
    facets = [{
        "normalized_facet": "Wake-time Consistency",
        "facet_type": "conversation_observable",
        "conversation_observable": True,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] in {"insufficient_evidence", "not_observable"}
    assert res[0]["score"] is None


def test_hallucination_trap_third_party_friend_patient_abstains():
    """Example 3: 'My friend is extremely patient.' Must NOT score target candidate as patient."""
    conv = "My friend is extremely patient."
    facets = [{
        "normalized_facet": "Patience: Resistance to Anger",
        "facet_type": "conversation_observable",
        "conversation_observable": True,
    }]
    res = _heuristic_offline_score_batch(conv, facets)
    assert len(res) == 1
    assert res[0]["status"] == "insufficient_evidence"
    assert res[0]["score"] is None
