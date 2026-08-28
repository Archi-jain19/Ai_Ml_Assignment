"""
Regression test suite for evidence-based scoring and abstention guardrails.

Verifies:
- Direct behavioral evidence scoring (Perseverance, Troubleshooting, Teamwork, Deadlines)
- Weak evidence and no-evidence abstention
- Anti-hallucination guardrails (Medical diagnoses, External telemetry, Biographical data, Structural headers)
- Speaker attribution (Quoted speech and third-party subjects do NOT attribute to target candidate)
- Sarcasm / sentiment inversion detection
- Brevity evaluation for short utterances
"""

import pytest
from src.scoring import _heuristic_offline_score_batch


def test_1_strong_persistence_and_troubleshooting():
    conv = "I kept trying different approaches to solve the problem. My first three attempts failed, but instead of giving up, I researched the bug, tested each fix, and stayed up until I solved it."
    facets = [
        {"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Measures persistence"},
        {"normalized_facet": "Troubleshooting Technical Issues", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Measures debugging"},
        {"normalized_facet": "Volunteer Work", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Measures community service"},
        {"normalized_facet": "Blood Pressure", "facet_type": "medical_health", "conversation_observable": False, "scoring_definition": "Medical vitals"},
    ]
    results = {r["facet"]: r for r in _heuristic_offline_score_batch(conv, facets)}

    # Observable supported traits score
    assert results["Perseverance"]["status"] == "scored"
    assert results["Perseverance"]["score"] in [4, 5]
    assert results["Troubleshooting Technical Issues"]["status"] == "scored"
    assert results["Troubleshooting Technical Issues"]["score"] in [4, 5]

    # Unsupported observable trait abstains
    assert results["Volunteer Work"]["status"] == "insufficient_evidence"
    assert results["Volunteer Work"]["score"] is None

    # Medical trait is not observable
    assert results["Blood Pressure"]["status"] == "not_observable"
    assert results["Blood Pressure"]["score"] is None


def test_2_weak_and_no_evidence_abstains():
    # Weak evidence
    conv_weak = "I had a difficult day at work."
    facets = [{"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True}]
    res_weak = _heuristic_offline_score_batch(conv_weak, facets)
    assert res_weak[0]["status"] == "insufficient_evidence"
    assert res_weak[0]["score"] is None

    # No evidence
    conv_none = "Today was a normal day. I had cereal for breakfast."
    res_none = _heuristic_offline_score_batch(conv_none, facets)
    assert res_none[0]["status"] == "insufficient_evidence"
    assert res_none[0]["score"] is None


def test_3_hallucination_and_taxonomy_guardrails():
    conv = "I have been feeling exhausted and low on energy because of the cloudy weather."
    facets = [
        {"normalized_facet": "Diabetes", "facet_type": "medical_health", "conversation_observable": False},
        {"normalized_facet": "Serotonin Transporter Availability", "facet_type": "medical_health", "conversation_observable": False},
        {"normalized_facet": "Commute Time/Day", "facet_type": "external_evidence", "conversation_observable": False},
        {"normalized_facet": "Caffeine Intake (mg/day)", "facet_type": "external_evidence", "conversation_observable": False},
        {"normalized_facet": "Nationality", "facet_type": "biographical", "conversation_observable": False},
        {"normalized_facet": "Numerical Reasoning Subcomponents:", "facet_type": "malformed_header", "conversation_observable": False},
    ]
    results = {r["facet"]: r for r in _heuristic_offline_score_batch(conv, facets)}

    assert results["Diabetes"]["status"] == "not_observable"
    assert results["Serotonin Transporter Availability"]["status"] == "not_observable"
    assert results["Commute Time/Day"]["status"] == "not_observable"
    assert results["Caffeine Intake (mg/day)"]["status"] == "not_observable"
    assert results["Nationality"]["status"] == "insufficient_evidence"
    assert results["Numerical Reasoning Subcomponents:"]["status"] == "unsuitable"


def test_4_speaker_attribution_third_party_subject():
    """Third party attributes must NOT be scored on candidate."""
    conv = "My brother is incredibly hardworking and works twelve hours every day."
    facets = [
        {"normalized_facet": "Hardworking", "facet_type": "conversation_observable", "conversation_observable": True},
        {"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True},
    ]
    results = {r["facet"]: r for r in _heuristic_offline_score_batch(conv, facets)}

    assert results["Hardworking"]["status"] == "insufficient_evidence"
    assert results["Hardworking"]["score"] is None
    assert "third party" in results["Hardworking"]["reason"].lower() or "brother" in results["Hardworking"]["reason"].lower()


def test_5_speaker_attribution_quoted_speech():
    """Quoted third party statements do not reflect speaker's own traits."""
    conv = "My manager walked in and yelled, 'You are all completely incompetent!' I just kept my voice calm."
    facets = [
        {"normalized_facet": "Hostility", "facet_type": "conversation_observable", "conversation_observable": True},
        {"normalized_facet": "Managing Emotions", "facet_type": "conversation_observable", "conversation_observable": True},
    ]
    results = {r["facet"]: r for r in _heuristic_offline_score_batch(conv, facets)}

    assert results["Hostility"]["status"] == "scored"
    assert results["Hostility"]["score"] == 1  # Speaker remained calm
    assert results["Managing Emotions"]["status"] == "scored"
    assert results["Managing Emotions"]["score"] == 5


def test_6_sarcasm_detection():
    conv = "Oh, wonderful! Another unexpected production outage. Truly the highlight of my week."
    facets = [
        {"normalized_facet": "Happiness", "facet_type": "conversation_observable", "conversation_observable": True},
        {"normalized_facet": "Discontentment", "facet_type": "conversation_observable", "conversation_observable": True},
    ]
    results = {r["facet"]: r for r in _heuristic_offline_score_batch(conv, facets)}

    assert results["Happiness"]["status"] == "scored"
    assert results["Happiness"]["score"] == 1
    assert results["Discontentment"]["status"] == "scored"
    assert results["Discontentment"]["score"] == 5


def test_7_collaboration_and_teamwork():
    conv = "We need to pair-program on the backend API and support each other through the release."
    facets = [
        {"normalized_facet": "Cooperation", "facet_type": "conversation_observable", "conversation_observable": True},
        {"normalized_facet": "Collaboration", "facet_type": "conversation_observable", "conversation_observable": True},
    ]
    results = {r["facet"]: r for r in _heuristic_offline_score_batch(conv, facets)}

    assert results["Cooperation"]["status"] == "scored"
    assert results["Cooperation"]["score"] == 5
    assert results["Collaboration"]["status"] == "scored"
    assert results["Collaboration"]["score"] == 5


def test_8_deadline_ontime_and_missed():
    # On time
    conv_ontime = "I submitted the project ahead of schedule before the deadline."
    facets = [{"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True}]
    res_ontime = _heuristic_offline_score_batch(conv_ontime, facets)
    assert res_ontime[0]["status"] == "scored"
    assert res_ontime[0]["score"] == 5

    # Missed
    conv_missed = "I missed the deadline and submitted it on Wednesday."
    res_missed = _heuristic_offline_score_batch(conv_missed, facets)
    assert res_missed[0]["status"] == "scored"
    assert res_missed[0]["score"] == 1


def test_9_brevity_evaluation():
    conv = "Ok. Sounds fine."
    facets = [{"normalized_facet": "Brevity", "facet_type": "conversation_observable", "conversation_observable": True}]
    res = _heuristic_offline_score_batch(conv, facets)
    assert res[0]["status"] == "scored"
    assert res[0]["score"] == 5


def test_10_negative_perseverance_surrender():
    conv = "I failed the test and just decided not to try again — there was no point."
    facets = [{"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True}]
    res = _heuristic_offline_score_batch(conv, facets)
    assert res[0]["status"] == "scored"
    assert res[0]["score"] == 1
