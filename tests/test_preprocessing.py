"""
Tests for preprocessing, taxonomy classification, and normalization.
"""

import pytest
from src.preprocessing import normalize_facet, detect_duplicates, load_raw_facets
from src.taxonomy import classify_facet, _is_header_like


def test_normalization_whitespace():
    assert normalize_facet("  Risk Taking  ") == "Risk Taking"
    assert normalize_facet("Multiple   Spaces   Inside") == "Multiple Spaces Inside"


def test_normalization_trailing_colon():
    assert normalize_facet("Democratic Leadership:") == "Democratic Leadership"
    assert normalize_facet("HonestyHumility:") == "Honesty Humility"


def test_normalization_numbered_prefix():
    assert normalize_facet("800. Sufi practice: retreat") == "Sufi Practice: Retreat"
    assert normalize_facet("754. I Ching hexagram 36 resonance level") == "I Ching Hexagram 36 Resonance Level"


def test_normalization_camelcase():
    assert normalize_facet("SelfEsteem") == "Self Esteem"
    assert normalize_facet("SelfDirectedness") == "Self Directedness"


def test_taxonomy_header_detection():
    assert _is_header_like("Numerical Reasoning Subcomponents:") is True
    assert _is_header_like("HEXACO Personality Inventory Facets:") is True
    assert _is_header_like("Leadership Styles and Behaviors:") is True

    res = classify_facet("Numerical Reasoning Subcomponents:", "Numerical Reasoning Subcomponents")
    assert res.facet_type == "malformed_header"
    assert res.conversation_observable is False


def test_taxonomy_medical_classification():
    med_facets = [
        ("FSH level", "Fsh Level"),
        ("Basophil count", "Basophil Count"),
        ("Serotonin transporter availability", "Serotonin Transporter Availability"),
        ("Sleep-disorder diagnosis", "Sleep-Disorder Diagnosis"),
        ("Polygenic risk: cardiovascular disease", "Polygenic Risk: Cardiovascular Disease"),
    ]
    for raw, norm in med_facets:
        res = classify_facet(raw, norm)
        assert res.facet_type == "medical_health"
        assert res.conversation_observable is False
        assert res.sensitivity == "critical"
        assert res.abstention_reason is not None


def test_taxonomy_external_measurable():
    ext_facets = [
        ("Pilgrimage participation count", "Pilgrimage Participation Count"),
        ("Commute time/day", "Commute Time/Day"),
        ("Public-transport km/week", "Public-Transport Km/Week"),
        ("Dance rehearsal hours/week", "Dance Rehearsal Hours/Week"),
    ]
    for raw, norm in ext_facets:
        res = classify_facet(raw, norm)
        assert res.facet_type == "external_evidence"
        assert res.conversation_observable is False


def test_taxonomy_conversation_observable():
    obs_facets = [
        ("Persistence", "Persistence"),
        ("Happiness", "Happiness"),
        ("Hostility", "Hostility"),
        ("Cooperation", "Cooperation"),
        ("Managing emotions", "Managing Emotions"),
        ("Brevity", "Brevity"),
    ]
    for raw, norm in obs_facets:
        res = classify_facet(raw, norm)
        assert res.facet_type == "conversation_observable"
        assert res.conversation_observable is True


def test_duplicate_detection():
    pairs = [
        ("Risk Taking", "Risk Taking"),
        ("risk-taking", "Risk Taking"),
        ("Unique Facet", "Unique Facet"),
    ]
    dupes = detect_duplicates(pairs)
    assert "Risk Taking" in dupes
    assert len(dupes["Risk Taking"]) == 2
