"""
Tests for retrieval pipeline and taxonomy filtering.
"""

import pytest
import pandas as pd
from src.retrieval import get_scoreable_facets, EXCLUDED_TYPES


def test_scoreable_facets_filters_medical_and_headers():
    data = [
        {"raw_facet": "Persistence", "normalized_facet": "Persistence", "facet_type": "conversation_observable", "conversation_observable": True},
        {"raw_facet": "FSH level", "normalized_facet": "Fsh Level", "facet_type": "medical_health", "conversation_observable": False},
        {"raw_facet": "Numerical Reasoning Subcomponents:", "normalized_facet": "Numerical Reasoning Subcomponents", "facet_type": "malformed_header", "conversation_observable": False},
        {"raw_facet": "Happiness", "normalized_facet": "Happiness", "facet_type": "conversation_observable", "conversation_observable": True},
    ]
    df = pd.DataFrame(data)
    filtered = get_scoreable_facets(df)

    assert len(filtered) == 2
    assert "Persistence" in filtered["normalized_facet"].values
    assert "Happiness" in filtered["normalized_facet"].values
    assert "Fsh Level" not in filtered["normalized_facet"].values
    assert "Numerical Reasoning Subcomponents" not in filtered["normalized_facet"].values


def test_excluded_types_contains_expected():
    assert "malformed_header" in EXCLUDED_TYPES
    assert "medical_health" in EXCLUDED_TYPES
