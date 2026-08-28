"""
Taxonomy classifier for facets.

Classifies each facet into a category using deterministic rules based on
keyword patterns and structural heuristics. This is NOT a perfect classifier —
the CSV is intentionally heterogeneous and some edge cases will be misclassified.

Taxonomy Categories
-------------------
A. conversation_observable
   Things that can reasonably be inferred from conversational evidence:
   personality traits, emotional states, behavioral tendencies, values,
   attitudes, communication styles.

B. external_evidence
   Requires quantitative data from outside the conversation:
   counts, metrics, frequencies, durations, percentages, scores from
   standardised instruments.

C. medical_health
   Lab values, diagnoses, biological markers, genetic data.
   Must NOT be inferred from casual conversation.

D. biographical
   Nationality, childhood experiences, personal history, demographics.
   Requires factual external information.

E. ambiguous
   Might be partially observable in conversation but cannot safely be
   inferred from weak evidence. Includes many psychological constructs
   and clinical scales.

F. malformed_header
   Section headers, category labels, or other non-facet entries
   (e.g. "Numerical Reasoning Subcomponents:").

Classification Approach
-----------------------
The classifier uses a priority-ordered rule cascade:
1. Structural rules first (header-like entries).
2. Domain-specific keyword matching (medical, external-measurable).
3. Prefix-based patterns (numbered entries, "Psychological construct:" etc).
4. Remaining entries examined for observability signals.

Limitations
-----------
- Some psychological constructs COULD be partially observed in conversation
  but are classified as 'ambiguous' to avoid false confidence.
- Religious/spiritual practice metrics are classified as 'external_evidence'
  because they measure quantities not observable in a conversation.
- The boundary between 'conversation_observable' and 'ambiguous' is inherently
  subjective. We err on the side of caution (classify as ambiguous when unsure).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaxonomyResult:
    """Result of classifying a single facet."""
    facet_type: str          # One of the taxonomy categories
    conversation_observable: bool
    sensitivity: str         # low, medium, high, critical
    abstention_reason: Optional[str]  # Why this facet should abstain, or None
    data_quality_flag: Optional[str]  # Quality issue found, or None


# ── Header-like patterns ──────────────────────────────────────────────────
_HEADER_SUFFIXES = [
    "Subcomponents:", "Facets:", "Types:", "Styles:", "Components:",
    "Parameters:", "Behaviors:", "Tendencies:", "Drivers:", "End Points:",
    "Themes:", "Inventory Facets:",
]

# ── Medical / biological keywords ─────────────────────────────────────────
_MEDICAL_KEYWORDS = [
    "fsh level", "basophil", "serotonin", "parathyroid", "chromatin",
    "polygenic risk", "metabolic rate", "sleep apnea", "immune-response",
    "caffeine sensitivity gene", "sleep-disorder diagnosis",
    "macronutrient ratio", "drug-use history", "chronic pain",
    "hypomania", "hysteria (hy)", "psychoticism",
    "depression (dep)", "depression symptoms",
]

# ── External measurable patterns ──────────────────────────────────────────
_EXTERNAL_PATTERNS = [
    r"\bcount\b", r"\bkm/week\b", r"\bmg/day\b", r"\bhours?/week\b",
    r"\b/year\b", r"\b/day\b", r"\bvisits/year\b", r"\bmonths\b",
    r"\btime/day\b", r"\byears\b", r"\bscore\b", r"\blevel\b",
    r"\bindex\b", r"\brate\b", r"\b%\b", r"\bfrequency\b",
    r"\bstamps\b", r"\bsubscri", r"\brendorsement",
    r"\bhours\b", r"\bsessions\b", r"\bcycles\b", r"\bdays\b",
    r"\bverses\b", r"\brepetitions\b",
]

# ── Biographical keywords ────────────────────────────────────────────────
_BIOGRAPHICAL_KEYWORDS = [
    "nationality", "childhood experiences", "attachment style",
    "intelligence quotient", "iq", "learning style",
]

# ── I Ching / esoteric patterns ──────────────────────────────────────────
_ESOTERIC_PATTERNS = [
    r"i ching", r"hexagram", r"aura-color", r"astrology",
    r"rising sign",
]

# ── Numbered prefix pattern (e.g. "800. Sufi practice:") ─────────────────
_NUMBERED_PREFIX = re.compile(r"^\d+\.\s+")

# ── Psychological construct prefix ───────────────────────────────────────
_PSYCH_PREFIX = "psychological construct:"
_SOCIAL_COG_PREFIX = "social-cognition variable:"
_WELL_BEING_PREFIX = "well-being component:"
_EMOTIONAL_INTEL_PREFIX = "emotional-intelligence measure:"
_DEFENSE_PREFIX = "defense-mechanism tendency:"
_CHAR_STRENGTH_PREFIX = "character strength:"
_BIG_FIVE_PREFIX = "big five facet"
_HEXACO_PREFIX = "hexaco"
_MINDFULNESS_PREFIX = "mindfulness facet:"

# ── Conversation-observable signals ──────────────────────────────────────
# Personality traits, emotions, communication styles that ARE observable
_OBSERVABLE_SIGNALS = [
    "ness$", "ity$", "ism$", "ment$", "ance$", "ence$",
    "enthusiasm", "compassion", "bravery", "persistence", "creativity",
    "assertive", "openness", "warmth", "hostility", "patience",
    "impulsiv", "curiosity", "humor", "optimis", "pessimis",
    "introvert", "extrovert", "agreeable", "conscientious",
    "cooperat", "collaborat", "adapt", "flexib", "resilien",
    "motivat", "initiat", "leadership", "communicat",
]


def classify_facet(raw_facet: str, normalized_facet: str) -> TaxonomyResult:
    """
    Classify a single facet into the taxonomy.

    Parameters
    ----------
    raw_facet : str
        The original facet value from the CSV.
    normalized_facet : str
        The cleaned/normalized facet name.

    Returns
    -------
    TaxonomyResult
    """
    low = normalized_facet.lower()
    raw_low = raw_facet.lower().strip()
    quality_flag = _detect_quality_issues(raw_facet)

    # ── Rule 1: Header-like entries ──────────────────────────────────────
    if _is_header_like(raw_facet):
        return TaxonomyResult(
            facet_type="malformed_header",
            conversation_observable=False,
            sensitivity="low",
            abstention_reason="Entry appears to be a section header, not a scoreable facet.",
            data_quality_flag=quality_flag or "header_like_entry",
        )

    # ── Rule 2: Medical / biological ─────────────────────────────────────
    if _matches_medical(low):
        return TaxonomyResult(
            facet_type="medical_health",
            conversation_observable=False,
            sensitivity="critical",
            abstention_reason="Medical/biological data cannot be inferred from conversation.",
            data_quality_flag=quality_flag,
        )

    # ── Rule 3: Esoteric / I Ching ──────────────────────────────────────
    if _matches_esoteric(low):
        return TaxonomyResult(
            facet_type="external_evidence",
            conversation_observable=False,
            sensitivity="medium",
            abstention_reason="Esoteric/spiritual metric requires external measurement.",
            data_quality_flag=quality_flag,
        )

    # ── Rule 4: Numbered religious/spiritual practice metrics ────────────
    if _NUMBERED_PREFIX.match(raw_facet.strip()):
        return TaxonomyResult(
            facet_type="external_evidence",
            conversation_observable=False,
            sensitivity="medium",
            abstention_reason="Quantitative spiritual/religious metric requires external data.",
            data_quality_flag=quality_flag or "numbered_prefix",
        )

    # ── Rule 5: Biographical ────────────────────────────────────────────
    if _matches_biographical(low):
        return TaxonomyResult(
            facet_type="biographical",
            conversation_observable=False,
            sensitivity="high",
            abstention_reason="Biographical fact requires external information.",
            data_quality_flag=quality_flag,
        )

    # ── Rule 6: External measurables ────────────────────────────────────
    if _matches_external(low):
        # Some external-pattern words overlap with observable traits
        # Check if it's clearly quantitative
        if _is_clearly_quantitative(low):
            return TaxonomyResult(
                facet_type="external_evidence",
                conversation_observable=False,
                sensitivity="medium",
                abstention_reason="Quantitative metric requires external measurement data.",
                data_quality_flag=quality_flag,
            )

    # ── Rule 7: Psychological constructs (prefix-based) ─────────────────
    if low.startswith(_PSYCH_PREFIX) or low.startswith(_SOCIAL_COG_PREFIX):
        return TaxonomyResult(
            facet_type="ambiguous",
            conversation_observable=False,
            sensitivity="high",
            abstention_reason="Standardised psychological measure requires validated instrument.",
            data_quality_flag=quality_flag,
        )

    # ── Rule 8: Clinical scales (HEXACO, Big Five named facets, etc.) ────
    if any(low.startswith(p) for p in [_BIG_FIVE_PREFIX, _HEXACO_PREFIX]):
        return TaxonomyResult(
            facet_type="ambiguous",
            conversation_observable=False,
            sensitivity="medium",
            abstention_reason="Named psychometric facet requires validated instrument for precise scoring.",
            data_quality_flag=quality_flag,
        )

    # ── Rule 9: Clinical depression / mental health specifics ────────────
    if "depression" in low and ("feelings" in low or "sadness" in low):
        return TaxonomyResult(
            facet_type="medical_health",
            conversation_observable=False,
            sensitivity="critical",
            abstention_reason="Clinical depression assessment requires professional diagnosis.",
            data_quality_flag=quality_flag,
        )

    # ── Rule 10: Defense mechanisms, well-being components ───────────────
    if any(low.startswith(p) for p in [_DEFENSE_PREFIX, _WELL_BEING_PREFIX,
                                        _EMOTIONAL_INTEL_PREFIX]):
        return TaxonomyResult(
            facet_type="ambiguous",
            conversation_observable=False,
            sensitivity="medium",
            abstention_reason="Composite psychological measure; partial signals may appear in conversation but safe scoring requires validated tools.",
            data_quality_flag=quality_flag,
        )

    # ── Rule 11: Remaining — check for conversation-observable signals ───
    if _has_observable_signals(low):
        sensitivity = "high" if _is_sensitive_trait(low) else "low"
        return TaxonomyResult(
            facet_type="conversation_observable",
            conversation_observable=True,
            sensitivity=sensitivity,
            abstention_reason=None,
            data_quality_flag=quality_flag,
        )

    # ── Default: conversation_observable with medium confidence ──────────
    # Most remaining facets are personality/behavioral traits
    sensitivity = "high" if _is_sensitive_trait(low) else "low"
    return TaxonomyResult(
        facet_type="conversation_observable",
        conversation_observable=True,
        sensitivity=sensitivity,
        abstention_reason=None,
        data_quality_flag=quality_flag,
    )


def _is_header_like(raw: str) -> bool:
    """Check if the entry is a section header."""
    stripped = raw.strip()
    for suffix in _HEADER_SUFFIXES:
        if stripped.endswith(suffix):
            return True
    # Also catch entries like "Enneagram Personality Types:"
    if stripped.endswith(":") and any(w in stripped for w in [
        "Subcomponent", "Component", "Parameter", "Behavior",
        "Tendency", "Driver", "Style", "End Point", "Type",
        "Facet", "Inventory", "Theme",
    ]):
        return True
    return False


def _matches_medical(low: str) -> bool:
    return any(kw in low for kw in _MEDICAL_KEYWORDS)


def _matches_esoteric(low: str) -> bool:
    return any(re.search(p, low) for p in _ESOTERIC_PATTERNS)


def _matches_biographical(low: str) -> bool:
    return any(kw in low for kw in _BIOGRAPHICAL_KEYWORDS)


def _matches_external(low: str) -> bool:
    return any(re.search(p, low) for p in _EXTERNAL_PATTERNS)


def _is_clearly_quantitative(low: str) -> bool:
    """
    Distinguish between truly quantitative facets and traits that
    happen to contain words like 'level' or 'score'.
    """
    quant_signals = [
        r"\bcount\b", r"\bkm/", r"\bmg/", r"\bhours?/", r"/year",
        r"/day", r"/week", r"\bstamps\b", r"\bsubscri", r"\bcycles\b",
        r"\brendorsement", r"\bsessions\b", r"\brepetitions\b",
        r"\bverses\b", r"\bvisits/", r"\btime/day\b",
        r"\bpresence\b", r"\bdiagnosis\b",
        r"ratio:", r"age$", r"\bgene\b",
        # Patterns like "Pilgrimage participation count"
        r"participation count", r"subscriber count",
        r"endorsements count", r"skill-endorsements",
    ]
    # Trait-like patterns that contain 'level' but are still observable
    trait_exceptions = [
        "comfort level", "confidence", "contentment level",
        "self-efficacy", "stress recovery", "attitude",
    ]
    if any(te in low for te in trait_exceptions):
        return False
    return any(re.search(p, low) for p in quant_signals)


def _has_observable_signals(low: str) -> bool:
    """Check if the facet name contains signals of conversation-observability."""
    for pattern in _OBSERVABLE_SIGNALS:
        if re.search(pattern, low):
            return True
    return False


def _is_sensitive_trait(low: str) -> bool:
    """Flag facets touching sensitive personal areas."""
    sensitive_words = [
        "sexual", "kink", "drug", "violence", "abuse",
        "suicide", "self-harm", "eating disorder", "addiction",
        "hatred", "racist", "extremis",
    ]
    return any(sw in low for sw in sensitive_words)


def _detect_quality_issues(raw: str) -> Optional[str]:
    """Detect data quality issues in the raw facet value."""
    issues = []
    stripped = raw.strip()

    if raw != stripped:
        issues.append("whitespace")
    if stripped.endswith(":") and not _is_header_like(raw):
        issues.append("trailing_colon")
    if _NUMBERED_PREFIX.match(stripped):
        issues.append("numbered_prefix")
    if re.search(r"[a-z][A-Z]", stripped) and " " not in stripped.replace(":", ""):
        issues.append("camelcase_compound")

    return "; ".join(issues) if issues else None
