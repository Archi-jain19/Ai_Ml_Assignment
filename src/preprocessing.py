"""
Preprocessing pipeline for facet data.

Takes the raw 'Facets Assignment.csv' and produces an enriched CSV with:
- raw_facet: original value, untouched
- normalized_facet: cleaned/normalized name
- facet_type: taxonomy category
- conversation_observable: bool
- sensitivity: low/medium/high/critical
- scoring_definition: what each score level means for this facet
- score_1_anchor through score_5_anchor: per-level descriptions
- abstention_reason: why this facet cannot be scored, or empty
- data_quality_flag: quality issues found, or empty

The raw CSV is NEVER modified.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import RAW_CSV_PATH, ENRICHED_CSV_PATH
from src.taxonomy import classify_facet, TaxonomyResult

logger = logging.getLogger(__name__)


# ── Normalization ─────────────────────────────────────────────────────────

def normalize_facet(raw: str) -> str:
    """
    Deterministic normalization of a facet name.

    Steps:
    1. Strip leading/trailing whitespace
    2. Remove numbered prefixes (e.g. "800. ")
    3. Remove trailing colons (unless part of a meaningful label like "fat:")
    4. Expand CamelCase compounds (SelfEsteem → Self Esteem)
    5. Collapse repeated whitespace
    6. Title case

    The original semantic meaning is preserved.
    """
    s = raw.strip()

    # Remove numbered prefix (e.g. "800. Sufi practice: ...")
    s = re.sub(r"^\d+\.\s+", "", s)

    # Remove trailing colon if it's just formatting noise
    if s.endswith(":"):
        s = s[:-1].strip()

    # Expand CamelCase: "SelfEsteem" → "Self Esteem"
    # But don't break things like "HEXACO" or "IQ"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)

    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Title case for consistency
    s = _smart_title_case(s)

    return s


def _smart_title_case(s: str) -> str:
    """
    Title case that preserves acronyms and handles special cases.
    """
    words = s.split()
    result = []
    # Words to keep lowercase (prepositions, etc.) unless first word
    lower_words = {"a", "an", "the", "in", "on", "at", "to", "for",
                   "of", "and", "or", "vs", "vs."}
    for i, word in enumerate(words):
        # Keep fully uppercase short words (likely acronyms)
        if word.isupper() and len(word) <= 5:
            result.append(word)
        elif i > 0 and word.lower() in lower_words:
            result.append(word.lower())
        else:
            result.append(word.capitalize())
    return " ".join(result)


# ── Scoring Definitions ──────────────────────────────────────────────────

def generate_scoring_definition(normalized: str, facet_type: str) -> dict:
    """
    Generate a scoring definition and 5 anchor descriptions for a facet.

    For conversation-observable facets, provides meaningful anchors.
    For non-observable facets, returns empty anchors since they should abstain.
    """
    if facet_type != "conversation_observable":
        return {
            "scoring_definition": f"Not directly scoreable from conversation ({facet_type}).",
            "score_1_anchor": "",
            "score_2_anchor": "",
            "score_3_anchor": "",
            "score_4_anchor": "",
            "score_5_anchor": "",
        }

    # Generate generic but meaningful anchors based on the facet name
    low = normalized.lower()

    # Detect if this is a negative trait
    negative_traits = [
        "hostility", "dishonesty", "hatefulness", "slothfulness",
        "impracticalness", "inefficiency", "disagreeableness",
        "discontentment", "disrespect", "harmfulness", "impudence",
        "immaturity", "coarseness", "cantankerousness", "clumsiness",
        "servility", "rebelliousness", "inattentiveness", "moroseness",
        "withdrawnness", "suspicion", "irritability",
    ]
    is_negative = any(nt in low for nt in negative_traits)

    if is_negative:
        return {
            "scoring_definition": (
                f"Measures the degree of {normalized.lower()} demonstrated "
                f"in the conversation, based on observable language, tone, "
                f"and behavioral descriptions."
            ),
            "score_1_anchor": f"No signs of {low} in the conversation.",
            "score_2_anchor": f"Minimal/isolated signs of {low}.",
            "score_3_anchor": f"Moderate {low} apparent in several statements.",
            "score_4_anchor": f"Clear and repeated {low} throughout.",
            "score_5_anchor": f"Pervasive {low} dominating the conversation.",
        }

    return {
        "scoring_definition": (
            f"Measures the degree of {normalized.lower()} demonstrated "
            f"in the conversation, based on observable language, tone, "
            f"and behavioral descriptions."
        ),
        "score_1_anchor": f"Very low / no evidence of {low}.",
        "score_2_anchor": f"Slight evidence of {low}; brief or weak signals.",
        "score_3_anchor": f"Moderate {low}; clear but not dominant evidence.",
        "score_4_anchor": f"Strong {low}; multiple clear indicators.",
        "score_5_anchor": f"Very strong {low}; pervasive and compelling evidence.",
    }


# ── Duplicate Detection ──────────────────────────────────────────────────

def detect_duplicates(facets: list[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Detect potential duplicates after normalization.

    Returns a mapping of normalized_facet → list of raw_facet values
    for any normalized name that appears more than once.
    """
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for raw, norm in facets:
        groups[norm].append(raw)
    return {k: v for k, v in groups.items() if len(v) > 1}


# ── Main Pipeline ────────────────────────────────────────────────────────

def load_raw_facets(path: Optional[Path] = None) -> list[str]:
    """Load raw facet values from CSV, skipping header and empty rows."""
    path = path or RAW_CSV_PATH
    facets = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        logger.info(f"CSV header: {header}")
        for row in reader:
            val = row[0] if row else ""
            if val.strip():
                facets.append(val)
    logger.info(f"Loaded {len(facets)} raw facets from {path}")
    return facets


def preprocess_facets(
    raw_facets: list[str],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    raw_facets : list[str]
        Raw facet values from the CSV.
    output_path : Path, optional
        Where to save the enriched CSV. Defaults to ENRICHED_CSV_PATH.

    Returns
    -------
    pd.DataFrame
        Enriched facet dataframe.
    """
    output_path = output_path or ENRICHED_CSV_PATH
    records = []

    for raw in raw_facets:
        norm = normalize_facet(raw)
        taxonomy: TaxonomyResult = classify_facet(raw, norm)
        scoring = generate_scoring_definition(norm, taxonomy.facet_type)

        records.append({
            "raw_facet": raw,
            "normalized_facet": norm,
            "facet_type": taxonomy.facet_type,
            "conversation_observable": taxonomy.conversation_observable,
            "sensitivity": taxonomy.sensitivity,
            "scoring_definition": scoring["scoring_definition"],
            "score_1_anchor": scoring["score_1_anchor"],
            "score_2_anchor": scoring["score_2_anchor"],
            "score_3_anchor": scoring["score_3_anchor"],
            "score_4_anchor": scoring["score_4_anchor"],
            "score_5_anchor": scoring["score_5_anchor"],
            "abstention_reason": taxonomy.abstention_reason or "",
            "data_quality_flag": taxonomy.data_quality_flag or "",
        })

    df = pd.DataFrame(records)

    # Detect and log duplicates
    pairs = list(zip(df["raw_facet"], df["normalized_facet"]))
    dupes = detect_duplicates(pairs)
    if dupes:
        logger.warning(f"Found {len(dupes)} potential duplicate groups after normalization:")
        for norm, raws in dupes.items():
            logger.warning(f"  '{norm}' ← {raws}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved enriched CSV to {output_path} ({len(df)} rows)")

    # Log summary
    _log_audit_summary(df)

    return df


def _log_audit_summary(df: pd.DataFrame) -> None:
    """Log a summary of the facet audit."""
    logger.info("\n" + "=" * 60)
    logger.info("FACET AUDIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total facets: {len(df)}")
    logger.info(f"\nBy facet type:")
    for ft, count in df["facet_type"].value_counts().items():
        logger.info(f"  {ft}: {count}")
    logger.info(f"\nConversation observable: {df['conversation_observable'].sum()}")
    logger.info(f"Non-observable: {(~df['conversation_observable']).sum()}")
    logger.info(f"\nBy sensitivity:")
    for s, count in df["sensitivity"].value_counts().items():
        logger.info(f"  {s}: {count}")
    quality_issues = df[df["data_quality_flag"] != ""]
    logger.info(f"\nData quality issues: {len(quality_issues)}")
    if len(quality_issues) > 0:
        flags = quality_issues["data_quality_flag"].value_counts()
        for flag, count in flags.items():
            logger.info(f"  {flag}: {count}")
    logger.info("=" * 60)


def generate_audit_report(df: pd.DataFrame) -> str:
    """Generate a text audit report for documentation."""
    lines = []
    lines.append("# Facet Data Audit Report")
    lines.append(f"\nTotal facets: {len(df)}")
    lines.append(f"\n## Classification Breakdown\n")
    for ft, count in df["facet_type"].value_counts().items():
        pct = 100 * count / len(df)
        lines.append(f"- **{ft}**: {count} ({pct:.1f}%)")

    lines.append(f"\n## Observability\n")
    obs = df["conversation_observable"].sum()
    lines.append(f"- Conversation-observable: {obs} ({100*obs/len(df):.1f}%)")
    lines.append(f"- Non-observable: {len(df)-obs} ({100*(len(df)-obs)/len(df):.1f}%)")

    lines.append(f"\n## Sensitivity\n")
    for s, count in df["sensitivity"].value_counts().items():
        lines.append(f"- {s}: {count}")

    lines.append(f"\n## Data Quality Issues\n")
    quality_issues = df[df["data_quality_flag"] != ""]
    lines.append(f"Total entries with issues: {len(quality_issues)}")
    if len(quality_issues) > 0:
        for flag, count in quality_issues["data_quality_flag"].value_counts().items():
            lines.append(f"- {flag}: {count}")

    lines.append(f"\n## Examples by Category\n")
    for ft in df["facet_type"].unique():
        subset = df[df["facet_type"] == ft]
        lines.append(f"\n### {ft} (showing up to 5)")
        for _, row in subset.head(5).iterrows():
            lines.append(f"- `{row['raw_facet']}` -> `{row['normalized_facet']}`")

    return "\n".join(lines)
