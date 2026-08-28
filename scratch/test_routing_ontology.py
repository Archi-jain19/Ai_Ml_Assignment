import sys
import re
import math
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval import load_enriched_facets, get_scoreable_facets
from src.embeddings import embed_texts

df = load_enriched_facets()
scoreable = get_scoreable_facets(df)

# ── 1. DOMAIN & TOPIC ONTOLOGY ──────────────────────────────────────────────
# Maps broad semantic domains to keyword signals and facet classifiers.
# Generalizes across infinite phrasings without hardcoding individual sentences.

DOMAINS = {
    "work_effort": {
        "signals": [r"\b(work\w*|job|shift|overtime|office|project|task|deadlines?|stayed\s+late|long\s+hours|eight\s+hours|twelve\s+hours|labor|diligence|industrious\w*|career|colleague\w*|boss|manager|client|deliver\w*|complet\w*)\b"],
        "facet_patterns": [r"\b(work\w*|hardworking|job|career|labor|industrious|diligence|peer-collaboration|delegat\w*)\b"],
    },
    "persistence_adversity": {
        "signals": [r"\b(fail\w*|rejection|reject\w*|setback\w*|difficult\w*|adversity|obstacle\w*|gave\s+up|give\s+up|giving\s+up|persist\w*|persever\w*|grit|tenacity|tri\w*\s+again|appl\w*\s+again|struggl\w*|surrender\w*|never\s+quit)\b"],
        "facet_patterns": [r"\b(persever\w*|persist\w*|grit|resilience|tenacity|doggedness|striving)\b"],
    },
    "learning_growth": {
        "signals": [r"\b(learn\w*|studi\w*|exam\w*|test\w*|interview\w*|practice\w*|prepar\w*|feedback|improve\w*|weak\w*|presentation\w*|professor|teach\w*|course|grade|coach\w*|skill\w*)\b"],
        "facet_patterns": [r"\b(learning|self-improvement|growth|education|study|skill|achievement|intellect)\b"],
    },
    "deadlines_timeliness": {
        "signals": [r"\b(deadline\w*|due\s+date|due|submit\w*|on\s+time|ahead\s+of\s+schedule|late|punctual\w*|schedule\w*|turn\w*\s+in|deliver\w*\s+by)\b"],
        "facet_patterns": [r"\b(deadline\w*|timeliness|punctuality|submission)\b"],
    },
    "technical_problem_solving": {
        "signals": [r"\b(code|coding|debug\w*|bug\w*|error\s+logs?|race\s+condition|memory\s+leak|bottleneck|server|algorithm|software|tech\w*|system)\b"],
        "facet_patterns": [r"\b(troubleshooting|technical|data\s+analysis|debugging|algorithm)\b"],
    },
    "emotion_patience": {
        "signals": [r"\b(ang\w*|calm|yell\w*|scream\w*|frustrat\w*|patient\w*|composure|temper|rage|screaming|provok\w*|emotion\w*)\b"],
        "facet_patterns": [r"\b(emotion|patience|anger|hostility|calm|composure|aggression)\b"],
    },
    "community_volunteering": {
        "signals": [r"\b(volunteer\w*|food\s+bank|charity|shelter|civic|non-profit|community\s+service|donate|donating)\b"],
        "facet_patterns": [r"\b(volunteer|community|charity|civic)\b"],
    },
    "yoga_physical_practice": {
        "signals": [r"\b(yoga|asanas?|pranayama|vinyasa|stretching\s+routine|yoga\s+class)\b"],
        "facet_patterns": [r"\b(yoga)\b"],
    },
    "spiritual_religious": {
        "signals": [r"\b(sufi|dhikr|zohar|kabbalah|quran|khatam|bhagavad|gita|vrata|kirtan|sukkot|lulav|archon|reiki|meditation|mantra|prayer|synagogue|mosque|temple|scripture)\b"],
        "facet_patterns": [r"\b(spiritual|sufi|dhikr|zohar|kabbalah|quran|gita|vrata|kirtan|sukkot|lulav|archon|reiki|meditation|mantra|jewish|islamic|hindu|sikh|buddhist|gnostic)\b"],
    },
    "caffeine_beverage": {
        "signals": [r"\b(coffee|espresso|caffeine|tea|latte|cappuccino|energy\s+drinks?)\b"],
        "facet_patterns": [r"\b(caffeine)\b"],
    },
    "transit_commute": {
        "signals": [r"\b(commute|commuting|subway|bus|metro|public\s+transit|train|transit|driving\s+to\s+work)\b"],
        "facet_patterns": [r"\b(commute|transport)\b"],
    },
}

def extract_conversational_intents(text: str) -> set[str]:
    """Detect broad topic domains present in the conversation."""
    text_low = text.lower()
    intents = set()
    for domain, config in DOMAINS.items():
        for pat in config["signals"]:
            if re.search(pat, text_low):
                intents.add(domain)
                break
    return intents

def classify_facet_domains(facet_name: str, defn: str = "") -> set[str]:
    """Determine which domains a facet belongs to based on name and definition."""
    combined = f"{facet_name} {defn}".lower()
    facet_domains = set()
    for domain, config in DOMAINS.items():
        for pat in config["facet_patterns"]:
            if re.search(pat, combined):
                facet_domains.add(domain)
                break
    return facet_domains

# Pre-tag all scoreable facets with their domain sets
scoreable_domains = [
    classify_facet_domains(row["normalized_facet"], str(row.get("scoring_definition", "")))
    for _, row in scoreable.iterrows()
]

# Specific niche domains that require explicit conversational signals
NICHE_DOMAINS = {
    "yoga_physical_practice",
    "spiritual_religious",
    "caffeine_beverage",
    "transit_commute",
}

print(f"Tagged {len(scoreable_domains)} scoreable facets across {len(DOMAINS)} domain categories.")
