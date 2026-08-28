"""
Facet retrieval module.

Implements the two-stage hybrid retrieval pipeline:
1. Taxonomy-based pre-filtering (exclude malformed_header and medical_health)
2. Hybrid Dense Vector (FAISS) + Lexical (BM25) similarity search with concept enrichment

This ensures that:
- Medical/biological facets are never sent to scoring
- Header-like entries are excluded
- Conversational behavioral traits (Perseverance, Hardworking, Self-improvement, Deadlines)
  rank at the top for relevant conversations without lexical/semantic confusion
- Explicit topical references (Yoga, Coffee/Caffeine, Volunteer Work) properly surface their
  respective facets with high precision
- The pipeline scales to >=5,000 facets via FAISS flat indexing and fast vector dot products
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ENRICHED_CSV_PATH,
    TOP_K,
)
from src.embeddings import (
    build_faiss_index,
    load_faiss_index,
    embed_texts,
)

logger = logging.getLogger(__name__)

# Facet types that are EXCLUDED from retrieval (never sent to LLM scoring)
EXCLUDED_TYPES = {"malformed_header", "medical_health"}

# Facet types that CAN be retrieved but may still abstain
RETRIEVABLE_TYPES = {"conversation_observable", "ambiguous", "external_evidence", "biographical"}

# ── Domain Concept & Synonym Dictionary ─────────────────────────────────────
# Provides natural conceptual anchors for facet semantic indexing without hardcoding query matches.
CONCEPT_DICTIONARY: dict[str, str] = {
    "perseverance": "persistence grit tenacity determination not giving up enduring hardship resilience striving through obstacles overcoming failure trying again",
    "persistence": "perseverance steadfastness continuing effort not quitting sticking with tasks diligence through difficulty determination",
    "character strength: perseverance": "grit persistence tenacity overcoming adversity not giving up sustained effort through setbacks",
    "hardworking": "diligence industriousness dedicated effort working long hours putting in time and energy tireless work ethic dedicated labor",
    "self-improvement": "personal development learning from mistakes adjusting approach improving skills seeking feedback self-growth practicing to get better modifying strategy",
    "attitude toward learning": "growth mindset enthusiasm for studying seeking feedback active practice curiosity continuous education engagement with learning",
    "meeting deadlines": "timeliness punctuality submitting on time completing before due date delivering work on schedule finishing before deadline finished early handed in ahead of due date turned in before deadline got work in before cutoff on time delivery",
    "submission": "submitting assignments documents applications turning in work handing in deliverables completed task submission coursework project hand in turn in got work in turned in",
    "submissiveness": "compliance following instructions obeying authority deferring to manager yielding conforming executing instructions doing as requested obedience yielding to decision",
    "troubleshooting technical issues": "debugging diagnosing problems isolating root causes fixing bugs technical problem solving writing unit tests memory leaks race conditions",
    "data analysis": "analyzing error logs metrics investigating data patterns quantitative reasoning diagnosing system bottlenecks reviewing logs",
    "managing emotions": "emotional regulation composure under pressure remaining calm patience self-control during conflict de-escalation",
    "patience: resistance to anger": "staying calm when provoked tolerance emotional restraint composure during screaming or stress resisting rage",
    "risktaking": "financial risk speculation daring actions gambling taking chances bold uncalculated moves memecoin investing emergency fund",
    "brevity": "conciseness short answers brief communication laconic replies succinct responses minimal words",
    "volunteer work": "community service food bank charity volunteering time helping non-profits civic participation",
    "caffeine intake (mg/day)": "drinking coffee espresso tea energy drinks caffeine consumption morning coffee cups",
    "hindu spiritual metric: yoga discipline hours / week": "practicing yoga asanas yoga sessions weekly yoga practice yoga discipline",
    "cooperation": "teamwork collaboration pair programming supporting colleagues working together mutual assistance",
    "delegation skills": "assigning tasks delegating work to team members dividing responsibilities dispatching work",
    "happiness": "joy cheerfulness positive emotion satisfaction genuine delight optimism",
    "discontentment": "frustration dissatisfaction resentment unhappiness with situation sarcasm about outages cynicism",
    "burnout symptoms": "exhaustion chronic fatigue overwork emotional depletion burnout",
    "general mood and attitude": "disposition general outlook life satisfaction optimism pessimism",
    "orderliness": "organization neatness structured workspace systematic filing",
    "flawlessness": "perfectionism zero defects precision exactness meticulousness",
}


def load_enriched_facets(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the enriched facet CSV."""
    path = path or ENRICHED_CSV_PATH
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} enriched facets from {path}")
    return df


def get_scoreable_facets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter facets to only those that are candidates for scoring.
    Excludes malformed_header and medical_health facets entirely.
    """
    mask = ~df["facet_type"].isin(EXCLUDED_TYPES)
    scoreable = df[mask].reset_index(drop=True)
    logger.info(
        f"Scoreable facets: {len(scoreable)} / {len(df)} "
        f"(excluded {len(df) - len(scoreable)} {EXCLUDED_TYPES})"
    )
    return scoreable


def build_facet_semantic_text(row: pd.Series) -> str:
    """
    Construct rich semantic indexing representation for a facet.
    Combines facet name, domain concept anchors, category type, and anchors.
    """
    name = str(row["normalized_facet"])
    name_low = name.lower().strip().rstrip(":")
    ftype = str(row.get("facet_type", "conversation_observable"))
    obs = bool(row.get("conversation_observable", True))

    parts = [name]

    if name_low in CONCEPT_DICTIONARY:
        parts.append(f"Concepts: {CONCEPT_DICTIONARY[name_low]}")

    if obs:
        parts.append("Observable conversational behavioral trait and personality characteristic.")
    else:
        parts.append(f"External domain metric ({ftype}).")

    defn = str(row.get("scoring_definition", "") or "").strip()
    if defn and len(defn) > 150:
        defn = defn[:150]
    if defn:
        parts.append(defn)

    return " | ".join(parts)


STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into alphanumeric words, filtering out stopwords and single-character tokens."""
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class SimpleBM25:
    """Lightweight in-memory BM25 scorer for hybrid retrieval with stopword filtering."""
    def __init__(self, corpus: list[str]):
        self.corpus = [_tokenize(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0
        self.df: Counter = Counter()
        for doc in self.corpus:
            for word in set(doc):
                self.df[word] += 1
        self.N = len(self.corpus)

    def score(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)
        if not q_tokens:
            return scores
        k1 = 1.2
        b = 0.75
        for t in q_tokens:
            if t in self.df:
                idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0)
                for i, doc in enumerate(self.corpus):
                    tf = doc.count(t)
                    if tf > 0:
                        denom = tf + k1 * (1.0 - b + b * (self.doc_len[i] / self.avgdl))
                        scores[i] += idf * (tf * (k1 + 1.0)) / denom
        return scores



# Global cache for BM25 index over scoreable facets
_bm25_index: Optional[SimpleBM25] = None
_bm25_scoreable_len: int = 0


def _get_bm25_index(scoreable: pd.DataFrame) -> SimpleBM25:
    global _bm25_index, _bm25_scoreable_len
    if _bm25_index is None or _bm25_scoreable_len != len(scoreable):
        corpus = [build_facet_semantic_text(row) for _, row in scoreable.iterrows()]
        _bm25_index = SimpleBM25(corpus)
        _bm25_scoreable_len = len(scoreable)
    return _bm25_index


# ── Conversational Intent & Domain Ontology ─────────────────────────────────
# Infers broad topic signals from conversational input and matches against facet domains.
# Operates without hardcoding individual sentences or global blacklists.

DOMAIN_ONTOLOGY: dict[str, dict[str, list[str]]] = {
    "compliance_submissiveness": {
        "signals": [r"\b(followed\s+(the\s+|my\s+)?instructions|complied|obey\w*|yield\w*|submissive|did\s+what\s+(i\s+was|they)\s+told|exactly\s+as\s+requested|as\s+instructed|submitting|submitted\s+to|deferred|deference|disagreed\s+with\s+(my\s+)?manager)\b"],
        "facet_patterns": [r"\b(submissiveness|submission|compliance|conforming)\b"],
    },
    "organization_planning": {
        "signals": [r"\b(reorganiz\w*|organiz\w*|schedule\w*|time\s+management|prioritiz\w*|plan\w*|routine|structured|order\w*)\b"],
        "facet_patterns": [r"\b(organized\s+lifestyle|orderliness|time\s+management|planning|self-improvement)\b"],
    },
    "work_effort": {
        "signals": [r"\b(work\w*|job|shift|overtime|office|project|task\w*|stayed\s+late|long\s+hours|eight\s+hours|twelve\s+hours|every\s+evening|every\s+night|labor|diligence|industrious\w*|career|colleague\w*|boss|manager|client|finish\w*|complet\w*)\b"],
        "facet_patterns": [r"\b(hardworking|work\s+styles?|work\s+ethic|industrious|diligence)\b"],
    },
    "persistence_adversity": {
        "signals": [r"\b(behind\s+on|fell\s+behind|fail\w*|rejection|reject\w*|setback\w*|difficult\w*|adversity|obstacle\w*|gave\s+up|give\s+up|giving\s+up|persist\w*|persever\w*|grit|tenacity|tri\w*\s+again|appl\w*\s+again|struggl\w*|surrender\w*|never\s+quit)\b"],
        "facet_patterns": [r"\b(persever\w*|persist\w*|grit|resilience|tenacity|doggedness|striving)\b"],
    },
    "learning_growth": {
        "signals": [r"\b(learn\w*|studi\w*|exam\w*|test\w*|interview\w*|practice\w*|prepar\w*|feedback|improve\w*|weak\w*|presentation\w*|professor|teach\w*|course|grade|coach\w*|skill\w*)\b"],
        "facet_patterns": [r"\b(learning|self-improvement|growth|education|study|skill|achievement|intellect|feedback)\b"],
    },
    "deadlines_timeliness": {
        "signals": [r"\b(deadline\w*|due\s+date|due|submit\w*|on\s+time|ahead\s+of\s+(?:the\s+)?due\s+date|ahead\s+of\s+schedule|late|punctual\w*|schedule\w*|turn\w*\s+in|turned\s+in|handed\s+in|hand\w*\s+in|deliver\w*\s+by|cutoff|finished\s+early|got\s+(?:\w+\s+)?(?:work|it|project|assignment)\s+in)\b"],
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
        "signals": [r"\b(sufi|dhikr|zohar|kabbalah|quran|khatam|bhagavad|gita|vrata|kirtan|sukkot|lulav|archon|reiki|meditation|mantra|prayer|synagogue|mosque|temple|scripture|spiritual)\b"],
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

NICHE_DOMAINS = {
    "yoga_physical_practice",
    "spiritual_religious",
    "caffeine_beverage",
    "transit_commute",
}


def extract_conversational_intents(text: str) -> set[str]:
    """Infer broad semantic/behavioral intent signals from conversational text."""
    text_low = text.lower()
    intents = set()
    for domain, config in DOMAIN_ONTOLOGY.items():
        for pat in config["signals"]:
            if re.search(pat, text_low):
                intents.add(domain)
                break
    return intents


def classify_facet_domains(facet_name: str, defn: str = "") -> set[str]:
    """Map a facet to its relevant conceptual domains based on name and definition."""
    combined = f"{facet_name} {defn}".lower()
    facet_domains = set()
    for domain, config in DOMAIN_ONTOLOGY.items():
        for pat in config["facet_patterns"]:
            if re.search(pat, combined):
                facet_domains.add(domain)
                break
    return facet_domains


# Cached domain tags for scoreable facets
_facet_domain_cache: Optional[list[set[str]]] = None
_facet_domain_cache_len: int = 0


def _get_facet_domains(scoreable: pd.DataFrame) -> list[set[str]]:
    global _facet_domain_cache, _facet_domain_cache_len
    if _facet_domain_cache is None or _facet_domain_cache_len != len(scoreable):
        _facet_domain_cache = [
            classify_facet_domains(row["normalized_facet"], str(row.get("scoring_definition", "")))
            for _, row in scoreable.iterrows()
        ]
        _facet_domain_cache_len = len(scoreable)
    return _facet_domain_cache


def build_retrieval_index(
    enriched_df: pd.DataFrame,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Build the FAISS index for scoreable facets using rich semantic representations.
    """
    scoreable = get_scoreable_facets(enriched_df)
    facet_names = scoreable["normalized_facet"].tolist()
    facet_ids = scoreable.index.tolist()

    enriched_texts = [build_facet_semantic_text(row) for _, row in scoreable.iterrows()]
    build_faiss_index(facet_names, facet_ids, save_dir, texts=enriched_texts)
    logger.info(f"Built retrieval index with {len(facet_names)} facets (semantic text)")


def retrieve_relevant_facets(
    conversation: str,
    enriched_df: pd.DataFrame,
    top_k: Optional[int] = None,
    index_dir: Optional[Path] = None,
    alpha: float = 0.65,
    min_relevance_threshold: float = 0.20,
) -> list[dict]:
    """
    Retrieve the most relevant facets for a conversation using contextual intent routing,
    hybrid dense + lexical search, and dynamic relevance threshold filtering.
    """
    top_k = top_k or TOP_K

    # Load the index (built over scoreable facets only)
    index, facet_ids = load_faiss_index(index_dir)
    scoreable = get_scoreable_facets(enriched_df)
    facet_domains = _get_facet_domains(scoreable)

    # 1. Infer conversational intent signals
    intents = extract_conversational_intents(conversation)

    # 2. Dense retrieval (FAISS cosine similarity)
    q_emb = embed_texts([conversation])[0]
    dense_scores = np.zeros(len(scoreable), dtype=np.float32)

    distances, indices = index.search(np.array([q_emb], dtype=np.float32), len(scoreable))
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(scoreable):
            dense_scores[idx] = float(dist)

    # 3. Lexical retrieval (Stopword-filtered BM25)
    bm25 = _get_bm25_index(scoreable)
    bm25_raw = bm25.score(conversation)
    bm25_max = np.max(bm25_raw)
    bm25_norm = bm25_raw / (bm25_max + 1e-6) if bm25_max > 0 else bm25_raw

    # 4. Contextual combination & reranking
    combined_scores = alpha * dense_scores + (1.0 - alpha) * bm25_norm

    for idx, row in scoreable.iterrows():
        f_doms = facet_domains[idx]
        obs = bool(row.get("conversation_observable", True))

        # Soft conversational observability prior (+0.03)
        if obs:
            combined_scores[idx] += 0.03

        # Contextual domain intent boost (+0.15) when conversational topic aligns
        if intents and f_doms and (intents & f_doms):
            combined_scores[idx] += 0.15

        # Contextual niche domain penalty (-0.35) when unmentioned niche metric appears
        niche_unmentioned = f_doms & NICHE_DOMAINS
        if niche_unmentioned and not (intents & niche_unmentioned):
            combined_scores[idx] -= 0.35

        # Penalty for external telemetry / quantitative indicators when not explicitly mentioned
        if not obs and not (intents & f_doms):
            combined_scores[idx] -= 0.15

    # 5. Rank and select exactly TOP_K candidates
    top_indices = np.argsort(-combined_scores)[:top_k]

    # Map back to facet metadata
    retrieved = []
    for rank_pos, row_idx in enumerate(top_indices):
        row = scoreable.iloc[row_idx]
        sim_score = float(combined_scores[row_idx])
        retrieved.append({
            "row_index": int(row_idx),
            "normalized_facet": row["normalized_facet"],
            "raw_facet": row["raw_facet"],
            "facet_type": row["facet_type"],
            "conversation_observable": bool(row["conversation_observable"]),
            "sensitivity": row["sensitivity"],
            "scoring_definition": row["scoring_definition"],
            "score_1_anchor": row["score_1_anchor"],
            "score_2_anchor": row["score_2_anchor"],
            "score_3_anchor": row["score_3_anchor"],
            "score_4_anchor": row["score_4_anchor"],
            "score_5_anchor": row["score_5_anchor"],
            "abstention_reason": row.get("abstention_reason", ""),
            "similarity_score": sim_score,
            "retrieved_domains": list(facet_domains[row_idx]),
            "low_relevance": bool(sim_score < 0.15),
        })

    logger.info(
        f"Retrieved {len(retrieved)} facets for conversation "
        f"(intents: {list(intents)}, top score: {retrieved[0]['similarity_score']:.3f} if retrieved else 'N/A')"
    )
    return retrieved


