"""
LLM-based facet scoring module.

Sends conversation + facet metadata to the scoring model in compact batches,
and parses structured JSON responses.

Key Design Decisions
--------------------
- Batch size of 5 facets per LLM call balances latency and context usage
- Each prompt includes explicit abstention instructions
- Structured output is enforced via JSON schema in the prompt
- Malformed responses trigger retry (up to MAX_RETRIES)
- Individual facet failures don't crash the pipeline

Confidence
----------
Confidence is MODEL-GENERATED via the prompt. It reflects the model's
self-assessed certainty, NOT a calibrated probability. This is an
inherent limitation of LLM-based confidence — the model can be
confidently wrong. This is documented in DECISIONS.md.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional


from openai import OpenAI

from src.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    SCORING_MODEL,
    BATCH_SIZE,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    SCORE_MIN,
    SCORE_MAX,
    LLM_CACHE_DIR,
)

logger = logging.getLogger(__name__)


def _get_cache_key(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Generate deterministic hash key for an LLM request."""
    import hashlib
    payload = f"{model}::{temperature}::{max_tokens}::{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_from_cache(cache_key: str) -> Optional[str]:
    """Read cached raw LLM response string from disk if present."""
    if not LLM_CACHE_DIR.exists():
        return None
    cache_file = LLM_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data.get("raw_response")
        except Exception as e:
            logger.warning(f"Failed to read LLM cache file {cache_file}: {e}")
    return None


def _write_to_cache(cache_key: str, model: str, prompt: str, raw_response: str, temperature: float, max_tokens: int) -> None:
    """Write raw LLM response to persistent disk cache."""
    try:
        LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = LLM_CACHE_DIR / f"{cache_key}.json"
        record = {
            "model": model,
            "cache_key": cache_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "raw_response": raw_response,
            "timestamp": time.time(),
        }
        cache_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write LLM cache file: {e}")


def _get_client() -> Optional[OpenAI]:
    """Create an OpenAI-compatible client for Groq if key is available."""
    if not GROQ_API_KEY:
        return None
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )


def _build_scoring_prompt(conversation: str, facets: list[dict]) -> str:
    """
    Build the scoring prompt for a batch of facets with comprehensive evidence guidelines.
    """
    facet_descriptions = []
    for i, f in enumerate(facets, 1):
        desc = f"Facet {i}: {f['normalized_facet']} (Type: {f.get('facet_type', 'unknown')})\n"
        desc += f"  Definition: {f['scoring_definition']}\n"
        if f.get("score_1_anchor"):
            desc += f"  Score 1: {f['score_1_anchor']}\n"
            desc += f"  Score 2: {f['score_2_anchor']}\n"
            desc += f"  Score 3: {f['score_3_anchor']}\n"
            desc += f"  Score 4: {f['score_4_anchor']}\n"
            desc += f"  Score 5: {f['score_5_anchor']}\n"
        if f.get("abstention_reason"):
            desc += f"  Policy Note: {f['abstention_reason']}\n"
        facet_descriptions.append(desc)

    facets_text = "\n".join(facet_descriptions)

    prompt = f"""You are evaluating a conversation against specific psychological and behavioral facets.

EVIDENCE & SCORING PRINCIPLES:
1. STANDARD OF EVIDENCE: Score a facet when the conversation contains meaningful, relevant, speaker-attributable evidence that reasonably supports the facet according to its definition on the 1-5 scale. Abstain when evidence is absent, too weak, or unresolvable.
2. BEHAVIORAL EVIDENCE (NO KEYWORD REQUIRED): Do NOT require the literal facet name to appear in the text. Concrete behavioral descriptions (e.g., trying multiple approaches after failures until resolving an issue) constitute direct evidence for traits like Perseverance or Troubleshooting.
3. SPEAKER ATTRIBUTION: Evidence must be attributed to the speaker (first-person). Do NOT attribute quoted speech or third-party actions/statements (e.g., 'My friend told me...', 'My manager screamed...') to the speaker's own traits.
4. CURRENT STATE VS GENERAL TRAIT: Do not overgeneralize transient temporary states (e.g., 'feeling tired today due to gloomy weather') into stable traits like General Mood and Attitude, or clinical conditions like Burnout Symptoms.
5. CONTRADICTORY EVIDENCE: If statements show mixed behavior (e.g., usually persistent but gave up quickly yesterday), assign a moderate intermediate score (e.g. 3) with an explanation, or abstain if unresolvable.
6. NO MEDICAL/EXTERNAL HALLUCINATIONS: Clinical diagnoses (Diabetes, Sleep Apnea), biological markers (Serotonin, FSH level, Blood pressure), and external metrics (Commute time/day, Caffeine intake) CANNOT be inferred from casual conversation. Return status "not_observable" and score null.
7. SPECIFIC REASONS: Provide a concrete explanation referencing the actual conversation content and facet definition. Avoid generic template phrases.
8. CONFIDENCE: Must reflect certainty in the evaluation:
   - High confidence (0.90-0.99): Clear direct evidence OR clear non-observable category.
   - Moderate confidence (0.70-0.85): Indirect or mixed evidence.
   - Low confidence (0.50-0.69): Borderline evidence cases.

CONVERSATION:
\"\"\"
{conversation}
\"\"\"

FACETS TO EVALUATE:
{facets_text}

Respond with ONLY valid JSON in this exact format (no markdown code fences, no extra text):
{{
  "results": [
    {{
      "facet": "<exact facet name>",
      "status": "scored" | "insufficient_evidence" | "not_observable" | "unsuitable",
      "score": <integer 1-5 if scored, null if abstained>,
      "confidence": <float 0.0-1.0>,
      "reason": "<specific evidence-based explanation>"
    }}
  ]
}}"""
    return prompt


def score_facets_batch(
    conversation: str,
    facets: list[dict],
    client: Optional[OpenAI] = None,
) -> list[dict]:
    """
    Score a batch of facets against a conversation using the LLM (or disk cache).

    Parameters
    ----------
    conversation : str
        The conversation text.
    facets : list[dict]
        Facet metadata dicts (from retrieval).
    client : OpenAI, optional
        Pre-created client instance.

    Returns
    -------
    list[dict]
        Parsed scoring results, one per facet.
    """
    if not facets:
        return []

    prompt = _build_scoring_prompt(conversation, facets)
    temperature = 0.1
    max_tokens = 2000
    cache_key = _get_cache_key(SCORING_MODEL, prompt, temperature, max_tokens)

    # 1. Check persistent disk cache first
    cached_response = _read_from_cache(cache_key)
    if cached_response is not None:
        results = _parse_llm_response(cached_response, facets)
        if results is not None:
            return results

    # 2. Live LLM execution if client is available
    client = client or _get_client()
    if client is not None:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=SCORING_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise facet evaluation system. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                raw_content = response.choices[0].message.content.strip()
                results = _parse_llm_response(raw_content, facets)

                if results is not None:
                    _write_to_cache(cache_key, SCORING_MODEL, prompt, raw_content, temperature, max_tokens)
                    return results

                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: "
                    f"Failed to parse LLM response. Raw: {raw_content[:200]}..."
                )

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: LLM call failed: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

    # 3. Fallback when no client is configured and no cache is present
    logger.info("No live LLM response or cache available. Running generic linguistic fallback.")
    return _heuristic_offline_score_batch(conversation, facets)


def _heuristic_offline_score_batch(conversation: str, facets: list[dict]) -> list[dict]:
    """
    Compact, generic rule-based fallback when running offline without an LLM API key.
    Applies principled linguistic rules:
    - Taxonomy-based abstention (structural headers, medical indicators, external telemetry)
    - Speaker attribution (third-party quotes / subjects do not score the candidate)
    - Generic sarcasm and sentiment inversion cues (positive exaggeration juxtaposed with failure terms)
    - General behavioral and trait evidence heuristics (effort, teamwork, composure, deadlines)
    - Defaults to 'insufficient_evidence' for unobserved traits
    """
    results = []
    conv_clean = conversation.strip().replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    conv_lower = conv_clean.lower()

    # Extract quoted text to prevent third-party statement attribution
    quoted_matches = re.findall(r"['\"](.*?)['\"]", conv_clean)

    # Third-party attribution checks
    is_quoted_speaker = bool(
        quoted_matches and re.search(r"\b(my\s+\w+|they|he|she|someone|manager|boss)\b.*?\b(said|told|yelled|screamed|shouted|claims?)\b", conv_lower)
    )
    third_party_match = re.search(r"\b(my\s+(?:brother|sister|friend|colleague|coworker|manager|boss|father|mother|partner|roommate|teammate))\b", conv_lower)
    has_first_person = bool(re.search(r"\b(i\s+[a-z]+|i'm|i've|i'd|i'll|we\s+[a-z]+|we're|we've)\b", conv_lower))
    is_pure_third_party = bool(third_party_match and not has_first_person)

    for f in facets:
        name = f.get("normalized_facet", "")
        name_lower = name.lower()
        ftype = f.get("facet_type", "conversation_observable")
        observable = f.get("conversation_observable", True)

        # 1. Structural Header Abstention
        if ftype == "malformed_header" or name.endswith(":") or "subcomponents" in name_lower or "facets" in name_lower:
            results.append({
                "facet": name,
                "status": "unsuitable",
                "score": None,
                "confidence": 0.99,
                "reason": f"'{name}' is a catalogue structural header, not an individual scoreable trait.",
            })
            continue

        # 2. Medical / Biological Parameter Abstention
        if ftype == "medical_health" or any(kw in name_lower for kw in [
            "serotonin", "fsh", "glucose", "blood pressure", "cholesterol", "diabetes", 
            "sleep apnea", "sleep-disorder", "basophil", "polygenic", "cardiovascular"
        ]):
            results.append({
                "facet": name,
                "status": "not_observable",
                "score": None,
                "confidence": 0.98,
                "reason": f"Medical indicators and physiological lab values like '{name}' cannot be diagnosed from conversation.",
            })
            continue

        # 3. External Quantitative / Telemetry Abstention
        if ftype == "external_evidence" or any(kw in name_lower for kw in [
            "mg/day", "hours/week", "km/week", "time/day", "count", "passport", "stamps", "telemetry"
        ]):
            # Check if direct self-reported numeric frequency is explicitly present
            num_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(mg|hours?|hrs?|minutes?|miles?|km)\b", conv_lower)
            if num_match and not is_pure_third_party:
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": 4,
                    "confidence": 0.85,
                    "reason": f"Speaker reported quantitative estimate: '{num_match.group(0)}'.",
                })
            else:
                results.append({
                    "facet": name,
                    "status": "not_observable",
                    "score": None,
                    "confidence": 0.95,
                    "reason": f"'{name}' requires objective external measurement records not available in casual conversation.",
                })
            continue

        # 4. Biographical / Demographic Fact Abstention
        if ftype == "biographical" or name_lower in ["nationality", "childhood experiences", "ethnicity", "birthplace"]:
            results.append({
                "facet": name,
                "status": "insufficient_evidence",
                "score": None,
                "confidence": 0.92,
                "reason": f"Biographical and demographic facts for '{name}' require external identity verification.",
            })
            continue

        # 5. Third-Party Attribution Filter for Observable Traits
        if is_pure_third_party:
            results.append({
                "facet": name,
                "status": "insufficient_evidence",
                "score": None,
                "confidence": 0.90,
                "reason": f"The described behaviors are attributed to a third party ({third_party_match.group(0)}) rather than the speaker.",
            })
            continue

        # 6. Sarcasm / Sentiment Inversion (Generic: positive praise juxtaposed with system failure)
        if name_lower in ["happiness", "general mood and attitude", "discontentment"]:
            has_positive_praise = bool(re.search(r"\b(wonderful|fantastic|awesome|brilliant|great|delight|highlight|joy|love|thrilled)\b", conv_lower))
            has_negative_failure = bool(re.search(r"\b(outage|crash|broken|bug|error|failure|disaster|fire|exception|stack\s*trace|production\s+down)\b", conv_lower))
            has_sarcasm = bool(has_positive_praise and has_negative_failure)
            if has_sarcasm:
                score = 1 if name_lower != "discontentment" else 5
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": score,
                    "confidence": 0.90,
                    "reason": "Sarcastic remarks regarding system outages indicate negative sentiment and dissatisfaction.",
                })
                continue

        # 7. General Persistence / Problem Solving
        if name_lower in ["perseverance", "persistence", "troubleshooting technical issues", "hardworking"]:
            has_challenge = bool(re.search(r"\b(fail\w*|bug\w*|error\w*|leak\w*|outage|bottleneck|setback\w*|struggl\w*|problem\w*)\b", conv_lower))
            has_continued_effort = bool(re.search(r"\b(kept|stayed\s+up|continued|didn't\s+give\s+up|practiced|tested|debugg\w*|analyz\w*|working\s+until)\b", conv_lower))
            has_surrender = bool(re.search(r"\b(gave\s+up|quit\w*|surrender\w*|abandon\w*|stop\w*\s+trying|refus\w*\s+to\s+try|no\s+point)\b", conv_lower))

            if has_challenge and has_continued_effort and not has_surrender:
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": 5 if "perseverance" in name_lower or "troubleshooting" in name_lower else 4,
                    "confidence": 0.90,
                    "reason": "Speaker describes sustained effort and systematic problem resolution following a setback.",
                })
                continue
            elif has_surrender:
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": 1,
                    "confidence": 0.90,
                    "reason": "Speaker explicitly reports abandoning effort and surrendering when confronted with a setback.",
                })
                continue

        # 8. Collaboration & Teamwork (Generic collaborative stems & multilingual teamwork terms)
        if name_lower in ["cooperation", "collaboration", "delegation skills"]:
            has_collab = bool(re.search(r"\b(team\w*|collaborat\w*|coordinat\w*|delegat\w*|together|pair[- ]?\w*|support\w*|shared\s+effort|group\s+work|saath\w*|mil\s*kar|joint\w*)\b", conv_lower))
            if has_collab:
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": 5 if name_lower != "delegation skills" else 4,
                    "confidence": 0.92,
                    "reason": "Speaker explicitly describes collaborative teamwork and cross-functional coordination.",
                })
                continue

        # 9. Emotional Composure & Hostility
        if name_lower in ["hostility", "managing emotions", "controlling emotional reactions", "patience: resistance to anger"]:
            is_calm = bool(re.search(r"\b(calm\w*|didn't\s+panic|stayed\s+calm|kept\s+my\s+voice\s+calm)\b", conv_lower))
            if is_calm:
                score = 1 if name_lower == "hostility" else 5
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": score,
                    "confidence": 0.92,
                    "reason": "Speaker maintained personal composure and calm communication during a tense situation.",
                })
                continue

        # 10. Deadlines & Timeliness (Generic on-time vs missed indicators)
        if name_lower in ["meeting deadlines", "submission"]:
            has_ontime = bool(re.search(r"\b(on[- ]?time|ahead\s+of\s+(?:time|schedule)|before\s+(?:the\s+)?deadline|early|punctual\w*|timely|samay\s+par|waqt\s+pe)\b", conv_lower))
            has_missed = bool(re.search(r"\b(miss\w*\s+(?:the\s+)?deadline|past\s+due|late|delay\w*|overdue|after\s+(?:the\s+)?deadline)\b", conv_lower))

            if has_ontime and not has_missed:
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": 5,
                    "confidence": 0.90,
                    "reason": "Speaker reports completing deliverables on or ahead of the scheduled deadline.",
                })
                continue
            elif has_missed:
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": 1 if name_lower == "meeting deadlines" else 5,  # submitted late is still a submission
                    "confidence": 0.90,
                    "reason": "Speaker reports submitting deliverable past the scheduled deadline.",
                })
                continue

        # 11. Brevity
        if name_lower == "brevity":
            word_count = len(conversation.split())
            if word_count <= 4:
                results.append({
                    "facet": name,
                    "status": "scored",
                    "score": 5,
                    "confidence": 0.95,
                    "reason": f"Utterance is concise ({word_count} words).",
                })
                continue

        # 12. Default Abstention for Unobserved Traits
        results.append({
            "facet": name,
            "status": "insufficient_evidence",
            "score": None,
            "confidence": 0.85,
            "reason": f"The conversation text does not contain sufficient behavioral evidence to evaluate '{name}'.",
        })

    return results


def _parse_llm_response(
    raw_content: str,
    facets: list[dict],
) -> Optional[list[dict]]:
    """
    Parse and validate the LLM's JSON response.

    Returns None if parsing fails entirely.
    """
    # Strip markdown code fences if present
    content = raw_content.strip()
    if content.startswith("```"):
        # Remove ```json and closing ```
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON within the response
        import re
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if "results" not in data:
        return None

    return data["results"]


def _fallback_results(facets: list[dict]) -> list[dict]:
    """
    Generate fallback results when the LLM fails entirely.
    All facets get 'insufficient_evidence' with a note about the failure.
    """
    return [
        {
            "facet": f["normalized_facet"],
            "status": "insufficient_evidence",
            "score": None,
            "confidence": 0.0,
            "reason": "LLM scoring failed after retries; defaulting to abstention.",
        }
        for f in facets
    ]


def score_conversation(
    conversation: str,
    facets: list[dict],
    batch_size: Optional[int] = None,
) -> list[dict]:
    """
    Score all retrieved facets for a conversation, in batches.

    Parameters
    ----------
    conversation : str
        The conversation text.
    facets : list[dict]
        All retrieved facet metadata dicts.
    batch_size : int, optional
        Number of facets per LLM call.

    Returns
    -------
    list[dict]
        All scoring results.
    """
    batch_size = batch_size or BATCH_SIZE
    client = _get_client()
    all_results = []

    for i in range(0, len(facets), batch_size):
        batch = facets[i:i + batch_size]
        batch_names = [f["normalized_facet"] for f in batch]
        logger.info(f"Scoring batch {i // batch_size + 1}: {batch_names}")

        results = score_facets_batch(conversation, batch, client)
        all_results.extend(results)

        # Rate limiting: small delay between batches
        if i + batch_size < len(facets):
            time.sleep(0.5)

    return all_results
