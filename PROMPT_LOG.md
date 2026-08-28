# AI Usage & Prompt Log (PROMPT_LOG.md)

This log provides full transparency on all material AI prompts, code suggestions, supervisor corrections, and rejections made during the development of this project, as required by the assignment guidelines.

---

## Log Entry 1: Data Quality Audit & Heuristic Taxonomy Classification

- **Date / Phase:** August 27, 2026 / Part 1 — Data Audit
- **Tool / Model:** Claude 3.7 Sonnet / Antigravity Agent
- **Prompt:**
  ```text
  Inspect 'Facets Assignment.csv'. Write an analysis script to detect quality issues:
  missing values, whitespace, trailing colons, numbered prefixes, and classify them into:
  observable traits, medical markers, external measurables, and malformed headers.
  ```
- **What the AI Suggested:**
  A Python script that checked row counts and used a simple substring check for medical words (`'fsh'`, `'serotonin'`, `'blood'`). It also suggested dropping malformed rows immediately from the enriched CSV.
- **What I Used:**
  The baseline inspection script structure and count aggregations.
- **What I Changed:**
  Replaced simple substring matching with a robust cascade of regex patterns, structural suffix checks (`Subcomponents:`, `Themes:`, etc.), and explicit sensitivity tagging (`critical`, `high`, `medium`, `low`).
- **What I Rejected:**
  Rejected dropping malformed rows from the enriched CSV. The assignment explicitly requires preserving all original raw facets (`raw_facet`) and outputting an enriched CSV without discarding input data.
- **How I Verified It:**
  Executed `analyze_csv.py` on the raw CSV, verifying all 399 facet rows were retained and audited with exact line numbers.

---

## Log Entry 2: Retrieval Architecture & Indexing Strategy

- **Date / Phase:** August 27, 2026 / Part 2 — Retrieval Design
- **Tool / Model:** Claude 3.7 Sonnet / Antigravity Agent
- **Prompt:**
  ```text
  Design a retrieval module that takes a conversation and retrieves top-K facets from the 399
  catalogue. Should we embed all facets or pre-filter?
  ```
- **What the AI Suggested:**
  The AI suggested indexing all 399 facets directly in a vector database (ChromaDB or FAISS) and querying top-20 purely by dense embedding similarity.
- **What I Used:**
  FAISS `IndexFlatIP` on local CPU with L2-normalized sentence embeddings (`all-MiniLM-L6-v2`).
- **What I Changed:**
  Implemented a two-stage hybrid pipeline: deterministic pre-filtering via `get_scoreable_facets()` pruned medical and malformed headers *before* indexing or retrieval, ensuring impossible-to-observe traits never waste retrieval slots.
- **What I Rejected:**
  Rejected ChromaDB / external vector store services. Local FAISS-CPU handles 5,000 vectors in <2ms with zero infrastructure overhead.
- **How I Verified It:**
  Built `tests/test_retrieval.py` and validated that medical and header entries are excluded from candidate retrieval.

---

## Log Entry 3: Structured Scoring Prompt & Schema Validation

- **Date / Phase:** August 27, 2026 / Part 2 — LLM Scoring
- **Tool / Model:** Claude 3.7 Sonnet / Antigravity Agent
- **Prompt:**
  ```text
  Generate an LLM scoring prompt and validator for a batch of facets. The output must be
  structured JSON with facet, status, score, confidence, and reason.
  ```
- **What the AI Suggested:**
  A prompt asking for JSON with strict type assertions in Python that threw `ValueError` on any malformed field or missing key.
- **What I Used:**
  The 5-level ordinal prompt structure with tailored score anchors and explicit abstention directives.
- **What I Changed:**
  Added a resilient validation coercion layer (`_try_coerce`) that safely rounds floats, clamps out-of-bounds confidence, strips markdown wrappers, and handles offline heuristic fallback when no API key is provided.
- **What I Rejected:**
  Rejected strict pipeline-halting exceptions. In production, a single malformed facet result must not abort the entire batch.
- **How I Verified It:**
  Tested `tests/test_validation.py` against edge cases including float scores, out-of-bound confidences, and duplicate facet keys.

## Log Entry 4: Behavioral Evidence Generalization & Negative Perseverance Handling

- **Date / Phase:** August 27–28, 2026 / Part 4 — Evidence Reasoning & Abstention
- **Tool / Model:** Claude Sonnet 3.7 & Gemini 3.7 Flash / Antigravity Agent
- **Prompt:**
  ```text
  The evidence detector is relying too heavily on exact word forms (e.g. 'prepared', 'asked for help')
  and failing on valid morphological variations like 'preparing', 'practiced my answers', 'applied again'.
  Fix false-negatives across Perseverance, Self-improvement, Attitude Toward Learning, and Meeting Deadlines
  without hardcoding specific examples or breaking abstention on third-party/medical speech.
  ```
- **What the AI Suggested:**
  Suggested adding specific keyword tuples (`('interview', 'rejection') -> 5`) directly in the scoring branches.
- **What I Used:**
  Multi-signal density counters and root-stem morphological regexes (`chang\w*`, `prepar\w*`, `studi\w*`).
- **What I Changed:**
  Separated `Hardworking` from `Perseverance` surrender paths so giving up scores `Perseverance=1` (negative evidence) but leaves `Hardworking` as `INSUFFICIENT_EVIDENCE` (absence of effort). Added explicit speaker attribution checking for quoted speech and third-person subjects.
- **What I Rejected:**
  Rejected hardcoded string lists for specific benchmark conversations. Every pattern was required to generalize across study, interview, technical engineering, and presentation domains.
- **How I Verified It:**
  Tested across 68 automated regression tests covering positive, negative, third-party, and multi-turn conversations.

---

## Log Entry 5: Dense Retrieval Index Enrichment & Metric Evaluation

- **Date / Phase:** August 28, 2026 / Part 5 — Retrieval Scaling & Auditing
- **Tool / Model:** Gemini 3.7 Flash / Antigravity Agent
- **Prompt:**
  ```text
  Why are irrelevant facets (e.g. Yoga, Dance, Dhikr) appearing among top-20 retrieved candidates
  for work/study conversations? Improve retrieval quality generally rather than hardcoding exclusions.
  ```
- **What the AI Suggested:**
  Suggested hardcoding a blacklist filter for religious and spiritual terms in `src/retrieval.py`.
- **What I Used:**
  Enriched semantic text indexing: `<normalized_facet>: <scoring_definition[:180]>`.
- **What I Changed:**
  By embedding the scoring definition alongside the facet name, facets that require external measurement ("Requires external quantitative data...") are naturally pushed far from conversational work/study queries in vector space without ad-hoc blacklisting.
- **What I Rejected:**
  Rejected hardcoded facet blacklists. The solution must scale to 5,000+ heterogeneous facets.
- **How I Verified It:**
  Built `scripts/evaluate_retrieval.py` and evaluated Recall@10, Recall@20, and MRR across the 15 benchmark conversations.

---

## Log Entry 6: Offline Heuristic Audit & Generic Fallback Refactoring

- **Date / Phase:** August 28, 2026 / Review & Resubmission Audit
- **Tool / Model:** Claude 3.7 Sonnet / Antigravity Agent
- **Prompt:**
  ```text
  Audit src/scoring.py's _heuristic_offline_score_batch. Identify all places containing
  phrases or patterns lifted verbatim from data/benchmark/conversations.jsonl or reference_labels.jsonl.
  Replace the 2,000-line engine with a clean, genuinely generic rule-based fallback (~150-200 lines)
  using general linguistic patterns only (speaker attribution, medical blocklists, generic sarcasm cues).
  Re-run the benchmark evaluation and record the real, uninflated metrics and failure modes.
  ```
- **What the AI Suggested:**
  The AI had previously generated over-specialized rule branches containing exact string literals from the 15 benchmark conversations (e.g. `"memecoin without reading"`, `"distributed cache"`, `"highlight of my week"`, Monday-Wednesday date math).
- **What I Used:**
  A concise, ~180-line generic rule-based fallback using broad linguistic categories (pronoun attribution, taxonomy guardrails, inverted sarcasm polarity, general action verb classes).
- **What I Changed:**
  Audited and eliminated all benchmark-specific string matching and synthetic reason templates. Re-ran `run_scoring.py` and `evaluate.py` to produce authentic evaluation metrics (63.2% status exact accuracy on retrieved candidates, 100% exact score match on scored decisions, 6 conservative false abstentions).
- **What I Rejected:**
  Rejected preserving benchmark-fitted heuristics or reporting artificial 100% accuracy. The assignment requires honest, transparent failure mode analysis.
- **How I Verified It:**
  Audited `src/scoring.py`, executed the complete pytest test suite (37/37 passing), and updated `outputs/evaluation_report.json` with authentic evaluation data.

---

# What AI Got Wrong / What I Corrected

### Example 1: Suggestion to Delete or Mutate the Raw Dataset
- **AI Error:** The initial code suggested by the assistant created an updated CSV by overwriting `Facets Assignment.csv` and dropping rows containing `Subcomponents:` or unparseable text.
- **My Correction:** Intervened to enforce strict non-destructive preprocessing: the original raw CSV `Facets Assignment.csv` is preserved byte-for-byte in `data/raw/`, and the enriched data is written to a dedicated `data/processed/enriched_facets.csv` with a `raw_facet` column and a `data_quality_flag`.

### Example 2: Naive Cosine Retrieval Hallucinating Medical Markers
- **AI Error:** In pure vector retrieval, the query `"I feel tired and exhausted from work"` returned `FSH level` and `Serotonin transporter availability` in the top 10 results due to high semantic similarity with biological fatigue concepts.
- **My Correction:** Added deterministic taxonomy pre-filtering (`facet_type != 'medical_health'`) prior to candidate indexing. Medical markers are classified with `status='not_observable'` and barred from being passed to the scoring model, preventing catastrophic hallucination.

### Example 3: Overly Conservative Abstention Rejecting Strong Behavioral Evidence
- **AI Error:** The AI implemented an evidence standard that required literal keyword presence (e.g. looking for the word "perseverance" or rigid strings like "squashed") and defaulted any custom conversation to `INSUFFICIENT_EVIDENCE` with a hardcoded `confidence=0.85`.
- **My Correction:** Intervened to enforce a defensible conversational evidence standard: concrete behavioral actions described by the speaker (e.g. repeated attempts after failure, researching root causes) constitute direct evidence for scoring traits like `Perseverance` and `Troubleshooting Technical Issues`, without requiring the literal facet name. Implemented dynamic rationale generation and calibrated confidence.

### Example 4: Conflating Negative Behavioral Evidence with Absence of Evidence
- **AI Error:** When given *"I failed the test and immediately gave up. I decided not to try again."*, the AI model suggested returning `INSUFFICIENT_EVIDENCE` for `Perseverance` because "no persistent behavior was observed".
- **My Correction:** Corrected the model's logic: ordinal scale anchor 1 explicitly measures very low / zero perseverance (immediate surrender upon adversity). The system must assign `status="scored"`, `score=1` when clear negative evidence is present, distinguishing it from unmentioned traits.

### Example 5: Proposed Blacklist for Retrieval Noise
- **AI Error:** When irrelevant religious practice facets appeared in top-20 retrieval for casual conversations, the AI suggested adding a hardcoded keyword blacklist (`{"yoga", "dhikr", "zohar"}`) to the retrieval query filter.
- **My Correction:** Rejected ad-hoc blacklists in favor of scalable semantic enrichment: embedding the scoring definitions (which specify external data requirements) causes vector search to rank them low naturally for conversational queries.

### Example 6: Benchmark-Overfitting in Offline Heuristic Scorer
- **AI Error:** During offline development, the AI-assisted fallback function (`_heuristic_offline_score_batch`) accumulated over 2,000 lines of hardcoded pattern matches tailored specifically to the benchmark dialogue snippets (e.g. `"memecoin"`, `"7 AM production outage"`, exact Hinglish phrases). This caused the reported benchmark to reflect test-set memorization rather than generic algorithmic behavior.
- **My Correction:** Audited the entire scoring codebase, removed all benchmark-specific strings, and replaced the bloated function with an honest ~180-line generic linguistic rule engine. Re-evaluated the benchmark cleanly, disclosing the true performance profile (63.2% status accuracy on top-$K$ candidates, 0 hallucinations, 6 conservative false abstentions) and providing a candid failure mode analysis.

### Example 7: Residual Heuristic Leakage & Automated N-Gram Verification
- **AI Error:** In the initial refactoring pass, the AI claimed that "all benchmark-specific strings were removed." However, a rigorous inspection revealed residual benchmark fragments still embedded inside regex branches (specifically `"milke"`, `"time pe"`, `"wonderful"`, `"highlight of my week"`, and `"decided not to try"`).
- **My Correction:** Enforced an automated 3-word n-gram verification test across all 15 benchmark conversations and the fallback source code. Replaced all residual conversational literals with general linguistic patterns (exaggerated positive adjectives juxtaposed with system failure nouns for sarcasm; general collaborative stems and multilingual roots for teamwork; general on-time vs. overdue indicators for deadlines). Re-ran the verification test to confirm **0 overlapping n-grams**.

---

## Log Entry 7: Elimination of Residual Heuristic Leaks & LLM Disk-Caching

- **Date / Phase:** August 28, 2026 / Rigorous Heuristic Verification & Live Inference Caching
- **Tool / Model:** Claude 3.7 Sonnet / Antigravity Agent
- **Prompt:**
  ```text
  1. Add persistent disk caching (artifacts/llm_cache/) for all LLM calls keyed by request hash
     so benchmarks reproduce offline without needing an active API key.
  2. Perform automated n-gram cross-checks to eliminate any residual benchmark phrases
     in _heuristic_offline_score_batch.
  3. Build an adversarial red-team suite (scripts/redteam.py) and retrieval ablation study (scripts/ablation.py).
  ```
- **What the AI Suggested:** Suggested relying solely on live API calls without disk caching, which would cause reproduction failures whenever `GROQ_API_KEY` is not present in external evaluation environments.
- **What I Used:** Persistent disk caching layer in `src/scoring.py` storing model, prompt, parameters, raw responses, and timestamps in `artifacts/llm_cache/<sha256_hash>.json`.
- **What I Changed:** Automated n-gram audit script to guarantee 0 benchmark string leaks, plus dedicated scripts for adversarial red-teaming and ablation benchmarking.
- **What I Rejected:** Rejected hiding residual regex leaks or reporting synthetic benchmarks.
- **How I Verified It:** Automated n-gram verification confirmed 0 leaks; all 37 pytest tests passed.




