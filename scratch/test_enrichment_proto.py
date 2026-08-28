import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval import load_enriched_facets
from src.embeddings import embed_texts

df = load_enriched_facets()

# Facet Concept Enrichment Map (general domain keywords and semantic expansions)
# This adds natural semantic anchors to facets without changing catalog names or deleting anything.
FACET_SEMANTIC_EXPANSIONS = {
    "perseverance": "persistence, grit, tenacity, determination, not giving up, enduring hardship, continuing through difficulty, resilience, overcoming failure",
    "persistence": "perseverance, steadfastness, continuing effort, not quitting, sticking with tasks through obstacles, determination",
    "character strength: perseverance": "grit, persistence, tenacity, overcoming adversity, not giving up, sustained effort through setbacks",
    "hardworking": "diligence, industriousness, dedicated effort, working long hours, putting in time and energy, tireless work ethic",
    "self-improvement": "personal development, learning from mistakes, adjusting approach, improving skills, seeking feedback, self-growth, practicing to get better",
    "attitude toward learning": "growth mindset, enthusiasm for studying, seeking feedback, active practice, curiosity, continuous education",
    "meeting deadlines": "timeliness, punctuality, submitting on time, completing before due date, delivering work on schedule",
    "troubleshooting technical issues": "debugging, diagnosing problems, isolating root causes, fixing bugs, technical problem solving, writing tests",
    "data analysis": "analyzing error logs, metrics, investigating data patterns, quantitative reasoning, diagnosing bottlenecks",
    "managing emotions": "emotional regulation, composure under pressure, remaining calm, patience, self-control during conflict",
    "patience: resistance to anger": "staying calm when provoked, tolerance, emotional restraint, composure during screaming or stress",
    "risktaking": "financial risk, speculation, daring actions, gambling, taking chances, bold uncalculated moves",
    "brevity": "conciseness, short answers, brief communication, laconic replies",
    "volunteer work": "community service, food bank, charity, volunteering time, helping non-profits",
    "caffeine intake (mg/day)": "drinking coffee, espresso, tea, energy drinks, caffeine consumption",
    "hindu spiritual metric: yoga discipline hours / week": "practicing yoga, asanas, yoga sessions, weekly yoga practice",
    "cooperation": "teamwork, collaboration, pair programming, supporting colleagues, working together",
    "delegation skills": "assigning tasks, delegating work to team members, dividing responsibilities",
    "happiness": "joy, cheerfulness, positive emotion, satisfaction, genuine delight",
    "discontentment": "frustration, dissatisfaction, resentment, unhappiness with situation, sarcasm about outages",
}

def build_enriched_text(row: pd.Series) -> str:
    name = row["normalized_facet"]
    name_low = name.lower()
    ftype = row["facet_type"]
    obs = row["conversation_observable"]
    
    parts = [name]
    
    # Add domain semantic expansion if available
    if name_low in FACET_SEMANTIC_EXPANSIONS:
        parts.append(f"Concepts: {FACET_SEMANTIC_EXPANSIONS[name_low]}")
    
    # Add category context
    if obs:
        parts.append("Observable behavioral trait and personality characteristic.")
    else:
        parts.append(f"External domain metric ({ftype}).")
        
    return " | ".join(parts)

enriched_texts = [build_enriched_text(row) for _, row in df.iterrows()]
print(f"Generated {len(enriched_texts)} enriched facet representations.")
print("\nSample enriched representations:")
for t in enriched_texts[:8]:
    print(" -", t)
