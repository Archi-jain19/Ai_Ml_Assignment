import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run_pipeline

cases = [
    ("My Manager Rejected Proposal (First-Person Actor)", "My manager rejected my proposal, but I calmly explained why I disagreed, defended my reasoning, and accepted the final decision."),
    ("Pure Third-Party Brother", "My brother studied every night and never gave up. He changed his approach and eventually passed.")
]

for label, conv in cases:
    print("=" * 80)
    print(f"TEST: {label}")
    print(f"Conversation: \"{conv}\"\n")
    res = run_pipeline(conv)
    for i, r in enumerate(res["results"], 1):
        status = r["status"].upper()
        score_str = f"Score: {r['score']}/5" if r['score'] is not None else "Score: null"
        conf = r["confidence"]
        reason = r["reason"]
        facet = r["facet"]
        if status == "SCORED" or i <= 10:
            print(f"{i:2d}. {facet:<45} [{status:<21}] {score_str:<12} (Conf: {conf})")
            print(f"    Reason: {reason}\n")
