import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run_pipeline

conv = "“I disagreed with my team lead’s approach, but I calmly explained my reasoning, listened to their response, and accepted the final decision.”"
res = run_pipeline(conv)
print(f"Conversation: \"{conv}\"\n")
for i, r in enumerate(res["results"], 1):
    status = r["status"].upper()
    score_str = f"Score: {r['score']}/5" if r['score'] is not None else "Score: null"
    conf = r["confidence"]
    reason = r["reason"]
    facet = r["facet"]
    print(f"{i:2d}. {facet:<45} [{status:<21}] {score_str:<12} (Conf: {conf})")
    print(f"    Reason: {reason}\n")
