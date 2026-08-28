import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run_pipeline

conv = "I was struggling to keep up with my coursework. Instead of giving up, I created a weekly schedule, studied for two hours every evening, and followed the schedule for the entire semester. My assignments are now consistently completed before their deadlines."

res = run_pipeline(conv)

print("=" * 80)
print(f"Conversation: \"{conv}\"\n")
for i, r in enumerate(res["results"], 1):
    status = r["status"].upper()
    score_str = f"Score: {r['score']}/5" if r['score'] is not None else "Score: null"
    conf = r["confidence"]
    reason = r["reason"]
    facet = r["facet"]
    print(f"{i:2d}. {facet:<35} [{status:<21}] {score_str:<15} (Conf: {conf})")
    print(f"    Reason: {reason}\n")
print("=" * 80)
