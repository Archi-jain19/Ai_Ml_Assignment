import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run_pipeline

test_cases = [
    ("Explicit Negative Evidence", "I failed twice and decided not to try again."),
    ("Ambiguous / Undecided Evidence", "I failed twice and haven't decided what to do yet."),
    ("Early Submission with Deadline", "I submitted my assignment two days before the deadline."),
    ("Late Submission Past Deadline", "The assignment was due on Monday, but I submitted it on Wednesday."),
    ("Upcoming Deadline Not Passed", "It's due tomorrow and I haven't submitted it."),
    ("Submission Without Deadline Mention", "I submitted my assignment yesterday."),
    ("Compliance vs Document Submission", "I followed my manager's instructions throughout the project."),
    ("Perseverance Independence", "I kept studying every day despite failing twice.")
]

for label, conv in test_cases:
    print("=" * 80)
    print(f"TEST: {label}")
    print(f"Conversation: \"{conv}\"\n")
    res = run_pipeline(conv)
    # Print top 5 candidates
    for i, r in enumerate(res["results"][:6], 1):
        status = r["status"].upper()
        score_str = f"Score: {r['score']}/5" if r['score'] is not None else "Score: null"
        conf = r["confidence"]
        reason = r["reason"]
        facet = r["facet"]
        print(f"{i:2d}. {facet:<35} [{status:<21}] {score_str:<12} (Conf: {conf})")
        print(f"    Reason: {reason}\n")
