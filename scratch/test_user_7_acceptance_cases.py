import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline

def main():
    test_cases = [
        ("TEST 1", "I submitted my assignment two days before the deadline."),
        ("TEST 2", "My assignment was due Monday, but I submitted it Wednesday."),
        ("TEST 3", "My assignment is due tomorrow, but I haven't submitted it yet."),
        ("TEST 4", "I used to miss deadlines, but after reorganizing my schedule I started submitting all my assignments at least two days early."),
        ("TEST 5", "I failed my exam twice, but I kept studying every day and passed on my third attempt."),
        ("TEST 6", "I consume 300 mg of caffeine every day."),
        ("TEST 7", "I practice yoga for five hours every week."),
    ]

    print("=" * 80)
    print("USER ACCEPTANCE TEST SUITE (CASES 1 - 7)")
    print("=" * 80)

    for label, conv in test_cases:
        print("\n" + "=" * 80)
        print(f"[{label}] Conversation: \"{conv}\"\n")
        res = run_pipeline(conv, conversation_id=label.lower().replace(" ", "_"))
        for i, r in enumerate(res.get("results", [])[:20], 1):
            facet = r["facet"]
            status = r["status"].upper()
            score = f"{r['score']}/5" if r['score'] is not None else "null"
            conf = r["confidence"]
            reason = r["reason"]
            # Highlight scored facets or top candidates
            if status == "SCORED":
                print(f" {i:2d}. {facet:<40} [{status:<21}] Score: {score:<4} (Conf: {conf:.2f})")
                print(f"     Reason: {reason}\n")
            elif i <= 3:
                print(f" {i:2d}. {facet:<40} [{status:<21}] Score: {score:<4} (Conf: {conf:.2f})")
                print(f"     Reason: {reason}\n")
        print("=" * 80)

if __name__ == "__main__":
    main()
