import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import score_facets_batch

def run():
    cases = [
        (
            "1. 'I submitted my assignment yesterday.'",
            "I submitted my assignment yesterday.",
            [
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "2. 'I submitted the application before the deadline.'",
            "I submitted the application before the deadline.",
            [
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "3. 'I always comply with my manager's instructions.'",
            "I always comply with my manager's instructions.",
            [
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "4. 'I submitted to my manager's decision even though I disagreed.'",
            "I submitted to my manager's decision even though I disagreed.",
            [
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "5. 'My assignment is due Friday, but I haven't submitted it yet.'",
            "My assignment is due Friday, but I haven't submitted it yet.",
            [
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "6. 'I submitted my assignment two days before the deadline.'",
            "I submitted my assignment two days before the deadline.",
            [
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
    ]

    print("=" * 80)
    print("USER'S 6 TARGET DISAMBIGUATION REGRESSION CASES")
    print("=" * 80)

    for label, conv, facets in cases:
        print(f"\n[{label}]")
        results = score_facets_batch(conv, facets, client=None)
        for r in results:
            score_str = f"{r['score']}/5" if r['score'] is not None else "null"
            print(f"  {r['facet']:<20} -> {r['status'].upper():<22} (Score: {score_str})")
            print(f"    Reason: {r['reason']}")

if __name__ == "__main__":
    run()
