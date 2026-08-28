import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import score_facets_batch

def run_tests():
    print("=" * 80)
    print("SUBMISSION & MEETING DEADLINES BEHAVIORAL DISAMBIGUATION TEST")
    print("=" * 80)

    test_cases = [
        (
            "Case 1: Direct assignment submission (no deadline)",
            "I submitted my assignment yesterday.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "Case 2: Submission before deadline",
            "I submitted my assignment two days before the deadline.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "Case 3: Synonymous turn-in (no deadline)",
            "I turned in my assignment yesterday.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "Case 4: Habitual on-time submission",
            "I submit my assignments on time.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "Case 5: Due date stated, no submission evidence",
            "My assignment is due Friday.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "Case 6: Future intention, not completed delivery",
            "I plan to submit my assignment tomorrow.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "Case 7: Third-party reported attribution",
            "My professor said I submitted my assignment.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
        (
            "Case 8: Interpersonal Submissiveness (True psychological Submission)",
            "I gave in to their demands and complied with everything they ordered.",
            [
                {"normalized_facet": "Submission", "facet_type": "conversation_observable", "conversation_observable": True},
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True},
            ]
        ),
    ]

    for label, conv, facets in test_cases:
        print(f"\n[{label}] \"{conv}\"")
        for r in score_facets_batch(conv, facets, client=None):
            print(f"  {r['facet']:<22} -> {r['status'].upper():<22} (Score: {str(r.get('score')):<5})")
            print(f"    Reason: {r['reason']}")

if __name__ == "__main__":
    run_tests()
