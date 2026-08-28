import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import score_facets_batch

def test_grounded_reasons():
    cases = [
        (
            "TEST 1",
            "I failed the exam once and decided not to try again.",
            [
                {"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Perseverance"},
                {"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Attitude toward learning"},
            ]
        ),
        (
            "TEST 2",
            "I had a difficult day at work and went home.",
            [
                {"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Perseverance"},
            ]
        ),
        (
            "TEST 3",
            "I practice yoga for five hours every week.",
            [
                {"normalized_facet": "Hindu Spiritual Metric: Yoga Discipline Hours / Week", "facet_type": "external_evidence", "conversation_observable": False, "scoring_definition": "Yoga hours"},
            ]
        ),
        (
            "TEST 4",
            "I drink three cups of coffee every morning.",
            [
                {"normalized_facet": "Caffeine Intake (mg/day)", "facet_type": "external_evidence", "conversation_observable": False, "scoring_definition": "Caffeine intake in mg/day"},
            ]
        ),
        (
            "TEST 5",
            "I consume 300 mg of caffeine every day.",
            [
                {"normalized_facet": "Caffeine Intake (mg/day)", "facet_type": "external_evidence", "conversation_observable": False, "scoring_definition": "Caffeine intake in mg/day"},
            ]
        ),
        (
            "TEST 6",
            "My brother works twelve hours every day.",
            [
                {"normalized_facet": "Hardworking", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Hardworking"},
            ]
        ),
        (
            "TEST 7",
            "The assignment was due Monday, but I submitted it Sunday.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Meeting deadlines"},
            ]
        ),
        (
            "TEST 8",
            "My assignment is due Friday, and I'm still working on it.",
            [
                {"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Meeting deadlines"},
            ]
        ),
    ]

    print("=" * 80)
    print("STRICT EVIDENCE-GROUNDED REASON GENERATION AUDIT")
    print("=" * 80)

    for label, conv, facets in cases:
        results = score_facets_batch(conv, facets, client=None)
        print(f"\n[{label}] \"{conv}\"")
        for r in results:
            score_str = f"{r['score']}/5" if r['score'] is not None else "null"
            print(f"  Facet:  {r['facet']}")
            print(f"  Status: {r['status'].upper()} (Score: {score_str}, Conf: {r['confidence']:.2f})")
            print(f"  Reason: {r['reason']}")

if __name__ == "__main__":
    test_grounded_reasons()
