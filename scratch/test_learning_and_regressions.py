import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import score_facets_batch

def run_tests():
    print("=" * 80)
    print("EVALUATION OF ATTITUDE TOWARD LEARNING & ALL 9 REGRESSION TESTS")
    print("=" * 80)

    # 1. Main Failure Case
    conv_main = "I enjoy learning new things. Whenever I don't understand a topic, I spend extra time studying it until I understand it."
    facets_main = [
        {"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Measures enthusiasm for learning and deliberate study to understand."},
        {"normalized_facet": "Self-improvement", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Adapting methods after failure."},
        {"normalized_facet": "Learning Style", "facet_type": "biographical", "conversation_observable": False, "scoring_definition": "Sensory learning modality."},
        {"normalized_facet": "Learning Through Movement", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Kinesthetic learning."},
        {"normalized_facet": "Intellect", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Abstract intellectual reasoning."},
        {"normalized_facet": "Information Retention", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Memory retention capacity."},
        {"normalized_facet": "Time Outdoors/day (h)", "facet_type": "external_evidence", "conversation_observable": False, "scoring_definition": "Outdoor hours."},
    ]
    res_main = score_facets_batch(conv_main, facets_main, client=None)
    print(f"\n[MAIN TEST] \"{conv_main}\"")
    for r in res_main:
        print(f"  {r['facet']:<32} -> {r['status'].upper()} (Score: {r['score']}, Conf: {r['confidence']:.2f})")
        print(f"    Reason: {r['reason']}")

    # 2. Semantic Variations of Learning
    variations = [
        "I love learning new things.",
        "I like discovering new topics.",
        "I enjoy studying difficult subjects.",
        "When I don't understand something, I keep studying until I get it.",
        "I actively look for opportunities to learn.",
        "I spend extra time learning things I don't understand.",
        "Learning is something I genuinely enjoy.",
        "I always try to understand a topic instead of memorizing it."
    ]
    print("\n" + "=" * 80)
    print("SEMANTIC VARIATIONS TEST (Attitude Toward Learning)")
    print("=" * 80)
    for v in variations:
        res_v = score_facets_batch(v, [{"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Attitude toward learning"}], client=None)[0]
        print(f"  \"{v}\"\n    -> {res_v['status'].upper()} (Score: {res_v['score']}) | Reason: {res_v['reason']}")

    # 3. Target 9 Regression Cases
    print("\n" + "=" * 80)
    print("NINE REGRESSION TESTS")
    print("=" * 80)
    tests = [
        ("TEST 1 (Learning Attitude)", "I enjoy learning new things. Whenever I don't understand a topic, I spend extra time studying it until I understand it.", [{"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True}]),
        ("TEST 2 (Perseverance Exam)", "I failed my exam twice, but I kept studying every day and passed on my third attempt.", [{"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True}, {"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True}]),
        ("TEST 3 (Negative Perseverance)", "I failed the exam once and decided not to try again.", [{"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True}, {"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True}]),
        ("TEST 4 (Caffeine Cups)", "I drink three cups of coffee every morning.", [{"normalized_facet": "Caffeine Intake (mg/day)", "facet_type": "external_evidence", "conversation_observable": False}]),
        ("TEST 5 (Caffeine 300mg)", "I consume 300 mg of caffeine every day.", [{"normalized_facet": "Caffeine Intake (mg/day)", "facet_type": "external_evidence", "conversation_observable": False}]),
        ("TEST 6 (Deadline Early)", "The assignment was due Monday, but I submitted it Sunday.", [{"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True}]),
        ("TEST 7 (Deadline In Progress)", "My assignment is due Friday and I'm still working on it.", [{"normalized_facet": "Meeting Deadlines", "facet_type": "conversation_observable", "conversation_observable": True}]),
        ("TEST 8 (Yoga Discipline)", "I practice yoga for five hours every week.", [{"normalized_facet": "Hindu Spiritual Metric: Yoga Discipline Hours / Week", "facet_type": "external_evidence", "conversation_observable": False}, {"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True}]),
        ("TEST 9 (Third-Party Brother)", "My brother works twelve hours every day.", [{"normalized_facet": "Hardworking", "facet_type": "conversation_observable", "conversation_observable": True}]),
    ]

    for label, conv, facet_list in tests:
        print(f"\n[{label}] \"{conv}\"")
        for r in score_facets_batch(conv, facet_list, client=None):
            print(f"  {r['facet']:<45} -> {r['status'].upper()} (Score: {r['score']})")
            print(f"    Reason: {r['reason']}")

if __name__ == "__main__":
    run_tests()
