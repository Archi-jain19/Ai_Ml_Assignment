import sys
sys.path.insert(0, ".")
from src.scoring import _heuristic_offline_score_batch

def run_tests():
    # Case 1: Yoga
    c1 = "I practice yoga for five hours every week."
    f1 = [
        {"normalized_facet": "Hindu Spiritual Metric: Yoga Discipline Hours / Week", "facet_type": "external_evidence", "conversation_observable": False, "scoring_definition": "Yoga hours"},
        {"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Learning attitude"},
        {"normalized_facet": "Hardworking", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Hardworking"},
        {"normalized_facet": "Self-improvement", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Self improvement"},
        {"normalized_facet": "Achievement Motivation", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Achievement"},
    ]
    r1 = {r["facet"]: r for r in _heuristic_offline_score_batch(c1, f1)}
    print("--- Case 1: 'I practice yoga for five hours every week.' ---")
    for k, v in r1.items():
        print(f"{k:50} -> {v['status']:22} Score: {v.get('score')} | {v['reason']}")

    # Case 2: 8 Hours Work
    c2 = "I work eight hours every day."
    f2 = [
        {"normalized_facet": "Hardworking", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Hardworking"},
        {"normalized_facet": "Hindu Spiritual Metric: Yoga Discipline Hours / Week", "facet_type": "external_evidence", "conversation_observable": False, "scoring_definition": "Yoga hours"},
        {"normalized_facet": "Achievement Motivation", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Achievement"},
        {"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Perseverance"},
    ]
    r2 = {r["facet"]: r for r in _heuristic_offline_score_batch(c2, f2)}
    print("\n--- Case 2: 'I work eight hours every day.' ---")
    for k, v in r2.items():
        print(f"{k:50} -> {v['status']:22} Score: {v.get('score')} | {v['reason']}")

    # Case 3: Failed Exam & Kept Studying
    c3 = "I failed my exam twice but kept studying every night until I passed."
    f3 = [
        {"normalized_facet": "Perseverance", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Perseverance"},
        {"normalized_facet": "Attitude Toward Learning", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Learning attitude"},
        {"normalized_facet": "Self-improvement", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Self improvement"},
        {"normalized_facet": "Hardworking", "facet_type": "conversation_observable", "conversation_observable": True, "scoring_definition": "Hardworking"},
    ]
    r3 = {r["facet"]: r for r in _heuristic_offline_score_batch(c3, f3)}
    print("\n--- Case 3: 'I failed my exam twice but kept studying every night until I passed.' ---")
    for k, v in r3.items():
        print(f"{k:50} -> {v['status']:22} Score: {v.get('score')} | {v['reason']}")

if __name__ == "__main__":
    run_tests()
