import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline

conv = "I failed the interview three times, but I kept preparing after every rejection. I practiced my answers, asked for feedback, and applied again until I finally got the job."
res = run_pipeline(conv, conversation_id="interview_test")

print(f"Total retrieved: {res['num_facets_retrieved']}")
print(f"Total results: {res['num_results']}")
print(f"Validation errors: {res['num_validation_errors']}")

for r in res['results']:
    if r['status'] == 'scored':
        print(f"  [SCORED] {r['facet']} -> Score: {r['score']}/5 (Conf: {r['confidence']}) | Reason: {r['reason']}")
    elif r['status'] == 'not_observable':
        print(f"  [NOT_OBSERVABLE] {r['facet']} | Reason: {r['reason']}")
