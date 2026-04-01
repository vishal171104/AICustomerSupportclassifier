import sys
from pathlib import Path
import pickle

BASE_DIR = Path("/Users/vishalsi/CustomerSupportTicketAI")
try:
    with open(BASE_DIR / "model" / "category_pipeline.pkl", "rb") as f:
        cat_pipe = pickle.load(f)
    with open(BASE_DIR / "model" / "priority_pipeline.pkl", "rb") as f:
        pri_pipe = pickle.load(f)
except Exception as e:
    print(f"Failed to load pipelines: {e}")
    sys.exit(1)

test_sentences = [
    ("the next page is loading for a long time, shows server error and error 404 not found", "High"),
    ("system is completely down, no users can access the platform", "Critical"),
    ("how do I reset my password", "Low"),
    ("the dashboard loads slowly sometimes", "Medium")
]

all_passed = True
for text, expected in test_sentences:
    pri = pri_pipe.predict([text])[0]
    result = "PASS" if pri == expected else "FAIL"
    if result == "FAIL":
        all_passed = False
    print(f"TEXT: {text}")
    print(f"PREDICTED: {pri} | EXPECTED: {expected} | {result}\n")

if all_passed:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED.")
