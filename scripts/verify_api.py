import requests
import json
import time
import subprocess
import os
from pathlib import Path

# Samples for verification - chosen to be "Not Simple"
test_samples = [
    {
        "description": "URGENT: The payment portal is behaving unexpectedly and my card was flagged. I need this checked now.",
        "note": "Clear Priority, Mixed Intent (Billing/Technical)"
    },
    {
        "description": "I noticed some inconsistencies in the navigation layer after the morning update. It feels degraded.",
        "note": "Technical, Low/Medium priority signals"
    },
    {
        "description": "My profile data is not accessible and it returns a generic error. This is a critical blocker for our Q1 billing cycle.",
        "note": "Ambiguous Category (Account/Technical), Critical priority"
    },
    {
        "description": "The latest summary shows a discrepancy. Why is my card info showing twice? No hurry on this one.",
        "note": "Billing, Low priority"
    },
    {
        "description": "SSL handshake failed in the gateway. Production is at a standstill! Help fast.",
        "note": "High Technical complexity, Critical priority"
    }
]

def run_verification():
    print("=== Model Confidence Verification Demo ===")
    
    url = "http://127.0.0.1:8000/predict"
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest #{i}: {sample['note']}")
        print(f"Input: \"{sample['description']}\"")
        
        try:
            response = requests.post(url, json={"description": sample["description"]})
            if response.status_code == 200:
                res = response.json()
                print(f"-> Predicted Category: {res['category']} ({res['category_confidence']*100:.1f}%)")
                print(f"-> Predicted Priority: {res['priority']} ({res['priority_confidence']*100:.1f}%)")
            else:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Connection Error: {str(e)}")
            print("Is the FastAPI server running? (uvicorn api.main:app --reload)")
            break

if __name__ == "__main__":
    run_verification()
