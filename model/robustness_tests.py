import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text
from model.pipelines import create_pipeline

DATA_PATH = BASE_DIR / "data" / "tickets.csv"
REPORTS_DIR = BASE_DIR / "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["description"].fillna("").apply(clean_text)
    return df

def run_shuffle_test():
    """
    Step 1: Shuffle Test
    Randomly shuffle category labels and retrain.
    If accuracy drops to ~33% (for 3 classes), then no leakage.
    If it remains high -> leakage exists.
    """
    print("\n" + "="*50)
    print("STEP 1: SHUFFLE TEST (LEAKAGE DETECTION)")
    print("="*50)
    
    df = load_data()
    orig_labels = df["category"].copy()
    
    # Shuffle labels
    df["category"] = df["category"].sample(frac=1, random_state=42).values
    
    X = df["clean_text"]
    y = df["category"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model on shuffled labels...")
    model = create_pipeline("svm")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy with Shuffled Labels: {acc:.4f}")
    print(f"Random Guess Baseline (3 classes): ~0.3333")
    
    if acc < 0.40:
        print("\nRESULT: [PASS] - Accuracy is near random. No significant label-leakage found.")
    else:
        print("\nRESULT: [FAIL] - Accuracy remains high despite shuffled labels! Possible leakage in features.")
    
    return acc

def run_adversarial_test():
    """
    Step 2: Harder Test Split / Adversarial Examples
    Test the model on hand-crafted ambiguous/cross-category examples.
    """
    print("\n" + "="*50)
    print("STEP 2: ADVERSARIAL TEST (AMBIGUITY HANDLING)")
    print("="*50)
    
    df = load_data()
    X = df["clean_text"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = create_pipeline("svm")
    model.fit(X_train, y_train)
    
    adversarial_samples = [
        {
            "desc": "Payment portal login fails due to SSL certificate.",
            "expected": "Technical",
            "reason": "Mentions 'Payment' (Billing) and 'Login' (Account) but core issue is 'SSL certificate' (Technical)."
        },
        {
            "desc": "I am unable to see my invoice in the dashboard, it keeps timing out.",
            "expected": "Billing",
            "reason": "Mentions 'Dashboard/Timeout' (Technical) but objective is 'Invoice' (Billing)."
        },
        {
            "desc": "My profile settings shows an incorrect billing address and I can't update it.",
            "expected": "Account",
            "reason": "Mentions 'Billing' but core task is 'Profile settings' (Account)."
        }
    ]
    
    print(f"{'Description':<60} | {'Expected':<10} | {'Predicted':<10} | {'Status'}")
    print("-" * 100)
    
    adv_clean = [clean_text(s["desc"]) for s in adversarial_samples]
    preds = model.predict(adv_clean)
    
    correct = 0
    for i, sample in enumerate(adversarial_samples):
        pred = preds[i]
        status = "CORRECT" if pred == sample["expected"] else "WRONG"
        if status == "CORRECT": correct += 1
        print(f"{sample['desc'][:60]:<60} | {sample['expected']:<10} | {pred:<10} | {status}")

    acc = correct / len(adversarial_samples)
    print(f"\nAdversarial Accuracy: {acc:.4f}")
    if acc < 1.0:
        print("RESULT: [REALISTIC] - Model captures complexity but struggles with deep ambiguity as expected.")
    else:
        print("RESULT: [SUSPICIOUS] - Model got 100% on adversarial cases. Might be overfitted to specific keywords.")

def run_noise_test():
    """
    Step 3: Add Noise & Cross-category words
    """
    print("\n" + "="*50)
    print("STEP 3: NOISE & CROSS-CATEGORY INFLUENCE")
    print("="*50)
    
    df = load_data()
    X = df["clean_text"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = create_pipeline("svm")
    model.fit(X_train, y_train)
    
    noise_samples = [
        "Profile shows invoice mismatch and login failure.", # Mixed
        "I need help with everything, the system is down and my card failed.", # Mixed
        "The thingy is not doing the stuff.", # No info
    ]
    
    print(f"{'Noisy Text':<60} | {'Prediction':<10} | {'Confidence'}")
    print("-" * 100)
    
    noise_clean = [clean_text(t) for t in noise_samples]
    preds = model.predict(noise_clean)
    probs = model.predict_proba(noise_clean)
    
    for i, text in enumerate(noise_samples):
        pred = preds[i]
        conf = np.max(probs[i])
        print(f"{text[:60]:<60} | {pred:<10} | {conf:.4f}")

if __name__ == "__main__":
    run_shuffle_test()
    run_adversarial_test()
    run_noise_test()
