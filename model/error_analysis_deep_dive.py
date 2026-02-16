import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text
from model.pipelines import create_pipeline

# Paths
REPORTS_DIR = BASE_DIR / "reports" / "error_analysis"
os.makedirs(REPORTS_DIR, exist_ok=True)
DATA_PATH = BASE_DIR / "data" / "tickets.csv"

def analyze_error_patterns():
    print("🚀 Starting Phase 4: Error Analysis Deep Dive")
    
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["description"].fillna("").apply(clean_text)
    
    # Analyze Priority since it had lower accuracy (approx 51%)
    X = df["clean_text"]
    y = df["priority"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipe = create_pipeline("svm", ngram_range=(1,1))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    
    results_df = pd.DataFrame({
        "Text": X_test.values,
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Confidence": max_probs
    })
    
    errors = results_df[results_df["Actual"] != results_df["Predicted"]].copy()
    
    # 1. Misclassification Taxonomy
    # Define rules to categorize errors
    def categorize_error(row):
        text = row["Text"].lower()
        # Semantic Ambiguity: Words that belong to multiple categories or contexts
        if "login" in text and ("bill" in text or "payment" in text):
            return "Semantic Ambiguity (Billing vs Tech)"
        # Keyword Traps: Sarcasm or distracting keywords
        if "urgent" in text and row["Actual"] == "Low":
            return "Keyword Trap (Sarcastic/Minor 'Urgent')"
        if "not" in text or "don't" in text or "can't" in text:
            return "Negation/Context Loss"
        # Low Confidence
        if row["Confidence"] < 0.4:
            return "Low Confidence / Noise"
        return "Miscellaneous / Out-of-vocab"

    errors["Error_Type"] = errors.apply(categorize_error, axis=1)
    
    taxonomy_table = errors["Error_Type"].value_counts().reset_index()
    taxonomy_table.columns = ["Error Pattern", "Count"]
    taxonomy_table.to_csv(REPORTS_DIR / "error_taxonomy.csv", index=False)
    
    print("\n✅ Error Taxonomy Summary:")
    print(taxonomy_table.to_string(index=False))

    # 2. Confidence Distribution Histograms (Item 7 Requirement)
    print("\n  Generating Confidence Distribution Histograms...")
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df[results_df["Actual"] == results_df["Predicted"]]["Confidence"], label="Correct", color="green", kde=True, alpha=0.5)
    sns.histplot(results_df[results_df["Actual"] != results_df["Predicted"]]["Confidence"], label="Incorrect", color="red", kde=True, alpha=0.5)
    plt.title("Confidence Distribution: Correct vs Incorrect Predictions")
    plt.xlabel("Model Confidence (Max Probability)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(REPORTS_DIR / "confidence_histogram.png")
    plt.close()

    # 3. Export specific examples for the report
    examples = errors.sort_values("Confidence", ascending=False).head(10)
    examples.to_csv(REPORTS_DIR / "misclassification_samples.csv", index=False)
    print(f"  Saved 10 high-confidence misclassifications to CSV.")

    # 4. Adversarial Response Pattern (Item 4 Requirement)
    print("  Simulating Adversarial Response Patterns...")
    ads_texts = [
        "THIS IS URGENT BUT ACTUALLY FINE", # Keyword trap
        "system working well but login button is green instead of blue", # Sentiment vs Triage
        "i will cancel my subscription if this is not fixed in 5 minutes", # Hard pressure
        "p ay m e n t is fail ing", # Character spacing
    ]
    ads_preds = pipe.predict(ads_texts)
    ads_probs = np.max(pipe.predict_proba(ads_texts), axis=1)
    
    ads_df = pd.DataFrame({"Adversarial_Input": ads_texts, "Prediction": ads_preds, "Confidence": ads_probs})
    ads_df.to_csv(REPORTS_DIR / "adversarial_patterns.csv", index=False)
    
    print("\n✅ Error Analysis complete.")

if __name__ == "__main__":
    analyze_error_patterns()
