import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text

# Paths
REPORTS_DIR = BASE_DIR / "reports" / "ablation"
os.makedirs(REPORTS_DIR, exist_ok=True)
DATA_PATH = BASE_DIR / "data" / "tickets.csv"

def run_ablation():
    print("🚀 Starting Phase 3: Model Ablation Study")
    
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["description"].fillna("").apply(clean_text)
    
    X = df["clean_text"]
    y = df["priority"] # Use priority as it's the harder task
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ablation_results = []

    # 1. TF-IDF Ablation (N-grams)
    print("  Testing N-grams...")
    for ngram in [(1,1), (1,2), (1,3)]:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram, stop_words='english')),
            ("svc", SVC(kernel='linear', C=1.0, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        ablation_results.append({
            "Component": "N-gram Range",
            "Variant": str(ngram),
            "Accuracy": round(acc, 4)
        })

    # 2. Stopwords Ablation
    print("  Testing Stopwords...")
    for stop in [None, 'english']:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words=stop)),
            ("svc", SVC(kernel='linear', C=1.0, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        ablation_results.append({
            "Component": "Stopwords",
            "Variant": "ON" if stop else "OFF",
            "Accuracy": round(acc, 4)
        })

    # 3. SVM Kernel Ablation
    print("  Testing SVM Kernels...")
    for kernel in ['linear', 'rbf', 'poly']:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
            ("svc", SVC(kernel=kernel, C=1.0, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        ablation_results.append({
            "Component": "SVM Kernel",
            "Variant": kernel,
            "Accuracy": round(acc, 4)
        })

    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(REPORTS_DIR / "ablation_results.csv", index=False)
    print("\n✅ Ablation Study Results:")
    print(df_ablation.to_string(index=False))

    # Summary table as requested
    summary_table = df_ablation.pivot(index="Component", columns="Variant", values="Accuracy")
    summary_table.to_csv(REPORTS_DIR / "ablation_summary_table.csv")

if __name__ == "__main__":
    run_ablation()
