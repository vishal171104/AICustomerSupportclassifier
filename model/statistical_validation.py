import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve, CalibrationDisplay

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text
from model.pipelines import create_pipeline

# Paths
REPORTS_DIR = BASE_DIR / "reports" / "statistical"
os.makedirs(REPORTS_DIR, exist_ok=True)
DATA_PATH = BASE_DIR / "data" / "tickets.csv"

def run_statistical_suite():
    print("🚀 Starting Phase 5: Statistical Validation Suite")
    
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["description"].fillna("").apply(clean_text)
    
    X = df["clean_text"]
    y = df["priority"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Feature Importance (TF-IDF Top Terms)
    print("  Computing Permutation Feature Importance...")
    pipe = create_pipeline("svm", ngram_range=(1,1)) # Based on ablation, unigram is better
    pipe.fit(X_train, y_train)
    
    # We need to transform X_test manually for permutation importance if we want to see terms
    # Or just use the pipe's internal steps
    tfidf = pipe.named_steps['tfidf']
    X_test_transformed = tfidf.transform(X_test).toarray()
    clf = pipe.named_steps['clf']
    
    result = permutation_importance(clf, X_test_transformed, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    feature_names = tfidf.get_feature_names_out()
    sorted_idx = result.importances_mean.argsort()[-20:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names[sorted_idx], result.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Top 20 Technical/Priority Predictors (TF-IDF Terms)")
    plt.savefig(REPORTS_DIR / "feature_importance.png")
    plt.close()
    
    # 2. Confidence Calibration Plots
    print("  Generating Calibration Plots...")
    # Multi-class calibration plot (one vs rest)
    classes = sorted(y.unique())
    y_test_bin = pd.get_dummies(y_test)
    
    plt.figure(figsize=(10, 10))
    for i, cls in enumerate(classes):
        prob_pos = pipe.predict_proba(X_test)[:, i]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_bin.iloc[:, i], prob_pos, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{cls}")
        
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Reliability Diagram (Calibration Plot)")
    plt.legend(loc="lower right")
    plt.savefig(REPORTS_DIR / "calibration_plot.png")
    plt.close()

    # 3. Bootstrap Confidence Intervals
    print("  Computing Bootstrap Confidence Intervals (Accuracy)...")
    n_iterations = 1000
    stats = []
    y_pred = pipe.predict(X_test)
    for _ in range(n_iterations):
        indices = np.random.choice(range(len(y_test)), size=len(y_test), replace=True)
        if len(np.unique(y_test.iloc[indices])) < 2:
            continue
        acc = accuracy_score(y_test.iloc[indices], y_pred[indices])
        stats.append(acc)
    
    confidence = 0.95
    lower = np.percentile(stats, (1 - confidence) / 2 * 100)
    upper = np.percentile(stats, (1 + confidence) / 2 * 100)
    print(f"  95% CI for Accuracy: [{lower:.4f}, {upper:.4f}]")
    
    with open(REPORTS_DIR / "statistical_summary.txt", "w") as f:
        f.write(f"Confidence Intervals for Accuracy (1000 iterations):\n")
        f.write(f"Lower Bound (2.5%): {lower:.4f}\n")
        f.write(f"Upper Bound (97.5%): {upper:.4f}\n")
        f.write(f"Mean: {np.mean(stats):.4f}\n")

    print(f"✅ Statistical suite complete. Artifacts saved in {REPORTS_DIR}")

if __name__ == "__main__":
    run_statistical_suite()
