import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.dummy import DummyClassifier

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text
from model.pipelines import create_pipeline

# Paths
REPORTS_DIR = BASE_DIR / "reports" / "advanced"
os.makedirs(REPORTS_DIR, exist_ok=True)
DATA_PATH = BASE_DIR / "data" / "tickets.csv"

def plot_roc_pr_curves(X_test, y_test, pipeline, model_name, task_name):
    """
    Plots ROC and PR curves for multi-class tasks.
    """
    classes = sorted(y_test.unique())
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Get probabilities
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)
    else:
        # Fallback for models without predict_proba if any
        return

    # ROC Curve
    plt.figure(figsize=(12, 5))
    
    # ROC
    plt.subplot(1, 2, 1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'ROC {classes[i]} (AUC = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} ({task_name})')
    plt.legend(loc="lower right")

    # PR Curve
    plt.subplot(1, 2, 2)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], label=f'PR {classes[i]} (AP = {average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {model_name} ({task_name})')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{task_name.lower()}_{model_name.lower()}_curves.png")
    plt.close()

def get_per_class_metrics(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose().iloc[:-3, :3] # Only classes, only P, R, F1
    return df_report

def run_phase1():
    print("🚀 Starting Phase 1: Metrics Explosion")
    
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["description"].fillna("").apply(clean_text)
    
    # Tasks
    tasks = {
        "Category": {"y": df["category"], "labels": sorted(df["category"].unique())},
        "Priority": {"y": df["priority"], "labels": ["Low", "Medium", "High", "Critical"]}
    }
    
    models_to_test = ["svm", "logreg", "nb", "ensemble"]
    
    # 1. & 4. Per-class tables and CV scores
    all_cv_results = []
    
    for task_name, task_data in tasks.items():
        print(f"\nProcessing {task_name}...")
        X = df["clean_text"]
        y = task_data["y"]
        labels = task_data["labels"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        per_class_summary = []
        
        for m_type in models_to_test:
            print(f"  Evaluating {m_type}...")
            pipe = create_pipeline(m_type)
            
            # Cross-validation with std
            cv_scores = cross_val_score(pipe, X, y, cv=5)
            all_cv_results.append({
                "Task": task_name,
                "Model": m_type.upper(),
                "CV_Mean": np.mean(cv_scores),
                "CV_Std": np.std(cv_scores)
            })
            
            # Fit and evaluate
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            # Per-class table data
            report_df = get_per_class_metrics(y_test, y_pred, labels)
            report_df['Model'] = m_type.upper()
            per_class_summary.append(report_df)
            
            # 2. ROC/PR Curves for Priority (only for main models)
            if task_name == "Priority":
                plot_roc_pr_curves(X_test, y_test, pipe, m_type, task_name)
            
            # 3. Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix: {m_type.upper()} ({task_name})")
            plt.savefig(REPORTS_DIR / f"{task_name.lower()}_{m_type.lower()}_cm.png")
            plt.close()

        # Combine per-class metrics for this task
        combined_report = pd.concat(per_class_summary)
        combined_report.to_csv(REPORTS_DIR / f"{task_name.lower()}_per_class_metrics.csv")
        print(f"  Saved per-class metrics for {task_name}")

    # 4. Save CV Results table
    df_cv = pd.DataFrame(all_cv_results)
    df_cv.to_csv(REPORTS_DIR / "cross_validation_results.csv", index=False)
    print("\n✅ CV Results saved.")
    print(df_cv.to_string(index=False))

    # 5. Baseline Comparison Table
    print("\nComputing Baselines...")
    baseline_results = []
    for task_name, task_data in tasks.items():
        X = df["clean_text"]
        y = task_data["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Random
        random_clf = DummyClassifier(strategy="uniform", random_state=42)
        random_clf.fit(X_train, y_train)
        random_acc = accuracy_score(y_test, random_clf.predict(X_test))
        
        # Majority
        majority_clf = DummyClassifier(strategy="most_frequent", random_state=42)
        majority_clf.fit(X_train, y_train)
        majority_acc = accuracy_score(y_test, majority_clf.predict(X_test))
        
        # Human (Target/Goal)
        human_acc = 0.88 # Estimated high-quality human triage
        
        # Best model for this task from CV
        best_model_acc = df_cv[df_cv["Task"] == task_name]["CV_Mean"].max()
        
        baseline_results.append({
            "Task": task_name,
            "Random": round(random_acc, 4),
            "Majority": round(majority_acc, 4),
            "Best ML": round(best_model_acc, 4),
            "Human (Est.)": human_acc
        })
    
    df_baselines = pd.DataFrame(baseline_results)
    df_baselines.to_csv(REPORTS_DIR / "baseline_comparison.csv", index=False)
    print("\n✅ Baseline Comparison:")
    print(df_baselines.to_string(index=False))

if __name__ == "__main__":
    run_phase1()
