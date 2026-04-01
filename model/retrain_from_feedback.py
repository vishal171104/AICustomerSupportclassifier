"""
retrain_from_feedback.py
=========================
Novel contribution: Human-in-the-loop retraining from agent corrections.

Usage:
  python model/retrain_from_feedback.py

What it does:
  1. Loads accepted corrections from the feedback table in predictions.db
  2. Merges with original training data
  3. Retrains the LR pipeline, saves versioned model
  4. Evaluates improvement vs. baseline
  5. Logs delta metrics to reports/feedback_improvement.csv
"""
import sys, pickle, time, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from utils.preprocessing import clean_text

DATA_PATH = BASE_DIR / "data" / "tickets.csv"
DB_PATH   = BASE_DIR / "data" / "predictions.db"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports" / "publication"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CV   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                  stop_words="english", min_df=2)),
        ("clf",  LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=SEED, solver="lbfgs")),
    ])


def load_feedback():
    """Pull accepted agent corrections from SQLite."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        feedback = pd.read_sql_query(
            "SELECT description, corrected_category, corrected_priority "
            "FROM feedback WHERE accepted = 1", conn)
        conn.close()
        return feedback
    except Exception:
        return pd.DataFrame()


def evaluate(pipe, X, y, task):
    scores = cross_val_score(pipe, X, y, cv=CV, scoring="f1_macro", n_jobs=-1)
    return scores.mean(), scores.std()


def run_retraining():
    print("\n" + "="*60)
    print("  FEEDBACK RETRAINING PIPELINE")
    print("="*60)

    # ── Load base data ────────────────────────────────────────────
    df_base = pd.read_csv(DATA_PATH)
    df_base["clean_text"] = df_base["description"].fillna("").apply(clean_text)

    # ── Load feedback ─────────────────────────────────────────────
    feedback = load_feedback()
    n_feedback = len(feedback)
    print(f"  Agent corrections available: {n_feedback}")

    if n_feedback > 0:
        feedback["clean_text"] = feedback.get(
            "description", pd.Series([""] * n_feedback)).fillna("").apply(clean_text)
        if "corrected_category" in feedback.columns:
            feedback = feedback.rename(columns={
                "corrected_category": "category",
                "corrected_priority": "priority"
            })
        df_augmented = pd.concat([df_base, feedback], ignore_index=True)
    else:
        df_augmented = df_base.copy()
        print("  No feedback found — demonstrating pipeline with base data.\n"
              "  (In production, corrections come from dashboard ✏️ buttons)")

    X_base = df_base["clean_text"].values
    X_aug  = df_augmented["clean_text"].values

    results = []

    for task, col in [("Category", "category"), ("Priority", "priority")]:
        y_base = df_base[col].values
        y_aug  = df_augmented[col].values if col in df_augmented.columns else y_base

        # Baseline (original data only)
        base_f1, base_std = evaluate(make_pipeline(), X_base, y_base, task)

        # Augmented (original + feedback)
        aug_f1, aug_std = evaluate(make_pipeline(), X_aug, y_aug, task)

        delta = aug_f1 - base_f1
        print(f"\n  [{task}]")
        print(f"    Baseline  F1 (5-CV): {base_f1:.4f} ± {base_std:.4f}")
        print(f"    Augmented F1 (5-CV): {aug_f1:.4f} ± {aug_std:.4f}")
        print(f"    Delta:               {delta:+.4f}")

        results.append({
            "Task": task,
            "Baseline F1": round(base_f1, 4),
            "Baseline Std": round(base_std, 4),
            "Augmented F1": round(aug_f1, 4),
            "Augmented Std": round(aug_std, 4),
            "Delta F1": round(delta, 4),
            "Feedback Samples": n_feedback,
        })

        # Save retrained model
        pipe_full = make_pipeline()
        pipe_full.fit(X_aug, y_aug)
        v2_path = MODEL_DIR / f"{col}_pipeline_v2.pkl"
        with open(v2_path, "wb") as f:
            pickle.dump(pipe_full, f)
        print(f"    ✅ Model v2 saved → {v2_path.name}")

    # ── Save results table ────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = REPORTS_DIR / "feedback_improvement.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    tasks = results_df["Task"].values
    x = np.arange(len(tasks))
    w = 0.35
    bars1 = ax.bar(x - w/2, results_df["Baseline F1"], w, label="Baseline",
                   color="#95a5a6", yerr=results_df["Baseline Std"], capsize=5)
    bars2 = ax.bar(x + w/2, results_df["Augmented F1"], w, label="+ Feedback",
                   color="#2ecc71", yerr=results_df["Augmented Std"], capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0.5, 1.05)
    ax.set_title("Baseline vs. Feedback-Augmented Model", fontweight="bold")
    ax.legend()
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feedback_improvement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved → feedback_improvement.png")

    print("\n" + "="*60)
    print("  ✅ Retraining complete")
    print("="*60)


if __name__ == "__main__":
    run_retraining()
