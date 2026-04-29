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
  6. Tracks active learning efficiency and saves active_learning_report.csv
"""
import sys, pickle, time, sqlite3, warnings, datetime
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


UNCERTAINTY_THRESHOLD = 0.5


def query_active_learning_stats():
    """
    Pull active learning telemetry from predictions.db:
      - total tickets reviewed by humans
      - how many were high-uncertainty (uncertainty_score > threshold)
      - how many human corrections were rejected by the model (accepted=0)
    Returns a dict with all counts and the efficiency score.
    """
    stats = {
        "total_reviews": 0,
        "uncertain_reviews": 0,
        "corrections_made": 0,
        "efficiency_score": 0.0,
    }
    if not DB_PATH.exists():
        return stats
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Total tickets a human has reviewed
        c.execute("SELECT COUNT(*) FROM tickets WHERE reviewed = 1")
        stats["total_reviews"] = c.fetchone()[0]

        # Of those, how many were flagged as high-uncertainty
        c.execute(
            "SELECT COUNT(*) FROM tickets WHERE reviewed = 1 AND uncertainty_score > ?",
            (UNCERTAINTY_THRESHOLD,)
        )
        stats["uncertain_reviews"] = c.fetchone()[0]

        # Human corrections stored in feedback (accepted=0 means the agent
        # prediction was wrong and the human provided the correct label)
        try:
            c.execute("SELECT COUNT(*) FROM feedback WHERE accepted = 0")
            stats["corrections_made"] = c.fetchone()[0]
        except Exception:
            stats["corrections_made"] = 0  # table may not exist yet

        conn.close()
    except Exception as e:
        print(f"  [warn] Could not query active learning stats: {e}")
        return stats

    # Efficiency: fraction of reviews that actually needed a correction
    if stats["total_reviews"] > 0:
        stats["efficiency_score"] = round(
            stats["corrections_made"] / stats["total_reviews"], 4
        )
    return stats


def evaluate(pipe, X, y, task):
    scores = cross_val_score(pipe, X, y, cv=CV, scoring="f1_macro", n_jobs=-1)
    return scores.mean(), scores.std()


def run_retraining():
    print("\n" + "="*60)
    print("  FEEDBACK RETRAINING PIPELINE")
    print("="*60)

    # ── Active Learning Telemetry ─────────────────────────────────
    al_stats = query_active_learning_stats()
    print(f"\n  [Active Learning Telemetry]")
    print(f"    Total human reviews:       {al_stats['total_reviews']}")
    print(f"    High-uncertainty reviews:  {al_stats['uncertain_reviews']}")
    print(f"    Corrections made:          {al_stats['corrections_made']}")
    print(f"    Active learning efficiency: "
          f"{al_stats['efficiency_score']*100:.1f}% "
          f"(of reviews that led to a correction)")

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
    # Store per-task F1 values for the active learning report
    f1_by_task = {}  # {"Category": (base, aug, delta), "Priority": (base, aug, delta)}

    for task, col in [("Category", "category"), ("Priority", "priority")]:
        y_base = df_base[col].values
        y_aug  = df_augmented[col].values if col in df_augmented.columns else y_base

        # Baseline (original data only)
        base_f1, base_std = evaluate(make_pipeline(), X_base, y_base, task)

        # Augmented (original + feedback)
        aug_f1, aug_std = evaluate(make_pipeline(), X_aug, y_aug, task)

        delta = aug_f1 - base_f1
        f1_by_task[task] = (round(base_f1, 4), round(aug_f1, 4), round(delta, 4))

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

    # ── Save results table (existing) ─────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = REPORTS_DIR / "feedback_improvement.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    # ── Active Learning Report (append mode) ──────────────────────
    cat_base, cat_aug, cat_delta = f1_by_task.get("Category", (0.0, 0.0, 0.0))
    pri_base, pri_aug, pri_delta = f1_by_task.get("Priority", (0.0, 0.0, 0.0))

    al_row = pd.DataFrame([{
        "timestamp":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_cat_f1":    cat_base,
        "augmented_cat_f1":   cat_aug,
        "cat_f1_delta":       cat_delta,
        "baseline_pri_f1":    pri_base,
        "augmented_pri_f1":   pri_aug,
        "pri_f1_delta":       pri_delta,
        "total_reviews":      al_stats["total_reviews"],
        "uncertain_reviews":  al_stats["uncertain_reviews"],
        "corrections_made":   al_stats["corrections_made"],
        "efficiency_score":   al_stats["efficiency_score"],
    }])

    al_report_path = REPORTS_DIR / "active_learning_report.csv"
    if al_report_path.exists():
        al_row.to_csv(al_report_path, mode="a", header=False, index=False)
    else:
        al_row.to_csv(al_report_path, index=False)
    print(f"  Active learning report → {al_report_path}")
    print(f"  Efficiency score: {al_stats['efficiency_score']*100:.1f}%")

    # ── Existing feedback_improvement plot ────────────────────────
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

    # ── Active Learning grouped bar chart ─────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    al_tasks  = ["Category F1", "Priority F1"]
    base_vals = [cat_base, pri_base]
    aug_vals  = [cat_aug,  pri_aug]
    x2 = np.arange(len(al_tasks))
    w2 = 0.35

    bars_b = ax2.bar(x2 - w2/2, base_vals, w2, label="Baseline",
                     color="#7f8c8d", zorder=3)
    bars_a = ax2.bar(x2 + w2/2, aug_vals,  w2, label="Augmented (+ Feedback)",
                     color="#2980b9", zorder=3)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(al_tasks, fontsize=12)
    ax2.set_ylabel("Macro F1", fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("F1 improvement from active learning feedback",
                  fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)

    for b in list(bars_b) + list(bars_a):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    # Annotate deltas
    for i, (bv, av) in enumerate(zip(base_vals, aug_vals)):
        delta_v = av - bv
        color = "#27ae60" if delta_v >= 0 else "#c0392b"
        ax2.annotate(f"Δ {delta_v:+.3f}",
                     xy=(x2[i] + w2/2, av + 0.04),
                     ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    al_plot_path = REPORTS_DIR / "active_learning_improvement.png"
    plt.savefig(al_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Active learning plot → {al_plot_path}")

    print("\n" + "="*60)
    print("  ✅ Retraining complete")
    print("="*60)


if __name__ == "__main__":
    run_retraining()
