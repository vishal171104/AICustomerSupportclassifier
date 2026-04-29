"""
publication_eval.py
====================
Section 5 of the paper: "Experimental Evaluation"
Produces:
  - reports/publication/model_comparison.csv      ← main results table
  - reports/publication/model_comparison.tex       ← LaTeX-ready table
  - reports/publication/mcnemar_tests.csv          ← statistical significance
  - reports/publication/bootstrap_ci.csv           ← 95% CI per model
  - reports/publication/calibration_ece.csv        ← ECE scores

Run:  python model/publication_eval.py
"""

import sys, os, time, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from utils.preprocessing import clean_text

REPORTS_DIR = BASE_DIR / "reports" / "publication"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = BASE_DIR / "data" / "tickets.csv"

SEED = 42
CV_FOLDS = 5
N_BOOTSTRAP = 1000

# ─── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["clean_text"] = df["description"].fillna("").apply(clean_text)
X = df["clean_text"].values
y_cat = df["category"].values
y_pri = df["priority"].values


# ─── Model Zoo ───────────────────────────────────────────────────────────────
def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                  stop_words="english", min_df=2)),
        ("clf", clf)
    ])

MODELS = {
    "Logistic Regression (LR)": make_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=1000,
                           random_state=SEED, solver="lbfgs")),
    "Multinomial NB": make_pipeline(MultinomialNB()),
    "Linear SVM": make_pipeline(
        CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000, random_state=SEED))),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=100, class_weight="balanced",
                               random_state=SEED, n_jobs=-1)),
}


# ─── Cross-Validated Evaluation ──────────────────────────────────────────────
def evaluate_models(X, y, task_name):
    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    predictions = {}   # store per-sample OOF preds for McNemar

    for name, pipe in MODELS.items():
        t0 = time.time()
        oof_preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        elapsed = time.time() - t0

        acc  = accuracy_score(y, oof_preds)
        f1   = f1_score(y, oof_preds, average="macro", zero_division=0)
        prec = precision_score(y, oof_preds, average="macro", zero_division=0)
        rec  = recall_score(y, oof_preds, average="macro", zero_division=0)

        # Bootstrap CI on accuracy
        boot_accs = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(len(y), len(y), replace=True)
            boot_accs.append(accuracy_score(y[idx], oof_preds[idx]))
        ci_lo = np.percentile(boot_accs, 2.5)
        ci_hi = np.percentile(boot_accs, 97.5)

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "95% CI": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "CV Time (s)": round(elapsed, 2),
        })
        predictions[name] = oof_preds

        print(f"  {name:<28} Acc={acc:.4f}  F1={f1:.4f}  95%CI=[{ci_lo:.4f},{ci_hi:.4f}]  {elapsed:.1f}s")

    results_df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    results_df.to_csv(REPORTS_DIR / f"model_comparison_{task_name.lower().replace(' ','_')}.csv", index=False)

    # LaTeX table
    latex = results_df[["Model", "Accuracy", "Macro F1", "Precision", "Recall", "95% CI"]].to_latex(
        index=False, float_format="%.4f",
        caption=f"Model Comparison — {task_name} Task (5-fold CV)",
        label=f"tab:results_{task_name.lower().replace(' ','_')}",
        escape=False
    )
    with open(REPORTS_DIR / f"model_comparison_{task_name.lower().replace(' ','_')}.tex", "w") as f:
        f.write(latex)

    return results_df, predictions, y


# ─── McNemar's Test ──────────────────────────────────────────────────────────
def mcnemar_tests(predictions, y, task_name):
    print(f"\n  McNemar's Test ({task_name}):")
    model_names = list(predictions.keys())
    rows = []
    for m1, m2 in itertools.combinations(model_names, 2):
        p1 = predictions[m1]
        p2 = predictions[m2]
        c1 = (p1 == y)
        c2 = (p2 == y)
        # Contingency: both correct / m1 only / m2 only / both wrong
        b = np.sum(c1 & ~c2)  # m1 correct, m2 wrong
        c = np.sum(~c1 & c2)  # m1 wrong, m2 correct
        table = [[np.sum(c1 & c2), b], [c, np.sum(~c1 & ~c2)]]
        try:
            result = mcnemar(table, exact=False, correction=True)
            pval = result.pvalue
        except Exception:
            pval = float("nan")
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
        rows.append({"Model A": m1, "Model B": m2, "p-value": round(pval, 4), "Significance": sig})
        print(f"    {m1} vs {m2}: p={pval:.4f} {sig}")
    df_mc = pd.DataFrame(rows)
    df_mc.to_csv(REPORTS_DIR / f"mcnemar_{task_name.lower().replace(' ','_')}.csv", index=False)
    return df_mc


# ─── ECE (Expected Calibration Error) ────────────────────────────────────────
def compute_ece(probs, y_bin, n_bins=10):
    """Lower is better. 0 = perfectly calibrated."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_bin[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / len(probs)


def calibration_analysis(X, y, task_name):
    print(f"\n  Calibration Analysis ({task_name}):")
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    classes = np.unique(y)
    rows = []
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5), sharey=True)

    for ax, (name, pipe) in zip(axes, MODELS.items()):
        pipe_copy = Pipeline(pipe.steps)  # fresh copy
        pipe_copy.fit(X_tr, y_tr)
        try:
            probs = pipe_copy.predict_proba(X_te)
        except Exception:
            rows.append({"Model": name, "ECE": "N/A"})
            ax.set_title(name[:20])
            continue

        # Macro ECE across classes
        ece_total = 0.0
        for i, cls in enumerate(classes):
            y_bin = (y_te == cls).astype(int)
            ece_total += compute_ece(probs[:, i], y_bin)
            # Plot calibration curve
            from sklearn.calibration import calibration_curve
            frac_pos, mean_pred = calibration_curve(y_bin, probs[:, i], n_bins=8)
            ax.plot(mean_pred, frac_pos, marker="s", label=cls, linewidth=1.5)

        ece = ece_total / len(classes)
        rows.append({"Model": name, "ECE": round(ece, 4)})
        print(f"    {name:<28} ECE={ece:.4f}")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_title(f"{name[:20]}\n(ECE={ece:.4f})", fontsize=9)
        ax.set_xlabel("Mean Predicted Probability")
        ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel("Fraction of Positives")
    plt.suptitle(f"Reliability Diagrams — {task_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"reliability_diagrams_{task_name.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    df_ece = pd.DataFrame(rows)
    df_ece.to_csv(REPORTS_DIR / f"ece_{task_name.lower().replace(' ','_')}.csv", index=False)
    return df_ece


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PUBLICATION EVALUATION SUITE")
    print("  Output →", REPORTS_DIR)
    print("="*60)

    # Category task
    res_cat, preds_cat, y_cat_arr = evaluate_models(X, y_cat, "Category")
    mcnemar_tests(preds_cat, y_cat_arr, "Category")
    calibration_analysis(X, y_cat, "Category")

    # Priority task
    res_pri, preds_pri, y_pri_arr = evaluate_models(X, y_pri, "Priority")
    mcnemar_tests(preds_pri, y_pri_arr, "Priority")
    calibration_analysis(X, y_pri, "Priority")

    print("\n" + "="*60)
    print("  ✅ All publication artifacts saved to:", REPORTS_DIR)
    print("="*60)
