"""
error_analysis_publication.py
==============================
Section 6 of the paper: "Error Analysis"
Produces:
  - reports/publication/error_analysis_category.md   ← human-readable report
  - reports/publication/error_analysis_priority.md
  - reports/publication/confusion_examples.csv        ← text examples per error cell
  - reports/publication/confidence_distribution.png   ← correct vs incorrect

Run:  python model/error_analysis_publication.py
"""
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from utils.preprocessing import clean_text

REPORTS_DIR = BASE_DIR / "reports" / "publication"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
SEED = 42


def make_lr_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                  stop_words="english", min_df=2)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=SEED, solver="lbfgs")),
    ])


def run_error_analysis(X, y, raw_texts, task_name):
    print(f"\n{'='*55}")
    print(f"  Error Analysis: {task_name}")
    print(f"{'='*55}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = make_lr_pipeline()
    oof_preds = cross_val_predict(pipe, X, y, cv=cv, method="predict", n_jobs=-1)
    oof_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    max_conf = oof_proba.max(axis=1)

    classes = sorted(np.unique(y))
    errors_mask = oof_preds != y

    # ── 1. Classification Report ─────────────────────────────────────────────
    report = classification_report(y, oof_preds, target_names=classes, zero_division=0)
    print(report)

    # ── 2. Confusion Matrix Plot ─────────────────────────────────────────────
    cm = confusion_matrix(y, oof_preds, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Counts",      "d"),
        (axes[1], cm_norm, "Normalised",  ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix ({title})")
    plt.suptitle(f"Logistic Regression — {task_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{task_name.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Confidence Distribution ───────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    bins = np.linspace(0, 1, 25)
    plt.hist(max_conf[~errors_mask], bins=bins, alpha=0.65, color="#2ecc71",
             label=f"Correct  (n={int((~errors_mask).sum())})", density=True)
    plt.hist(max_conf[errors_mask],  bins=bins, alpha=0.65, color="#e74c3c",
             label=f"Incorrect (n={int(errors_mask.sum())})",  density=True)
    plt.axvline(max_conf[errors_mask].mean(),  color="#c0392b", ls="--", lw=1.5,
                label=f"Mean incorrect conf = {max_conf[errors_mask].mean():.2f}")
    plt.xlabel("Max Predicted Probability (Confidence)")
    plt.ylabel("Density")
    plt.title(f"Confidence Distribution — {task_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confidence_dist_{task_name.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. Text Examples per Confusion Cell ──────────────────────────────────
    example_rows = []
    for true_cls in classes:
        for pred_cls in classes:
            if true_cls == pred_cls:
                continue
            mask = (y == true_cls) & (oof_preds == pred_cls)
            sample_texts = np.array(raw_texts)[mask]
            sample_conf  = max_conf[mask]
            if len(sample_texts) == 0:
                continue
            # Pick the highest-confidence mistake (most surprising to reviewer)
            top_idx = np.argsort(sample_conf)[::-1][:2]
            for i in top_idx:
                example_rows.append({
                    "Actual": true_cls,
                    "Predicted": pred_cls,
                    "Confidence": round(float(sample_conf[i]), 3),
                    "Text (first 120 chars)": str(sample_texts[i])[:120],
                })

    examples_df = pd.DataFrame(example_rows).sort_values("Confidence", ascending=False)
    examples_df.to_csv(REPORTS_DIR / f"confusion_examples_{task_name.lower().replace(' ','_')}.csv",
                       index=False)

    # ── 5. Low-Confidence Bucket Analysis ───────────────────────────────────
    thresholds = [0.5, 0.6, 0.7, 0.8]
    bucket_rows = []
    for thr in thresholds:
        mask_low = max_conf < thr
        err_in_bucket = errors_mask[mask_low].sum()
        total_in_bucket = mask_low.sum()
        bucket_rows.append({
            "Conf < Threshold": thr,
            "Tickets in Bucket": int(total_in_bucket),
            "Errors in Bucket": int(err_in_bucket),
            "Error Rate": f"{(err_in_bucket / total_in_bucket * 100):.1f}%" if total_in_bucket > 0 else "N/A"
        })
    low_conf_df = pd.DataFrame(bucket_rows)
    low_conf_df.to_csv(REPORTS_DIR / f"low_confidence_analysis_{task_name.lower().replace(' ','_')}.csv",
                       index=False)

    # ── 6. Markdown Report ───────────────────────────────────────────────────
    total_errors = int(errors_mask.sum())
    total = len(y)
    out = []
    out.append(f"# Error Analysis Report — {task_name}\n")
    out.append(f"**Model**: Logistic Regression + TF-IDF (1–2-grams, 5000 features)  ")
    out.append(f"**Data**: {total} samples, 5-fold cross-validation\n")
    out.append(f"## Overall Performance\n```\n{report}\n```\n")
    out.append(f"## Error Summary\n")
    out.append(f"- **Total errors**: {total_errors} / {total} = {total_errors/total*100:.2f}%\n")
    out.append(f"- **Mean confidence on incorrect predictions**: {max_conf[errors_mask].mean():.3f}\n")
    out.append(f"- **Mean confidence on correct predictions**: {max_conf[~errors_mask].mean():.3f}\n")
    out.append(f"\n## Low-Confidence Bucket Analysis\n")
    out.append(low_conf_df.to_markdown(index=False))
    out.append(f"\n\n## Top Confusion Pairs\n")
    out.append(examples_df.head(10).to_markdown(index=False))
    out.append(f"\n\n## Interpretation\n")
    for _, row in examples_df.head(5).iterrows():
        out.append(f"- **{row['Actual']} → {row['Predicted']}** "
                   f"(conf={row['Confidence']}): *\"{row['Text (first 120 chars)']}\"*\n")

    report_path = REPORTS_DIR / f"error_analysis_{task_name.lower().replace(' ','_')}.md"
    with open(report_path, "w") as f:
        f.write("\n".join(out))
    print(f"  ✅ Report saved → {report_path.name}")
    print(f"  Total errors: {total_errors}/{total} ({total_errors/total*100:.2f}%)")
    print(f"  Mean conf on errors: {max_conf[errors_mask].mean():.3f}")
    print(low_conf_df.to_string(index=False))


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["description"].fillna("").apply(clean_text)
    X = df["clean_text"].values
    raw = df["description"].fillna("").values

    run_error_analysis(X, df["category"].values, raw, "Category")
    run_error_analysis(X, df["priority"].values, raw, "Priority")
    print("\n✅ Error analysis complete.")
