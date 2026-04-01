"""
Script to regenerate notebook/eda.ipynb with complete:
- Data preprocessing
- Feature engineering
- Label encoding + train/test split
- Logistic Regression model training
- Confusion Matrix visualization
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "notebook" / "eda.ipynb"

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🔬 Feature Engineering & Exploratory Data Analysis (EDA)\n",
                "## Customer Support Ticket Triage AI\n",
                "\n",
                "This notebook documents the **end-to-end data processing pipeline**: raw ticket text →  "
                "feature engineering → ML model training → confusion matrix evaluation.\n",
                "\n",
                "**Model Used**: Logistic Regression (TF-IDF + Unigrams/Bigrams, balanced class weights)"
            ]
        },
        # ── Cell 1: Imports ────────────────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay\n",
                "from sklearn.preprocessing import LabelEncoder\n",
                "from sklearn.feature_selection import chi2\n",
                "\n",
                "# ── Project path setup ────────────────────────────────\n",
                "BASE_DIR = Path(os.getcwd()).resolve().parent\n",
                "sys.path.append(str(BASE_DIR))\n",
                "\n",
                "from utils.preprocessing import clean_text\n",
                "\n",
                "# ── Plot aesthetics ───────────────────────────────────\n",
                "plt.style.use('ggplot')\n",
                "sns.set_palette('viridis')\n",
                "print('Imports OK')"
            ]
        },
        # ── Section 1 Header ───────────────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 📥 Section 1 — Data Loading & Initial Inspection\n",
                "We start by loading the raw CSV and checking for shape, dtypes, and missing values."
            ]
        },
        # ── Cell 2: Load data ──────────────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "DATA_PATH = BASE_DIR / 'data' / 'tickets.csv'\n",
                "df = pd.read_csv(DATA_PATH)\n",
                "\n",
                "print(f'Dataset shape : {df.shape}')\n",
                "print(f'Columns       : {list(df.columns)}')\n",
                "print('\\nMissing values per column:')\n",
                "print(df.isnull().sum())\n",
                "\n",
                "# ── Ensure no NaNs in the text column ────────────────\n",
                "df['description'] = df['description'].fillna('')\n",
                "\n",
                "df.head()"
            ]
        },
        # ── Cell 3: Class distribution ─────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "df['category'].value_counts().plot(kind='bar', ax=axes[0], color=sns.color_palette('viridis', 3))\n",
                "axes[0].set_title('Category Distribution')\n",
                "axes[0].set_xlabel('Category')\n",
                "axes[0].set_ylabel('Count')\n",
                "axes[0].tick_params(axis='x', rotation=20)\n",
                "\n",
                "priority_order = ['Low', 'Medium', 'High', 'Critical']\n",
                "priority_counts = df['priority'].value_counts().reindex(priority_order, fill_value=0)\n",
                "priority_counts.plot(kind='bar', ax=axes[1], color=sns.color_palette('rocket', 4))\n",
                "axes[1].set_title('Priority Distribution')\n",
                "axes[1].set_xlabel('Priority')\n",
                "axes[1].set_ylabel('Count')\n",
                "axes[1].tick_params(axis='x', rotation=20)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # ── Section 2 Header ───────────────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 🧹 Section 2 — Data Preprocessing\n",
                "### Step 1: Text Cleaning\n",
                "The `clean_text()` utility (in `utils/preprocessing.py`) applies:\n",
                "- Lowercase normalization\n",
                "- URL removal (`http\\S+`)\n",
                "- Digit stripping\n",
                "- Punctuation removal\n",
                "- Whitespace normalization"
            ]
        },
        # ── Cell 4: Clean text ─────────────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['clean_text'] = df['description'].apply(clean_text)\n",
                "\n",
                "print('Before cleaning:')\n",
                "print(df.iloc[0]['description'][:200])\n",
                "print('\\nAfter cleaning:')\n",
                "print(df.iloc[0]['clean_text'][:200])"
            ]
        },
        # ── Cell 5: Ticket length feature ─────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Structural Feature — Ticket Length\n",
                "`text_len` (number of characters in the cleaned description) acts as a simple structural proxy "
                "for ticket complexity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['text_len'] = df['clean_text'].apply(len)\n",
                "\n",
                "print(df.groupby('category')['text_len'].describe().round(1))\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "sns.boxplot(x='category', y='text_len', data=df, palette='viridis')\n",
                "plt.title('Ticket Length Distribution by Category')\n",
                "plt.xlabel('Category')\n",
                "plt.ylabel('Character Count (clean text)')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # ── Section 3 Header ───────────────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 🔢 Section 3 — Feature Engineering\n",
                "### Step 3: TF-IDF Vectorization\n",
                "We use **Term Frequency–Inverse Document Frequency** to weight token importance:\n",
                "- `ngram_range=(1, 2)` → captures unigrams and bigrams\n",
                "- `max_features=5000` → caps vocabulary to prevent overfitting\n",
                "- `stop_words='english'` → removes low-signal filler words\n",
                "- `min_df=2` → ignores tokens that appear fewer than 2 times"
            ]
        },
        # ── Cell 6: TF-IDF ────────────────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tfidf = TfidfVectorizer(\n",
                "    ngram_range=(1, 2),\n",
                "    max_features=5000,\n",
                "    stop_words='english',\n",
                "    min_df=2\n",
                ")\n",
                "X_tfidf = tfidf.fit_transform(df['clean_text'])\n",
                "\n",
                "print(f'TF-IDF Feature Matrix Shape: {X_tfidf.shape}')\n",
                "print(f'Vocabulary size            : {len(tfidf.vocabulary_)}')"
            ]
        },
        # ── Cell 7: Chi2 top features ─────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Top Discriminating Terms (Chi² Scoring)\n",
                "Chi² analysis identifies which TF-IDF features are most statistically associated with each category label."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_names = tfidf.get_feature_names_out()\n",
                "\n",
                "for category in sorted(df['category'].unique()):\n",
                "    labels = (df['category'] == category).astype(int)\n",
                "    chi2_scores = chi2(X_tfidf, labels)[0]\n",
                "    top_idx = np.argsort(chi2_scores)[::-1][:10]\n",
                "    top_terms = [(feature_names[i], round(chi2_scores[i], 2)) for i in top_idx]\n",
                "    print(f'\\n--- Top 10 Chi² Features for [{category}] ---')\n",
                "    for term, score in top_terms:\n",
                "        print(f'  {term:<30} {score}')"
            ]
        },
        # ── Section 4 Header ───────────────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 🏷️ Section 4 — Label Encoding & Train/Test Split\n",
                "`LabelEncoder` maps category strings to integers for sklearn compatibility. "
                "We use **stratified 80/20 split** to preserve class balance."
            ]
        },
        # ── Cell 8: Encode + split ─────────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "le = LabelEncoder()\n",
                "y = le.fit_transform(df['category'])\n",
                "\n",
                "print(f'Classes (encoded → label): {dict(enumerate(le.classes_))}')\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X_tfidf, y,\n",
                "    test_size=0.2,\n",
                "    random_state=42,\n",
                "    stratify=y\n",
                ")\n",
                "\n",
                "print(f'\\nTraining samples : {X_train.shape[0]}')\n",
                "print(f'Test samples     : {X_test.shape[0]}')"
            ]
        },
        # ── Section 5 Header ───────────────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 🤖 Section 5 — Model Training: Logistic Regression\n",
                "**Logistic Regression** is our primary classifier. It is:\n",
                "- Computationally efficient for high-dimensional sparse TF-IDF features\n",
                "- Naturally probabilistic (outputs confidence scores per class)\n",
                "- Easily interpretable via learned weights\n",
                "\n",
                "> `class_weight='balanced'` adjusts for any imbalanced label distribution."
            ]
        },
        # ── Cell 9: Train model ────────────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = LogisticRegression(\n",
                "    class_weight='balanced',\n",
                "    max_iter=1000,\n",
                "    random_state=42,\n",
                "    solver='lbfgs',\n",
                "    multi_class='auto'\n",
                ")\n",
                "\n",
                "model.fit(X_train, y_train)\n",
                "y_pred = model.predict(X_test)\n",
                "\n",
                "test_acc = (y_pred == y_test).mean()\n",
                "print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)')"
            ]
        },
        # ── Section 6 Header ───────────────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 📊 Section 6 — Confusion Matrix Visualization\n",
                "A confusion matrix shows prediction accuracy per class and which classes are confused with each other.\n",
                "\n",
                "- **Diagonal cells** (top-left to bottom-right) represent **correct predictions**.\n",
                "- **Off-diagonal cells** represent **misclassifications**.\n",
                "- The normalized version shows each cell as a proportion of the row total (recall per class)."
            ]
        },
        # ── Cell 10: Confusion matrix ──────────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))\n",
                "cm_norm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))), normalize='true')\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
                "\n",
                "# ── Raw counts ──────────────────────────────────────────────────────────────\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
                "            xticklabels=le.classes_, yticklabels=le.classes_,\n",
                "            linewidths=0.5, ax=axes[0])\n",
                "axes[0].set_title('Confusion Matrix — Raw Counts\\n(Model: Logistic Regression + TF-IDF)', fontsize=13)\n",
                "axes[0].set_xlabel('Predicted Category', fontsize=11)\n",
                "axes[0].set_ylabel('Actual Category', fontsize=11)\n",
                "\n",
                "# ── Normalized (recall per class) ───────────────────────────────────────────\n",
                "sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',\n",
                "            xticklabels=le.classes_, yticklabels=le.classes_,\n",
                "            linewidths=0.5, vmin=0, vmax=1, ax=axes[1])\n",
                "axes[1].set_title('Confusion Matrix — Normalized (Recall)\\n(Model: Logistic Regression + TF-IDF)', fontsize=13)\n",
                "axes[1].set_xlabel('Predicted Category', fontsize=11)\n",
                "axes[1].set_ylabel('Actual Category', fontsize=11)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # ── Cell 11: Classification report ────────────────────────────────────────────
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Classification Report (Logistic Regression)')\n",
                "print('=' * 50)\n",
                "print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))"
            ]
        },
        # ── Summary ───────────────────────────────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## ✅ Pipeline Summary\n",
                "\n",
                "| Step | Technique | Purpose |\n",
                "|------|-----------|---------|\n",
                "| 1 | `fillna('')` + `clean_text()` | Remove noise & missing values |\n",
                "| 2 | `text_len` (char count) | Structural ticket complexity proxy |\n",
                "| 3 | TF-IDF (unigrams + bigrams, stopwords, min_df=2) | Semantic feature representation |\n",
                "| 4 | Chi² Scoring | Identify top discriminating terms per class |\n",
                "| 5 | `LabelEncoder` + stratified 80/20 split | Encode labels, preserve class ratios |\n",
                "| 6 | **Logistic Regression** (`balanced`, `lbfgs`) | Efficient probabilistic text classifier |\n",
                "| 7 | Confusion Matrix (raw + normalized) | Audit per-class prediction errors |"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print(f"Notebook written to: {NOTEBOOK_PATH}")
