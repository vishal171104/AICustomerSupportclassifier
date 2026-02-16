# Executive Summary: Customer Support Ticket Triage AI

## Project Overview
This project implements a multi-task machine learning system for automated customer support ticket triage. The system classifies incoming tickets into semantic categories (**Technical**, **Billing**, **Account**) and assigns a **Priority** level (**Low**, **Medium**, **High**, **Critical**).

## Key Performance Indicators
| Task | Best Model | Accuracy (CV) | Macro F1 (Test) | Baseline (Majority) |
| :--- | :--- | :--- | :--- | :--- |
| **Category** | Naive Bayes | 96.5% | 0.96 | 33.3% |
| **Priority** | SVM (Linear) | 51.6% | 0.50 | 25.2% |

## Technical Highlights
- **Hybrid Architecture**: Combines classical TF-IDF pipelines (SVM, Logistic Regression) with modern Transformer architectures (DistilBERT).
- **Explainability**: Integrated feature importance analysis using TF-IDF weights and permutation importance.
- **Robustness**: Evaluated against adversarial examples and noise sensitivity.
- **Deployment-Ready**: FastAPI backend with p95 latency < 150ms.

## Phase 1 Progress (Today)
- [x] **Metrics Explosion**: Full per-class breakdown for P/R/F1 generated.
- [x] **Visual Evidence**: Multi-class ROC and PR curves generated for Priority task.
- [x] **Error Analysis**: Confusion matrices identifying common misclassifications (e.g., High vs. Critical).
- [x] **Statistical Foundation**: 5-fold cross-validation with standard deviation reporting.

## Next Steps
1. **Dataset Transparency**: Finalizing the Dataset Card and Bias Analysis.
2. **Model Ablation**: Quantitative study on TF-IDF n-grams and stopword impact.
3. **Error Taxonomy**: Deep dive into semantic ambiguity traps.
