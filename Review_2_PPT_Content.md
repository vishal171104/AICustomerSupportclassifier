## Slide 1: Title
**Comparative Experimental Study of Classical ML Models for Multi-Class Ticket Classification and Priority Prediction**

## Slide 2: Problem Statement
- Support triage is not just keyword matching.
- Real-world tickets are messy, ambiguous, and contain typos.
- **Challenge**: Manual prioritization is inconsistent; we need a statistically sound, automated system.

## Slide 3: Research Methodology (Academic Rigor)
- **Baseline**: Random Stratified (28%)
- **Optimization**: Formal `GridSearchCV` for hyperparameter tuning.
- **Ensemble**: Combining SVM and Logistic Regression via Soft Voting.
- **Assessment**: **Learning Curve Analysis** to verify model stability and generalization.

## Slide 4: Data Engineering & Hardening
- **Diversity**: Mixed intents, noise phrases ("Sent from iPhone"), and 30% typo rate.
- **Ambiguity**: Removed obvious triggers (e.g., "refund") to force the model to learn context.

## Slide 5: Hyperparameter Optimization Results
| Model | Best Params | CV Accuracy |
| :--- | :--- | :--- |
| **SVM** | C=10, linear | 0.50 |
| **LogReg** | C=10 | 0.47 |
- GridSearch ensures the model is optimally regularized for the dataset.

## Slide 6: Stability Analysis (Learning Curves)
- *Self-Critique Slide*: Points to `reports/lc_svm.png`.
- **Finding**: Convergence between Training and CV scores indicates the model is generalizing well and is ready for the next data scale up (1000+ samples).

## Slide 7: Architectural Comparison (Feature Engineering)
- **TF-IDF + SVM**: **67% Accuracy** (The strongest baseline).
- **Fine-tuned DistilBERT**: **63% Accuracy**.
- **S-BERT (MiniLM)**: **41% Accuracy**.
- **Insight**: Deep learning/Embeddings are not always superior. For ticket priority, high-frequency "urgency tokens" are highly informative and better captured by TF-IDF bigrams.

## Slide 8: Why TF-IDF + Classical ML? (The Defense)
- **High Signal-to-Noise**: Support tickets use specific professional vocabulary. TF-IDF Bigrams capture these "anchor tokens" more effectively than generic pre-trained embeddings on small datasets.
- **Inference Speed**: <5ms vs 100ms+ (DistilBERT). Crucial for high-volume support APIs.
- **Interpretability**: We can prove *why* a ticket was flagged as 'Critical' by looking at feature weights.
- **Calibration**: Use of `CalibratedClassifierCV` provides reliable probability estimates.

## Slide 9: Model Comparison (Final Lift)
- **Random Baseline**: 28%
- **Our Ensemble Model**: 60%
- **Lift**: **+32%** absolute improvement in a highly ambiguous environment.

## Slide 8: Production-Ready Architecture
- **Probability Calibration**: Every prediction comes with a confidence score.
- **Modular Pipelines**: Built for high-performance inference via FastAPI.
- **Error Analysis**: Continuous monitoring of misclassifications for linguistic refinement.
