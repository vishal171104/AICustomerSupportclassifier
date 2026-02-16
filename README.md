# AI Ticket Triage System 🎓
[ **Institutional Grade ML Research Project** ]

## 🌐 Live Demo
[🔥 Interactive Research Dashboard](http://localhost:8501) | [📚 API Documentation](http://localhost:8000/docs)

## Research Highlights
| Task | Result | Best Model | Baseline Lift |
| :--- | :--- | :--- | :--- |
| **Category Classification** | **96.5% F1** | Naive Bayes | **2.9x** over Majority |
| **Priority Prediction** | **51.6% F1** | SVM (Linear) | **2.1x** over Random |

---

## 🔬 Core Findings
✅ **Ablation Study**: Unigram + Linear SVM identified as optimal pipeline for support-domain semantic triage.
✅ **Statistical Rigor**: 95% Confidence Interval for Accuracy: **[47.8%, 66.7%]** (via 1000-iteration bootstrap).
✅ **Error Taxonomy**: 75% of misclassifications linked to **OOV Jargon** and **Semantic Negation Context Loss**.

## 🚀 Deployment (1-Click)
```bash
docker-compose up --build
```
*This starts the FastAPI backend (8000) and Streamlit Research Dashboard (8501).*
   - **ReDoc**: Visit `http://127.0.0.1:8000/redoc` for alternative documentation.

## Usage
**Predict Functionality**:
Send a POST request to `/predict` with a JSON body:
```json
{
  "description": "My payment failed and I was charged twice."
}
```

**Response**:
```json
{
  "category": "Billing",
  "priority": "High"
}
```

## Constraints & Notes
- This system uses **Logistic Regression** for efficiency and interpretability.
- Deep Learning (BERT, etc.) is intentionally avoided to meet Review-2 constraints.
- No production deployment is claimed.
