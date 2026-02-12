import sys
from pathlib import Path
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Customer Support Ticket AI - Phase 2",
    description="Automated triage system for ticket categorization and prioritization."
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In research demo, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CAT_MODEL_PATH = BASE_DIR / "model" / "category_pipeline.pkl"
PRI_MODEL_PATH = BASE_DIR / "model" / "priority_pipeline.pkl"

# Load models at startup
try:
    with open(CAT_MODEL_PATH, "rb") as f:
        category_pipeline = pickle.load(f)
    with open(PRI_MODEL_PATH, "rb") as f:
        priority_pipeline = pickle.load(f)
    logger.info(f"Successfully loaded pipelines from {BASE_DIR / 'model'}")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise RuntimeError(f"Could not load models. Please run training pipeline first.")

class Ticket(BaseModel):
    description: str

class PredictionResponse(BaseModel):
    category: str
    priority: str

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

import numpy as np

def get_top_keywords(pipeline, text, predicted_class, n=3):
    """
    Extract top N influential keywords for the predicted class.
    """
    try:
        tfidf = pipeline.named_steps['tfidf']
        clf = pipeline.named_steps['clf']
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        # Transform text to TF-IDF vector
        X_tfidf = tfidf.transform([text]).toarray()[0]
        
        # Determine coefficients and classes
        if hasattr(clf, 'calibrated_classifiers_'):
            # For CalibratedClassifierCV
            cal_clf = clf.calibrated_classifiers_[0]
            # Try different attribute names for the internal estimator
            base_clf = getattr(cal_clf, 'estimator', getattr(cal_clf, 'base_estimator', None))
            if base_clf is not None:
                coef = base_clf.coef_
            else:
                return []
            classes = clf.classes_
        elif hasattr(clf, 'estimators_'):
            # For VotingClassifier, try to use the LogisticRegression part
            lr_part = next((e for name, e in clf.named_estimators_.items() if name == 'lr'), None)
            if lr_part:
                coef = lr_part.coef_
                classes = clf.classes_
            else:
                return []
        elif hasattr(clf, 'coef_'):
            coef = clf.coef_
            classes = clf.classes_
        else:
            return []

        # Find class index
        class_idx = list(classes).index(predicted_class)
        
        # Handle multiclass vs binary coef_ shapes
        if coef.shape[0] == 1:
            # Binary classification (or one class vs rest in some contexts)
            # For 2 classes, coef[0] is for the 'positive' class (index 1)
            # If our predicted class is index 0, we negate the weights
            weight_vector = coef[0] if class_idx == 1 else -coef[0]
        else:
            # Multiclass (one row per class)
            weight_vector = coef[class_idx]

        # Calculate influence: TF-IDF value * Coefficient
        # Focus on features present in the input text
        present_indices = np.where(X_tfidf > 0)[0]
        if len(present_indices) == 0:
            return []
            
        # Get influence for present features
        influence = X_tfidf[present_indices] * weight_vector[present_indices]
        
        # Map back to original feature indices
        top_local_indices = influence.argsort()[-n:][::-1]
        top_global_indices = [present_indices[i] for i in top_local_indices if influence[i] > 0]
        
        return [feature_names[i] for i in top_global_indices]
    except Exception as e:
        logger.error(f"Error extracting keywords for {predicted_class}: {str(e)}")
        return []

@app.post("/predict", response_model=dict)
def predict(ticket: Ticket):
    if not ticket.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")
        
    try:
        text = clean_text(ticket.description)
        
        # Category Prediction
        cat_pred = category_pipeline.predict([text])[0]
        cat_probs = category_pipeline.predict_proba([text])[0]
        cat_conf = float(max(cat_probs))
        cat_keywords = get_top_keywords(category_pipeline, text, cat_pred)
        
        # Priority Prediction
        pri_pred = priority_pipeline.predict([text])[0]
        pri_probs = priority_pipeline.predict_proba([text])[0]
        pri_conf = float(max(pri_probs))
        pri_keywords = get_top_keywords(priority_pipeline, text, pri_pred)
        
        return {
            "category": cat_pred,
            "category_confidence": round(cat_conf, 4),
            "category_keywords": cat_keywords,
            "priority": pri_pred,
            "priority_confidence": round(pri_conf, 4),
            "priority_keywords": pri_keywords
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
