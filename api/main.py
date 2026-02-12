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
        
        # Priority Prediction
        pri_pred = priority_pipeline.predict([text])[0]
        pri_probs = priority_pipeline.predict_proba([text])[0]
        pri_conf = float(max(pri_probs))
        
        return {
            "category": cat_pred,
            "category_confidence": round(cat_conf, 2),
            "priority": pri_pred,
            "priority_confidence": round(pri_conf, 2)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
