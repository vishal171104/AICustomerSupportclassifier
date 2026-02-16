import sys
import os
import sqlite3
import time
from pathlib import Path
import pickle
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Observability Layer (SQLite Logging) ---
DB_PATH = BASE_DIR / "data" / "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (timestamp TEXT, input_text TEXT, category TEXT, priority TEXT, 
                  cat_conf REAL, pri_conf REAL, latency_ms REAL)''')
    conn.commit()
    conn.close()

init_db()

def log_prediction(text, cat, pri, cat_conf, pri_conf, latency):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (time.strftime('%Y-%m-%d %H:%M:%S'), text, cat, pri, cat_conf, pri_conf, latency))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log to DB: {e}")

# --- API Setup ---
app = FastAPI(
    title="Customer Support Ticket AI - Production Rigor",
    version="2.1.0",
    description="Automated triage system with hardened API and observability."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
CAT_MODEL_PATH = BASE_DIR / "model" / "category_pipeline.pkl"
PRI_MODEL_PATH = BASE_DIR / "model" / "priority_pipeline.pkl"

try:
    with open(CAT_MODEL_PATH, "rb") as f:
        category_pipeline = pickle.load(f)
    with open(PRI_MODEL_PATH, "rb") as f:
        priority_pipeline = pickle.load(f)
    logger.info("Successfully loaded production pipelines.")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise RuntimeError("Could not load models.")

# --- Helper Functions ---
def get_top_keywords(pipeline, text, predicted_class, n=3):
    try:
        tfidf = pipeline.named_steps['tfidf']
        clf = pipeline.named_steps['clf']
        feature_names = tfidf.get_feature_names_out()
        X_tfidf = tfidf.transform([text]).toarray()[0]
        
        if hasattr(clf, 'calibrated_classifiers_'):
            cal_clf = clf.calibrated_classifiers_[0]
            base_clf = getattr(cal_clf, 'estimator', getattr(cal_clf, 'base_estimator', None))
            coef = base_clf.coef_ if base_clf is not None else None
            classes = clf.classes_
        elif hasattr(clf, 'coef_'):
            coef = clf.coef_
            classes = clf.classes_
        else:
            return []

        if coef is None: return []
        class_idx = list(classes).index(predicted_class)
        weight_vector = coef[0] if coef.shape[0] == 1 and class_idx == 1 else (-coef[0] if coef.shape[0] == 1 else coef[class_idx])
        
        present_indices = np.where(X_tfidf > 0)[0]
        if len(present_indices) == 0: return []
        influence = X_tfidf[present_indices] * weight_vector[present_indices]
        top_local_indices = influence.argsort()[-n:][::-1]
        top_global_indices = [present_indices[i] for i in top_local_indices if influence[i] > 0]
        return [feature_names[i] for i in top_global_indices]
    except:
        return []

# --- Metrics & Rate Limiting (Phase 7) ---
METRICS = {
    "total_predictions": 0,
    "error_count": 0,
    "avg_latency": 0.0,
    "last_confidence": 0.0
}

RATE_LIMITS = {} 

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    if client_ip in RATE_LIMITS:
        last_time, count = RATE_LIMITS[client_ip]
        if current_time - last_time < 60:
            if count >= 30: # Relaxed slightly for testing
                return JSONResponse(status_code=429, content={"detail": "Too many requests"})
            RATE_LIMITS[client_ip] = (last_time, count + 1)
        else:
            RATE_LIMITS[client_ip] = (current_time, 1)
    else:
        RATE_LIMITS[client_ip] = (current_time, 1)
        
    response = await call_next(request)
    return response

@app.get("/metrics")
def metrics():
    lines = [
        f"total_predictions {METRICS['total_predictions']}",
        f"error_count {METRICS['error_count']}",
        f"avg_latency_ms {round(METRICS['avg_latency'], 4)}"
    ]
    return Response(content="\n".join(lines), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time(), "models": ["category", "priority"]}

@app.get("/model_info")
def model_info():
    return {
        "version": "2.1.0",
        "category_model": str(category_pipeline.named_steps['clf']),
        "priority_model": str(priority_pipeline.named_steps['clf']),
        "tfidf_params": category_pipeline.named_steps['tfidf'].get_params()
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 20", conn)
    conn.close()
    html_table = df.to_html(classes='table table-striped', index=False)
    return f"""
    <html>
        <head>
            <title>ML Observability Dashboard</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            <style>body {{ padding: 20px; background: #f8f9fa; }} .card {{ margin-bottom: 20px; }}</style>
        </head>
        <body>
            <div class="container">
                <h1 class="mb-4">🚀 Production ML Observability</h1>
                <div class="row">
                    <div class="col-md-4">
                        <div class="card text-white bg-primary">
                            <div class="card-body">
                                <h5 class="card-title">Total Inferences</h5>
                                <p class="card-text h2">{METRICS['total_predictions']}</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card mt-4">
                    <div class="card-header">Latest Predictions</div>
                    <div class="card-body" style="overflow-x: auto;">
                        {html_table}
                    </div>
                </div>
            </div>
        </body>
    </html>
    """

class Ticket(BaseModel):
    text: Optional[str] = None
    description: Optional[str] = None

@app.post("/predict")
def predict(ticket: Ticket):
    start_time = time.time()
    try:
        raw_text = ticket.text if ticket.text else ticket.description
        if not raw_text:
             raise HTTPException(status_code=400, detail="Text or description is required")
             
        text = clean_text(raw_text)
        cat_pred = category_pipeline.predict([text])[0]
        cat_conf = float(np.max(category_pipeline.predict_proba([text])))
        pri_pred = priority_pipeline.predict([text])[0]
        pri_conf = float(np.max(priority_pipeline.predict_proba([text])))
        
        latency = (time.time() - start_time) * 1000
        METRICS["total_predictions"] += 1
        METRICS["avg_latency"] = (METRICS["avg_latency"] * (METRICS["total_predictions"] - 1) + latency) / METRICS["total_predictions"]
        
        log_prediction(raw_text, cat_pred, pri_pred, cat_conf, pri_conf, latency)
        
        return {
            "category": cat_pred,
            "category_confidence": round(cat_conf, 4),
            "category_keywords": get_top_keywords(category_pipeline, text, cat_pred),
            "priority": pri_pred,
            "priority_confidence": round(pri_conf, 4),
            "priority_keywords": get_top_keywords(priority_pipeline, text, pri_pred),
            "latency_ms": round(latency, 2)
        }
    except Exception as e:
        METRICS["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(tickets: List[Ticket]):
    results = []
    for t in tickets:
        results.append(predict(t))
    return {"batch_results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

