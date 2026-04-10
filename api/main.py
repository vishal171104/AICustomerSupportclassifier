import sys
import os
import sqlite3
import time
from pathlib import Path
import pickle
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
    c.execute('''CREATE TABLE IF NOT EXISTS tickets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  subject TEXT, body TEXT, sender_email TEXT, 
                  received_at TEXT, ingested_at TEXT, source TEXT, 
                  predicted_label TEXT, priority TEXT, confidence_score REAL,
                  status TEXT DEFAULT 'open')''')
    c.execute('''CREATE TABLE IF NOT EXISTS config
                 (watched_email TEXT, registered_at TEXT)''')
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
    raw_params = category_pipeline.named_steps['tfidf'].get_params()
    safe_params = {
        k: v if isinstance(v, (int, float, bool, str, type(None))) else str(v)
        for k, v in raw_params.items()
    }
    return {
        "version": "2.1.0",
        "category_model": str(category_pipeline.named_steps['clf']),
        "priority_model": str(priority_pipeline.named_steps['clf']),
        "tfidf_params": safe_params
    }

# Make sure the UI directory exists
os.makedirs(BASE_DIR / "ui" / "dashboard", exist_ok=True)
app.mount("/dashboard_assets", StaticFiles(directory=str(BASE_DIR / "ui" / "dashboard")), name="dashboard_assets")

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    index_path = BASE_DIR / "ui" / "dashboard" / "index.html"
    if index_path.exists():
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard UI not built yet.</h1>", status_code=404)

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
    except HTTPException:
        raise
    except Exception as e:
        METRICS["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(tickets: List[Ticket]):
    results = []
    for t in tickets:
        results.append(predict(t))
    return {"batch_results": results}

class EmailConfig(BaseModel):
    email: str

class IngestEmail(BaseModel):
    subject: str
    body: str
    sender_email: str
    recipient_email: str
    received_at: str
    source: str = "gmail"

@app.post("/api/config/email")
def set_config_email(config: EmailConfig):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM config")
    c.execute("INSERT INTO config (watched_email, registered_at) VALUES (?, ?)", 
              (config.email, time.strftime('%Y-%m-%dT%H:%M:%SZ')))
    conn.commit()
    conn.close()
    return {"success": True, "email": config.email}

@app.get("/api/config/email")
def get_config_email():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT watched_email FROM config LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row and row[0]:
        return {"email": row[0]}
    return {"email": None}

@app.delete("/api/config/email")
def delete_config_email():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM config")
    conn.commit()
    conn.close()
    return {"success": True, "email": None}

@app.post("/api/tickets/ingest")
def ingest_tickets(emails: List[IngestEmail]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT watched_email FROM config LIMIT 1")
    row = c.fetchone()
    watched_email = row[0] if row else None
    
    if not watched_email:
        conn.close()
        raise HTTPException(status_code=400, detail="No watched email configured.")
        
    ingested = 0
    failed = 0
    
    for email in emails:
        if email.recipient_email != watched_email:
            continue
        try:
            raw_text = f"{email.subject} {email.body}"
            text = clean_text(raw_text)
            
            cat_pred = category_pipeline.predict([text])[0]
            cat_conf = float(np.max(category_pipeline.predict_proba([text])))
            pri_pred = priority_pipeline.predict([text])[0]
            pri_conf = float(np.max(priority_pipeline.predict_proba([text])))
            
            c.execute('''INSERT INTO tickets 
                         (subject, body, sender_email, received_at, ingested_at, source, 
                          predicted_label, priority, confidence_score) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (email.subject, email.body, email.sender_email, email.received_at, 
                       time.strftime('%Y-%m-%dT%H:%M:%SZ'), email.source, 
                       cat_pred, pri_pred, max(cat_conf, pri_conf)))
            ingested += 1
        except Exception as e:
            failed += 1
            logger.error(f"Ingestion error: {e}")
            
    conn.commit()
    conn.close()
    return {"ingested": ingested, "failed": failed}

@app.get("/api/tickets")
def get_tickets(sort: str = "Time (Newest First)", filter: str = "all", source: str = "all", search: str = "", status: str = "open"):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = "SELECT * FROM tickets WHERE status = ?"
    params = [status.lower()]
    
    if filter != "all":
        query += " AND LOWER(priority) = ?"
        params.append(filter.lower())
        
    if source != "all":
        query += " AND LOWER(source) = ?"
        params.append(source.lower())
        
    if search:
        query += " AND (LOWER(subject) LIKE ? OR LOWER(body) LIKE ?)"
        params.extend([f"%{search.lower()}%", f"%{search.lower()}%"])
        
    if sort == "Priority (High→Low)":
        query += " ORDER BY CASE WHEN LOWER(priority)='critical' THEN 1 WHEN LOWER(priority)='high' THEN 2 WHEN LOWER(priority)='medium' THEN 3 ELSE 4 END ASC"
    elif sort == "Priority (Low→High)":
        query += " ORDER BY CASE WHEN LOWER(priority)='critical' THEN 1 WHEN LOWER(priority)='high' THEN 2 WHEN LOWER(priority)='medium' THEN 3 ELSE 4 END DESC"
    elif sort == "Time (Oldest First)":
        query += " ORDER BY ingested_at ASC, received_at ASC"
    else: 
        query += " ORDER BY ingested_at DESC, received_at DESC"
        
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    return [dict(ix) for ix in rows]

@app.get("/api/tickets/stats")
def get_ticket_stats(status: str = "open"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM tickets WHERE status = ?", (status.lower(),))
    total = c.fetchone()[0]
    
    c.execute("SELECT LOWER(priority), COUNT(*) FROM tickets WHERE status = ? GROUP BY priority", (status.lower(),))
    pri_counts = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute("SELECT LOWER(source), COUNT(*) FROM tickets WHERE status = ? GROUP BY source", (status.lower(),))
    src_counts = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute("SELECT AVG(confidence_score) FROM tickets WHERE status = ?", (status.lower(),))
    row = c.fetchone()
    avg_conf = row[0] if row and row[0] is not None else 0.0
    
    c.execute("SELECT MAX(ingested_at) FROM tickets WHERE status = ?", (status.lower(),))
    last_upd = c.fetchone()[0]
    
    conn.close()
    
    by_priority = {
        "critical": pri_counts.get("critical", 0),
        "high": pri_counts.get("high", 0),
        "medium": pri_counts.get("medium", 0),
        "low": pri_counts.get("low", 0)
    }
    
    by_source = {
        "gmail": src_counts.get("gmail", 0),
        "manual": src_counts.get("manual", 0)
    }
    
    return {
        "total": total,
        "by_priority": by_priority,
        "by_source": by_source,
        "avg_confidence": avg_conf,
        "last_updated": last_upd
    }

class TicketStatusUpdate(BaseModel):
    status: str

@app.patch("/api/tickets/{ticket_id}/status")
def update_ticket_status(ticket_id: int, status_update: TicketStatusUpdate):
    if status_update.status.lower() not in ["open", "resolved"]:
        raise HTTPException(status_code=400, detail="Invalid status")
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE tickets SET status = ? WHERE id = ?", (status_update.status.lower(), ticket_id))
    conn.commit()
    conn.close()
    return {"success": True, "ticket_id": ticket_id, "status": status_update.status.lower()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

