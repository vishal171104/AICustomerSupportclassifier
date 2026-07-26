import sys
import os
import sqlite3
import time
from pathlib import Path
import pickle
import logging
from typing import List, Optional
import threading
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as scipy_entropy
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
                  status TEXT DEFAULT 'open',
                  uncertainty_score REAL DEFAULT 0.0,
                  reviewed INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS config
                 (watched_email TEXT, registered_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  similarity_score REAL,
                  window_size INTEGER,
                  alert INTEGER DEFAULT 0)''')
    # Migrate existing databases — safe no-op if columns already exist
    for _col_sql in [
        "ALTER TABLE tickets ADD COLUMN uncertainty_score REAL DEFAULT 0.0",
        "ALTER TABLE tickets ADD COLUMN reviewed INTEGER DEFAULT 0",
        "ALTER TABLE tickets ADD COLUMN explanation TEXT"
    ]:
        try:
            c.execute(_col_sql)
        except Exception:
            pass
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

# --- Temporal Drift Detection ---
drift_counter = 0
drift_last_check = time.time()
drift_lock = threading.Lock()
drift_status = {
    "score": None,
    "alert": False,
    "timestamp": None,
    "window_size": 0
}
reference_vector = None
global_vectorizer = None

@app.on_event("startup")
def startup_event():
    global reference_vector, global_vectorizer
    try:
        csv_path = BASE_DIR / "data" / "tickets.csv"
        if not csv_path.exists():
            raise FileNotFoundError("tickets.csv not found")
        df = pd.read_csv(csv_path)
        texts = [clean_text(str(t)) for t in df["description"].tolist()]
        
        vectorizer = category_pipeline.named_steps.get('tfidfvectorizer', category_pipeline.named_steps.get('tfidf'))
        if vectorizer is None:
            raise ValueError("TfidfVectorizer not found in category_pipeline")
            
        global_vectorizer = vectorizer
        tfidf_matrix = vectorizer.transform(texts)
        reference_vector = np.mean(tfidf_matrix.toarray(), axis=0)
        logger.info(f"Computed reference_vector of shape {reference_vector.shape} for drift detection.")
    except Exception as e:
        logger.warning(f"Could not compute reference_vector: {e}")
        reference_vector = None
        global_vectorizer = None

def run_drift_check():
    global drift_status
    if reference_vector is None or global_vectorizer is None:
        return
        
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT body FROM tickets ORDER BY id DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return
            
        texts = [clean_text(row["body"]) for row in rows]
        tfidf_matrix = global_vectorizer.transform(texts)
        window_vector = np.mean(tfidf_matrix.toarray(), axis=0)
        
        score = cosine_similarity([window_vector], [reference_vector])[0][0]
        alert = int(score < 0.3)
        window_size = len(texts)
        ts_now = time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT INTO drift_logs (timestamp, similarity_score, window_size, alert)
                     VALUES (?, ?, ?, ?)""", (ts_now, score, window_size, alert))
        conn.commit()
        conn.close()
        
        with drift_lock:
            drift_status = {
                "score": float(score),
                "alert": bool(alert),
                "timestamp": ts_now,
                "window_size": window_size
            }
        logger.info(f"Drift check complete. Score: {score:.3f}, Alert: {bool(alert)}")
    except Exception as e:
        logger.error(f"Error in run_drift_check: {e}")

def maybe_run_drift_check():
    global drift_counter, drift_last_check
    run_it = False
    with drift_lock:
        drift_counter += 1
        now = time.time()
        time_elapsed = now - drift_last_check
        if drift_counter >= 50 or time_elapsed >= 600:
            drift_counter = 0
            drift_last_check = now
            run_it = True
            
    if run_it:
        run_drift_check()

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

# --- Active Learning: Uncertainty Scoring ---
UNCERTAINTY_THRESHOLD = 0.5

def compute_uncertainty(cat_conf: float, pri_conf: float) -> float:
    """Combined entropy score over category and priority binary distributions."""
    cat_ent = float(scipy_entropy([cat_conf, 1.0 - cat_conf], base=2))
    pri_ent = float(scipy_entropy([pri_conf, 1.0 - pri_conf], base=2))
    return round(cat_ent + pri_ent, 6)

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
            if count >= 200: # Raised for demo — simulation fires many requests
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
def ingest_tickets(emails: List[IngestEmail], background_tasks: BackgroundTasks):
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
            
            uncertainty = compute_uncertainty(cat_conf, pri_conf)
            c.execute('''INSERT INTO tickets 
                         (subject, body, sender_email, received_at, ingested_at, source, 
                          predicted_label, priority, confidence_score, uncertainty_score) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (email.subject, email.body, email.sender_email, email.received_at, 
                       time.strftime('%Y-%m-%dT%H:%M:%SZ'), email.source, 
                       cat_pred, pri_pred, max(cat_conf, pri_conf), uncertainty))
            ingested += 1
        except Exception as e:
            failed += 1
            logger.error(f"Ingestion error: {e}")
            
    conn.commit()
    conn.close()
    
    background_tasks.add_task(maybe_run_drift_check)
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
        query += " ORDER BY CASE WHEN LOWER(priority)='critical' THEN 1 WHEN LOWER(priority)='high' THEN 2 WHEN LOWER(priority)='medium' THEN 3 ELSE 4 END ASC, id DESC"
    elif sort == "Priority (Low→High)":
        query += " ORDER BY CASE WHEN LOWER(priority)='critical' THEN 1 WHEN LOWER(priority)='high' THEN 2 WHEN LOWER(priority)='medium' THEN 3 ELSE 4 END DESC, id DESC"
    elif sort == "Time (Oldest First)":
        query += " ORDER BY id ASC"
    else: 
        query += " ORDER BY id DESC"
        
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

    c.execute(
        "SELECT COUNT(*) FROM tickets WHERE status = ? AND reviewed = 0 AND uncertainty_score > ?",
        (status.lower(), UNCERTAINTY_THRESHOLD)
    )
    uncertain_count = c.fetchone()[0]

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
        "last_updated": last_upd,
        "uncertain_count": uncertain_count
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

# --- Active Learning: Uncertain Tickets & Review Endpoints ---

@app.get("/api/tickets/uncertain")
def get_uncertain_tickets():
    """Return open, unreviewed tickets sorted by uncertainty score (most uncertain first)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM tickets
        WHERE status = 'open' AND reviewed = 0
        ORDER BY uncertainty_score DESC
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


class ReviewUpdate(BaseModel):
    correction: Optional[str] = None


@app.patch("/api/tickets/{ticket_id}/review")
def mark_ticket_reviewed(ticket_id: int, review: Optional[ReviewUpdate] = None):
    """Mark a ticket as human-reviewed after inspection or correction."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE tickets SET reviewed = 1 WHERE id = ?", (ticket_id,))
    conn.commit()
    conn.close()
    return {"success": True, "ticket_id": ticket_id, "reviewed": True}


# --- AI Explanation Endpoint (Gemini-powered, DB-cached) ---
from google import genai as google_genai

_gemini_key = os.getenv("GEMINI_API_KEY")
if _gemini_key:
    _gemini_client = google_genai.Client(api_key=_gemini_key)
    logger.info("Gemini client initialised (google.genai SDK)")
else:
    _gemini_client = None
    logger.warning("GEMINI_API_KEY not set — AI explanations disabled")


@app.post("/api/tickets/{ticket_id}/explain")
async def explain_ticket(ticket_id: int):
    """Generate a 2-sentence Gemini explanation, cached in the tickets table."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # ── Return cached explanation — zero API cost ───────────────────
    c.execute("SELECT explanation FROM tickets WHERE id = ?", (ticket_id,))
    cached = c.fetchone()
    if cached and cached[0]:
        conn.close()
        return {"explanation": cached[0]}

    # ── Fetch ticket details ─────────────────────────────────────────
    c.execute(
        "SELECT body, predicted_label, priority, confidence_score FROM tickets WHERE id = ?",
        (ticket_id,)
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return {"explanation": "Ticket not found."}

    body = row["body"]
    category = row["predicted_label"]
    priority = row["priority"]
    confidence_pct = round((row["confidence_score"] or 0) * 100, 1)

    # ── Guard: key must be configured ───────────────────────────────
    if not _gemini_client:
        mock_explanation = f"Based on the linguistic features, the classifier confidently assigned this to '{category}'. The presence of urgent keywords and domain terms resulted in a '{priority}' priority designation."
        c.execute("UPDATE tickets SET explanation = ? WHERE id = ?", (mock_explanation, ticket_id))
        conn.commit()
        conn.close()
        return {"explanation": mock_explanation}

    # ── Call Gemini API ──────────────────────────────────────────────
    try:
        prompt = (
            f"A support ticket classifier predicted:\n"
            f"Category: {category}\n"
            f"Priority: {priority}\n"
            f"Confidence: {confidence_pct}%\n\n"
            f"Ticket text: {body}\n\n"
            "In exactly 2 sentences, explain why this ticket was classified this way. "
            "Be specific — reference actual words or phrases from the ticket text. "
            "Do not use bullet points. Plain sentences only."
        )
        response = _gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        explanation = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error for ticket {ticket_id}: {e}")
        conn.close()
        return {"explanation": "Explanation unavailable."}

    # ── Cache in DB — same ticket never calls API twice ─────────────
    c.execute("UPDATE tickets SET explanation = ? WHERE id = ?", (explanation, ticket_id))
    conn.commit()
    conn.close()
    return {"explanation": explanation}

@app.get("/api/drift/status")
def get_drift_status():
    return drift_status

@app.get("/api/drift/history")
def get_drift_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT id, timestamp, similarity_score, window_size, alert FROM drift_logs ORDER BY timestamp DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return []



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

