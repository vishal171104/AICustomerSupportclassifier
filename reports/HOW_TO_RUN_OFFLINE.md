# 🚀 How to Run TicketIQ Offline — Complete Guide

---

## ✅ Pre-Check: Everything Already Installed

No internet needed. Everything is local.

| Component | Location | Status |
|---|---|---|
| Python virtual env | `venv/` | ✅ All packages installed |
| Trained ML models | `model/*.pkl` | ✅ Ready |
| Dashboard (HTML/JS/CSS) | `ui/dashboard/` | ✅ All local, no CDN |
| Database | `data/predictions.db` | ✅ SQLite, no server needed |
| Web server | FastAPI + uvicorn | ✅ Inside venv |

---

## ▶️ Option 1 — One Command (Recommended)

Open **Terminal** and run:

```bash
cd /Users/vishalsi/CustomerSupportTicketAI
python start_demo.py
```

**This automatically:**
1. Clears old demo tickets so the board starts fresh
2. Configures the demo email (`demo@ticketiq.ai`)
3. Starts the API server on port 8000
4. Opens the dashboard in your browser

**Browser opens at:** `http://localhost:8000/dashboard`

---

## ▶️ Option 2 — Manual (If Option 1 fails)

### Step 1 — Navigate to project folder

```bash
cd /Users/vishalsi/CustomerSupportTicketAI
```

### Step 2 — Activate the virtual environment

```bash
source venv/bin/activate
```

You will see `(venv)` at the start of your terminal prompt.

### Step 3 — Start the server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Wait for this output:
```
INFO: Successfully loaded production pipelines.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 4 — Open dashboard in browser

Go to this URL in Chrome or Safari:
```
http://localhost:8000/dashboard
```

---

## 🔴 If venv activation fails

Use the full path to the venv Python directly:

```bash
/Users/vishalsi/CustomerSupportTicketAI/venv/bin/python \
  -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 🎬 Demo Walkthrough Once Running

1. **Click the green ● Live Demo button** in the header
2. **Left panel — Classify Live:**
   - Click **Sample** to load an example ticket
   - Click **Classify with ML Model**
   - See: Category + Priority + Confidence % + Keywords + Latency
3. **Right panel — Auto-Simulate:**
   - Click **Start Simulation** (Normal 1.5s speed)
   - Watch 12 realistic emails arrive and get classified one by one
   - Ticket table updates live
4. **Explore the table:**
   - Filter by Priority (Critical / High / Medium / Low)
   - Filter by Source (Demo)
   - Search by keyword
   - Sort by newest or priority
5. **Expand a ticket:** Click the 👁 eye icon → see full body + ML explanation
6. **Resolve a ticket:** Click ✓ → ticket moves to Resolved tab

---

## 🛑 To Stop the Server

Press `Ctrl + C` in the Terminal window.

---

## ⚠️ Troubleshooting

### "Port 8000 already in use"
```bash
lsof -ti:8000 | xargs kill -9
python start_demo.py
```

### "No module named fastapi" or "No module named uvicorn"
```bash
source venv/bin/activate
pip install fastapi uvicorn scikit-learn pandas numpy pydantic
python start_demo.py
```

### Dashboard shows "Connection error"
- Make sure the Terminal server is still running
- Refresh browser with `Cmd + Shift + R`

### Red badge "No email configured"
- Click the ⚙ gear icon → type `demo@ticketiq.ai` → Save Email

### Tickets not appearing after simulation
- Click the blue **Refresh** button in the header

---

## 📁 Key Files Reference

```
CustomerSupportTicketAI/
│
├── start_demo.py              ← RUN THIS — starts everything
│
├── api/main.py                ← FastAPI server (all endpoints)
│
├── model/
│   ├── category_pipeline.pkl  ← Category classifier (TF-IDF + LogReg)
│   └── priority_pipeline.pkl  ← Priority classifier (TF-IDF + LogReg)
│
├── ui/dashboard/
│   ├── index.html             ← Dashboard (fully offline)
│   ├── app.js                 ← Tickets table, filters, chart logic
│   ├── demo.js                ← Live demo engine (12 demo tickets)
│   ├── tailwind.min.js        ← Styling (LOCAL — no CDN)
│   ├── chart.min.js           ← Charts (LOCAL — no CDN)
│   └── fontawesome/           ← Icons (LOCAL — no CDN)
│
├── data/
│   ├── tickets.csv            ← Training data (1,083 tickets)
│   └── predictions.db         ← SQLite DB (tickets + logs)
│
└── venv/                      ← All Python packages (offline)
```

---

## 📊 All 12 Demo Tickets — Verified Predictions

| # | Ticket | Category | Priority |
|---|---|---|---|
| 1 | Payment gateway broken | Technical | 🔴 Critical |
| 2 | Login 500 server error | Technical | 🔴 Critical |
| 3 | Security breach detected | Technical | 🔴 Critical |
| 4 | Entire system down | Technical | 🔴 Critical |
| 5 | App crashes at checkout | Technical | 🟠 High |
| 6 | Refund not processed | Billing | 🟠 High |
| 7 | Charged twice | Billing | 🟠 High |
| 8 | Dashboard loads slowly | Technical | 🟡 Medium |
| 9 | Export button intermittent | Technical | 🟡 Medium |
| 10 | Reset MFA settings | Account | 🟡 Medium |
| 11 | Profile showing inactive | Account | 🟡 Medium |
| 12 | Minor invoice question | Billing | 🟢 Low |

> All 12 verified at **100% accuracy** against the trained model.
