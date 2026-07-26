# TicketIQ — Live Demo Script (Review-Ready)
> **Fully offline** · No internet required · Real ML model running locally

---

## ⚡ Start the demo (1 command)

```bash
cd /Users/vishalsi/CustomerSupportTicketAI
python start_demo.py
```

The browser opens automatically at **http://localhost:8000/dashboard**

---

## 🎬 Demo Flow (5–7 minutes)

### Step 1 — Show the Dashboard (30 sec)
Point out:
- **Header**: Live clock, watching email badge (`demo@ticketiq.ai`)
- **5 stat cards**: Total, Critical, High, Low/Med, Needs Review
- **Priority Distribution chart** (doughnut)
- **Ticket table**: subject, sender, category, priority, confidence bar

---

### Step 2 — Live Classify (2 min)
Click the **green ● Live Demo button** → Demo panel opens

**Left panel — "Type a Ticket":**
1. Click **Sample** a few times to cycle through examples, OR type:
   > `"Payment gateway completely broken since 2 hours, users cannot checkout"`
2. Click **Classify with ML Model**
3. Show the result: **Category → Technical, Priority → Critical, Confidence ~94%**
4. Point out **keywords** extracted by TF-IDF and **latency** (~15ms)

Try a low-priority example:
> `"Dashboard loads a bit slowly sometimes, not urgent"`
→ Should show **Medium** priority

Try a billing example:
> `"I was charged twice this month, please refund the duplicate"`
→ Should show **Billing / Critical** or **High**

---

### Step 3 — Submit as Ticket (1 min)
1. Type any support message in the text box
2. Click **Submit as Ticket**
3. Watch the toast notification appear: *"✓ Ticket classified and added to dashboard!"*
4. Scroll to the ticket table — the new ticket appears **instantly at the top**
5. Point out: **Category badge** (purple/orange/blue), **Priority badge** (red/orange/green), **Confidence bar**

---

### Step 4 — Auto-Simulation (2 min)
**Right panel — "Auto-Simulate Email Arrival":**
1. Click **Start Simulation** (Normal 1.5s speed)
2. Watch the status log: `→ [1/12] Sending: "URGENT: Payment gateway..."` → `✓ Classified`
3. The ticket table **auto-updates** as each email is processed
4. After ~10 seconds, 6–8 tickets appear — show the mix of Critical, High, Medium, Low
5. Click **Stop** when enough tickets are visible

---

### Step 5 — Explore the Ticket Table (1 min)
- **Sort by Priority (High→Low)** — all Criticals jump to top
- **Filter by "Critical"** — only red badges shown
- **Filter by "Demo"** source — shows only simulation tickets
- **Search** for "payment" — filters instantly
- Click the **👁 eye icon** on any ticket → expanded view shows:
  - Raw body text
  - ML Category + Priority cards
  - Model confidence bar
  - AI Explanation (generated from keywords)

---

### Step 6 — Resolve a Ticket (30 sec)
- Click the **✓ check icon** on any ticket → toast: *"Ticket #X marked as resolved!"*
- Switch the **Status dropdown to "Resolved"** — shows the resolved list
- Switch back to "Open" — resolved ticket is gone

---

### Step 7 — Model Drift (optional, 30 sec)
- Point to the **Model Info sidebar**: TF-IDF + Logistic Regression, 1,083 training tickets, ~89% accuracy
- Mention: drift detection runs every 50 predictions against reference TF-IDF vectors

---

## 🗣 Key Talking Points

| Feature | What to say |
|---|---|
| **No internet needed** | "Entire stack runs on localhost — model, API, and UI are all local" |
| **Real model** | "This is the actual trained sklearn pipeline, not a mock — TF-IDF + Logistic Regression" |
| **Latency** | "Classification happens in ~10–20ms — FastAPI + sklearn is extremely fast" |
| **Confidence scores** | "We compute entropy-based uncertainty — uncertain tickets are flagged for human review" |
| **Category + Priority** | "Two independent classifiers: one for category (Technical/Billing/Account), one for priority" |
| **Drift detection** | "System computes cosine similarity between incoming tickets and training vectors — alerts if distribution shifts" |

---

## 🚨 Troubleshooting

| Problem | Fix |
|---|---|
| Server not starting | `source venv/bin/activate && python -m uvicorn api.main:app --port 8000` |
| "No email configured" badge | Click ⚙ Settings → type any email → Save |
| Simulation fails | Check server is running — look at terminal for errors |
| Tickets not appearing | Click **Refresh** button in the header |
| Page won't load | Make sure port 8000 is free: `lsof -i :8000` |

---

## 📁 Key Files

```
CustomerSupportTicketAI/
├── start_demo.py              ← Run this to start everything
├── api/main.py                ← FastAPI backend (predict, ingest, tickets)
├── model/
│   ├── category_pipeline.pkl  ← Trained category classifier
│   └── priority_pipeline.pkl  ← Trained priority classifier
├── ui/dashboard/
│   ├── index.html             ← Dashboard UI (offline assets)
│   ├── app.js                 ← Dashboard logic
│   ├── demo.js                ← Live demo engine
│   ├── tailwind.min.js        ← Local (no CDN)
│   ├── chart.min.js           ← Local (no CDN)
│   └── fontawesome/           ← Local (no CDN)
└── data/
    ├── tickets.csv            ← Training dataset (1,083 tickets)
    └── predictions.db         ← SQLite — all tickets + logs
```
