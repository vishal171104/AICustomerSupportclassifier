# System Architecture

## Mermaid Diagram (embed in paper)

```mermaid
flowchart LR
    subgraph EXT["Chrome Extension (MV3)"]
        A[Gmail OAuth\nIdentity API] --> B[background.js\nService Worker]
        B -->|"Poll every 5 min\nalarms API"| A
        B --> C[popup.js\nPriority Queue UI]
    end

    subgraph API["FastAPI Backend (localhost:8000)"]
        D["/api/tickets/ingest\nPOST"] --> E[Text Cleaning\nutils/preprocessing.py]
        E --> F[TF-IDF Vectorizer\n5000 features, 1–2-grams]
        F --> G[Logistic Regression\nCategory Pipeline]
        F --> H[Logistic Regression\nPriority Pipeline]
        G & H --> I[SQLite\npredictions.db]
        I --> J["/api/tickets GET\nSort/Filter/Search"]
        I --> K["/api/tickets/stats\nPriority Breakdown"]
        I --> L["/api/tickets/{id}/correction\nPOST (Feedback Loop)"]
    end

    subgraph DASH["Dashboard (Vanilla JS)"]
        M[index.html\napp.js] --> J & K
        M -->|"Resolve / Correct"| L
    end

    subgraph TRAIN["Offline Retraining"]
        L -->|"Accepted corrections\nfeedback table"| N[retrain_from_feedback.py]
        N -->|"Versioned model\ncategory_pipeline_v2.pkl"| G
    end

    B -->|"POST emails"| D
    J --> M
```

## Component Descriptions

| Component | Technology | Role |
|---|---|---|
| Chrome Extension | JavaScript MV3 | Reads Gmail via OAuth, pushes to backend |
| FastAPI Backend | Python 3.10, Uvicorn | REST API, ML inference, SQLite management |
| ML Pipeline | scikit-learn 1.5 | TF-IDF + Logistic Regression (Category & Priority) |
| SQLite | `predictions.db` | Tickets, config, feedback, prediction logs |
| Dashboard | Vanilla JS + CSS | Ticket queue, stats, resolve/correct UI |
| Feedback Loop | Python script | Retrains model from agent corrections (novel) |

## Data Flow (End-to-End)

```
[Gmail Inbox]
      │  OAuth (gmail.readonly)
      ▼
[background.js] ──── POST /api/tickets/ingest ────▶ [FastAPI]
                                                         │
                                              clean_text() 
                                                         │
                                              TF-IDF transform
                                                         │
                                         ┌───────────────┴──────────────┐
                                         ▼                              ▼
                                 Category LR                      Priority LR
                                 (Account/Billing/Tech)           (Low→Critical)
                                         │                              │
                                         └──────────── SQLite ──────────┘
                                                           │
                                               ┌───────────┴───────────┐
                                               ▼                       ▼
                                       Dashboard UI              Chrome popup
                                    (sort, filter, resolve)    (priority queue)
                                               │
                                   Agent corrects label ✏️
                                               │
                                         feedback table
                                               │
                                    retrain_from_feedback.py
                                               │
                                    category_pipeline_v2.pkl
```
