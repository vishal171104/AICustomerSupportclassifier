# 🎫 AI Ticket Triage — Chrome Extension

> **VIT Major Project** | Customer Support Intelligence System  
> Chrome Extension + Gmail API + FastAPI AI Backend

---

## Overview

This Chrome Extension reads unread emails from Gmail, treats each as a **support ticket**, sends the text to the local FastAPI AI backend (`http://localhost:8000/predict`), and displays a classified, priority-sorted ticket queue inside the popup.

---

## File Structure

```
chrome-extension/
├── manifest.json          ← Manifest V3 config
├── background.js          ← Service worker: Gmail auth, email fetch, FastAPI calls
├── popup.html             ← Extension popup UI
├── popup.js               ← Popup controller / UI logic
├── generate_icons.js      ← Script to auto-generate PNG icons
├── icons/
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
└── SETUP.md               ← This file
```

---

## Step 1 — Enable Gmail API & Get OAuth2 Credentials

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (or select your existing one)
3. Enable the **Gmail API**:
   - Navigate to **APIs & Services → Library**
   - Search for "Gmail API" → Enable it
4. Create OAuth credentials:
   - Go to **APIs & Services → Credentials**
   - Click **Create Credentials → OAuth client ID**
   - Application type: **Chrome Extension**
   - For "Application ID", you'll need your Extension's ID (see Step 3)
   - Copy the **Client ID** (format: `xxxxxxxxxx.apps.googleusercontent.com`)

---

## Step 2 — Configure `manifest.json`

Open `manifest.json` and replace the placeholder:

```json
"oauth2": {
  "client_id": "YOUR_GOOGLE_OAUTH2_CLIENT_ID.apps.googleusercontent.com",
  ...
}
```

Replace `YOUR_GOOGLE_OAUTH2_CLIENT_ID.apps.googleusercontent.com` with your actual Client ID.

---

## Step 3 — Load the Extension in Chrome

1. Open Chrome and navigate to: `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **"Load unpacked"**
4. Select the `chrome-extension/` folder
5. The extension will appear in your toolbar
6. **Copy the Extension ID** shown on the extension card (e.g., `abcdefghijklmnopqrstuvwxyzabcdef`)

---

## Step 4 — Add Extension ID to OAuth Console

1. Go back to [console.cloud.google.com → Credentials](https://console.cloud.google.com/apis/credentials)
2. Edit your OAuth Client ID
3. Under **Authorized JavaScript origins**, add:
   ```
   chrome-extension://YOUR_EXTENSION_ID
   ```
4. Save

---

## Step 5 — Configure OAuth Consent Screen

1. Go to **APIs & Services → OAuth consent screen**
2. Choose **External** (for demo) or **Internal** (if using Google Workspace)
3. Fill in required fields:
   - App name: `AI Ticket Triage`
   - User support email: your email
4. Add scopes: `https://www.googleapis.com/auth/gmail.readonly`
5. Add your Gmail (`ticket.triage@gmail.com`) as a **Test User**

---

## Step 6 — Start Your FastAPI Backend

```bash
cd /path/to/CustomerSupportTicketAI
source venv/bin/activate  # or venv_prod
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Verify it's running:  
```bash
curl http://localhost:8000/health
```

---

## Step 7 — Use the Extension

1. Click the extension icon in Chrome's toolbar
2. Click **"Connect Gmail"** → Sign in with `ticket.triage@gmail.com`
3. Grant `gmail.readonly` permission
4. Click **"Activate Ticket Triage"**
5. The extension will:
   - Fetch up to 20 unread emails
   - Send each to FastAPI for classification
   - Display a sorted ticket queue (Critical → High → Medium → Low)
6. **Click any row** to see full details, keywords, and confidence scores

---

## Features

| Feature | Details |
|---|---|
| Gmail OAuth2 | Uses `chrome.identity.getAuthToken` — secure, no stored secrets |
| Email parsing | Extracts plain-text body from multipart MIME emails |
| AI classification | POST to `http://localhost:8000/predict` |
| Priority sorting | Critical → High → Medium → Low |
| Detail modal | Full body, confidence bars, keyword chips |
| Toast errors | "Backend API not reachable" if FastAPI is down |
| Persistent storage | Tickets cached in `chrome.storage.local` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Connect Gmail" button shown after clicking | Extension ID not added to OAuth credentials → see Step 4 |
| `Error: OAuth Token failed` | OAuth consent screen not configured → Step 5 |
| "Backend API not reachable" toast | FastAPI server not running → Step 6 |
| No emails shown | Gmail account has no unread inbox messages |
| Extension ID keeps changing | It changes on re-load if you remove and re-add; only add final ID to OAuth |

---

## API Contract

**Request:**
```json
POST http://localhost:8000/predict
Content-Type: application/json

{ "description": "Subject line and email body" }
```

**Response:**
```json
{
  "category": "Technical",
  "category_confidence": 0.91,
  "priority": "Critical",
  "priority_confidence": 0.88,
  "latency_ms": 55,
  "category_keywords": ["payment", "error", "gateway"],
  "priority_keywords": ["production", "down"]
}
```

---

## Security Notes

- **No secrets stored**: OAuth is handled entirely by Chrome's identity API
- **Read-only scope**: Only `gmail.readonly` is requested — no write access
- **Local API only**: FastAPI is called at `localhost:8000` — no external data transfer
- **`chrome.storage.local`**: Ticket data stays in your browser only

---

*Built for VIT Major Project — AI Customer Support Ticket Classification System*
