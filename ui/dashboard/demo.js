// =====================================================================
// demo.js — Live Demo Engine for TicketIQ
// Uses the REAL /predict and /api/tickets/ingest endpoints (localhost)
// Fully offline — no external APIs required
// =====================================================================

// --- Demo ticket scenarios (realistic, varied) ---
const DEMO_TICKETS = [
    {
        subject: "URGENT: Payment gateway completely broken",
        body: "Our payment gateway is completely broken since 2 hours. None of our customers can complete checkout. We are losing revenue every minute. Please fix immediately.",
        sender_email: "ops@retailco.com",
        source: "demo"
    },
    {
        subject: "Login page returns 500 Internal Server Error",
        body: "NOW: Login page returns 500 internal server error. Production is completely blocked. No users can access the system. IMMEDIATELY fix needed. This is blocking my work.",
        sender_email: "devteam@startup.io",
        source: "demo"
    },
    {
        subject: "Minor: Question about my invoice amount",
        body: "No hurry: I noticed a discrepancy in my invoice INV-202, the amount is different from what I agreed to. Could you check when you have time? Minor issue.",
        sender_email: "customer@gmail.com",
        source: "demo"
    },
    {
        subject: "Security breach detected on our account",
        body: "CRITICAL: We have detected unauthorized access on our account. Multiple logins from unknown IPs. We are losing customer data right now. Need immediate response.",
        sender_email: "cto@financeapp.com",
        source: "demo"
    },
    {
        subject: "Dashboard loads slowly sometimes",
        body: "The dashboard is loading slowly sometimes, about 5-8 seconds. It still works but it's a bit frustrating. Not urgent, just reporting in case it helps.",
        sender_email: "user123@company.com",
        source: "demo"
    },
    {
        subject: "How do I reset my MFA settings?",
        body: "Routine: How do I reset the MFA system? I am not seeing any updates. I need to change settings. Need this fixed soon.",
        sender_email: "john.doe@acme.com",
        source: "demo"
    },
    {
        subject: "BLOCKER: Entire system down, no one can login",
        body: "The entire system is down. No one in our organization can login. This is a production blocker affecting 200+ users. We need immediate escalation.",
        sender_email: "manager@enterprise.com",
        source: "demo"
    },
    {
        subject: "Refund not processed after 7 days",
        body: "Important: My refund billing request was not processed after 7 days. The billing amount of $245 is still showing. Please review my account billing and process the refund. Urgent attention needed.",
        sender_email: "angry.customer@yahoo.com",
        source: "demo"
    },
    {
        subject: "Export button not working sometimes",
        body: "The export to CSV button is not working sometimes. It works maybe 70% of the time. When it fails, no error message is shown. Feature is slow but still mostly functional.",
        sender_email: "analyst@bigcorp.com",
        source: "demo"
    },
    {
        subject: "App crashes when trying to checkout",
        body: "Our mobile app crashes every time a user tries to complete checkout. This happens on both iOS and Android. Cannot complete payment, getting server error. High impact.",
        sender_email: "mobile-team@ecommerce.com",
        source: "demo"
    },
    {
        subject: "Profile data showing as inactive incorrectly",
        body: "My profile data is showing as inactive even though I'm actively using the account. The settings area also feels strange. Can someone check the profile panel?",
        sender_email: "p.kumar@tech.net",
        source: "demo"
    },
    {
        subject: "Billing discrepancy — charged twice",
        body: "Important: Why is there a billing discrepancy? I was charged twice this month for the same plan. The amount is unusually high. Please check and refund the duplicate. Urgent attention needed.",
        sender_email: "billing-issue@mail.com",
        source: "demo"
    }
];

let simInterval = null;
let simIndex = 0;
let demoPanelOpen = false;
let WATCHED_EMAIL = 'demo@ticketiq.ai';

// Ensure demo email is always configured so ingest works
async function ensureDemoEmail() {
    try {
        const res = await fetch('/api/config/email');
        const data = await res.json();
        if (!data.email) {
            await fetch('/api/config/email', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: WATCHED_EMAIL })
            });
        } else {
            WATCHED_EMAIL = data.email;
        }
    } catch (e) {
        console.warn('Could not ensure demo email:', e);
    }
}
ensureDemoEmail();

// Toggle demo panel
window.toggleDemoPanel = function () {
    const panel = document.getElementById('demo-panel');
    demoPanelOpen = !demoPanelOpen;
    if (demoPanelOpen) {
        panel.classList.remove('hidden');
        // Smooth scroll to panel
        panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
        panel.classList.add('hidden');
        stopSimulation();
    }
};

// Classify demo input using real model
window.classifyDemoInput = async function () {
    const text = document.getElementById('demo-input').value.trim();
    if (!text) {
        showToast('Please type a ticket message first', 'warning');
        return;
    }
    const btn = document.getElementById('classify-btn');
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>Classifying...';
    btn.disabled = true;

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!res.ok) throw new Error('API error');
        const data = await res.json();
        showClassifyResult(data);
    } catch (e) {
        showToast('API error — is the server running? (python -m uvicorn api.main:app)', 'error');
    } finally {
        btn.innerHTML = '<i class="fa-solid fa-brain"></i> Classify with ML Model';
        btn.disabled = false;
    }
};

function showClassifyResult(data) {
    const result = document.getElementById('classify-result');
    result.classList.remove('hidden');

    const priColors = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e' };
    const pri = (data.priority || '').toLowerCase();
    const color = priColors[pri] || '#94a3b8';

    document.getElementById('res-category').textContent = data.category || '—';
    document.getElementById('res-priority').innerHTML = `<span style="color:${color}">${data.priority || '—'}</span>`;
    document.getElementById('res-cat-conf').textContent = ((data.category_confidence || 0) * 100).toFixed(1) + '%';
    document.getElementById('res-pri-conf').textContent = ((data.priority_confidence || 0) * 100).toFixed(1) + '%';

    const kw = [...(data.category_keywords || []), ...(data.priority_keywords || [])].filter(Boolean);
    document.getElementById('res-keywords').textContent = kw.length ? kw.join(', ') : 'none';
    document.getElementById('res-latency').textContent = (data.latency_ms || 0).toFixed(1);
}

// Submit demo input as a ticket via real ingest endpoint
window.injectDemoInput = async function () {
    const text = document.getElementById('demo-input').value.trim();
    if (!text) {
        showToast('Please type a ticket message first', 'warning');
        return;
    }

    const btn = document.getElementById('inject-btn');
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>Submitting...';
    btn.disabled = true;

    await ensureDemoEmail();

    const now = new Date().toISOString();
    const payload = [{
        subject: text.substring(0, 60) + (text.length > 60 ? '...' : ''),
        body: text,
        sender_email: 'demo-user@ticketiq.ai',
        recipient_email: WATCHED_EMAIL,
        received_at: now,
        source: 'demo'
    }];

    try {
        const res = await fetch('/api/tickets/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.ingested > 0) {
            showToast('✓ Ticket classified and added to dashboard!', 'success');
            document.getElementById('demo-input').value = '';
            document.getElementById('classify-result').classList.add('hidden');
            if (typeof fetchData === 'function') fetchData();
        } else {
            showToast('Ingestion failed — check email config', 'error');
        }
    } catch (e) {
        showToast('API error — server not reachable', 'error');
    } finally {
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Submit as Ticket';
        btn.disabled = false;
    }
};

// Auto-simulation
window.startSimulation = async function () {
    if (simInterval) return;

    await ensureDemoEmail();

    document.getElementById('sim-start-btn').disabled = true;
    document.getElementById('sim-stop-btn').disabled = false;

    const speed = parseInt(document.getElementById('sim-speed').value);
    simIndex = 0;
    logSimStatus('<span class="text-emerald-400">▶ Simulation started.</span> Sending tickets...');

    await sendNextSimTicket();
    simInterval = setInterval(async () => {
        if (simIndex >= DEMO_TICKETS.length) {
            stopSimulation();
            logSimStatus('<span class="text-emerald-400">✓ All demo tickets sent!</span> Check the dashboard above.');
            return;
        }
        await sendNextSimTicket();
    }, speed);
};

async function sendNextSimTicket() {
    if (simIndex >= DEMO_TICKETS.length) return;
    const ticket = DEMO_TICKETS[simIndex];
    simIndex++;

    logSimStatus(`<span class="text-yellow-300">→ [${simIndex}/${DEMO_TICKETS.length}]</span> Sending: <span class="text-white">"${ticket.subject.substring(0, 45)}..."</span>`);

    const now = new Date().toISOString();
    const payload = [{
        subject: ticket.subject,
        body: ticket.body,
        sender_email: ticket.sender_email,
        recipient_email: WATCHED_EMAIL,
        received_at: now,
        source: ticket.source
    }];

    try {
        const res = await fetch('/api/tickets/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.ingested > 0) {
            logSimStatus(`<span class="text-emerald-400">✓ Classified</span> → "${ticket.subject.substring(0, 35)}..."`);
            if (typeof fetchData === 'function') fetchData();
        } else {
            logSimStatus(`<span class="text-red-400">✗ Failed</span> — ticket skipped`);
        }
    } catch (e) {
        logSimStatus(`<span class="text-red-400">✗ API error</span> — is the server running?`);
    }
}

window.stopSimulation = function () {
    if (simInterval) {
        clearInterval(simInterval);
        simInterval = null;
    }
    document.getElementById('sim-start-btn').disabled = false;
    document.getElementById('sim-stop-btn').disabled = true;
    logSimStatus('<span class="text-slate-400">■ Simulation stopped.</span>');
};

function logSimStatus(html) {
    const el = document.getElementById('sim-status');
    el.innerHTML = html;
}

// Pre-load demo text samples on click
const SAMPLE_TEXTS = [
    "Payment gateway completely broken since 2 hours, users cannot checkout",
    "My account was charged twice this month, please refund the duplicate",
    "How do I reset my 2FA settings? I got a new phone",
    "System is completely down, no one can login, production blocker",
    "Dashboard loads slowly sometimes but still works fine",
];
let sampleIdx = 0;

// Add sample text button helper (called from HTML if needed)
window.loadSampleText = function () {
    document.getElementById('demo-input').value = SAMPLE_TEXTS[sampleIdx % SAMPLE_TEXTS.length];
    sampleIdx++;
};
