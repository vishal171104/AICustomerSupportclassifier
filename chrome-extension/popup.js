const BACKEND_URL = "http://localhost:8000";

document.addEventListener('DOMContentLoaded', async () => {
    const authSection = document.getElementById('auth-section');
    const syncSection = document.getElementById('sync-section');
    const btnAuth = document.getElementById('btn-auth');
    const btnSync = document.getElementById('btn-sync');

    // 1. Check if we already have a valid OAuth token
    chrome.identity.getAuthToken({ interactive: false }, async (token) => {
        if (chrome.runtime.lastError || !token) {
            authSection.style.display = 'block';
            syncSection.style.display = 'none';
        } else {
            authSection.style.display = 'none';
            syncSection.style.display = 'flex';
            await checkDashboardConfig();
            updateLastSyncedUI();
        }
    });

    // 2. Auth Flow — interactive login
    btnAuth.addEventListener('click', () => {
        chrome.identity.getAuthToken({ interactive: true }, async (token) => {
            if (chrome.runtime.lastError || !token) {
                showToast("Failed to connect Gmail. Please try again.", true);
                return;
            }
            authSection.style.display = 'none';
            syncSection.style.display = 'flex';
            await checkDashboardConfig();
            updateLastSyncedUI();
        });
    });

    // 3. Sync Flow — runs entirely in popup, no background worker needed
    btnSync.addEventListener('click', async () => {
        btnSync.disabled = true;
        btnSync.innerHTML = '<span class="spinner"></span> Syncing...';

        try {
            const result = await doSync();
            if (result.ingested > 0) {
                showToast(`✅ ${result.ingested} ticket(s) ingested successfully!`);
            } else {
                showToast(`No new unread emails found for this address.`);
            }
            // Save last sync time
            await chrome.storage.local.set({ last_sync_time: Date.now() });
            updateLastSyncedUI();
        } catch (err) {
            showToast(`Sync Failed: ${err.message}`, true);
            console.error('[TicketIQ] Sync error:', err);
        } finally {
            btnSync.disabled = false;
            btnSync.innerText = 'Sync Now';
        }
    });
});

// ─── Core sync function (runs in popup context, no message passing) ───────────

async function doSync() {
    // Step 1: Get watched email from backend API (source of truth)
    const configRes = await fetch(`${BACKEND_URL}/api/config/email`);
    if (!configRes.ok) throw new Error("Cannot reach backend API at localhost:8000");
    const configData = await configRes.json();
    const watchedEmail = configData.email;
    if (!watchedEmail) throw new Error("No watched email configured on the Dashboard.");

    // Also sync to storage so background auto-sync works
    await chrome.storage.local.set({ watched_email: watchedEmail });

    // Step 2: Get OAuth token (non-interactive — popup already proved we have one)
    const token = await new Promise((resolve, reject) => {
        chrome.identity.getAuthToken({ interactive: false }, (t) => {
            if (chrome.runtime.lastError || !t) {
                // Try interactive as fallback
                chrome.identity.getAuthToken({ interactive: true }, (t2) => {
                    if (chrome.runtime.lastError || !t2) {
                        reject(new Error("Gmail auth failed. Please reconnect."));
                    } else {
                        resolve(t2);
                    }
                });
            } else {
                resolve(t);
            }
        });
    });

    // Step 3: Query Gmail for emails sent TO the watched address (don't rely on unread status)
    const query = `to:${watchedEmail}`;
    const listRes = await fetch(
        `https://gmail.googleapis.com/gmail/v1/users/me/messages?q=${encodeURIComponent(query)}&maxResults=50`,
        { headers: { 'Authorization': `Bearer ${token}` } }
    );

    if (!listRes.ok) {
        // Token might be stale — remove and ask to re-auth
        if (listRes.status === 401) {
            await new Promise(r => chrome.identity.removeCachedAuthToken({ token }, r));
            throw new Error("Gmail token expired. Please click 'Connect Gmail' to re-authenticate.");
        }
        throw new Error(`Gmail API error: ${listRes.status}`);
    }

    const listData = await listRes.json();
    const messages = listData.messages || [];
    if (messages.length === 0) return { ingested: 0 };

    // Step 4: Load already-processed IDs to avoid duplicates
    const stored = await new Promise(r => chrome.storage.local.get(['processed_ids'], r));
    let processedIds = stored.processed_ids || [];

    // Step 5: Fetch each message detail and build ingest payload
    const toIngest = [];
    const newProcessedIds = [...processedIds];

    for (const msg of messages) {
        if (processedIds.includes(msg.id)) continue;

        const msgRes = await fetch(
            `https://gmail.googleapis.com/gmail/v1/users/me/messages/${msg.id}?format=full`,
            { headers: { 'Authorization': `Bearer ${token}` } }
        );
        if (!msgRes.ok) continue;

        const msgData = await msgRes.json();
        const headers = msgData.payload.headers;

        let subject = "No Subject";
        let senderEmail = "unknown@example.com";
        let receivedAt = new Date().toISOString();

        for (const h of headers) {
            if (h.name.toLowerCase() === 'subject') subject = h.value;
            if (h.name.toLowerCase() === 'from') {
                const match = h.value.match(/<([^>]+)>/);
                senderEmail = match ? match[1] : h.value.trim();
            }
        }
        if (msgData.internalDate) {
            receivedAt = new Date(parseInt(msgData.internalDate)).toISOString();
        }

        let body = getMessageBody(msgData.payload) || "No plain text content.";
        body = body.trim();

        toIngest.push({
            subject,
            body,
            sender_email: senderEmail,
            recipient_email: watchedEmail,
            received_at: receivedAt,
            source: "gmail"
        });
        newProcessedIds.push(msg.id);
    }

    if (toIngest.length === 0) return { ingested: 0 };

    // Step 6: POST to backend ingest endpoint
    const ingestRes = await fetch(`${BACKEND_URL}/api/tickets/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toIngest)
    });

    if (!ingestRes.ok) {
        const errText = await ingestRes.text();
        throw new Error(`Backend ingest failed (${ingestRes.status}): ${errText}`);
    }

    const ingestData = await ingestRes.json();

    // Save updated processed IDs (keep last 500)
    if (newProcessedIds.length > 500) newProcessedIds.splice(0, newProcessedIds.length - 500);
    await chrome.storage.local.set({ processed_ids: newProcessedIds });

    return { ingested: ingestData.ingested || toIngest.length, failed: ingestData.failed || 0 };
}

// ─── Check backend config and update UI ──────────────────────────────────────

async function checkDashboardConfig() {
    const btnSync = document.getElementById('btn-sync');
    const statusEl = document.getElementById('config-status');
    const errorEl = document.getElementById('config-error');

    try {
        const res = await fetch(`${BACKEND_URL}/api/config/email`);
        const data = await res.json();

        if (data.email) {
            statusEl.style.display = 'block';
            statusEl.innerText = `🟢 Watching: ${data.email}`;
            errorEl.style.display = 'none';
            btnSync.disabled = false;
            // Keep storage in sync with API
            await chrome.storage.local.set({ watched_email: data.email });
        } else {
            statusEl.style.display = 'none';
            errorEl.style.display = 'block';
            errorEl.innerText = '⚠️ Set a watched email on the Dashboard first.';
            btnSync.disabled = true;
            await chrome.storage.local.remove('watched_email');
        }
    } catch (err) {
        statusEl.style.display = 'none';
        errorEl.style.display = 'block';
        errorEl.innerText = "❌ Cannot connect to backend (localhost:8000).";
        btnSync.disabled = true;
    }
}

// ─── Last synced display ──────────────────────────────────────────────────────

function updateLastSyncedUI() {
    chrome.storage.local.get(['last_sync_time'], (res) => {
        const el = document.getElementById('last-synced');
        if (!el) return;
        if (res.last_sync_time) {
            el.innerText = new Date(res.last_sync_time).toLocaleString();
        } else {
            el.innerText = "Never";
        }
    });
}

// ─── Toast notifications ──────────────────────────────────────────────────────

function showToast(msg, isError = false) {
    const t = document.getElementById('toast');
    if (!t) return;
    t.innerText = msg;
    t.style.background = isError ? '#ef4444' : '#22c55e';
    t.style.display = 'block';
    setTimeout(() => { t.style.display = 'none'; }, 5000);
}

// ─── Gmail message body extraction ───────────────────────────────────────────

function getMessageBody(payload) {
    let plainBody = "";
    let htmlBody = "";

    function traverse(parts) {
        for (const part of parts) {
            if (part.mimeType === "text/plain" && part.body?.data) {
                plainBody += decodeBase64Url(part.body.data);
            } else if (part.mimeType === "text/html" && part.body?.data) {
                htmlBody += decodeBase64Url(part.body.data);
            } else if (part.parts) {
                traverse(part.parts);
            }
        }
    }

    if (payload.parts) {
        traverse(payload.parts);
    } else if (payload.body?.data) {
        if (payload.mimeType === "text/html") {
            htmlBody = decodeBase64Url(payload.body.data);
        } else {
            plainBody = decodeBase64Url(payload.body.data);
        }
    }

    if (plainBody) return plainBody;
    if (htmlBody) return stripHtml(htmlBody);
    return "";
}

function stripHtml(html) {
    return html
        .replace(/<br\s*[/]?>/gi, '\n')
        .replace(/<\/p>/gi, '\n\n')
        .replace(/<[^>]+>/g, '')
        .replace(/&nbsp;/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .trim();
}

function decodeBase64Url(str) {
    try {
        let b64 = str.replace(/-/g, '+').replace(/_/g, '/');
        const pad = b64.length % 4;
        if (pad) b64 += '='.repeat(4 - pad);
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return new TextDecoder('utf-8').decode(bytes);
    } catch {
        return "";
    }
}
