const BACKEND_URL = "http://localhost:8000";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "sync_tickets") {
        handleSync()
            .then(res => sendResponse(res))
            .catch(err => {
                console.error("Sync error:", err);
                sendResponse({ success: false, error: err.message });
            });
        return true;
    }
});

chrome.alarms.create("bg_sync", { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === "bg_sync") {
        console.log("[TicketIQ Sync] Running background auto-sync...");
        handleSync().catch(e => console.error("Auto sync failed:", e));
    }
});

async function handleSync() {
    // 1. Get watched email config and processed IDs
    const storageResult = await new Promise(resolve => {
        chrome.storage.local.get(['watched_email', 'processed_ids'], resolve);
    });

    const watchedEmail = storageResult.watched_email;
    if (!watchedEmail) {
        throw new Error("No watched email configured on the Dashboard.");
    }

    let processedIds = storageResult.processed_ids || [];

    // 2. Ensure Auth Token
    const token = await new Promise((resolve, reject) => {
        chrome.identity.getAuthToken({ interactive: false }, function (token) {
            if (chrome.runtime.lastError || !token) {
                reject(new Error("Missing OAuth Token. Please Connect Gmail."));
            } else {
                resolve(token);
            }
        });
    });

    // 3. Build Gmail query strictly enforcing the recipient
    let query = `is:unread to:${watchedEmail}`;

    const listResponse = await fetch(`https://gmail.googleapis.com/gmail/v1/users/me/messages?q=${encodeURIComponent(query)}&maxResults=50`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });

    if (!listResponse.ok) {
        throw new Error("Failed to search Gmail inbox via API.");
    }

    const listData = await listResponse.json();
    const messages = listData.messages || [];

    if (messages.length === 0) {
        await chrome.storage.local.set({ last_sync_time: Date.now() });
        return { success: true, ingested: 0 };
    }

    // 4. Extract email details
    const ingestedEmails = [];
    let newProcessedIds = [...processedIds];

    for (const msg of messages) {
        // Skip if already processed to prevent duplicate ingestion
        if (processedIds.includes(msg.id)) continue;
        const msgResponse = await fetch(`https://gmail.googleapis.com/gmail/v1/users/me/messages/${msg.id}?format=full`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!msgResponse.ok) continue;

        const msgData = await msgResponse.json();
        const headers = msgData.payload.headers;

        let subject = "No Subject";
        let senderEmail = "unknown@example.com";
        let receivedAt = new Date().toISOString();

        for (const h of headers) {
            if (h.name.toLowerCase() === 'subject') subject = h.value;
            if (h.name.toLowerCase() === 'from') {
                const match = h.value.match(/<([^>]+)>/);
                senderEmail = match ? match[1] : h.value;
            }
        }

        receivedAt = msgData.internalDate ? new Date(parseInt(msgData.internalDate)).toISOString() : new Date().toISOString();

        let body = getMessageBody(msgData.payload);
        body = body ? body.trim() : "No plain text content.";

        console.log("[TicketIQ Sync] Extracted ticket:", {
            subject: subject,
            body: body,
            sender_email: senderEmail,
            received_at: receivedAt
        });

        ingestedEmails.push({
            subject: subject,
            body: body,
            sender_email: senderEmail,
            recipient_email: watchedEmail,
            received_at: receivedAt,
            source: "gmail"
        });

        newProcessedIds.push(msg.id);
    }

    // 5. POST to Ingestion endpoint if valid emails found
    if (ingestedEmails.length > 0) {
        const backendRes = await fetch(`${BACKEND_URL}/api/tickets/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(ingestedEmails)
        });

        if (!backendRes.ok) {
            throw new Error(`Ingest API failed: ${backendRes.status}`);
        }

        const backendData = await backendRes.json();

        // Keep array manageable
        if (newProcessedIds.length > 200) newProcessedIds = newProcessedIds.slice(-200);

        await chrome.storage.local.set({
            last_sync_time: Date.now(),
            processed_ids: newProcessedIds
        });
        return { success: true, ingested: backendData.ingested, failed: backendData.failed };
    }

    await chrome.storage.local.set({ last_sync_time: Date.now() });
    return { success: true, ingested: 0 };
}

function getMessageBody(payload) {
    let body = "";
    let htmlBody = "";

    function traverseParts(parts) {
        for (const part of parts) {
            if (part.mimeType === "text/plain" && part.body && part.body.data) {
                body += decodeBase64Url(part.body.data);
            } else if (part.mimeType === "text/html" && part.body && part.body.data) {
                htmlBody += decodeBase64Url(part.body.data);
            } else if (part.parts) {
                traverseParts(part.parts);
            }
        }
    }

    if (payload.parts) {
        traverseParts(payload.parts);
    } else if (payload.body && payload.body.data) {
        if (payload.mimeType === "text/html") {
            htmlBody = decodeBase64Url(payload.body.data);
        } else {
            body = decodeBase64Url(payload.body.data);
        }
    }

    if (body) {
        return body;
    }

    if (htmlBody) {
        return stripHtmlTags(htmlBody);
    }

    return "";
}

function stripHtmlTags(html) {
    let text = html.replace(/<br\s*[\/]?>/gi, '\n');
    text = text.replace(/<\/p>/gi, '\n\n');
    text = text.replace(/<[^>]+>/g, '');
    text = text.replace(/&nbsp;/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
    return text.trim();
}

function decodeBase64Url(base64UrlStr) {
    try {
        let base64 = base64UrlStr.replace(/-/g, '+').replace(/_/g, '/');
        const padding = base64.length % 4;
        if (padding > 0) {
            base64 += '='.repeat(4 - padding);
        }
        const binaryStr = atob(base64);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
        }
        return new TextDecoder('utf-8').decode(bytes);
    } catch (e) {
        return "";
    }
}
