// State & Variables
let currentEmail = null;
let priorityChart = null;
let tickets = [];
let autoRefreshTimer = null;
let isRefreshing = false;

// DOM Elements
const REFRESH_BTN = document.getElementById('refresh-btn');
const REFRESH_ICON = document.getElementById('refresh-icon');
const AUTO_TOGGLE = document.getElementById('auto-refresh-toggle');
const SEARCH_INPUT = document.getElementById('search-input');
const FILTER_SORT = document.getElementById('filter-sort');
const FILTER_PRI = document.getElementById('filter-priority');
const FILTER_SRC = document.getElementById('filter-source');
const FILTER_STATUS = document.getElementById('filter-status');

// Chart Colors mapping
const PRI_COLORS = {
    'critical': '#ef4444',
    'high': '#f97316',
    'medium': '#eab308',
    'low': '#22c55e'
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initClock();
    initChart();
    loadConfig();
    fetchData();

    // Event Listeners
    REFRESH_BTN.addEventListener('click', () => { manualRefresh(); });

    AUTO_TOGGLE.addEventListener('change', (e) => {
        if (e.target.checked) {
            autoRefreshTimer = setInterval(fetchData, 60000);
            showToast('Auto-refresh enabled (60s)', 'info');
        } else {
            clearInterval(autoRefreshTimer);
            showToast('Auto-refresh disabled', 'info');
        }
    });

    [FILTER_SORT, FILTER_PRI, FILTER_SRC, FILTER_STATUS].forEach(el => {
        el.addEventListener('change', fetchData);
    });

    // Debounce search
    let searchTimeout;
    SEARCH_INPUT.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(fetchData, 300);
    });

    // Settings
    document.getElementById('btn-save-email').addEventListener('click', saveEmail);
    document.getElementById('btn-remove-email').addEventListener('click', removeEmail);

    // Card filters
    document.querySelectorAll('.card-filter').forEach(card => {
        card.addEventListener('click', () => {
            const f = card.dataset.filter;
            if (f === 'all') FILTER_PRI.value = 'all';
            else if (f === 'critical') FILTER_PRI.value = 'critical';
            else if (f === 'high') FILTER_PRI.value = 'high';
            else if (f === 'lowmed') FILTER_PRI.value = 'all'; // Show all tickets when clicking Low/Med card
            fetchData();
        });
    });
});

// Clock
function initClock() {
    setInterval(() => {
        const d = new Date();
        document.getElementById('live-clock').textContent = d.toLocaleTimeString('en-US', { hour12: false });
    }, 1000);
}

// Chart.js init
function initChart() {
    const ctx = document.getElementById('priorityChart').getContext('2d');
    priorityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'High', 'Medium', 'Low'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: [PRI_COLORS.critical, PRI_COLORS.high, PRI_COLORS.medium, PRI_COLORS.low],
                borderWidth: 0,
                cutout: '70%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            onHover: (event, chartElement) => {
                event.native.target.style.cursor = chartElement.length ? 'pointer' : 'default';
            },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    const map = ['critical', 'high', 'medium', 'low'];
                    FILTER_PRI.value = map[idx];
                    fetchData();
                }
            }
        }
    });
}

async function manualRefresh() {
    if (isRefreshing) return;
    isRefreshing = true;
    REFRESH_BTN.disabled = true;
    REFRESH_ICON.classList.add('spin');

    if (!currentEmail) {
        showToast('Configure a support email in Settings before syncing.', 'warning');
    }

    await fetchData();

    showToast('Updated just now', 'success');

    setTimeout(() => {
        REFRESH_ICON.classList.remove('spin');
        REFRESH_BTN.disabled = false;
        isRefreshing = false;
    }, 3000);
}

// Data Fetching
async function fetchData() {
    try {
        const sort = FILTER_SORT.value;
        const pri = FILTER_PRI.value;
        const src = FILTER_SRC.value;
        const search = SEARCH_INPUT.value;
        const status = FILTER_STATUS.value;

        const params = new URLSearchParams({ sort, filter: pri, source: src, search, status });
        const statParams = new URLSearchParams({ status });
        const [ticketsRes, statsRes] = await Promise.all([
            fetch(`/api/tickets?${params}`),
            fetch(`/api/tickets/stats?${statParams}`)
        ]);

        tickets = await ticketsRes.json();
        const stats = await statsRes.json();

        renderStats(stats);
        renderChart(stats);
        renderTable(tickets);
    } catch (err) {
        console.error('Failed to fetch data', err);
        showToast('Connection error connecting to API', 'error');
    }
}

function renderStats(stats) {
    document.getElementById('stat-total').textContent = stats.total || 0;
    document.getElementById('stat-critical').textContent = stats.by_priority.critical || 0;
    document.getElementById('stat-high').textContent = stats.by_priority.high || 0;
    document.getElementById('stat-lowmed').textContent = (stats.by_priority.medium + stats.by_priority.low) || 0;
}

function renderChart(stats) {
    const data = [
        stats.by_priority.critical || 0,
        stats.by_priority.high || 0,
        stats.by_priority.medium || 0,
        stats.by_priority.low || 0
    ];

    const sum = data.reduce((a, b) => a + b, 0);
    const canvas = document.getElementById('priorityChart');
    const emptyState = document.getElementById('chart-empty');

    if (sum === 0) {
        canvas.style.opacity = '0.1';
        emptyState.classList.remove('hidden');
        data[3] = 1; // dummy data to draw grey ring
        priorityChart.data.datasets[0].backgroundColor = Array(4).fill('#cbd5e1'); // grey out all segments
    } else {
        canvas.style.opacity = '1';
        emptyState.classList.add('hidden');
        priorityChart.data.datasets[0].backgroundColor = [PRI_COLORS.critical, PRI_COLORS.high, PRI_COLORS.medium, PRI_COLORS.low];
    }

    priorityChart.data.datasets[0].data = data;
    priorityChart.update();
}

function renderTable(data) {
    const tbody = document.getElementById('ticket-table-body');
    const emptyState = document.getElementById('empty-state');

    tbody.innerHTML = '';

    if (data.length === 0) {
        emptyState.classList.remove('hidden');
        if (!currentEmail) {
            document.getElementById('empty-message').textContent = 'Configure a support email in Settings to begin.';
        } else {
            document.getElementById('empty-message').textContent = 'No tickets match the current filters. Connect Gmail and click Sync to ingest incoming email.';
        }
        return;
    }

    emptyState.classList.add('hidden');

    data.forEach((t, rowIdx) => {
        // Priority Badge
        let badgeColor = 'bg-slate-100 text-slate-800';
        let p = t.priority ? t.priority.toLowerCase() : '';
        if (p === 'critical') badgeColor = 'bg-red-100 text-red-800';
        if (p === 'high') badgeColor = 'bg-orange-100 text-orange-800';
        if (p === 'medium') badgeColor = 'bg-yellow-100 text-yellow-800';
        if (p === 'low') badgeColor = 'bg-green-100 text-green-800';

        // Confidence bar
        const conf = t.confidence_score ? Math.round(t.confidence_score * 100) : 0;
        const cbColor = conf > 80 ? 'bg-green-500' : (conf > 50 ? 'bg-yellow-500' : 'bg-red-500');

        // Source
        const isGmail = t.source && t.source.toLowerCase() === 'gmail';
        const srcHtml = isGmail ?
            `<span title="Received via Gmail sync at ${new Date(t.ingested_at).toLocaleString()}"><i class="fa-brands fa-google text-slate-400"></i></span>` :
            `<span class="text-xs text-slate-500">Manual</span>`;

        const receivedStr = new Date(t.received_at || t.ingested_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

        const statusView = FILTER_STATUS.value;
        let actionButtons = ``;
        if (statusView === 'open') {
            actionButtons = `
                <button class="text-accent hover:text-indigo-900 p-1" onclick="toggleExpand(${t.id})" title="View Details"><i class="fa-regular fa-eye"></i></button>
                <button class="text-green-600 hover:text-green-900 p-1" onclick="resolveTicket(${t.id})" title="Resolve"><i class="fa-solid fa-check"></i></button>
                <button class="text-red-500 hover:text-red-700 p-1" onclick="escalateTicket(${t.id})" title="Escalate"><i class="fa-solid fa-arrow-trend-up"></i></button>
            `;
        } else {
            actionButtons = `
                <button class="text-accent hover:text-indigo-900 p-1" onclick="toggleExpand(${t.id})" title="View Details"><i class="fa-regular fa-eye"></i></button>
                <button class="text-blue-500 hover:text-blue-700 p-1" onclick="reopenTicket(${t.id})" title="Reopen Ticket"><i class="fa-solid fa-rotate-left"></i></button>
            `;
        }

        const tr = document.createElement('tr');
        tr.id = `ticket-row-${t.id}`;
        tr.className = 'hover:bg-slate-50 transition border-b border-slate-100';
        tr.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap">
                <div class="text-sm font-semibold text-slate-700">${rowIdx + 1}</div>
                <div class="text-xs text-slate-400">id:${t.id}</div>
            </td>
            <td class="px-6 py-4">
                <div class="text-sm font-medium text-slate-900 truncate max-w-xs" title="${t.subject}">${t.subject || 'No Subject'}</div>
                <div class="text-xs text-slate-500 break-all">${t.sender_email || 'Unknown'}</div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-500">${receivedStr}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-center">${srcHtml}</td>
            <td class="px-6 py-4 whitespace-nowrap">
                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${badgeColor}">
                    ${t.priority || 'Unknown'}
                </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <div class="flex items-center">
                    <span class="text-xs text-slate-500 w-8">${conf}%</span>
                    <div class="w-20 bg-slate-200 rounded-full h-1.5 ml-2">
                        <div class="${cbColor} h-1.5 rounded-full" style="width: ${conf}%"></div>
                    </div>
                </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium flex gap-2 justify-end">
                ${actionButtons}
            </td>
        `;

        // Expansion row
        const trExp = document.createElement('tr');
        trExp.id = `expand-${t.id}`;
        trExp.className = 'bg-slate-50 text-sm hidden'; // Use hidden utility class
        trExp.innerHTML = `
            <td colspan="7" class="px-8 py-6 border-b border-slate-200">
                <div class="flex flex-col gap-4">
                    <div>
                        <span class="font-semibold text-slate-700 block mb-1">Raw Ticket Body:</span>
                        <div class="p-3 bg-white border border-slate-200 rounded-md text-slate-600 whitespace-pre-wrap font-mono text-xs max-h-40 overflow-y-auto">${t.body || 'No content...'}</div>
                    </div>
                    <div>
                        <span class="font-semibold text-slate-700 block mb-1">ML Inference Explanation:</span>
                        <div class="flex gap-4">
                            <div class="bg-indigo-50 text-indigo-800 px-3 py-2 rounded-md border border-indigo-100 flex-1">
                                <span class="block text-xs uppercase tracking-wider font-semibold opacity-70 mb-1">Category</span>
                                <span class="font-medium">${t.predicted_label || 'Unknown'}</span>
                            </div>
                            <div class="bg-orange-50 text-orange-800 px-3 py-2 rounded-md border border-orange-100 flex-1">
                                <span class="block text-xs uppercase tracking-wider font-semibold opacity-70 mb-1">Priority</span>
                                <span class="font-medium">${t.priority || 'Unknown'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </td>
        `;

        tbody.appendChild(tr);
        tbody.appendChild(trExp);
    });
}

window.toggleExpand = function (id) {
    const el = document.getElementById(`expand-${id}`);
    if (el) {
        if (el.classList.contains('hidden')) el.classList.remove('hidden');
        else el.classList.add('hidden');
    }
}

window.resolveTicket = async function (id) {
    try {
        await fetch(`/api/tickets/${id}/status`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'resolved' })
        });
        showToast(`Ticket #${id} marked as resolved!`, 'success');
        fetchData();
    } catch (e) {
        showToast('Error resolving ticket', 'error');
    }
};

window.reopenTicket = async function (id) {
    try {
        await fetch(`/api/tickets/${id}/status`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'open' })
        });
        showToast(`Ticket #${id} reopened!`, 'success');
        fetchData();
    } catch (e) {
        showToast('Error reopening ticket', 'error');
    }
};

window.escalateTicket = function (id) {
    showToast(`Ticket #${id} escalated to human intervention queue!`, 'warning');
};

// Settings & Config
async function loadConfig() {
    try {
        const res = await fetch('/api/config/email');
        const data = await res.json();

        currentEmail = data.email;
        updateEmailIndicator();
    } catch (err) {
        console.error('Failed to load config', err);
    }
}

function updateEmailIndicator() {
    const ind = document.getElementById('email-indicator');
    const input = document.getElementById('settings-email');
    const rmvBtn = document.getElementById('btn-remove-email');

    if (currentEmail) {
        ind.className = "ml-6 px-3 py-1 rounded-full text-xs font-semibold bg-green-500 text-white cursor-pointer hover:bg-green-600 transition flex items-center shadow-sm";
        ind.innerHTML = `<i class="fa-solid fa-envelope-circle-check mr-1"></i> Watching: ${currentEmail}`;
        input.value = currentEmail;
        rmvBtn.disabled = false;

        // Update empty state if showing
        const emp = document.getElementById('empty-message');
        if (emp && tickets.length === 0) emp.textContent = 'No tickets match the current filters. Connect Gmail and click Sync to ingest incoming email.';
    } else {
        ind.className = "ml-6 px-3 py-1 rounded-full text-xs font-semibold bg-red-500 text-white cursor-pointer hover:bg-red-600 transition flex items-center shadow-sm";
        ind.innerHTML = `<i class="fa-solid fa-circle-exclamation mr-1"></i> No email configured`;
        input.value = '';
        rmvBtn.disabled = true;

        const emp = document.getElementById('empty-message');
        if (emp && tickets.length === 0) emp.textContent = 'Configure a support email in Settings to begin.';
    }
}

async function saveEmail() {
    const email = document.getElementById('settings-email').value.trim();
    if (!email || !email.includes('@')) {
        showToast('Please enter a valid email address', 'error');
        return;
    }

    const btn = document.getElementById('btn-save-email');
    btn.innerHTML = '<i class="fa-solid fa-spinner spin mr-2"></i> Saving...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/config/email', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });

        const data = await res.json();
        if (data.success) {
            currentEmail = data.email;
            updateEmailIndicator();
            closeSettings();
            showToast(`${currentEmail} is now being watched`, 'success');
        } else {
            showToast('Failed to save email configuration', 'error');
        }
    } catch (err) {
        showToast('API connection error', 'error');
    } finally {
        btn.innerHTML = 'Save Email';
        btn.disabled = false;
    }
}

async function removeEmail() {
    const btn = document.getElementById('btn-remove-email');
    btn.innerHTML = '<i class="fa-solid fa-spinner spin mr-2"></i> Removing...';
    btn.disabled = true;

    try {
        await fetch('/api/config/email', { method: 'DELETE' });
        currentEmail = null;
        updateEmailIndicator();
        closeSettings();
        showToast('No email configured. Sync is paused.', 'warning');
    } catch (err) {
        showToast('API connection error', 'error');
    } finally {
        btn.innerHTML = 'Remove';
        btn.disabled = false;
    }
}

// openSettings and closeSettings are defined in index.html inline script
// to ensure they are available before app.js loads. These window aliases
// keep backward compatibility with onclick attributes in the HTML.
if (!window.openSettings) {
    window.openSettings = function () {
        document.getElementById('settings-modal').classList.remove('hidden');
        setTimeout(() => document.getElementById('settings-email').focus(), 100);
    };
}
if (!window.closeSettings) {
    window.closeSettings = function () {
        document.getElementById('settings-modal').classList.add('hidden');
    };
}


// Toasts
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');

    let colors = 'bg-slate-800 text-white';
    let icon = 'fa-info-circle text-blue-400';
    if (type === 'success') { colors = 'bg-green-800 text-white'; icon = 'fa-check-circle text-green-300'; }
    if (type === 'error') { colors = 'bg-red-800 text-white'; icon = 'fa-circle-xmark text-red-300'; }
    if (type === 'warning') { colors = 'bg-yellow-800 text-white'; icon = 'fa-triangle-exclamation text-yellow-300'; }

    toast.className = `flex items-center p-4 rounded-lg shadow-lg ${colors} min-w-[250px] transform transition-all duration-300 translate-y-10 opacity-0`;
    toast.innerHTML = `<i class="fa-solid ${icon} mr-3 text-lg"></i> <span class="text-sm font-medium">${message}</span>`;

    container.appendChild(toast);

    // Animate in
    setTimeout(() => {
        toast.style.transform = 'translateY(0)';
        toast.style.opacity = '1';
    }, 10);

    // Remove after 3s
    setTimeout(() => {
        toast.style.transform = 'translateY(100%)';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
