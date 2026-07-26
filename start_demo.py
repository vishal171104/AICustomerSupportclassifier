#!/usr/bin/env python3
"""
start_demo.py  —  TicketIQ Offline Demo Launcher
=================================================
Usage:  python start_demo.py
        (or double-click in Finder)

What it does:
  1. Clears previous demo tickets so the board starts fresh
  2. Seeds the DB with the demo email (demo@ticketiq.ai)
  3. Starts the FastAPI server on port 8000
  4. Opens the dashboard in your default browser automatically
"""
import subprocess, sys, time, sqlite3, os, webbrowser, threading
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
DB_PATH    = BASE_DIR / "data" / "predictions.db"
DEMO_EMAIL = "demo@ticketiq.ai"
PORT       = 8000

def get_python():
    candidates = [
        BASE_DIR / "venv" / "bin" / "python",
        BASE_DIR / "venv_prod" / "bin" / "python",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable

# ── Reset previous demo data ─────────────────────────────────────────
def reset_demo():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM tickets WHERE source = 'demo'")
        n = c.rowcount
        conn.commit(); conn.close()
        print(f"  ✓ Cleared {n} stale demo tickets")
    except Exception as e:
        print(f"  ⚠ Could not clear demo tickets: {e}")

# ── Seed configured email ────────────────────────────────────────────
def seed_email():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT watched_email FROM config LIMIT 1")
        row = c.fetchone()
        if not row or not row[0]:
            c.execute("DELETE FROM config")
            c.execute(
                "INSERT INTO config (watched_email, registered_at) VALUES (?, ?)",
                (DEMO_EMAIL, time.strftime('%Y-%m-%dT%H:%M:%SZ'))
            )
            conn.commit()
            print(f"  ✓ Demo email set: {DEMO_EMAIL}")
        else:
            print(f"  ✓ Email already set: {row[0]}")
        conn.close()
    except Exception as e:
        print(f"  ⚠ Could not seed email: {e}")

# ── Open browser after short delay ───────────────────────────────────
def open_browser():
    time.sleep(3)
    url = f"http://localhost:{PORT}/dashboard"
    print(f"\n  🌐  Opening browser → {url}")
    webbrowser.open(url)

# ── Main ─────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   TicketIQ — AI Customer Support Classifier          ║")
    print("║   OFFLINE LIVE DEMO LAUNCHER                         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("📦  Preparing environment...")
    reset_demo()
    seed_email()

    python = get_python()
    print(f"\n🚀  Starting server  →  http://localhost:{PORT}")
    print(f"    Dashboard        →  http://localhost:{PORT}/dashboard")
    print("\n    Then click the green ● Live Demo button!")
    print("\n    Press Ctrl+C to stop.\n")
    print("──────────────────────────────────────────────────────")

    threading.Thread(target=open_browser, daemon=True).start()

    os.environ["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [python, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(PORT), "--reload"],
        cwd=str(BASE_DIR)
    )

if __name__ == "__main__":
    main()
