"""
100-Ticket Prediction Test
==========================
Runs 100 curated test cases directly through the trained model pipelines
(no HTTP, bypasses rate limiter). Checks category and priority predictions
against expected values. Saves a full report to reports/test_100_results.md.

Usage:
    python scripts/test_100_predictions.py
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text

CAT_MODEL_PATH = BASE_DIR / "model" / "category_pipeline.pkl"
PRI_MODEL_PATH = BASE_DIR / "model" / "priority_pipeline.pkl"
REPORTS_DIR = BASE_DIR / "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 100 Curated Test Cases
# Format: (description, expected_category, expected_priority)
# Covers all 3 categories x 4 priorities with real-world, edge-case & noisy text
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- TECHNICAL / CRITICAL (15) ---
    ("system is completely down, no users can access the platform", "Technical", "Critical"),
    ("the entire system is down, no one can login", "Technical", "Critical"),
    ("payment gateway is completely broken since 2 hours", "Technical", "Critical"),
    ("security breach detected on our account", "Technical", "Critical"),
    ("we are losing customer data right now", "Technical", "Critical"),
    ("CRITICAL: The dashboard is not loading and all users are blocked", "Technical", "Critical"),
    ("Getting a status code 500 in the main view. Production is impacted.", "Technical", "Critical"),
    ("HELP FAST: There is a glitches when I upload data. This is a major issue.", "Technical", "Critical"),
    ("NOW: The system is not responding correctly. I was trying to sync but it just stops working.", "Technical", "Critical"),
    ("BLOCKER: Getting a handshake failure in the main view after the weekend.", "Technical", "Critical"),
    ("IMMEDIATELY: The file is not appearing as expected in the gateway. This is blocking my work.", "Technical", "Critical"),
    ("URGENT: Getting a status code 500 in the profile panel. It's a critical blocker.", "Technical", "Critical"),
    ("The portal is not loading properly and it just returns an error.", "Technical", "Critical"),
    ("CRITICAL: There is a fails to complete when I sync. Very annoying.", "Technical", "Critical"),
    ("Getting a internal error in the profile panel after the weekend. Production is impacted.", "Technical", "Critical"),

    # --- TECHNICAL / HIGH (15) ---
    ("page shows server error and 404 not found", "Technical", "High"),
    ("login page returns 500 internal server error", "Technical", "High"),
    ("app crashes when trying to checkout", "Technical", "High"),
    ("database connection error on every request", "Technical", "High"),
    ("getting a server error", "Technical", "High"),
    ("cannot complete payment, getting server error", "Technical", "High"),
    ("Important: Getting a unexpected response in the navigation layer after the weekend. Urgent attention needed.", "Technical", "High"),
    ("Getting a timeout in the profile panel after the last update. Need this fixed soon.", "Technical", "High"),
    ("Getting a handshake failure in the side section after my last login. Need this fixed soon.", "Technical", "High"),
    ("The interface is showing inconsistencies. I was trying to access the feature but it just fails to complete.", "Technical", "High"),
    ("Urgent: The portal is not responding correctly. I was trying to proceed but it just glitches.", "Technical", "High"),
    ("Important: Getting a internal error in the profile panel after the weekend. This is blocking my work.", "Technical", "High"),
    ("The module is acting strange. I was trying to submit but it just fails to complete.", "Technical", "High"),
    ("Getting a unexpected response in the navigation layer after the last update. Production is impacted.", "Technical", "High"),
    ("The file is not appearing as expected in the module. Can you check the mobile device logs? Urgent attention needed.", "Technical", "High"),

    # --- TECHNICAL / MEDIUM (10) ---
    ("UI is not rendering correctly on mobile", "Technical", "Medium"),
    ("the dashboard loads slowly sometimes", "Technical", "Medium"),
    ("export button is not working sometimes", "Technical", "Medium"),
    ("feature is slow but still works", "Technical", "Medium"),
    ("Query: Getting a handshake failure in the main view after the last update. This is blocking my work.", "Technical", "Medium"),
    ("Follow up: There is a returns an error when I sync. This is very annoying.", "Technical", "Medium"),
    ("Getting a generic error in the main view after my last login. Need this fixed soon.", "Technical", "Medium"),
    ("The dashboard is behaving unexpectedly. I was trying to submit but it just glitches.", "Technical", "Medium"),
    ("The entry is not appearing as expected in the system. Can you check the web browser logs? Urgent attention needed.", "Technical", "Medium"),
    ("Getting a status code 500 in the settings area. This happened after this morning. It's a critical blocker.", "Technical", "Medium"),

    # --- TECHNICAL / LOW (5) ---
    ("Maybe later: The system is not loading properly. I was trying to save changes but it just returns an error.", "Technical", "Low"),
    ("Minor: The file is not appearing as expected in the module. Can you check the mobile device logs? Urgent attention needed.", "Technical", "Low"),
    ("FYI: There is a fails to complete when I upload data. This is very annoying.", "Technical", "Low"),
    ("No hurry: Getting a generic error in the side section after the last update. It's a critical blocker.", "Technical", "Low"),
    ("Suggestion: There is a fails to complete when I access the feature. This is very annoying.", "Technical", "Low"),

    # --- BILLING / CRITICAL (10) ---
    ("NOW: My card was denied last night. I also have a question about contact email.", "Billing", "Critical"),
    ("BLOCKER: My transfer was stuck last night. I also have a question about login information.", "Billing", "Critical"),
    ("IMMEDIATELY: The invoice from my last login seems incorrect. I checked the profile and it looks okay.", "Billing", "Critical"),
    ("HELP FAST: The summary from the last update seems not what I agreed to.", "Billing", "Critical"),
    ("NOW: The plan process did not behave as expected and not what I agreed to.", "Billing", "Critical"),
    ("IMMEDIATELY: I was expecting a email notification but I received a notification about denied. This is blocking my work.", "Billing", "Critical"),
    ("CRITICAL: Why is there a discrepancy in the invoice? It says INV-202 but the amount is different.", "Billing", "Critical"),
    ("URGENT: Why is there a discrepancy in the summary? It says TXN-101 but the amount is different.", "Billing", "Critical"),
    ("IMMEDIATELY: Why is there a discrepancy in the latest charge? It says order #55 but the amount is different.", "Billing", "Critical"),
    ("My account was stuck last night. I also have a question about security settings.", "Billing", "Critical"),

    # --- BILLING / HIGH (8) ---
    ("Important: Why is there a discrepancy in the statement? It says order #55 but the amount is different.", "Billing", "High"),
    ("My card was not going through last night. I also have a question about login information.", "Billing", "High"),
    ("By Monday: Why is there a discrepancy in the latest charge? It says ref 99 but the amount is different. Production is impacted.", "Billing", "High"),
    ("Important: My transfer was not going through last night. I also have a question about profile data.", "Billing", "High"),
    ("ASAP: Why is there a discrepancy in the latest charge? It says TXN-101 but the amount is different.", "Billing", "High"),
    ("The add-on process did not behave as expected and unusually high. Sorry for the trouble.", "Billing", "High"),
    ("By Monday: I was expecting a receipt but I received a notification about flagged. It's a critical blocker.", "Billing", "High"),
    ("Why is there a discrepancy in the summary? It says INV-202 but the amount is different. It's a critical blocker.", "Billing", "High"),

    # --- BILLING / MEDIUM (7) ---
    ("Why is there a discrepancy in the latest charge? It says TXN-101 but the amount is different. Production is impacted.", "Billing", "Medium"),
    ("My card was not going through last night. I also have a question about security settings.", "Billing", "Medium"),
    ("Why is there a discrepancy in the statement? It says order #55 but the amount is different. This is blocking my work.", "Billing", "Medium"),
    ("Query: My card was denied last night. I also have a question about profile data.", "Billing", "Medium"),
    ("I was expecting a email notification but I received a notification about not going through. Urgent attention needed.", "Billing", "Medium"),
    ("The invoice from this morning seems showing twice. I checked the card info and it looks okay.", "Billing", "Medium"),
    ("Routine: Why is there a discrepancy in the statement? It says order #55 but the amount is different.", "Billing", "Medium"),

    # --- BILLING / LOW (5) ---
    ("No hurry: The plan process did not behave as expected and showing twice.", "Billing", "Low"),
    ("FYI: My PayPal was flagged last night. I also have a question about profile data.", "Billing", "Low"),
    ("The subscription process did not behave as expected and not what I agreed to.", "Billing", "Low"),
    ("Minor: Why is there a discrepancy in the invoice? It says order #55 but the amount is different.", "Billing", "Low"),
    ("FYI: The summary from this morning seems unusually high. I checked the card info and it looks okay.", "Billing", "Low"),

    # --- ACCOUNT / CRITICAL (10) ---
    ("IMMEDIATELY: How do I reset the SMS system? I am not seeing any updates. This is blocking my work.", "Account", "Critical"),
    ("HELP FAST: How do I modify the message system? I am not seeing any updates. This is blocking my work.", "Account", "Critical"),
    ("BLOCKER: The verification settings are showing inconsistencies. I need to review my login information.", "Account", "Critical"),
    ("URGENT: My login information is under review. Also, the profile panel feels behaving unexpectedly.", "Account", "Critical"),
    ("IMMEDIATELY: My security settings is showing as inactive. Also, the main view feels not loading properly.", "Account", "Critical"),
    ("The sign in level is insufficient for my contact email. It keeps stops working.", "Account", "Critical"),
    ("BLOCKER: I am having trouble with password credentials when I try to change settings.", "Account", "Critical"),
    ("I am having trouble with password credentials when I try to sign in.", "Account", "Critical"),
    ("CRITICAL: The SAML settings are not responding correctly. I need to review my login information.", "Account", "Critical"),
    ("URGENT: My contact email is not accessible. Also, the settings area feels showing inconsistencies.", "Account", "Critical"),

    # --- ACCOUNT / HIGH (8) ---
    ("ASAP: The update profile level is insufficient for my login information. It keeps glitches.", "Account", "High"),
    ("Important: The verification settings are showing inconsistencies. I need to set up my contact email.", "Account", "High"),
    ("By Monday: The verification settings are behaving unexpectedly. I need to set up my security settings.", "Account", "High"),
    ("ASAP: I am having trouble with keys when I try to sign in.", "Account", "High"),
    ("Urgent: I am having trouble with login when I try to sign in.", "Account", "High"),
    ("Important: How do I review the SMS system? I am not seeing any updates. This is blocking my work.", "Account", "High"),
    ("The MFA settings are acting strange. I need to set up my login information. Production is impacted.", "Account", "High"),
    ("ASAP: The sign in level is insufficient for my profile data. It keeps glitches.", "Account", "High"),

    # --- ACCOUNT / MEDIUM (4) ---
    ("Routine: I am having trouble with login when I try to update profile.", "Account", "Medium"),
    ("The auth settings are showing inconsistencies. I need to review my login information.", "Account", "Medium"),
    ("I am having trouble with keys when I try to change settings.", "Account", "Medium"),
    ("How do I set up the message system? I am not seeing any updates. Need this fixed soon.", "Account", "Medium"),

    # --- ACCOUNT / LOW (3) ---
    ("No hurry: The auth settings are acting strange. I need to review my contact email. Production is impacted.", "Account", "Low"),
    ("Maybe later: The sign in level is insufficient for my profile data. It keeps stops working.", "Account", "Low"),
    ("Minor: The MFA settings are degraded. I need to set up my contact email. Production is impacted.", "Account", "Low"),
]

assert len(TEST_CASES) == 100, f"Expected 100 test cases, got {len(TEST_CASES)}"


def load_models():
    with open(CAT_MODEL_PATH, "rb") as f:
        cat_pipeline = pickle.load(f)
    with open(PRI_MODEL_PATH, "rb") as f:
        pri_pipeline = pickle.load(f)
    return cat_pipeline, pri_pipeline


def run_tests(cat_pipeline, pri_pipeline):
    results = []
    for desc, exp_cat, exp_pri in TEST_CASES:
        clean = clean_text(desc)
        pred_cat = cat_pipeline.predict([clean])[0]
        cat_conf = float(np.max(cat_pipeline.predict_proba([clean])))
        pred_pri = pri_pipeline.predict([clean])[0]
        pri_conf = float(np.max(pri_pipeline.predict_proba([clean])))

        cat_pass = pred_cat == exp_cat
        pri_pass = pred_pri == exp_pri
        overall = "PASS" if (cat_pass and pri_pass) else "FAIL"

        results.append({
            "description": desc[:70],
            "expected_category": exp_cat,
            "predicted_category": pred_cat,
            "cat_confidence": round(cat_conf, 3),
            "cat_ok": cat_pass,
            "expected_priority": exp_pri,
            "predicted_priority": pred_pri,
            "pri_confidence": round(pri_conf, 3),
            "pri_ok": pri_pass,
            "status": overall,
        })
    return results


def print_results(results):
    passes = [r for r in results if r["status"] == "PASS"]
    fails = [r for r in results if r["status"] == "FAIL"]

    print("\n" + "=" * 110)
    print(f"  100-TICKET MODEL TEST RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

    header = f"{'#':<4} {'Status':<6} {'Desc':<50} {'ExpCat':<10} {'GotCat':<10} {'CatConf':<8} {'ExpPri':<9} {'GotPri':<9} {'PriConf'}"
    print(header)
    print("-" * 110)

    for i, r in enumerate(results, 1):
        status_str = "✅" if r["status"] == "PASS" else "❌"
        print(
            f"{i:<4} {status_str:<6} {r['description']:<50} "
            f"{r['expected_category']:<10} {r['predicted_category']:<10} {r['cat_confidence']:<8} "
            f"{r['expected_priority']:<9} {r['predicted_priority']:<9} {r['pri_confidence']}"
        )

    print("=" * 110)
    print(f"\n  TOTAL: {len(results)} | ✅ PASSED: {len(passes)} | ❌ FAILED: {len(fails)}")

    if fails:
        print(f"\n  ⚠️  FAILURES ({len(fails)}):")
        for r in fails:
            cat_status = "✅" if r["cat_ok"] else f"❌ expected={r['expected_category']} got={r['predicted_category']}"
            pri_status = "✅" if r["pri_ok"] else f"❌ expected={r['expected_priority']} got={r['predicted_priority']}"
            print(f"    - [{r['description'][:60]}]")
            print(f"        Category: {cat_status}")
            print(f"        Priority: {pri_status}")
    else:
        print("\n  🎉 ALL 100 TESTS PASSED!")

    return passes, fails


def save_report(results, passes, fails):
    report_path = REPORTS_DIR / "test_100_results.md"
    with open(report_path, "w") as f:
        f.write(f"# 100-Ticket Model Test Report\n\n")
        f.write(f"**Run at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Result:** {'✅ ALL PASSED' if not fails else f'❌ {len(fails)} FAILED'}\n\n")
        f.write(f"| # | Status | Description | Exp Cat | Got Cat | Cat Conf | Exp Pri | Got Pri | Pri Conf |\n")
        f.write(f"|---|--------|-------------|---------|---------|----------|---------|---------|----------|\n")
        for i, r in enumerate(results, 1):
            status = "✅" if r["status"] == "PASS" else "❌"
            f.write(
                f"| {i} | {status} | {r['description'][:55]} | "
                f"{r['expected_category']} | {r['predicted_category']} | {r['cat_confidence']} | "
                f"{r['expected_priority']} | {r['predicted_priority']} | {r['pri_confidence']} |\n"
            )
        f.write(f"\n**Total:** {len(results)} | **Passed:** {len(passes)} | **Failed:** {len(fails)}\n")
    print(f"\n  Report saved → {report_path}")
    return report_path


if __name__ == "__main__":
    print("Loading models...")
    cat_pipeline, pri_pipeline = load_models()
    print(f"  Category pipeline: {cat_pipeline.named_steps['clf'].__class__.__name__}")
    print(f"  Priority pipeline: {pri_pipeline.named_steps['clf'].__class__.__name__}")

    print("\nRunning 100 prediction tests...")
    results = run_tests(cat_pipeline, pri_pipeline)
    passes, fails = print_results(results)
    save_report(results, passes, fails)

    sys.exit(0 if not fails else 1)
