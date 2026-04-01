"""
Send 1000 test support emails via Gmail SMTP
=============================================
From: vishal1711004@gmail.com
To:   vishalsi1711@gmail.com

Uses Gmail SMTP with App Password (OAuth not needed).
Run: python scripts/send_test_emails.py

SETUP: You need a Gmail App Password for vishal1711004@gmail.com
  1. Go to https://myaccount.google.com/security
  2. Enable 2-Step Verification if not already done
  3. Search "App passwords" → Generate one for "Mail"
  4. Paste it when prompted (or set env var GMAIL_APP_PASSWORD)
"""

import smtplib
import os
import sys
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

SENDER = "vishal1711004@gmail.com"
RECEIVER = "vishalsi1711@gmail.com"

# 100 unique ticket scenarios  
TICKET_TEMPLATES = [
    # Technical / Critical
    ("CRITICAL: System Down - All Users Blocked",
     "The entire system is down. No users can log in. We are losing data every minute. Production completely impacted. Please fix immediately."),
    ("BLOCKER: Payment Gateway Broken Since 2 Hours",
     "Payment gateway has been broken for 2 hours. Customers cannot complete purchases. Revenue loss continuing. Urgent fix needed."),
    ("URGENT: Security Breach Detected",
     "We detected a potential security breach on our account. Unauthorized access suspected. Please investigate immediately."),
    ("IMMEDIATELY: Data Loss In Progress",
     "We are actively losing customer data right now. This is a critical emergency. Immediate response required."),
    ("NOW: Login Page Returns 500 Error",
     "Login page returns 500 internal server error for all users. Production is completely impacted."),
    ("HELP FAST: App Crashes On Checkout",
     "App crashes every time a user tries to checkout. Revenue loss is continuing with every minute of downtime."),
    ("CRITICAL: Database Connection Error",
     "Database connection error on every request. The site is essentially down for all users."),
    ("BLOCKER: SSL Certificate Error",
     "SSL certificate error is preventing all secure connections. Users see browser security warnings and cannot proceed."),
    ("IMMEDIATELY: API Not Responding",
     "Our API integration is timing out on every request. Cannot use your service at all. Production completely down."),
    ("URGENT: Account Suspended Unexpectedly",
     "Our account appears suspended. We cannot access any features. We need emergency assistance."),

    # Technical / High
    ("Important: UI Rendering Issues on Mobile",
     "UI is not rendering correctly on mobile devices. Large portion of users affected. Need this fixed soon."),
    ("Dashboard Loading Slowly After Update",
     "Dashboard is loading very slowly since the latest update. Performance degraded significantly. Need this fixed soon."),
    ("Export Button Not Working For Team",
     "Export button is not working for any user. Data exports are completely blocked. Need resolution soon."),
    ("Server Error and 404 on Multiple Pages",
     "Multiple pages show server error and 404 not found. Many key sections of the app are broken."),
    ("Urgent: App Returning Unexpected Errors",
     "App is throwing unexpected errors intermittently. Users are affected on and off throughout the day."),
    ("Getting Timeout in Profile Panel",
     "Profile panel times out after 10 seconds. Cannot manage account settings. Need this fixed soon."),
    ("Handshake Failure in Main Dashboard View",
     "Getting handshake failure errors when accessing the main dashboard. Happens intermittently since last update."),
    ("Module Not Responding Correctly",
     "The main module is not responding as expected. Trying to submit data but it just fails. Need help soon."),
    ("File Not Appearing in System",
     "Files are not appearing in the module after upload. Need to check the upload pipeline. Urgent attention needed."),
    ("Interface Showing Inconsistencies",
     "Interface is showing inconsistencies across different sections. Data shown doesn't match expected values."),

    # Technical / Medium
    ("Dashboard Behaving Unexpectedly",
     "Dashboard is behaving unexpectedly when trying to submit reports. Glitches happen intermittently."),
    ("Settings Area Returning 500 Sometimes",
     "Settings area returns 500 error occasionally. Not consistent but happening multiple times today."),
    ("Generic Error in Main View After Login",
     "Getting a generic error in the main view after logging in. Happens every few hours."),
    ("Navigation Layer Showing Unexpected Response",
     "Unexpected response in the navigation layer after the latest update. Not urgent but concerning."),
    ("Entry Not Appearing in System View",
     "An entry is not appearing as expected in the system view. Could you check the configuration?"),
    ("Status Code 500 in Settings Area",
     "Getting status code 500 in the settings area since this morning. Not always but frequently."),
    ("Query: Handshake Failure in Main View",
     "Getting a handshake failure in the main view after the last update. Not critical but needs attention."),
    ("Follow Up: Returns Error When Syncing",
     "There is an error when I try to sync data. It's very annoying but I can work around it for now."),
    ("The System Not Loading Properly",
     "The system is not loading properly. Could be a caching issue. Please investigate when you get a chance."),
    ("Feature Slow But Still Working",
     "A key feature is slow but still functional. Takes about 30 seconds to load which is frustrating."),

    # Technical / Low
    ("Maybe Later: System Not Loading Save Changes",
     "The system is not saving changes properly. I was trying to save updates but it just returns an error. No hurry."),
    ("Minor: File Not Appearing in Module",
     "A minor issue with files not appearing in the module. Can you check when you have time? No urgency."),
    ("FYI: Fails To Complete Upload",
     "Just a heads up - the upload feature sometimes fails to complete. Not blocking my work. Low priority."),
    ("No Hurry: Generic Error in Side Section",
     "Getting a generic error in the side section occasionally. No hurry on this one. Just logging it."),
    ("Suggestion: Feature Access Issue",
     "There is a minor issue when I try to access a certain feature. Not urgent at all. Just a FYI."),

    # Billing / Critical
    ("NOW: Card Denied - Cannot Make Payments",
     "My card was denied last night and I cannot make any payments at all. This is completely blocking me. Critical issue."),
    ("BLOCKER: Transfer Stuck In Processing",
     "My transfer has been stuck in processing for over 24 hours. This is a critical billing issue."),
    ("IMMEDIATELY: Invoice Discrepancy INV-202",
     "The invoice INV-202 shows the wrong amount. This is a critical billing error that must be corrected immediately."),
    ("URGENT: Statement Amount Is Completely Wrong",
     "My account statement shows an amount that is completely wrong. This is a critical discrepancy."),
    ("CRITICAL: Unauthorized Charge Detected",
     "I see an unauthorized charge on my account. This is a critical billing issue requiring immediate investigation."),
    ("NOW: PayPal Payment Failed Blocking Work",
     "My PayPal payment failed last night and this is completely blocking my work. Critical issue."),
    ("IMMEDIATELY: Subscription Charge Wrong Amount",
     "The subscription charge this month is 3x higher than what was agreed. Critical billing error."),
    ("BLOCKER: Account Balance Shows Incorrect",
     "Account balance shows incorrect numbers. All billing is compromised. Emergency assistance needed."),
    ("URGENT: Duplicate Charge This Month",
     "We were billed twice this month. Duplicate charge on our account. This needs immediate correction."),
    ("CRITICAL: Refund Not Processing",
     "A refund that should have been processed hasn't been. This is making our accounts inaccurate."),

    # Billing / High
    ("Important: Discrepancy in Statement Order-55",
     "There is a discrepancy in my statement for order #55. The charged amount does not match what was agreed. Need this fixed."),
    ("Card Not Going Through Since Last Night",
     "My card was not going through last night. I have a question about my login and payment method. Need help soon."),
    ("ASAP: Latest Charge TXN-101 Amount Different",
     "The latest charge TXN-101 shows a different amount than expected. Need this resolved ASAP."),
    ("The Add-On Process Unusually High",
     "The add-on charge is unusually high this month. This is a billing error that needs correction soon."),
    ("By Monday: Discrepancy Latest Charge REF-99",
     "There is a discrepancy in the latest charge reference REF-99. The amounts don't match. Need by Monday."),
    ("Transfer Not Going Through Question Profile",
     "My transfer was not going through last night. I also have a question about my profile data."),
    ("ASAP: Billing Portal Not Loading",
     "Cannot access the billing portal to update my payment method. Need this fixed urgently."),
    ("Important: Why Is There A Discrepancy INV-202",
     "Why is there a discrepancy in my invoice INV-202? It says the amount is different from what I expected."),
    ("By Monday: Receipt Shows Wrong Details",
     "My receipt shows the wrong order details. The total doesn't match what I was shown at checkout."),
    ("Expected Email Received Denied Notification",
     "I was expecting a confirmation email but received a notification about my payment being denied."),

    # Billing / Medium
    ("Query: Card Denied Last Night About Profile",
     "My card was denied last night. I also have a question about my profile data. Can you look into this?"),
    ("Why Discrepancy In Summary TXN-101 Amount",
     "Why is there a discrepancy in the summary? It says TXN-101 but the amount is different. Routine query."),
    ("Invoice From This Morning Showing Twice",
     "The invoice from this morning is showing twice in my account. Could you verify this is not a double charge?"),
    ("Routine: Why Discrepancy Statement Order 55",
     "A routine inquiry about a discrepancy in my statement for order #55. Amount seems slightly off."),
    ("Email Notification About Not Going Through",
     "I received a notification about a payment not going through. Would like this checked when possible."),
    ("Card Not Going Through Question Security",
     "My card was not going through yesterday. I also had a question about my security settings."),
    ("Latest Charge TXN-101 Amount Different",
     "The latest charge TXN-101 shows a different amount than what I expected. Please investigate."),
    ("Discrepancy in Statement for Order-55",
     "There is a discrepancy in my statement for order #55. The amount charged seems slightly off."),

    # Billing / Low
    ("No Hurry: Plan Process Showed Twice",
     "The plan process showed twice in my account summary. No rush on this but please investigate."),
    ("FYI: PayPal Flagged Last Night Profile Data",
     "Just a note - my PayPal was flagged last night. Also have a question about profile data. Low priority."),
    ("Subscription Process Not What Agreed",
     "The subscription renewal shows terms slightly different from what I agreed to. Not urgent."),
    ("Minor: Invoice Order-55 Amount Discrepancy",
     "Minor discrepancy in invoice for order #55. The amount is slightly different. No urgency."),
    ("FYI: Summary From Morning Unusually High",
     "The account summary from this morning looked unusually high. Could just be a display issue. FYI only."),

    # Account / Critical
    ("IMMEDIATELY: Reset SMS System Not Seeing Updates",
     "How do I reset the SMS system? I am not seeing any updates. This is completely blocking my work."),
    ("HELP FAST: Modify Message System Blocking Work",
     "I need to modify the message system but cannot. Not seeing any updates. This is blocking my work."),
    ("BLOCKER: Verification Settings Inconsistencies",
     "My verification settings are showing inconsistencies. I cannot access my login information. Critical blocker."),
    ("URGENT: Login Information Under Review Blocked",
     "My login information is under review and I cannot access my account. Profile panel is behaving unexpectedly."),
    ("Sign In Level Insufficient Contact Email",
     "The sign-in level is insufficient for my contact email. It keeps failing. I cannot access my account."),
    ("CRITICAL: SAML Settings Not Responding",
     "The SAML settings are not responding correctly. I need to review my login information. Critical issue."),
    ("URGENT: Contact Email Not Accessible",
     "My contact email is completely inaccessible. The settings area is also showing inconsistencies. Blocking all work."),
    ("BLOCKER: Password Credentials Change Settings",
     "I am having trouble with my password credentials when I try to change settings. This is blocking me."),
    ("I Am Having Trouble With Password Sign In",
     "I am having trouble with my password credentials when I try to sign in. Cannot access my account at all."),
    ("IMMEDIATELY: Security Settings Showing Inactive",
     "My security settings are showing as inactive. The main view is also not loading properly. Critical."),

    # Account / High
    ("ASAP: Update Profile Level Insufficient Login",
     "The profile update level is insufficient for my login information. It keeps glitching. Need fix ASAP."),
    ("Important: Verification Settings Inconsistent",
     "My verification settings are showing inconsistencies. Need to set up my contact email. Urgent attention."),
    ("By Monday: Verification Settings Unexpected",
     "Verification settings are behaving unexpectedly. Need to set up my security settings by Monday."),
    ("ASAP: Keys Not Working Sign In",
     "I am having trouble with my access keys when I try to sign in. Need this fixed ASAP."),
    ("Urgent: Login Trouble When Trying To Sign In",
     "I have persistent trouble with login when I try to sign in. Need this resolved urgently."),
    ("Important: Review SMS System Not Seeing Updates",
     "I need to review the SMS system but I am not seeing any updates. This is blocking my work."),
    ("MFA Settings Acting Strange Login Information",
     "My MFA settings are acting strange. I need to set up my login information. Production is impacted."),
    ("ASAP: Sign In Level Insufficient Profile Data",
     "The sign-in level is insufficient for my profile data. It keeps glitching. I need this fixed ASAP."),

    # Account / Medium
    ("Routine: Trouble With Login Update Profile",
     "Routine issue - having trouble with login when I try to update my profile. Not urgent."),
    ("Auth Settings Showing Inconsistencies Login",
     "My auth settings are showing inconsistencies. Need to review my login information when possible."),
    ("Trouble With Keys When Change Settings",
     "Having trouble with my access keys when I try to change settings. Can someone look into this?"),
    ("How Do I Set Up The Message System",
     "I need help setting up the message system. I am not seeing any updates. Need this fixed soon."),

    # Account / Low
    ("No Hurry: Auth Settings Acting Strange",
     "My auth settings are acting a bit strange. Need to review my contact email eventually. No rush."),
    ("Maybe Later: Sign In Level Insufficient Profile",
     "The sign-in level seems insufficient for my profile data. It occasionally stops working. Low priority."),
    ("Minor: MFA Settings Degraded Contact Email",
     "My MFA settings appear slightly degraded. I need to set up my contact email at some point. Minor issue."),
]

def create_mime_message(sender, receiver, subject, body):
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    return msg

def send_emails(password: str, count: int = 1000):
    print(f"\nConnecting to Gmail SMTP...")
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30)
        server.login(SENDER, password)
        print(f"✅ Connected. Sending {count} emails from {SENDER} → {RECEIVER}\n")
    except Exception as e:
        print(f"❌ SMTP connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure 2-Step Verification is ON in Google Account")
        print("2. Generate an App Password: myaccount.google.com/apppasswords")
        print("3. Use the 16-char app password (not your regular Gmail password)")
        return

    sent = 0
    failed = 0
    start = time.time()

    for i in range(count):
        template = TICKET_TEMPLATES[i % len(TICKET_TEMPLATES)]
        batch = i // len(TICKET_TEMPLATES) + 1
        subject = template[0] + (f" (Batch {batch})" if batch > 1 else "")
        body = template[1] + f"\n\nTicket ID: TICKET-{i+1:04d}\nTimestamp: {datetime.now().isoformat()}"

        try:
            msg = create_mime_message(SENDER, RECEIVER, subject, body)
            server.sendmail(SENDER, RECEIVER, msg.as_string())
            sent += 1
        except smtplib.SMTPException as e:
            failed += 1
            print(f"  ❌ Email {i+1} failed: {e}")
            # Re-connect if connection dropped
            try:
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30)
                server.login(SENDER, password)
            except:
                pass

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  Progress: {i+1}/{count} | ✅ {sent} sent | ❌ {failed} failed | {elapsed:.0f}s elapsed")
        
        # Small delay to avoid Gmail rate limiting (500 emails/day limit)
        if i < count - 1:
            time.sleep(0.05)

    server.quit()
    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"COMPLETE: {sent}/{count} sent, {failed} failed in {elapsed:.0f}s")
    print(f"Check {RECEIVER} inbox for the emails.")

if __name__ == "__main__":
    print("="*50)
    print("TicketIQ - Gmail Test Email Sender")
    print("="*50)
    print(f"FROM: {SENDER}")
    print(f"TO:   {RECEIVER}")
    print()

    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not password:
        print("Enter Gmail App Password for vishal1711004@gmail.com")
        print("(Get it from: myaccount.google.com/apppasswords)")
        import getpass
        password = getpass.getpass("App Password: ")

    try:
        count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    except ValueError:
        count = 1000

    send_emails(password, count)
