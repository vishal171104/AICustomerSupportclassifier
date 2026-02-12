import pandas as pd
import random

new_tickets = [
    # Billing
    ("Double charge on my invoice INV-9902.", "Billing", "High"),
    ("Requesting a refund for the service I cancelled last month.", "Billing", "Medium"),
    ("My payment was declined but my bank says it went through.", "Billing", "High"),
    ("Need a copy of my yearly tax statement for the previous fiscal year.", "Billing", "Low"),
    ("The subscription price increased without prior notification.", "Billing", "Medium"),
    ("Cannot update my credit card details on the billing portal.", "Billing", "High"),
    ("Where can I find my transaction history for the last 3 years?", "Billing", "Low"),
    ("Applied a promo code but the discount is not reflected in the total.", "Billing", "Medium"),
    ("Unrecognized charge from your company on my bank statement.", "Billing", "Critical"),
    ("The autopay feature is not working as expected, had to pay manually.", "Billing", "Medium"),
    ("Inconsistent pricing between the mobile app and the website.", "Billing", "Low"),
    ("My account is locked due to an outstanding balance I already paid.", "Billing", "Critical"),
    ("How do I change my currency settings to USD?", "Billing", "Low"),
    ("Received an invoice for a service that was supposed to be free trial.", "Billing", "Medium"),
    ("The VAT amount on my receipt seems incorrectly calculated.", "Billing", "Medium"),
    ("I was overcharged for the premium tier features.", "Billing", "High"),
    
    # Account
    ("I cannot reset my password, the 'Forgot Password' link is broken.", "Account", "High"),
    ("My account was breached and my email was changed.", "Account", "Critical"),
    ("How do I delete my account permanently and remove all data?", "Account", "Medium"),
    ("The two-factor authentication code is not arriving on my phone.", "Account", "Critical"),
    ("Need to merge two accounts under the same business email.", "Account", "Medium"),
    ("My profile picture upload keeps failing with a size error.", "Account", "Low"),
    ("Unauthorized login attempt detected from an unknown location.", "Account", "High"),
    ("Update my legal name and address in the business profile.", "Account", "Low"),
    ("The session keeps logging me out every 5 minutes.", "Account", "Medium"),
    ("I am unable to change my primary email address.", "Account", "High"),
    ("Verification email is not being received in the inbox or spam.", "Account", "High"),
    ("Account suspended for no apparent reason, please investigate.", "Account", "Critical"),
    ("Can I change my username or is it permanent once created?", "Account", "Low"),
    ("My organization members cannot access the shared workspace.", "Account", "High"),
    ("Permission denied error when I try to access the settings panel.", "Account", "Medium"),
    
    # Technical
    ("The mobile app crashes every time I open the analytics tab.", "Technical", "High"),
    ("API documentation link is returning a 404 page not found.", "Technical", "Low"),
    ("Database connection refused when trying to sync large datasets.", "Technical", "Critical"),
    ("The website is loading very slowly in the European region.", "Technical", "Medium"),
    ("CSS elements are overlapping on the mobile view of the dashboard.", "Technical", "Low"),
    ("Feature X is not working after the latest system update.", "Technical", "High"),
    ("The export to PDF function is generating blank documents.", "Technical", "Medium"),
    ("Search results are not refreshing when I apply new filters.", "Technical", "Low"),
    ("Integration with Slack is failing with an invalid token error.", "Technical", "High"),
    ("Server side error 500 when adding new items to the inventory.", "Technical", "Critical"),
    ("The webhook is not triggering when a new order is placed.", "Technical", "High"),
    ("Image compression is making the uploaded photos look blurry.", "Technical", "Low"),
    ("The SSL certificate for our custom domain has expired.", "Technical", "Critical"),
    ("Dark mode is causing readability issues on the reports page.", "Technical", "Low"),
    ("Auto-save is not working in the text editor component.", "Technical", "Medium"),
    ("System latency is too high during the morning peak hours.", "Technical", "Medium"),
    ("Broken links in the help center articles need fixing.", "Technical", "Low"),
    ("The dashboard doesn't load on Safari browser.", "Technical", "High"),
    ("Real-time notifications are delayed by several minutes.", "Technical", "Medium")
]

# Shuffle to be safe
random.shuffle(new_tickets)

df_new = pd.DataFrame(new_tickets, columns=["description", "category", "priority"])
# ticket_id will be added by appending or re-indexing
df_orig = pd.read_csv("data/tickets.csv")

# Ensure unique ticket IDs
max_id = df_orig["ticket_id"].max()
df_new["ticket_id"] = range(max_id + 1, max_id + 1 + len(df_new))

df_final = pd.concat([df_orig, df_new], ignore_index=True)
df_final.to_csv("data/tickets.csv", index=False)

print(f"Added {len(df_new)} new tickets to data/tickets.csv.")
