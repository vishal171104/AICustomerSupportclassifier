import pandas as pd
import random
import re
from pathlib import Path

# Categories and Priorities
categories = ["Technical", "Billing", "Account"]
priorities = ["Critical", "High", "Medium", "Low"]

# Ambiguous and "Non-Trivial" templates
# We remove direct words like "refund", "password", "crash" in some cases 
# and use descriptive language or mix the intents.
templates = {
    "Technical": [
        "The {component} is {issue}. I was trying to {action} but it just {fault}. {noise}",
        "Getting a {error_type} in the {app_part}. This happened after {time_period}. {urgency}",
        "I noticed the {app_part} is {issue} on {platform}. Also, {mixed_intent_billing}. {noise}",
        "The {resource} is not appearing as expected in the {component}. Can you check the {platform} logs? {urgency}",
        "There is a {fault} when I {action}. This is {description_adj}. Sent from my {platform}.",
    ],
    "Billing": [
        "The {bill_type} from {time_period} seems {charge_type}. I checked the {billing_info} and it looks okay. {noise}",
        "I was expecting a {document} but I received a notification about {payment_status}. {urgency}",
        "My {payment_method} was {payment_status} last night. I also have a question about {account_detail}. {noise}",
        "Why is there a discrepancy in the {bill_type}? It says {id} but the amount is different. {urgency}",
        "The {service} process did not behave as expected and {charge_type}. {noise}",
    ],
    "Account": [
        "I am having trouble with {credentials} when I try to {access_type}. {noise}",
        "The {security_feature} settings are {issue}. I need to {account_action} my {account_detail}. {urgency}",
        "My {account_detail} is {account_status}. Also, the {app_part} feels {issue}. {noise}",
        "How do I {account_action} the {notification} system? I am not seeing any updates. {urgency}",
        "The {access_type} level is insufficient for my {account_detail}. It keeps {fault}. {noise}",
    ]
}

fillers = {
    "component": ["gateway", "interface", "system", "portal", "dashboard", "module"],
    "issue": ["acting strange", "not responding correctly", "behaving unexpectedly", "showing inconsistencies", "degraded", "not loading properly"],
    "error_type": ["unexpected response", "internal error", "handshake failure", "timeout", "generic error", "status code 500"],
    "action": ["proceed", "save changes", "upload data", "submit", "sync", "access the feature"],
    "app_part": ["main view", "side section", "navigation layer", "settings area", "profile panel"],
    "fault": ["stops working", "hangs", "returns an error", "glitches", "fails to complete"],
    "resource": ["item", "data entry", "token", "file", "record", "entry"],
    "platform": ["mobile device", "web browser", "client app", "iOS", "Android", "desktop"],
    "time_period": ["the last update", "yesterday", "this morning", "the weekend", "my last login"],
    "urgency": ["This is blocking my work.", "Need this fixed soon.", "It's a critical blocker.", "Urgent attention needed.", "Production is impacted."],
    "noise": ["Hope you're having a good day.", "Sent from my iPhone.", "Thanks for the help.", "Best regards.", "Sorry for the trouble."],
    "charge_type": ["higher than expected", "incorrect", "showing twice", "not what I agreed to", "unusually high"],
    "service": ["plan", "feature", "subscription", "add-on"],
    "payment_status": ["not going through", "denied", "stuck", "flagged"],
    "bill_type": ["summary", "statement", "latest charge", "invoice"],
    "document": ["confirmation", "receipt", "email notification", "PDF file"],
    "billing_info": ["details", "address", "card info", "profile"],
    "id": ["TXN-101", "INV-202", "order #55", "ref 99"],
    "payment_method": ["card", "transfer", "account", "PayPal"],
    "account_action": ["check", "modify", "review", "set up", "reset"],
    "account_detail": ["login information", "contact email", "security settings", "profile data"],
    "account_status": ["not accessible", "showing as inactive", "locked", "under review"],
    "access_type": ["sign in", "view logs", "update profile", "change settings"],
    "security_feature": ["SAML", "MFA", "verification", "auth"],
    "description_adj": ["very annoying", "frustrating", "a major issue", "preventing progress"],
    "credentials": ["keys", "login", "password credentials", "authentication info"],
    "notification": ["alert", "email", "message", "SMS"],
    "mixed_intent_billing": ["also my last payment failed", "and check my refund", "but I have a billing question", "and the invoice is wrong"]
}

def add_typos(text):
    if random.random() > 0.3: # Increased typo frequency to 30%
        return text
    words = text.split()
    if not words: return text
    idx = random.randint(0, len(words) - 1)
    word = words[idx]
    if len(word) > 4:
        mode = random.choice(['swap', 'miss', 'double'])
        if mode == 'swap':
            i = random.randint(0, len(word)-2)
            word = word[:i] + word[i+1] + word[i] + word[i+2:]
        elif mode == 'miss':
            i = random.randint(0, len(word)-1)
            word = word[:i] + word[i+1:]
        elif mode == 'double':
            i = random.randint(0, len(word)-1)
            word = word[:i] + word[i] + word[i] + word[i+1:]
        words[idx] = word
    return " ".join(words)

def generate_ticket(cat, pri):
    template = random.choice(templates[cat])
    placeholders = re.findall(r'\{(.*?)\}', template)
    data = {p: random.choice(fillers.get(p, ["data"])) for p in placeholders}
    description = template.format(**data)
    
    # Priority signals - less obvious
    urgency_markers = {
        "Critical": ["NOW", "IMMEDIATELY", "CRITICAL", "URGENT", "BLOCKER", "HELP FAST"],
        "High": ["Important", "ASAP", "By Monday", "Urgent", "Needs checking"],
        "Medium": ["Question", "Query", "Follow up", "Routine", "Update"],
        "Low": ["No hurry", "FYI", "Suggestion", "Minor", "Maybe later"]
    }
    
    if random.random() > 0.4:
        description = f"{random.choice(urgency_markers[pri])}: {description}"
    
    # Add length variations
    if random.random() > 0.7:
        description += " " + " ".join(random.sample(list(fillers["noise"]), 2))
        
    description = add_typos(description)
    return description

def main():
    new_data = []
    # 500 tickets total for Phase 3
    per_combination = 42 
    for cat in categories:
        for pri in priorities:
            for _ in range(per_combination):
                desc = generate_ticket(cat, pri)
                new_data.append({"description": desc, "category": cat, "priority": pri})
    
    random.shuffle(new_data)
    df = pd.DataFrame(new_data)
    df.index += 1
    df.index.name = "ticket_id"
    df.reset_index(inplace=True)
    
    data_path = Path("/Users/vishalsi/CustomerSupportTicketAI/data/tickets.csv")
    df.to_csv(data_path, index=False)
    print(f"Generated {len(df)} non-trivial research-grade tickets.")

if __name__ == "__main__":
    main()
