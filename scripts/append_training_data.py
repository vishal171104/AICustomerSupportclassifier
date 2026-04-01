import pandas as pd
import random
from pathlib import Path

DATA_PATH = Path("/Users/vishalsi/CustomerSupportTicketAI/data/tickets.csv")

critical_templates = [
    "the entire system is down, no one can login",
    "system is completely down, no users can access the platform",
    "the entire system is completely down",
    "no users can access the platform",
    "we are losing customer data right now",
    "payment gateway is completely broken since 2 hours",
    "security breach detected on our account",
]

high_templates = [
    "page shows server error and 404 not found",
    "the next page is loading for a long time and shows error 404",
    "the next page is loading for a long time, shows server error and error 404 not found",
    "cannot complete payment, getting server error",
    "app crashes when trying to checkout",
    "login page returns 500 internal server error",
    "database connection error on every request",
    "getting a server error",
]

medium_templates = [
    "feature is slow but still works",
    "UI is not rendering correctly on mobile",
    "export button is not working sometimes",
    "the dashboard loads slowly sometimes",
    "the dashboard loads slowly",
]

low_templates = [
    "how do I change my profile picture",
    "I want to update my billing address",
    "can you add dark mode",
    "how do I reset my password",
    "I forgot my password, how do I reset it",
]

# Provide around 100 examples per class
noise_words = ["please help", "thanks", "hello", "hi", "can someone check this", "not sure why", "regards", "best"]

data = []

def generate_examples(templates, priority, count=120):
    for i in range(count):
        text = random.choice(templates)
        # Add a tiny bit of random noise to prevent identical duplicates
        if random.random() > 0.5:
            text += " " + random.choice(noise_words)
        if random.random() > 0.8:
            text = random.choice(noise_words) + " " + text
            
        data.append({
            "description": text,
            "category": "Technical",  # For priority, category doesn't matter too much, but Technical covers most cases
            "priority": priority
        })

generate_examples(critical_templates, "Critical", 120)
generate_examples(high_templates, "High", 120)
generate_examples(medium_templates, "Medium", 120)
generate_examples(low_templates, "Low", 120)

new_df = pd.DataFrame(data)

# Read existing and append
existing_df = pd.read_csv(DATA_PATH)
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Reindex
combined_df.index = range(1, len(combined_df) + 1)
if 'ticket_id' in combined_df.columns:
    combined_df = combined_df.drop(columns=['ticket_id'])
combined_df.index.name = "ticket_id"
combined_df.reset_index(inplace=True)

combined_df.to_csv(DATA_PATH, index=False)
print(f"Appended 480 new explicit examples to {DATA_PATH}. Total size: {len(combined_df)}")
