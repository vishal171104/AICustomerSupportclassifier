# 3-Minute Demo Script: Ticket Triage AI

## 0:00 - The Problem (The "Pain")
"Meet Sarah. She manages support for a high-growth retail app. Every day, 500+ tickets flood her inbox. Manually sorting them into 'Billing' vs 'Technical' and deciding what's 'Critical' takes her team 4 hours a day. That's 4 hours they aren't actually *solving* customer problems."

## 0:30 - The Solution (Live Prediction)
"Enter our Automated Ticket Triage AI. Watch what happens when I paste a complex, frustrated message: 'I can't access my recent invoices and the app keeps crashing on the login screen.'
[Visual: Prediction bar fills up]
In 150ms, the AI identifies this as primarily **Technical** (94% confidence) but flags the **Billing** sub-intent. Most importantly, it assigns **High Priority** because of the login failure."

## 1:00 - Model Performance
"We didn't just build a model; we built a research-grade pipeline. Our ensemble of SVM and Logistic Regression achieves 96% accuracy on semantic categorization. Even on the harder task of priority prediction, we outperform a majority-class baseline by 2x."

## 1:30 - Explainability & Statistical Rigor
"Why did the AI say 'High Priority'? Our feature importance layer highlights 'access', 'crashing', and 'login' as the primary drivers. We've validated these results using McNemar’s significance tests and 1000-iteration bootstrap confidence intervals."

## 2:15 - Production Ready
"The system is deployed via a hardened FastAPI backend. It features SQLite-based observability for drift monitoring, batch processing for bulk uploads, and automated input validation for security."

## 2:45 - The Impact
"By automating the first 5 minutes of every ticket's lifecycle, we return 500 hours a month back to the support team. This is AI that doesn't just predict—it professionalizes."

## 3:00 - Outro
"Check out the full technical report and GitHub repo for details. Thanks for watching!"
