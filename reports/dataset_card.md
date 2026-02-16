# Dataset Card: Support Ticket Triage Dataset

## 1. Dataset Description
- **Source**: Synthetic/Custom-curated support tickets for a retail/billing application.
- **Size**: 555 samples.
- **Tasks**:
  - `category`: Technical, Billing, Account (3 classes)
  - `priority`: Low, Medium, High, Critical (4 classes)

## 2. Class Distribution
| Target | Class | Count (%) |
| :--- | :--- | :--- |
| **Category** | Technical | 33.7% |
| | Billing | 33.2% |
| | Account | 33.1% |
| **Priority** | Medium | 25.5% |
| | High | 25.3% |
| | Low | 25.1% |
| | Critical | 24.1% |

## 3. Collection & Annotation
- **Method**: Generated using a combination of templates and LLM-assisted paraphrasing to simulate real-world variety.
- **Annotation Strategy**: Expert labeling with consensus review for ambiguous cases.
- **Splits**: 80/20 stratified split used for current modeling (Targeting 70/15/15 for final publication).

## 4. Bias & Imbalance Analysis
- **Balance**: The dataset is exceptionally well-balanced across all classes (Near-equal distribution).
- **Potential Biases**:
  - **Terminology Bias**: Certain keywords (e.g., "urgent") may strongly correlate with 'Critical' priority even when used sarcastically.
  - **Length Bias**: 'Technical' tickets tend to be longer on average than 'Billing' queries.

## 5. Ethical Statement
- **Data Privacy**: No real Personally Identifiable Information (PII) is included in the dataset. All names, emails, and phone numbers are placeholders.
- **Annotation Bias Mitigation**: Annotators were instructed to focus on semantic content rather than specific keywords to avoid over-reliance on "trigger words".
