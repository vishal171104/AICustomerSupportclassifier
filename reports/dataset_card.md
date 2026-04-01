# Dataset Card: Customer Support Ticket Triage Dataset v1.0

## Overview
A curated, balanced dataset of support ticket descriptions designed for evaluating
automated triage systems combining **category classification** (3 classes) and
**priority prediction** (4 classes) — two simultaneous NLP classification tasks.

---

## Dataset Provenance

| Field | Value |
|---|---|
| **Version** | 1.0 |
| **Size** | 1,082 samples |
| **Format** | CSV (UTF-8) |
| **License** | CC BY 4.0 |
| **SHA-256 (tickets.csv)** | `5f2778374fe934f5ba33671d39ab4950f3688fba98993f17c13ef87d5bcdfa78` |
| **Source** | Synthetic, LLM-assisted paraphrasing of real-world helpdesk templates |
| **Language** | English |

---

## Tasks

| Task | Type | Classes |
|---|---|---|
| `category` | Multi-class classification | Technical, Billing, Account |
| `priority` | Multi-class classification | Low, Medium, High, Critical |

---

## Class Distribution

### Category (n=1,082)

| Class | Count | % |
|---|---|---|
| Technical | 683 | 63.1% |
| Billing | 214 | 19.8% |
| Account | 185 | 17.1% |

> **Note**: Technical class is intentionally larger, reflecting realistic helpdesk distributions
> where infrastructure/product issues dominate.

### Priority (n=1,082)

| Class | Count | % |
|---|---|---|
| Critical | 294 | 27.2% |
| High | 278 | 25.7% |
| Medium | 264 | 24.4% |
| Low | 246 | 22.7% |

> Priority distribution is near-uniform, ensuring all severity levels are well-represented
> for robust classifier evaluation.

---

## Schema

| Column | Type | Description |
|---|---|---|
| `ticket_id` | int | Unique identifier |
| `description` | str | Raw ticket text (avg. 86.3 chars) |
| `category` | str | Ground-truth category label |
| `priority` | str | Ground-truth priority label |

---

## Data Collection and Annotation

- **Method**: Template-based generation with GPT-4-assisted paraphrasing
  to simulate natural language variety across departments
- **Quality Control**: Expert review of 100% of samples for label consistency
- **Ambiguous cases**: ~3% samples independently reviewed by two annotators;
  consensus label used; disagreements logged in `data/annotation_log.md`
- **Text cleaning applied**: lowercase, URL removal, digit stripping,
  punctuation normalisation (see `utils/preprocessing.py`)

---

## Train / Validation / Test Splits

| Split | Size | % | Method |
|---|---|---|---|
| Train | 865 | 80% | Stratified random split |
| Test | 217 | 20% | Stratified random split |
| **CV** | — | — | 5-fold stratified CV (primary evaluation) |

> Random seed: **42** throughout. See `model/publication_eval.py` for reproducibility.

---

## Bias and Limitations

| Bias Type | Description |
|---|---|
| **Keyword Trap** | "urgent" correlates with Critical priority even when used informally |
| **Length Bias** | Technical tickets average longer than Billing/Account tickets |
| **Domain Scope** | Tickets simulate SaaS/software helpdesk only — may not generalise to healthcare, legal, or hardware support domains |
| **Language** | English only — multilingual tickets not represented |
| **Synthetic origin** | Some phrasing patterns may be less naturalistic than 100% real-world data |

---

## Ethical Statement

- **PII**: Zero real Personally Identifiable Information. All names, emails, IDs
  are randomly generated placeholders.
- **Annotation Bias Mitigation**: Annotators were instructed to focus on semantic
  content rather than trigger keywords (e.g., "urgent") to avoid label bias.
- **Intended Use**: Research, benchmarking NLP classifiers for support triage.
- **Prohibited Use**: Direct deployment on private user data without additional
  privacy review.

---

## Citation

If you use this dataset, please cite:
```
@misc{customerticketai2026,
  title  = {Customer Support Ticket Triage Dataset v1.0},
  author = {Vishal S I},
  year   = {2026},
  note   = {CC BY 4.0. SHA-256: 5f2778374fe934f5ba33671d39ab4950f3688fba98993f17c13ef87d5bcdfa78}
}
```
