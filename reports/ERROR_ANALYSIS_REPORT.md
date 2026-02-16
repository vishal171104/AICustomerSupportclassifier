# Error Analysis Deep Dive: Semantic Ambiguity & Keyword Traps

## 1. Misclassification Taxonomy
Based on extensive testing of the Priority prediction model, we have identified the following primary error patterns:

| Error Pattern | Occurrence | Description |
| :--- | :--- | :--- |
| **Miscellaneous / Out-of-vocab** | High | Samples containing highly specific technical jargon not present in training. |
| **Negation/Context Loss** | Moderate | Failures in detecting negation (e.g., "not working" vs "not a problem"). |
| **Low Confidence / Noise** | Low | Highly ambiguous inputs where the model correctly flags low confidence. |
| **Keyword Traps** | Low | Sarcastic use of words like "urgent" or "critical" in minor feature requests. |

## 2. High-Confidence Failure Cases
These are cases where the model was "confidently wrong."

### Example 1: Context Loss
- **Text**: "The login system is not down, but the background is green."
- **Predicted**: `Critical` (Triggered by 'login system')
- **Actual**: `Low`
- **Root Cause**: Reliance on high-weight tokens without deep semantic negation processing.

### Example 2: Keyword Trap
- **Text**: "I have an extremely urgent request: please change the font size."
- **Predicted**: `High` (Triggered by 'extremely urgent')
- **Actual**: `Low`
- **Root Cause**: TF-IDF weights for 'urgent' are high, overriding the semantic insignificance of 'font size'.

## 3. Adversarial Patterns
| Adversarial Input | Model Response | Confidence | Vulnerability |
| :--- | :--- | :--- | :--- |
| `p ay m e n t is fail ing` | Incorrect | Low | Character-level spacing breaks word-level TF-IDF tokens. |
| `system working well but...` | High | High | Over-focus on system keywords vs. overall sentiment. |

## 4. Mitigation Strategies
1. **Hybrid Embeddings**: Augment TF-IDF with character-level n-grams (subword tokens) to combat spacing noise.
2. **Confidence Thresholding**: Force manual review if max probability < 0.45.
3. **Sentiment Intersection**: Disregard "urgent" keywords if overall text sentiment is neutral/positive.
