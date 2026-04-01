# Error Analysis Report — Category

**Model**: Logistic Regression + TF-IDF (1–2-grams, 5000 features)  
**Data**: 1082 samples, 5-fold cross-validation

## Overall Performance
```
              precision    recall  f1-score   support

     Account       0.99      1.00      0.99       185
     Billing       1.00      1.00      1.00       214
   Technical       1.00      1.00      1.00       683

    accuracy                           1.00      1082
   macro avg       1.00      1.00      1.00      1082
weighted avg       1.00      1.00      1.00      1082

```

## Error Summary

- **Total errors**: 2 / 1082 = 0.18%

- **Mean confidence on incorrect predictions**: 0.524

- **Mean confidence on correct predictions**: 0.846


## Low-Confidence Bucket Analysis

|   Conf < Threshold |   Tickets in Bucket |   Errors in Bucket | Error Rate   |
|-------------------:|--------------------:|-------------------:|:-------------|
|                0.5 |                   8 |                  1 | 12.5%        |
|                0.6 |                  38 |                  1 | 2.6%         |
|                0.7 |                  98 |                  2 | 2.0%         |
|                0.8 |                 214 |                  2 | 0.9%         |


## Top Confusion Pairs

| Actual    | Predicted   |   Confidence | Text (first 120 chars)                                                                                                   |
|:----------|:------------|-------------:|:-------------------------------------------------------------------------------------------------------------------------|
| Technical | Account     |        0.664 | Settings area returning 500 sometimes. Not consistent but happening today.                                               |
| Technical | Account     |        0.384 | I noticed the side section is showing inconsistencies on iOS. Also, but I have a billing question. Sorry for the trouble |


## Interpretation

- **Technical → Account** (conf=0.664): *"Settings area returning 500 sometimes. Not consistent but happening today."*

- **Technical → Account** (conf=0.384): *"I noticed the side section is showing inconsistencies on iOS. Also, but I have a billing question. Sorry for the trouble"*
