# Error Analysis Report — Priority

**Model**: Logistic Regression + TF-IDF (1–2-grams, 5000 features)  
**Data**: 1082 samples, 5-fold cross-validation

## Overall Performance
```
              precision    recall  f1-score   support

    Critical       0.76      0.76      0.76       294
        High       0.72      0.74      0.73       278
         Low       0.76      0.75      0.75       246
      Medium       0.69      0.67      0.68       264

    accuracy                           0.73      1082
   macro avg       0.73      0.73      0.73      1082
weighted avg       0.73      0.73      0.73      1082

```

## Error Summary

- **Total errors**: 291 / 1082 = 26.89%

- **Mean confidence on incorrect predictions**: 0.355

- **Mean confidence on correct predictions**: 0.704


## Low-Confidence Bucket Analysis

|   Conf < Threshold |   Tickets in Bucket |   Errors in Bucket | Error Rate   |
|-------------------:|--------------------:|-------------------:|:-------------|
|                0.5 |                 511 |                284 | 55.6%        |
|                0.6 |                 565 |                290 | 51.3%        |
|                0.7 |                 585 |                290 | 49.6%        |
|                0.8 |                 626 |                291 | 46.5%        |


## Top Confusion Pairs

| Actual   | Predicted   |   Confidence | Text (first 120 chars)                                                                                                   |
|:---------|:------------|-------------:|:-------------------------------------------------------------------------------------------------------------------------|
| Critical | High        |        0.727 | Internal server error in the profile panel. System is down. Production is impacted.                                      |
| Critical | High        |        0.551 | Internal server error profile panel blocking all users. Production down since weekend.                                   |
| High     | Low         |        0.55  | Important: The MFA settings are degraded. I need to check my security settings. Production is impacted.                  |
| Medium   | High        |        0.528 | Why is there a discrepancy in the latest charge? It says INV-202 but the amount is different. Need this fixed soon.      |
| Low      | Critical    |        0.525 | Why is there a discrepancy in the invoice? It says INV-202 but the amount is different. It's a critical blocker. Best re |
| Medium   | Low         |        0.507 | i was expecting email notification received notification about not going through medium billing                          |
| High     | Low         |        0.499 | The latets charge from the last update seems higher than expected. I checked the address and it looks okay. Sorry for th |
| High     | Critical    |        0.487 | The view logs level is insufficient for my contact email. It keeps hangs. Thanks for the help.                           |
| Low      | Medium      |        0.484 | The dashboard is degraded. I was trying to submit but it just glitches. Sorry for the trouble.                           |
| Medium   | High        |        0.483 | Rotine: The item is not appearing as expected in the module. Can you check the mobile device logs? Need this fixed soon. |


## Interpretation

- **Critical → High** (conf=0.727): *"Internal server error in the profile panel. System is down. Production is impacted."*

- **Critical → High** (conf=0.551): *"Internal server error profile panel blocking all users. Production down since weekend."*

- **High → Low** (conf=0.55): *"Important: The MFA settings are degraded. I need to check my security settings. Production is impacted."*

- **Medium → High** (conf=0.528): *"Why is there a discrepancy in the latest charge? It says INV-202 but the amount is different. Need this fixed soon."*

- **Low → Critical** (conf=0.525): *"Why is there a discrepancy in the invoice? It says INV-202 but the amount is different. It's a critical blocker. Best re"*
