# Current evaluation report

This report describes `banking77-0.1.0` and `policy-0.1.0`, generated from the release manifest. It is intentionally separate from the application code so a UI claim can be traced back to an artifact.

| Measurement | Value |
|---|---:|
| Official test rows | 3,080 |
| Intent labels | 77 |
| Accuracy | 0.8951 |
| Macro-F1 | 0.8942 |
| Policy-validation rows | 1,500 |
| Confidence threshold | 0.45 |
| Margin threshold | 0.05 |
| Automatic-routing coverage on test | 77.2% |
| Selective accuracy on accepted test rows | 96.0% |
| Accepted test rows | 2,379 |

The model is a word/character TF-IDF feature union with logistic regression, calibrated with sigmoid calibration on a disjoint partition. The thresholds were selected only on the policy-validation partition. Security-sensitive intents remain review-required regardless of score.

The official test was not used to fit features, calibrate probabilities, or choose thresholds. Seven normalized train/test duplicate texts were audited and removed from development fitting. The test file remains unchanged. These numbers describe a benchmark artifact and do not establish real-bank performance, live accuracy, multilingual behavior, or out-of-domain safety.
