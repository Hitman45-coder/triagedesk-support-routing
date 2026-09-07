# TriageDesk model card

## Current artifact

`banking77-0.1.0`, policy `policy-0.1.0`. The full artifact uses a word and character TF-IDF feature union with logistic regression, then fits a sigmoid calibrator on a disjoint calibration partition. A separate policy-validation partition selects the persisted confidence and margin thresholds.

## Evaluation

The model was fit on 6,997 rows, calibrated on 1,499 rows, and policy-validated on 1,500 rows. Seven normalized train/test duplicates were removed from development rows. The recorded results are accuracy `0.8951` and macro-F1 `0.8942` on 3,080 test rows across 77 labels. The frozen policy accepted 2,379 test rows (`77.2%` coverage) with `96.0%` selective accuracy.

These results are benchmark evidence for this artifact only. They do not establish performance on live traffic, unseen intents, other languages, or real banking operations. The application still requires human review for policy-sensitive and below-threshold cases, and representative out-of-domain evaluation remains incomplete.

## Intended output

The model suggests an intent and queue. Security-related or uncertain outputs are sent to human review. It must not trigger payments, card changes, fraud actions, or customer communications.
