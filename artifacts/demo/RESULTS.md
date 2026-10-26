# Synthetic benchmark results

Seed: 42. Source: synthetic data only.
Train days 0–59; calibration days 60–89; holdout days 90–119.
The same synthetic employees appear in each period; unseen-employee generalization is not measured.

- **autoencoder**: AP=0.8756; precision=0.8452; recall=0.7889; FPR=0.0076; TP=71, FP=13, FN=19, TN=1697.
- **pca**: AP=0.8369; precision=0.8415; recall=0.7667; FPR=0.0076; TP=69, FP=13, FN=21, TN=1697.
- **isolation_forest**: AP=0.6571; precision=0.7455; recall=0.4556; FPR=0.0082; TP=41, FP=14, FN=49, TN=1696.
- **max_abs_z_rule**: AP=0.8590; precision=0.7500; recall=0.8000; FPR=0.0140; TP=72, FP=24, FN=18, TN=1686.

AP is average precision, not trapezoidal PR-AUC. Test anomaly prevalence is 5%.
No architecture or threshold selection is performed on the holdout.
This is a generator-specific demonstration, not evidence of fraud detection quality at a bank.
Model comparisons are reported as observed; an autoencoder need not beat simpler baselines.
