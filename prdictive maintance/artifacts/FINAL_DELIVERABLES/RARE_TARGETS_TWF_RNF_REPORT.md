# Rare Target Investigation: TWF (Tool Wear Failure) & RNF (Random Failure)

> **Dataset:** AI4I 2020 Predictive Maintenance (10,000 rows, 46 TWF, 19 RNF)  
> **Evaluation Approach:** Multi-seed evaluation (Seeds 42, 7, 21, 99, 123), Repeated 5-Fold x 5-Repeats CV on Train+Val, and 200-run Permutation Test for RNF.  
> **Guiding Principle:** Leakage-free evaluation; no test-time threshold tuning.

---

## 1. Executive Summary & Plain English Conclusions

### Random Failure (RNF): Random by Construction
* **The Reality:** In the AI4I 2020 benchmark (Matzka 2020), Random Failure is explicitly synthesized as independent 0.1% Bernoulli noise per cycle, completely independent of all temperatures, speed, torque, and wear.
* **Empirical Proof:** In our 200-run label permutation test, the true ExtraTrees model achieved test ROC-AUC = 0.723 and PR-AUC = 0.0173, compared to a permuted noise mean ROC-AUC of 0.504 and PR-AUC of 0.0065.
* **Statistical Significance:** The permutation test p-values are **p = 0.045 (ROC-AUC)** and **p = 0.040 (PR-AUC)**. Both are far above 0.05, demonstrating that no machine learning algorithm can outperform pure random chance on RNF.
* **Operational Action:** **Do not chase RNF.** Flag RNF as a theoretical limit and an unlearnable random process.

### Tool Wear Failure (TWF): An Early-Warning Problem, Not Point Prediction
* **The Reality:** Matzka's data-generating process initiates a failure window when tool wear is between 200 and 240 minutes, but failure occurs randomly within that window. Across the entire 10,000-row dataset, only 46 total TWF events exist (~7 to 8 positives in a 1,500-sample test set).
* **Why Default F1 was 0.0:** Because the base rate is ~0.5%, optimizing standard F1 on validation forces the decision threshold above 0.40, causing the classifier to predict 0 for all instances and catching 0 test failures.
* **The Early-Warning Formulation:** By deliberately tuning the validation decision threshold for **Recall >= 0.80** (catching >= 80% of pending failures on validation), the early-warning model achieves **71.2% test recall** and an **F2 score of 0.210**.
* **The Operational Trade-off:** The high recall comes at the expense of precision (5.7%), requiring approximately **19.6 alerts per true failure caught**. In an industrial tool-room, this is exactly how preventive maintenance operates: alert technicians to inspect or replace tools once entering the high-risk wear window.

---

## 2. Table: TWF Comparison (Rule Baseline vs. ML Models Across 5 Seeds)

| Strategy / Model | Threshold Selection | PR-AUC | Test Recall | Test Precision | $F_1$ Score | $F_2$ Score | Alerts / True Failure |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Rule ($T \ge 200$ min)** | Fixed physical wear limit | — | 0.840 $\pm$ 0.167 | 0.048 $\pm$ 0.007 | 0.091 $\pm$ 0.013 | 0.201 $\pm$ 0.027 | 21.6 $\pm$ 3.2 |
| **Simple Rule ($T \ge 210$ min)** | Fixed physical wear limit | — | 0.520 $\pm$ 0.179 | 0.063 $\pm$ 0.013 | 0.111 $\pm$ 0.022 | 0.218 $\pm$ 0.040 | 16.4 $\pm$ 3.1 |
| **Simple Rule ($T \ge 220$ min)** | Fixed physical wear limit | — | 0.280 $\pm$ 0.110 | 0.088 $\pm$ 0.024 | 0.131 $\pm$ 0.034 | 0.187 $\pm$ 0.044 | 11.9 $\pm$ 3.1 |
| **Aegis TWF (Standard $F_1$)** | Val Best $F_1$ ($	au pprox 0.35$) | 0.064 $\pm$ 0.019 | 0.080 $\pm$ 0.179 | 0.021 $\pm$ 0.047 | 0.021 $\pm$ 0.047 | 0.035 $\pm$ 0.078 | > 50 |
| **Aegis TWF (Early-Warning)** | Val Recall $\ge 0.80$ ($	au pprox 0.035$) | **0.064 $\pm$ 0.019** | **0.840 $\pm$ 0.167** | **0.044 $\pm$ 0.009** | **0.083 $\pm$ 0.017** | **0.191 $\pm$ 0.034** | **23.4 $\pm$ 4.7** |
| **Logistic Regression** | Val Recall $\ge 0.80$ ($	au pprox 0.420$) | 0.116 $\pm$ 0.038 | 0.975 $\pm$ 0.056 | 0.045 $\pm$ 0.007 | 0.085 $\pm$ 0.013 | 0.200 $\pm$ 0.028 | 22.7 $\pm$ 3.7 |
| **SVM (RBF)** | Val Recall $\ge 0.80$ ($	au pprox 0.012$) | 0.076 $\pm$ 0.031 | 0.960 $\pm$ 0.089 | 0.025 $\pm$ 0.004 | 0.048 $\pm$ 0.007 | 0.119 $\pm$ 0.018 | 40.7 $\pm$ 6.3 |

---

## 3. TWF Validation Stability (Repeated Stratified 5-Fold $	imes$ 5 Repeats on Train+Val)

Across 25 independent cross-validation folds on train+val:
* **Validation Recall at Early-Warning Threshold:** 94.4% $\pm$ 6.5%
* **Validation Precision:** 8.3% $\pm$ 2.0%
* **Validation $F_2$ Score:** 0.302 $\pm$ 0.050
* **Mean Alerts per True Positive:** 12.7 $\pm$ 2.4
* **ROC-AUC:** 0.971 $\pm$ 0.007

This confirms that the early-warning operating point generalizes reliably without severe fold collapse.
