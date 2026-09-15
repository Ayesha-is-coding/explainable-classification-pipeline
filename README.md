# Explainable Classification & Imbalance-Handling Pipeline

A reusable methodology for binary classification: model comparison via cross-validation, a class-imbalance-handling ablation, and SHAP-based interpretability — demonstrated here on the Breast Cancer Wisconsin (Diagnostic) dataset.

## Problem
Binary classification of breast tumor diagnoses (malignant vs. benign) from digitized cell-nuclei measurements, with a focus on comparing models and explaining what drives predictions rather than optimizing a single accuracy number.

## Data
Breast Cancer Wisconsin (Diagnostic) dataset, 569 samples, 30 numeric features, class balance 62.7% benign / 37.3% malignant, no missing values.

## Method
Logistic Regression, Random Forest, and HistGradientBoosting compared via 5-fold stratified cross-validation. Class-imbalance handling (class weighting, SMOTE) tested against a plain baseline. SHAP TreeExplainer used to interpret the best-performing tree model.

## Results — model comparison (5-fold CV)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.980 | 0.980 | 0.989 | 0.984 |
| HistGradientBoosting | 0.971 | 0.969 | 0.986 | 0.977 |
| Random Forest | 0.954 | 0.959 | 0.968 | 0.963 |

## Results — imbalance-handling ablation (minority class = malignant)
| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Plain Random Forest | 0.951 | 0.929 | 0.940 |
| Class-weighted Random Forest | 0.929 | 0.929 | 0.929 |
| SMOTE-resampled Random Forest | 0.891 | 0.976 | 0.932 |

## Explainability
Top 5 SHAP features: worst area, worst concave points, worst radius, mean concave points, worst perimeter — see `shap_summary.png`.

## Finding
On this dataset, explicit imbalance-handling (class weighting, SMOTE) did not outperform the plain baseline — imbalance here is mild (62.7/37.3), so there was little room for these techniques to help, and SMOTE's synthetic minority oversampling slightly reduced precision in exchange for recall. This suggests imbalance-handling techniques are not universally beneficial and their payoff scales with how severe the imbalance actually is — a hypothesis worth testing on a severely imbalanced dataset (e.g. <1% positive class fraud data).

## Next steps
Re-run the identical pipeline on a severely imbalanced binary classification dataset (e.g. Kaggle's Credit Card Fraud Detection, ULB) to test whether the imbalance-ablation finding changes under more extreme class skew.
