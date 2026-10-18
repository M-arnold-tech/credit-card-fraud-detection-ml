# credit-card-fraud-detection-ml

# Comparative Analysis of Traditional ML vs. Deep Learning for Credit Card Fraud Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository hosts the machine learning code and evaluation suite for an end-to-end classification benchmark on credit card transaction fraud. The project contrasts traditional statistical machine learning (Logistic Regression, Random Forest, XGBoost) with deep learning architectures (TensorFlow/Keras Multilayer Perceptrons) across 284,807 transactions exhibiting extreme empirical class imbalance (0.172% fraud, 577:1 ratio).

## Project Links
* **Google Drive Artifacts (Raw Data & Saved Models):** [Google Drive Link](YOUR_SHARED_GOOGLE_DRIVE_URL)
* **Video Presentation Defense:** [YouTube / Presentation Link](YOUR_VIDEO_URL)
* **Academic Paper:** See `Academic_Summative_Report.pdf`

---

## Benchmark Progression (7 Experiments)

| Exp ID | Paradigm | Model Architecture | Key Settings | Test PR-AUC | Test ROC-AUC | Recall | Precision | F1-Score |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **E1** | Classical ML | Balanced Logistic Regression | $C=0.01$, L2 penalty, balanced weights | 0.7182 | 0.9745 | 0.8919 | 0.0555 | 0.1044 |
| **E2** | Classical ML | Balanced Random Forest | 150 trees, max depth 12, subsample weights | 0.8521 | 0.9790 | 0.8243 | 0.8472 | 0.8356 |
| **E3** | Classical ML | Cost-Sensitive XGBoost | 200 trees, scale_pos_weight=577.9, depth 4 | **0.8694** | **0.9842** | 0.8514 | 0.8182 | 0.8344 |
| **E4** | Deep Learning | Sequential MLP (Baseline) | 2 layers [64, 32], unweighted, Adam | 0.7410 | 0.9412 | 0.6351 | 0.8868 | 0.7399 |
| **E5** | Deep Learning | Weighted Sequential MLP | 2 layers [64, 32], inverse frequency weights | 0.8124 | 0.9754 | 0.8378 | 0.5636 | 0.6739 |
| **E6** | Deep Learning | Regularized Deep MLP | 3 layers [128, 64, 16], BatchNorm, Dropout, L2 | 0.8492 | 0.9815 | 0.8243 | 0.7625 | 0.7922 |
| **E7** | Optimization | Tuned Deep MLP (Optimal Cutoff) | Model E6 evaluated at validation $\tau^* = 0.72$ | 0.8580 | 0.9815 | 0.8108 | **0.8824** | **0.8451** |

---

## Technical Highlights
1. **Zero Data Leakage:** Preprocessing transformations (`RobustScaler` on `Time` and `Amount`) are fit strictly on the training partition ($70\%$) and applied out-of-sample to validation ($15\%$) and test ($15\%$) sets.
2. **Asynchronous Deep Learning Pipelines:** Data ingestion is handled via `tf.data.Dataset` using batching ($N=2048$) and `prefetch(buffer_size=tf.data.AUTOTUNE)` to eliminate GPU/accelerator starvation.
3. **Threshold Calibration:** Mitigates cost-sensitive probability distortion by discovering the optimal decision frontier $\tau^*$ via validation Precision-Recall sweeps, reducing false alarms by $58.3\%$.

---

## Repository Structure
```text
├── credit_card_fraud_pipeline.ipynb  # Fully executed notebook with inline outputs
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Omits raw CSV, zip files, and checkpoints
├── README.md                         # Project documentation and summary
└── figures/                          # Exported diagnostic plots (PR/ROC curves, CMs)
