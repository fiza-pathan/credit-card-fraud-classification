# Credit Card Fraud Detection - Classification Algorithm Battle

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Finance%20%2F%20FinTech-blue)
![Type](https://img.shields.io/badge/Type-Classification-purple)

> **Logistic Regression · K-Nearest Neighbours · Naive Bayes · Support Vector Machine**  
> With SMOTE oversampling · Cost-sensitive evaluation · Business impact analysis · SHAP explainability

**[View Kaggle Notebook](#)** | **[Portfolio](#)**

---

## The Problem

Credit card fraud costs the global economy **$32 billion annually**. This is not an accuracy problem — it is a **cost-sensitive decision problem** where:

- A **missed fraud** (False Negative) = direct financial loss to the bank
- A **false alarm** (False Positive) = customer friction + investigation cost

A model predicting *"not fraud"* for every transaction achieves **99.83% accuracy** and catches **zero fraud**. This is why standard accuracy is the wrong metric here entirely.

---

## Key Questions Answered

1. Which classifier best detects fraud in severely imbalanced data (0.17% fraud rate)?
2. How does SMOTE oversampling change algorithm rankings?
3. Which model **minimises total financial loss**, not just maximises accuracy?
4. What is the cost-optimal classification threshold, and why is it never 0.5?

---

## Results Summary

| Model | ROC-AUC | Recall | F1 | Fraud Caught | Total Cost |
|---|---|---|---|---|---|
| Logistic Reg + SMOTE | **0.974** | **0.891** | **0.873** | **89.1%** | **$12,450** |
| SVM (RBF) + SMOTE | 0.961 | 0.864 | 0.851 | 86.4% | $15,830 |
| KNN + SMOTE | 0.943 | 0.812 | 0.798 | 81.2% | $21,200 |
| Naive Bayes + SMOTE | 0.861 | 0.784 | 0.701 | 78.4% | $28,900 |

*Results approximate — actual values depend on run. Key findings hold across runs.*

**Winner: Logistic Regression + SMOTE**, fastest inference, highest AUC, most interpretable for regulatory compliance.

---

## What Makes This Project Different

### 1. Business Cost Framework
Every model is evaluated not just on F1, but on **total financial loss**:
- False Negative cost = 100% of the missed fraud transaction amount
- False Positive cost = $10 investigation cost per false alarm
- Models are ranked by dollars saved, not statistical metrics alone

### 2. SMOTE Deep Dive
Explains *why* SMOTE works, *how* synthetic samples are generated, and compares it against class_weight balancing, two different philosophies for the same problem.

### 3. Threshold Optimisation
Demonstrates that the default 0.5 threshold is rarely optimal for fraud detection. Shows how to find the cost-minimising threshold for any business tolerance level.

### 4. Precision-Recall vs ROC
Explains why PR curves are more honest than ROC curves for imbalanced datasets, and shows the difference visually.

### 5. SHAP Explainability
Identifies which PCA-transformed features drive fraud predictions, essential for regulatory compliance in financial AI.

---

## Algorithm Decision Guide

| Algorithm | Use When |
|---|---|
| **Logistic Regression** | Production baseline, fast, interpretable, regulatory-friendly, scales to billions of transactions |
| **SVM (RBF)** | Non-linear decision boundaries needed, dataset fits in memory |
| **KNN** | Local neighbourhood patterns matter, no strong distributional assumptions |
| **Naive Bayes** | Real-time streaming at microsecond latency, acceptable accuracy tradeoff |

---

## Project Structure

```
credit-card-fraud-detection/
├── credit-card-fraud-detection-classification.ipynb
├── outputs/
│   ├── 01_class_imbalance.png
│   ├── 02_time_patterns.png
│   ├── 03_feature_discriminability.png
│   ├── 04_top_feature_distributions.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_smote_effect.png
│   ├── 07_model_comparison.png
│   ├── 08_roc_pr_curves.png
│   ├── 09_confusion_matrices.png
│   ├── 10_business_cost.png
│   ├── 11_threshold_analysis.png
│   ├── 12_shap_importance.png
│   └── 13_shap_beeswarm.png
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group
- **Size:** 284,807 transactions × 31 features
- **Fraud rate:** 492 frauds = **0.172%** — extreme class imbalance
- **Features:** V1–V28 (PCA-anonymised), Time, Amount

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `scikit-learn` | All classifiers, pipelines, cross-validation, metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `shap` | Model explainability |
| `matplotlib`, `seaborn` | Visualisation (13 charts) |
| `numpy`, `pandas` | Data manipulation |

---

## How to Run

```bash
git clone https://github.com/fiza-pathan/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
# Download dataset from Kaggle and place in /data/
jupyter notebook notebooks/credit_card_fraud_detection.ipynb
```

---

## Part of My AI/ML Portfolio

**ML Phase:** Regression ✅ → **Classification** ✅ → Ensemble → Time Series → Unsupervised → Capstone  
**Next:** Project 3 — Customer Churn & Lifetime Value with Random Forest, XGBoost & LightGBM

[GitHub Profile](https://github.com/fiza-pathan) · [Kaggle](https://www.kaggle.com/fizapathan21)
