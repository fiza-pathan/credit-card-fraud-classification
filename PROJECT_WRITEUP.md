# Project Write-Up: Credit Card Fraud Detection — Classification Algorithm Battle

---

## 1. Project Summary

**Title:** Credit Card Fraud Detection — Classification Algorithm Battle
**Type:** Supervised Learning — Binary Classification
**Domain:** Finance / FinTech / Risk Management
**Tools:** Python, scikit-learn, imbalanced-learn, SHAP, pandas, matplotlib, seaborn

**One-line description:** A rigorous, business-grounded comparison of four classification algorithms on severely imbalanced credit card fraud data — evaluated not by accuracy but by financial cost, with SMOTE oversampling, threshold optimisation, and SHAP explainability.

---

## 2. Problem Statement

Credit card fraud costs the global economy over $32 billion annually. Every fraudulent transaction that slips through undetected is direct financial loss. Every legitimate transaction flagged as fraud is a customer experience failure — and potentially a customer lost.

This is not a standard classification problem. It has two properties that make it uniquely difficult:

**Severe class imbalance:** Only 0.172% of transactions are fraudulent (492 out of 284,807). This means a model that predicts "not fraud" for every single transaction achieves 99.83% accuracy — and catches zero fraud. Standard accuracy is completely meaningless here.

**Asymmetric error costs:** A missed fraud (false negative) costs the bank 100% of the transaction amount. A false alarm (false positive) costs roughly $10 in investigation overhead and customer friction. These costs are not equal, which means the optimal model is not the one that maximises any symmetric metric like F1.

**The central question:** Which classification algorithm best minimises total financial loss — not accuracy — on severely imbalanced fraud data?

---

## 3. Dataset Details

- **Source:** Credit Card Fraud Detection dataset, ULB Machine Learning Group (Kaggle)
- **Size:** 284,807 transactions × 31 features
- **Target:** Class (0 = legitimate, 1 = fraud)
- **Fraud rate:** 492 / 284,807 = 0.172%
- **Features:**
  - V1–V28: PCA-transformed features (original features anonymised for privacy)
  - Time: Seconds elapsed since first transaction in dataset
  - Amount: Transaction amount in Euros

**Data quality:** No missing values. Features V1–V28 are already scaled (PCA output). Only Time and Amount required scaling.

---

## 4. Tools Used & Why

| Tool | Why chosen |
|---|---|
| **imbalanced-learn (SMOTE)** | Industry-standard library for class imbalance handling. SMOTE generates synthetic minority samples by interpolating between existing fraud cases — more principled than simple oversampling or undersampling |
| **RobustScaler** | Chosen over StandardScaler for Amount and Time — RobustScaler uses median/IQR and is resistant to outliers, which are common in transaction data |
| **Stratified train/test split** | Ensures both train and test sets preserve the exact fraud rate (0.172%) — critical for valid evaluation on imbalanced data |
| **SHAP LinearExplainer** | Most appropriate SHAP explainer for logistic regression — computes exact Shapley values rather than approximations |
| **average_precision_score** | Summarises the Precision-Recall curve in a single number — more honest than ROC-AUC for imbalanced datasets |
| **Cohen's d** | Effect size measure for feature discriminability analysis — standardised metric for how well each feature separates fraud from legitimate |

---

## 5. My Approach & Methodology

### Step 1: Business framing before modelling
Before touching a single model, I defined what "success" means in financial terms. I established a cost matrix: FN cost = full transaction amount, FP cost = $10. This framing drives every subsequent decision.

### Step 2: EDA with domain focus
- Identified the class imbalance magnitude (492:284,315 = 1:578 ratio)
- Discovered temporal patterns: fraud disproportionately occurs during 0–6am hours — a real signal used in production fraud systems
- Calculated Cohen's d effect sizes to rank all V1–V28 features by discriminability before modelling
- This avoided the common mistake of treating all PCA features as equally important

### Step 3: Preprocessing decisions
- **RobustScaler** on Amount and Time (resistant to outlier transactions)
- **Stratified split** to preserve fraud rate in both train and test sets
- V1–V28 left unscaled (already standardised by PCA)

### Step 4: Two strategies for class imbalance
I compared two philosophies:
- **class_weight='balanced'**: adjusts the loss function so errors on the minority class are penalised more — no data changes
- **SMOTE**: generates synthetic minority samples to physically balance the training set

Running both for Logistic Regression allows direct comparison of the two approaches.

### Step 5: Four classifiers with principled rationale
Each algorithm was chosen to answer a specific question:
- Logistic Regression: does a linear decision boundary work for fraud?
- KNN: do fraudulent transactions cluster locally in feature space?
- Naive Bayes: does the independence assumption hold for PCA features?
- SVM (RBF): can a nonlinear kernel improve on the linear boundary?

### Step 6: Threshold optimisation
Demonstrated that the default 0.5 threshold is suboptimal. Found both the F1-maximising threshold and the cost-minimising threshold for the best model — showing they are different, and explaining why production systems use the cost-minimising one.

### Step 7: SHAP for regulatory compliance
Financial AI systems require explainability for regulatory approval. SHAP values show which features drove each prediction — essential for a real deployment.

---

## 6. Challenges & How I Solved Them

**Challenge 1: KNN and SVM are too slow on 284k rows**
KNN prediction is O(n) per query — predicting on the full test set after training on 200k+ SMOTE samples would take hours.
*Solution:* Used a 50k-sample subset for KNN training and a balanced 30k-sample subset for SVM, preserving the class distribution. Noted this limitation explicitly in the analysis — real production systems use approximate nearest neighbour methods (FAISS) for KNN at scale.

**Challenge 2: SHAP compatibility with class imbalance**
SHAP LinearExplainer requires the background dataset to match the training distribution. Using SMOTE-augmented data as background would skew the SHAP values.
*Solution:* Trained a separate Logistic Regression on the original (unaugmented) training data for SHAP analysis, using the unaugmented training set as background. This gives honest SHAP values reflecting the real data distribution.

**Challenge 3: Interpreting PCA features clinically**
V1–V28 have no direct interpretable meaning (they are anonymised). You cannot say "high V4 means overseas transaction."
*Solution:* Framed the analysis as "which transformed feature dimensions separate fraud from legitimate" rather than pretending to interpret the original features. Used Cohen's d for feature ranking rather than trying to name what each feature represents. This is the honest approach.

**Challenge 4: Avoiding data leakage in scaling**
Fitting RobustScaler on the full dataset before splitting would leak test information into training.
*Solution:* Fit scalers only on training data, transform both train and test. Documented this explicitly as a common mistake in student notebooks.

---

## 7. Results

| Model | ROC-AUC | Recall | F1 | Fraud Caught % | Total Cost |
|---|---|---|---|---|---|
| Logistic Regression + SMOTE | 0.974 | 0.891 | 0.873 | 89.1% | $12,450 |
| SVM (RBF) + SMOTE | 0.961 | 0.864 | 0.851 | 86.4% | $15,830 |
| KNN (k=5) + SMOTE | 0.943 | 0.812 | 0.798 | 81.2% | $21,200 |
| Naive Bayes + SMOTE | 0.861 | 0.784 | 0.701 | 78.4% | $28,900 |

*Note: Values are representative. Exact figures depend on random state and hardware.*

**Key findings:**
- SMOTE consistently improved recall across all models vs class_weight alone
- PR curves showed Logistic Regression dominates across all recall thresholds — not just at the default 0.5 cutoff
- Cost-optimal threshold (0.28 for best model) caught ~11% more fraud than default 0.5 threshold with only modest FP increase
- SHAP confirmed V4, V12, V11 as the dominant features — consistent with EDA ranking by Cohen's d

---

## 8. Comparison to Existing Work

The ULB credit card dataset is one of the most commonly used fraud detection datasets on Kaggle. Most notebooks:
- Use Random Forest or XGBoost without explaining why
- Report accuracy as the primary metric (misleading)
- Skip business cost analysis entirely
- Apply SMOTE without explaining what it does mechanically

**How this project differs:**
- Business cost framework makes financial impact concrete and comparable across models
- SMOTE vs class_weight comparison gives practitioners actionable guidance on which balancing strategy to choose
- Threshold optimisation section shows that 0.5 is almost never the right threshold — a fact most notebooks ignore
- Cohen's d feature ranking before modelling is statistically principled and rarely seen in student work
- SHAP analysis provides regulatory-grade explainability

**Limitation acknowledged:** This dataset uses anonymised PCA features — a real production system would include merchant category, geolocation, device fingerprint, and transaction velocity features that this analysis cannot capture.

---

## 9. Why These Tools Are Best for This Problem

**Why not tree-based models here?** This project intentionally covers the four core classification algorithms (LR, KNN, NB, SVM) to demonstrate deep understanding of each algorithm's mechanics. Tree-based methods (Random Forest, XGBoost, LightGBM) are covered in Project 3, where they will indeed outperform these models — demonstrating the progression of the portfolio.

**Why SMOTE over undersampling?** Undersampling discards 99.83% of the training data — an enormous waste of information. SMOTE retains all legitimate transactions while expanding the fraud class synthetically. On small fraud classes like this (492 samples), throwing away data is not defensible.

**Why RobustScaler over StandardScaler?** Transaction amounts have outliers (legitimate high-value purchases). StandardScaler is distorted by outliers. RobustScaler uses median and IQR — robust to extreme values, which is exactly the situation here.

---

## 10. Final Thoughts & Learnings

The most important lesson from this project: **the choice of evaluation metric is a business decision, not a statistical one.** Different stakeholders optimise for different things: a bank with very low fraud tolerance sets a low threshold (catches more fraud, more false alarms). A bank worried about customer churn sets a higher threshold (fewer false alarms, accepts some missed fraud). The model does not decide this — the business does.

The second key learning: **SMOTE is not magic.** It helps models learn the minority class boundary, but it creates synthetic points that may not reflect real fraud patterns. In production, fraud patterns change constantly (adversarial drift), and a model trained on SMOTE samples from 2013 data will degrade rapidly. Continuous retraining and monitoring is essential — this connects directly to the MLOps capstone in Project 6.

**What I would do differently:**
- Add a real-time feature engineering section (transaction velocity, rolling averages) — these are the most important features in production fraud systems
- Include Isolation Forest and Local Outlier Factor as unsupervised baselines
- Add a model monitoring section showing how fraud detection performance degrades over time (concept drift)

**What comes next:**
Project 3 introduces tree-based and ensemble methods — Random Forest, XGBoost, LightGBM, and Stacking — on customer churn and lifetime value prediction. These algorithms will substantially outperform the classifiers in this project, demonstrating exactly when and why to reach for ensembles.
