# Recipe Traffic Decision Engine

## Project Overview

This project develops a **machine learning decision framework** to identify and promote recipes with high expected traffic.
Instead of focusing only on classification accuracy, the project applies concepts from **credit risk modeling and portfolio strategy** to optimize promotion decisions under uncertainty.

The goal is not only to predict which recipes will generate high traffic, but to determine:

* Which recipes should be promoted
* How many should be promoted
* The trade-off between opportunity and risk
* The promotion strategy that maximizes expected value

---

## Business Motivation

Platforms promoting digital content face a key decision problem:

> Promoting too many recipes wastes visibility resources, while promoting too few misses potential high-traffic opportunities.

This project reframes the problem similarly to **credit approval strategy** used in banking:

| Credit Risk Concept | Recipe Platform Equivalent |
| ------------------- | -------------------------- |
| Default probability | Low traffic probability    |
| Acceptance rate     | Promotion rate             |
| Bad rate            | Failed promotions          |
| Portfolio value     | Traffic value generated    |

---

## Methodology

### 1. Data Preparation

* Feature engineering and cleaning
* Train/Test split with stratification
* Class imbalance handling using undersampling

---

### 2. Modeling

Models evaluated:

* Logistic Regression
* XGBoost Classifier (final selected model)

Evaluation metrics:

* Classification report
* Macro F1-score
* ROC Curve & AUC

---

### 3. Model Calibration

Calibration curves were used to evaluate whether predicted probabilities could be interpreted as confidence levels.

A well-calibrated model ensures that:

> predicted probabilities reflect real-world outcomes.

---

### 4. Promotion Strategy (Decision Layer)

Instead of using a fixed classification threshold:

* Promotion decisions are based on **quantiles**.
* Different promotion rates were tested (10%–100%).

Example:

* Promote only the top 20% highest predicted traffic recipes.

---

### 5. Bad Rate Analysis

Bad Rate measures:

> Percentage of promoted recipes that did NOT achieve high traffic.

This allows evaluation of operational risk in promotion decisions.

---

### 6. Strategy Optimization

A strategy table was created evaluating:

* Promotion Rate
* Threshold
* Bad Rate
* Number of Promoted Recipes
* Estimated Business Value

This produces a **risk–reward frontier**, similar to portfolio optimization in finance.

---

## Key Insights

* Higher promotion rates increase opportunity but also increase failure risk.
* Optimal promotion level emerges from balancing expected gains and promotion errors.
* Machine learning becomes a **decision system**, not just a prediction tool.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib

---

## Future Improvements

* Probability calibration (Platt / Isotonic)
* Cost-sensitive learning
* Bayesian optimization of promotion thresholds
* Real-time deployment pipeline

---

## Author

Data Science & Decision Modeling Project
Focused on applied machine learning, risk modeling, and strategy optimization.
