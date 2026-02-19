# Introduction
Credit risk is one of the core pillars of the financial industry, as it represents the possibility that a borrower will fail to meet their repayment obligations. Poor management of credit risk can lead to significant financial losses, making robust quantitative models for estimating the **Probability of Default (PD)** essential for informed decision-making.

This project develops a comprehensive credit risk modeling framework using machine learning techniques to predict the probability that a loan will default based on financial, demographic, and credit-related information of the borrower. The analysis follows an end-to-end industry-style workflow, including data exploration, cleaning, outlier detection, and missing value treatment, as well as model training, evaluation, and calibration.

Two widely used modeling approaches in credit risk are implemented and compared:

- Logistic Regression, valued for its interpretability and transparency, allowing direct analysis of each variable’s impact on default risk.
- Gradient Boosted Trees (XGBoost), which provide higher predictive power by capturing non-linear relationships and complex feature interactions.

Beyond basic model training, the project evaluates performance using advanced metrics such as **ROC-AUC, recall, precision, F1-score, and confusion matrices**. It further **explores probability threshold selection, estimated monetary impact, class imbalance handling through undersampling, cross-validation, and probability calibration curves**, directly linking statistical performance to real-world business and risk management implications.

The result is a practical and extensible credit risk modeling pipeline that emphasizes not only predictive accuracy, but also interpretability, stability, and real-world applicability for financial decision-making.