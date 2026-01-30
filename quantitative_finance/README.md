This repository contains a project related to finance, actuarial science, and quantitative modeling.  
It aims to demonstrate a solid foundation in applied mathematics, programming, and financial analytics.

---

### **Quantitative Finance Project**
A developing project focused on creating reusable modules for **financial simulation, market data processing, and quantitative model implementation**.


---

## Module Overview

### `market_universe/`
Stores historical price data (CSV files) for financial assets.  
Each file represents a single instrument and serves as input for all models.

---

### `market_data.py`
Handles **market data ingestion, preprocessing, synchronization, and statistical analysis**.

Key features:
- Log-return computation
- Cross-asset time series synchronization
- Annualized return and volatility
- Sharpe Ratio, Value at Risk (VaR)
- Skewness, kurtosis, Jarque–Bera normality test
- Distribution fitting using Tukey Lambda (PPCC optimization)

---

### `capm.py`
Implements **CAPM-based factor modeling and hedging strategies**.

Key features:
- Beta and alpha estimation
- Correlation and R² analysis
- Multi-asset correlation screening
- Beta-neutral and delta-neutral hedging
- Regularized hedge weight optimization

---

### `portfolio_class.py`
Implements **portfolio construction and optimization** based on Modern Portfolio Theory.

Key features:
- Equally weighted and minimum variance portfolios
- Long-only and constrained optimization
- Covariance and correlation matrix estimation
- Expected return, volatility, and Sharpe Ratio

---

### `financial_quant.py`
Experimental and validation script used to test models and run quantitative experiments.

---

## Design Principles

- Modular and class-based architecture
- Clear separation between data, models, and optimization
- Statistically and financially grounded methods
- Focus on reproducibility and extensibility

---

 

#### **Supporting Modules**
Contains auxiliary files and helper functions used across different parts of the project.  
These modules enhance modularity and maintainability by handling specific subtasks, such as:
- Data formatting and transformation  
- Utility math functions  
- File handling or configuration management  

---


## Development Notes
- The repository is under **active development**.   