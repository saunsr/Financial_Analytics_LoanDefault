# Loan Default Prediction — Financial Analytics

## Project Overview
This project demonstrates an **end-to-end machine learning workflow** for predicting loan default risk. Using a public Kaggle dataset, an entire process — from loading raw data to delivering model performance reports is shown, that can be used by stakeholders to guide decision-making.

---

## Objective
Given a dataset of customer employment status, bank balances, and salaries, the goal was to **predict whether a customer will default on a loan**.  
The business value is to **prioritize high-risk customers for further review or intervention**, reducing financial losses while maintaining efficiency.

---

## Project Structure
```
├── config/
│ └── config.yaml # File paths and settings
│
├── data/
│ ├── raw/ # Raw dataset
│ └── processed/ # Cleaned & engineered datasets
│
├── notebooks/
│ ├── 01_load_EDA_data.ipynb # Load & explore data (EDA)
│ ├── 02_feature_engineering.ipynb# Create new features
│ ├── 03_modelling_baseline.ipynb # Train baseline models
│ └── 04_summary_report.ipynb # Summary and insights
│
├── reports/
│ ├── baseline_summary.csv # Model performance metrics
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Load config and datasets
│ └── preprocess.py # Cleaning & feature prep
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Process Workflow
1. **Data Loading**  
   Load processed data via a config-driven `data_loader.py` to keep paths flexible.

2. **Exploratory Data Analysis (EDA)**  
   - Class balance check (default vs non-default)
   - Missing value check
   - Distribution plots for numerical features
   - Feature vs target visualizations
   - Correlation analysis

3. **Feature Engineering**  
   Derived features:
   - `log_bank_balance`
   - `log_annual_salary`
   - `balance_to_salary`
   - `employed_x_log_salary`

4. **Modelling (Baseline)**  
   - Logistic Regression
   - Random Forest Classifier  
   Metrics: ROC-AUC, PR-AUC, top-K capture rate.

5. **Evaluation & Insights**  
   - Logistic Regression: ROC-AUC 0.95, PR-AUC 0.46  
   - Random Forest: ROC-AUC 0.89, PR-AUC 0.39  
   - Logistic Regression performed better for rare-event capture.

6. **Reporting**  
   - Metrics saved to `reports/baseline_summary.csv`
   - Summary plots and key insights documented in `04_summary_report.ipynb`
