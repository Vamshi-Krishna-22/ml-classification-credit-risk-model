# ml-classification-credit-risk-model
# 📊 VK Finance: Credit Risk Modeling Web App

A production-ready machine learning project simulating how a fintech company like **VK Finance** would assess the credit risk of loan applicants using advanced analytics, machine learning, and real-time scoring.

---

## 🚀 Project Summary

This end-to-end ML solution takes you through the full credit modeling lifecycle:

1. **Data Collection & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering & Domain-Based Transformations**
4. **Feature Selection using VIF & IV/WOE**
5. **Encoding, Scaling & Data Preparation**
6. **Model Training & Hyperparameter Tuning (Optuna, RandomSearchCV)**
7. **Class Imbalance Handling (SMOTETomek, Undersampling)**
8. **Evaluation using ROC, AUC, KS, Gini**
9. **Model Interpretation & Feature Importance Visualization**
10. **Deployment using Streamlit Web App**

---

## 🧠 Problem Statement

Given applicant and loan-related data, predict the **probability of default** and return a **custom credit score** (300–900) along with a **rating category** (Poor → Excellent).

---

## 🧰 Tools & Technologies

| Area                | Stack Used                            |
|---------------------|----------------------------------------|
| Language            | Python                                 |
| ML Models           | Logistic Regression, XGBoost, Random Forest |
| Feature Selection   | VIF, WOE/IV                            |
| Scaling             | MinMaxScaler                           |
| Class Imbalance     | SMOTETomek, Undersampling              |
| UI/Deployment       | Streamlit                              |
| Storage             | Joblib                                 |

---

## 📂 Project Structure

```plaintext
├── main.py                      # Streamlit frontend
├── prediction_helper.py         # Backend logic for prediction, scaling, scoring
├── artifacts/
│   └── model_data.joblib        # Contains trained model, scaler, features
├── notebooks/                   # EDA, Feature Selection, Model Tuning
│   └── credit_risk_pipeline.ipynb
├── requirements.txt
└── README.md
