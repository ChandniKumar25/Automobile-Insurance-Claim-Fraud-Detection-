# 🚗 Insurance Claim Fraud Detection System

An Explainable AI-powered Insurance Claim Fraud Detection System built using **XGBoost**, **SHAP Explainability**, and **Streamlit**.

This project predicts whether an automobile insurance claim is **fraudulent or genuine** while providing transparent explanations for each prediction using SHAP (SHapley Additive exPlanations).

---

## Project Overview

Insurance fraud causes significant financial losses for insurance companies every year. This project applies machine learning techniques to automatically identify potentially fraudulent claims and explains the reasoning behind every prediction.

The application combines high predictive performance with Explainable AI to make model decisions transparent and easier to interpret.

---

## Features

- Insurance claim fraud prediction
- XGBoost machine learning model
- Explainable AI using SHAP
- Fraud probability estimation
- Genuine claim probability estimation
- Model confidence score
- SHAP Waterfall Plot
- Top feature contribution analysis
- Plain-English explanation of predictions
- Modern Streamlit dashboard

---

## Machine Learning Pipeline

- Data preprocessing
- Label Encoding
- Train-Test Split
- SMOTE for class balancing
- Feature Selection
- XGBoost Classifier
- Model Evaluation
- SHAP Explainability
- Streamlit Deployment

---

## Technologies Used

- Python
- Streamlit
- XGBoost
- SHAP
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

---

## Dataset

Dataset:
Automobile Insurance Claim Fraud Dataset

File used:

```
carclaims.csv
```

---

## Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **95.07%** |
| ROC-AUC | **0.936** |

The model demonstrates strong predictive performance while maintaining interpretability through Explainable AI.

---

## Project Structure

```
Insurance-Fraud-Detection/
│
├── app.py
├── best_xgb.pkl
├── label_encoders.pkl
├── carclaims.csv
├── requirements.txt
├── README.md
├── notebook/
│   └── Insurance_Fraud_Detection.ipynb
├── images/
└── .streamlit/
    └── config.toml
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Insurance-Fraud-Detection.git
```

Move into the project directory

```bash
cd Insurance-Fraud-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Explainable AI

The application integrates SHAP (SHapley Additive Explanations) to improve transparency.

For every prediction, the application provides:

- SHAP Waterfall Plot
- Feature contribution ranking
- Natural language explanation of the prediction

---

## Future Enhancements

- Deep Learning-based fraud detection
- Real-time fraud detection API
- Cloud deployment
- User authentication
- Database integration
- Claims management dashboard

---

## Author

Chandni Kumar

