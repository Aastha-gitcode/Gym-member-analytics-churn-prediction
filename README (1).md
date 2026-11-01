
# Gym Member Analytics & Churn Prediction

## Project Overview
A machine learning and data analytics project that analyzes gym membership data, identifies cancellation patterns, and classifies existing member status using machine learning.

## Features
- Gym membership analysis
- Active vs inactive member visualization
- Cancellation reason analysis
- Membership plan distribution
- Member visit analysis
- Business insights and recommendations
- Machine learning classification using Logistic Regression and Random Forest
- Interactive Streamlit dashboard

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Models

| Model | Accuracy |
|---|---:|
| Logistic Regression | 55.00% |
| Random Forest | 56.67% |

Random Forest performed slightly better on the current test split.

## Project Files
- app.py – Streamlit dashboard
- gym_churn_model.pkl – Trained Random Forest model
- gym_member_analysis.csv – Processed dataset
- model_comparison.csv – Model evaluation results
- confusion_matrix.png – Model confusion matrix
- requirements.txt – Required Python libraries

## How to Run

Install dependencies:

```bash
pip install -r requirements.txtRun the dashboard:

```bash
streamlit run app.py
```

## Important Note
The current model classifies existing membership status. It does not forecast future cancellations. Model performance is limited and requires further improvement.

## Author
Aastha Bansal
