# 📊 Customer Churn Prediction – Logistic Regression (Python)

This project predicts customer churn using the Telco dataset.  
It includes data cleaning, feature engineering, logistic regression modeling, and model evaluation with visual outputs.

---

## 🧠 Problem Statement

Telecom companies struggle with customer retention. This project builds a logistic regression model to predict whether a customer will churn based on account and service features.

---

## 🔧 Tools Used

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

---

## 🚀 Key Steps

- Cleaned and preprocessed 20+ features
- Converted categorical values and handled missing data
- Scaled numeric variables (`MonthlyCharges`, `TotalCharges`, `tenure`)
- Handled class imbalance using `class_weight='balanced'`
- Trained and evaluated a logistic regression model

---

## 📈 Model Evaluation

- **Accuracy:** ~78.7%
- **Precision (Churn):** 62%
- **Recall (Churn):** 52%
- **F1-Score (Churn):** 56%

---

## 📷 Visualizations

### Confusion Matrix  
![Confusion Matrix](visuals/improved_confusion_matrix.png)

### Feature Importance  
![Feature Importance](visuals/improved_feature_importance.png)

---

## 📁 Project Structure

customer-churn-prediction/
└── visuals/
├── confusion_matrix.png
└── feature_importance.png
├── churn_prediction.py
├── README.md
├── insights-summary.md
├── telco_churn.csv

---

## 📬 Author

**Sofiia Hrabova**  
[LinkedIn →](https://www.linkedin.com/in/sofiia-hrabova-1380a7338)