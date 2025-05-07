# 📊 Insights Summary – Customer Churn Prediction

This project analyzed a Telco customer dataset to predict churn using logistic regression. The goal was to identify patterns in customer behavior that lead to cancellation and help improve retention.

---

## 🧠 Key Insights

### 1. Churn Rate
- **Overall churn rate:** ~26%
- Indicates the dataset is imbalanced and requires model adjustments (handled via class weighting)

### 2. Top Predictors of Churn
- Longer tenure, lower monthly charges → less likely to churn
- Electronic check users and month-to-month contracts → more likely to churn
- Paperless billing and no internet security → moderate churn risk

### 3. Model Performance
- **Accuracy:** 78.7%
- **Precision (Churn):** 62%
- **Recall (Churn):** 52%
- Balanced performance for both classes using class weighting

---

## 📊 Visual Highlights

- **Confusion Matrix:** Showed a fair number of false positives, but good true negative rate
- **Feature Importance:** Tenure, MonthlyCharges, PaymentMethod_ElectronicCheck stood out

---

## 🔍 Recommendations

- Offer incentives for long-term contracts
- Improve support for electronic check users
- Promote bundled services to reduce churn risk

---

## ✅ Final Notes

The model performs well with simple logistic regression and thoughtful feature scaling. For further improvement, try tree-based models (e.g. Random Forest or XGBoost) or SMOTE to synthetically balance classes.

