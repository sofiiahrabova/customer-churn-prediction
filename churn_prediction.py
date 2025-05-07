"""
Customer Churn Prediction – Logistic Regression (Improved)
Author: Sofiia Hrabova
Date: May 2025

Description:
This version improves prediction by scaling numeric features,
handling class imbalance, and tuning model parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Load dataset ---
df = pd.read_csv("telco_churn.csv")
df.columns = df.columns.str.strip()
df.drop("customerID", axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# --- Convert Churn column ---
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
churn_rate = df['Churn'].mean()
print(f"Overall churn rate: {churn_rate:.2%}")

# --- Convert Yes/No columns ---
yes_no_cols = df.select_dtypes(include='object').columns
for col in yes_no_cols:
    df[col] = df[col].replace({'Yes': 1, 'No': 0})

# --- One-hot encode remaining categoricals ---
df = pd.get_dummies(df, drop_first=True)

# --- Separate features and target ---
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- Scale numeric features ---
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train logistic regression model (improved) ---
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Create output folder if not present ---
if not os.path.exists("visuals"):
    os.makedirs("visuals")

# --- Confusion matrix plot ---
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("visuals/confusion_matrix.png")
plt.show()

# --- Feature importance ---
importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
importance.head(10).plot(kind='barh')
plt.title("Top 10 Features Influencing Churn")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("visuals/feature_importance.png")
plt.show()
