# !pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Load and clean dataset
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# --------- Your Provided SHAP Features (Both Lists) ----------
features_1 = [
    'Country', 'LanguageCode', 'MonthlyPayment', 'LoanDuration',
    'Interest', 'Amount', 'Education', 'VerificationType',
    'DebtToIncome', 'Age', 'FreeCash', 'LiabilitiesTotal',
    'IncomeTotal', 'IncomeFromPrincipalEmployer', 'OccupationArea'
]

features_2 = features_1  # both lists were same in your input, so union is same

# --------- Take Union of Both Lists and Remove Duplicates ----------
union_features = list(set(features_1 + features_2))
print(f"\n✅ Total Union Features: {len(union_features)}")
print("📋 Union Feature Names:")
print(union_features)

# --------- Split Data on Union Features ----------
X_union = X[union_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_union, y, test_size=0.2, stratify=y, random_state=42
)

# --------- Train Random Forest on Union Features ----------
rf_union = RandomForestClassifier(n_estimators=100, random_state=42)
rf_union.fit(X_train, y_train)

# --------- Select Top 15 by Random Forest Importance ----------
importances = pd.Series(rf_union.feature_importances_, index=X_union.columns)
top15_features = importances.sort_values(ascending=False).head(15)
print("\n🔝 Top 15 Features by Random Forest Importance:")
print(top15_features)

# --------- Subset to Top 15 ----------
X_train_top15 = X_train[top15_features.index]
X_test_top15 = X_test[top15_features.index]

# --------- Train Again on Top 15 Features ----------
rf_top15 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top15.fit(X_train_top15, y_train)

# --------- Evaluate ----------
y_pred = rf_top15.predict(X_test_top15)
y_prob = rf_top15.predict_proba(X_test_top15)[:, 1]
