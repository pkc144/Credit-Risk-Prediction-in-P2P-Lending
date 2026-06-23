# !pip install shap scikit-learn pandas matplotlib seaborn

import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------------------------
# 1. Load & Clean Dataset
# ---------------------------
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# ---------------------------
# 2. Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. Train Full Gradient Boosting Model
# ---------------------------
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# ---------------------------
# 4. SHAP Feature Importance
# ---------------------------
explainer = shap.Explainer(gb_model, X_train)
shap_values = explainer(X_train)

mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns)
top15_shap = mean_abs_shap.sort_values(ascending=False).head(15)

print("\n🔝 Top 15 SHAP Features (Gradient Boosting):")
print(top15_shap)

# ---------------------------
# 5. Retrain on SHAP Top 15 Features
# ---------------------------
X_train_top15 = X_train[top15_shap.index]
X_test_top15 = X_test[top15_shap.index]

gb_shap15 = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_shap15.fit(X_train_top15, y_train)

# ---------------------------
# 6. Evaluate
# ---------------------------
y_pred = gb_shap15.predict(X_test_top15)
y_prob = gb_shap15.predict_proba(X_test_top15)[:, 1]

