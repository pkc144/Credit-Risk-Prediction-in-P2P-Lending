# !pip install catboost shap scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# -------------------------
# 1. Load and Clean Dataset
# -------------------------
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# -------------------------
# 2. Split Data
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -------------------------
# 3. Train Initial CatBoost Model
# -------------------------
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 4. SHAP Feature Importance
# -------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)[1]  # for class 1

# Calculate mean SHAP importance
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
top15_shap = mean_abs_shap.sort_values(ascending=False).head(15)

print("\n🔝 Top 15 SHAP Features (CatBoost):")
print(top15_shap)

# -------------------------
# 5. Retrain on Top 15 SHAP Features
# -------------------------
X_train_top15 = X_train[top15_shap.index]
X_test_top15 = X_test[top15_shap.index]

model_shap = CatBoostClassifier(verbose=0, random_state=42)
model_shap.fit(X_train_top15, y_train)

# -------------------------
# 6. Evaluation
# -------------------------
y_pred = model_shap.predict(X_test_top15)
y_prob = model_shap.predict_proba(X_test_top15)[:, 1]

