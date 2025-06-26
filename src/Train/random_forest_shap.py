
import pandas as pd
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Load and clean data
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Full-feature Random Forest (better performance)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Sample 2000 rows for SHAP
shap_sample = X_train.sample(n=2000, random_state=42)

# SHAP with TreeExplainer (optimized for RF)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(shap_sample)[1]  # for class 1

# Rank by SHAP importance
shap_importance = pd.DataFrame({
    'feature': shap_sample.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)

top15_features = shap_importance.head(15)['feature'].tolist()

# ✅ Print top 15 features
print("\n🔝 Top 15 SHAP Features (Improved Sampling):")
print(shap_importance.head(15))

# Subset to top 15
X_train_top15 = X_train[top15_features]
X_test_top15 = X_test[top15_features]

# Train RF on top 15
rf_top15 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top15.fit(X_train_top15, y_train)

