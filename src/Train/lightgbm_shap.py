import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# Load dataset
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

# Top 15 features by importance
feature_importance = pd.Series(lgb_model.feature_importances_, index=X.columns)
top15_features = feature_importance.sort_values(ascending=False).head(15)
print("\n🔝 Top 15 LightGBM Features (Model Importance):")
print(top15_features)

# Retrain on top 15
X_train_top15 = X_train[top15_features.index]
X_test_top15 = X_test[top15_features.index]
lgb_top15 = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_top15.fit(X_train_top15, y_train)

y_pred = lgb_top15.predict(X_test_top15)
y_prob = lgb_top15.predict_proba(X_test_top15)[:, 1]

