# !pip install lightgbm shap scikit-learn pandas matplotlib seaborn

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt
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
# 3. Train Full LightGBM Model
# ---------------------------
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

# ---------------------------
# 4. Get SHAP Top 15 Features
# ---------------------------
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_train)[1]  # for class 1

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
top15_shap = mean_abs_shap.sort_values(ascending=False).head(15)

print("\n🔝 Top 15 SHAP Features (LightGBM):")
print(top15_shap)

# ---------------------------
# 5. Get LightGBM Top 15 Model Features
# ---------------------------
feature_importance = pd.Series(lgb_model.feature_importances_, index=X.columns)
top15_model = feature_importance.sort_values(ascending=False).head(15)

print("\n🔝 Top 15 Model Features (LightGBM Importance):")
print(top15_model)

# ---------------------------
# 6. Union of SHAP + Model Features
# ---------------------------
union_features = list(set(top15_shap.index.tolist() + top15_model.index.tolist()))
print(f"\n✅ Total Union Features: {len(union_features)}")
print("📋 Union Feature Names:")
print(union_features)

# ---------------------------
# 7. Retrain LightGBM on Union Features
# ---------------------------
X_train_union = X_train[union_features]
X_test_union = X_test[union_features]

model_union = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model_union.fit(X_train_union, y_train)

# ---------------------------
# 8. Select Top 15 from Union via Model Importance
# ---------------------------
union_importance = pd.Series(model_union.feature_importances_, index=union_features)
top15_union = union_importance.sort_values(ascending=False).head(15)

print("\n🔝 Top 15 Features from SHAP + Model Union (Ranked by Model):")
print(top15_union)

# ---------------------------
# 9. Retrain Final Model on Union Top 15
# ---------------------------
X_train_top15 = X_train_union[top15_union.index]
X_test_top15 = X_test_union[top15_union.index]

final_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train_top15, y_train)

# ---------------------------
# 10. Evaluate Final Model
# ---------------------------
y_pred = final_model.predict(X_test_top15)
y_prob = final_model.predict_proba(X_test_top15)[:, 1]




