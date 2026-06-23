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
# 4. Top 15 Model-Based Features
# ---------------------------
model_importance = pd.Series(gb_model.feature_importances_, index=X.columns)
top15_model = model_importance.sort_values(ascending=False).head(15)

# ---------------------------
# 5. Top 15 SHAP-Based Features
# ---------------------------
explainer = shap.Explainer(gb_model, X_train)
shap_values = explainer(X_train)

mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns)
top15_shap = mean_abs_shap.sort_values(ascending=False).head(15)

print("\n🔝 SHAP Top 15 Features:")
print(top15_shap)
print("\n🔝 Model Top 15 Features:")
print(top15_model)

# ---------------------------
# 6. Union of SHAP + Model Features
# ---------------------------
union_features = list(set(top15_shap.index.tolist() + top15_model.index.tolist()))
print(f"\n Total Union Features: {len(union_features)}")
print(" Union Feature Names:")
print(union_features)

# ---------------------------
# 7. Retrain on Union Features
# ---------------------------
X_train_union = X_train[union_features]
X_test_union = X_test[union_features]

gb_union = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_union.fit(X_train_union, y_train)

# ---------------------------
# 8. Get Final Top 15 Features from Union
# ---------------------------
union_importance = pd.Series(gb_union.feature_importances_, index=union_features)
top15_union = union_importance.sort_values(ascending=False).head(15)

print("\n🔝 Final Top 15 Features from SHAP + Model Union:")
print(top15_union)

# ---------------------------
# 9. Retrain on Final Top 15 Features
# ---------------------------
X_train_top15 = X_train_union[top15_union.index]
X_test_top15 = X_test_union[top15_union.index]

gb_final = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_final.fit(X_train_top15, y_train)

# ---------------------------
# 10. Evaluate
# ---------------------------
y_pred = gb_final.predict(X_test_top15)
y_prob = gb_final.predict_proba(X_test_top15)[:, 1]

