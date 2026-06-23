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
# 3. Train CatBoost Model on Full Data
# -------------------------
model_full = CatBoostClassifier(verbose=0, random_state=42)
model_full.fit(X_train, y_train)

# -------------------------
# 4. Get Top 15 SHAP Features
# -------------------------
explainer = shap.TreeExplainer(model_full)
shap_values = explainer.shap_values(X_train)[1]

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
top15_shap = mean_abs_shap.sort_values(ascending=False).head(15)

# -------------------------
# 5. Get Top 15 Model Feature Importances
# -------------------------
feature_importance = pd.Series(model_full.get_feature_importance(), index=X.columns)
top15_model = feature_importance.sort_values(ascending=False).head(15)

# -------------------------
# 6. Create Union of Top 15 SHAP and Model Features
# -------------------------
union_features = list(set(top15_shap.index.tolist() + top15_model.index.tolist()))
print(f"\nTotal Union Features: {len(union_features)}")
print(" Union Feature Names:")
print(union_features)

# -------------------------
# 7. Train CatBoost on Union Features
# -------------------------
X_train_union = X_train[union_features]
X_test_union = X_test[union_features]

model_union = CatBoostClassifier(verbose=0, random_state=42)
model_union.fit(X_train_union, y_train)

# -------------------------
# 8. Select Top 15 from Union by Model Importance
# -------------------------
union_importance = pd.Series(model_union.get_feature_importance(), index=union_features)
top15_union = union_importance.sort_values(ascending=False).head(15)

print("\n🔝 Top 15 Features by Model Importance from Union Set:")
print(top15_union)

# -------------------------
# 9. Retrain on Top 15 from Union
# -------------------------
X_train_top15_union = X_train[top15_union.index]
X_test_top15_union = X_test[top15_union.index]

model_top15_union = CatBoostClassifier(verbose=0, random_state=42)
model_top15_union.fit(X_train_top15_union, y_train)

# -------------------------
# 10. Evaluate
# -------------------------
y_pred = model_top15_union.predict(X_test_top15_union)
y_prob = model_top15_union.predict_proba(X_test_top15_union)[:, 1]

