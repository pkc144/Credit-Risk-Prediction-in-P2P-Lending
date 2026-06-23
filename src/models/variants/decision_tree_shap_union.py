# !pip install shap scikit-learn pandas matplotlib seaborn

import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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
# 3. Train Full Decision Tree
# ---------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# ---------------------------
# 4. Get Model-Based Top 15 Features
# ---------------------------
model_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
top15_model = model_importance.sort_values(ascending=False).head(15)

# ---------------------------
# 5. Get SHAP-Based Top 15 Features
# ---------------------------
explainer = shap.TreeExplainer(dt_model)
shap_values = explainer.shap_values(X_train)[1]  # class 1

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
top15_shap = mean_abs_shap.sort_values(ascending=False).head(15)

# ---------------------------
# 6. Union of Both Sets
# ---------------------------
union_features = list(set(top15_model.index.tolist() + top15_shap.index.tolist()))
print(f"\n✅ Total Union Features: {len(union_features)}")
print("📋 Union Feature Names:")
print(union_features)

# ---------------------------
# 7. Retrain on Union, Then Select Top 15 Again
# ---------------------------
X_train_union = X_train[union_features]
X_test_union = X_test[union_features]

dt_union = DecisionTreeClassifier(random_state=42)
dt_union.fit(X_train_union, y_train)

# Select top 15 from union via model importance
union_importance = pd.Series(dt_union.feature_importances_, index=union_features)
top15_union = union_importance.sort_values(ascending=False).head(15)

print("\n🔝 Final Top 15 Features from Union (Ranked by Model Importance):")
print(top15_union)

# ---------------------------
# 8. Retrain on Final Top 15 Union Features
# ---------------------------
X_train_top15 = X_train_union[top15_union.index]
X_test_top15 = X_test_union[top15_union.index]

dt_final = DecisionTreeClassifier(random_state=42)
dt_final.fit(X_train_top15, y_train)

# ---------------------------
# 9. Evaluation
# ---------------------------
y_pred = dt_final.predict(X_test_top15)
y_prob = dt_final.predict_proba(X_test_top15)[:, 1]

