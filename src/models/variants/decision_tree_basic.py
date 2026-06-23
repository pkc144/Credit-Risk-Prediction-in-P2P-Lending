
import pandas as pd
import numpy as np
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
# 4. Get Top 15 Features by Model Importance
# ---------------------------
feature_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
top15_features = feature_importance.sort_values(ascending=False).head(15)

print("\n🔝 Top 15 Decision Tree Features (Model Importance):")
print(top15_features)

# ---------------------------
# 5. Retrain on Top 15 Features
# ---------------------------
X_train_top15 = X_train[top15_features.index]
X_test_top15 = X_test[top15_features.index]

dt_top15 = DecisionTreeClassifier(random_state=42)
dt_top15.fit(X_train_top15, y_train)

# ---------------------------
# 6. Evaluate Model
# ---------------------------
y_pred = dt_top15.predict(X_test_top15)
y_prob = dt_top15.predict_proba(X_test_top15)[:, 1]

