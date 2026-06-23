
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Load dataset
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

# Prepare features and target
X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

print(f"✅ Total features used: {X.shape[1]}")
print("📋 Feature names:", list(X.columns))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Random Forest on all features
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)

# 🔝 Rank top 15 features
importances = pd.Series(rf_full.feature_importances_, index=X.columns)
top15_features = importances.sort_values(ascending=False).head(15)
print("\n🔝 Top 15 Features by Importance (Random Forest):")
print(top15_features)

# Train again using only top 15 features
X_train_top15 = X_train[top15_features.index]
X_test_top15 = X_test[top15_features.index]

rf_top15 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top15.fit(X_train_top15, y_train)

# Predict
y_pred = rf_top15.predict(X_test_top15)
y_prob = rf_top15.predict_proba(X_test_top15)[:, 1]

