
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Load the dataset
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

# Prepare features and target
X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# Train base XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=4)
model.fit(X, y)

# Use TreeExplainer to get SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values(by='shap_importance', ascending=False)

# Select top 15 features
top15_features = shap_importance.head(15)['feature'].tolist()

# Subset data
X_top15 = X[top15_features]
X_train, X_test, y_train, y_test = train_test_split(X_top15, y, test_size=0.2, stratify=y, random_state=42)

# Retrain on top 15
model_top15 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=4)
model_top15.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_top15.predict(X_test)
y_prob = model_top15.predict_proba(X_test)[:, 1]


