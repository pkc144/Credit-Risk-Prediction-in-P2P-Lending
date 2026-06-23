

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load the dataset
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')

# 2. Clean the data
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

# 3. Encode categorical features & separate target
X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# 4. Use XGBoost to get top 15 important features
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X, y)

importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
top15_features = importances.sort_values(ascending=False).head(15)

print("\nTop 15 Features Selected by XGBoost:")
for rank, (feature, score) in enumerate(top15_features.items(), start=1):
    print(f"{rank}. {feature}: {score:.4f}")

# 5. Use only top 15 features
X_top15 = X[top15_features.index]

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_top15, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# 8. Predictions
y_pred = gb_model.predict(X_test)
y_prob = gb_model.predict_proba(X_test)[:, 1]

