
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Load the dataset
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')

# Drop the 'EmploymentPosition' column if it exists
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Separate features and target
X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# Train an initial XGBoost model to get feature importances
initial_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
initial_model.fit(X, y)

# Get feature importances and select top 15 features
importances = pd.Series(initial_model.feature_importances_, index=X.columns)
top15_features = importances.sort_values(ascending=False).head(15)

# Print top 15 features with their importance scores
print("\nTop 15 Features Ranked by Importance:")
for rank, (feature, importance) in enumerate(top15_features.items(), start=1):
    print(f"{rank}. {feature}: {importance:.4f}")

# Subset the data to top 15 features
X_top15 = X[top15_features.index]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_top15, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model on top 15 features
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

