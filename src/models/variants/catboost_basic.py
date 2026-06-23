from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Train CatBoost
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train, y_train)

# Get top 15 features by CatBoost importance
feature_importances = pd.Series(model.get_feature_importance(), index=X.columns)
top15_features = feature_importances.sort_values(ascending=False).head(15)

print(" Top 15 Features by CatBoost Importance:")
print(top15_features)

# Train again on top 15
X_train_top15 = X_train[top15_features.index]
X_test_top15 = X_test[top15_features.index]
model_top15 = CatBoostClassifier(verbose=0, random_state=42)
model_top15.fit(X_train_top15, y_train)

