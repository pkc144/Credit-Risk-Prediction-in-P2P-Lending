
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Load data
df = pd.read_csv('/kaggle/input/final-balanced/final_balanced_dataset (1).csv')
df.drop(columns=['EmploymentPosition'], errors='ignore', inplace=True)
df.dropna(inplace=True)

X = pd.get_dummies(df.drop(columns=['Defaulted']))
y = df['Defaulted']

# Given feature lists
shap_features = [
    'MonthlyPayment', 'Country', 'LoanDuration', 'LanguageCode',
    'AmountOfPreviousLoansBeforeLoan', 'PreviousRepaymentsBeforeLoan',
    'Interest', 'Education', 'Amount', 'DebtToIncome', 'Age',
    'VerificationType', 'LiabilitiesTotal', 'HomeOwnershipType', 'FreeCash'
]

importance_features = [
    'Country', 'LoanDuration', 'LanguageCode', 'MonthlyPayment', 'Education',
    'VerificationType', 'PreviousRepaymentsBeforeLoan', 'IncomeFromSocialWelfare',
    'AmountOfPreviousLoansBeforeLoan', 'Interest', 'HomeOwnershipType',
    'ExistingLiabilities', 'DebtToIncome', 'NewCreditCustomer', 'Gender'
]

# Union of features
union_features = list(set(shap_features + importance_features))

# Rank top 15 from union based on XGBoost
model_rank = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=4)
model_rank.fit(X[union_features], y)

# Get top 15 by importance
importances = pd.Series(model_rank.feature_importances_, index=union_features)
top15_features = importances.sort_values(ascending=False).head(15).index.tolist()

# Prepare data with top 15
X_top15 = X[top15_features]
X_train, X_test, y_train, y_test = train_test_split(X_top15, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model_final = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=4)
model_final.fit(X_train, y_train)

# Evaluate
y_pred = model_final.predict(X_test)
y_prob = model_final.predict_proba(X_test)[:, 1]

