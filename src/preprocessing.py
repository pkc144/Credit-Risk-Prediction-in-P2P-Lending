import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path, save_path='final_cleaned_p2p.csv', target_col='Defaulted'):
    # Step 1: Load dataset
    df = pd.read_csv(file_path, low_memory=False)
    print(f" Dataset loaded with shape: {df.shape}")

    # Step 2: Drop leaky/ID columns
    columns_to_drop = ['UserName', 'LoanNumber', 'PrincipalBalance', 'Status']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(" Dropped columns:", columns_to_drop)

    # Step 3: Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        elif col != target_col:
            df[col] = df[col].fillna(df[col].mean())
    print(" Missing values handled")

    # Step 4: Encode categorical variables
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_enc.fit_transform(df[col].astype(str))
    print(" Categorical features encoded")

    # Step 5: Check/confirm binary target
    print(f"Target column `{target_col}` classes:", df[target_col].unique())

    # Step 6: Winsorize numerical features
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col != target_col:
            lower = df[col].quantile(0.001)
            upper = df[col].quantile(0.999)
            df[col] = df[col].clip(lower, upper)
    print("Winsorization applied")

    # Step 7: Remove high-correlation columns
    cor_matrix = df.corr().abs()
    upper_triangle = cor_matrix.where(
        pd.np.triu(pd.np.ones(cor_matrix.shape), k=1).astype(bool)
    )
    to_drop_corr = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)
    ]
    df.drop(columns=to_drop_corr, inplace=True)
    print(" Dropped high-correlation columns:", to_drop_corr)

    # Save cleaned file
    df.to_csv(save_path, index=False)
    print(f"Final cleaned dataset saved to: {save_path}")
    return df
