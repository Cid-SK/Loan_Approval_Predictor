import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(input_file, output_folder_data, output_folder_model):
    # Load the dataset
    df = pd.read_csv(input_file)

    # 1. Remove rows with negative residential assets value
    df = df[df['residential_assets_value'] >= 0]

    # 2. Create a new feature: total_asset_value
    df['total_asset_value'] = (df['residential_assets_value'] + 
                          df['commercial_assets_value'] + 
                          df['luxury_assets_value'] + 
                          df['bank_asset_value'])

    # 3. Drop unnecessary columns
    columns_to_drop = ['residential_assets_value', 'commercial_assets_value', 
                       'luxury_assets_value', 'bank_asset_value', 'loan_id']
    df.drop(columns=columns_to_drop, inplace=True)

    # 4. Label encode categorical columns
    label_encoders = {}
    categorical_columns = ['education', 'self_employed', 'loan_status']
    for col in categorical_columns:
        # Clean the data by stripping whitespace from categorical columns
        df[col] = df[col].str.strip()

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save label encoders for future use
    pd.to_pickle(label_encoders, os.path.join(output_folder_model, 'label_encoders.pkl'))

    # 5. Separate features and target variable
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # 6. Scale the numerical features using StandardScaler
    scaler = StandardScaler()
    numeric_columns = ['income_annum','loan_amount','loan_term','cibil_score','total_asset_value']
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # Save the scaler for future use
    pd.to_pickle(scaler, os.path.join(output_folder_model, 'scaler.pkl'))

    # 7. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the processed data
    df.to_csv(os.path.join(output_folder_data, 'preprocessed_data.csv'), index=False)
    X.to_csv(os.path.join(output_folder_data, 'X.csv'), index=False)
    y.to_csv(os.path.join(output_folder_data, 'y.csv'), index=False)
    X_train.to_csv(os.path.join(output_folder_data, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_folder_data, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_folder_data, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_folder_data, 'y_test.csv'), index=False)

    print(f"Data preprocessing complete. Files saved in '{output_folder_data}' and '{output_folder_model}' folders.")


input_file = "S:/DS/new_projects/loan_prediction/Loan_Approval_Predictor/data/loan_approval_dataset.csv"
output_folder_data = "S:/DS/new_projects/loan_prediction/Loan_Approval_Predictor/data"
output_folder_model = "S:/DS/new_projects/loan_prediction/Loan_Approval_Predictor/models"


preprocess_data(input_file, output_folder_data, output_folder_model)
