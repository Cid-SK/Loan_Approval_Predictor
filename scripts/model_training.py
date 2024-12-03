import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving models


def train_and_select_best_model(input_folder_data, output_folder_model):
    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(input_folder_data, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_folder_data, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(input_folder_data, 'y_train.csv')).values.ravel()  # Flatten y_train
    y_test = pd.read_csv(os.path.join(input_folder_data, 'y_test.csv')).values.ravel()  # Flatten y_test

    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier()
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy for {model_name}: {accuracy}")
        print(f"Classification Report for {model_name}:\n", classification_report(y_test, y_pred))
        print(f"Confusion Matrix for {model_name}:\n", confusion_matrix(y_test, y_pred))

        # Check if this model is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    # Save the best model 
    if best_model:
        best_model_filename = os.path.join(
            output_folder_model, f'best_model_{best_model_name.replace(" ", "_")}.pkl'
        )
        joblib.dump(best_model, best_model_filename)
        print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")
        print(f"Best model saved as '{best_model_filename}'")


# Set input and output folder paths
input_folder_data = "S:/DS/new_projects/loan_prediction/Loan_Approval_Predictor/data"
output_folder_model = "S:/DS/new_projects/loan_prediction/Loan_Approval_Predictor/models"


# Train and select the best model
train_and_select_best_model(input_folder_data, output_folder_model)
