import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib

def evaluate_model(input_folder_data, model_path):
    # Load preprocessed test data
    X_test = pd.read_csv(os.path.join(input_folder_data, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(input_folder_data, 'y_test.csv')).values.ravel()

    # Load the saved best model
    best_model = joblib.load(model_path)

    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    # Evaluate metrics
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC Score: {roc_auc}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()
    else:
        print("Model does not support probability predictions.")

# Path to the best model
model_path = "S:/DS/new_projects/loan_prediction/Loan_Approval_Predictor/models/best_model_XGBoost.pkl"
input_folder_data = "S:/DS/new_projects/loan_prediction/Loan_Approval_Predictor/data"

evaluate_model(input_folder_data, model_path)
