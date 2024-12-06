from flask import Flask, jsonify, render_template, request
import pickle
import pandas as pd

# Load the trained model and preprocessing tools
MODEL_PATH = "models/best_model_XGBoost.pkl"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODERS_PATH = "models/label_encoders.pkl"

# Load pre-trained model
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Load scaler
with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load label encoders
with open(LABEL_ENCODERS_PATH, "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)


def feature_engineering(data):
    
    data['total_asset_value'] = (data['residential_assets_value'] +
                              data['commercial_assets_value'] +
                              data['luxury_assets_value'] +
                              data['bank_asset_value'])
    
    columns_to_drop = ['residential_assets_value', 'commercial_assets_value', 
                       'luxury_assets_value', 'bank_asset_value']
    data.drop(columns=columns_to_drop, inplace=True)

    return data


app = Flask(__name__)

# Feature definitions
numerical_features = [
    'income_annum', 'loan_amount', 'loan_term',
    'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value',
    'bank_asset_value', 'no_of_dependents'
]
categorical_features = ['education', 'self_employed']


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
        # Collect data from the form

        data = {
            "income_annum": float(request.form["income_annum"]),
            "loan_amount": float(request.form["loan_amount"]),
            "loan_term": float(request.form["loan_term"]),
            "cibil_score": float(request.form["cibil_score"]),
            "residential_assets_value": float(request.form["residential_assets_value"]),
            "commercial_assets_value": float(request.form["commercial_assets_value"]),
            "luxury_assets_value": float(request.form["luxury_assets_value"]),
            "bank_asset_value": float(request.form["bank_asset_value"]),
            "no_of_dependents": int(request.form["no_of_dependents"]),
            "education": request.form["education"],
            "self_employed": request.form["self_employed"],
        }

        # Prepare a DataFrame for input
        input_data = pd.DataFrame([data])

        # Feature engineering
        input_data = feature_engineering(input_data)

        # Encode categorical features
        for col in categorical_features:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # Scale numerical features
        numeric_columns = [
            "income_annum", "loan_amount", "loan_term", 
            "cibil_score", "total_asset_value"
        ]
        input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

        # order column
        desired_order = [
            "no_of_dependents", "education", "self_employed",
            "income_annum", "loan_amount", "loan_term",
            "cibil_score", "total_asset_value"
        ]

        # Reorder DataFrame columns
        input_data = input_data[desired_order]

        # Predict using the trained model
        prediction = model.predict(input_data)

        # Decode prediction
        prediction_label = label_encoders["loan_status"].inverse_transform(prediction)[0]

        # Return prediction result
        return render_template("home.html", predictions=prediction_label)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')

        response = {
            'status': 'success',
            'message': 'Form submitted successfully!',
            'data': {
                'name': name,
                'email': email,
                'message': message
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
