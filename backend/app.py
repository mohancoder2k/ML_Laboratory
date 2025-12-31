import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from services import DigitRecognitionService

app = Flask(__name__)
CORS(app)   # Allow frontend requests

# ------------------ PATH SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOAN_MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")
SALARY_MODEL_PATH = os.path.join(BASE_DIR, "model", "salary_model.pkl")

# ------------------ LOAD MODELS ------------------
loan_model = joblib.load(LOAN_MODEL_PATH)
salary_model = joblib.load(SALARY_MODEL_PATH)

# ------------------ DIGIT SETUP ------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

digit_service = DigitRecognitionService()

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return "MLLAB API is running"

# ---- Digit Recognition ----
@app.route("/api/predict", methods=["POST"])
def predict_digit():
    if "digit" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["digit"]
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    prediction = digit_service.predict_digit(img_path)

    return jsonify({
        "predicted_digit": prediction
    })


# ---- Loan Prediction ----
@app.route("/predict", methods=["POST"])
def predict_loan():
    data = request.get_json()
    income = float(data["ApplicantIncome"])

    prediction = loan_model.predict(np.array([[income]]))

    return jsonify({
        "PredictedLoanAmount": round(prediction[0], 2)
    })


# ---- Salary Prediction ----
@app.route("/predict/salary", methods=["POST"])
def predict_salary():
    data = request.get_json()
    experience = float(data["YearsExperience"])

    prediction = salary_model.predict(np.array([[experience]]))

    return jsonify({
        "PredictedSalary": round(prediction[0], 2)
    })


# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    app.run(debug=True)
