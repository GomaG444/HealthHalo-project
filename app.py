# app.py (complete and final version)

import os
from datetime import datetime
import pandas as pd
import joblib
from dotenv import load_dotenv
from openai import OpenAI
import json

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import your modular ML prediction script
try:
    from ml_model import predict_csv
except ImportError:
    predict_csv = None 

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Setup directories and allowed extensions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SUMMARY_DIR = os.path.join(BASE_DIR, "summaries")
ALLOWED_EXT = {".csv"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# --- Helper functions ---
def latest_summary_fallback():
    """Return the latest saved data or a default placeholder."""
    path = os.path.join(SUMMARY_DIR, "latest.json")
    if os.path.exists(path):
        with open(path, "r") as fp:
            return json.load(fp)
    # Return a default empty state
    return {"summary": "No recent summary yet.", "risk_score": None, "predicted_class": None}

def load_ml_model():
    """Load the trained ML pipeline model."""
    try:
        model_path = os.path.join(BASE_DIR, "HealthHalo-project", "logistic_model.joblib")
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

# --- Initialize Flask app and resources ---
app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, "static"),
            template_folder=os.path.join(BASE_DIR, "templates"))

client = OpenAI(api_key=openai_api_key)
model = load_ml_model()

# --- Flask Routes ---

@app.route("/")
def dashboard():
    """Dashboard page with latest data injected for initial page load."""
    summary_data = latest_summary_fallback()
    return render_template("index.html", data=summary_data)

@app.route("/upload", methods=["GET", "POST"])
def handle_upload():
    """GET: show form · POST: ingest CSV, call OpenAI, store summary & predict risk."""
    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("file")
    if not file or not file.filename:
        return render_template("upload.html", message="No file selected.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("upload.html", message="Please upload a .csv file.")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = secure_filename(f"{timestamp}_{file.filename}")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)
        from utilities import df_quick_overview
        digest = df_quick_overview(df)
        
        if predict_csv:
            risk_counts = predict_csv(save_path)
        else:
            risk_counts = None
        
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a clinical data scientist. Turn raw CSV statistics into a concise, friendly heart-health summary for the patient. Avoid jargon."},
                {"role": "user", "content": digest},
            ],
        )
        summary_text = chat.choices[0].message.content.strip()

        # Calculate average risk score from CSV
        X = df.drop('target', axis=1) if 'target' in df.columns else df
        risk_probs = model.predict_proba(X)[:, 1]
        avg_risk_score = round(risk_probs.mean() * 100, 2)
        avg_pred_class = 1 if avg_risk_score > 50 else 0 # A simple threshold for high/low

        # --- Update: Save to the same JSON file as the chatbot ---
        data_to_save = {
            "summary": summary_text,
            "risk_score": avg_risk_score,
            "predicted_class": avg_pred_class
        }
        path = os.path.join(SUMMARY_DIR, "latest.json")
        with open(path, "w") as fp:
            json.dump(data_to_save, fp)
        # --- End Update ---

        return render_template(
            "upload.html",
            message="Upload successful! Summary generated.",
            summary=summary_text,
            prediction=risk_counts,
            avg_risk_score=avg_risk_score,
        )
    except Exception as exc:
        print(f"Error during upload processing: {exc}")
        return render_template("upload.html", message=f"An error occurred: {exc}")

@app.route("/chatbot")
def chatbot():
    """Chatbot page (passes key to browser JS)."""
    # Fix: Pass an empty string if the key is not found to prevent template rendering errors.
    key = os.getenv("OPENAI_API_KEY", "")
    return render_template("chatbot.html", openai_api_key=key)

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint returning risk score & class from a pre-loaded model.
    This version correctly handles feature name mapping and provides defaults."""
    if model is None:
        return jsonify({"error": "ML model not loaded."}), 500

    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid request: 'features' key is missing."}), 400

        user_features = data['features']
        
        # Define the expected features and their default values
        required_features = {
            'age': 55, 'sex': 1, 'cp': 0, 'trestbps': 120, 'chol': 200, 'fbs': 0,
            'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 1.0, 'slope': 2,
            'ca': 0, 'thal': 2
        }

        input_data = required_features.copy()

        if 'age' in user_features: input_data['age'] = user_features['age']
        if 'sex' in user_features: input_data['sex'] = user_features['sex']
        if 'cholesterol' in user_features: input_data['chol'] = user_features['cholesterol']
        if 'blood_pressure' in user_features: input_data['trestbps'] = user_features['blood_pressure']
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[list(required_features.keys())]

        prob = model.predict_proba(input_df)[0][1]
        pred_cls = int(model.predict(input_df)[0])

        return jsonify({"predicted_class": pred_cls, "risk_score": prob})
    except Exception as exc:
        print(f"Prediction API error: {exc}")
        return jsonify({"error": f"An error occurred during prediction: {exc}"}), 500

@app.route("/save_data", methods=["POST"])
def save_data():
    """Endpoint to receive and save both LLM summary and ML prediction from the chatbot."""
    try:
        data = request.get_json()
        summary_text = data.get("summary")
        risk_score = data.get("risk_score")
        predicted_class = data.get("predicted_class")

        if not all([summary_text, risk_score, predicted_class is not None]):
            return jsonify({"status": "error", "message": "Missing data"}), 400

        summary_data = {
            "summary": summary_text,
            "risk_score": risk_score,
            "predicted_class": predicted_class
        }
        
        path = os.path.join(SUMMARY_DIR, "latest.json")
        with open(path, "w") as fp:
            json.dump(summary_data, fp)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/get_latest_data")
def get_latest_data():
    """Returns the latest saved data for real-time dashboard updates."""
    data = latest_summary_fallback()
    return jsonify(data)


if __name__ == "__main__":
    print("App.py folder:", BASE_DIR)
    if model:
        print("✅ ML model loaded successfully.")
    else:
        print("❌ Failed to load ML model.")
    app.run(debug=True)