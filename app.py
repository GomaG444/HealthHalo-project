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
            data = json.load(fp)
            if "lstm_summary" not in data:
                data["lstm_summary"] = "No recent health trends available."
            return data
    return {
        "summary": "No recent summary yet.",
        "risk_score": None,
        "predicted_class": None,
        "lstm_summary": "No recent health trends available."
    }

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
    summary_data = latest_summary_fallback()
    return render_template("index.html", data=summary_data)

@app.route("/upload", methods=["GET", "POST"])
def handle_upload():
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

        # --- ML risk counts (if predict_csv exists) ---
        if predict_csv:
            risk_counts = predict_csv(save_path)
        else:
            risk_counts = None

        # --- LLM summary ---
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical data scientist. Summarize the patient CSV data "
                        "into a short, friendly heart-health report suitable for a patient. "
                        "Focus only on important trends and anomalies. "
                        "Avoid repeating information and keep it concise (3-4 sentences max). "
                        "Do not mention exact ages or ranges unless critical. "
                        "Avoid jargon and unnecessary statistics."
                    )
                },
                {"role": "user", "content": digest},
            ],
        )
        summary_text = chat.choices[0].message.content.strip()

        # --- Map CSV for logistic regression ---
        df_mapped = pd.DataFrame()
        df_mapped['age'] = df['age']
        df_mapped['sex'] = df['sex'].map({'M': 1, 'F': 0})
        df_mapped['cp'] = 0
        df_mapped['trestbps'] = df['systolic']
        df_mapped['chol'] = df['cholesterol']
        df_mapped['fbs'] = 0
        df_mapped['restecg'] = 0
        df_mapped['thalach'] = df['heart_rate']
        df_mapped['exang'] = 0
        df_mapped['oldpeak'] = 0.0
        df_mapped['slope'] = 2
        df_mapped['ca'] = 0
        df_mapped['thal'] = 2

        df_mapped = df_mapped[['age','sex','cp','trestbps','chol','fbs','restecg',
                               'thalach','exang','oldpeak','slope','ca','thal']]

        # --- ML predictions ---
        risk_probs = model.predict_proba(df_mapped)[:, 1]
        avg_risk_score = round(risk_probs.mean() * 100, 2)
        avg_pred_class = 1 if avg_risk_score > 50 else 0

        # --- Save summary and risk to JSON ---
        data_to_save = {
            "summary": summary_text,
            "risk_score": avg_risk_score,
            "predicted_class": avg_pred_class,
            "lstm_summary": "Stable heart rate and blood pressure trends."
        }
        path = os.path.join(SUMMARY_DIR, "latest.json")
        with open(path, "w") as fp:
            json.dump(data_to_save, fp)

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
    key = os.getenv("OPENAI_API_KEY", "")
    return render_template("chatbot.html", openai_api_key=key)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        summary_data = latest_summary_fallback()
        previous_risk = summary_data.get("risk_score")
        predicted_class = summary_data.get("predicted_class")
        summary_text = summary_data.get("summary")
        previous_lstm_summary = summary_data.get("lstm_summary")

        friendly_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        user_lower = user_message.lower()
        if any(greet in user_lower for greet in friendly_greetings):
            ai_reply = "Hello! How are you feeling today? Any heart-related updates like BP, heart rate, or missed medication?"
            risk_score = previous_risk
            lstm_summary = previous_lstm_summary
        else:
            system_prompt = (
                "You are a helpful heart health assistant. "
                "The patient may provide recent health data such as blood pressure, heart rate, "
                "medication adherence, fatigue, or other symptoms. "
                "Based on this information, estimate the patient's readmission risk (0-100%) and summarize any trends in heart health. "
                "Always return a JSON object in this format: "
                "{\"reply\": \"friendly message to user\", \"risk_score\": 82, \"lstm_summary\": \"summary of heart trends\"}. "
                "If the user does not provide numeric data, base estimates on previous saved data. "
                "If the user asks unrelated questions, politely redirect them to heart health."
            )
            if previous_risk is not None:
                system_prompt += f" The patient's last known readmission risk was {previous_risk:.2f}%."

            chat_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            ai_raw = chat_response.choices[0].message.content.strip()

            try:
                ai_json = json.loads(ai_raw)
                ai_reply = ai_json.get("reply", ai_raw)
                risk_score = ai_json.get("risk_score", previous_risk)
                lstm_summary = ai_json.get("lstm_summary", previous_lstm_summary)
            except json.JSONDecodeError:
                ai_reply = ai_raw
                risk_score = previous_risk
                lstm_summary = previous_lstm_summary

        return jsonify({
            "reply": ai_reply,
            "summary": summary_text,
            "risk_score": risk_score,
            "predicted_class": predicted_class,
            "lstm_summary": lstm_summary
        })

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "ML model not loaded."}), 500

    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid request: 'features' key is missing."}), 400

        user_features = data['features']
        
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
    try:
        data = request.get_json()
        summary_text = data.get("summary")
        risk_score = data.get("risk_score")
        predicted_class = data.get("predicted_class")
        lstm_summary = data.get("lstm_summary", "Stable heart rate and blood pressure trends.")

        if summary_text is None or risk_score is None or predicted_class is None:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        summary_data = {
            "summary": summary_text,
            "risk_score": risk_score,
            "predicted_class": predicted_class,
            "lstm_summary": lstm_summary
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
    data = latest_summary_fallback()
    return jsonify(data)

@app.route("/test_post", methods=["POST"])
def test_post():
    data = request.get_json()
    print("Test POST received data:", data)
    return jsonify({"status": "success", "data": data}), 200

# --- ADD THIS: Reports route ---
@app.route("/reports")
def reports():
    # Get the latest saved summary
    summary = latest_summary_fallback()
    
    # Map keys to what the template expects
    report = {
        "risk_trend": f"{summary.get('risk_score', 'N/A')}%",
        "hr_flags": summary.get("predicted_class", "No anomalies"),
        "recent_symptoms": summary.get("summary", "No recent symptoms")
    }
    
    return render_template("reports.html", report=report)


if __name__ == "__main__":
    print("App.py folder:", BASE_DIR)
    if model:
        print("✅ ML model loaded successfully.")
    else:
        print("❌ Failed to load ML model.")
    app.run(debug=True, port=5050)
