# app.py
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ────────────────────────────────
# 1.  ENV & DIRECTORIES
# ────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
SUMMARY_DIR  = os.path.join(BASE_DIR, "summaries")
ALLOWED_EXT  = {".csv"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ────────────────────────────────
# 2.  UTILITIES
# ────────────────────────────────
from utilities import df_quick_overview  # helper that summarises a DataFrame

client = OpenAI(api_key=openai_api_key)

def latest_summary_fallback() -> str:
    """Return the latest LLM summary or a default placeholder."""
    path = os.path.join(SUMMARY_DIR, "latest.txt")
    if os.path.exists(path):
        with open(path) as fp:
            return fp.read()
    return "No recent summary yet."

# ────────────────────────────────
# 3.  LOAD ML MODEL
# ────────────────────────────────
model_path = os.path.join(BASE_DIR, "HealthHalo-project", "logistic_model.joblib")
model = joblib.load(model_path)

# ────────────────────────────────
# 4.  FLASK APP
# ────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates")
)

# ────────────────────────────────
# 5.  ROUTES
# ────────────────────────────────
@app.route("/")
def dashboard():
    """Dashboard page with latest LLM summary injected."""
    return render_template("index.html", llm_summary=latest_summary_fallback())


@app.route("/upload", methods=["GET", "POST"])
def handle_upload():
    """GET: show form · POST: ingest CSV, call OpenAI, store summary."""
    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("file")
    if not file:
        return render_template("upload.html", message="No file selected.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("upload.html", message="Please upload a .csv file.")

    # 1. Save upload with timestamp prefix to avoid collisions
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = secure_filename(f"{timestamp}_{file.filename}")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)

    # 2. Pandas summary
    df      = pd.read_csv(save_path)
    digest  = df_quick_overview(df)

    # 3. Ask OpenAI for friendly analysis
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical data scientist. "
                    "Turn raw CSV statistics into a concise, friendly heart-health summary "
                    "for the patient. Avoid jargon."
                ),
            },
            {"role": "user", "content": digest},
        ],
    )
    summary_text = chat.choices[0].message.content.strip()

    # 4. Persist summary for dashboard
    with open(os.path.join(SUMMARY_DIR, "latest.txt"), "w") as fp:
        fp.write(summary_text)

    return render_template("upload.html", message="Upload successful! Summary generated.")


@app.route("/chatbot")
def chatbot():
    """Chatbot page (passes key to browser JS)."""
    return render_template("chatbot.html", openai_api_key=openai_api_key)


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint returning risk score & class from logistic model."""
    try:
        features = request.get_json()["features"]
        input_df = pd.DataFrame([features])
        prob     = model.predict_proba(input_df)[0][1]
        pred_cls = int(model.predict(input_df)[0])
        return jsonify({"predicted_class": pred_cls, "risk_score": prob})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ────────────────────────────────
# 6.  MAIN
# ────────────────────────────────
if __name__ == "__main__":
    print("App.py folder:", BASE_DIR)
    app.run(debug=True)
