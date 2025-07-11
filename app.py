

# app.py
import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Print the folder path for debugging
print("App.py folder:", os.path.dirname(os.path.abspath(__file__)))

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML Model
model_path = os.path.join(BASE_DIR, 'HealthHalo-project', 'logistic_model.joblib')
model = joblib.load(model_path)

# Initialize Flask App
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates')
)

# Routes
@app.route('/')
def dashboard():
    return render_template('index.html')  # Dashboard page

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', openai_api_key=openai_api_key)  # Chatbot page with API key

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint for ML Prediction"""
    data = request.get_json()
    try:
        features = data['features']
        input_df = pd.DataFrame([features])
        prob = model.predict_proba(input_df)[0][1]  # Risk Score
        pred_class = int(model.predict(input_df)[0])  # Class (e.g., 0 or 1)
        response = {
            'predicted_class': pred_class,
            'risk_score': prob
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

    