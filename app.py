import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Print the path of the current file (for debugging)
print("App.py folder:", os.path.dirname(os.path.abspath(__file__)))

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the saved model file (same folder as app.py)
model_path = os.path.join(BASE_DIR, 'HealthHalo-project', 'logistic_model.joblib')


# Path to templates and static folders (you'll adjust these if needed)
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates')
)

# Load model
model = joblib.load(model_path)

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = data['features']
        input_df = pd.DataFrame([features])
        prob = model.predict_proba(input_df)[0][1]
        pred_class = int(model.predict(input_df)[0])
        response = {
            'predicted_class': pred_class,
            'risk_score': prob
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
