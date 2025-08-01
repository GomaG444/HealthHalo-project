import os
import pandas as pd
import joblib

def predict_csv(file_path):
    # Get this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Full path to model.pkl inside HealthHalo-project folder
    model_path = os.path.join(script_dir, 'model.pkl')

    model = joblib.load(model_path)  # Load model with full path

    df = pd.read_csv(file_path)

    if 'target' in df.columns:
        df = df.drop('target', axis=1)

    X = df
    predictions = model.predict(X)

    risk_counts = {
        'low_risk': int((predictions == 0).sum()),
        'high_risk': int((predictions == 1).sum())
    }

    return risk_counts
