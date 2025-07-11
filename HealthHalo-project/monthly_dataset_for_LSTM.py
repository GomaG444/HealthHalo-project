import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for consistency
np.random.seed(42)

# Parameters
num_people = 5
readings_per_day = 48 # every 30 minutes
days = 30
total_readings = readings_per_day * days

# Generate timestamps
start_time = datetime(2025, 6, 1, 0, 0)
timestamps = [start_time + timedelta(minutes=30 * i) for i in range(total_readings)]

# Function to simulate a person’s data
def generate_person_data(person_id):
data = {
'person_id': [person_id] * total_readings,
'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
'age': np.random.randint(40, 70),
'sex': np.random.randint(0, 2),
'cp': np.random.randint(0, 4, total_readings),
'trestbps': np.random.randint(90, 180, total_readings),
'chol': np.random.randint(150, 350, total_readings),
'fbs': np.random.randint(0, 2, total_readings),
'restecg': np.random.randint(0, 2, total_readings),
'thalach': np.random.randint(100, 202, total_readings),
'exang': np.random.randint(0, 2, total_readings),
'oldpeak': np.round(np.random.uniform(0, 6.2, total_readings), 1),
'slope': np.random.randint(0, 3, total_readings),
'ca': np.random.randint(0, 4, total_readings),
'thal': np.random.choice([0, 1, 2, 3], total_readings),
}

# Calculate a fake risk score
risk_score = (
(np.array(data['age']) > 50).astype(int)
+ (np.array(data['chol']) > 240).astype(int)
+ (np.array(data['thalach']) < 140).astype(int)
+ (np.array(data['trestbps']) > 130).astype(int)
+ (np.array(data['oldpeak']) > 2.5).astype(int)
)
data['heart_disease'] = (risk_score >= 3).astype(int)
return pd.DataFrame(data)

# Combine data for 5 people
df = pd.concat([generate_person_data(i + 1) for i in range(num_people)], ignore_index=True)
