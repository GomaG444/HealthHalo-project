import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model and prepare scaler
model = load_model("your_model.h5") # Save your trained model as this
scaler = MinMaxScaler()

# Load your existing dataset to fit the scaler
df = pd.read_csv("healthhalo_monthly_data_multiple_users.csv")
features = df.drop(columns=['person_id', 'timestamp', 'heart_disease'])
scaler.fit(features)

# Function to simulate incoming data (every 5 seconds)
def simulate_data_stream(seq_length=10):
buffer = []
for _ in range(100): # simulate 100 time steps
# Generate random new reading based on realistic bounds
new_data = {
'age': np.random.randint(40, 70),
'sex': np.random.randint(0, 2),
'cp': np.random.randint(0, 4),
'trestbps': np.random.randint(90, 180),
'chol': np.random.randint(150, 350),
'fbs': np.random.randint(0, 2),
'restecg': np.random.randint(0, 2),
'thalach': np.random.randint(100, 202),
'exang': np.random.randint(0, 2),
'oldpeak': np.round(np.random.uniform(0, 6.2), 1),
'slope': np.random.randint(0, 3),
'ca': np.random.randint(0, 4),
'thal': np.random.choice([0, 1, 2, 3])
}

df_new = pd.DataFrame([new_data])
df_scaled = scaler.transform(df_new)
buffer.append(df_scaled[0])

# Once buffer has enough data
if len(buffer) >= seq_length:
input_seq = np.array(buffer[-seq_length:]).reshape(1, seq_length, -1)
prediction = model.predict(input_seq)[0][0]

if prediction > 0.5:
print(f"🚨 ALERT: High risk of heart disease detected! (Risk: {prediction:.2f})")
else:
print(f"✅ Normal status. (Risk: {prediction:.2f})")

# Simulate delay between readings
time.sleep(5)

# Run simulation
simulate_data_stream()