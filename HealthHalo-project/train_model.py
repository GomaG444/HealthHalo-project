# train_model.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'heart.csv')

# Load dataset
data = pd.read_csv(csv_path)

# Explore
print("Data Info:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())
print("\nSummary statistics:")
print(data.describe())
print(data.head())

# Prepare features/target
X = data.drop('target', axis=1)
y = data['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
importances = np.abs(model.coef_[0])
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.bar(range(X.shape[1]), importances[indices], color="skyblue", align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# Save the trained model
model_path = os.path.join(script_dir, 'model.pkl')
joblib.dump(model, model_path)
print(f"\n✅ Model saved to {model_path}")
