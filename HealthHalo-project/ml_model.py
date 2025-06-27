import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the CSV file
csv_path = os.path.join(script_dir, 'heart.csv')

# Load the dataset
data = pd.read_csv(csv_path)

# Explore the dataset
print("Data Info:")
print(data.info())

print("\nMissing values per column:")
print(data.isnull().sum())

print("\nSummary statistics:")
print(data.describe())

# Quick look at data
print(data.head())

# Prepare features and target
X = data.drop('target', axis=1)  # assuming 'target' is the label column
y = data['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importances (coefficients) from the trained Logistic Regression model
importances = np.abs(model.coef_[0])
features = X.columns

# Sort feature importances
indices = importances.argsort()[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.bar(range(X.shape[1]), importances[indices], color="skyblue", align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
