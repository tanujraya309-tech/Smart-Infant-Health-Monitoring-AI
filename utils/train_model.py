import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
X = np.load("data/X_features.npy")
y = np.load("data/y_labels.npy")

print("Dataset Loaded")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=500,max_depth=30,
    min_samples_split=2,
    random_state=42)
model.fit(X_train, y_train)

print("Model training completed")

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model and scaler
joblib.dump(model, "data/baby_cry_model.pkl")
joblib.dump(scaler, "data/scaler.pkl")

print(classification_report(y_test, y_pred))

print("Model saved successfully")

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved")

