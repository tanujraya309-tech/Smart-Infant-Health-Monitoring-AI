import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

print("\n====== BABY CRY XGBOOST TRAINING STARTED ======\n")

# -----------------------------------
# Load Dataset
# -----------------------------------

X = np.load("../data/X_features.npy")
y = np.load("../data/y_labels.npy")

print("Dataset Loaded")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# -----------------------------------
# Fix NaN / Inf Values
# -----------------------------------

X = np.nan_to_num(X)

# -----------------------------------
# Encode Labels (IMPORTANT)
# -----------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nClasses Found:", encoder.classes_)

# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Done")

# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Samples:", X_train.shape[0])
print("Test Samples:", X_test.shape[0])

# -----------------------------------
# XGBoost Model (MULTICLASS SAFE)
# -----------------------------------

num_classes = len(np.unique(y))

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("\nModel Training Completed")

# -----------------------------------
# Model Evaluation
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------")
print("MODEL ACCURACY:", accuracy)
print("-----------------------------\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------------
# Save Everything
# -----------------------------------

joblib.dump(model, "../data/xgboost_babycry_model.pkl")
joblib.dump(scaler, "../data/scaler.pkl")
joblib.dump(encoder, "../data/label_encoder.pkl")

print("scaler.pkl")
print("label_encoder.pkl")

print("\n====== TRAINING FINISHED SUCCESSFULLY ======\n")

