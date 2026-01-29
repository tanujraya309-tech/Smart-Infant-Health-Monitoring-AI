import librosa
import numpy as np
import joblib
import sys
import os

# ===== CHECK INPUT =====

if len(sys.argv) < 2:
    print("Usage: python predict.py test_audio.wav")
    exit()

audio_path = sys.argv[1]

if not os.path.exists(audio_path):
    print("Audio file not found!")
    exit()

# ===== LOAD MODEL =====

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===== CLASS LABELS (MUST MATCH TRAINING) =====

labels = [
    "belly_pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired",
    "non_cry"
]

# ===== LOAD AUDIO =====

audio, sr = librosa.load(audio_path, sr=None)

# ===== FEATURE EXTRACTION =====

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

mfcc_mean = np.mean(mfcc.T, axis=0)

X = mfcc_mean.reshape(1, -1)

# ===== SCALE FEATURES =====

X = scaler.transform(X)

# ===== PREDICTION =====

prediction = model.predict(X)[0]

print("\n🎯 Prediction Result")
print("-------------------")
print("Cry Type:", labels[prediction])

