import os
import librosa
import numpy as np

# ==============================
# DATASET PATH
# ==============================

DATA_PATH = "data/cry_audio"

# ==============================
# CLASS LABELS (FOLDER NAMES)
# ==============================

labels = [
    "belly_pain",
    "burping",
    "old_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired",
    "non_cry"
]

# Create numeric label mapping
label_map = {label: i for i, label in enumerate(labels)}

print("Label Mapping:")
print(label_map)


# ==============================
# MFCC FEATURE FUNCTION
# ==============================

def extract_mfcc(file_path):
    try:
        # Load WAV audio file
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        # Convert variable length to fixed size
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        return mfcc_scaled

    except Exception as e:
        print("Error processing:", file_path)
        return None


# ==============================
# FEATURE EXTRACTION PROCESS
# ==============================

X = []   # Feature list
y = []   # Label list

print("\nStarting Feature Extraction...\n")

for label in labels:
    
    folder_path = os.path.join(DATA_PATH, label)

    print("Processing folder:", label)

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):

            file_path = os.path.join(folder_path, file)

            features = extract_mfcc(file_path)

            if features is not None:
                X.append(features)
                y.append(label_map[label])


# ==============================
# CONVERT TO NUMPY ARRAYS
# ==============================

X = np.array(X)
y = np.array(y)

print("\nExtraction Completed!")
print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)


# ==============================
# SAVE DATASET
# ==============================

np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\nSaved Files:")
print("X_features.npy")
print("y_labels.npy")


# ==============================
# VERIFY SAVED DATA
# ==============================

X_test = np.load("X_features.npy")
y_test = np.load("y_labels.npy")

print("\nVerification Successful!")
print("Loaded Feature Shape:", X_test.shape)
print("Loaded Label Shape:", y_test.shape)

