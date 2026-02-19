import os
import librosa
import numpy as np

# Dataset path
DATA_PATH = "data/cry_audio"

# Your 8 baby cry classes
labels = [
    "belly_pain",
    "burping",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired","non_cry"
]

# Auto label mapping
label_map = {i: label for i, label in enumerate(labels)}

X = []
y = []


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zcr, axis=1),
        np.mean(spec_centroid, axis=1)
    ])
    return features


for label_index, label in enumerate(labels):

    folder_path = os.path.join(DATA_PATH, label)

    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        continue

    print("\nProcessing:", label)

    files = os.listdir(folder_path)

    for count, file in enumerate(files):

        # Ignore non-wav and hidden files
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            features = extract_features(file_path)

            X.append(features)
            y.append(label_index)

            print(f"{label} file processed:", count + 1)

        except Exception as e:
            print("Error in file:", file)
            print(e)


# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save dataset
np.save("data/X_features.npy", X)
np.save("data/y_labels.npy", y)
np.save("data/label_map.npy", label_map)


print(" DATASET EXTRACTION COMPLETED")
print("Total samples:", X.shape[0])
print("Feature size:", X.shape[1])
print("Total classes:", len(labels))

