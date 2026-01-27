import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
X = np.load("data/X_features.npy")
y = np.load("data/y_labels.npy")
label_map = np.load("data/label_map.npy", allow_pickle=True).item()

print("Dataset loaded")
print("Total samples:", len(X))

# -------- CLASS DISTRIBUTION --------

class_counts = Counter(y)

print("\nClass Distribution:")
for key, value in class_counts.items():
    print(label_map[key], ":", value)

# Plot bar graph
plt.figure(figsize=(10,5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Class Index")
plt.ylabel("Number of Samples")
plt.title("Baby Cry Dataset Distribution")
plt.show()


# -------- REMOVE BAD SAMPLES --------

# Remove rows with NaN or Inf values
valid_rows = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)

X_clean = X[valid_rows]
y_clean = y[valid_rows]

print("\nAfter Cleaning:")
print("Samples left:", len(X_clean))


# -------- NORMALIZATION --------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_clean)

# Save cleaned dataset
np.save("data/X_clean.npy", X_normalized)
np.save("data/y_clean.npy", y_clean)

print("\nCleaned and normalized dataset saved")

