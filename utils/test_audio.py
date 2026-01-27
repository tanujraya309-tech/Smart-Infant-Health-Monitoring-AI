import librosa
import numpy as np

file_path = "data/cry_audio/hungry/02c3b725-26e4-4a2c-9336-04ddc58836d9-1430726196216-1.7-m-04-hu.wav"

audio, sample_rate = librosa.load(file_path)

mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

features = np.mean(mfcc.T, axis=0)

print("MFCC Features Extracted")
print("Feature Vector Length:", len(features))
print(features)
