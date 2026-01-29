import librosa
import os
import numpy as np
import soundfile as sf

DATA_PATH = "data/cry_audio"
CLEAN_PATH = "data/clean_audio"

MIN_DURATION = 0.5

processed = 0
skipped = 0

for label_folder in os.listdir(DATA_PATH):

    class_path = os.path.join(DATA_PATH, label_folder)

    if not os.path.isdir(class_path):
        continue

    # Create same class folder in clean_audio
    save_class_path = os.path.join(CLEAN_PATH, label_folder)
    os.makedirs(save_class_path, exist_ok=True)

    print("Processing class:", label_folder)

    for file in os.listdir(class_path):

        if not file.lower().endswith(".wav"):
            continue

        try:
            file_path = os.path.join(class_path, file)

            audio, sr = librosa.load(file_path, sr=None, mono=True)

            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=30)

            # Skip very short clips
            if len(audio) / sr < MIN_DURATION:
                skipped += 1
                continue

            # Normalize safely
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # Save cleaned audio
            save_path = os.path.join(save_class_path, file)
            sf.write(save_path, audio, sr)

            processed += 1

        except Exception as e:
            print("Error:", file, e)
            skipped += 1


print("\nDONE")
print("Processed files:", processed)
print("Skipped files:", skipped)


