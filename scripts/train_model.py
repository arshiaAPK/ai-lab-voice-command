from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from scipy.io import wavfile
from scipy.signal import resample
from python_speech_features import mfcc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

MANIFEST = Path("data/processed/manifest.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def extract_features(path):
    rate, audio = wavfile.read(path)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    features = mfcc(audio, samplerate=rate, numcep=13)
    return np.concatenate([features.mean(axis=0), features.std(axis=0)])

df = pd.read_csv(MANIFEST)

X = np.array([extract_features(p) for p in df["file_path"]])
y = df["label"].values
splits = df["split"].values

X_train, X_test = X[splits == "train"], X[splits == "test"]
y_train, y_test = y[splits == "train"], y[splits == "test"]

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train_encoded)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test_encoded, predictions))
print("\nClassification report:")
print(classification_report(
    y_test_encoded,
    predictions,
    target_names=encoder.classes_,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test_encoded, predictions))

joblib.dump(
    {"model": model, "encoder": encoder},
    MODEL_DIR / "voice_command_model.joblib"
)

print("\nمدل ذخیره شد.")
