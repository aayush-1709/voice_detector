import os
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

DATASET_PATH = "dataset"

X = []
y = []

def extract_features(file):
    audio, sr = librosa.load(file, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

print("🔵 Starting feature extraction...")

# HUMAN = 0
human_files = os.listdir(f"{DATASET_PATH}/human")
print(f"🟢 Human samples found: {len(human_files)}")

for f in tqdm(human_files, desc="Processing HUMAN"):
    feat = extract_features(f"{DATASET_PATH}/human/{f}")
    X.append(feat)
    y.append(0)

# AI = 1
ai_files = os.listdir(f"{DATASET_PATH}/ai")
print(f"🔴 AI samples found: {len(ai_files)}")

for f in tqdm(ai_files, desc="Processing AI"):
    feat = extract_features(f"{DATASET_PATH}/ai/{f}")
    X.append(feat)
    y.append(1)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples loaded: {len(X)}")

print("\n🧠 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Training model...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

print("📊 Evaluating model...")
acc = model.score(X_test, y_test)
print(f"\n🎯 Validation accuracy: {acc:.4f}")

joblib.dump(model, "voice_model.pkl")
print("\n💾 Model saved as voice_model.pkl")

print("\n🏁 Training complete!")
