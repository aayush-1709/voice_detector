import os
import base64
import io
from fastapi import FastAPI, Header, HTTPException
from dotenv import load_dotenv
import librosa
import numpy as np
import joblib

# Load .env
load_dotenv()

API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    raise RuntimeError("API_KEY not found in .env file")

app = FastAPI()

print("🔵 Loading ML model...")
model = joblib.load("voice_model.pkl")
print("✅ Model loaded")

def extract_features_bytes(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

@app.post("/api/voice-detection")
async def detect(data: dict, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        language = data["language"]
        audio_b64 = data["audioBase64"]

        audio_bytes = base64.b64decode(audio_b64)

        features = extract_features_bytes(audio_bytes).reshape(1, -1)

        prob = model.predict_proba(features)[0][1]

        classification = "AI_GENERATED" if prob > 0.5 else "HUMAN"

        explanation = (
            "Synthetic spectral patterns detected"
            if classification == "AI_GENERATED"
            else "Natural speech variations detected"
        )

        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": float(round(prob, 4)),
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
