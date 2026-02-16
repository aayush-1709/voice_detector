# Voice Detector


Simple audio classifier to detect AI-generated vs human audio using MFCC features
and a lightweight Logistic Regression model.

## Project Overview

This project trains a classifier to distinguish AI-generated speech from human
speech. It extracts MFCC features from audio files and trains a Logistic
Regression model (scikit-learn). A FastAPI app exposes a simple HTTP API for
inference.

## Project Structure

- `app.py` - FastAPI inference server (POST `/api/voice-detection`)
- `train.py` - training script that builds `voice_model.pkl`
- `requirements.txt` - Python dependencies


## Dataset Description

Organize audio samples under `dataset/ai/` (AI-generated) and
`dataset/human/` (human-recorded). Each file should be a short speech clip
(WAV/MP3). During pre-processing we resample to 16 kHz and convert to mono
before feature extraction.

Recommended dataset layout:

- `dataset/ai/`
- `dataset/human/`

Larger, diverse datasets with multiple speakers and recording conditions
improve robustness.

## Model Type

- Feature extraction: MFCC (Mel-frequency cepstral coefficients), 40 coefficients
	with mean and standard deviation aggregation.
- Classifier: Logistic Regression (scikit-learn). Model file: `voice_model.pkl`.

## Training Details

Training (implemented in `train.py`) follows these steps:

1. Load audio files from `dataset/ai/` and `dataset/human/`.
2. Resample to 16 kHz and convert to mono.
3. Extract MFCC features (e.g. 40 MFCCs), aggregate via mean and std to form
	 a fixed-length feature vector per file.
4. Fit a `sklearn.linear_model.LogisticRegression` (or `SGDClassifier` with
	 logistic loss for large datasets).
5. Save the trained model with `joblib.dump(model, "voice_model.pkl")`.

Quick train command:

```
python train.py
```

Hyperparameters and tips:

- Standardize features (zero-mean, unit-variance) before training.
- Use cross-validation to select regularization (`C`) for Logistic Regression.
- Augment data (noise, pitch/time shifts) if dataset is small.

## Deployment Steps

1. Ensure `voice_model.pkl` exists at the project root (created by `train.py`).
2. Create a `.env` file with `API_KEY=your_secret_key`.
3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the FastAPI app (recommended with `uvicorn`):

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

5. The API endpoint is POST `/api/voice-detection` and requires the
	 `X-API-KEY` header (set to the value in `.env`).

## API Usage Example

Endpoint: `POST /api/voice-detection`

Request JSON body fields:

- `language` (string): language code or description (optional for model)
- `audioBase64` (string): base64-encoded raw audio bytes (WAV/MP3)

Required header:

- `X-API-KEY`: your API key from `.env`

Example request body:

```json
{
	"language": "en-US",
	"audioBase64": "<BASE64_AUDIO_STRING>"
}
```

Example success response:

```json
{
	"status": "success",
	"classification": "AI_GENERATED",
	"confidenceScore": 0.8723,
}
```

## Sample curl Request

Replace `<API_KEY>` and `<BASE64_AUDIO>` with your values.

```bash
curl -X POST "http://localhost:8000/api/voice-detection" \
	-H "Content-Type: application/json" \
	-H "X-API-KEY: <API_KEY>" \
	-d '{"language":"en-US","audioBase64":"<BASE64_AUDIO>"}'
```

Tip: To produce `<BASE64_AUDIO>` from a local file `sample.wav`:

```bash
base64 sample.wav | tr -d '\n' > sample_b64.txt
```

Then paste the contents of `sample_b64.txt` into the JSON `audioBase64` field
or programmatically read/encode the file when issuing requests.