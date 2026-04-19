# Aphasia Rehabilitation Speech Evaluation API

Production-ready FastAPI backend for evaluating patient speech against a reference pronunciation using **audio similarity** (WavLM + DTW), with optional Whisper transcription.

## Features

- Audio-based pronunciation similarity (not text-only comparison)
- `POST /evaluate` endpoint for multipart audio evaluation
- WavLM embeddings from HuggingFace Transformers (loaded once)
- DTW alignment using `fastdtw`
- Similarity-to-accuracy conversion and deviation position detection
- Optional Whisper transcription for UI display
- Alignment-based mismatch indices
- `/health` endpoint
- Processing time in response
- File-size and invalid-audio error handling
- Structured logging with Loguru

## Project Structure

```
aphasia-api/
│
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── services/
│   │   ├── wavlm_service.py
│   │   ├── dtw_service.py
│   │   ├── whisper_service.py
│   │   └── scoring_service.py
│   ├── models/
│   │   └── response.py
│   └── utils/
│       └── audio.py
│
├── requirements.txt
└── README.md
```

## API Contract

### `POST /evaluate`

**Input:** multipart/form-data
- `reference_audio`: file
- `user_audio`: file

**Response example:**

```json
{
  "accuracy": 91.2,
  "deviation_position": "MIDDLE",
  "similarity_score": 0.912,
  "deviation_index": 34,
  "reference_text": "hello world",
  "user_text": "hello word",
  "mismatch_indices": [34, 35, 36],
  "dtw_distance": 57.44,
  "alignment_path_length": 128,
  "processing_time_ms": 483.1
}
```

### `GET /health`

```json
{
  "status": "ok"
}
```

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

From the `aphasia-api` directory:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables (Optional)

- `LOG_LEVEL` (default: `INFO`)
- `WAVLM_MODEL_NAME` (default: `microsoft/wavlm-base-plus`)
- `WHISPER_MODEL_NAME` (default: `base`)

## Notes

- Scoring is based on WavLM + DTW only.
- Whisper output is optional and for display purposes.
- First request may be slower because model weights are loaded.
- Whisper failures do not break scoring; transcription falls back to `null`.
