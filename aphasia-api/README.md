# Aphasia Rehabilitation Speech Evaluation API

Production-ready FastAPI backend for evaluating patient speech against a reference pronunciation using **dual-model analysis**:

1. **Audio-level**: WavLM embeddings + DTW alignment for pronunciation similarity scoring
2. **Text-level**: Wav2Vec2 CTC transcription + CER/WER metrics for word-level accuracy

## Features

- **Dual-model evaluation** — audio similarity (WavLM + DTW) combined with text accuracy (Wav2Vec2 + CER/WER)
- **Combined scoring** — weighted blend of audio and text metrics for clinically meaningful accuracy
- **Human-readable feedback** — patient-friendly messages with specific guidance
- **Silence trimming** — automatically removes leading/trailing silence (important for aphasia patients)
- **Therapist text override** — optional `reference_text` field bypasses ASR on the reference
- `POST /evaluate` endpoint for multipart audio evaluation
- `GET /health` endpoint
- CORS enabled for frontend integration
- Models pre-loaded at startup (no cold-start latency)
- File-size and invalid-audio error handling
- Structured logging with Loguru

## Project Structure

```
aphasia-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory, CORS, lifespan
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # /evaluate and /health endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Settings with env var support
│   │   └── logging.py             # Loguru configuration
│   ├── services/
│   │   ├── __init__.py
│   │   ├── wavlm_service.py       # WavLM audio embeddings
│   │   ├── wav2vec2_service.py     # Wav2Vec2 CTC transcription
│   │   ├── dtw_service.py         # Dynamic Time Warping alignment
│   │   └── scoring_service.py     # Combined scoring + feedback
│   ├── models/
│   │   ├── __init__.py
│   │   └── response.py            # Pydantic response schemas
│   └── utils/
│       ├── __init__.py
│       ├── audio.py               # Audio loading, trimming, normalisation
│       └── text_metrics.py        # CER/WER computation
│
├── .env.example
├── requirements.txt
└── README.md
```

## API Contract

### `POST /evaluate`

**Input:** multipart/form-data
- `reference_audio`: file (required)
- `user_audio`: file (required)
- `reference_text`: string (optional — therapist can provide expected text)

**Response example:**

```json
{
    "combined_accuracy": 82.1,
    "feedback_message": "Good effort! You scored 82%. The main difficulty was in the middle of the phrase. Try focusing on the word 'sat'.",
    "accuracy": 85.4,
    "audio_similarity": 0.854,
    "deviation_position": "MIDDLE",
    "deviation_index": 34,
    "mismatch_indices": [34, 35, 36],
    "dtw_distance": 57.44,
    "alignment_path_length": 128,
    "reference_text": "the cat sat on the mat",
    "user_text": "the cat at on the mat",
    "text_similarity": 0.789,
    "character_error_rate": 0.211,
    "word_error_rate": 0.333,
    "processing_time_ms": 320.5
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

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `WAVLM_MODEL_NAME` | `microsoft/wavlm-base-plus` | WavLM model for audio embeddings |
| `WAV2VEC2_MODEL_NAME` | `facebook/wav2vec2-large-960h` | Wav2Vec2 model for transcription |

See `.env.example` for the full list.

## Models Used

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| WavLM Base Plus | Audio embedding extraction | ~380 MB | [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) |
| Wav2Vec2 Large 960h | CTC speech-to-text | ~1.2 GB | [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) |

> **Tip:** For lower memory usage, set `WAV2VEC2_MODEL_NAME=facebook/wav2vec2-base-960h` (~360 MB).

## Notes

- **Combined scoring** uses 70% audio similarity + 30% text similarity by default.
- Audio scoring is based on WavLM + DTW only — captures pronunciation nuances beyond text.
- Text scoring uses Wav2Vec2 CTC + CER/WER — captures word-level accuracy.
- Silence is automatically trimmed from audio inputs.
- Models are pre-loaded at startup. First startup may take 30-60s depending on network speed and hardware.
- Wav2Vec2 transcription failures are handled gracefully — scoring falls back to audio-only.
