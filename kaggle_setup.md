# Run EVA on Kaggle GPU

1. Create a Kaggle Notebook, choose **GPU T4 x2** (or P100), and enable
   Internet for the first model download. Upload this repository as a Dataset
   or add it with `!git clone`.
2. In a notebook cell, run:

```python
%cd /kaggle/working/eva
!pip install -q -r requirements.txt
```

3. Upload a WAV/M4A/MP3 audio file through the Kaggle file panel, then run:

```python
!python app/main.py --input-file /kaggle/input/YOUR_DATASET/recording.wav
```

Kaggle has no microphone device, so `--record-seconds` is for your local
computer only. Record in Flutter (or locally) and upload the resulting file.

## Flutter-ready API

For local development or a deployed GPU machine, start the API:

```python
!uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Flutter sends `multipart/form-data` with its audio file in a field named
`audio` to `POST /v1/voice`. The JSON response has `transcription`, `emotion`,
and `response` fields. `GET /health` is a simple readiness check.

Kaggle notebook URLs are not persistent public API endpoints, so do not point
a released Flutter app at Kaggle. Move this exact FastAPI service to a hosted
GPU endpoint before release.
