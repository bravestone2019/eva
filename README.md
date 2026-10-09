# EVA

EVA is a voice-first assistant with a React chat interface and a FastAPI backend.

## Run the API

Install Python 3.12 for Windows from [python.org](https://www.python.org/downloads/windows/). PyTorch on Windows currently supports Python 3.9 through 3.12. From the project root, create an environment and install the CPU dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

For an NVIDIA GPU setup, install `requirements-gpu.txt` in the environment instead of installing `torch` and `requirements.txt` separately. The first chat request loads EVA's models, which can take a while and may download model files.

## Run the React interface

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Install Node.js from [nodejs.org](https://nodejs.org/en/download/) if `npm` is not recognized, then reopen PowerShell. Open the local URL printed by Vite. The interface connects to `http://localhost:8000` by default. Set `VITE_API_BASE_URL` before starting Vite to use a different API address. Microphone recording requires browser permission and a secure context (localhost is allowed).

The interface supports text chat, browser speech recognition, microphone recording as a fallback, and audio-file upload. Voice conversation mode sends each spoken turn after a short pause, reads EVA's reply aloud once in the feminine en-US Aria Neural voice, then listens for the next turn until stopped. Speech audio requires an internet connection. Conversations are saved in the browser's local storage, listed under “Your Space,” and can be started with `Alt+K`; the FastAPI service does not persist typed chat history.
