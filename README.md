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

The interface supports text chat, browser speech recognition, microphone recording, and audio-file upload. Voice conversation mode sends each spoken turn after a short pause, reads EVA's reply aloud once, then listens for the next turn until stopped. Open **Settings** to choose browser speech recognition or local Whisper input, and Edge Aria Neural TTS or a voice installed on the device. Browser recognition may use a browser speech service; Aria Neural requires internet. Local Whisper runs on the EVA backend, so it needs the Whisper model downloaded there.

Chats remain in browser storage until you sign in. Use the profile area at the bottom of the sidebar to create an email/password account or sign in; EVA then syncs that browser's chats to the server and loads them on another browser signed in to the same server. Account email addresses must use Gmail (`gmail.com`) or a Yahoo domain; new email/password registrations require exactly six numeric digits (existing accounts can continue using their current password). The server stores salted password hashes and conversation text in `data/eva_accounts.sqlite3`, never passwords in plain text; the signing key is stored in `data/eva_accounts.secret`. Back up both files to preserve accounts and sessions. Set `EVA_AUTH_SECRET` to a strong secret in deployments where the server database may move between machines. Use HTTPS before exposing account login over a network. Local browser storage remains available when signed out. Chats are listed under “Your Space,” and a new chat can be started with `Alt+K`.

Google sign-in is supported when configured. In Google Cloud Console, create a Web application OAuth client and add your frontend URL (for local Vite, `http://localhost:5173` and `http://127.0.0.1:5173`) to its authorized JavaScript origins. Install the server dependency and configure the same client ID for the backend and frontend:

```powershell
python -m pip install google-auth
```

Put `EVA_GOOGLE_CLIENT_ID=YOUR_CLIENT_ID.apps.googleusercontent.com` in the project-root `.env` file and `VITE_GOOGLE_CLIENT_ID=YOUR_CLIENT_ID.apps.googleusercontent.com` in `frontend/.env.local`. The API loads the root `.env` at startup, and Vite loads `frontend/.env.local`; restart both after editing. Google credentials are verified by the backend; Google sign-in accepts only verified Gmail or supported Yahoo addresses. The Google button remains visible before configuration and explains that it needs an OAuth client ID when clicked. These local environment files are ignored by Git.

For a frontend hosted on another origin, set `EVA_ALLOWED_ORIGINS` to a comma-separated list of trusted frontend origins before starting the API (for example, `https://eva.example.com`). This starter account system does not include email verification, password reset, or deployment-grade abuse protection; keep it on a trusted server and add those controls before public release.

The **Settings** panel also controls reply-style adaptation and wake-phrase listening (both default on), and manages local knowledge files. Add UTF-8 `.txt` or `.md` files up to 2 MB; EVA splits them into overlapping chunks and retrieves likely matches with local TF-IDF before answering. Files live under `data/documents/`, are shared by all accounts on that EVA server, and are not uploaded to an external vector service. This is a lightweight local retrieval baseline, not a semantic embedding index.

Wake-phrase listening starts on page load by default. Toggle it in **Settings** to stop standby listening. Say “Hello EVA,” “Hi EVA,” “Hey EVA,” or “EVA listening” to start a new hands-free conversation. During a conversation, say “end the conversation,” “end the chat,” “stop the conversation,” “goodbye EVA,” or “that’s all” to end it. Wake phrases still rely on the browser's recognition support and microphone permission, including when local Whisper is selected for voice turns.

## Emotion-aware response style prototype

The **Adapt reply style** setting is on by default and can be switched off in **Settings**. EVA estimates broad cues from typed text and, for voice turns, from the captured audio alongside the transcript. During supported browser speech mode, rolling two-second audio windows are analyzed with one-second steps and smoothed; the style is applied after the turn ends. Local Whisper mode analyzes the completed recording. The expanded voice label set is `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, and `surprised`; GoEmotions text cues can additionally include `confused`. Likely anger favors concise replies, fear and sadness favor warm replies, and happiness favors encouraging replies. These are uncertain classifier outputs, not diagnoses or calibrated confidence probabilities.


On first use, the backend may download the LLM and active affect classifiers from Hugging Face and cache them locally. Configure the affect classifiers with `EVA_AFFECT_TEXT_MODEL`, `EVA_AFFECT_AUDIO_MODEL`, and `EVA_AFFECT_CONFIDENCE_THRESHOLD`. If `checkpoints/eva-text-affect-go-emotions/` exists and `EVA_AFFECT_TEXT_MODEL` is unset, EVA uses that local GoEmotions fine-tune; otherwise it uses the pretrained GoEmotions model. If `checkpoints/eva-voice-affect-ravdess-cremad-8class/` exists and `EVA_AFFECT_AUDIO_MODEL` is unset, EVA prefers that expanded checkpoint. It falls back to the older `checkpoints/eva-voice-affect-ravdess-cremad/` four-class checkpoint, then to pretrained SUPERB. The pretrained classifier and legacy checkpoint can only report angry, happy, sad, and neutral; train and configure the eight-class checkpoint before expecting the other four voice labels. On the GoEmotions validation split, fine-tuning raised macro-F1 from 0.448 to 0.500. Its held-out test macro-F1 was 0.489. These are English Reddit metrics and do not establish improved accuracy on EVA conversations. The earlier four-class voice checkpoint scored 0.836 macro-F1 on held-out acted RAVDESS/CREMA-D speech; this does not establish accuracy on natural conversation or the expanded eight-class model. The feed-forward `EmotionModel`, acoustic feature extraction, and `EmotionState` fusion modules remain experimental scaffolding and do not drive replies. EVA skips the unused WavLM download at startup. Use the consented, speaker-separated evaluation workflow in `evaluation/` to evaluate real-world speech.

Optional training workflows for consented voice affect, English/Hindi FLEURS Whisper, and GoEmotions text affect are documented in [evaluation/README.md](evaluation/README.md). They do not run automatically: install the optional dependencies, choose/download data, and explicitly launch a selected script. Dataset and checkpoint folders remain local and are ignored by Git.

For natural voice-affect training, `scripts/prepare_natural_voice_manifest.py` creates a private annotation template with speaker-separated splits. It trains on each speaker's self-reported emotion and can report listener-perceived emotion separately; see the natural-speech section of `evaluation/README.md`.
