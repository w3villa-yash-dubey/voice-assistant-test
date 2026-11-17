# LiveKit Voice Agent (local Mithi v2 Flash TTS)

Browser UI → **LiveKit** → agent (STT → LLM → local **Mithi** TTS).

TTS model: [Oomni/mithi-v2-flash](https://huggingface.co/Oomni/mithi-v2-flash)

## Run on your laptop / Jetson

You need **3 terminals** (skip LiveKit if it is already running).

### 0. One-time setup

```bash
cd voice-assistant-test
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install "misaki[en]>=0.9.4" "espeakng-loader>=0.2.4" "phonemizer-fork>=3.3.2" "spacy>=3.8"
python -m spacy download en_core_web_sm

cp .env.example .env   # set OPENAI_API_KEY
python scripts/download_mithi.py
# ensure .env has: MITHI_MODEL_DIR=models/mithi-v2-flash
```

### 1. LiveKit server

```bash
docker compose up -d
```

Skip if LiveKit is already up on `ws://127.0.0.1:7880`.

### 2. Voice agent worker

```bash
source .venv/bin/activate
python agent.py start
```

Wait for `registered worker`.

### 3. Small UI

```bash
source .venv/bin/activate
python web_server.py
```

Open **http://127.0.0.1:8080** → **Start talking** → allow mic.

## `.env` essentials

```bash
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
AGENT_NAME=mithi-voice-agent
OPENAI_API_KEY=sk-...
UI_PORT=8080
MITHI_MODEL_DIR=models/mithi-v2-flash
MITHI_LANG=en
MITHI_VOICE=ahana
MITHI_DEVICE=auto
```

## Layout

```
agent.py                 # LiveKit agent worker (Mithi TTS)
web_server.py            # UI + /api/token (dispatches agent)
static/                  # small talk UI
livekit.yaml             # local LiveKit server
docker-compose.yml
app/plugins/mithi_tts.py # local Mithi TTS plugin
app/services/mithi_tts.py
scripts/download_mithi.py
```
