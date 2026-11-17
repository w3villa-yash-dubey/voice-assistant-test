# LiveKit Voice Agent (local Goonj TTS)

Browser UI → **LiveKit** → agent (STT → LLM → local **Goonj** TTS).

## Run on your laptop

You need **3 terminals**.

### 1. LiveKit server

```bash
cd voice-assistant-test
docker compose up -d
```

### 2. Voice agent worker

```bash
source .venv/bin/activate
# first time:
# pip install -r requirements.txt && pip install kokoro misaki --no-deps
# cp .env.example .env   # set OPENAI_API_KEY
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
AGENT_NAME=goonj-voice-agent
OPENAI_API_KEY=sk-...
UI_PORT=8080
GOONJ_PERSONA=hi_meera
```

## Layout

```
agent.py                 # LiveKit agent worker
web_server.py            # UI + /api/token (dispatches agent)
static/                  # small talk UI
livekit.yaml             # local LiveKit server
docker-compose.yml
app/plugins/goonj_tts.py # local Goonj TTS plugin
```
