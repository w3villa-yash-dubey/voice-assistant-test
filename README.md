# LiveKit Voice Agent (local Goonj TTS)

Browser/mic → **LiveKit** → STT → LLM → local **Goonj** TTS.

## Quick start

### 1. Start LiveKit server

```bash
cd voice-assistant-test
docker compose up -d
```

Credentials (`devkey` / `secret`): `ws://127.0.0.1:7880`

### 2. Install the agent

```bash
# macOS: brew install espeak
# Ubuntu/Jetson: sudo apt-get install -y espeak-ng

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install kokoro misaki --no-deps
cp .env.example .env                  # set OPENAI_API_KEY
python -m scripts.download_goonj      # prefetch Goonj weights
```

### 3. Run the LiveKit agent

```bash
python agent.py dev
```

Open [Agents Playground](https://agents-playground.livekit.io/), set URL `ws://127.0.0.1:7880`, key `devkey`, secret `secret`, start a session, and talk.

## Goonj settings

| Env | Default | Notes |
|-----|---------|-------|
| `GOONJ_PERSONA` | `hi_meera` | `hi_*`, `en_*`, or `bed_hindi` |
| `GOONJ_LANG` | `auto` | `auto` / `hi` / `en` / `hing` |
| `GOONJ_SPEED` | `1.0` | speaking rate |
| `GOONJ_DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `HF_TOKEN` | | only if HF gated/private |

## Layout

```
agent.py                 # LiveKit agent worker
livekit.yaml             # local LiveKit server config
docker-compose.yml       # LiveKit server
app/services/tts.py      # local Goonj synthesizer
app/plugins/goonj_tts.py # LiveKit TTS plugin
```
