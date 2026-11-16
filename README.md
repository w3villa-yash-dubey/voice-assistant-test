# WebRTC Voice Assistant (Jetson test)

Simple voice agent over **plain WebRTC** (no LiveKit). TTS is **only Goonj** — [`BH-Builds/goonj-1-82M`](https://huggingface.co/BH-Builds/goonj-1-82M).

## What it does

1. Browser mic → WebRTC → Python server  
2. **VAD** detects end of utterance  
3. **STT** → **LLM** (OpenAI-compatible API)  
4. **Goonj TTS** streams audio back over WebRTC  
5. Speak while it talks → **barge-in** stops TTS  

## Quick start

```bash
# system dep for Hindi/English G2P
# macOS:  brew install espeak
# Ubuntu/Jetson: sudo apt-get install -y espeak-ng

cd voice-assistant-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install kokoro misaki --no-deps   # required (avoids spacy)
cp .env.example .env                  # set OPENAI_API_KEY
python -m scripts.download_goonj      # prefetch weights
python -m app.main
```

Open `http://127.0.0.1:8080` → **Start** → talk.

## Goonj settings

| Env | Default | Notes |
|-----|---------|-------|
| `GOONJ_PERSONA` | `hi_meera` | `hi_*`, `en_*`, or `bed_hindi` |
| `GOONJ_LANG` | `auto` | `auto` / `hi` / `en` / `hing` |
| `GOONJ_SPEED` | `1.0` | speaking rate |
| `GOONJ_DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `HF_TOKEN` | | only if HF gated/private |

Personas: `hi_atul`, `hi_meera`, `hi_ravi`, `hi_shivani`, `en_aman`, `en_ananya`, `en_arjun`, `en_dev`, `en_divya`, `en_kabir`, `en_nisha`, `en_priya`, `en_sameer`, `en_tara`, `bed_hindi`.

## Jetson deploy (isolated)

```bash
ssh user@jetson
git clone <YOUR_REPO_URL> ~/voice-assistant-test
cd ~/voice-assistant-test
bash deploy/setup.sh          # venv + espeak-ng + Goonj weights + systemd
# edit /opt/voice-assistant/.env
sudo systemctl start voice-assistant
```

Open `http://<jetson-ip>:8080`.

Docker:

```bash
docker build -t voice-assistant .
docker run --rm -p 8080:8080 --env-file .env voice-assistant
```

## Success check on Nano

- Hear Goonj voice replies  
- Mid-sentence interrupt works  
- Service restarts cleanly  
- Prefer cloud STT/LLM + local Goonj on Nano (82M is edge-friendly)
