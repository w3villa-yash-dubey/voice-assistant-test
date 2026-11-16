#!/usr/bin/env bash
# Run on Jetson after cloning the repo. Creates an isolated venv + optional systemd unit.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/voice-assistant}"
SERVICE_USER="${SERVICE_USER:-$USER}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Installing into ${APP_DIR} (user=${SERVICE_USER})"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  # espeak-ng is required for Goonj Hindi / English G2P (misaki.espeak)
  sudo apt-get install -y python3-venv python3-dev build-essential rsync espeak-ng libespeak-ng1
fi

sudo mkdir -p "${APP_DIR}"
sudo rsync -a --delete \
  --exclude '.venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  "${REPO_DIR}/" "${APP_DIR}/"
sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"

cd "${APP_DIR}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Avoid misaki[en] → spacy (heavy / often broken). We only need KModel + EspeakG2P.
pip install kokoro misaki --no-deps

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Created .env — edit OPENAI_API_KEY / GOONJ_PERSONA before starting"
fi

echo "==> Prefetching Goonj weights from Hugging Face…"
python -m scripts.download_goonj || echo "==> Prefetch skipped (will download on first start)"

# Optional: install systemd unit
UNIT_SRC="${APP_DIR}/deploy/voice-assistant.service"
UNIT_DST="/etc/systemd/system/voice-assistant.service"
if [[ -f "${UNIT_SRC}" ]]; then
  tmp="$(mktemp)"
  sed -e "s|/opt/voice-assistant|${APP_DIR}|g" \
      -e "s|User=nvidia|User=${SERVICE_USER}|g" \
      -e "s|Group=nvidia|Group=${SERVICE_USER}|g" \
      "${UNIT_SRC}" > "${tmp}"
  sudo cp "${tmp}" "${UNIT_DST}"
  rm -f "${tmp}"
  sudo systemctl daemon-reload
  sudo systemctl enable voice-assistant.service
  echo "==> systemd unit installed. Start with: sudo systemctl start voice-assistant"
fi

echo "==> Done. Manual run: cd ${APP_DIR} && source .venv/bin/activate && python -m app.main"
echo "==> Open: http://<jetson-ip>:8080"
