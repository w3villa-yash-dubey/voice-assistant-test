FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "misaki[en]>=0.9.4" "espeakng-loader>=0.2.4" \
       "phonemizer-fork>=3.3.2" "spacy>=3.8" \
    && python -m spacy download en_core_web_sm

COPY app ./app
COPY agent.py ./agent.py
COPY scripts ./scripts
COPY livekit.yaml ./livekit.yaml
COPY .env.example ./.env.example

ENV LIVEKIT_URL=ws://host.docker.internal:7880
ENV LIVEKIT_API_KEY=devkey
ENV LIVEKIT_API_SECRET=secret
ENV MITHI_MODEL_DIR=/models/mithi-v2-flash
ENV MITHI_LANG=en
ENV MITHI_VOICE=ahana
ENV MITHI_DEVICE=auto

CMD ["python", "agent.py", "start"]
