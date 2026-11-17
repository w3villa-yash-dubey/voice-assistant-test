FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir kokoro misaki --no-deps

COPY app ./app
COPY agent.py ./agent.py
COPY scripts ./scripts
COPY livekit.yaml ./livekit.yaml
COPY .env.example ./.env.example

ENV LIVEKIT_URL=ws://host.docker.internal:7880
ENV LIVEKIT_API_KEY=devkey
ENV LIVEKIT_API_SECRET=secret
ENV GOONJ_PERSONA=hi_meera

CMD ["python", "agent.py", "start"]
