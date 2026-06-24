# docker/ids/Dockerfile
# Immagine Docker per l'IDS (Intrusion Detection System)
#
# Variabili d'ambiente richieste:
#   IDS_PORT         → porta del servizio (default 8005)
#   ALERT_THRESHOLD  → soglia alert (default 0.7)

FROM python:3.12-slim

LABEL maintainer="ChargeShield-FL"
LABEL version="0.3.0"
LABEL description="IDS — Intrusion Detection System (difesa)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_DIR=/app \
    CERTS_DIR=/app/certs \
    IDS_PORT=8005 \
    ALERT_THRESHOLD=0.7

RUN apt-get update && apt-get install -y \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_DIR}

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY src/ ./src/

EXPOSE ${IDS_PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${IDS_PORT}/health || exit 1

CMD ["python3", "-m", "src.ids.charging_ids"]
