# docker/auditor/Dockerfile
# Immagine Docker per il Privacy Auditor (Membership Inference Attacker)
#
# Variabili d'ambiente richieste:
#   AUDITOR_PORT → porta del servizio (default 8004)

FROM python:3.12-slim

LABEL maintainer="ChargeShield-FL"
LABEL version="0.3.0"
LABEL description="Privacy Auditor — Membership Inference Attacker"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_DIR=/app \
    CERTS_DIR=/app/certs \
    AUDITOR_PORT=8004

RUN apt-get update && apt-get install -y \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_DIR}

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY src/ ./src/

EXPOSE ${AUDITOR_PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${AUDITOR_PORT}/health || exit 1

CMD ["python3", "-m", "src.auditor.privacy_auditor"]
