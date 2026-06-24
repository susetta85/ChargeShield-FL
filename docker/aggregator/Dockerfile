# docker/aggregator/Dockerfile
# Immagine Docker per il server FL (NVIDIA FLARE Aggregator)
#
# Contiene:
# - Python 3.12
# - NVIDIA FLARE
# - Framework ChargeShield-FL (src/)
# - Configurazione mTLS
#
# Variabili d'ambiente richieste:
#   FLARE_ROLE     → server
#   SERVER_PORT    → porta FL (default 8002)
#   MIN_CLIENTS    → quorum minimo per round FL

FROM python:3.12-slim

LABEL maintainer="ChargeShield-FL"
LABEL version="0.3.0"
LABEL description="FLARE Aggregator — server FL per ChargeShield-FL"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_DIR=/app \
    CERTS_DIR=/app/certs \
    FLARE_ROLE=server \
    SERVER_PORT=8002 \
    MIN_CLIENTS=3

RUN apt-get update && apt-get install -y \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_DIR}

COPY pyproject.toml .
# Installa anche le dipendenze FLARE
RUN pip install --no-cache-dir -e ".[dev,flare]"

COPY src/ ./src/

EXPOSE ${SERVER_PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

CMD ["python3", "-m", "src.flare.flare_connector"]
