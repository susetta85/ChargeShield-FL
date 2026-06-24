# docker/fl-admin/Dockerfile
# Immagine Docker per FL Admin (NVIDIA FLARE Admin)
# Gestisce il ciclo di vita del server FL:
# avvio, shutdown, monitoring dei round
#
# Variabili d'ambiente richieste:
#   ADMIN_PORT → porta admin (default 8003)

FROM python:3.12-slim

LABEL maintainer="ChargeShield-FL"
LABEL version="0.3.0"
LABEL description="FL Admin — gestione ciclo di vita FLARE"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_DIR=/app \
    CERTS_DIR=/app/certs \
    FLARE_ROLE=admin \
    ADMIN_PORT=8003

RUN apt-get update && apt-get install -y \
    openssl \
    curl \
    && rm -rf /var/lib/abd/lists/*

WORKDIR ${APP_DIR}

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev,flare]"

COPY src/ ./src/

EXPOSE ${ADMIN_PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${ADMIN_PORT}/health || exit 1

CMD ["python3", "-m", "src.flare.fl_admin"]
