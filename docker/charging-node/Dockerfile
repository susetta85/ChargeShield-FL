# docker/charging-node/Dockerfile
# Immagine Docker per ChargingNode — ChargeShield-FL
#
# Contiene:
# - Python 3.12
# - Framework ChargeShield-FL (src/)
# - Dipendenze da pyproject.toml
# - Configurazione mTLS
#
# Build:
#   docker build -t chargeshield/charging-node:latest -f docker/charging-node/Dockerfile .
#
# Variabili d'ambiente richieste (settate in topology.clab.yml):
#   NODE_ID        → identificatore del nodo (es. highway-01)
#   CLUSTER_ID     → cluster di appartenenza (es. highway)
#   PROTOCOL       → protocollo (OCPP_16 | OCPP_20 | MQTT_v5)
#   MAX_POWER_KW   → potenza massima in kW
#   AGGREGATOR_URL → URL del server FLARE (es. aggregator:8002)

FROM python:3.12-slim

# Metadata immagine
LABEL maintainer="ChargeShield-FL"
LABEL version="0.3.0"
LABEL description="ChargingNode — FL client per colonnine EV"

# Variabili d'ambiente di default
# Sovrascrivibili da topology.clab.yml
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_DIR=/app \
    CERTS_DIR=/app/certs \
    CONFIG_DIR=/app/config \
    NODE_ID=unknown \
    CLUSTER_ID=unknown \
    PROTOCOL=OCPP_16 \
    MAX_POWER_KW=22 \
    AGGREGATOR_URL=aggregator:8002

# Installa dipendenze di sistema
# openssl: per mTLS
# curl: per healthcheck
RUN apt-get update && apt-get install -y \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Directory di lavoro
WORKDIR ${APP_DIR}

# Copia e installa dipendenze Python
# Copiamo prima solo pyproject.toml per sfruttare la cache Docker
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copia il codice sorgente del framework
COPY src/ ./src/

# Le directory seguenti vengono montate via bind in topology.clab.yml
# e NON vengono copiate nell'immagine:
#   /app/config/     → config/
#   /app/certs/      → certs/NODE_ID/
#   /app/datasets/   → datasets/acn/

# Healthcheck: verifica che il nodo sia online
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Entrypoint del nodo
CMD ["python3", "-m", "src.nodes.charging_node"]
