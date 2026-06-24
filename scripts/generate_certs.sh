#!/bin/bash
# scripts/generate_certs.sh
# Genera certificati mTLS per tutti i nodi di ChargeShield-FL
#
# Struttura generata:
#   certs/
#   ├── ca/
#   │   ├── ca.crt      → Certificate Authority (condiviso da tutti)
#   │   └── ca.key      → Chiave privata CA (non distribuita ai nodi)
#   ├── highway-01/
#   │   ├── node.crt    → Certificato del nodo
#   │   ├── node.key    → Chiave privata del nodo
#   │   └── ca.crt      → CA pubblica (per verificare gli altri nodi)
#   └── ...
#
# Uso:
#   bash scripts/generate_certs.sh certs "highway-01 urban-01 ..."
#   oppure tramite Makefile:
#   make certs
#
# Requisiti: openssl installato sul sistema

set -e  # Interrompi su qualsiasi errore

# ─── Argomenti ────────────────────────────────────────────────────────────────

CERTS_DIR="${1:-certs}"
NODES="${2:-highway-01 highway-02 highway-03 urban-01 urban-02 urban-03 residential-01 residential-02 residential-03 corporate-01 corporate-02 corporate-03 aggregator auditor ids fl-admin mqtt-broker}"

# Validità certificati in giorni
CERT_DAYS=365

# Informazioni CA
CA_SUBJECT="/C=IT/ST=Tuscany/L=Lucca/O=ChargeShield-FL/CN=ChargeShield-CA"

echo "────────────────────────────────────────────"
echo " ChargeShield-FL — Generazione certificati mTLS"
echo "────────────────────────────────────────────"

# ─── Verifica openssl ─────────────────────────────────────────────────────────

if ! command -v openssl &> /dev/null; then
    echo "ERRORE: openssl non trovato. Installalo con:"
    echo "  brew install openssl  (macOS)"
    echo "  apt install openssl   (Debian/Ubuntu)"
    exit 1
fi

# ─── Crea directory base ──────────────────────────────────────────────────────

mkdir -p "${CERTS_DIR}/ca"
echo "→ Directory certificati: ${CERTS_DIR}/"

# ─── Genera Certificate Authority (CA) ───────────────────────────────────────
# La CA è il trust anchor del sistema.
# Tutti i nodi devono avere la CA pubblica per verificare
# i certificati degli altri nodi.

echo "→ Generazione CA..."

# Chiave privata CA (4096 bit — solo per la CA)
openssl genrsa -out "${CERTS_DIR}/ca/ca.key" 4096 2>/dev/null

# Certificato CA self-signed
openssl req -new -x509 \
    -key "${CERTS_DIR}/ca/ca.key" \
    -out "${CERTS_DIR}/ca/ca.crt" \
    -days "${CERT_DAYS}" \
    -subj "${CA_SUBJECT}" \
    2>/dev/null

echo "✓ CA generata: ${CERTS_DIR}/ca/ca.crt"

# ─── Genera certificati per ogni nodo ────────────────────────────────────────
# Ogni nodo riceve:
# - node.key  → chiave privata (solo del nodo, non condivisa)
# - node.crt  → certificato firmato dalla CA
# - ca.crt    → CA pubblica (per verificare gli altri nodi)
#
# In mTLS ogni nodo presenta il proprio certificato
# e verifica quello dell'interlocutore tramite la CA condivisa.

for NODE in ${NODES}; do
    echo "→ Generazione certificato per ${NODE}..."

    NODE_DIR="${CERTS_DIR}/${NODE}"
    mkdir -p "${NODE_DIR}"

    # Chiave privata del nodo (2048 bit)
    openssl genrsa -out "${NODE_DIR}/node.key" 2048 2>/dev/null

    # Certificate Signing Request (CSR)
    # CN = node_id — usato per identificare il nodo durante mTLS
    openssl req -new \
        -key "${NODE_DIR}/node.key" \
        -out "${NODE_DIR}/node.csr" \
        -subj "/C=IT/ST=Tuscany/L=Lucca/O=ChargeShield-FL/CN=${NODE}" \
        2>/dev/null

    # Firma il certificato con la CA
    openssl x509 -req \
        -in "${NODE_DIR}/node.csr" \
        -CA "${CERTS_DIR}/ca/ca.crt" \
        -CAkey "${CERTS_DIR}/ca/ca.key" \
        -CAcreateserial \
        -out "${NODE_DIR}/node.crt" \
        -days "${CERT_DAYS}" \
        2>/dev/null

    # Copia la CA pubblica nella directory del nodo
    # (necessaria per verificare i certificati degli altri nodi)
    cp "${CERTS_DIR}/ca/ca.crt" "${NODE_DIR}/ca.crt"

    # Rimuovi il CSR — non serve più dopo la firma
    rm "${NODE_DIR}/node.csr"

    echo "  ✓ ${NODE}: node.crt, node.key, ca.crt"
done

# ─── Imposta permessi sicuri ──────────────────────────────────────────────────
# Le chiavi private devono essere leggibili solo dal proprietario

chmod 600 "${CERTS_DIR}/ca/ca.key"
find "${CERTS_DIR}" -name "*.key" -exec chmod 600 {} \;
find "${CERTS_DIR}" -name "*.crt" -exec chmod 644 {} \;

echo "────────────────────────────────────────────"
NODE_COUNT=$(echo ${NODES} | wc -w | tr -d ' ')
echo "✓ Certificati generati per ${NODE_COUNT} nodi"
echo "✓ Validità: ${CERT_DAYS} giorni"
echo "✓ Directory: ${CERTS_DIR}/"
echo ""
echo "IMPORTANTE: aggiungi certs/ca/ca.key al .gitignore"
echo "            Non committare mai le chiavi private!"
echo "────────────────────────────────────────────"
