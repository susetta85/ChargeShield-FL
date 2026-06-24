# Makefile — ChargeShield-FL
# Automazione build, deploy, test e certificati mTLS
#
# Comandi disponibili:
#   make help        → mostra questo help
#   make build       → build tutte le immagini Docker
#   make deploy      → deploy topologia Containerlab
#   make destroy     → teardown topologia
#   make certs       → genera certificati mTLS per tutti i nodi
#   make test        → esegui unit test
#   make lint        → controlla qualità del codice
#   make experiment  → lancia un round FL completo
#   make clean       → rimuovi artefatti temporanei

# ─── Variabili ────────────────────────────────────────────────────────────────

PROJECT     := chargeshield-fl
VERSION     := 0.3.0
TOPOLOGY    := containerlab/topology.clab.yml
CERTS_DIR   := certs
SCRIPTS_DIR := scripts
PYTHON      := python3
PYTEST      := python3 -m pytest

# Nodi per cui generare certificati mTLS
NODES := \
    highway-01 highway-02 highway-03 \
    urban-01 urban-02 urban-03 \
    residential-01 residential-02 residential-03 \
    corporate-01 corporate-02 corporate-03 \
    aggregator auditor ids fl-admin

# Immagini Docker da buildare
IMAGES := \
    charging-node \
    aggregator \
    auditor \
    ids \
    fl-admin

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "ChargeShield-FL — Makefile"
	@echo "─────────────────────────────────────────"
	@echo "  make build       Build tutte le immagini Docker"
	@echo "  make deploy      Deploy topologia Containerlab"
	@echo "  make destroy     Teardown topologia"
	@echo "  make certs       Genera certificati mTLS"
	@echo "  make test        Esegui unit test"
	@echo "  make lint        Controlla qualità del codice"
	@echo "  make experiment  Lancia un round FL completo"
	@echo "  make clean       Rimuovi artefatti temporanei"
	@echo ""

# ─── Test ─────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	@echo "→ Esecuzione unit test..."
	$(PYTEST) tests/ -v

.PHONY: test-coverage
test-coverage:
	@echo "→ Esecuzione test con coverage..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

# ─── Lint ─────────────────────────────────────────────────────────────────────

.PHONY: lint
lint:
	@echo "→ Controllo qualità codice con ruff..."
	ruff check src/ tests/

# ─── Certificati mTLS ─────────────────────────────────────────────────────────

.PHONY: certs
certs:
	@echo "→ Generazione certificati mTLS..."
	@bash $(SCRIPTS_DIR)/generate_certs.sh $(CERTS_DIR) "$(NODES)"
	@echo "✓ Certificati generati in $(CERTS_DIR)/"

.PHONY: clean-certs
clean-certs:
	@echo "→ Rimozione certificati..."
	rm -rf $(CERTS_DIR)/
	@echo "✓ Certificati rimossi"

# ─── Docker ───────────────────────────────────────────────────────────────────

.PHONY: build
build:
	@echo "→ Build immagini Docker..."
	@for image in $(IMAGES); do \
		echo "  Building chargeshield/$$image:latest ..."; \
		docker build \
			-t chargeshield/$$image:latest \
			-f docker/$$image/Dockerfile \
			.; \
	done
	@echo "✓ Build completato"

.PHONY: push
push:
	@echo "→ Push immagini Docker..."
	@for image in $(IMAGES); do \
		docker push chargeshield/$$image:latest; \
	done

# ─── Containerlab ─────────────────────────────────────────────────────────────

.PHONY: deploy
deploy: certs build
	@echo "→ Deploy topologia Containerlab..."
	containerlab deploy -t $(TOPOLOGY)
	@echo "✓ Topologia attiva"

.PHONY: destroy
destroy:
	@echo "→ Teardown topologia Containerlab..."
	containerlab destroy -t $(TOPOLOGY)
	@echo "✓ Topologia rimossa"

.PHONY: inspect
inspect:
	@echo "→ Stato topologia..."
	containerlab inspect -t $(TOPOLOGY)

# ─── Esperimento FL ───────────────────────────────────────────────────────────

.PHONY: experiment
experiment:
	@echo "→ Avvio esperimento FL..."
	$(PYTHON) scripts/run_experiment.py \
		--config config/flare.yaml \
		--nodes config/nodes.yaml \
		--output experiments/
	@echo "✓ Esperimento completato — risultati in experiments/"

# ─── Clean ────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	@echo "→ Pulizia artefatti temporanei..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage
	@echo "✓ Pulizia completata"

.PHONY: clean-all
clean-all: clean clean-certs destroy
	@echo "✓ Pulizia completa"
