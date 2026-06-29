# Makefile — ChargeShield-FL
# Sprint 5: OrbStack + Containerlab + NVFLARE 2.7.2
#
# Flusso tipico:
#   1. make build       → build immagine Docker con NVFLARE
#   2. make provision   → genera workspace NVFLARE (una volta sola)
#   3. make deploy      → deploya topologia Containerlab
#   4. make experiment  → esegui esperimento FedMIA
#
# Altri target:
#   make test           → unit test
#   make experiment-sweep → sweep epsilon per privacy/utility trade-off
#   make destroy        → rimuovi topologia
#   make clean          → rimuovi artefatti temporanei

# ─── Variabili ────────────────────────────────────────────────────────────────
PROJECT      := chargeshield-fl
VERSION      := 0.5.0
IMAGE        := chargeshield-fl:latest
TOPOLOGY     := containerlab/topology.clab.yml
PROJECT_YML  := nvflare/project.yml
WORKSPACE    := nvflare/workspace
SCRIPTS_DIR  := scripts
PYTHON       := python3
PYTEST       := python3 -m pytest
NVFLARE      := nvflare
CLAB         := sudo containerlab
EXPERIMENTS  := experiments

# ─── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "ChargeShield-FL v$(VERSION) — Makefile"
	@echo "════════════════════════════════════════"
	@echo "  make build             Build immagine Docker con NVFLARE 2.7.2"
	@echo "  make provision         Genera workspace NVFLARE (una volta sola)"
	@echo "  make deploy            Deploy topologia Containerlab"
	@echo "  make destroy           Rimuovi topologia"
	@echo "  make status            Stato container"
	@echo "  make logs              Log server + highway"
	@echo "  make experiment        Esegui esperimento FedMIA (config default)"
	@echo "  make experiment-sweep  Sweep epsilon 0.1→5.0 (50 round)"
	@echo "	 make experiment-full-sweep	Sweep roundxepsilon (100-1000 x 0.1-5.0)"
	@echo "  make experiment-dry    Dry run (verifica config e dataset)"
	@echo "  make test              Tutti i test unitari"
	@echo "  make test-sprint4      Solo Sprint 4"
	@echo "  make test-sprint5      Solo Sprint 5"
	@echo "  make lint              Controllo qualità codice"
	@echo "  make clean             Rimuovi __pycache__ e artefatti"
	@echo "  make clean-workspace   Rimuovi workspace NVFLARE"
	@echo "  make clean-experiments Rimuovi risultati esperimenti"
	@echo ""

# ─── Build ────────────────────────────────────────────────────────────────────
.PHONY: build
build:
	@echo "→ Building $(IMAGE)..."
	docker build -f Dockerfile.flare -t $(IMAGE) .
	@echo "✓ Build completato: $(IMAGE)"

# ─── NVFLARE Provisioning ─────────────────────────────────────────────────────
# Genera workspace con certificati mTLS e startup scripts.
# Eseguire UNA SOLA VOLTA dopo il primo build.
.PHONY: provision
provision:
	@echo "→ NVFLARE provisioning..."
	@mkdir -p $(WORKSPACE)
	$(NVFLARE) provision -p $(PROJECT_YML) -w $(WORKSPACE)
	@echo "✓ Workspace generato in: $(WORKSPACE)/chargeshield_fl/prod_00/"

# ─── Containerlab ─────────────────────────────────────────────────────────────
.PHONY: deploy
deploy:
	@echo "→ Deploy topologia ChargeShield-FL..."
	$(CLAB) deploy -t $(TOPOLOGY) --reconfigure
	@echo "✓ Topologia attiva"

.PHONY: destroy
destroy:
	@echo "→ Teardown topologia..."
	$(CLAB) destroy -t $(TOPOLOGY) --cleanup
	@echo "✓ Topologia rimossa"

.PHONY: status
status:
	@docker ps --filter "name=clab-chargeshield" \
		--format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

.PHONY: logs
logs:
	@echo "=== aggregator ==="
	docker logs clab-chargeshield-fl-aggregator --tail 50
	@echo "=== highway ==="
	docker logs clab-chargeshield-fl-highway --tail 20

# ─── Esperimento FL ───────────────────────────────────────────────────────────
.PHONY: experiment
experiment:
	@echo "→ Avvio esperimento FedMIA..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml
	@echo "✓ Risultati salvati in: $(EXPERIMENTS)/"

.PHONY: experiment-sweep
experiment-sweep:
	@echo "→ Epsilon sweep: 0.1, 0.5, 1.0, 2.0, 5.0"
	@mkdir -p $(EXPERIMENTS)
	@for eps in 0.1 0.5 1.0 2.0 5.0; do \
		echo "=== epsilon=$$eps ==="; \
		$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
			--epsilon $$eps --rounds 100; \
	done
	@echo "✓ Sweep completato — risultati in: $(EXPERIMENTS)/"

# Sweep completo: rounds × epsilon → heat map per il paper
# rounds ∈ {100, 200, 500, 1000} × epsilon ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
# Stima: ~8-12 ore su CPU
.PHONY: experiment-full-sweep
experiment-full-sweep:
	@echo "→ Full sweep: rounds × epsilon"
	@mkdir -p $(EXPERIMENTS)
	@for rounds in 100 200 500 1000; do \
		for eps in 0.1 0.5 1.0 2.0 5.0; do \
			echo "=== rounds=$$rounds epsilon=$$eps ==="; \
			$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
				--epsilon $$eps --rounds $$rounds; \
		done; \
	done
	@echo "✓ Full sweep completato — risultati in: $(EXPERIMENTS)/"

.PHONY: experiment-dry
experiment-dry:
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml --dry-run

# ─── Test ─────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	@echo "→ Esecuzione unit test..."
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-sprint4
test-sprint4:
	$(PYTEST) tests/test_sprint4.py -v --tb=short

.PHONY: test-sprint5
test-sprint5:
	$(PYTEST) tests/test_sprint5.py -v --tb=short

.PHONY: test-coverage
test-coverage:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

# ─── Lint ─────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	@echo "→ Controllo qualità codice..."
	ruff check src/ tests/

# ─── Clean ────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo "→ Pulizia artefatti..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage
	@echo "✓ Pulizia completata"

.PHONY: clean-workspace
clean-workspace:
	@echo "→ Rimozione workspace NVFLARE..."
	rm -rf $(WORKSPACE)
	@echo "✓ Workspace rimosso — ri-esegui 'make provision'"

.PHONY: clean-experiments
clean-experiments:
	@echo "→ Rimozione risultati esperimenti..."
	rm -rf $(EXPERIMENTS)
	@echo "✓ Esperimenti rimossi"

.PHONY: clean-all
clean-all: clean clean-workspace destroy
	@echo "✓ Pulizia completa"

# ─── All ──────────────────────────────────────────────────────────────────────
.PHONY: all
all: build provision deploy experiment
