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
	@echo "  make experiment            Esegui esperimento FedMIA (config default)"
	@echo "  make experiment-smoke      Smoke test (5 round, no-DP, n_shadow=4) — verifica pipeline"
	@echo "  make experiment-nodp       Baseline no-DP (10 round, n_shadow=8) — Yeom+Shadow+LiRA"
	@echo "  make experiment-dp         Con DP (10 round, ε=EPS, n_shadow=8) — Yeom+Shadow+LiRA"
	@echo "  [Confronto automatico in foglio 'Attack Comparison' dell'Excel]"
	@echo "  make experiment-sweep      Sweep epsilon 0.1→5.0 (100 round) [legacy]"
	@echo "  make experiment-full-sweep Sweep rounds×epsilon (100-1000 × 0.1-5.0) — crea experiments/exp{N}/"
	@echo "  make experiment-byzantine-sweep Byzantine sweep (5 seed × 5 epsilon) — IDS validation"
	@echo "  make experiment-dry        Dry run (verifica config e dataset)"
	@echo "  make install           Installa dipendenze runtime (torch, numpy, ecc.) — nuovo ambiente"
	@echo "  make install-dev       Installa runtime + dev tools (pytest, ruff, mypy)"
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

# LEGACY: non isola i sweep (nessuna --sweep-dir). Usare experiment-full-sweep.
.PHONY: experiment-sweep
experiment-sweep:
	@echo "→ Epsilon sweep (legacy): 0.1, 0.5, 1.0, 2.0, 5.0 — usa experiment-full-sweep per sweep isolati"
	@mkdir -p $(EXPERIMENTS)
	@for eps in 0.1 0.5 1.0 2.0 5.0; do \
		echo "=== epsilon=$$eps ==="; \
		$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
			--config config/experiment.yaml --epsilon $$eps --rounds 100; \
	done
	@echo "✓ Sweep completato — risultati in: $(EXPERIMENTS)/"

# Sweep completo: rounds × epsilon → heat map per il paper
# rounds ∈ {100, 200, 500, 1000} × epsilon ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
# Stima: ~36h su CPU sequenziale (20 config × ~1.8h media; molto più veloce su Mac M-series nativo)
.PHONY: experiment-full-sweep
# Sweep completo rounds × epsilon con directory numerata automatica.
# Ogni esecuzione crea experiments/exp{N}/ con i JSON e exp{N}.xlsx separati.
# Non mischia mai risultati di sweep distinti.
experiment-full-sweep:
	@mkdir -p $(EXPERIMENTS); \
	SWEEP_NUM=$$(find $(EXPERIMENTS) -maxdepth 1 -type d -name 'exp[0-9]*' 2>/dev/null | wc -l | tr -d ' '); \
	SWEEP_NUM=$$((SWEEP_NUM + 1)); \
	SWEEP_DIR=$(EXPERIMENTS)/exp$$SWEEP_NUM; \
	mkdir -p "$$SWEEP_DIR"; \
	LOG="$$SWEEP_DIR/sweep_log.txt"; \
	echo "→ Full sweep #$$SWEEP_NUM: rounds × epsilon — $$SWEEP_DIR" | tee "$$LOG"; \
	for rounds in 100 200 500 1000; do \
		for eps in 0.1 0.5 1.0 2.0 5.0; do \
			echo "=== rounds=$$rounds epsilon=$$eps ===" | tee -a "$$LOG"; \
			$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
				--config config/experiment.yaml \
				--epsilon $$eps \
				--rounds $$rounds \
				--sweep-dir "$$SWEEP_DIR" 2>&1 | tee -a "$$LOG"; \
		done; \
	done; \
	echo "✓ Full sweep #$$SWEEP_NUM completato — $$SWEEP_DIR/" | tee -a "$$LOG"

# IDS Validation: Byzantine gradient scaling sweep — SEPARATO dai test MIA.
#
# IMPORTANTE: questo sweep NON è un test di privacy risk.
# Serve SOLO a validare che l'IDS (Krum) rilevi correttamente nodi malevoli.
# I risultati vanno in experiments/ids_validation/ — MAI in experiments/exp{N}/.
# La sequenza exp{N} è riservata esclusivamente agli esperimenti MIA puliti
# (nessun attacco attivo, DP variabile, per misurare AUC-ROC vs epsilon).
#
# Attacco: highway moltiplica pesi ×10 → Krum score ≈4.0 > threshold 1.5 → alert.
# 25 run: 5 seed × 5 epsilon per robustezza statistica della detection.
IDS_VALIDATION_DIR := $(EXPERIMENTS)/ids_validation
.PHONY: experiment-byzantine-sweep
experiment-byzantine-sweep:
	@mkdir -p $(IDS_VALIDATION_DIR); \
	LOG="$(IDS_VALIDATION_DIR)/sweep_log.txt"; \
	echo "→ IDS validation sweep (highway ×10, 5 seed × 5 epsilon)" | tee "$$LOG"; \
	echo "  NOTA: risultati in experiments/ids_validation/ — separati da exp{N} (MIA)" | tee -a "$$LOG"; \
	for seed in 42 123 456 789 1234; do \
		for eps in 0.1 0.5 1.0 2.0 5.0; do \
			echo "=== seed=$$seed epsilon=$$eps ===" | tee -a "$$LOG"; \
			$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
				--config config/experiment.yaml \
				--epsilon $$eps \
				--rounds 100 \
				--seed $$seed \
				--byzantine \
				--byzantine-node highway \
				--scale-factor 10 \
				--sweep-dir "$(IDS_VALIDATION_DIR)" 2>&1 | tee -a "$$LOG"; \
		done; \
	done; \
	echo "✓ IDS validation sweep completato — $(IDS_VALIDATION_DIR)/" | tee -a "$$LOG"

# ─── Sequenza sperimentale raccomandata per DSN 2027 ─────────────────────────
#
# IMPORTANTE: ogni run esegue SEMPRE tutti e tre gli attacchi in parallelo:
#   • Yeom 2018         (loss-based MIA sul modello globale — baseline debole)
#   • Shadow MIA        (calibrated attack sul modello globale — medio)
#   • LiRA              (server-side, raw_updates PRE-aggregazione — primario ★)
# Il confronto avviene automaticamente nel foglio "Attack Comparison" dell'Excel.
# Non serve eseguire run separati per Yeom vs LiRA: stesse condizioni, stesso JSON.
#
# Sequenza raccomandata:
#   Passo 1: make experiment-smoke    → verifica pipeline (5 round)
#   Passo 2: make experiment-nodp     → baseline no-DP (10 round): AUC > 0.55?
#   Passo 3: make experiment-dp       → con DP (10 round, ε=1.0): DP sopprime l'AUC?
#   Passo 4: make experiment-full-sweep → sweep completo rounds × epsilon (paper)
#
# EPS ?= 1.0   — override epsilon: make experiment-dp EPS=0.5

EPS ?= 1.0
N_SHADOW ?= 8

# Smoke test: 5 round, no-DP, n_shadow=4 — solo per verificare che la pipeline giri.
# AUC non interpretabile con 5 round (troppo poco training).
.PHONY: experiment-smoke
experiment-smoke:
	@echo "→ Smoke test: 5 round, no-DP, n_shadow=4 (Yeom + Shadow + LiRA)..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--no-dp \
		--rounds 5 \
		--n-shadow 4
	@echo "✓ Smoke test completato — se nessun errore, pipeline OK"

# Baseline no-DP: esegue Yeom + Shadow + LiRA senza rumore DP (σ=0).
# Obiettivo: verificare che ALMENO UN attacco (preferibilmente LiRA) ottenga AUC > 0.55.
# AUC > 0.55 → Scenario A confermato → il segnale di membership esiste → ha senso testare DP.
# Se AUC ≈ 0.5 anche qui → modello non memorizza → serve più training o dati diversi.
.PHONY: experiment-nodp
experiment-nodp:
	@echo "→ Baseline no-DP (σ=0, 10 round, n_shadow=$(N_SHADOW)) — Yeom + Shadow + LiRA..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--no-dp \
		--rounds 10 \
		--n-shadow $(N_SHADOW)
	@echo "✓ Baseline no-DP completato — controlla Attack Comparison nell'Excel"

# Baseline con DP: esegue Yeom + Shadow + LiRA con DP attivo (ε=EPS, default 1.0).
# Confrontare ogni attacco (Yeom, Shadow, LiRA) con il corrispondente no-DP.
# Se LiRA no-DP > 0.55 ma LiRA DP ≈ 0.50 → DP sopprime LiRA → claim paper valido.
.PHONY: experiment-dp
experiment-dp:
	@echo "→ Esperimento con DP (ε=$(EPS), 10 round, n_shadow=$(N_SHADOW)) — Yeom + Shadow + LiRA..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--epsilon $(EPS) \
		--rounds 10 \
		--n-shadow $(N_SHADOW)
	@echo "✓ Esperimento DP completato — confronta con no-DP nel foglio Attack Comparison"


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

# Installa SOLO le dipendenze runtime (torch, numpy, sklearn, ecc.)
# Usare su Mac nativo o in nuovi ambienti prima di eseguire esperimenti.
# torch>=2.0 è dichiarato in pyproject.toml [project.dependencies].
.PHONY: install
install:
	@echo "→ Installazione dipendenze runtime (torch, numpy, sklearn, pandas, openpyxl, pyyaml)..."
	pip install -e "." --break-system-packages
	@echo "✓ Runtime installato. Per dev tools (pytest, ruff, mypy): make install-dev"

# Installa dipendenze runtime + dev tools (pytest, ruff, mypy).
# Equivale a: pip install -e ".[dev]"  (include torch e tutti i runtime deps).
.PHONY: install-dev
install-dev:
	@echo "→ Installazione dipendenze (runtime + dev)..."
	pip install -e ".[dev]" --break-system-packages
	@echo "✓ Installato: torch, numpy, scikit-learn, pandas, openpyxl, pyyaml, pytest, pytest-cov, ruff, mypy"

.PHONY: test-coverage
test-coverage: install-dev
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
