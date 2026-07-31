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
	@echo "  make experiment-smoke       Smoke test (5 round, no-DP, seed=SEED, n_shadow=2)"
	@echo "  make experiment-nodp        Baseline no-DP singolo seed (10 round)"
	@echo "  make experiment-dp          Con DP singolo seed (10 round, ε=EPS)"
	@echo "  make experiment-nodp-sweep  no-DP multi-seed (SEEDS='42 123 456 789 1234') → mean±std"
	@echo "  make experiment-dp-sweep    DP multi-seed (ε=EPS, SEEDS) → mean±std vs no-DP"
	@echo "  [Excel 11 sheet: Attack Comparison + per-attacco + Seed Aggregation mean±std]"
	@echo "  make experiment-sweep      Sweep epsilon 0.1→5.0 (100 round) [legacy]"
	@echo "  make experiment-full-sweep Sweep rounds×epsilon (100-1000 × 0.1-5.0) — crea experiments/full-sweep{N}/"
	@echo "  make experiment-byzantine-sweep Byzantine sweep (5 seed × 5 epsilon) — IDS validation"
	@echo "  make experiment-dry        Dry run (verifica config e dataset)"
	@echo "  make install           Installa dipendenze runtime (torch, numpy, ecc.) — nuovo ambiente"
	@echo "  make install-dev       Installa runtime + dev tools (pytest, ruff, mypy)"
	@echo "  make install-flare     Installa torch + nvflare==2.7.2 (extra 'flare' di pyproject.toml)"
	@echo "  make nvflare-sim-smoke Simulatore NVFLARE, 1 client (caltech) — smoke test round-trip"
	@echo "  make nvflare-sim       Simulatore NVFLARE, 3 siti reali (caltech, jpl, office1)"
	@echo "  make clean-nvflare-sim Rimuovi workspace del simulatore NVFLARE"
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

# NOTA (2026-07-22, review indipendente fresh-pass): target non ancora
# funzionale indipendentemente da questo fix — containerlab/topology.clab.yml
# non esiste ancora (l'integrazione Containerlab resta un task aperto), quindi
# nessuno di questi container esiste oggi. Nome sito aggiornato comunque da
# "highway" (schema fittizio a 4 cluster, superato) a "caltech" (uno dei 3
# siti reali ACN-Data, Sprint 10) per coerenza quando quel task verrà ripreso.
.PHONY: logs
logs:
	@echo "=== aggregator ==="
	docker logs clab-chargeshield-fl-aggregator --tail 50
	@echo "=== caltech ==="
	docker logs clab-chargeshield-fl-caltech --tail 20

# ─── Controllo dipendenze ─────────────────────────────────────────────────────
# _check-deps: verifica che torch (e le altre dipendenze) siano installate.
# Fix 2026-07-24 (richiesta esplicita dell'utente — "vorrei che l'ambiente
# venisse configurato all'avvio dalla macchina"): prima si limitava a
# STAMPARE un avviso e uscire con errore, lasciando all'utente il compito di
# ricordarsi di lanciare 'make install' a mano prima di ogni sessione di
# lavoro su una macchina nuova o dopo una pulizia dell'ambiente. Ora, se le
# dipendenze mancano, le installa da solo chiamando 'make install' — nessun
# passo manuale separato da ricordare prima dei run reali. Se le dipendenze
# ci sono già, l'import silenzioso non fa nulla (nessun overhead su ogni
# singolo esperimento lanciato in sequenza).
# Tutti i target sperimentali dipendono da questo.
.PHONY: _check-deps
_check-deps:
	@$(PYTHON) -c "import torch, numpy, sklearn, openpyxl, yaml" 2>/dev/null || \
		(echo "" && \
		 echo "→ Dipendenze mancanti (torch/numpy/sklearn/openpyxl/yaml) — installazione automatica..." && \
		 $(MAKE) install && \
		 echo "✓ Dipendenze installate — proseguo con il target richiesto." && \
		 echo "") || \
		(echo "✗ Installazione automatica fallita (make install) — impossibile proseguire." >&2; exit 1)

# ─── Esperimento FL ───────────────────────────────────────────────────────────
.PHONY: experiment
experiment: _check-deps
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
# Ogni esecuzione crea experiments/full-sweep{N}/ con i JSON e full-sweep{N}.xlsx separati.
# Nome prefissato per tipo (full-sweep / nodp-sweep / dp-sweep) — contatori indipendenti,
# così i nomi restano leggibili anche mescolando i tre tipi di sweep (fix 2026-07-21,
# prima tutti e tre condividevano lo stesso contatore generico exp{N}, ambiguo).
experiment-full-sweep: _check-deps
	@mkdir -p $(EXPERIMENTS); \
	SWEEP_NUM=$$(find $(EXPERIMENTS) -maxdepth 1 -type d -name 'full-sweep[0-9]*' 2>/dev/null | wc -l | tr -d ' '); \
	SWEEP_NUM=$$((SWEEP_NUM + 1)); \
	SWEEP_DIR=$(EXPERIMENTS)/full-sweep$$SWEEP_NUM; \
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
# I risultati vanno in experiments/ids_validation/ — MAI in experiments/nodp-sweep{N}/,
# dp-sweep{N}/ o full-sweep{N}/ (fix 2026-07-22: nome aggiornato, il vecchio
# schema generico "exp{N}" non esiste più — vedi nota sotto experiment-full-sweep).
# Le sequenze nodp-sweep{N}/dp-sweep{N}/full-sweep{N} sono riservate esclusivamente
# agli esperimenti MIA puliti (nessun attacco attivo, DP variabile, per misurare
# AUC-ROC vs epsilon).
#
# Attacco: synthetic_1 (client fittizio, mai un sito reale — vedi
# inject_synthetic_client_indices() in run_experiments.py, n=5 = 3 siti reali
# + 2 sintetici, garanzia Krum n≥2f+3 con f=1) moltiplica pesi ×10 → Krum
# score elevato → alert. FedMIA/Shadow/LiRA vengono saltati automaticamente
# per questo sweep (byzantine_attack.enabled=true): non è una fonte di numeri
# di privacy, solo di validazione IDS/Krum.
# 25 run: 5 seed × 5 epsilon per robustezza statistica della detection.
IDS_VALIDATION_DIR := $(EXPERIMENTS)/ids_validation
.PHONY: experiment-byzantine-sweep
experiment-byzantine-sweep: _check-deps
	@mkdir -p $(IDS_VALIDATION_DIR); \
	LOG="$(IDS_VALIDATION_DIR)/sweep_log.txt"; \
	echo "→ IDS validation sweep (synthetic_1 ×10, 5 seed × 5 epsilon)" | tee "$$LOG"; \
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
				--byzantine-node synthetic_1 \
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

# N_SHADOW alzato da 8 a 16 (2026-07-23): il round 8 dello sweep ε=0.5/0.1
# del 2026-07-22 (seed=42) mostrava un gap LiRA invertito, causato da
# instabilità di calibrazione con un ensemble di soli 8 shadow/cluster
# (confermato non essere un effetto reale legato a ε: seed=123 sullo stesso
# ε=0.5 non mostra l'inversione — vedi esperimenti/anomaly_check). Più
# shadow stabilizzano la stima IN/OUT round per round, riducendo il rischio
# di ripetere questo tipo di instabilità sui prossimi sweep "seri".
EPS      ?= 1.0
N_SHADOW ?= 16
SEED     ?= 42
# SEEDS: 5 seed per mean±std (minimo DSN 2027); 10 seed per test Wilcoxon di significatività.
# Override: make experiment-nodp-sweep SEEDS="42 123 456"
SEEDS    ?= 42 123 456 789 1234

# Lock anti-concorrenza per gli sweep multi-seed (2026-07-31, trovato un caso reale:
# due 'make experiment-*-sweep' avviati per errore in parallelo hanno fatto
# competere due training FL/LiRA per la stessa CPU/RAM sulla stessa macchina —
# uno dei due e' stato terminato silenziosamente dall'OS (probabile OOM kill: il
# processo e' sparito senza alcun traceback Python nel suo stesso log, senza
# produrre risultati). Le run FL/LiRA sono deliberatamente CPU/RAM-intensive
# (50 epoche/round, fino a 16 shadow model per cluster) — due insieme sulla
# stessa macchina non sono solo piu' lente, rischiano di far fallire una delle
# due senza alcun errore visibile. Questo lock blocca un secondo sweep finche'
# il primo non e' finito (o la sua lock e' stale, cioe' il PID che la deteneva
# non esiste piu' — es. macchina riavviata, processo killato manualmente).
SWEEP_LOCK := $(EXPERIMENTS)/.sweep_running.lock

.PHONY: _sweep_lock
_sweep_lock:
	@mkdir -p $(EXPERIMENTS); \
	if [ -f "$(SWEEP_LOCK)" ]; then \
		OLD_PID=$$(head -1 "$(SWEEP_LOCK)" | cut -d' ' -f1); \
		OLD_INFO=$$(cat "$(SWEEP_LOCK)"); \
		if [ -n "$$OLD_PID" ] && kill -0 "$$OLD_PID" 2>/dev/null; then \
			echo "✗ Un altro sweep e' gia' in corso (PID $$OLD_PID): $$OLD_INFO"; \
			echo "  Due sweep FL/LiRA in parallelo competono per CPU/RAM sulla stessa"; \
			echo "  macchina e possono causare un OOM kill silenzioso (visto il 2026-07-31"; \
			echo "  su dp-sweep3 — processo sparito senza traceback, nessun risultato)."; \
			echo "  Attendi che finisca, oppure interrompilo (kill $$OLD_PID) se e' bloccato,"; \
			echo "  prima di lanciarne un altro. Se sei SICURO che non stia girando nulla"; \
			echo "  (es. la macchina e' stata riavviata), rimuovi il lock a mano:"; \
			echo "    rm $(SWEEP_LOCK)"; \
			exit 1; \
		else \
			echo "⚠ Lock stale trovato ($$OLD_INFO, PID non piu' attivo) — lo rimuovo e procedo."; \
			rm -f "$(SWEEP_LOCK)"; \
		fi; \
	fi

# Smoke test: 5 round, no-DP, n_shadow=2, shadow-epochs-cap=20 — verifica pipeline rapida.
# --shadow-epochs-cap 20: LiRA usa max 20 epoche per shadow model invece di 250 (formula default).
#   Riduce il tempo da ~30 min a ~3-5 min. NON usare nei run sperimentali reali.
# --sweep-dir experiments/smoke: isola i risultati — non contamina exp{N}/ né l'Excel globale.
# AUC non interpretabile (troppo poco training + shadow sottoadatti) — solo "la pipeline gira?".
SMOKE_DIR := $(EXPERIMENTS)/smoke
.PHONY: experiment-smoke
experiment-smoke: _check-deps
	@echo "→ Smoke test: 5 round, no-DP, seed=$(SEED), n_shadow=2, shadow-cap=20..."
	@mkdir -p $(SMOKE_DIR)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--no-dp \
		--rounds 5 \
		--seed $(SEED) \
		--n-shadow 2 \
		--shadow-epochs-cap 20 \
		--sweep-dir $(SMOKE_DIR)
	@echo "✓ Smoke test completato — risultati in $(SMOKE_DIR)/; se no errori, pipeline OK"

# Baseline no-DP: seed singolo. Usare experiment-nodp-sweep per 5 seed (mean±std).
.PHONY: experiment-nodp
experiment-nodp: _check-deps
	@echo "→ Baseline no-DP (σ=0, 10 round, seed=$(SEED), n_shadow=$(N_SHADOW)) — Yeom + Shadow + LiRA..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--no-dp \
		--rounds 10 \
		--seed $(SEED) \
		--n-shadow $(N_SHADOW)
	@echo "✓ Baseline no-DP completato — controlla Attack Comparison nell'Excel"

# Con DP: seed singolo. Usare experiment-dp-sweep per 5 seed (mean±std).
# Placement DP-FedAvg (default, McMahan 2017) — clip+noise per client prima di FedAvg.
.PHONY: experiment-dp
experiment-dp: _check-deps
	@echo "→ Esperimento con DP-FedAvg (ε=$(EPS), 10 round, seed=$(SEED), n_shadow=$(N_SHADOW)) — Yeom + Shadow + LiRA..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--epsilon $(EPS) \
		--rounds 10 \
		--seed $(SEED) \
		--n-shadow $(N_SHADOW) \
		--dp-mode dp-fedavg
	@echo "✓ Esperimento DP-FedAvg completato — confronta con no-DP nel foglio Attack Comparison"

# Central DP (2026-07-22, vedi docs/CaseStudies.md §2.4.3): client clippano SENZA
# rumorizzare, il server aggrega pulito e aggiunge UN SOLO rumore all'aggregato.
# Atteso: LiRA sul singolo update NON mostra soppressione, a nessun ε (non un bug).
.PHONY: experiment-central-dp
experiment-central-dp: _check-deps
	@echo "→ Esperimento con Central DP (ε=$(EPS), 10 round, seed=$(SEED), n_shadow=$(N_SHADOW))..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--epsilon $(EPS) \
		--rounds 10 \
		--seed $(SEED) \
		--n-shadow $(N_SHADOW) \
		--dp-mode central
	@echo "✓ Esperimento Central DP completato — confronta LiRA con dp-fedavg: atteso NESSUNA soppressione"

# Local DP (2026-07-22): stesso meccanismo per-client di dp-fedavg, ma server/IDS
# non vede mai l'update raw — run_ids() degrada (atteso, vedi docstring run_ids()).
.PHONY: experiment-local-dp
experiment-local-dp: _check-deps
	@echo "→ Esperimento con Local DP (ε=$(EPS), 10 round, seed=$(SEED), n_shadow=$(N_SHADOW))..."
	@mkdir -p $(EXPERIMENTS)
	$(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
		--config config/experiment.yaml \
		--epsilon $(EPS) \
		--rounds 10 \
		--seed $(SEED) \
		--n-shadow $(N_SHADOW) \
		--dp-mode local
	@echo "✓ Esperimento Local DP completato — IDS degradato per design, vedi docstring run_ids()"

# ─── Multi-seed sweep per DSN 2027 ───────────────────────────────────────────
#
# METODOLOGIA: ogni attacco (Yeom, Shadow, LiRA) deve essere riportato come
#   mean ± std su ≥5 seed indipendenti (minimo accettato dalle venue ML top-tier).
# Tutti i seed girano nella STESSA sweep-dir → Excel aggrega mean±std per (ε, rounds, no_dp).
#
# Passo tipico DSN 2027:
#   1. make experiment-nodp-sweep   → Scenario A confermato? (LiRA mean AUC > 0.55?)
#   2. make experiment-dp-sweep     → DP sopprime LiRA? (mean AUC → 0.50?)
#   3. make experiment-full-sweep   → sweep completo rounds × ε per le figure del paper
#
# SEEDS override: make experiment-nodp-sweep SEEDS="42 123 456"  (3 seed per test rapido)
# SEED deve restare in every-run single target; SEEDS è per il loop multi-seed.

.PHONY: experiment-nodp-sweep
experiment-nodp-sweep: _check-deps _sweep_lock
	@mkdir -p $(EXPERIMENTS); \
	echo "$$$$ experiment-nodp-sweep avviato $$(date '+%Y-%m-%d %H:%M:%S')" > $(SWEEP_LOCK); \
	trap 'rm -f $(SWEEP_LOCK)' EXIT INT TERM; \
	SWEEP_NUM=$$(find $(EXPERIMENTS) -maxdepth 1 -type d -name 'nodp-sweep[0-9]*' 2>/dev/null | wc -l | tr -d ' '); \
	SWEEP_NUM=$$((SWEEP_NUM + 1)); \
	SWEEP_DIR=$(EXPERIMENTS)/nodp-sweep$$SWEEP_NUM; \
	mkdir -p "$$SWEEP_DIR"; \
	LOG="$$SWEEP_DIR/sweep_log.txt"; \
	echo "→ no-DP multi-seed sweep #$$SWEEP_NUM — seeds: $(SEEDS)" | tee "$$LOG"; \
	echo "  n_shadow=$(N_SHADOW), rounds=10, no-DP" | tee -a "$$LOG"; \
	FAILED=0; \
	for seed in $(SEEDS); do \
		echo "=== seed=$$seed ===" | tee -a "$$LOG"; \
		{ $(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
			--config config/experiment.yaml \
			--no-dp \
			--rounds 10 \
			--seed $$seed \
			--n-shadow $(N_SHADOW) \
			--sweep-dir "$$SWEEP_DIR"; echo $$? > /tmp/_cs_rc_$$$$; } 2>&1 | tee -a "$$LOG"; \
		RC=$$(cat /tmp/_cs_rc_$$$$ 2>/dev/null || echo 1); rm -f /tmp/_cs_rc_$$$$; \
		if [ "$$RC" != "0" ]; then \
			echo "✗ seed=$$seed FALLITO (exit $$RC)" | tee -a "$$LOG"; \
			FAILED=$$((FAILED + 1)); \
		fi; \
	done; \
	if [ $$FAILED -eq 0 ]; then \
		echo "✓ no-DP sweep #$$SWEEP_NUM completato — controlla Seed Aggregation nell'Excel" | tee -a "$$LOG"; \
	else \
		echo "✗ no-DP sweep #$$SWEEP_NUM: $$FAILED/$(words $(SEEDS)) seed falliti — NON e' completo, vedi $$LOG" | tee -a "$$LOG"; \
		exit 1; \
	fi

.PHONY: experiment-dp-sweep
experiment-dp-sweep: _check-deps _sweep_lock
	@mkdir -p $(EXPERIMENTS); \
	echo "$$$$ experiment-dp-sweep avviato $$(date '+%Y-%m-%d %H:%M:%S')" > $(SWEEP_LOCK); \
	trap 'rm -f $(SWEEP_LOCK)' EXIT INT TERM; \
	SWEEP_NUM=$$(find $(EXPERIMENTS) -maxdepth 1 -type d -name 'dp-sweep[0-9]*' 2>/dev/null | wc -l | tr -d ' '); \
	SWEEP_NUM=$$((SWEEP_NUM + 1)); \
	SWEEP_DIR=$(EXPERIMENTS)/dp-sweep$$SWEEP_NUM; \
	mkdir -p "$$SWEEP_DIR"; \
	LOG="$$SWEEP_DIR/sweep_log.txt"; \
	echo "→ DP multi-seed sweep #$$SWEEP_NUM — ε=$(EPS), seeds: $(SEEDS)" | tee "$$LOG"; \
	echo "  n_shadow=$(N_SHADOW), rounds=10, DP ε=$(EPS)" | tee -a "$$LOG"; \
	FAILED=0; \
	for seed in $(SEEDS); do \
		echo "=== seed=$$seed ===" | tee -a "$$LOG"; \
		{ $(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
			--config config/experiment.yaml \
			--epsilon $(EPS) \
			--rounds 10 \
			--seed $$seed \
			--n-shadow $(N_SHADOW) \
			--sweep-dir "$$SWEEP_DIR"; echo $$? > /tmp/_cs_rc_$$$$; } 2>&1 | tee -a "$$LOG"; \
		RC=$$(cat /tmp/_cs_rc_$$$$ 2>/dev/null || echo 1); rm -f /tmp/_cs_rc_$$$$; \
		if [ "$$RC" != "0" ]; then \
			echo "✗ seed=$$seed FALLITO (exit $$RC)" | tee -a "$$LOG"; \
			FAILED=$$((FAILED + 1)); \
		fi; \
	done; \
	if [ $$FAILED -eq 0 ]; then \
		echo "✓ DP sweep #$$SWEEP_NUM completato (ε=$(EPS)) — confronta con no-DP in Seed Aggregation" | tee -a "$$LOG"; \
	else \
		echo "✗ DP sweep #$$SWEEP_NUM (ε=$(EPS)): $$FAILED/$(words $(SEEDS)) seed falliti — NON e' completo, vedi $$LOG" | tee -a "$$LOG"; \
		exit 1; \
	fi

# Central DP multi-seed sweep (2026-07-22) — CS4 candidate, docs/CaseStudies.md §2.4.3.
.PHONY: experiment-central-dp-sweep
experiment-central-dp-sweep: _check-deps _sweep_lock
	@mkdir -p $(EXPERIMENTS); \
	echo "$$$$ experiment-central-dp-sweep avviato $$(date '+%Y-%m-%d %H:%M:%S')" > $(SWEEP_LOCK); \
	trap 'rm -f $(SWEEP_LOCK)' EXIT INT TERM; \
	SWEEP_NUM=$$(find $(EXPERIMENTS) -maxdepth 1 -type d -name 'central-sweep[0-9]*' 2>/dev/null | wc -l | tr -d ' '); \
	SWEEP_NUM=$$((SWEEP_NUM + 1)); \
	SWEEP_DIR=$(EXPERIMENTS)/central-sweep$$SWEEP_NUM; \
	mkdir -p "$$SWEEP_DIR"; \
	LOG="$$SWEEP_DIR/sweep_log.txt"; \
	echo "→ Central DP multi-seed sweep #$$SWEEP_NUM — ε=$(EPS), seeds: $(SEEDS)" | tee "$$LOG"; \
	echo "  n_shadow=$(N_SHADOW), rounds=10, Central DP ε=$(EPS)" | tee -a "$$LOG"; \
	FAILED=0; \
	for seed in $(SEEDS); do \
		echo "=== seed=$$seed ===" | tee -a "$$LOG"; \
		{ $(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
			--config config/experiment.yaml \
			--epsilon $(EPS) \
			--rounds 10 \
			--seed $$seed \
			--n-shadow $(N_SHADOW) \
			--dp-mode central \
			--sweep-dir "$$SWEEP_DIR"; echo $$? > /tmp/_cs_rc_$$$$; } 2>&1 | tee -a "$$LOG"; \
		RC=$$(cat /tmp/_cs_rc_$$$$ 2>/dev/null || echo 1); rm -f /tmp/_cs_rc_$$$$; \
		if [ "$$RC" != "0" ]; then \
			echo "✗ seed=$$seed FALLITO (exit $$RC)" | tee -a "$$LOG"; \
			FAILED=$$((FAILED + 1)); \
		fi; \
	done; \
	if [ $$FAILED -eq 0 ]; then \
		echo "✓ Central DP sweep #$$SWEEP_NUM completato — atteso: LiRA NON soppressa (vedi CaseStudies.md §2.4.3)" | tee -a "$$LOG"; \
	else \
		echo "✗ Central DP sweep #$$SWEEP_NUM (ε=$(EPS)): $$FAILED/$(words $(SEEDS)) seed falliti — NON e' completo, vedi $$LOG" | tee -a "$$LOG"; \
		exit 1; \
	fi

# Local DP multi-seed sweep (2026-07-22).
.PHONY: experiment-local-dp-sweep
experiment-local-dp-sweep: _check-deps _sweep_lock
	@mkdir -p $(EXPERIMENTS); \
	echo "$$$$ experiment-local-dp-sweep avviato $$(date '+%Y-%m-%d %H:%M:%S')" > $(SWEEP_LOCK); \
	trap 'rm -f $(SWEEP_LOCK)' EXIT INT TERM; \
	SWEEP_NUM=$$(find $(EXPERIMENTS) -maxdepth 1 -type d -name 'local-sweep[0-9]*' 2>/dev/null | wc -l | tr -d ' '); \
	SWEEP_NUM=$$((SWEEP_NUM + 1)); \
	SWEEP_DIR=$(EXPERIMENTS)/local-sweep$$SWEEP_NUM; \
	mkdir -p "$$SWEEP_DIR"; \
	LOG="$$SWEEP_DIR/sweep_log.txt"; \
	echo "→ Local DP multi-seed sweep #$$SWEEP_NUM — ε=$(EPS), seeds: $(SEEDS)" | tee "$$LOG"; \
	echo "  n_shadow=$(N_SHADOW), rounds=10, Local DP ε=$(EPS)" | tee -a "$$LOG"; \
	FAILED=0; \
	for seed in $(SEEDS); do \
		echo "=== seed=$$seed ===" | tee -a "$$LOG"; \
		{ $(PYTHON) $(SCRIPTS_DIR)/run_experiments.py \
			--config config/experiment.yaml \
			--epsilon $(EPS) \
			--rounds 10 \
			--seed $$seed \
			--n-shadow $(N_SHADOW) \
			--dp-mode local \
			--sweep-dir "$$SWEEP_DIR"; echo $$? > /tmp/_cs_rc_$$$$; } 2>&1 | tee -a "$$LOG"; \
		RC=$$(cat /tmp/_cs_rc_$$$$ 2>/dev/null || echo 1); rm -f /tmp/_cs_rc_$$$$; \
		if [ "$$RC" != "0" ]; then \
			echo "✗ seed=$$seed FALLITO (exit $$RC)" | tee -a "$$LOG"; \
			FAILED=$$((FAILED + 1)); \
		fi; \
	done; \
	if [ $$FAILED -eq 0 ]; then \
		echo "✓ Local DP sweep #$$SWEEP_NUM completato — IDS degradato per design (vedi docstring run_ids())" | tee -a "$$LOG"; \
	else \
		echo "✗ Local DP sweep #$$SWEEP_NUM (ε=$(EPS)): $$FAILED/$(words $(SEEDS)) seed falliti — NON e' completo, vedi $$LOG" | tee -a "$$LOG"; \
		exit 1; \
	fi


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

# Test di integrazione sulla pipeline reale (run_fl_rounds/run_fedmia/run_lira/run_ids),
# con dati sintetici e round/epoche/shadow ridotti (~secondi, non minuti).
# A differenza di test_sprint4.py/test_sprint5.py (che mockano NVFLARE), questi test
# eseguono il codice reale di scripts/run_experiments.py end-to-end.
.PHONY: test-integration
test-integration:
	@echo "→ Esecuzione test di integrazione (pipeline run_experiments.py)..."
	$(PYTEST) tests/test_run_experiments_integration.py -v --tb=short

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

# ─── NVFLARE Simulator (2026-07-24) ───────────────────────────────────────────
# Esecuzione REALE del job nvflare/jobs/chargeshield_poc/ tramite
# `nvflare simulator` — processi/thread locali, NESSUN bisogno di Docker o
# Containerlab (a differenza di build/provision/deploy sopra, che restano per
# la fase Containerlab, deliberatamente successiva — vedi docs/
# NVFlareIntegration.md, "Suggested next steps" #7: quello resta il PROSSIMO
# passo dopo che questi target avranno validato il job). Mai eseguito prima
# d'ora in nessun ambiente (torch/nvflare non installabili nel sandbox usato
# per scrivere il codice) — il primo run è quasi certamente il primo posto
# dove emergeranno bug reali (vedi i punti VERIFY ancora aperti in
# docs/NVFlareIntegration.md: contatori di round locali vs fl_ctx).
#
# install-flare installa l'extra "flare" di pyproject.toml (nvflare==2.7.2,
# che porta con sé torch>=2.0 tramite le dipendenze runtime base).
.PHONY: install-flare
install-flare:
	@echo "→ Installazione runtime + NVFLARE 2.7.2 (torch incluso)..."
	pip install -e ".[flare]" --break-system-packages
	@echo "✓ Installato. Verifica: python3 -c 'import nvflare; print(nvflare.__version__)'"

# Stesso principio di _check-deps sopra (fix 2026-07-24): installazione
# automatica invece di solo avvisare, così anche i target nvflare-sim* non
# richiedono un 'make install-flare' manuale separato prima del primo uso.
.PHONY: _check-nvflare-deps
_check-nvflare-deps:
	@$(PYTHON) -c "import nvflare, torch" 2>/dev/null || \
		(echo "" && \
		 echo "→ NVFLARE/torch mancanti — installazione automatica..." && \
		 $(MAKE) install-flare && \
		 echo "✓ Dipendenze installate — proseguo con il target richiesto." && \
		 echo "") || \
		(echo "✗ Installazione automatica fallita (make install-flare) — impossibile proseguire." >&2; exit 1)

# Directory separata dal workspace di provisioning Containerlab (nvflare/workspace,
# variabile WORKSPACE sopra) — `nvflare simulator` genera il proprio workspace
# locale ad ogni run (nessun PKI/provisioning reale coinvolto), ricreato da zero
# ogni volta per evitare stato residuo tra run successivi.
NVFLARE_SIM_WORKSPACE := nvflare/sim_workspace

# CHARGESHIELD_PROJECT_ROOT (2026-07-24, fix da bug reale trovato al primo run):
# chargeshield_executor.py/chargeshield_aggregator.py risolvevano la project
# root con Path(__file__).resolve().parents[N], che si è rotto non appena
# `nvflare simulator` ha copiato quei file dentro il workspace a una profondità
# diversa (vedi commenti nei due file). Passare la root esplicitamente qui
# elimina la dipendenza da "dove NVFLARE decide di copiare il codice".
# CHARGESHIELD_MIN_CLIENTS=1 (2026-07-24, fix da bug reale trovato dall'utente
# sul primo tentativo): config_fed_server.json ha min_clients=3 hardcoded (corretto
# per il deploy reale a 3 siti), ma questo rende STRUTTURALMENTE impossibile
# completare un'aggregazione con un solo client — FedAvgAggregator.aggregate()
# avrebbe sempre restituito None ("partecipanti validi insufficienti: 1 < 3"),
# non un crash ma un round vuoto per sempre, che impediva allo smoke test di
# validare qualunque cosa oltre al semplice round-trip DXO/accept(). L'override
# (letto da ChargeShieldAggregator.__init__) permette allo smoke test di
# completare un'aggregazione vera con un client solo.
.PHONY: nvflare-sim-smoke
nvflare-sim-smoke: _check-nvflare-deps
	@echo "→ NVFLARE simulator — smoke test (1 client: caltech, min_clients=1)..."
	@rm -rf $(NVFLARE_SIM_WORKSPACE)
	CHARGESHIELD_PROJECT_ROOT=$(CURDIR) CHARGESHIELD_MIN_CLIENTS=1 \
		$(NVFLARE) simulator nvflare/jobs/chargeshield_poc \
		-w $(NVFLARE_SIM_WORKSPACE) \
		-n 1 -c caltech
	@echo "✓ Smoke test NVFLARE completato — controlla $(NVFLARE_SIM_WORKSPACE)/ e experiments/nvflare_*"

# Run con i 3 siti reali insieme — SOLO dopo che nvflare-sim-smoke passa.
.PHONY: nvflare-sim
nvflare-sim: _check-nvflare-deps
	@echo "→ NVFLARE simulator — 3 siti reali (caltech, jpl, office1)..."
	@rm -rf $(NVFLARE_SIM_WORKSPACE)
	CHARGESHIELD_PROJECT_ROOT=$(CURDIR) $(NVFLARE) simulator nvflare/jobs/chargeshield_poc \
		-w $(NVFLARE_SIM_WORKSPACE) \
		-n 3 -c caltech,jpl,office1
	@echo "✓ NVFLARE simulator completato — controlla $(NVFLARE_SIM_WORKSPACE)/, experiments/nvflare_ids_audit_results.json, experiments/nvflare_fl_results.pkl"

.PHONY: clean-nvflare-sim
clean-nvflare-sim:
	@echo "→ Rimozione workspace del simulatore NVFLARE..."
	rm -rf $(NVFLARE_SIM_WORKSPACE)
	@echo "✓ Rimosso"

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
clean-all: clean clean-workspace clean-nvflare-sim destroy
	@echo "✓ Pulizia completa"

# ─── All ──────────────────────────────────────────────────────────────────────
.PHONY: all
all: build provision deploy experiment
