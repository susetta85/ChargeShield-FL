# NVFLARE / Containerlab Integration — Status and Plan

> **Stato: documento OPERATIVO, di riferimento. Deployment NVFlare e Containerlab, divergenze note dalla simulazione. Attenzione alla segnalazione 1 sul seed dello snapshot. I rimandi a `DSN2027_Positioning.md`, `TestRoadmap_DSN2027.md` e altri documenti eliminati il 2026-09-22 puntano alla storia git: lo stato corrente è in `STATO.md` ed `ESPERIMENTI.md`.**

**Started:** 2026-07-22
**Status:** Job scaffold + client Executor + custom Aggregator + DP wiring + structured exports (fase 1-5) written. **First real execution: 2026-07-24** (see "First real run" section below) — 4 real bugs found and fixed across three attempts; **the third attempt completed all 10 rounds successfully** (`make nvflare-sim-smoke`, 1 client/caltech, `min_clients=1`: "Round 9 finished" → "Finished ScatterAndGather Training", no errors). This confirms the DXO/Executor/Aggregator/DP/IDS-export pipeline runs end-to-end for a single client. **Same day, follow-up fix**: raw exports are now timestamped per run (see "Fix 2026-07-24: export non più sovrascritti tra run" below) so a second/accidental run can no longer silently overwrite a prior run's results.

> **Status correction (2026-09-09, found during a documentation audit, task #81) — the rest of this
> intro paragraph was never updated after it was overtaken by events.** It used to end here with
> "Not yet run: the 3-real-site shape (`make nvflare-sim`, `-n 3 -c caltech,jpl,office1`) — that is
> the next concrete step, not yet attempted." That framing is long stale: the 3-site simulator shape
> ran to completion the same week (see "Verified 2026-08-31" below), the real multi-container
> Containerlab deployment (5 nodes: `server`/`caltech`/`jpl`/`office1`/`fl-admin`) went through
> several real bug-fix cycles from 2026-08-01 through 2026-08-31 (see the dated updates further
> down), and — most recently — was **independently re-verified end-to-end on 2026-09-09**: "Step A"
> redeployed the containers fresh with the latest code and ran a full clean 10-round FedAvg job with
> no errors, and "Step B" ran a 5-seed statistical campaign on `dp_mode=dp-fedavg`, `epsilon=1.0` —
> all 5 seeds completed cleanly — and is currently being cross-validated against a `central` dp_mode
> variant. The real Containerlab multi-container deployment is a working, repeatedly-verified path
> today, not a "next step" or a simulator-only story — treat every "not yet attempted" phrase
> elsewhere in this document's older dated sections as historical (true when written), superseded by
> the runs recorded below.

**Update (2026-07-22, later same day):** the 4 fictional same-site "clusters" (`highway`/`urban`/`residential`/`corporate`) referenced throughout the fase 1-5 sections below have been replaced project-wide with the 3 real ACN-Data sites (`caltech`/`jpl`/`office1` — see README "Real multi-site experiment" and the JPL/Caltech mislabeling correction in the same section). `nvflare/project.yml`, `chargeshield_executor.py`, `config_fed_client.json`, and `config_fed_server.json` (`min_clients: 4→3`) were all updated to match. The fase 1-5 narrative and `VERIFY:` points below are left as originally written (historical record of that work) except where explicitly annotated as updated; read `highway`/`urban`/`residential`/`corporate` in what follows as referring to the old 4-cluster scheme this superseded, not the current client set.
**Why:** the environment used to write this code cannot install `torch` (proxy blocks `download.pytorch.org`) or, by extension, verify `nvflare` behaviour (nvflare depends on torch). Every NVFLARE API call below was written from documented/standard NVFLARE 2.x patterns and careful reading of the existing `src/ml/`/`src/auditor/`/`src/ids/` code, but **none of it has run**. Treat this as a first draft to debug on a machine with the real dependencies installed, not as working code. **Update (2026-07-24)**: re-checked — `pip install torch`/`pip install nvflare==2.7.2` now resolve their dependency graphs fine in this sandbox (no proxy block observed today), but the actual wheel downloads are large enough (CUDA toolkit dependencies pulled in alongside torch) to exceed this session's per-command execution time budget, so a full install still wasn't completed here. This is a sandbox time-limit constraint, not necessarily a hard network block anymore — worth trying a plain `pip install torch nvflare==2.7.2` on a normal (non-time-boxed) machine before assuming it will fail the same way.

This document exists because the prior state of the repo's Containerlab/NVFLARE scaffolding was audited (2026-07-21, see `docs/CaseStudies.md` §2.4.3's "the privacy pipeline does not run on the containerised network" limitation) and found to be unused: `src/flare/flare_connector.py` is an explicit Sprint-3 placeholder that never imports `nvflare` and simulates gradients with `random.gauss()`; `nvflare/project.yml` only provisions PKI/network participants, no job/app existed; the `docker/` Dockerfiles are orphaned (unreferenced, and their `CMD`s have no `if __name__ == "__main__"` guard, so they'd crash on start). This document and the files under `nvflare/jobs/chargeshield_poc/` are the first concrete step toward closing that gap — not a completed integration.

## First real run (2026-07-24) — `make nvflare-sim-smoke`, real bugs found and fixed

The user ran `make install-flare` (nvflare 2.8.1 installed successfully — no proxy/dependency issue
on the real machine, unlike this sandbox) followed by `make nvflare-sim-smoke` (`-n 1 -c caltech`).
This is the **first execution of any of this code, ever**, in any environment. It crashed, exactly
as expected for code that had only ever been `py_compile`-checked — but it crashed in an
informative way, and both root causes were real, fixable bugs rather than fundamental design
problems:

1. **`_PROJECT_ROOT` resolution broke.** Both `chargeshield_executor.py` and
   `chargeshield_aggregator.py` computed `_PROJECT_ROOT = Path(__file__).resolve().parents[4]`,
   correct only if the file stayed at its original location
   (`nvflare/jobs/chargeshield_poc/app/custom/`). `nvflare simulator` instead copies `custom/` into
   the workspace (observed: `nvflare/sim_workspace/server/simulate_job/app_server/custom/`), a
   different depth — so `_PROJECT_ROOT` resolved to a path *inside* `sim_workspace/`, and every
   downstream path built from it (dataset directory, `config/auditor.yaml`) pointed nowhere real.
   Observed symptoms: `[caltech] Directory dataset non trovata:
   .../sim_workspace/datasets/acn/caltech` and `FileNotFoundError: Auditor config not found:
   .../sim_workspace/config/auditor.yaml`. **Fixed**: both files now use `_find_project_root()`,
   which checks the `CHARGESHIELD_PROJECT_ROOT` environment variable first (now set by `make
   nvflare-sim`/`nvflare-sim-smoke` to `$(CURDIR)`) and falls back to walking up from `__file__`
   looking for `pyproject.toml` with `name = "chargeshield-fl"`, for anyone invoking `nvflare
   simulator` directly without the Makefile.
2. **`ChargeShieldAggregator._ensure_components()`'s "already initialized" guard was fragile.**
   It checked `if self._fedavg is not None: return`. At round 0, `PrivacyAuditor.__init__()` raised
   `FileNotFoundError` (bug 1 above) — *after* `self._fedavg` had already been assigned but
   *before* `self._gm` was. The next round's call to `_ensure_components()` saw `self._fedavg` set
   and concluded initialization was complete, permanently skipping re-init — so `self._gm` stayed
   `None` forever, even once bug 1 was fixed. This surfaced as a second, seemingly unrelated crash:
   `AttributeError: 'NoneType' object has no attribute 'privatize'` in `aggregate()` at round 1.
   **Fixed**: a new `self._components_ready` flag, set to `True` only after every component in
   `_ensure_components()` has been constructed successfully — so a partial failure now causes a
   full retry on the next round instead of a false "already done."

**Second attempt, same day, after fixing 1-2 above**: the user re-ran `make nvflare-sim-smoke` and
it progressed much further (round 1 accepted caltech's contribution — the DXO/Executor/Aggregator
round-trip genuinely works now) but then hit a third issue, and the user correctly noticed and
interrupted the run rather than assuming it would resolve itself:

3. **`min_clients=3` (correct for the real 3-site deployment) made the 1-client smoke test
   structurally unable to ever complete an aggregation.** Observed: `Round 2 — partecipanti validi
   insufficienti: 1 < 3 (update raccolti: 1, di cui 0 invalidi)`, followed by `FedAvgAggregator non
   ha prodotto un aggregato — restituisco Shareable vuoto`. Not a crash (no exception) — `-n 1
   -c caltech` will only ever collect 1 valid update per round, and `FedAvgAggregator.aggregate()`
   correctly refuses to aggregate below its configured `min_participants`, so every round after the
   first produces an empty, no-op global model forever. This is a mismatch between what the
   "smoke test" was documented to validate (the transport contract, which it now does) and what
   `-n 1` can structurally deliver against a config hardcoded for 3 real sites. **Fixed**: a new
   `CHARGESHIELD_MIN_CLIENTS` environment variable read by `ChargeShieldAggregator.__init__`,
   defaulting to the config value (3) but overridable — `make nvflare-sim-smoke` now sets it to `1`,
   so the 1-client smoke test can complete a genuine (if trivial) single-client aggregation instead
   of silently producing empty rounds forever. `make nvflare-sim` (the real 3-site run) does not set
   this override, so the real deployment's `min_clients=3` is untouched.

A round-4 independent code review (separate from the two real-execution attempts above) found one
more real bug in the same files, not yet exercised by either attempt because it doesn't crash —
it silently produces wrong data instead:

4. **The Executor's duplicated `_enrich_sessions()` was missing the 2026-07-22 timezone-localization
   fix already applied to the simulation's `enrich_sessions()`.** It still computed `hour_of_day`
   as `float(start.hour)` on the raw UTC timestamp — exactly the pre-fix behavior that was found and
   corrected in `scripts/run_experiments.py` on 2026-07-22 (ACN-Data's timestamps carry a misleading
   "GMT" suffix but are genuine UTC; the real local hour requires localizing via each session's
   `timezone` field). This directly contradicted the function's own comment claiming "stessa formula
   esatta, nessuna deviazione di logica" (same exact formula, no deviation) relative to the
   simulation — true for `_compute_feature_stats()`/`_normalize_sessions()`, false for
   `_enrich_sessions()`. Not caught by the 2026-07-22 "2652/2652 sessioni valide, feature nel range
   [0,1] atteso" empirical check (see fase 3-5 section above) because that check verifies
   non-emptiness and post-normalization range, not whether the underlying value is *correct* — a
   consistent-but-wrong offset passes both checks. **Fixed**: ported the identical `ZoneInfo`-based
   localization block from `scripts/run_experiments.py::enrich_sessions()`.

**Invalidation check**: none of the four bugs touch `scripts/run_experiments.py` or anything the
single-process simulation depends on — these are NVFLARE-job-only files (the executor/aggregator
under `nvflare/jobs/chargeshield_poc/`). No existing experiment result (including the Central DP
numbers reported elsewhere in this document and in `docs/PrivacyExposureScore_v1.md`) is affected.
All four fixes are `py_compile`-verified only from this side (no torch/nvflare in this sandbox) —
but **confirmed by a successful real run**: after all four fixes, the user re-ran `make
nvflare-sim-smoke` and it completed all 10 configured rounds without error (`Round 9 finished` →
`Finished ScatterAndGather Training`). This is the first time any of this code has ever run to
completion. **Next step, not yet attempted**: `make nvflare-sim` (`-n 3 -c caltech,jpl,office1`,
`min_clients=3`) — the real 3-site deployment shape. A single successful 1-client run derisks the
transport/DP/IDS-export pipeline considerably but does not guarantee the 3-site case is
bug-free (e.g. the per-cluster `cluster_id` derivation, IDS Krum quorum at n=3, and multi-client
`FedAvgAggregator` weighting are all only exercised for the first time at `-n 3`).

Also noted, not yet acted on: the simulator printed `WARNING: 'nvflare simulator' is deprecated.
Use 'python job.py' with SimEnv instead.` — nvflare 2.8.1 (installed) vs. `>=2.7.2` (pinned in
`pyproject.toml`) still works today via the deprecated path, but migrating to the `SimEnv` API is
worth a follow-up task before this becomes a hard blocker in a future nvflare release.

## Fix 2026-07-24: export non più sovrascritti tra run

Dopo il primo `nvflare-sim-smoke` riuscito (sezione sopra), l'utente ha per sbaglio avviato e
subito interrotto un secondo run — senza danno in quel caso, ma ha fatto notare un rischio reale:
`ChargeShieldAggregator.__init__` scriveva i due export raw fase 4/5
(`experiments/nvflare_ids_audit_results.json`, `experiments/nvflare_fl_results.pkl`) con nomi
**fissi**. A differenza di `scripts/run_experiments.py`, che già produce un
`experiment_{timestamp}.json` univoco per ogni run della simulazione single-process, un secondo run
NVFLARE (anche solo uno smoke test di verifica) avrebbe silenziosamente sovrascritto l'export del
run precedente — inclusi risultati riusciti, senza alcun avviso.

**Fix**: `ChargeShieldAggregator.__init__` ora cattura un timestamp una sola volta all'avvio (`_run_ts
= datetime.now().strftime("%Y%m%d_%H%M%S")`) e lo inserisce nel nome di entrambi i file
(`nvflare_ids_audit_results_<timestamp>.json`, `nvflare_fl_results_<timestamp>.pkl`). I default in
`config_fed_server.json` restano invariati (nomi "puliti", senza timestamp) — il timestamp è
aggiunto a runtime, non in config. `scripts/run_nvflare_mia.py` (che consuma il dump pickle) non ha
più un default fisso per `--fl-results`: se omesso, una nuova `_resolve_fl_results_path()` sceglie
automaticamente il `nvflare_fl_results_*.pkl` più recente per data di modifica sotto `experiments/`,
avvisando (via `logger.warning`) se ne trova più di uno.

Verificato: `python3 -m py_compile` su entrambi i file modificati (`chargeshield_aggregator.py`,
`run_nvflare_mia.py`) e l'intera test suite non-torch (71 passed, nessuna regressione) — non
eseguito con torch/nvflare reali in questo sandbox, stesso limite di sempre. Non tocca la
simulazione single-process né alcun risultato pubblicato (Central DP, PES_v1): riguarda solo i due
export raw NVFLARE.

**Non ancora deciso**: se estendere la stessa garanzia di unicità-per-run ai report Excel della
pipeline principale (`scripts/run_experiments.py`/`run_sweep.py`). Questi ultimi seguono di
proposito un design "aggregato per sweep-dir" (più run/seed nello stesso file, per il foglio "Seed
Aggregation" mean±std) — non un file-per-run — quindi non è lo stesso problema e non va cambiato
senza una decisione esplicita.

## Review indipendente post-fase-5 (2026-07-22, notte) — bug reali trovati e corretti

Dopo la fase 5, è stata condotta una review indipendente (agente separato, nessun contesto della conversazione originale) su fase 3+4+5. Ha trovato due bug funzionali reali (non solo VERIFY/ipotesi) e un'osservazione minore:

1. **CRITICO, corretto**: `chargeshield_executor.py::_setup()` non chiamava mai l'equivalente di `enrich_sessions()`/`normalize_sessions()` (scripts/run_experiments.py) prima di passare le sessioni ad `AutoencoderTrainer`. `AutoencoderTrainer.CONTINUOUS_FEATURES` include `hour_of_day`/`duration_hours`, calcolati SOLO da `enrich_sessions()` a partire da `start_time`/`end_time` — assenti nei sample grezzi di `ACNDataset`. Verificato empiricamente (eseguendo davvero il codice in questo sandbox, senza bisogno di torch): **0 sessioni su 10609** sarebbero risultate valide per `_sessions_to_tensor()` senza il fix — ogni client NVFLARE reale avrebbe addestrato su un tensore vuoto, un fallimento silenzioso totale del training, mai catturato da `py_compile`. Fix: `_enrich_sessions()`/`_compute_feature_stats()`/`_normalize_sessions()` (duplicati da `scripts/run_experiments.py`, non importati, per non riconfigurare `logging.basicConfig()` dentro un processo client NVFLARE reale) ora chiamati in `_setup()`. Rieseguito lo stesso test empirico dopo il fix: **2652/2652 sessioni valide** per il cluster `highway`, tutte le feature nel range `[0,1]` atteso.
2. **CRITICO, corretto**: `scripts/run_nvflare_mia.py` (appena scritto in fase 5) caricava le sessioni con `load_sessions(cfg)` da `config/experiment.yaml` (che combina `jpl_2019`+`jpl_2020`) e le shuffle-ava prima dello split train/hold-out — un dataset e un ordinamento **diversi** da quelli che i client NVFLARE reali vedono davvero (`chargeshield_executor.py` carica SOLO il file indicato da `dataset_path` in `config_fed_client.json`, MAI shuffled, split contiguo per indice). Poiché `run_lira()`/`run_fedmia()` ricostruiscono l'appartenenza ai cluster assumendo lo stesso ordine/split dei client reali, questo mismatch avrebbe reso gli AUC di LiRA/Shadow/Yeom **non significativi senza generare alcun errore** — il tipo di fallimento silenzioso più pericoloso. Fix: nuove funzioni `load_client_sessions()` (legge `dataset_path` da `--client-config`, nessuno shuffle) e `load_holdout_sessions()` (carica un file dataset genuinamente mai visto dai client, es. l'anno successivo — dedotto automaticamente per nome file, con override esplicito via `--holdout-dataset`). Verificata la logica di lettura/deduzione path con un test isolato reale (senza torch).
3. **Non risolto, documentato esplicitamente**: al round 1, `ChargeShieldAggregator.aggregate()` chiama `GradientManager.privatize()` per `dp_mode="dp-fedavg"` con `reference_weights=self._prev_global_weights=None` (non ancora assegnato) — questo attiva il fallback storico "clip assoluto" invece di "clip sul delta" (diverso da ogni round successivo e dalla simulazione, dove il riferimento è sempre concreto anche al round 1). Non risolto qui: un fix corretto richiederebbe che il client invii i propri pesi pre-round nel DXO (plumbing aggiuntivo non tentato alla cieca in un sandbox senza possibilità di esecuzione) — commentato in dettaglio nel codice (`chargeshield_aggregator.py`, vicino alla chiamata `privatize()` per `dp-fedavg`) invece di tentare un fix speculativo non verificabile.

Osservazione minore accettata senza fix: l'export fase 4 (`_run_ids_analysis()`) non include il campo `drift_detected` (sempre `False`/inutilizzato anche nella simulazione) — cosmetico, non funzionale.

## What exists now

```
nvflare/jobs/chargeshield_poc/
  meta.json                           — job metadata, deploy_map (single "app" to all sites)
  app/config/config_fed_server.json   — ScatterAndGather workflow + ChargeShieldAggregator (fase 2)
  app/config/config_fed_client.json   — points at the custom Executor below
  app/custom/chargeshield_executor.py — wraps AutoencoderTrainer.train_local()
  app/custom/chargeshield_aggregator.py — wraps FedAvgAggregator + PrivacyAuditor + ByzantineDetector (fase 2, 2026-07-22)
```

**Server side** (`config_fed_server.json`) still uses NVFLARE's `ScatterAndGather` workflow (round orchestration: broadcast → wait for clients → aggregate → persist), ma da oggi (fase 2) l'aggregatore built-in `InTimeAccumulateWeightedAggregator` è stato sostituito da `ChargeShieldAggregator` — un `Aggregator` NVFLARE custom che al suo interno chiama le classi **vere e già testate** della simulazione: `FedAvgAggregator` (src/ml/fedavg_aggregator.py) per la media pesata, e `PrivacyAuditor`+`ByzantineDetector` (src/auditor, src/ids) per l'analisi privacy/IDS per-round, con la stessa logica di normalizzazione peer-relative (mediana) di `scripts/run_experiments.py::run_ids()`. `PTFileModelPersistor`/`FullModelShareableGenerator` restano built-in (nessun motivo per sostituirli). Da oggi pomeriggio (fase 3) anche `GradientManager`/DP è collegato — vedi sezione dedicata sotto.

Fase 1 (2026-07-22, mattina) era transport-only con l'aggregatore built-in. Fase 2 (stesso giorno, dopo) ha introdotto `ChargeShieldAggregator`. Fase 3 (stesso giorno, pomeriggio) ha aggiunto la DP client-side (Executor) e server-side (Aggregator). Questo documento è stato aggiornato ad ogni passaggio.

## DP wiring (fase 3, 2026-07-22, non verificato)

`dp_mode`/`epsilon`/`delta`/`max_grad_norm` sono ora parametri sia dell'Executor (`config_fed_client.json`) sia dell'Aggregator (`config_fed_server.json`) — devono combaciare tra i due file, nessuna validazione incrociata automatica esiste ancora. Semantica dei 3 `dp_mode`, mirror esatto di quella già implementata e testata in `scripts/run_experiments.py`/`src/ml/gradient_manager.py`:

- **`dp-fedavg`**: il client invia l'update grezzo (nessuna operazione lato Executor). Il server, dentro `ChargeShieldAggregator.aggregate()`, chiama `GradientManager.privatize()` (clip sul delta rispetto a `self._prev_global_weights` + noise) su ciascun update ricevuto, PRIMA di passarlo a `FedAvgAggregator` — il server vede il valore grezzo transitoriamente. Questo placement (rumore per-client, prima dell'aggregazione) è una variante più restrittiva e non-standard, non descritta letteralmente nell'Algoritmo 1 di McMahan et al. 2018 (vedi `central` sotto per il mode a cui quel paper corrisponde davvero).
- **`central`** [McMahan et al. 2018 — questo è esattamente il meccanismo del loro Algoritmo 1 (DP-FedAvg): clip lato client, un singolo draw di rumore lato server sull'aggregato]: il client, in `chargeshield_executor.py::execute()`, chiama `GradientManager.clip_only()` (clip sul delta rispetto a `pre_round_weights`, nessun noise) prima di inviare. Il server chiama `GradientManager.privatize_aggregate()` una sola volta sul risultato di `FedAvgAggregator`, aggiungendo un singolo draw di rumore che beneficia della riduzione di sensitività 1/n.
- **`local`**: il client chiama `GradientManager.privatize()` (clip + noise) prima di inviare — il server/IDS non vede mai il valore grezzo, nemmeno transitoriamente.

L'analisi IDS/Auditor (`ChargeShieldAggregator._run_ids_analysis()`) gira sempre su `received_updates`, cioè la vista "più grezza disponibile" per quella modalità: grezza vera per `dp-fedavg`, clippata-non-rumorosa per `central`, clippata+rumorosa per `local` — coerente con la degradazione IDS sotto local DP già documentata per la simulazione single-process.

Entrambi i lati (`chargeshield_executor.py`, `chargeshield_aggregator.py`) sono stati verificati solo con `python3 -m py_compile` — **non eseguiti**, stesso limite ambientale (niente torch/nvflare in questo sandbox) di tutto il resto di questo documento.

## Export strutturato (fase 4) e raw-update extraction (fase 5), 2026-07-22

`ChargeShieldAggregator` ora scrive due file dopo ogni round (path configurabili in `config_fed_server.json`, entrambi sotto `experiments/` — stessa directory, già in `.gitignore`, usata dalla simulazione):

- `experiments/nvflare_ids_audit_results_<timestamp>.json` (fase 4): cronologia IDS/Auditor per round, stesso formato di `ids_results` in `run_ids()` — alerts, `byzantine_detected`, `low_similarity_nodes`, più un blocco `per_client_audit` (privacy_score/epsilon/threats_detected).
- `experiments/nvflare_fl_results_<timestamp>.pkl` (fase 5): dump **pickle** (non JSON — contiene `GradientUpdate` con `torch.Tensor`) con esattamente lo stesso schema che `run_fl_rounds()` produce in memoria per la simulazione: `mean_loss`, `n_participants`, `updates`, `raw_updates`, `raw_global_weights`, `global_weights` per round.
  (Nomi fissi, senza `<timestamp>`, fino al fix 2026-07-24 descritto più sotto — vedi "Fix 2026-07-24: export non più sovrascritti tra run".)

**Decisione di design per la fase 5** (perché LiRA non gira "dal vivo" dentro `aggregate()`): `run_lira()` è già, anche nella simulazione, un'analisi post-hoc che itera sull'intero dict `fl_results` dopo che tutti i round sono finiti, e ha richiesto cinque round di fix empirici (vedi la sua docstring in `scripts/run_experiments.py`) trovati eseguendo davvero il codice. Riscriverla alla cieca per girare dentro l'Aggregator, senza poter eseguire nulla in questo sandbox, sarebbe un secondo tentativo con alta probabilità di bug nuovi e silenziosi. Scelta fatta: `ChargeShieldAggregator` si limita a esportare il dump; un nuovo script, `scripts/run_nvflare_mia.py`, lo carica e chiama `run_lira()`/`run_ids()`/`run_fedmia()`/`run_fedmia_shadow()`/`save_results()` **invariati** — zero rischio di regressione sulla logica di attacco già validata su `nodp-sweep1`/`dp-sweep1`.

Verificato realmente in questo sandbox (senza torch, quindi solo la parte non torch-dipendente): il round-trip pickle di `GradientUpdate` attraverso lo stesso `sys.path` setup usato in produzione — scrittura, lettura, ricostruzione degli oggetti — eseguito con successo con dati fittizi (pesi come liste di float invece di tensor). Il resto (compreso l'intero `scripts/run_nvflare_mia.py`, che importa `torch` a livello di modulo come `run_experiments.py`) resta solo `py_compile`-verificato.

**Limite noto, non risolto in questa fase**: né l'Executor né l'Aggregator hanno un equivalente di `--no-dp` (bypass completo del rumore) — `dp_mode` è sempre uno dei 3 valori. `scripts/run_nvflare_mia.py` chiama sempre `run_ids()`/`run_lira()` con `no_dp=False`; per un run NVFLARE "senza DP" servirebbe un `epsilon` molto grande nei config, un'approssimazione non equivalente esatto al bypass della simulazione.

**Nota su "central" DP nel dump**: sotto `dp_mode="central"`, `received_updates` (esportati come `raw_updates`) sono già clippati dal client (fase 3) — a differenza della simulazione, dove `raw_updates` è il valore prima del clip (stesso processo, ordine di codice diverso). Non è un mismatch: `run_ids()`/`run_lira()` vogliono "la vista meno offuscata dal rumore DP disponibile al server", che per central DP è esattamente il valore clippato-non-rumorizzato — la stessa cosa, raggiunta per una via architetturalmente diversa (client-side invece che stessa riga di codice in-process).

**Client side** (`chargeshield_executor.py`) reuses the *real, already-tested* `AutoencoderTrainer` class from `src/ml/autoencoder_trainer.py` — the same class `scripts/run_experiments.py::run_fl_rounds()` uses in single-process simulation. It does **not** reimplement training logic. What it does: on `START_RUN`, instantiate `AutoencoderTrainer` and load this client's slice of sessions (same contiguous per-cluster split as the simulation, for now — see limitations below); on each `train` task, convert NVFLARE's incoming `Shareable`/`DXO` into the weight list `AutoencoderTrainer.set_weights()`/`apply_global_model()` expects, call `train_local()`, and convert the resulting `GradientUpdate` back into an outgoing `DXO`.

## What is explicitly NOT done yet

- ~~No DP.~~ **Done 2026-07-22 (fase 3).** `GradientManager.clip_only()`/`privatize()`/`privatize_aggregate()` are now called in both the Executor (client-side, `central`/`local` modes) and `ChargeShieldAggregator` (server-side, `dp-fedavg`/`central` modes) — see the "DP wiring" section above. Still unexecuted/untested like everything else in this document.
- ~~IDS/audit is wired in but only logs, doesn't export.~~ **Done 2026-07-22 (fase 4).** `ChargeShieldAggregator._run_ids_analysis()` now builds a structured per-round dict (same shape as `ids_results` in `scripts/run_experiments.py::run_ids()`: alerts, `byzantine_detected`, `low_similarity_nodes`, plus a new `per_client_audit` block with each `AuditReport`'s `privacy_score`/`epsilon`/`threats_detected`) and `_export_results()` overwrites a JSON file (`experiments/nvflare_ids_audit_results.json` by default, configurable via `results_export_path`) with the full history after every round. Only the JSON-serialization logic itself was actually executed (isolated from nvflare/torch) — the rest is `py_compile`-checked only. **Not decided/verified**: whether writing to a project-relative path is appropriate once this runs across a real multi-process/containerised deployment (the Aggregator only ever runs server-side, so it's a single process/filesystem — but the actual working directory NVFLARE uses at runtime is unconfirmed).
- ~~No LiRA/Shadow/Yeom attacks.~~ **Done 2026-07-22 (fase 5), as an offline step, by design.** `ChargeShieldAggregator` now exports a per-round pickle (`experiments/nvflare_fl_results.pkl`) with the exact schema `run_fl_rounds()` produces in the simulation, and the new `scripts/run_nvflare_mia.py` loads it and calls `run_lira()`/`run_ids()`/`run_fedmia()`/`run_fedmia_shadow()`/`save_results()` unchanged. LiRA is deliberately NOT run live inside `aggregate()` — see the "Export strutturato... raw-update extraction" section above for the rationale (LiRA took 5 rounds of empirically-found fixes; a blind live port risked new silent bugs). Only the pickle round-trip itself was actually executed in this sandbox (no torch); the rest is `py_compile`-checked only, same as everything else here.
- **Not on Containerlab.** This targets NVFLARE's own simulator/POC mode (`nvflare simulator`, in-process or local multi-process), not the `topology.clab.yml` containerised network. Docker/Containerlab wiring is a separate, later step (see Open Items below).
- ~~Per-client dataset access is fake.~~ **Done 2026-07-22 (3 real sites).** Every client used to load the *same* shared JSON file and take a slice by index. `chargeshield_executor.py::_setup()` now loads all `.json` files under `datasets/acn/<cluster_id>/` — that client's own real site directory (`caltech`/`jpl`/`office1`), all available years combined — so each of the 3 NVFLARE clients genuinely trains on only its own site's real data. One documented simplification remains: feature enrichment/normalisation stats are computed per-client on that site's own data only, not on a shared global pool as in the simulation — a real difference, not a bug, and worth flagging if NVFLARE vs. simulation numbers are ever compared directly. Synthetic clients (`synthetic_1`/`synthetic_2`, used in the simulation only for the IDS/Krum validation sweep, see README "Real multi-site experiment") are **not** ported to NVFLARE — out of scope for this pass; NVFLARE currently provisions only the 3 real sites (`min_clients: 3`).
- ~~Dataset is 2019-only.~~ **Done 2026-07-22.** Superseded by the per-site-directory loading above: each client now combines every year available for its site (Caltech/JPL: 2018–2021; Office1: 2019–2021, no 2018 published for that site) instead of a single hardcoded file.

## Fix applied 2026-07-22 (independent review, finding A1)

The first draft of `meta.json`/`config_fed_client.json` deployed a single shared `app/` to `"@ALL"` sites with `cluster_id` hardcoded to `"highway"`. Since `nvflare/project.yml` names the 4 client sites exactly `highway`/`urban`/`residential`/`corporate`, this meant **every** client would have instantiated `cluster_id="highway"` and trained on the identical 25% data slice — the opposite of the per-cluster heterogeneity the whole simulation (and this integration) is built around. Caught by an independent review pass, not by the original author — a good example of why a fresh second read matters even on unexecuted code.

Fixed in `chargeshield_executor.py::_setup()`: `cluster_id` is now derived from `fl_ctx.get_identity_name()` (the NVFLARE site name) when it matches one of the 4 known clusters, falling back to the config value with an explicit warning only if the site name isn't recognized. `config_fed_client.json`'s `"cluster_id": "highway"` (now `"caltech"` — the 3 real sites superseded the original 4 synthetic ones, see below) remains as the fallback default, not the active path, once `nvflare provision` assigns each site its real name.

**Verified 2026-08-31, twice — first indirectly (2026-08-04 job, below), then directly (2026-08-31 job, `make nvflare-sim` with Fase 7+8 ML Plane wiring active).** In the 2026-08-31 run's raw log, the fallback-mismatch warning fires exactly as designed, live, for the two clients whose real identity differs from the config default: `"cluster_id da config (caltech) diverso dal nome del sito NVFLARE (jpl) — uso il nome del sito"` and the same for `office1` — direct proof `get_identity_name()` returned `"jpl"`/`"office1"` correctly (not `None`, not `"caltech"`), each triggering the explicit divergence-from-config warning rather than the not-recognized fallback. This supersedes the indirect argument below as the primary evidence.

**Verified 2026-08-31 (indirectly, via a real 3-site job, not a unit test).** `experiments/nvflare_ids_audit_results_20260804_133100_58d089.json` — the real 3-client NVFLARE run (`make nvflare-sim`, `caltech`/`jpl`/`office1`) — has `per_client_audit` keyed by all **three distinct** site names in every round. This is decisive because all three clients share the *same* `config_fed_client.json`, whose fallback default is a single hardcoded value (`"caltech"`). If `get_identity_name()` had failed or returned something unrecognized on the `jpl`/`office1` clients, both would have silently fallen back to `cluster_id="caltech"` (with a logged warning) — producing at most 2 distinct keys, not 3, and colliding two sites' data under one label. Seeing 3 genuinely distinct keys is only possible if `get_identity_name()` correctly returned `"jpl"` and `"office1"` for those clients (`"caltech"` alone is ambiguous with the fallback, but is corroborated by the other two). **VERIFY point #4 is resolved** — not by unit test, but by the strongest evidence available short of one (a real run whose fallback path is distinguishable from its success path by construction). The single-client smoke test (`make nvflare-sim-smoke`, `-c caltech`) run on 2026-08-31 does *not* by itself confirm this, since its one client's config default and real identity coincide — that ambiguity is what this earlier 3-site job resolves.

## Points marked `VERIFY:` in the code — check these first

Search `chargeshield_executor.py` and `chargeshield_aggregator.py` for `VERIFY:`. Originally seven
points, all reasoned from documented NVFLARE patterns but none confirmed by actually running
anything. **Update (2026-07-24, preparatory pass before the first real execution attempt)**: with
`torch`/`nvflare` still uninstallable in this sandbox (confirmed again today — `pip install torch`
resolves dependencies but the actual wheel download exceeds this session's per-command time budget;
this is a sandbox constraint, not a fundamentally-blocked install, so it may well work in a normal
environment with no time-boxed shell), five of the seven were instead checked against NVFLARE's
**actual public source** (`nvflare.app_common.abstract.aggregator`,
`nvflare.app_common.shareablegenerators.full_model_shareable_generator`, both fetched directly from
the `NVIDIA/NVFlare` GitHub `main` branch) and official docs/discussions — this is not the same as
running our code, but it is stronger evidence than "reasoned from general patterns," and resolves
most of the framework-compatibility risk before the first real attempt.

1. **DXO data format — CONFIRMED.** `dxo.data` for `DataKind.WEIGHTS` is exactly `dict[str,
   np.ndarray]` keyed by `state_dict()` variable names — confirmed both by
   `FullModelShareableGenerator.shareable_to_learnable()`'s source (`weights = dxo.data`, stored
   directly under `ModelLearnableKey.WEIGHTS`, no wrapping) and by NVFLARE's own PyTorch examples
   (`new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}`). Our assumption in both
   `chargeshield_executor.py` and `chargeshield_aggregator.py` matches exactly — no code change
   needed, downgraded from "assumed" to "confirmed against upstream source."
2. **Aggregation weighting — MOOT, not just resolved.** This VERIFY point asked which
   `MetaKey`/`InTimeAccumulateWeightedAggregator` convention we need to match — but
   `ChargeShieldAggregator` doesn't use `InTimeAccumulateWeightedAggregator` at all; it's a
   from-scratch `Aggregator` subclass that reads its own custom `dxo.meta["n_samples"]` key
   directly (`chargeshield_executor.py` line ~488, `chargeshield_aggregator.py` line ~366). The
   original VERIFY comment in `chargeshield_executor.py` already noted this ("irrilevante per
   ChargeShieldAggregator, che legge n_samples direttamente") — this document's own VERIFY list
   just hadn't caught up to that. No `MetaKey` convention to match; nothing to fix.
3. **Round number — still genuinely open.** The Executor and Aggregator both count rounds with a
   local `self._round_num += 1` instead of reading `fl_ctx`'s authoritative round (e.g.
   `AppConstants.CURRENT_ROUND`). No amount of documentation reading resolves this — it depends on
   whether `ScatterAndGather` ever retries a task or re-invokes `accept()`/`execute()` outside a
   strict one-call-per-round-per-client pattern, which needs a real multi-round run to observe.
   Real risk, but degrades gracefully for a first straight-through smoke test (no retries expected
   at `-n 1`/`-n 3`, no failure injection).
4. **Site identity → cluster_id — CONFIRMED directionally, exact string form still open.**
   NVFLARE's own `FLContext` docs confirm `get_identity_name()` "returns the unique name of the
   peer site (client name or server name)" — consistent with our assumption that it returns
   `project.yml`'s site names (`caltech`/`jpl`/`office1`) verbatim. What's still unconfirmed:
   whether NVFLARE ever decorates this (org suffix, case normalization) — the existing
   fallback-with-warning path in `_setup()` already handles that gracefully if so; check the logs
   for that warning on first run.
5. **Client identity in the Aggregator via `dxo.meta["cluster_id"]` — reframed, not really an
   NVFLARE-API risk.** This is a private contract between our own Executor (which sets the key)
   and our own Aggregator (which reads it) — nothing in NVFLARE's API constrains this either way,
   so there was never real framework-compatibility risk here, only an internal-consistency
   requirement, which the two files already satisfy (both use the literal string `"cluster_id"`).
6. **round_num in the Aggregator — same status as point 3**, still open for the same reason (no
   dependency on `fl_ctx`'s authoritative round counter).
7. **Aggregator base class contract — CONFIRMED exactly.** Fetched
   `nvflare/app_common/abstract/aggregator.py` directly from GitHub (`main` branch): `accept(self,
   shareable: Shareable, fl_ctx: FLContext) -> bool` and `aggregate(self, fl_ctx: FLContext) ->
   Shareable` are the only two `@abstractmethod`s; `reset(self, fl_ctx)` has a concrete no-op
   default (`pass`), so **not** overriding it (as `ChargeShieldAggregator` currently does) is fine,
   not a gap. `ChargeShieldAggregator`'s method signatures match the abstract contract exactly.
   `FullModelShareableGenerator.shareable_to_learnable()`'s source also confirms our `aggregate()`
   return convention: a `DXO(DataKind.WEIGHTS, ...)` is read via `weights = dxo.data;
   base_model[WEIGHTS] = weights` — no additional wrapping expected, matches what
   `chargeshield_aggregator.py` returns.

**Net effect of this pass**: two real open risks remain (points 3/6, the round-number counter) —
everything else that could be checked without execution now has been. The two remaining risks are
exactly the kind that need a real `nvflare simulator` run to resolve, not more reading — see
"Suggested next steps" below, still step 1 in the list, unchanged by this pass.

## Suggested next steps (in order)

1. Install `nvflare==2.7.2` + `torch` in a real environment (not this sandbox — see the 2026-07-24
   update above; a plain `pip install` may just work outside this session's time-boxed shell) and
   run `nvflare simulator` against `nvflare/jobs/chargeshield_poc/`. **Automated 2026-07-24** —
   both steps are now Makefile targets instead of commands to remember/retype:
   ```bash
   make install-flare        # pip install -e ".[flare]" — torch + nvflare==2.7.2
   make nvflare-sim-smoke    # nvflare simulator, -n 1 -c caltech — validates the transport
                             # contract (DXO round-trip, Executor _setup()/execute()) without
                             # needing all three sites to behave correctly at once
   make nvflare-sim          # once the smoke test passes: -n 3 -c caltech,jpl,office1,
                             # the real 3-site deployment shape
   make clean-nvflare-sim    # wipe nvflare/sim_workspace/ between attempts
   ```
   All four targets are self-contained in `Makefile` (`_check-nvflare-deps` guards the two
   simulator targets the same way `_check-deps` already guards every `experiment-*` target — clear
   error pointing at `make install-flare` instead of a raw `ModuleNotFoundError`). `-w
   nvflare/sim_workspace` (gitignored, wiped and regenerated on every run — separate from
   `nvflare/workspace`, the Containerlab-provisioning workspace above, which has real mTLS certs
   the simulator doesn't need). `nvflare simulator` runs everything in local processes/threads — no
   `nvflare provision`, no Containerlab, no Docker needed for this step (see "Not on Containerlab"
   above). Expect the first run to fail somewhere — that is normal for code that has never executed
   once; the point is to find out *where*, which is far cheaper to do here than after also standing
   up containers.
2. Fix whatever breaks. The two real open `VERIFY:` points after 2026-07-24's documentation-based
   pass are both about the local round-number counters (points 3/6 above) — expect the first crash
   or silent-wrong-result to involve those, or something in the numpy/tensor conversion at the
   `execute()`/`accept()` boundary that no amount of reading could rule out in advance.
3. ~~Write a custom server-side Controller/Aggregator...~~ **Done 2026-07-22** — `ChargeShieldAggregator` (`app/custom/chargeshield_aggregator.py`) wraps `FedAvgAggregator` for the averaging and `PrivacyAuditor`/`ByzantineDetector` for per-round analysis, mirroring `run_ids()`. Not yet done: exporting IDS/Auditor results anywhere structured (currently log-only — see "What is explicitly NOT done yet").
4. ~~Add DP: call `GradientManager.privatize()`/`clip_only()` inside the Executor's `execute()`...~~ **Done 2026-07-22 (fase 3)** — see the "DP wiring" section above for the full client/server split. Both files still only `py_compile`-checked, not executed.
5. ~~Export IDS/Auditor results somewhere structured...~~ **Done 2026-07-22 (fase 4)** — `ChargeShieldAggregator._export_results()` writes the full per-round history to `experiments/nvflare_ids_audit_results_<timestamp>.json` (overwritten only within a single run, across rounds — **not** across different runs, since 2026-07-24, see below). See "What is explicitly NOT done yet" above for the one open verification point (working-directory/deployment assumption).
6. ~~Solve raw-update extraction for LiRA/Shadow...~~ **Done 2026-07-22 (fase 5)**, scoped as an offline step by design — `ChargeShieldAggregator._export_fl_results()` dumps the exact `run_fl_rounds()`-shaped data per round, and `scripts/run_nvflare_mia.py` runs the existing, already-validated `run_lira()`/`run_ids()`/`run_fedmia()` against it unchanged. See the dedicated section above for why this wasn't ported to run live inside `aggregate()`.
7. **Rewritten 2026-07-31 (user-requested: a real multi-container deployment is needed, not just the simulator)** — `containerlab/topology.clab.yml` (moved from the repo root, where it lived misplaced relative to its own header comment and every doc reference to it) rewritten from scratch for the actual 3-real-site architecture instead of the Sprint-5 vintage 4-fictional-cluster + 12-OT-node + separate-auditor/ids/mqtt-broker-container design, which never matched the code that was actually built (PrivacyAuditor/ByzantineDetector run server-side inside `ChargeShieldAggregator`, not as separate network services — confirmed by reading `config_fed_server.json`'s `components[]`). The new topology has exactly 5 nodes matching `nvflare/project.yml`'s real participants: `server`, `caltech`, `jpl`, `office1`, `fl-admin`, star-topology links to `server`, no OT/OCPP/MQTT layer. `Dockerfile.flare` fixed to set `CHARGESHIELD_PROJECT_ROOT=/app` (missing entirely before — without it, `_find_project_root()` fails or silently loads 0 sessions inside a container, the same bug class already found and fixed for `nvflare simulator` on 2026-07-24) and its header comment corrected to stop describing "auditor"/"IDS" as separate container roles.

   **Update 2026-08-02 — job submission attempted, second real bug found and fixed:** with the
   path-resolution fix above applied, all 5 containers deployed and came up `running`
   (`sudo containerlab inspect` confirmed it). The user connected to the FLARE admin console
   (`bash /workspace/startup/fl_admin.sh`, logged in as `admin@chargeshield.local`) and ran
   `submit_job /workspace/jobs/chargeshield_poc`. It failed with `OSError: [Errno 30] Read-only
   file system: '/workspace/jobs/chargeshield_poc/.__nvfl_sig.json'` (the `AttributeError:
   'AdminClient' object has no attribute 'do_submit_job'` printed just above it is a normal
   internal fallback in NVFLARE's CLI dispatcher, not the real error — the actual handler runs via
   `default()`/`push_folder()`). Root cause: `submit_job` writes a signature file into the job
   folder itself before uploading it (`nvflare/lighter/utils.py:sign_folders()`), but
   `topology.clab.yml`'s `fl-admin` node bound `../nvflare/jobs/:/workspace/jobs:ro` — read-only,
   because at design time this bind was assumed to be pure upload source, not something NVFLARE
   writes into first. Fixed by changing that one bind to `:rw`. Side effect worth knowing: this
   writes `.__nvfl_sig.json` files into the real `nvflare/jobs/chargeshield_poc/` directory on the
   host (not just inside the container), regenerated on every submit — added to `.gitignore`
   (`nvflare/jobs/**/.__nvfl_sig.json`) so they don't get committed by accident.

   Because bind mounts are fixed at container creation, this fix requires a redeploy, not just a
   config edit:
   ```bash
   sudo containerlab destroy -t containerlab/topology.clab.yml
   sudo containerlab deploy -t containerlab/topology.clab.yml
   ```
   No need to rebuild the Docker image or reprovision — only the bind changed. Then repeat the
   `fl_admin.sh` → login → `submit_job` sequence.

   **Update 2026-08-03 — job actually ran, third real finding: CPU thread oversubscription, not a
   hang.** With both fixes above applied, the job ran for real: all 3 clients registered, trained,
   and by the time an unrelated OrbStack auto-update rebooted the host VM (see below), 4 of 10
   rounds had completed and exported real data (`experiments/nvflare_ids_audit_results_
   20260802_130114_141883.json` / matching `.pkl`, config confirms `dp_mode=dp-fedavg,
   epsilon=1.0`). Initial concern was that ~5 hours/round was a hang or a deadlock; investigation
   found it wasn't — `docker stats` during a later run showed `caltech`/`jpl` at ~97% CPU (real
   work), `office1` near 0% (its site is much smaller, finished its round already), and client logs
   showed `31404 sessioni caricate da 4 file` for Caltech alone — the real multi-year dataset, not
   the small slice used by earlier smoke tests, so real per-round compute is legitimately larger
   than assumed. `AutoencoderTrainer.train_local()` was re-checked and found efficient (tensor
   conversion and `DataLoader` built once outside the epoch loop, no O(n²) pattern) — ruling out an
   application-level bug. The likely actual cause: no file in the project ever pinned PyTorch's
   thread count. PyTorch defaults to spawning as many threads as the host's logical CPUs (12 on
   this VM), while `topology.clab.yml`'s client nodes are capped to `cpu: 1.0` — under Docker's
   CFS cgroup quota, that mismatch is a well-documented cause of severe throttling/thrashing
   (many threads fighting over a tiny quota), independent of and on top of whatever the "real"
   1-core-limited runtime would be. Fixed by adding `ENV OMP_NUM_THREADS=1` / `ENV
   MKL_NUM_THREADS=1` to `Dockerfile.flare`, matching the `cpu: 1.0` default — if that per-client
   CPU limit is ever raised in `topology.clab.yml`, these should be raised (or removed) to match.
   Requires an image rebuild and a `--reconfigure` redeploy to take effect; not yet independently
   re-measured against a clean unthrottled baseline (no non-Docker environment with 31k+ real
   sessions and a hard 1-CPU limit to compare against), so treat the "this fixes it" claim as the
   leading hypothesis, not confirmed — worth timing round 1 of the next run and reporting back.

   Separately: an OrbStack host-VM auto-update triggered a reboot mid-run, which reset all
   container state (clients re-registered fresh, the job's `RUNNING` status in `list_jobs`
   survived the reboot as stale, inaccurate state). Until this pipeline is stable enough to
   trust for a long unattended run, disable OrbStack's automatic updates for the duration of any
   multi-hour job.

   **Update 2026-08-03 (later) — job completed for real, fourth real finding: no genuine hold-out
   existed for the offline MIA analysis.** With the thread-pin fix applied, a resubmitted job ran
   to completion (all 10 rounds). Before running `scripts/run_nvflare_mia.py` on its dump, a check
   of `datasets/acn/` found every downloaded year for every site already used for training
   (Caltech/JPL 2018-2021, Office1 2019-2021 — `config_fed_client.json`'s `dataset_path` points at
   the parent `datasets/acn` directory, and `ChargeShieldExecutor._setup()` loads every `.json` file
   under each site's subdirectory). `run_nvflare_mia.py` already refused to guess a held-out year
   automatically in this mode (see its own `ValueError` in `main()`) precisely to avoid silently
   picking a file that was actually part of training — a real, not hypothetical, risk. Downloading
   an additional year (2022) was attempted and found not viable — ACN-Data has no further sessions
   available for these 3 sites beyond what's already downloaded.

   Root fix (structural, not a workaround): `ChargeShieldExecutor._setup()` now applies the same
   seed-based 80/20 split `scripts/run_experiments.py::main()` already uses for the local
   simulation — `random.seed(seed); random.shuffle(sessions)`, 80% train / 20% reserved, computed
   independently per site — instead of training on 100% of each site's data. `run_nvflare_mia.py`'s
   `load_client_sessions()` reconstructs the identical per-site split (same file load order, same
   enrichment, same seed) to recover the 20% as a genuine non-member pool, with no external file
   needed; `--holdout-dataset` remains available as an explicit override. This means the job that
   had just completed (trained on 100% of the data, pre-fix) is superseded — it needs to be
   resubmitted with the fixed `chargeshield_executor.py` before its dump is usable for MIA analysis.
   Since this file isn't baked into the Docker image (NVFLARE distributes job code at submission
   time via `deploy_map`), this only requires `submit_job` again — no rebuild, no `containerlab
   --reconfigure`. Verified via `py_compile` on both changed files and the full 83-test suite; not
   yet run for real (pending a fresh `submit_job` and then `run_nvflare_mia.py` against its dump).

   **Update 2026-08-01 — first real attempt, bug found and fixed:** the user ran steps 1-3 for
   real on their Mac (Debian VM via OrbStack). `docker build` and `nvflare provision` succeeded,
   but `containerlab deploy` failed immediately with `stat .../containerlab/nvflare/workspace/
   chargeshield_fl/prod_00/caltech/local: no such file or directory` — note the `containerlab/`
   wrongly prepended to the path. Root cause: **containerlab resolves relative bind-mount source
   paths relative to the directory containing the `.clab.yml` file itself, not relative to the
   directory the command is invoked from.** Every bind in `topology.clab.yml` was written as if
   relative to the repo root (`src/`, `nvflare/workspace/...`, etc.) — correct for readability but
   wrong for containerlab's actual resolution rule, since the file lives in `containerlab/`, one
   level below the repo root. Fixed by prefixing every relative bind source with `../` (e.g.
   `../src/:/app/src:ro`, `../nvflare/workspace/.../caltech/local:/workspace/local:rw`), for all 4
   default binds and all 12 per-node binds (server/fl-admin/caltech/jpl/office1). Verified the
   fixed file still parses as valid YAML with the same 5 nodes and 12 total binds via PyYAML in
   the sandbox (containerlab itself still unavailable there, so the actual `deploy` retry has to
   happen on the user's machine again). This is the first of the "expect the first attempt to
   fail somewhere" candidates below to actually be hit — worth noting it wasn't even on the
   predicted list, which was about provisioning/networking/permissions, not path resolution.

   **Runbook (user reruns `containerlab deploy` with the fixed topology; steps 1-2 already done once above, no need to rebuild the image or reprovision unless something else changed):**
   ```bash
   # 1. Build the image (from repo root)
   docker build -f Dockerfile.flare -t chargeshield-fl:latest .

   # 2. Provision real mTLS startup kits for server/caltech/jpl/office1/admin
   #    (regenerate if nvflare/workspace/ already has a stale highway/urban/etc.
   #    provisioning from before the 3-real-site migration — see Sprint 10e's
   #    note on nvflare/workspace/ reappearing with stale names)
   rm -rf nvflare/workspace
   nvflare provision -p nvflare/project.yml -w nvflare/workspace

   # 3. Deploy the containers (Containerlab needs Linux — run this inside
   #    OrbStack's Linux VM, not natively on macOS)
   sudo containerlab deploy -t containerlab/topology.clab.yml

   # 4. Containers start in idle (no job running yet — see Dockerfile.flare's
   #    CMD). Submit the job from the fl-admin container's NVFLARE admin
   #    console (exact submit_job invocation to be confirmed on first real
   #    attempt — this is the one part of the runbook genuinely unverified
   #    even on paper, since job submission over a provisioned mTLS
   #    federation was never exercised by the simulator-mode work in
   #    steps 1-6, which bypasses provisioning/submission entirely).
   ```
   **Expect the first attempt to fail somewhere** — most likely candidates, in rough order of
   suspicion: (a) `nvflare provision`'s output directory naming (`prod_00` is nvflare's default
   first-provisioning folder name; a second `provision` run without wiping `nvflare/workspace`
   first produces `prod_01`, silently breaking every bind path in the topology that hardcodes
   `prod_00`); (b) the admin console's exact job-submission command/API, untested end-to-end here;
   (c) container-to-container DNS resolution for the `fed_learn_port`/`admin_port` NVFLARE expects
   from `nvflare/project.yml`, which Containerlab's bridged networking may or may not satisfy without
   extra config; (d) file permission mismatches between the containers' user and the bind-mounted
   host directories. None of these were fixable by more reading — they need a real first run,
   exactly like the 3 real bugs the simulator work found on its first execution (Sprint 10f).

## ML Plane reale + Privacy Auditor subscriber (fase 7+8, 2026-08-31)

Due gap identificati da una review esterna e confermati leggendo il codice (non assunti):

1. **ML Plane mai wired qui (fase 7).** Fino a oggi, `chargeshield_aggregator.py` non importava né
   istanziava `MLPlane`/`FLArtifactCollector` — `accept()`/`aggregate()` costruivano
   `_fl_results_history` da variabili Python locali, lo stesso pattern "morto" già corretto nella
   simulazione il 2026-07-22 (vedi README "Relation to Prior Work") ma mai riportato qui. Fix:
   stesse classi (`src/ml/ml_plane.py`) ora istanziate in `_ensure_components()` e wired a
   `GradientManager`/`FedAvgAggregator`; `accept()` — il punto esatto in cui il paper QRS 2026
   colloca il Privacy Auditor — emette l'evento `gradient_upload` direttamente lì, con un livello
   Purdue che dipende da `dp_mode` (1=raw per `dp-fedavg`/`central`, 2=privatizzato per `local`, dato
   che sotto local DP il server non vede mai nulla di meno rumoroso). `aggregate()` ora legge
   `updates_for_fedavg`/`raw_updates`/`raw_global_weights` dal collector invece che da liste locali.
2. **Privacy Auditor invocato imperativamente, non come subscriber (fase 8).** `_run_ids_analysis()`
   calcolava le delta peer-relative a mano e chiamava `auditor.audit()` direttamente. Fix: nuova
   `PrivacyAuditorSubscriber` (`src/auditor/privacy_auditor_subscriber.py`, vedi
   `docs/SISTEMA.md` sezione 6) sottoscritta allo stesso `MLPlane`, reagisce
   all'evento `"aggregation"` invece di essere chiamata da un loop esterno — stessa formula, stessi
   numeri, solo il meccanismo di attivazione cambia.
3. **Fix collegato: sensibilità DP pesata.** `GradientManager.privatize_aggregate()` (chiamato qui
   per `dp_mode="central"`) ora accetta `participant_n_samples` — `FedAvgAggregator.aggregate()`
   popola `AggregatedUpdate.metadata["participant_n_samples"]` e l'Aggregator lo passa a valle. Vedi
   il commento in `gradient_manager.py::privatize_aggregate()` per il perché (limite Office1: FedAvg
   pesa per n_samples, non uniformemente).

**Verificato**: `python3 -m py_compile` su tutti i file toccati (`chargeshield_aggregator.py`,
`privacy_auditor.py`, `privacy_auditor_subscriber.py`, `gradient_manager.py`,
`fedavg_aggregator.py`) + tutti gli 83 test non-torch passano — stesso limite ambientale di sempre
(niente torch/nvflare in questo sandbox) al momento in cui questa sezione fu scritta. I due
job reali completati il 2026-08-04 (citati sopra in questo documento) sono stati prodotti PRIMA di
questo fix. **Aggiornamento (2026-08-31, vedi "Verified 2026-08-31" più sopra) e di nuovo
2026-09-09 (Step A/B, vedi sezione dedicata sotto)**: il wiring del ML Plane/Privacy Auditor
subscriber è stato da allora effettivamente esercitato da run NVFLARE reali sui 3 siti — questo
paragrafo, che diceva "un nuovo run è necessario per confermare il wiring a runtime", è quindi
superato; quel run è stato fatto, più volte.

Steps 1-2 are a few hours of real debugging once someone has `nvflare` installed — done, repeatedly, since 2026-07-24. Steps 4-6 are the "multi-week" part of the original estimate — this document doesn't shrink that estimate, it just gives it a concrete starting point. Step 7 (fase 7+8, ML Plane + Privacy Auditor subscriber) was verified against a real NVFLARE run by 2026-08-31 (see above) and again by the 2026-09-09 Step A/B verification below — no longer "should work by design, not verified."

## Independent end-to-end re-verification (2026-09-09, "Step A" / "Step B", task #77/#81)

With `src/adapters/chargeplace_scotland_adapter.py` added the same day (see README/DeveloperGuide —
unrelated to NVFLARE, a second-dataset adapter, not yet run through any campaign), the team took the
opportunity to re-verify the real Containerlab/NVFLARE path end-to-end against the current codebase,
since the most recent fully-documented real multi-container run above was 2026-08-31 and a fair
amount of unrelated `src/` work had landed since.

- **Step A — fresh redeploy + clean job.** Rebuilt `chargeshield-fl:latest` from `Dockerfile.flare`,
  re-ran `nvflare provision`, redeployed the full 5-node Containerlab topology
  (`server`/`caltech`/`jpl`/`office1`/`fl-admin`) from scratch against the current `main`, and
  submitted `nvflare/jobs/chargeshield_poc/` through the admin console exactly as in the 2026-08-01
  through 2026-08-31 runbook above. Result: a full 10-round FedAvg job completed cleanly, no errors —
  the best single confirmation to date that the whole chain (image build → provisioning → Containerlab
  deploy → job submission → 10 real FL rounds → ML Plane/Privacy Auditor/ByzantineDetector wiring)
  still works end-to-end on top of everything landed since 2026-08-31.
- **Step B — statistical campaign on real containers, not just the simulator.** Building on Step A,
  ran a 5-seed statistical campaign on the real Containerlab deployment with `dp_mode=dp-fedavg`,
  `epsilon=1.0`. All 5 seeds completed cleanly, 10/10 rounds each, zero `byzantine_detected`. 2 of 5
  seeds show one HIGH cosine-similarity alert on `office1` (round 4 and round 7 respectively) — the
  same long-standing, non-NVFLARE-specific pattern documented in `docs/IDS.md` §12. A
  `dp_mode=central` run at the same epsilon (seed 42) also completed cleanly. **Correction
  (2026-09-10)**: the paragraph above previously claimed dp-fedavg and central "are expected to be
  numerically identical at the same epsilon in the current single-process simulation" — this was
  wrong. The pair that is expected (and verified, bit-for-bit, against real experiment data) to be
  numerically identical in the single-process simulation is **dp-fedavg and local**, not central —
  see `docs/MetricsReference_DSN2027.md` for the full architectural explanation (both call the same
  `GradientManager.privatize()` with no real client/server process boundary in the simulation;
  `central` uses the structurally different `privatize_aggregate()` and is not part of that
  identity). Comparing the real central-seed-42 run against the real dp-fedavg-seed-42 run
  round-by-round confirms this directly: round 1 is identical on all three clients, but the two
  runs diverge genuinely from round 2 onward — central is a real, independently-validated mechanism
  on the real deployment, not a duplicate of dp-fedavg.
- **MIA re-analysis on the real pickles** (`scripts/run_nvflare_mia.py`, reusing `run_lira()`
  unmodified): seed 42 (dp-fedavg) done — `mean_lira_auc_roc=0.499`, `privacy_risk=LOW`, no leakage,
  consistent with the simulation. Seeds 123/456/789/1234 and the central run still pending as of
  2026-09-10.
- **Methodological finding (2026-09-10, does not affect the MIA conclusions above, but worth
  recording):** round 1 in every `nvflare_ids_audit_results_*.json` (privacy_score/epsilon per
  client) is byte-identical across all 5 dp-fedavg seeds *and* the central run. Root cause: the
  NVFLARE campaign's `seed` value is only used client-side, for the train/holdout split
  (`chargeshield_executor.py`) and DataLoader shuffle — never to seed the initial global model's
  weights (the persistor always instantiates the same, unseeded `Autoencoder` checkpoint), unlike
  the single-process simulation, which calls `torch.manual_seed(seed)` before model creation. On
  top of that, round 1's clipping uses the absolute weight norm (not a delta, since
  `reference_weights` is `None` at round 1) — dominated by that shared, unseeded checkpoint, hence
  the exact match. From round 2 onward, `reference_weights` reflects genuinely seed-dependent
  training and the values diverge normally. This means round-1 audit-metric variance across seeds on
  NVFLARE is not informative (it's artificially zero) — a caveat for anyone tempted to cite
  cross-seed variance starting at round 1 specifically. It does not touch the actual LiRA/Yeom/Shadow
  MIA results, which read `raw_updates`/reconstruction loss from `nvflare_fl_results_*.pkl`, not this
  audit field.
- **Fixed, not just documented (2026-09-10).** `ChargeShieldAggregator.__init__()` now accepts an
  opt-in `seed` parameter (default `None`, fully backward-compatible) and, when provided, seeds
  `random`/`numpy`/`torch` before returning — which happens before NVFLARE constructs the next
  component in `config_fed_server.json`'s list, the persistor that instantiates the initial
  `Autoencoder`. `config_fed_server.json` now carries a `"seed"` field that must match the client's
  (same "no automatic cross-check" caveat already documented for `epsilon`/`delta`/`dp_mode`). This
  makes round-1 model initialization genuinely seed-dependent on NVFLARE, matching the
  single-process simulation's `torch.manual_seed(seed)` call in `main()`. **Not yet re-verified with
  a real run** — py_compile passes, but torch/nvflare aren't installable in this sandbox; the next
  NVFLARE job submission (whichever seed/mode is used next) is the real test that round 1 now
  differs across seeds. The 5 dp-fedavg seeds and the 1 central seed already collected predate this
  fix and keep their byte-identical round 1 — still valid for the MIA conclusions they support, just
  not for a round-1-specific cross-seed variance claim.
- **Verified with a real run (2026-09-10).** Submitted a job with `seed=999` in both config files
  (first NVFLARE seed ever used other than the paper's five) and compared round 1's
  `raw_global_weights` (the clean pre-DP-noise FedAvg average, from `nvflare_fl_results_*.pkl`)
  against the pre-fix `central`/seed=42 run via `scripts/verify_seed_fix_round1.py`. Every one of the
  22 weight tensors differs (max abs diff 0.006–1.04 depending on layer) — round 1 model
  initialization is now genuinely seed-dependent, confirming the fix works. Caveat worth recording:
  the per-round **audit-JSON epsilon** (`nvflare_ids_audit_results_*.json`, `per_client_audit.*.epsilon`)
  stayed identical across seeds even on this post-fix run (office1=0.053503, caltech=1.0,
  jpl=1.070064, matching the old pre-fix numbers exactly) — this is *not* evidence the fix failed,
  it's a separate, expected property of that particular metric: it reads as a closed-form function of
  static per-client config (participant sample counts, `max_grad_norm`, `delta`), not of the actual
  weight values, so it is not a useful signal for this specific check either way. Anyone re-verifying
  this fix on a future seed should compare `raw_global_weights` directly (as
  `scripts/verify_seed_fix_round1.py` does), not the audit-JSON epsilon.

This supersedes every earlier "not yet attempted"/"next step" framing about the real multi-container
deployment elsewhere in this document (including this document's own opening paragraph, corrected
above): the Containerlab path is not a speculative next step, it is a working, twice-independently-run
deployment target, with a real statistical campaign now underway on it directly (not only via
`make nvflare-sim`, the single-process simulator). **Update (2026-09-11): `central` now has all 5
paper seeds** (42 pre-fix, 123/456/789/1234 post-fix, all ε=1.0, all 10/10 rounds; `mean_lira_auc_roc`
for 123/456 already reanalyzed: 0.5000/0.5015, consistent with the rest of the campaign, 1234's
reanalysis pending as of this write-up). **Update 2026-09-13**: `local` mode on NVFLARE is now also
complete at all 5 seeds (42/123/456/789/1234, ε=1.0) — `mean_lira_auc_roc` per seed 0.4993/0.5007/
0.4991/0.5003/0.5007, all `privacy_risk=LOW`, bootstrap 95% CI [0.4994, 0.5006] (contains 0.5), same
null result as `central` and the single-process campaign (README Sprint 10zz+63). **Still missing for
a comparison fully aligned with the single-process 10-config × 5-seed campaign**: epsilon variation
(only ε=1.0 tested so far — a 5-seed sweep at ε∈{0.5, 0.1} for both `central` and `local` is in
progress as of 2026-09-14, per explicit user request for full rigor). Not blocking for submission —
the paper's primary claim rests on the single-process campaign, already complete and statistically
robust; this NVFLARE work is supplementary validation, not a replacement.

**ChargePlace Scotland on NVFLARE — no longer a structural limitation (2026-09-10, task #37/#93).**
The gap wasn't Containerlab or NVFLARE's provisioning — the 3 NVFLARE site identities
(`caltech`/`jpl`/`office1`, from `nvflare/project.yml`) don't need to change to serve a different
dataset. The actual gap was that `chargeshield_executor.py::_setup()` hardcoded
`from adapters.acn_dataset import ACNDataset`, with no equivalent of `run_experiments.py`'s
`dataset_adapter` dispatch. Fixed: `ChargeShieldExecutor` now accepts `dataset_adapter` (default
`"acn"`, fully backward-compatible) and, when set to `"chargeplace_scotland"`, a `chargeplace_scotland`
dict (`metadata_dir`, `session_files`, and `site_mapping` — mapping each fixed NVFLARE site identity
to a real Scotland council area, e.g. `caltech` → `Glasgow City`). New helper methods
`_load_acn_sessions()`/`_load_chargeplace_scotland_sessions()` replace the single hardcoded block;
`ChargePlaceScotlandDataset.load_with_metadata()` loads the shared monthly pool (all 32 council
areas together — ChargePlace Scotland has no per-site files, unlike ACN-Data) and each client
filters to its own mapped council area. A ready-to-use example,
`config_fed_client_chargeplace_scotland.json` (2 months only, for a first smoke test — same
reasoning as the single-process smoke-test-first plan), sits alongside the ACN one. **Not yet run**
— py_compile passes but torch/nvflare aren't installable in this sandbox; the next `submit_job`
with this config swapped in is the real test. Expect the same wall-clock caveat as the
single-process ChargePlace Scotland run (LiRA is the dominant cost, ~54 min/round there due to the
much larger per-site volume) — untested whether NVFLARE's per-round cost scales the same way.

**Canary positive control on NVFLARE (2026-09-11).** Until now the canary positive control (Carlini
"The Secret Sharer" 2019/Jagielski et al. 2020 — see the "Quanti shadow model..." section of
`docs/MetricsReference_DSN2027.md` and README Sprint 10vv–10zz+18) only existed in the single-process
simulation (`scripts/run_experiments.py::inject_canaries()`). Ported to NVFLARE, on the user's direct
request after asking whether `privacy_risk: LOW` could be hiding a silently-broken attack pipeline:
`ChargeShieldExecutor` now accepts an opt-in `canary` dict (`enabled`/`site`/`n_templates`/
`n_duplicates` — same schema as the single-process YAML, minus `n_nonmember_templates`, which only
matters offline). When `canary["site"]` matches this client's own `cluster_id`, `_inject_canary_members()`
duplicates `n_templates` real local sessions `n_duplicates` times each into the actual local training
set (same `_canary_group`/`_canary_role="member"` tagging as the simulation), right after the 80/20
split and before normalization — the other two sites, receiving the identical `config_fed_client.json`
via `deploy_map="@ALL"`, see `site != cluster_id` and stay untouched (no-op, same default-off guarantee
as `dataset_adapter`). The executor never keeps its own hold-out list, so it cannot construct the
non-member twins itself — those are reconstructed offline instead: `scripts/run_nvflare_mia.py::
load_client_sessions()` now also reads the same `canary` block from the same config file and applies
a mirrored `_inject_canaries_for_site()` per site, drawing `member_templates` from the reconstructed
`site_train` with the identical `random.Random(seed + 271828)` sequence the executor used (verified
with a standalone reproduction, same seed/content: identical 5 template indices selected on both
sides), then `nonmember_templates` from `site_holdout` (a second draw from the same RNG, exactly
mirroring `inject_canaries()`'s original member-then-nonmember order) — giving `run_lira()` the same
`_canary_group`/`_canary_role` tags it already knows how to bucket into `canary_auc_roc`/
`canary_raw_mse_auc_roc`, with zero changes to that function. **Not yet run for real** — py_compile
passes and the RNG-matching logic is verified standalone (pure Python, no torch/nvflare needed for
that part), but no actual `submit_job` with `canary.enabled=true` has been submitted yet; that's the
next step before trusting NVFLARE-side canary numbers the way the single-process ones are trusted.
