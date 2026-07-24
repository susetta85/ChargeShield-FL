# NVFLARE / Containerlab Integration — Status and Plan

**Started:** 2026-07-22
**Status:** Job scaffold + client Executor + custom Aggregator + DP wiring + structured exports (fase 1-5) written. **First real execution: 2026-07-24** (see "First real run" section below) — 4 real bugs found and fixed across three attempts; **the third attempt completed all 10 rounds successfully** (`make nvflare-sim-smoke`, 1 client/caltech, `min_clients=1`: "Round 9 finished" → "Finished ScatterAndGather Training", no errors). This confirms the DXO/Executor/Aggregator/DP/IDS-export pipeline runs end-to-end for a single client. **Not yet run**: the 3-real-site shape (`make nvflare-sim`, `-n 3 -c caltech,jpl,office1`) — that is the next concrete step, not yet attempted.

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
  app/custom/chargeshield_aggregator.py — wraps FedAvgAggregator + PrivacyAuditor + ChargingIDS (fase 2, 2026-07-22)
```

**Server side** (`config_fed_server.json`) still uses NVFLARE's `ScatterAndGather` workflow (round orchestration: broadcast → wait for clients → aggregate → persist), ma da oggi (fase 2) l'aggregatore built-in `InTimeAccumulateWeightedAggregator` è stato sostituito da `ChargeShieldAggregator` — un `Aggregator` NVFLARE custom che al suo interno chiama le classi **vere e già testate** della simulazione: `FedAvgAggregator` (src/ml/fedavg_aggregator.py) per la media pesata, e `PrivacyAuditor`+`ChargingIDS` (src/auditor, src/ids) per l'analisi privacy/IDS per-round, con la stessa logica di normalizzazione peer-relative (mediana) di `scripts/run_experiments.py::run_ids()`. `PTFileModelPersistor`/`FullModelShareableGenerator` restano built-in (nessun motivo per sostituirli). Da oggi pomeriggio (fase 3) anche `GradientManager`/DP è collegato — vedi sezione dedicata sotto.

Fase 1 (2026-07-22, mattina) era transport-only con l'aggregatore built-in. Fase 2 (stesso giorno, dopo) ha introdotto `ChargeShieldAggregator`. Fase 3 (stesso giorno, pomeriggio) ha aggiunto la DP client-side (Executor) e server-side (Aggregator). Questo documento è stato aggiornato ad ogni passaggio.

## DP wiring (fase 3, 2026-07-22, non verificato)

`dp_mode`/`epsilon`/`delta`/`max_grad_norm` sono ora parametri sia dell'Executor (`config_fed_client.json`) sia dell'Aggregator (`config_fed_server.json`) — devono combaciare tra i due file, nessuna validazione incrociata automatica esiste ancora. Semantica dei 3 `dp_mode`, mirror esatto di quella già implementata e testata in `scripts/run_experiments.py`/`src/ml/gradient_manager.py`:

- **`dp-fedavg`** [McMahan et al. 2017]: il client invia l'update grezzo (nessuna operazione lato Executor). Il server, dentro `ChargeShieldAggregator.aggregate()`, chiama `GradientManager.privatize()` (clip sul delta rispetto a `self._prev_global_weights` + noise) su ciascun update ricevuto, PRIMA di passarlo a `FedAvgAggregator` — il server vede il valore grezzo transitoriamente, coerente con l'architettura "server semi-trusted" del paper originale.
- **`central`**: il client, in `chargeshield_executor.py::execute()`, chiama `GradientManager.clip_only()` (clip sul delta rispetto a `pre_round_weights`, nessun noise) prima di inviare. Il server chiama `GradientManager.privatize_aggregate()` una sola volta sul risultato di `FedAvgAggregator`, aggiungendo un singolo draw di rumore che beneficia della riduzione di sensitività 1/n.
- **`local`**: il client chiama `GradientManager.privatize()` (clip + noise) prima di inviare — il server/IDS non vede mai il valore grezzo, nemmeno transitoriamente.

L'analisi IDS/Auditor (`ChargeShieldAggregator._run_ids_analysis()`) gira sempre su `received_updates`, cioè la vista "più grezza disponibile" per quella modalità: grezza vera per `dp-fedavg`, clippata-non-rumorosa per `central`, clippata+rumorosa per `local` — coerente con la degradazione IDS sotto local DP già documentata per la simulazione single-process.

Entrambi i lati (`chargeshield_executor.py`, `chargeshield_aggregator.py`) sono stati verificati solo con `python3 -m py_compile` — **non eseguiti**, stesso limite ambientale (niente torch/nvflare in questo sandbox) di tutto il resto di questo documento.

## Export strutturato (fase 4) e raw-update extraction (fase 5), 2026-07-22

`ChargeShieldAggregator` ora scrive due file dopo ogni round (path configurabili in `config_fed_server.json`, entrambi sotto `experiments/` — stessa directory, già in `.gitignore`, usata dalla simulazione):

- `experiments/nvflare_ids_audit_results.json` (fase 4): cronologia IDS/Auditor per round, stesso formato di `ids_results` in `run_ids()` — alerts, `byzantine_detected`, `low_similarity_nodes`, più un blocco `per_client_audit` (privacy_score/epsilon/threats_detected).
- `experiments/nvflare_fl_results.pkl` (fase 5): dump **pickle** (non JSON — contiene `GradientUpdate` con `torch.Tensor`) con esattamente lo stesso schema che `run_fl_rounds()` produce in memoria per la simulazione: `mean_loss`, `n_participants`, `updates`, `raw_updates`, `raw_global_weights`, `global_weights` per round.

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

Fixed in `chargeshield_executor.py::_setup()`: `cluster_id` is now derived from `fl_ctx.get_identity_name()` (the NVFLARE site name) when it matches one of the 4 known clusters, falling back to the config value with an explicit warning only if the site name isn't recognized. `config_fed_client.json`'s `"cluster_id": "highway"` remains as the fallback default, not the active path, once `nvflare provision` assigns each site its real name. **Still unverified**: whether `fl_ctx.get_identity_name()` returns the site name in exactly this form at `START_RUN` time — a 4th `VERIFY:`-class assumption, on top of the original 3 below, to check first when this actually runs.

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
3. ~~Write a custom server-side Controller/Aggregator...~~ **Done 2026-07-22** — `ChargeShieldAggregator` (`app/custom/chargeshield_aggregator.py`) wraps `FedAvgAggregator` for the averaging and `PrivacyAuditor`/`ChargingIDS` for per-round analysis, mirroring `run_ids()`. Not yet done: exporting IDS/Auditor results anywhere structured (currently log-only — see "What is explicitly NOT done yet").
4. ~~Add DP: call `GradientManager.privatize()`/`clip_only()` inside the Executor's `execute()`...~~ **Done 2026-07-22 (fase 3)** — see the "DP wiring" section above for the full client/server split. Both files still only `py_compile`-checked, not executed.
5. ~~Export IDS/Auditor results somewhere structured...~~ **Done 2026-07-22 (fase 4)** — `ChargeShieldAggregator._export_results()` writes the full per-round history to `experiments/nvflare_ids_audit_results.json` (overwritten after every round). See "What is explicitly NOT done yet" above for the one open verification point (working-directory/deployment assumption).
6. ~~Solve raw-update extraction for LiRA/Shadow...~~ **Done 2026-07-22 (fase 5)**, scoped as an offline step by design — `ChargeShieldAggregator._export_fl_results()` dumps the exact `run_fl_rounds()`-shaped data per round, and `scripts/run_nvflare_mia.py` runs the existing, already-validated `run_lira()`/`run_ids()`/`run_fedmia()` against it unchanged. See the dedicated section above for why this wasn't ported to run live inside `aggregate()`.
7. Only after 1-6 work against the NVFLARE simulator: wire this same job into `topology.clab.yml`/Docker — fix the orphaned `docker/charging-node` build, point the topology at whatever image actually gets built, reconcile the two PKI trees (`certs/` vs. the NVFLARE-provisioned workspace). Target environment: OrbStack's Linux VM on macOS (Containerlab needs Linux), even though single-process simulation experiments run natively on the physical machine for speed. **Found 2026-07-24, still true, not fixed (correctly out of scope until steps 1-6 pass)**: the existing `topology.clab.yml` is misplaced (lives at the repo root, not under `containerlab/` where its own header comment and every reference to it — including this document — assume) and is Sprint-5 vintage: 12 fictional OT "charging nodes" (`highway-01..03`/`urban-01..03`/etc.) plus 4 fictional FL clients (`highway`/`urban`/`residential`/`corporate`), OCPP/MQTT protocol simulation that was never wired to anything real, and Docker images (`chargeshield-fl:latest`, `chargeshield/charging-node:latest`) with no confirmed working build. None of this matches the current 3-real-site (`caltech`/`jpl`/`office1`) design, and rewriting it now would be guessing at a container topology before step 1 even confirms the simulator-level job works — left as-is deliberately, flagged here so nobody deploys it by mistake before it's rewritten as part of step 7.

Steps 1-2 are a few hours of real debugging once someone has `nvflare` installed. Steps 4-6 are the "multi-week" part of the original estimate — this document doesn't shrink that estimate, it just gives it a concrete starting point.
