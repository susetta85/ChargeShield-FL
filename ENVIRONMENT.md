# ChargeShield-FL — Prerequisiti d'ambiente

> Nota di posizionamento: `CLAUDE.md` §1 dice esplicitamente "non creare nuovi
> documenti in `docs/`: si aggiornano STATO, ESPERIMENTI, SISTEMA o le
> Segnalazioni". Questo file è quindi in root, non in `docs/`. Se preferisci
> che il contenuto viva altrove (es. dentro `SISTEMA.md`, o assorbito nella
> sezione "Installazione" del `README.md`), dimmelo e lo sposto.

Costruito il 2026-09-22 leggendo `pyproject.toml`, `Makefile`, `Dockerfile.flare`,
`Dockerfile.node`, `containerlab/topology.clab.yml` e gli `import` reali in
`src/` e `scripts/` (non dedotto da `README.md`/`CLAUDE.md` da soli). Non è un
documento canonico del protocollo scientifico: è un indice tecnico, va tenuto
aggiornato a mano se le dipendenze cambiano.

## 1. Python

- Richiesto: **>=3.10** (`pyproject.toml`, `requires-python`).
- Osservato in uso su questa macchina, dai bytecode cache in `src/**/__pycache__`:
  **3.10, 3.12 e 3.14** — il codice ha girato con successo su più versioni, ma
  nessun file del repo fissa quale sia quella "ufficiale" per riprodurre i
  numeri pubblicati (nessun `.python-version`, nessun pin stretto).
- Nei container Docker (`Dockerfile.flare`, `Dockerfile.node`): `python:3.10-slim`.

## 2. Dove vivono le dipendenze

- Fonte canonica: **`pyproject.toml`**. Non esiste un `requirements.txt` né un
  `environment.yml` nel repository.
- Installazione: `pip install -e ".[dev]"` (runtime + pytest/ruff/mypy) o
  `pip install -e "."` (solo runtime) — sempre con `--break-system-packages`
  su questa macchina (i target Makefile `install`/`install-dev`/`install-flare`
  lo passano già). Extra opzionali: `.[flare]` (NVFlare), `.[viz]` (matplotlib).

## 3. Dipendenze runtime core (`pyproject.toml`, `[project.dependencies]`)

| pacchetto | vincolo dichiarato | dove/perché |
|---|---|---|
| numpy | >=1.24 | preprocessing, tutte le pipeline |
| pyyaml | >=6.0 | lettura `config/*.yaml` |
| torch | >=2.0 | autoencoder, training — **sempre CPU-only**, nessun percorso di codice referenzia CUDA/MPS |
| scikit-learn | >=1.3 | `sklearn.metrics.roc_auc_score`/`roc_curve`, 8 punti in `run_experiments.py`, tutti import lazy dentro le funzioni |
| pandas | >=2.0 | adapter dataset (ACN-Data, ChargePlace Scotland) |
| openpyxl | >=3.1 | tutti i report `.xlsx` in `risultati/` |
| scipy | >=1.10 | `scipy.stats.wilcoxon` in `check_significance.py`, import lazy con fallback al sign-test se assente (dichiarato dal 2026-09-14; prima mancava e degradava in silenzio) |

## 4. Extra opzionali (`[project.optional-dependencies]`)

| extra | pacchetto | uso |
|---|---|---|
| `dev` | pytest>=7.0, pytest-cov>=4.0, mypy>=1.0, ruff>=0.1 | test e lint |
| `flare` | nvflare>=2.7.2 | deployment federato reale a 5 container; versione pinnata perché il codice custom in `nvflare/jobs/chargeshield_poc/` è scritto contro le API/semantiche esatte di 2.7.2 |
| `viz` | matplotlib>=3.7 | solo `scripts/plot_roc_log_scale.py`, import lazy — nessun altro modulo del progetto dipende da matplotlib |

## 5. Dipendenze usate nel codice ma NON dichiarate in `pyproject.toml`

Verificate leggendo gli `import` reali in `src/` e `scripts/`, non assunte:

- **`requests`** — `scripts/download_acn_sessions.py:39`, import a livello di
  modulo, nessun fallback. Serve solo per scaricare ACN-Data da
  ev.caltech.edu; non compare né nelle dipendenze core né in nessun extra.
  Chi clona il repo e lancia questo script senza averlo installato a parte
  ottiene un `ModuleNotFoundError` diretto (non degrada in silenzio, a
  differenza del caso sotto). Non era già in `Segnalazioni_tecniche`: la
  aggiungo come nuovo punto 41 in quel file, come da protocollo.
- **`dp-accounting`** — dichiarato in `pyproject.toml` dal 2026-09-24
  (segnalazione 4). Import lazy in `src/ml/record_dp_accounting.py`, che calcola
  `epsilon_record_dp` (RDP accountant per la DP a livello di record). Se manca,
  la run prosegue ma scrive un warning `[RECORD-DP] 'dp-accounting' non
  installato` nel log e lascia i campi a `None`; si ricalcolano dopo con
  `scripts/ricalcola_epsilon_record_dp.py`. `tests/test_record_dp_accounting.py`
  fallisce in raccolta se manca. Serve solo per E-B; non è sul percorso di E-A.
  Su un ambiente gia' installato, bloccando la versione di numpy presente:
  `python3 -m pip install dp-accounting "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')"`.
  Il solo `pip install dp-accounting` sul Mac del 2026-09-24 ha aggiornato numpy da
  1.26.4, la versione delle run gia' prodotte, a 2.5.3 (segnalazione 52).

## 6. Versioni pinnate "note-funzionanti" (da `Dockerfile.flare`)

Confermate buildate con successo end-to-end il 2026-09-09 su una VM Debian/aarch64
via OrbStack. Non sono il vincolo dichiarato in `pyproject.toml` (che usa `>=`),
ma sono l'unico punto del repo dove un set di versioni esatte è stato
effettivamente installato insieme e verificato:

| pacchetto | versione pinnata |
|---|---|
| torch | 2.3.1 (CPU-only, da `download.pytorch.org/whl/cpu`) |
| nvflare | 2.7.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.5.1 |
| pandas | 2.2.2 |
| pyyaml | 6.0.2 |
| cryptography | 42.0.8 |

Nota su torch: va installato con `--extra-index-url
https://download.pytorch.org/whl/cpu`, non `--index-url` — quest'ultimo
forza *tutte* le dipendenze transitive pure-Python di torch (typing-extensions,
jinja2, sympy, networkx, fsspec, filelock) sullo stesso indice, che non le ha
tutte come wheel prebuilt per ogni piattaforma; ha fatto fallire il build reale
del 2026-09-09 (dettagli nel commento in `Dockerfile.flare`).

## 7. Variabili d'ambiente

| variabile | dove serve | note |
|---|---|---|
| `ACN_TOKEN` | `scripts/download_acn_sessions.py` | token personale da ev.caltech.edu, solo in env, mai come argomento CLI |
| `CHARGESHIELD_PROJECT_ROOT` | esecuzione NVFlare (simulator e container) | senza, `_find_project_root()` può fallire silenziosamente e caricare 0 sessioni; impostata dal Makefile per il simulator, va impostata a mano per un `docker run` diretto |
| `CHARGESHIELD_MIN_CLIENTS` | smoke test NVFlare | override a 1 solo per `nvflare-sim-smoke`; il deployment reale usa `min_clients=3` (i 3 siti) |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `Dockerfile.flare` | fissate a 1: senza, torch apre un thread per core logico dell'host mentre ogni container client è limitato a 1 CPU in Containerlab — mismatch che ha causato throttling osservato (~5h/round invece di minuti) |

## 8. Dataset

- Non versionato in git (`datasets/` è in `.gitignore`).
- Si scarica con `python3 scripts/download_acn_sessions.py` (richiede
  `requests` e `ACN_TOKEN`, vedi sopra).
- Senza `datasets/`: 288 test passano (264 prima del 2026-09-24, piu' i 13 di
  `tests/test_record_dp_accounting.py`, i 5 di `tests/test_costo_e_per_record.py` e i 6 di
  `tests/test_etichetta_cella.py`),
  33 falliscono con `FileNotFoundError`
  (`test_acn_dataset.py`, `test_chargeplace_scotland_adapter.py`) invece di
  essere skippati — segnalazione 39, ancora aperta.
- Con `datasets/` scaricato: 321 test passano nella suite non-torch (comando di
  `CLAUDE.md` §5; 297 prima del 2026-09-24). `make test`, con torch, raccoglie 448
  test: il 2026-09-24 sul Mac 447 passati e 1 fallito, un test con dati non
  normalizzati (segnalazione 52, corretto: il file ora passa 23 su 23).

## 9. Sistema operativo e hardware

- Le run prodotte su macchine diverse da quella degli esperimenti stanno in
  `experiments_altre_macchine/` (non versionata), con un file `PROVENIENZA.txt`: oggi
  E-D dal secondo Mac (Python 3.13). Lo stesso config sui due Mac non da' numeri
  identici sul percorso no-DP (segnalazione 55).

- Sviluppo/esperimenti: macOS — gli sweep usano `caffeinate` (comando
  macOS-specifico) per impedire lo sleep durante run multi-ora.
- Windows (PC con i9, 2026-09-24, E-E): la simulazione gira, ma con i thread di
  default di torch il round 1 durava 6 min contro 57 s sul Mac; con
  `OMP_NUM_THREADS=1` e `MKL_NUM_THREADS=1` 1 min 56 s, loss identiche alla terza
  cifra. Stessa causa della riga `OMP_NUM_THREADS` della sezione 7. Al posto di
  `caffeinate` serve `SetThreadExecutionState` di kernel32; i comandi usati sono in
  `ESPERIMENTI.md`, sezione E-E. La macchina non e' registrata nei JSON.
- Deployment NVFlare/Containerlab: **non nativo su macOS**, gira nella VM
  Linux di OrbStack (`sudo containerlab deploy`). Richiede Docker + Containerlab
  + CLI `nvflare` (per `nvflare provision`) installati separatamente — nessuno
  dei tre è un pacchetto Python di `pyproject.toml`.
- Nessun requisito GPU in nessun percorso: torch è sempre CPU-only, sia in
  locale sia nei container.
- `Dockerfile.node` (nodi OT OCPP/MQTT: `paho-mqtt`, `pyyaml`, `numpy`,
  `pandas`) descrive un'infrastruttura **mai costruita end-to-end**
  (`SISTEMA.md` §1: "topologia 12 nodi... mai costruita"). Incluso qui solo
  per completezza dell'inventario: non è sul percorso critico di nessun
  esperimento RQ1-RQ3.

## 10. Comandi di verifica rapidi

```bash
python3 --version
pip install -e ".[dev]" --break-system-packages
python3 -c "import torch, numpy, sklearn, pandas, openpyxl, yaml, scipy; print('core OK')"
python3 -m pytest tests/ -q \
  --ignore=tests/test_privacy_auditor_subscriber.py \
  --ignore=tests/test_run_experiments_integration.py \
  --ignore=tests/test_sprint4.py \
  --ignore=tests/test_sprint5.py
```

Baseline attesa: 315 passed con `datasets/` presente e torch installato, 282
senza (`CLAUDE.md` §5). Richiede `dp-accounting`.
