# ChargeShield-FL

Studio empirico su differential privacy e membership inference in un autoencoder
federato addestrato su sessioni reali di ricarica per veicoli elettrici (ACN-Data,
tre siti: Caltech, JPL, Office 1). Obiettivo: misurare in quali condizioni la DP
riduce la capacità degli attacchi di membership inference e a quale costo per il
modello. Paper in preparazione per DSN 2027 (abstract 25 novembre 2026, paper 2
dicembre 2026).

## Dove leggere

| documento | cosa contiene |
|---|---|
| [Guida scientifica](docs/ChargeShield_FL_spina_dorsale_consolidata.md) | la linea del lavoro: obiettivo, RQ1-RQ3, fasi, regole. Canonica. |
| [Stato](docs/STATO.md) | risultati verificati, cosa è fatto, cosa manca, glossario. Canonico. |
| [Esperimenti](docs/ESPERIMENTI.md) | cosa lanciare, in che ordine, con quali prerequisiti. Canonico. |
| [Sistema](docs/SISTEMA.md) | cosa esiste davvero nel codice: dati, modello, DP, ML Plane, attacchi, NVFlare. |
| [Segnalazioni tecniche](docs/Segnalazioni_tecniche_2026-09-22.md) | bug e problemi aperti, con file e riga. |
| [Sprint-log](docs/SprintLog.md) | diario di lavoro storico, append-only. |

Documenti di riferimento, da leggere su richiesta: `docs/CanaryPositiveControl.md`,
`docs/VulnerabilitaPerRecord.md`, `docs/LimiteTeoricoDP.md`,
`docs/MetricsReference_DSN2027.md`, `docs/NVFlareIntegration.md`,
`docs/ReadingList_DSN2027.md`, `docs/LiteratureReview.md`.
`docs/ChargeShield-FL_CLAIM.md` è un'ipotesi di contributo metodologico
subordinata alla guida. I config sono indicizzati in `config/README.md`.

**Se lavori con un assistente AI**: dagli `CLAUDE.md` come prima istruzione. Contiene
cosa leggere, dove stanno i numeri e il protocollo di sessione. Claude Code lo
carica da solo.

I numeri vivono solo in `risultati/`: `Matrice_sintesi.xlsx`,
`matrice_run_completati.xlsx`, `matrice_confronti.xlsx`,
`decision_matrix_ACN_membership_DP.xlsx`, `worst_case/*.json`. Sono generati da
`scripts/genera_matrici_faseA.py` dai JSON in `experiments/`, che non è
versionato.

Il paper LaTeX vive su Overleaf; `docs/paper/latex_dsn2027/` è solo un mirror
(regole in `CLAUDE.md`).

## Installazione

```bash
pip install -e ".[dev]"
```

Dipendenze principali: torch, scikit-learn, numpy, pandas, scipy, openpyxl,
pyyaml. Per il record-level DP serve anche `dp-accounting` (da dichiarare in
`pyproject.toml`, segnalazione 4). NVFlare è opzionale: `pip install -e ".[flare]"`.

Dataset: `python3 scripts/download_acn_sessions.py` scarica i file ACN-Data per
sito e anno in `datasets/acn/<sito>/`.

## Una run

```bash
python3 scripts/run_experiments.py --config config/experiment.yaml --rounds 10 --seed 42 \
  --sweep-dir experiments/nodp-sweep2 --no-dp \
  --per-sample-dump experiments/nodp-sweep2/per_sample_seed42.json
```

Flag principali: `--dp-mode {dp-fedavg,central,local}`, `--epsilon`, `--no-dp`,
`--n-shadow`, `--epochs`, `--per-sample-dump`, `--roc-curve-dump-dir`. I target
Makefile `experiment-*-sweep` lanciano i cinque seed in sequenza con lock
anti-concorrenza. Le celle DP a ε in {1, 0.5, 0.1} hanno utility distrutta: gli
esperimenti utili sono in `docs/ESPERIMENTI.md`.

Dopo un sweep:

```bash
python3 scripts/check_significance.py
python3 scripts/genera_matrici_faseA.py
```

## Test

```bash
python3 -m pytest tests/ -q --ignore=tests/test_privacy_auditor_subscriber.py --ignore=tests/test_run_experiments_integration.py --ignore=tests/test_sprint4.py --ignore=tests/test_sprint5.py
```

Baseline attesa senza torch: 297 passed con `datasets/` presente, 264 senza (i 33
restanti leggono i file reali dei dataset). Con torch e dati: `make test`.

## Struttura

```
config/        config YAML per esperimento (rq1_*, rq2_*, canary_*, rdp_*, overfit_*)
scripts/       run_experiments.py (pipeline), check_significance.py, genera_matrici_faseA.py,
               analyze_worst_case_vulnerability.py, worst_case_livello_di_caso.py, run_nvflare_mia.py
src/           core, ml (trainer, DP, FedAvg, ML Plane), auditor, ids, plugins/attacks, adapters
nvflare/       job NVFlare (executor, aggregator, config, snapshot dei seed)
containerlab/  topologia a 5 nodi
risultati/     matrici e report, unica fonte dei numeri
docs/          documentazione, vedi tabella sopra
tests/         suite pytest
```

## Autori

Assunta Imperatrice (IMT Lucca, Università di Napoli Parthenope), Luigi Romano
(Università di Napoli Parthenope). Licenza in `LICENSE`.
