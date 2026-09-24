# ChargeShield-FL — segnalazioni tecniche e bug (revisione del 2026-09-22)

Contesto: revisione indipendente di codice, config, `risultati/*.xlsx`, paper e docs
allo stato del commit `f89c1cc`. Ogni voce indica file e riga; verificala sul codice e
sui JSON prima di correggere, non fidarti della lista. L'ambiente di revisione non aveva
`experiments/`, `logs/`, torch, pytest, quindi nessun numero grezzo è stato rieseguito.

## A. Da fare subito: tocca l'analisi NVFlare in corso

1. **Lo split train/holdout viene ricostruito con il seed sbagliato per 20 run su 25.**
   `scripts/run_nvflare_mia.py:686-700` legge il seed dello split da `--client-config`,
   che di default è il file live `config_fed_client.json` (oggi `seed: 1234`). Il flag
   `--seed` governa solo gli shadow. `scripts/analizza_dump_nvflare.sh` non passa
   `--client-config`, quindi i run per i seed 42/123/456/789 usano una partizione
   membri/non membri diversa da quella dei client reali: l'AUC tende a 0.5 per
   costruzione, senza errore. Nel log compare il warning "Seed ... diverso dal seed
   reale del client NVFLARE ... uso quest'ultimo".
   Fix: aggiungere a ogni riga dello script
   `--client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed<S>_<mode>_eps<E>.json`
   (i 25 snapshot esistono tutti). Scartare i 20 JSON già prodotti senza snapshot.
   **Aggiornamento 2026-09-22:** script rigenerato con `--client-config` e cartelle
   invalide cancellate; rilancio rinviato a dopo lo sweep di ε (vedi `ESPERIMENTI.md`).
   Hardening: registrare `seed` in `meta` del pickle
   (`chargeshield_aggregator.py:1025-1030` oggi salva solo dp_mode/epsilon/delta/max_grad_norm)
   e far fallire `run_nvflare_mia.py` quando `--seed` non coincide con il seed del client
   config, invece di proseguire con quello live.
2. Stesso problema, retroattivo: le run `nvflare-central-seed{123,456,789,1234}` con seed
   nel JSON "DISCORDE" (`risultati/matrice_run_completati.xlsx`) vanno riverificate: il
   registro le tratta come discordanza cosmetica, ma può essere lo stesso disallineamento
   dello split.
3. Per tutte le rianalisi NVFlare resta la discrepanza dichiarata a
   `run_nvflare_mia.py` (blocco "ATTENZIONE — discrepanza nota"): statistiche di
   normalizzazione globali offline contro per-sito nel client reale. Va quantificata o
   allineata, altrimenti i numeri NVFlare restano "indicativi".

## B. Bug che cambiano numeri citati nel paper

4. **`epsilon_record_dp` è sottostimato.** `scripts/run_experiments.py:136-137` legge
   `fl_rounds` e `delta` dalla radice di `cfg`, ma vivono in `cfg["experiment"]`
   (`run_experiments.py:1169`, `:226`): `rounds` vale sempre 1, quindi i passi
   dell'accountant sono 1/3 o 1/10 di quelli reali. Inoltre `n_sessions` è
   `len(train_sessions)` globale (`:7443`), mentre ogni client addestra sul proprio
   sottoinsieme, e l'accountant compone un `PoissonSampledDpEvent` (`:152`) mentre il
   training usa `DataLoader(shuffle=True, drop_last=True)` (`autoencoder_trainer.py:374-377`).
   Stesso bug in `scripts/fix_json_recorddp_fields.py`. L'ε=7.15 a σ=5 e il "~88" a
   σ=1 vanno ricalcolati. `dp_accounting` non è dichiarato in `pyproject.toml` e
   l'ImportError degrada in silenzio a `epsilon_record_dp: None`.
5. **Le celle record-DP sono etichettate no-DP.** Il warning `[NO-DP BASELINE]`
   (`run_experiments.py:1280`) scatta con `--no-dp` anche se `record_dp` è attivo.
   Nel registro (`scripts/genera_matrici_faseA.py`) nessuna delle 237 righe riporta
   `record_dp`: le 10 run record-DP compaiono come `no_dp=True`. Il warning deve
   controllare `record_dp.enabled`; il registro deve emettere il campo.
6. **I due script worst-case non calcolano lo stesso percentile.**
   `scripts/analyze_worst_case_vulnerability.py:95-110` usa mid-rank su membri e non
   membri; `scripts/worst_case_livello_di_caso.py:62-72` usa `bisect_left` sui soli
   membri. Risultati 411 vs 412 (no-DP) e 288 vs 290 (dp-fedavg). La docstring del
   secondo dichiara il criterio "identico". Sceglierne uno e ricalcolare; il foglio
   `Worst_case_per_record` pubblica 412/290. Due righe di quel foglio
   (central-sweep5, local-sweep3) non hanno il JSON grezzo in `risultati/worst_case/`.
7. **dp-fedavg e local sono lo stesso codice in simulazione**
   (`run_experiments.py:1396-1401`): la Tabella 2 ha 7 condizioni, non 10. Il docx lo
   dice, `sections/preliminaries.tex` li presenta ancora come tre punti distinti.
8. **ε=8 e ε=16 hanno un solo seed** e sono le uniche celle con utility non distrutta;
   nel foglio `Utility_privacy_limite` stanno accanto a celle a 14 run senza distinzione.
9. **Deduplica e seed.** `scripts/check_significance.py:281` deduplica per il seed nel
   config JSON; il registro usa il nome cartella. Sei run sono discordi: le due logiche
   danno n diversi.

## C. LiRA: non è un bug da patchare alla cieca, ma va deciso

10. Il floor di varianza simmetrico (`run_experiments.py:4409-4413`, default
    `floor_mode=symmetric`) è colpito nel 97-99% dei record: il log-likelihood ratio
    degenera in una funzione monotona della loss (= `_sablayrolles_score`). Su `central`
    `matched_formula_auc` vale 0.21-0.34 contro uno 0.50 riportato: il null può essere
    cancellazione di due artefatti. `member_scoring` resta `"real"` (asimmetrico,
    `:4662-4666`). Il floor è 10 volte più largo dell'intervallo dei dati perché
    `global_in_stats_per_cluster` mescola record e shadow (varianza fra record, non fra
    shadow).
11. Lo score composto somma i log-LR su un numero variabile di round senza normalizzare
    (`_cumulative_scores[_sid] += lira_score`): campioni scorati in 10 e in 3 round
    finiscono nella stessa classifica. Impatta i dump per-campione del worst-case.
12. `lira_debug_uncalibrated_skip_rate = 0.0000` in tutti i gruppi: il filtro 8σ non
    esclude nulla, perché la soglia è calcolata su σ già floorato. Nel paper è
    presentato come adattamento sostanziale.
13. LiRA non è riproducibile a seed fisso (canary_auc_roc cambia fra due tornate
    identiche), la loss grezza sì. Decidere e scrivere quale è la metrica primaria.

## D. Privacy Auditor

14. `config/auditor.yaml` ha `total_rounds_budget: 1000` su 10 round reali:
    `budget_ratio ≈ 0.01`, le soglie `PRIVACY_BUDGET_*` non possono scattare. Solo
    `auditor_tuned.yaml` (budget 3) è coerente, ed è usato da un config solo.
15. La sensibilità vale 1.000 in ogni round perché la norma L2 è saturata dal clipping:
    telemetria identica fra cella con memorizzazione e cella record-DP. Il docstring
    `src/auditor/privacy_auditor.py:254-270` afferma il contrario ("3-4% di
    max_grad_norm") e va corretto. Il proxy non misura rischio di appartenenza.
16. Il commit `f8bb42a` dichiara `docs/AuditorEvaluation.md` e `MetricsReference §5b`:
    nessuno dei due esiste nel repo.

## E. Canary

17. `run_experiments.py:942-985`: per ogni template non membro vengono inseriti due
    record identici (originale ritaggato più clone); il commento a `:944-947` dice
    "qui non ci sono duplicati". Il pool non membro è raddoppiato artificialmente.
18. Lato NVFlare (`chargeshield_executor.py:380-451`, `run_nvflare_mia.py:150-215`) il
    canary usa ancora la semantica legacy: non membri dall'holdout, niente
    `paired_split`. Un canary su NVFlare riprodurrebbe il confondente già diagnosticato.
19. Il protocollo bilanciato (k=20, swap, baseline, 5 seed) esiste solo su Office 1.
    Caltech e JPL hanno solo run a seed 42, sbilanciati, senza blocco canary nel JSON
    (marcati "incompleta"). Scotland ha il config e nessun run.

## F. Altri bug e incoerenze nel codice

20. `run_experiments.py:2297`: `run_yeom = run_yeom` è un no-op; l'alias `run_shadow`
    promesso in `src/plugins/attacks/shadow.py:10-11` non esiste.
21. `drift_detected` è hardcoded `False` (`run_experiments.py:6471, 6487`) mentre il
    CUSUM lo calcola.
22. `GradientManager.privatize_aggregate` (`src/ml/gradient_manager.py:375-387`) usa
    sensibilità `max_i(n_i/N)·C`: vale per adiacenza add/remove; sotto replace-one manca
    un fattore 2. Dichiarare l'adiacenza scelta.
23. `_compute_sigma` (`gradient_manager.py:214-227`) non controlla ε≤1, condizione del
    bound usato; il codice lo chiama con ε=1000 (`run_experiments.py:6357, 6362`).
24. `krum_threshold=3.5` è calibrato su 4 fette dello stesso sito e mai rivalidato per
    n=5 (`run_experiments.py:6300-6329`); `byzantine_attack.enabled` è false ovunque,
    il ByzantineDetector non è mai stato eseguito end-to-end.
25. NVFlare: al round 1 di dp-fedavg il clipping è sul vettore assoluto
    (`chargeshield_aggregator.py:718-736`); `epsilon` default 1.0 silenzioso lato server
    (`:542-546`); contatori di round locali mai validati contro NVFlare.
26. `config/experiment_overfit_control.yaml` è solo commenti: `yaml.safe_load` dà `None`
    e `load_config` crasha. `config/_smoke_auditor.yaml` è un config sperimentale
    record-DP con nome fuorviante. `config/datasets.yaml` e `config/nodes/cluster_*.yaml`
    sono legacy non letti.
27. `scripts/{repair_json_fields,add_n_sessions_to_save,fix_json_recorddp_fields,add_auditor_telemetry}.py`
    riscrivono `run_experiments.py` con splice testuale e sono già stati applicati:
    rilanciarli corrompe il sorgente. Rimuoverli o proteggerli.
28. File spazzatura tracciati: `wegrvb` e `cqEFG24HF2UPIQWèCOX+a` (log identici, commit
    `f8bb42a`), `*.md.docx` in root, `fix_canary_paired_split.patch`,
    `docs/doc_update_canary.patch`.
29. `experiments/` e `logs/` sono gitignorati: nessun numero è verificabile da un clone.
    Almeno i JSON di sintesi per cella, o i loro hash, vanno versionati.

## G. Documentazione e paper

*Nota 2026-09-22 sera: i punti 30-33 riguardano documenti eliminati nella pulizia del
contesto (Architecture, MLPlane, IDS, CaseStudies, errata, Roadmap); quanto di valido
contenevano è in `SISTEMA.md`. Restano da correggere solo i commenti nel codice citati
al punto 30 e i punti 34-36 sul paper. Il config del punto 26 `_smoke_auditor.yaml` è
stato rinominato `experiment_canary_balanced_recorddp_nm5.yaml`; `datasets.yaml`,
`nodes/` e `overfit_control` sono in `config/_legacy_unused/`; gli script del punto 27
sono in `scripts/_applicati/`.*

30. Conteggio parametri incoerente: 570 (`core/autoencoder.py:72`), 618 con BatchNorm,
    "~650" (`gradient_manager.py:419`), 1200 (`Architecture.md:498`).
31. `docs/MLPlane.md` §5.5 descrive StandardScaler; il codice fa min-max [0,1]. §8 dello
    stesso file è interamente stantio (12 client, 13073 sessioni, output linear).
32. `Architecture.md:467` dice epochs 3 (reale 50); cosine threshold 0.85 nella prosa,
    0.3 nel codice; `IDS.md` abstract parla di zero alert, §12.1 ne conta 36; deadline
    DSN diverse fra `Roadmap.md` e README; Makefile target `logs` punta a un container
    inesistente; `CaseStudies.md` §6 cita make target e script che non esistono.
33. `ChargeShield-FL_errata_paper_vs_codice.md` punto 4 è superato: `check_significance.py`
    legge già `composed_lira_auc_roc`. Va datato o aggiornato.
34. Paper LaTeX: titolo "CChargeShield"; "TO BE VERIFIED" in `introduction.tex:14`;
    `preliminaries.tex` usa T=3 come "operating point" mentre la campagna è a T=10;
    δ confrontato con |D|≈1344 vale solo per Office 1; blocco autori non anonimizzato
    per il double-blind.

## H. Aggiunte dal secondo check di coerenza (2026-09-22, sera)

35. **`sections/validation.tex` riporta i numeri canary PRE correzione.** Righe 103-107
    e 126: 0.7570 / 0.7164, Δ = +0.2367, t = 11.25. Dopo il fix del 2026-09-16 (il pool
    raw non è più filtrato dalla calibrazione LiRA, `CanaryPositiveControl.md` §6.6) i
    valori sono 0.7447 / 0.7152, Δ = +0.2299, t = 9.90. Allineare a Overleaf secondo la
    regola di `CLAUDE.md`.
36. **`sections/setup.tex:128-130` inverte Caltech e JPL sulla copertura dell'identificativo
    utente.** Il paper dice 94% Caltech e 52% JPL; la scheda
    `risultati/decision_matrix_ACN_membership_DP.xlsx` (misura del 2026-09-21) dice JPL
    93.5% e Caltech 52.1%. La cifra "31.452 sessioni identificate" non può essere di
    Caltech, che ne ha 31.404 in totale. Stesso errore in
    `docs/paper/paper_privacy_unit_e_ids.md`.
37. **Le run record-DP con `--no-dp` finiscono nel gruppo "no-DP baseline" e lo
    sostituiscono.** `record_dp` non disattiva il client-level: per una cella record-DP
    pulita serve `--no-dp`, e infatti tutte le run record-DP canary hanno `no_dp=True`.
    Ma `check_significance.py::discover_groups` (righe 96-114) etichetta `no_dp=True`
    come "no-DP baseline" e deduplica per `(label, seed)` tenendo il file più recente:
    una run `rq1-recorddp-nm1` a seed 42 sostituirebbe il seed 42 di `nodp-sweep2` senza
    alcun avviso. Stesso problema in `genera_matrici_faseA.py`. Fix: includere
    `record_dp.enabled` e `noise_multiplier` nell'etichetta del gruppo. Da chiudere
    prima di E-B. Il comando E-B nel vecchio TestRoadmap non aveva `--no-dp`:
    corretto in `ESPERIMENTI.md`.
38. **Non esiste un braccio B1, clipping senza rumore.** La Fase B della guida lo
    richiede per separare l'effetto del clipping da quello del rumore; `central` clippa
    ma rumorizza l'aggregato, e `_compute_sigma` non accetta σ = 0. Serve un flag
    `--clip-only` (client clip, nessun rumore, `epsilon_cumulative_*` a `None`), oppure
    il braccio va dichiarato non eseguito.
39. **33 test leggono i dataset reali senza skip.** `tests/test_acn_dataset.py` e
    `tests/test_chargeplace_scotland_adapter.py` aprono file sotto `datasets/`, che non è
    versionato: in un clone senza dati falliscono con `FileNotFoundError` invece di
    essere saltati (verificato il 2026-09-22: 264 passed, 2 failed, 31 errors). Aggiungere
    `pytest.mark.skipif(not Path(...).exists(), reason=...)` a livello di modulo, così la
    baseline resta leggibile ovunque; la funzionalità va coperta anche da un test su un
    file sintetico piccolo committato in `tests/`.
40. **Commenti e docstring nel codice rimandano a documenti eliminati il 2026-09-22**
    (TestRoadmap, Positioning, CaseStudies, DeveloperGuide, PES_v1): 22 occorrenze in
    `scripts/run_experiments.py`, 5 in `compute_pes.py`, 3 in `check_significance.py`,
    3 in `generate_excel_report.py`, 2 ciascuno in `compare_results.py`,
    `src/core/autoencoder.py`, `src/core/base_attack.py`, `src/plugins/attacks/__init__.py`,
    `src/adapters/chargeplace_scotland_adapter.py`, una in `autoencoder_trainer.py`,
    `gradient_manager.py`, `ml_plane.py`, `privacy_auditor.py`, `fedmia.py`. Non bloccano
    nulla: sostituirli con `docs/SISTEMA.md` o `docs/STATO.md` quando si tocca il file per
    altro, non con un commit dedicato mentre la campagna gira.

## I. Trovato costruendo l'inventario dell'ambiente d'esecuzione (2026-09-22)

41. **`requests` è usato ma non dichiarato in nessuna dipendenza di
    `pyproject.toml`.** `scripts/download_acn_sessions.py:39` fa `import requests`
    a livello di modulo, senza fallback — è l'unico modo nel repo per scaricare
    ACN-Data da ev.caltech.edu. Non compare né in `[project.dependencies]` né in
    nessun `[project.optional-dependencies]` (`dev`, `flare`, `viz`). A differenza
    di `dp-accounting` (segnalazione 4), qui non c'è degradazione silenziosa: chi
    non ce l'ha installato ottiene subito un `ModuleNotFoundError` lanciando lo
    script. Fix: aggiungerlo a `[project.dependencies]` in `pyproject.toml`.
    Verificato leggendo gli `import` reali di tutto `src/` e `scripts/`, non solo
    questo file — nessun'altra libreria di terze parti usata nel codice risulta
    mancante dalla dichiarazione (oltre a `dp-accounting`, già nota).

## J. Trovato preparando la riscrittura del paper (2026-09-23)

42. **Un attacco che fallisce non ferma lo sweep e la cella perde un seed in
    silenzio.** `scripts/run_experiments.py:6174` (`run_registered_attacks`)
    intercetta l'eccezione di ogni attacco registrato, scrive comunque il JSON con
    le metriche a `None` ed esce con codice 0, quindi il `set -e` degli sweep non
    si ferma. Caso reale in E-A: `experiments/rq1-eps2/experiment_20260922_180414.json`
    (epsilon = 2, quarto seed del ciclo, cioe' 789) ha Yeom, Shadow e LiRA falliti
    con `AttributeError: module 'numpy' has no attribute '_no_nep50_warning'`
    (`logs/rq1_eps_sweep.log`, righe 1540-1697, ore 18:04 del 2026-09-22), JSON da
    40 KB contro i 90 KB degli altri e nessun `per_sample_seed789.json`. La causa e'
    d'ambiente (numpy/scipy incoerenti in quel momento: le run successive sono
    andate), ma il codice la rende invisibile. Da fare: rilanciare la sola cella
    epsilon = 2, seed 789, a fine sweep, prima di `check_significance.py`; valutare
    un codice d'uscita non nullo quando falliscono tutti gli attacchi (tocca
    `run_experiments.py`: solo dopo la campagna).

## K. Trovato rigenerando le matrici dopo E-A (2026-09-24)

43. **`genera_matrici_faseA.py` duplicava i fogli a ogni esecuzione.** Le quattro
    funzioni che aggiungono fogli a `Matrice_sintesi.xlsx` (`scrivi_costo_per_sito`,
    `scrivi_utility_privacy`, `scrivi_worst_case`, `scrivi_glossario`, righe 561-718
    prima della correzione) aprono il file esistente e chiamano `create_sheet`: alla
    seconda esecuzione openpyxl crea `Utility_privacy_limite1` accanto al vecchio
    `Utility_privacy_limite`, che resta con i numeri precedenti. Chi apre il foglio
    col nome atteso legge dati vecchi. **Corretto il 2026-09-24**: ogni funzione
    elimina il foglio omonimo prima di ricrearlo; `Matrice_sintesi.xlsx` rigenerato
    da quello committato, sei fogli senza duplicati.
44. **Il foglio `Utility_privacy_limite` non compone le celle come
    `check_significance.py`.** `scrivi_utility_privacy` (stesso script) raggruppa per
    `(dp_mode, epsilon)` TUTTI i JSON di `experiments/*/`, senza escludere le cartelle
    `_*` e `nvflare-*` e senza deduplicare per seed; `check_significance.py` fa
    entrambe le cose. Effetti visti il 2026-09-24: a ε = 8 e 16 la run pilota a un
    seed finisce insieme alle cinque di E-A (n = 6; a ε = 8 la media della loss scende
    da 0.0655 a 0.0566); `local` ε = 0.5 e 0.1 includono una run NVFlare rianalizzata
    il 2026-09-22 (la loss di `local` ε = 0.5 è passata da 0.4235 a 0.4118 senza nuove
    simulazioni); le celle a 14 run sommano sweep diversi dello stesso seed. Va deciso
    insieme alla segnalazione 9 quale regola vale; finché non è deciso, `STATO.md`
    riporta E-A dai JSON, una run per seed.

## L. Trovato preparando E-E (2026-09-24)

45. **Le run no-DP di E-D ed E-E finiscono nel gruppo "no-DP baseline" e ne
    sostituiscono i seed.** `check_significance.py::discover_groups` (righe 273-330)
    etichetta "no-DP baseline" ogni run con `no_dp=True` e senza `record_dp`, senza
    guardare `partition_strategy` ne' `ml.proximal_mu`, che pure sono registrati nel
    JSON, e deduplica per `(etichetta, seed)` tenendo il file piu' recente. Le run di
    `rq2-per_site` e `rq2-iid` (E-D, 22-23 settembre) e quelle di E-E hanno
    `no_dp=True`, gli stessi cinque seed e timestamp successivi a `nodp-sweep2`
    (8 settembre): copiate in `experiments/`, sostituirebbero in silenzio i seed del
    riferimento no-DP di RQ1, e fra loro vincerebbe l'ultima in ordine di timestamp.
    `genera_matrici_faseA.py` fa lo stesso (cella "no-DP" alle righe 379 e 633, e il
    foglio `Utility_privacy_limite` non deduplica, segnalazione 44). Stessa classe
    della 37. Fix: includere `partition_strategy` e `proximal_mu` nell'etichetta del
    gruppo in entrambi gli script (tocca il codice: da autorizzare). Finche' non e'
    corretto, le cartelle di E-D ed E-E restano fuori da `experiments/` (per esempio
    in `experiments_altre_macchine/`) e si analizzano a parte.
