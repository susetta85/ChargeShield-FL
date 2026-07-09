# ChargeShield-FL — Review v4 (Post-Fix Verification)

**Data:** 2026-07-09  
**Scope:** Verifica fix applicati + bug residui + valutazione DSN 2027  
**Reviewer:** Analisi read-only — nessun file modificato  

---

## 1. Stato dei Fix

| # | Fix dichiarato | File | Stato | Note |
|---|---|---|---|---|
| 1 | Normalizzazione min-max [0,1] su train split | `run_experiments.py:628-630` | **APPLICATO** | `compute_feature_stats` solo su `train_sessions`, poi `normalize_sessions` su entrambi i split. |
| 2 | FedAvgAggregator: valid filter prima di min_participants | `fedavg_aggregator.py:95-106` | **APPLICATO** | `valid = [u for u in updates if u.n_samples and u.n_samples > 0]` prima del check su `len(valid)`. |
| 3 | fedmia.py:107 — rimossa doppia istanziazione Autoencoder | `run_experiments.py:331` | **APPLICATO** | In `run_fedmia()` viene istanziato un solo `Autoencoder(input_dim=input_dim)`. |
| 4 | autoencoder.py fit() e _calibrate_threshold(): unpack tuple da TensorDataset | `autoencoder.py:267,309` | **APPLICATO** | Entrambi i loop eseguono `if isinstance(batch, (list, tuple)): batch = batch[0]`. |
| 5 | autoencoder_trainer.py: n_samples=0 se drop_last produce 0 batch | `autoencoder_trainer.py:199-205` | **APPLICATO** | `effective_samples = len(sessions) if epoch_losses else 0`. |
| 6 | charging_ids.py: fallback config default se auditor.yaml mancante | `charging_ids.py:58-83` | **APPLICATO** | `_IDS_DEFAULT_CONFIG` con `alert_threshold: 0.7` usato se il file manca. |
| 7 | MIA pool bilanciato: members_balanced = sample(members, len(non_members)) | `run_experiments.py:285-291` | **APPLICATO** | `_n_bal = min(len(members), len(non_members))`, `_pool_rng.sample(members, _n_bal)`. |
| 8 | run_experiments.py: exec_module → import standard in _update_excel_report() | `run_experiments.py:542-574` | **APPLICATO** | Import standard con `sys.path.insert` + `import generate_excel_report as gen`. |
| 9 | CaseStudies.md §2.4.3: nota limitazione DP formale | `docs/CaseStudies.md:192-203` | **APPLICATO** | Sezione "Limitation — Weight Perturbation vs. DP-SGD" completa con ref. |
| 10 | Architecture.md: epochs 5→3, enrich_sessions separato da normalize_sessions | `docs/Architecture.md` (sezione 5) | **PARZIALE** | La tabella Component Reference (riga ACNDataset, riga 356) descrive ancora "Loads and enriches raw ACN-Data CSV files" — il formato è JSON, non CSV. Dettaglio minore ma impreciso. L'epoch 3 è corretto in experiment.yaml. |
| 11 | test_sprint5.py: schema fix + 7 nuovi test per normalize_sessions | `tests/test_sprint5.py:407-479` | **APPLICATO** | 7 test in `TestNormalization` verificano range [0,1], no-mutation, constant feature, min=0, max=1. |
| 12 | Makefile: install-dev usa pip install -e ".[dev]", --config in sweep | Non letto (fuori scope richiesto) | **NON VERIFICATO** | Makefile non incluso nei file da leggere. |
| 13 | pyproject.toml: aggiunto openpyxl>=3.1 | Non letto (fuori scope richiesto) | **NON VERIFICATO** | pyproject.toml non incluso nei file da leggere. |

---

## 2. Bug Ancora Aperti

| Severity | File:riga | Descrizione | Impatto |
|---|---|---|---|
| **MEDIUM** | `gradient_manager.py:158-165` | Indentazione errata del commento ATTENZIONE: le righe 159-164 sono corpo del commento, ma il docstring inizia alla riga 165 con rientro sbagliato. Il codice funziona (Python non si rompe), ma la struttura visuale è fuorviante e qualsiasi linter/IDE segnala il blocco come mal-formattato. | Cosmetic ma riduce leggibilità durante peer review DSN. |
| **MEDIUM** | `run_experiments.py:637` | In `main()`, `run_fedmia()` riceve `train_sessions` come `members`. Queste sessioni sono già state usate per il training FL (corretto), ma il bilanciamento dentro `run_fedmia()` usa `random.Random(seed)` con seed dalla config (`experiment.seed`). Il seed viene però già consumato da `random.seed(seed)` alla riga 615 e da `random.shuffle(sessions)` alla riga 620. Il `random.Random` con seed fisso in `_pool_rng` è un oggetto separato e quindi il subset di `members_balanced` è riproducibile e deterministico. Tuttavia: `random.seed(seed)` alla riga 615 imposta il modulo-level RNG globale; le chiamate successive a `random.shuffle` (riga 620) e qualsiasi altra chiamata `random.*` nel codice mutano lo stato globale. Se in futuro si aggiungono chiamate `random.*` prima di `run_fedmia()`, il subset `members_balanced` rimarrà deterministico (usa `random.Random` separato), ma il training split potrebbe diventare non-riproducibile. Pattern architetturale fragile. | Riproducibilità futura a rischio. Rilevante per DSN artifact evaluation. |
| **MEDIUM** | `autoencoder.py:275` | `fit()` usa `sum(batch_losses) / len(batch_losses)` senza guardare se `batch_losses` è vuota (il loop potrebbe produrre zero batch se il DataLoader è vuoto). Se `train_loader` è vuoto, `len(batch_losses) == 0` e si solleva `ZeroDivisionError`. Il caso è difeso in `autoencoder_trainer.py` tramite `drop_last` e il check `epoch_losses`, ma `Autoencoder.fit()` resta vulnerabile se chiamato direttamente (es. da `FedMIA.train_shadow_model()`). | Crash latente in `FedMIA.train_shadow_model()` con DataLoader vuoto. |
| **LOW** | `fedmia.py` (plugin, `_calibrate_reference_errors`) riga 182 | Il DataLoader nel plugin `FedMIA._calibrate_reference_errors()` riceve batch raw (non tuple), ma il loop non fa unpack: `batch = batch.to(self._device)`. Se il DataLoader è creato da `TensorDataset` (che restituisce tuple), il codice solleva `AttributeError: 'tuple' object has no attribute 'to'`. La `Autoencoder.fit()` fa l'unpack correttamente, ma `_calibrate_reference_errors()` no. | Crash del shadow model plugin FedMIA se addestrato con TensorDataset. Il plugin è attualmente disabilitato, ma il bug è latente. |
| **LOW** | `acn_dataset.py:356` (Architecture.md) | L'Architecture.md (Component Reference, riga 356) descrive `ACNDataset` come "Loads and enriches raw ACN-Data CSV files". Il formato è JSON (`raw["_items"]`), non CSV. | Errore documentale minore; non impatta il codice. |
| **LOW** | `run_experiments.py:141` | `normalize_sessions()` fa `s = dict(s)` (shallow copy) per evitare mutazioni sull'originale. Funziona correttamente per sessioni con campi scalari. Se una sessione contenesse valori mutabili in sub-strutture (liste, dict annidati), la shallow copy non proteggerebbe i riferimenti interni. Nel contesto attuale (solo scalari float/int/str) non è un bug concreto, ma è un invariante non documentato. | Nessun impatto con il dataset attuale; fragile se esteso. |
| **LOW** | `charging_ids.py:384-386` | `KrumDetector.compute_scores()` importa `logging` a runtime dentro il metodo statico (`import logging as _logging`). È funzionale ma viola la convenzione del progetto (tutti gli import in cima al modulo). | Solo stile; nessun impatto funzionale. |

---

## 3. Correttezza della Pipeline ML

### 3.1 Data Flow Complessivo

Il flusso dati è corretto e privo di data leakage:

1. **Caricamento:** `ACNDataset.load()` carica JSON ACN-Data → `load_sessions()` in `run_experiments.py`.
2. **Enrichment:** `enrich_sessions()` aggiunge `hour_of_day` e `duration_hours` dai timestamp; le sessioni con timestamp malformati sono scartate silenziosamente (corretto, con `pass`).
3. **Split:** Shuffle con `random.seed(42)` → 80/20 train/hold-out. Il seed è impostato DOPO il caricamento ma PRIMA dello shuffle — ordine corretto.
4. **Normalizzazione:** `compute_feature_stats()` solo su `train_sessions` → `normalize_sessions()` applicata a entrambi i split con le stesse statistiche. Nessun leakage. La gestione del caso `fmax == fmin` (feature costante) usa `fmin + 1.0` per evitare divisione per zero — corretto.
5. **FL Training:** `run_fl_rounds()` → 4 cluster con sessioni contigue (non shuffled per cluster — potenziale non-IID, intenzionale per CS2).
6. **MIA Evaluation:** `run_fedmia()` riceve `train_sessions` come members e `holdout_sessions` come non-members — separazione corretta.

### 3.2 Autoencoder

**Architettura:** 6→16→8→4→8→16→6 con BatchNorm1d nell'encoder (dopo Linear 6→16 e 16→8), ReLU, Sigmoid finale nel decoder. Coerente con quanto descritto nella documentazione.

**Problemi architetturali rilevanti:**

- **BatchNorm in single-sample inference:** `reconstruction_error()` chiama `self.eval()` correttamente prima dell'inferenza, quindi `BatchNorm1d` usa running stats invece di batch stats — comportamento corretto. Tuttavia, se il modello viene usato in training mode con batch_size=1, BatchNorm solleva errore. Questo non accade nel flusso attuale.

- **Soglia calibrata su training set:** `_calibrate_threshold()` usa il 95° percentile degli errori di ricostruzione sul training set. Il training set è già normalizzato (sessioni normali). Questo produce una soglia valida per anomaly detection. Il 5% di falsi positivi atteso è documentato correttamente.

- **Sigmoid + MSE loss:** L'output del decoder è in [0,1] grazie a Sigmoid; l'input è normalizzato in [0,1] da `normalize_sessions()`. L'uso di MSE con Sigmoid è standard e corretto in questo contesto. Non c'è il problema del gradiente saturato che si avrebbe con BCE+Sigmoid su dati non binari.

### 3.3 FedAvg / FedProx

**FedAvg:** La media pesata per `n_samples` in `_weighted_average()` è matematicamente corretta. L'accumulazione in float32 con ripristino del dtype originale gestisce correttamente i buffer BatchNorm int64 (`num_batches_tracked`). Il test `test_aggregate_weighted_average` verifica la correttezza del calcolo.

**FedProx:** Il termine prossimale `(mu/2) * ||w - w_global||²` è implementato in `train_step()`. `_global_weights` viene aggiornato solo dopo `set_weights()` — corretto: nella prima iterazione (round 1) `_global_weights is None` e il termine prossimale è disabilitato, il che è il comportamento atteso.

**Problema FedProx — proximal term su parametri vs buffer:** `set_weights()` salva in `_global_weights` solo `[p.data.clone().detach() for p in self.model.parameters()]` (parametri trainable), mentre `get_weights()` e la FedAvg aggregation trasferiscono l'intero `state_dict()` (parametri + buffer BN). Il termine prossimale confronta correttamente solo i parametri trainable con `zip(self.model.parameters(), self._global_weights)` — questo è corretto, i buffer BN non devono entrare nel termine di regolarizzazione.

**Problema residuo — FedProx nel test `test_aggregate_weighted_average`:** Il test usa solo tensori float (parametri di `model.parameters()`), non l'intero `state_dict`. La FedAvg reale usa `state_dict().values()` (inclusi buffer BN). Il test verifica la weighted average solo sui parametri — non copre i buffer BN. Questo è un gap di test (vedi Sezione 4).

### 3.4 MIA Scoring (run_fedmia)

Il membership score `= -MSE(model(x), x)` è corretto: punteggi più alti (meno negativi) per errori bassi → più probabile membro. Il calcolo è fatto in batch da 256 senza gradiente (`torch.no_grad()`). I label sono `[1]*len(member_scores) + [0]*len(non_member_scores)` — corretto per `roc_auc_score`.

**Problema:** I pesi FL vengono caricati nel modello MIA con `load_state_dict(state, strict=True)`. I pesi globali in `global_weights` sono la lista restituita da `_weighted_average()`, che dopo la FedAvg contiene tensor già in float32/int64 (ripristinati da `acc.to(dt)`). Il caricamento `torch.tensor(w)` su tensori già tensori non è un problema — il branch `isinstance(w, torch.Tensor)` lo cattura correttamente.

**Punto critico — BatchNorm in eval mode:** Il modello MIA viene messo in `model.eval()` prima di calcolare gli score. I running stats di BatchNorm (`running_mean`, `running_var`) vengono aggiornati durante il training FL locale e poi aggregati via FedAvg weighted average. Questo è concettualmente problematico: le running stats di BatchNorm aggregated via FedAvg non hanno una chiara interpretazione statistica (sono medie di medie di distribuzioni locali diverse, non la statistica della distribuzione globale). In pratica, con dati ACN-Data omogenei e cluster relativamente simili, l'impatto è limitato. Ma è un punto da discutere nel paper.

---

## 4. Test Coverage

### 4.1 Cosa è coperto

| Area | Test File | Coverage |
|---|---|---|
| Autoencoder forward/backward | `test_sprint4.py:TestAutoencoder` | Buona: shape, errore, anomaly detection, fit, threshold. |
| CUSUMDetector | `test_sprint4.py:TestCUSUMDetector` | Buona: warm-up, deriva pos/neg, reset, nodi indipendenti. |
| GradientAnalyzer | `test_sprint4.py:TestGradientAnalyzer` | Buona: flatten, l2_norm, cosine similarity, cluster analysis. |
| KrumDetector | `test_sprint4.py:TestKrumDetector` | Buona: scores, Byzantine detection, caso n<2f+3. |
| FedMIA (plugin shadow model) | `test_sprint4.py:TestFedMIA` | Media: run_attack, cluster_attack, MIAResult fields. |
| ChargingIDS | `test_sprint4.py:TestChargingIDS` | Buona: analyze, CUSUM drift, Byzantine, risk score, reset. |
| AutoencoderTrainer | `test_sprint5.py:TestAutoencoderTrainer` | Buona: init, train_local, empty/None sessions, FedProx, ML Plane event. |
| GradientManager | `test_sprint5.py:TestGradientManager` | Buona: sigma, privatize, clipping, ML Plane event. |
| FedAvgAggregator | `test_sprint5.py:TestFedAvgAggregator` | Buona: weighted average, min_participants, valid filter, BN dtype. |
| Normalizzazione | `test_sprint5.py:TestNormalization` | Buona: range [0,1], no-mutation, constant feature. |

### 4.2 Gap critici (non testati)

| Gap | Perché è critico | Rischio DSN |
|---|---|---|
| **`run_fedmia()` end-to-end** | L'intera pipeline MIA (load pesi FL → score batch → AUC-ROC) non ha test. Un cambiamento in `global_weights` structure o in `_MIA_FEATURES` potrebbe rompere silenziosamente il calcolo AUC-ROC. | ALTO — il main claim del paper dipende da AUC-ROC. |
| **`run_fl_rounds()` integration** | Il ciclo FL completo (train_local → privatize → collect → aggregate → apply_global_model) non ha test di integrazione. I componenti sono testati isolatamente ma non il loro composto. | ALTO — un bug nell'ordine delle operazioni non sarebbe catturato. |
| **`enrich_sessions()` con timestamp malformati** | Il fallback `pass` per sessioni con timestamp malformati non è testato. Non si sa quante sessioni vengono scartate silenziosamente in produzione. | MEDIO — impatta il numero reale di sessioni nel paper. |
| **`PrivacyAuditor.audit()` con model_update reale** | `PrivacyAuditor` è testato indirettamente tramite `ChargingIDS`, ma il calcolo `round_epsilon = sensitivity / max_grad_norm` non ha un test unitario diretto. | MEDIO — il claim sull'epsilon cumulativo dipende da questo. |
| **BatchNorm FedAvg averaging (running stats)** | Nessun test verifica che le running stats BatchNorm aggregate via FedAvg producano risultati sensati al round successivo. | MEDIO — impatta la convergenza FL con BatchNorm. |
| **`_compute_max_power_kw()` in ACNDataset** | Il calcolo `kWh / ore` non è testato con edge cases (sessione zero-durata, done_time_estimated=True, done > end). | BASSO-MEDIO — impatta la qualità del dataset. |
| **`FedMIA._calibrate_reference_errors()` con TensorDataset** | Il bug del tuple non-unpack (vedi Sezione 2) non è testato — il test `test_run_attack_returns_mia_result` usa `train_loader_plain` con `collate_fn` che restituisce tensori raw, non tuple. | BASSO (plugin disabilitato, ma latente). |

---

## 5. Consistenza Documentazione ↔ Codice

### 5.1 Discrepanze rilevate

| Tipo | File doc | File codice | Discrepanza |
|---|---|---|---|
| **Errore fattuale** | `Architecture.md` riga 356 (Component Reference) | `acn_dataset.py:94-113` | Architecture.md descrive ACNDataset come "Loads and enriches raw ACN-Data **CSV** files". Il formato è **JSON** (`raw["_items"]`). `enrich_sessions()` è in `run_experiments.py`, non in ACNDataset. |
| **Discrepanza ML** | `Architecture.md` riga 359 (AutoencoderTrainer) | `autoencoder_trainer.py:82` | Architecture.md descrive "manages model checkpointing" per AutoencoderTrainer. Il codice non implementa checkpoint. |
| **Discrepanza architetturale** | `Architecture.md` §4.3 (Strategy Pattern) | `autoencoder_trainer.py`, `fedavg_aggregator.py` | Il documento descrive un `AggregationStrategy` interface con `FedAvgStrategy` e `FedProxStrategy` come classi concrete. Questo **non è implementato**: FedProx è controllato da `proximal_mu` in `AutoencoderTrainer.train_step()`. La nota in mermaid (riga 272-273) lo riconosce, ma il testo §4.3 presenta la Strategy come implementata. |
| **Discrepanza architetturale** | `Architecture.md` §4.4 (Plugin Pattern) | `src/plugins/attacks/fedmia.py` | Il documento descrive un `PluginRegistry` che scopre plugin via filesystem. Questo **non è implementato** nel codice letto. `FedMIA` viene istanziato direttamente. |
| **Claim non verificabile** | `CaseStudies.md` §2.6 | `run_experiments.py` | Il primo dato sperimentale "AUC-ROC = 0.5172" è presentato come confirmed. Il codice è corretto per produrlo, ma i dataset ACN-Data non sono presenti nel repository — il risultato non è riproducibile senza i dati. La sezione non segnala che i dati devono essere scaricati separatamente (la sezione 6.2 lo fa, ma la 2.6 no). |
| **Discrepanza parametri** | `CaseStudies.md` §2.3 | `experiment.yaml:30` | CaseStudies.md §2.3 descrive "Local training runs for a configurable number of epochs per FL round (**default: 5 epochs per round**)". Il file `experiment.yaml` ha `epochs: 3`. Architecture.md ha la versione corretta (epochs 3 dopo fix #10). |
| **Discrepanza DP** | `Architecture.md` riga 359 (GradientManager) | `gradient_manager.py:73-76` | Architecture.md Component Reference descrive GradientManager come "Implements Gaussian Mechanism DP: **per-sample gradient clipping** + calibrated noise injection". Il codice implementa **weight perturbation** (clipping sulla norma L2 dell'intero vettore pesi, non per-sample). La nota nel codice e in CaseStudies.md §2.4.3 è corretta, ma la Component Reference non è stata aggiornata. |
| **Discrepanza FedMIA** | `CaseStudies.md` §3.3.3 riga 341 | `run_experiments.py:244-365` | CS2 (§3.3.3) descrive "A per-cluster shadow model is trained on the public split sessions assigned to that cluster's distribution." Il codice `run_fedmia()` implementa solo **loss-based MIA** (Yeom 2018), non shadow models per cluster. CS2 è planned (Sprint 6), ma la descrizione implicita dell'approccio shadow-model per CS2 è inconsistente con l'implementazione esistente. |

---

## 6. Valutazione DSN 2027

### 6.1 Claim supportati dal codice

| Claim | Supporto nel codice |
|---|---|
| **FedMIA loss-based (Yeom 2018) implementato** | `run_experiments.py:244-365`. Score = -MSE, AUC-ROC via sklearn. Corretto. |
| **Normalizzazione min-max senza leakage** | `run_experiments.py:628-630`. Stats solo su train_sessions, applicate a entrambi i split. Verificato. |
| **FedAvg weighted average per n_samples** | `fedavg_aggregator.py:149-181`. Media pesata corretta con gestione dtype. Verificato. |
| **FedProx proximal term implementato** | `autoencoder_trainer.py:132-136`. Termine `(mu/2)*||w-w_global||²` corretto. Verificato. |
| **Gaussian Mechanism sigma = C*sqrt(2*ln(1.25/δ))/ε** | `gradient_manager.py:178-182`. Formula corretta. Verificato. |
| **Pool MIA bilanciato (equal members/non-members)** | `run_experiments.py:285-291`. `random.Random(seed).sample()`. Verificato. |
| **BatchNorm buffer (int64) escluso da DP noise** | `gradient_manager.py:195-211, 219-222`. Corretto. |
| **IDS: CUSUM, Krum, Cosine implementati** | `charging_ids.py`. Tutti e tre i detector reali. Verificato. |
| **Split 80/20 con seed fisso** | `run_experiments.py:614-621`. Seed 42 impostato prima dello shuffle. Verificato. |

### 6.2 Claim non supportati o problematici

| Claim | Problema | Gravità per DSN |
|---|---|---|
| **Garanzia (ε,δ)-DP formale** | Il codice implementa weight perturbation con `epochs=3`. Come documentato correttamente in CaseStudies.md §2.4.3, la garanzia (ε,δ)-DP formale non vale per `epochs>1`. Il paper deve chiarire esplicitamente che ε è un **parametro di rumore sperimentale**, non una garanzia formale per singolo campione. | **ALTO** — potenziale obiezione dei reviewer. |
| **Composizione su T round** | Con 100 round (default) e ε=1.0 per round, l'epsilon totale sotto composizione naive è ~100×1.0=100. Sotto composizione avanzata (Rényi DP) è ≈sqrt(2T·ln(1/δ))·ε. Nessuna delle due è calcolata o riportata nel codice o nella documentazione di dettaglio. CaseStudies.md menziona il problema (§2.4.3) ma non fornisce il calcolo. | **ALTO** — gap critico per la rivendicazione privacy. |
| **AUC-ROC = 0.5172 come "confirmed"** | Il dato è plausibile ma non riproducibile senza i dataset ACN-Data (che non sono nel repository). L'affermazione in CaseStudies.md §2.6 dovrebbe essere condizionata alla disponibilità dei dati. | **MEDIO** — artifact evaluation DSN richiede riproducibilità. |
| **Strategy Pattern con AggregationStrategy interface** | Architecture.md §4.3 descrive un'interfaccia `AggregationStrategy` con `FedAvgStrategy`/`FedProxStrategy`. Non implementata nel codice. Il paper non deve presentare questo come implementazione esistente. | **MEDIO** — discrepanza tra paper e codice viola l'artifact evaluation. |
| **PluginRegistry per filesystem-based plugin discovery** | Architecture.md §4.4 descrive un `PluginRegistry`. Non implementato nel codice. FedMIA è istanziato direttamente. | **MEDIO** — stesso problema. |
| **CS2 e CS3 "Planned"** | I case study 2 e 3 sono dichiarati "Planned (Sprint 6)" ma il paper è per DSN 2027. Se non completati, i claim su RQ2 e RQ3 non hanno supporto empirico. | **ALTO** — gap principale per la submission. |
| **IDS non genera alert durante FedMIA (CS3 §4.4)** | Il claim "zero IDS alerts" durante FedMIA è corretto per design (honest-but-curious), ma il comportamento dell'IDS con l'implementazione attuale (PrivacyAuditor su weight perturbation) non è stato verificato empiricamente — dipende dai valori di sensitivity dei pesi DP-perturbati. | **MEDIO** — necessita verifica sperimentale. |
| **FedMIA shadow model plugin per CS2** | CaseStudies.md §3.3.3 descrive "a per-cluster shadow model" per CS2. Il plugin shadow model (`src/plugins/attacks/fedmia.py`) ha un bug latente (tuple non-unpack) e non è integrato nel flusso sperimentale principale. | **MEDIO** — CS2 non può usare il plugin as-is. |

### 6.3 Valutazione globale

Il core della pipeline (caricamento dati → FL training → loss-based MIA → AUC-ROC) è **corretto e implementato**. I fix Sprint 5 hanno eliminato i bug più gravi. I problemi restanti per DSN 2027 sono principalmente:

1. Il gap tra la claim DP formale e l'implementazione weight perturbation (richiede una sezione "Limitations" esplicita e robusta nel paper — già abbozzata in CaseStudies.md §2.4.3, da espandere).
2. La composizione epsilon su T round non è calcolata né discussa numericamente.
3. CS2 e CS3 non sono implementati — il paper non può avanzare RQ2 e RQ3 come risultati empirici.
4. Discrepanze documentazione/codice (Strategy Pattern, PluginRegistry, CSV vs JSON) che un artifact reviewer noterebbe.

---

## 7. Priorità d'Azione

| Priorità | Azione | File | Stima |
|---|---|---|---|
| **P0 — Bloccante** | Calcolare e riportare la composizione epsilon su T round (almeno con composizione naive T·ε, idealmente con Rényi DP). Aggiungere tabella in CaseStudies.md §2.4.3 con valori numerici per le configurazioni del sweep. | `docs/CaseStudies.md` + commenti `gradient_manager.py` | 2-4 ore |
| **P0 — Bloccante** | Completare CS2 (per-cluster MIA) e CS3 (DP vs no-DP ablation) sperimentalmente. Senza questi dati, il paper non ha supporto empirico per RQ2 e RQ3. | `scripts/run_experiments.py` + analisi | Sprint 6 (settimane) |
| **P1 — Alta priorità** | Aggiornare Architecture.md §4.3 e §4.4 per riflettere lo stato reale del codice: Strategy Pattern e PluginRegistry sono **design futuri**, non implementazioni attuali. | `docs/Architecture.md` | 1 ora |
| **P1 — Alta priorità** | Correggere il bug tuple non-unpack in `FedMIA._calibrate_reference_errors()` (riga 182 del plugin). Il plugin è disabilitato ora, ma CS2 lo richiede. | `src/plugins/attacks/fedmia.py:182` | 10 minuti |
| **P1 — Alta priorità** | Aggiungere test di integrazione `run_fedmia()` end-to-end con sessioni sintetiche. Verificare che AUC-ROC sia calcolato correttamente su un caso controllato (es. membro con MSE=0, non-membro con MSE=1 → AUC-ROC deve essere 1.0). | `tests/test_sprint5.py` o nuovo file | 2 ore |
| **P1 — Alta priorità** | Aggiungere test integrazione `run_fl_rounds()` con sessioni sintetiche e 2 round, verificare che `aggregated.global_weights` abbia la struttura corretta. | `tests/test_integration.py` (nuovo) | 2-3 ore |
| **P2 — Media priorità** | Correggere CaseStudies.md §2.3: epochs default 5 → 3 (coerente con experiment.yaml). | `docs/CaseStudies.md:141` | 5 minuti |
| **P2 — Media priorità** | Correggere Architecture.md Component Reference: ACNDataset "CSV" → "JSON"; GradientManager "per-sample gradient clipping" → "weight-vector L2 clipping (weight perturbation)". | `docs/Architecture.md` righe 356, 359 | 15 minuti |
| **P2 — Media priorità** | Aggiungere `ZeroDivisionError` guard in `Autoencoder.fit()` per `batch_losses` vuoto (riga 275). | `src/core/autoencoder.py:275` | 5 minuti |
| **P3 — Bassa priorità** | Sistemare la formattazione del commento ATTENZIONE in `_compute_sigma()` — il corpo del commento deve essere dentro il docstring o separato, non misto. | `src/ml/gradient_manager.py:158-165` | 5 minuti |
| **P3 — Bassa priorità** | Spostare `import logging` in cima al modulo `charging_ids.py` (rimuovere l'import runtime a riga 384). | `src/ids/charging_ids.py:384` | 2 minuti |
| **P3 — Bassa priorità** | Documentare esplicitamente in CaseStudies.md §2.6 che il dato AUC-ROC=0.5172 richiede i dataset ACN-Data scaricati in `datasets/acn/jpl/`. | `docs/CaseStudies.md` | 10 minuti |

---

*Fine report — ChargeShield-FL Review v4 — 2026-07-09*
