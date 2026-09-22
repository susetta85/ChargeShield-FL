# ChargeShield-FL — Il sistema com'è davvero

> **Stato: documento OPERATIVO.** Scritto il 2026-09-22 estraendo da Architecture,
> MLPlane e ThreatModel (eliminati) le sole parti che corrispondono al codice al
> commit `f89c1cc`. Descrive cosa gira, non cosa era stato progettato. Per la linea
> scientifica vedi la guida; per lo stato dei risultati `STATO.md`.

## 1. Cosa esiste e cosa no

| componente | stato | dove |
|---|---|---|
| federazione a 3 client reali, Caltech / JPL / Office 1 | reale | `config/experiment.yaml` sezione `sites`, `scripts/run_experiments.py::group_indices_by_site` |
| autoencoder 6-16-8-4-8-16-6 | reale | `src/core/autoencoder.py` |
| training locale FedProx, DP-SGD opzionale per record | reale | `src/ml/autoencoder_trainer.py` |
| DP client-level, tre placement | reale | `src/ml/gradient_manager.py` |
| FedAvg pesato per campioni | reale | `src/ml/fedavg_aggregator.py` |
| ML Plane, hub eventi e collector | reale | `src/ml/ml_plane.py`, `src/ml/base_ml.py` |
| Privacy Auditor come subscriber | reale, contatore di budget | `src/auditor/` |
| ByzantineDetector, Krum, coseno, CUSUM | implementato, mai eseguito end-to-end | `src/ids/charging_ids.py` |
| attacchi Yeom, Shadow, LiRA via registro | reale | `src/plugins/attacks/`, funzioni in `run_experiments.py` |
| canary positive control | reale | `run_experiments.py::inject_canaries` |
| deployment NVFlare a 5 container | reale, verificato end-to-end | `nvflare/`, `containerlab/topology.clab.yml` |
| adapter ChargePlace Scotland | reale, una run di prova | `src/adapters/chargeplace_scotland_adapter.py` |
| topologia 12 nodi / 4 cluster, OCPP, MQTT, WireGuard | mai costruita | solo nei documenti eliminati |
| `fedmia.py`, `flare_connector.py`, `ocpp16_adapter.py`, `base_node.py`, `charging_node.py`, `elaadnl_dataset.py` | codice morto o non collegato | banner nei file |
| PluginRegistry da filesystem, AggregationStrategy | mai implementati | — |

## 2. Dati

ACN-Data, tre siti reali, tutti gli anni disponibili: Caltech 31 404 sessioni,
JPL 33 629, Office 1 1 680, totale 66 713. Ogni sessione porta il proprio
`site_id`; i client sono i siti. Split 80/20 train/holdout casuale con seed,
oppure `entity_aware` per stazione. Normalizzazione min-max in [0, 1] calcolata
sul solo training. Sei feature continue: energia erogata, potenza media (chiamata
`max_power_kw` nel codice, è kWh su ore), energia richiesta, finestra dichiarata,
ora di connessione localizzata al fuso del sito, durata. Nel regime canary si
aggiunge `start_time_epoch`. Nessuna etichetta di intrusione: `anomaly_label` è
nullo ovunque.

## 3. Modello e training

Autoencoder denso 6-16-8-4-8-16-6, 570 parametri lineari più 48 di normalizzazione,
BatchNorm nell'encoder (GroupNorm obbligatoria con record-DP), Sigmoid in uscita,
MSE. Adam a 1e-3, batch 32. FedProx con mu = 0.01: il termine prossimale è attivo
dal round 2. Protocollo ordinario: 10 round, 50 epoche locali, tutti i client a
ogni round. Regime canary: 3 round, 1000 epoche, capacità (32, 16)/8.

## 4. Differential privacy: due unità, un solo caso con tre punti di osservazione

**Client-level** (`GradientManager`). A fine round l'update del client, cioè il
delta rispetto ai pesi ricevuti, viene clippato a norma L2 pari a `max_grad_norm`
= 1 e rumorizzato con σ = C·sqrt(2·ln(1.25/δ))/ε, δ = 1e-5, buffer di BatchNorm
esclusi. È weight perturbation con 50 epoche dentro il round: l'ε per round è un
parametro di calibrazione, non una garanzia verificata. Composizione riportata
come naive T·ε e come avanzata di Dwork e Roth; a T = 10 l'avanzata non è mai più
stretta.

| placement | chi clippa | chi rumorizza | cosa vede l'aggregatore |
|---|---|---|---|
| dp-fedavg | server, per client, all'arrivo | server, per client | A1, l'update grezzo |
| central | client | server, una volta sull'aggregato, σ scalato per max n_i/N | A2, clippato senza rumore |
| local | client | client | A3, clippato e rumorizzato |

In simulazione single-process dp-fedavg e local eseguivano lo stesso codice e
davano numeri identici fino al 15 settembre; da allora LiRA sotto dp-fedavg
attacca l'update grezzo. Su NVFlare i due placement inviano payload diversi sulla
rete. A1 è un avversario più forte del threat model: un insider sul client.

**Record-level** (`AutoencoderTrainer._train_step_record_dp`, `ml.record_dp` nel
config). DP-SGD: microbatch da 1, clipping del gradiente per esempio, rumore
gaussiano σ·C sulla somma, termine prossimale aggiunto dopo il rumore. Protegge la
stessa unità che gli attacchi misurano. Budget calcolato con `dp_accounting`
(RDP, evento Poisson) nel campo `epsilon_record_dp`; l'accountant ha un bug aperto
(segnalazione 4). Non disattiva il client-level: per una cella pulita serve
`--no-dp`. I tre punti di osservazione collassano in uno.

## 5. ML Plane

Ogni componente di training espone `emit_event` e `subscribe`; `MLPlane` è l'hub,
`FLArtifactCollector` raccoglie per round gli update grezzi (livello Purdue 1),
quelli privatizzati (livello 2) e l'aggregato, con semantica "ultimo vince" per
(round, nodo). `run_fl_rounds` costruisce i risultati leggendo dal collector, non da
variabili locali. Tre consumatori: Privacy Auditor e ByzantineDetector in linea,
la suite di attacchi offline sugli stessi `round_data`. Niente qui è specifico del
dominio EV. Nel paper è infrastruttura, non contributo.

## 6. Privacy Auditor

Subscriber dell'evento di aggregazione. Per ogni nodo calcola la norma L2
dell'update normalizzata sulla mediana dei pari, la converte in uno
pseudo-ε per round, accumula e produce un `privacy_score`; quattro allerte
(esplosione del gradiente, budget quasi esaurito, esaurito, sensibilità
sospettosamente bassa). Non è contabilità DP: dipende dai dati. Nella campagna la
sensibilità vale 1.000 per costruzione, la telemetria non distingue una cella con
memorizzazione da una senza, e con `total_rounds_budget` = 1000 su 10 round le
soglie di budget non scattano. Overhead misurato 0.3-0.7 ms per round. Va
descritto come contatore di budget.

## 7. Attacchi

Tutti derivano il giudizio dalla loss di ricostruzione del campione.

| attacco | superficie | scorer | note |
|---|---|---|---|
| Yeom | modello globale | −loss | nessuna calibrazione; è la metrica che regge |
| Shadow | modello globale | loss shadow − loss target, un solo shadow su metà dei membri | invertito nel regime canary |
| LiRA | update del singolo client | log-rapporto di verosimiglianza gaussiano, n_shadow = 16 per sito, shadow riaddestrati ogni round con warm start e privatizzati come il client | floor di varianza colpito nel 97-99% dei casi: degenera in una funzione monotona della loss; punteggio composto sommato sui round |

Metriche per tutti: AUC-ROC, TPR a FPR in {0.1%, 1%, 5%} con caso a TPR = FPR,
advantage di Youden, matrice di confusione, curva ROC completa a richiesta,
scorer di Sablayrolles e in scala log come varianti. Analisi per record:
percentile del `composed_score` dentro il seed, record segnalato se membro in
almeno 2 seed con percentile medio ≥ 90 e minimo ≥ 75, livello di caso per
permutazione dentro il seed.

Canary: `n_templates` sessioni reali del sito duplicate `n_duplicates` volte,
gruppo taggato atomico negli shadow, non membri estratti dallo stesso pool e
spostati nell'holdout (`paired_split`), scambio dei ruoli valido solo con gruppi
bilanciati, baseline a inizializzazione casuale.

## 8. Threat model

Scenario 1, l'unico valutato: aggregatore honest-but-curious, segue il protocollo
e registra ciò che riceve. Cosa riceve dipende dal placement, tabella in 4.
Yeom e Shadow modellano un partecipante qualunque che vede il modello globale;
LiRA modella l'aggregatore che vede l'update di un client. Le shadow di LiRA si
addestrano sui dati reali del sito, membri e holdout: è un accesso più ampio di
quello del threat model, quindi la misura è un limite superiore generoso.

Scenario 2, client malevolo: dominio del ByzantineDetector, mai eseguito su un
attacco reale, fuori dal paper. Scenari 3 e 4, gateway compromesso e
intercettazione: fuori scope. Scenario 5, attaccante adattivo: solo discussione.
Nessun meccanismo di integrità offre garanzie di confidenzialità e viceversa.

## 9. Deployment NVFlare

Job `nvflare/jobs/chargeshield_poc`: `ChargeShieldExecutor` lato client riusa
`AutoencoderTrainer`, carica tutti gli anni del proprio sito, fa lo split 80/20
con il seed del config; `ChargeShieldAggregator` lato server incapsula FedAvg,
GradientManager, Auditor e ByzantineDetector con l'ML Plane. Ogni round esporta
un JSON di audit e un pickle con updates, raw_updates e pesi globali; l'analisi
MIA è offline con `scripts/run_nvflare_mia.py`, che riusa le stesse funzioni della
simulazione. Topologia Containerlab: server, caltech, jpl, office1, fl-admin, mTLS.
Differenze note dalla simulazione: normalizzazione per sito lato client contro
globale nell'analisi offline; clipping assoluto al round 1 in dp-fedavg; nessun
`--no-dp`; il seed dello split va letto dallo snapshot del job, non dal config
live (segnalazione 1). Il seed governa il modello globale solo dai job successivi
al 2026-09-10.

Per lanciare una run e aggregare i risultati vedi il README; per gli esperimenti
previsti `ESPERIMENTI.md`.
