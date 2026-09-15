# Test Roadmap — DSN 2027 (creato 2026-08-27)

Stato di tutti i test necessari prima della submission: cosa va ripetuto per validare i risultati
post-fix (README Sprint 10x–10dd), cosa è nuovo e necessario per rispondere ai punti sollevati
dalla revisione esterna del 2026-08-27, e cosa è opzionale/stretch entro la deadline abstract
**2026-11-25** (paper: 2026-12-02). Aggiornare le checkbox qui sotto man mano, non ricreare il file.

## Vista d'insieme

| # | Test | Stato | Priorità | Blocca la submission? |
|---|---|---|---|---|
| 1 | Campagna 5-seed × 10-config (bootstrap CI) | ✅ **completata 2026-09-08** (task #52, sostituita/assorbita da #13 sotto) — tutte e 10 le config (dp-fedavg×3/central×3/local×3/no-DP), 5/5 seed, metriche complete (Advantage/Confusion/curve ROC/dump worst-case). Tabella finale in `docs/MetricsReference_DSN2027.md` §8 | **Must** | No — completato |
| 2 | Sanity-check positivo (overfitting) | ✅ **Chiuso 2026-08-31 su office1, chiuso cross-site 2026-09-02 (task #35 completato)** — 5 assi naturali negativi + canary insertion positivo a livello raw-loss (0.70-0.90 office1, post-fix 10zz+16 canary_auc 0.6375/0.6625/0.5875 composto 0.6875). Replica caltech (AUC 0.42-0.50, raw-loss 0.35-0.44) e replica jpl (AUC 0.5789/0.2632/0.6842 composto 0.536842, raw-loss 0.4737/0.4632/0.4947) **entrambe NON riproducono il segnale** — pattern coerente di scale-dependence (duplicati ~11% a office1 vs ~0.5% a caltech/jpl), non un'anomalia specifica di un sito. Vedi README Sprint 10zz+4/10zz+18 e `docs/DSN2027_Positioning.md` punto 8 (risolto 2026-09-02). | **Must** | Sì (soddisfatto — cross-site riconciliato) |
| 3 | Fix `check_significance.py` (mapping cartella→epsilon) | ✅ fatto (2026-08-27) | Must | Sì (era silenziosamente rotto) |
| 4 | Metrica TPR@low-FPR | ✅ **completato 2026-08-28** — si applica solo ai run futuri | **Should** | Fortemente consigliato |
| 5 | Test di significatività Wilcoxon | ✅ **completato 2026-08-28** — trovato e corretto anche un bug reale di conflazione dati in `check_significance.py` | Should | No, ma richiesto dal revisore |
| 6 | Replica su secondo dataset EV (ChargePlace Scotland) | 🟡 **adapter completato e testato 2026-09-09** (task #37, `src/adapters/chargeplace_scotland_adapter.py`, 30 test) — wiring in `run_experiments.py` + primo run ancora da fare, vedi nota sotto | Da decidere | Vedi nota sotto |
| 7 | Sweep IDS/Byzantine n=5 (task #50) | ⚪ mai eseguito | Nice-to-have | No |
| 8 | Seconda run del deployment reale ContainerLab | 🟡 **Step B in corso 2026-09-09** — dp-fedavg ε=1.0 × 5 seed (42/123/456/789/1234) completati sul deployment reale, tutti 10/10 round puliti (nessun byzantine_detected/threats_detected). File rinominati con seed nel nome in `experiments/nvflare_fl_results_seed<N>_*.pkl`. Analisi MIA offline (`run_nvflare_mia.py`) in corso. **dp_mode=central ε=1.0** lanciata subito dopo (stesso ε, per verificare se l'equivalenza dp-fedavg/central osservata in simulazione regge anche nel deployment reale). **Pianificato successivamente**: variazione di ε (0.1, il punto di privacy più forte già validato in simulazione) sullo stesso dp_mode, 1-2 seed — stessa logica "validazione ridotta", non un nuovo CI statistico formale | Nice-to-have | No |
| 9 | Confronto con gradient clipping adattivo (vs soglia fissa attuale) | ⚪ non iniziato, nuovo 2026-08-27 | Da decidere | No, ma richiesto da una revisione esterna |
| 10 | Modernizzazione completa di `docs/ThreatModel.md` (conteggi/cluster/versioni obsoleti in tutto il documento) | ⚪ non iniziato, trovato 2026-08-27 | Da decidere | No — ma contiene una vera contraddizione già corretta (vedi nota) |
| 11 | Controllo centralizzato-vs-federato (capacità vs FL-regolarizzatore, task #40) | ✅ **run a budget pieno completato 2026-09-01** — centralized_control_auc_roc=0.5003 vs LiRA composto 0.5030 vs Yeom 0.4981, stessa banda 0.48-0.54. Vedi README Sprint 10zz+6, `docs/DSN2027_Positioning.md` punto 11. ⚠ **Il numero di smoke test precedente (0.496674, 20 epoche) è SUPERSEDED — non citare, usare solo il run a 500 epoche.** ⚠ Singolo seed — serve trattamento multi-seed (task #43) prima di essere citabile nel paper. | Should | Consigliato, rafforza la sezione validazione |
| 12 | FedMIA-gradient (attacco post-hoc opt-in, task #39/#42/#44/#48) | ✅ **Chiuso 2026-09-09 (task #48).** Storico: fix del pooling (Sprint 10zz+8) verificato con run reale — ipotesi cross-cluster smentita (Sprint 10zz+9); normalizzazione L2 (Sprint 10zz+17/10zz+20) ha eliminato la separazione grossolana di scala ma lasciato un residuo molto rumoroso (AUC 0.0–0.93 a seconda del round/cluster, n_test=16/round troppo piccolo per interpretare). **Risoluzione finale (Sprint 10zz+50)**: pooling cross-round (Sprint 10zz+21) verificato con run reale a 10 round/n_shadow=32 — il pool composto (n_test=160/cluster) converge in modo pulito a livello del caso in tutti e 3 i cluster (caltech=0.511, jpl=0.510, office1=0.508), confermando che il rumore era un problema di campione piccolo, non un confondimento strutturale. **Non citabile nel paper** (attacco opt-in, mai nel registry di default) ma la domanda diagnostica aperta è ora chiusa. Storico completo degli INVALIDI in README Sprint 10zz-10zz+9/10zz+20/10zz+21/10zz+50. | Nice-to-have | No — attacco opt-in, non nel registry di default, nessun risultato pubblicato dipende da esso |
| 13 | **Rilancio campagna paper con metriche complete (task #50/#52/#53/#54)** | ✅ **completato 2026-09-08** — 10/10 step, 6485 minuti totali (~4.5 giorni), via `scripts/run_multiseed_consolidation.sh`. Include Sablayrolles (#58) e TPR@low-FPR corretto per Yeom/Shadow (#59). Al completamento trovato e corretto un bug reale di pseudo-replicazione in `discover_groups()` (Sprint 10zz+46, task #73) — n gonfiato a 10/15 per rerun che riusavano gli stessi 5 seed; dopo il fix tutti i 10 gruppi mostrano n=5 corretto, tutte le CI bootstrap contengono 0.5. Tabella definitiva in `docs/MetricsReference_DSN2027.md` §8. | **Must** | No — completato, tabella finale disponibile |

**Nota sulla portata dell'invalidazione (#11/#12, 2026-09-01):** i due fix di bug trovati nel codice
FedMIA-gradient (BatchNorm1d, split non stratificato) e il redesign `controlled_composition` erano
tutti confinati a `run_fedmia_gradient()`, una funzione nuova mai inserita in `ATTACK_REGISTRY` e
invocabile solo con `--include-fedmia-gradient` (mai usato in nessun run pubblicato). I parametri
opt-in che aggiunge a `run_lira()` (`capture_shadow_weights`, `controlled_composition`) hanno
default `False` e zero impatto confermato dalla suite di test non-torch (83→96→100→104 test, tutti
invariati a ogni fix). **Nessun risultato LiRA/Yeom/Shadow già pubblicato — campagna 5-seed×8-config,
entity-split, canary insertion, escalation di capacità — è invalidato da questi fix.** Solo i tre
tentativi di FedMIA-gradient stesso (riga #12) e il numero di smoke test del controllo centralizzato
(riga #11) vanno considerati non validi/superati.

**Decisione su #6 (2026-08-27, confermata dall'utente): rimandato, dichiarato come limite
esplicito nel paper** — validazione su un solo dataset EV, replica su ChargePlace Scotland come
lavoro futuro. Vedi `docs/DSN2027_Positioning.md`, sezione "Limitations and scope boundaries",
punto 5.

**Aggiornamento 2026-08-31 (decisione utente, priorità invertita ma non urgente): la replica su
ChargePlace Scotland non è più esclusa a priori** — l'utente vuole risultati confrontabili su dati
indipendenti per rendere il paper "inattaccabile". Resta però l'ULTIMO step della sequenza
concordata, da fare solo se resta tempo prima della deadline abstract (2026-11-25): prima vanno
chiusi (1) verifica NVFLARE Fase 7+8 con run reale 3-siti, (2) rilancio entity-aware split fino al
completamento, (3) estensione del canary positive control a caltech/jpl (oggi solo su office1), (4)
rilancio Central DP con sensibilità pesata corretta. Se il tempo non basta per ChargePlace
Scotland, resta valida la dichiarazione di limite esplicito già in `docs/DSN2027_Positioning.md`.

**Decisione su #9 (2026-08-27, confermata dall'utente): rimandato, dichiarato come limite
esplicito nel paper** — il confronto resta su placement DP (dp-fedavg/central/local, già
implementato), non su strategia di clipping. Vedi `docs/DSN2027_Positioning.md`, punto 6. Nota
importante da non perdere: il confronto Local-vs-Central DP **esiste già** (è quello che le 3
modalità già producono) — solo il clipping adattivo è il pezzo mancante.

**Decisione su #10 (2026-08-27, confermata dall'utente): rimandato.** La contraddizione fattuale
più grave (sezione 3.2, C6/L2 — l'attaccante affermato "non poter mai osservare i gradienti
pre-rumore", falso per `dp-fedavg`/`central`) è stata **già corretta** lo stesso giorno, insieme
all'aggiunta della sezione S5 (adaptive attacker). Il resto del documento (conteggio sessioni
N=13.073, 4 cluster fittizi, NVFLARE 2.7.2) resta stale solo cosmeticamente — già segnalato da una
nota di correzione in testa al documento dal 2026-07-24 — e non blocca la sottomissione.

---

## Prossimi esperimenti da eseguire — aggiornato 2026-09-03, in ordine di priorità

Sostituisce, per gli esperimenti ANCORA da lanciare, la lettura della vecchia tabella sopra (che
resta come storico). Contesto: nella sessione del 2026-09-03 sono state aggiunte metriche non
retroattive (MIA Advantage/Confusion Matrix, curve ROC complete, dump per-campione per il
controllo worst-case — task #41/#49/#50/#54) e una discussione su Carlini et al. 2022 ha esteso
queste metriche anche a Yeom/Shadow, non solo LiRA (task #53). **Nessun JSON storico le contiene**
— servono run reali. Nello stesso momento si è verificato che 2 delle 10 config della matrice del
paper (Central DP ε=0.5, Local DP ε=0.5) non sono MAI state eseguite, a nessun livello.

**Novità pratica**: il Makefile ha ora una variabile opt-in `DUMP_EXTRAS=1` (Sprint 10zz+30) che,
aggiunta a uno qualsiasi dei comandi sotto, include automaticamente `--per-sample-dump` e
`--roc-curve-dump-dir` per ogni seed (path unico, mai sovrascritti) — zero costo aggiuntivo,
raccoglie i dati per il controllo worst-case (task #50) e i plot ROC log-log (task #54) nello
stesso run che serve comunque per Advantage/Confusion. **Usarla sempre nei comandi sotto**, a meno
di non volere quei dati per un motivo specifico.

**Modo più semplice di lanciare tutto in sequenza** (aggiornato con l'ordine di priorità sotto,
usa `DUMP_EXTRAS=1` su ogni step):
```bash
cd /Users/susetta/Documents/ChargeShield-FL
mkdir -p logs
nohup caffeinate -dimsu ./scripts/run_multiseed_consolidation.sh > logs/consolidation_master.log 2>&1 &
disown
tail -f logs/consolidation_master.log
```
(`scripts/run_multiseed_consolidation.sh` riscritto in questa sessione con l'elenco `STEPS`
sotto, già in ordine di priorità — se preferisci lanciare i singoli step a mano, o solo un
sottoinsieme, i comandi equivalenti sono elencati singolarmente qui sotto.)

### P0 — informativo, nessuna azione richiesta

**Central DP ε=0.1 (`experiments/central-sweep4/`) è in corso in questo momento** (3/5 seed
completati oggi, 2026-09-03 — seed 789 e 1234 mancanti). Lascialo finire: è dati grezzi utili come
riserva, ma essendo partito PRIMA di questa sessione **non avrà comunque Confusion/curve
ROC/dump worst-case** (solo Advantage/TPR@low-FPR, già presenti) — è superseded dal punto 3 sotto,
che lo rifà da zero con tutte le metriche. Non serve killarlo né aspettarlo per procedere con la
lista sotto (sweep-dir diverse, nessun conflitto — ma non lanciare un ALTRO sweep Central DP
ε=0.1 finché questo non finisce, per via del lock anti-concorrenza del Makefile).

### P1 — buchi genuini nella matrice del paper (mai eseguiti, priorità massima)

**1. Central DP ε=0.5**
```bash
cd /Users/susetta/Documents/ChargeShield-FL
nohup caffeinate -dimsu make experiment-central-dp-sweep EPS=0.5 DUMP_EXTRAS=1 \
  > logs/central_eps05.log 2>&1 &
disown
tail -f logs/central_eps05.log
```

**2. Local DP ε=0.5**
```bash
cd /Users/susetta/Documents/ChargeShield-FL
nohup caffeinate -dimsu make experiment-local-dp-sweep EPS=0.5 DUMP_EXTRAS=1 \
  > logs/local_eps05.log 2>&1 &
disown
tail -f logs/local_eps05.log
```

### P2 — DP-FedAvg (meccanismo primario citato nel paper), backfill metriche complete

**3. DP-FedAvg ε=1.0 / 0.5 / 0.1** (dp-sweep1/2/3 esistenti hanno solo AUC/gap, dp-sweep3 anche
TPR@low-FPR — nessuno ha Advantage/Confusion/ROC/dump)
```bash
cd /Users/susetta/Documents/ChargeShield-FL
for EPS in 1.0 0.5 0.1; do
  nohup caffeinate -dimsu make experiment-dp-sweep EPS=$EPS DUMP_EXTRAS=1 \
    > logs/dp_eps${EPS}.log 2>&1 &
  wait   # uno alla volta — mai in parallelo, vedi nota OOM nel Makefile
done
```
(oppure lanciarli separatamente uno alla volta, stesso comando senza il `for`, se preferisci
controllare l'avanzamento di ognuno singolarmente)

### P3 — Central/Local DP ε=1.0/0.1, backfill metriche complete (dati preliminari già esistenti)

**4. Central DP ε=1.0** (central-sweep1/3 esistenti — sweep3 ha già Advantage/TPR, manca
Confusion/ROC/dump)
```bash
cd /Users/susetta/Documents/ChargeShield-FL
nohup caffeinate -dimsu make experiment-central-dp-sweep EPS=1.0 DUMP_EXTRAS=1 \
  > logs/central_eps10.log 2>&1 &
disown
tail -f logs/central_eps10.log
```

**5. Central DP ε=0.1** (rifà da zero central-sweep4, in corso in P0 — lanciare solo dopo che
central-sweep4 è finito o è stato fermato)
```bash
cd /Users/susetta/Documents/ChargeShield-FL
nohup caffeinate -dimsu make experiment-central-dp-sweep EPS=0.1 DUMP_EXTRAS=1 \
  > logs/central_eps01_full.log 2>&1 &
disown
tail -f logs/central_eps01_full.log
```

**6. Local DP ε=1.0 / 0.1** (local-sweep1/2 esistenti hanno TPR@low-FPR, non Advantage/Confusion/ROC/dump)
```bash
cd /Users/susetta/Documents/ChargeShield-FL
nohup caffeinate -dimsu make experiment-local-dp-sweep EPS=1.0 DUMP_EXTRAS=1 \
  > logs/local_eps10.log 2>&1 &
disown
tail -f logs/local_eps10.log
# poi, separatamente:
nohup caffeinate -dimsu make experiment-local-dp-sweep EPS=0.1 DUMP_EXTRAS=1 \
  > logs/local_eps01.log 2>&1 &
disown
```

### P4 — baseline no-DP, backfill metriche complete

**7. No-DP baseline** (nodp-sweep1 esistente manca tutto — TPR/Advantage/Confusion/ROC/dump)
```bash
cd /Users/susetta/Documents/ChargeShield-FL
nohup caffeinate -dimsu make experiment-nodp-sweep DUMP_EXTRAS=1 \
  > logs/nodp_full.log 2>&1 &
disown
tail -f logs/nodp_full.log
```

### P5 — nice-to-have, solo se resta tempo (invariato, vedi sezione dedicata sotto)

8. Sweep IDS/Byzantine n=5 (mai eseguito) — vedi test #7 sotto.
9. Seconda run del deployment reale ContainerLab — vedi test #8 sotto.
10. Replica su ChargePlace Scotland — vedi test #6 sotto.
11. Verifica pooling cross-round FedMIA-gradient (task #48) — attacco opt-in, non pubblicato,
    priorità bassa data la storia di problemi metodologici (vedi test #12 sopra).

### Dopo ogni sweep completato

```bash
python3 scripts/check_significance.py   # bootstrap CI + Wilcoxon/sign-test per gruppo
```
Per il controllo worst-case per-campione (task #50, richiede almeno 2 seed della STESSA config
con `DUMP_EXTRAS=1`, già garantito da tutti i comandi sopra):
```bash
python3 scripts/analyze_worst_case_vulnerability.py \
  experiments/<sweep-dir>/per_sample_seed42.json \
  experiments/<sweep-dir>/per_sample_seed123.json \
  experiments/<sweep-dir>/per_sample_seed456.json \
  experiments/<sweep-dir>/per_sample_seed789.json \
  experiments/<sweep-dir>/per_sample_seed1234.json
```
Per il plot ROC log-log (task #54):
```bash
python3 scripts/plot_roc_log_scale.py \
  experiments/<sweep-dir>/roc_curves_seed42/roc_curves_lira.json \
  --which composed --output figures/roc_<nome-config>.png
```

---

## Must — bloccanti per la submission

### 1. Campagna 5-seed × 10-config — ✅ COMPLETATA 2026-09-08 (task #52)

**Perché**: porta il risultato nullo (README Sprint 10dd) a rigore statistico pubblicabile — un
singolo seed per configurazione non basta per un paper table.

**Configurazioni** (`scripts/run_multiseed_consolidation.sh`): no-DP, DP-FedAvg ε∈{1.0, 0.5, 0.1},
Central DP ε∈{1.0, 0.5, 0.1}, Local DP ε∈{1.0, 0.5, 0.1} — 10 configurazioni totali, 5 seed
ciascuna (42, 123, 456, 789, 1234).

**Stato**: campagna completata 2026-09-08, 10/10 step, 6485 minuti totali. Al completamento
trovato e corretto un bug reale di pseudo-replicazione in `discover_groups()` (task #73) — dopo il
fix, tutti i 10 gruppi mostrano n=5 corretto, tutte le CI bootstrap contengono 0.5, confermato con
Wilcoxon reale (p tra 0.3125 e 1.0000). Tabella definitiva in
`docs/MetricsReference_DSN2027.md` §8.

**Nota**: `local` ε=1.0/0.1 sono attesi numericamente identici ai corrispondenti DP-FedAvg in
questa simulazione single-process (README, nota 2026-08-06) — non è un bug se i due risultano
uguali, è la conferma della nota già documentata.

**Done**: tutte le 10 sweep-dir hanno 5/5 seed completati (Excel "Seed Aggregation" = N=5
ovunque, verificato), e `python3 scripts/check_significance.py` mostra un CI per ogni gruppo.

### 2. Sanity-check positivo (dimostrare che LiRA sa rilevare overfitting)

**Perché**: senza questo, un revisore può legittimamente sospettare che l'AUC~0.50 nei test
principali sia dovuto a un attacco/harness insensibile o a underfitting del modello (Scenario B),
non a una vera assenza di leakage. Sollevato dalla revisione esterna del 2026-08-27.

**Riprogettato 2026-08-27 in due fasi**, invece di un singolo run a `epochs=500` scelto senza
base empirica (quel primo tentativo, `config/experiment_overfit_control.yaml`, è ora un file
segnaposto che punta qui — non usarlo). Aggiunto anche un vero flag `--epochs` a
`scripts/run_experiments.py` (mancava — prima si poteva cambiare `ml.epochs` solo editando il
config a mano), e `epochs` viene ora salvato nel `config` di ogni JSON di risultato, altrimenti
sarebbe impossibile distinguere dopo i file di una sweep di calibrazione a epoche diverse.

**Fase A — calibrazione economica (trova il valore giusto, non lo indovina).**
`config/experiment_overfit_calibration.yaml`: **un solo sito** (office1, ~1341 sessioni — il più
piccolo, quindi il più facile da overfittare), 3 round invece di 10, no-DP. Lanciare più run
variando solo `--epochs`, guardando `lira_auc_roc` nel JSON di ognuno:
```
cd /Users/susetta/Documents/ChargeShield-FL
mkdir -p logs
for EPOCHS in 50 150 300 500 1000; do
  python3 scripts/run_experiments.py \
    --config config/experiment_overfit_calibration.yaml \
    --no-dp --seed 42 --n-shadow 8 --epochs $EPOCHS \
    --sweep-dir experiments/_calibration_overfit \
    >> logs/calibration_overfit.log 2>&1
done
```
(`--n-shadow 8` invece di 16 per velocità — questa fase serve solo a trovare la direzione, non a
produrre numeri paper-quality.) Ogni run è breve (1 sito, 3 round) — l'intero ciclo dovrebbe
richiedere molto meno della singola run a 500 epoche/3 siti/10 round originariamente pianificata.
Guardare a quale valore di `--epochs` `lira_auc_roc` si stacca chiaramente da ~0.50 (es. >0.65-0.70)
— quello è il valore calibrato, non un numero scelto a priori.

**Fase B — run ufficiale (quella citata nel paper).** Una volta trovato il valore calibrato,
lanciarlo sulla config **standard** (3 siti reali, 10 round — il protocollo del resto della
campagna, non un file diverso):
```
cd /Users/susetta/Documents/ChargeShield-FL
nohup caffeinate -dimsu python3 scripts/run_experiments.py \
  --config config/experiment.yaml \
  --no-dp --seed 42 --n-shadow 16 --epochs <VALORE_CALIBRATO> \
  --sweep-dir experiments/_control_overfit \
  < /dev/null > logs/control_overfit.log 2>&1 &
disown
tail -f logs/control_overfit.log
```

**Interpretazione**:
- `lira_auc_roc` > 0.80 (o comunque nettamente sopra 0.5, indicativamente ≥0.60 per un pass più
  permissivo) → l'harness funziona, il ~0.50 della campagna principale è un risultato reale.
  Procedere a scrivere il paper con la tesi "nessun leakage rilevabile".
- `lira_auc_roc` resta ~0.50 anche al valore di epochs più alto testato in Fase A → problema
  ulteriore, non ancora diagnosticato, da investigare PRIMA di scrivere qualunque claim — passare
  al percorso di escalation sotto.

**Può girare in parallelo alla campagna 5-seed** (già fatto con successo in questo progetto per
run locali multiple — vedi README, run no-DP/Central DP concorrenti del 2026-08-25/26).

**Risultato Fase A (completata 2026-08-27/28)**: LiRA composto rimane ~0.48–0.52 su TUTTI i
valori testati (epochs 50/300/500/1000), nessun trend di salita verso overfitting anche a 1000
epoche locali su un singolo client (office1, no-DP). `raw_mse_auc` per-round nello stesso range
(0.487–0.538), floor_hit ancora 96–99%. **Il positive-control non è stato raggiunto tramite sole
epoche** → si passa al percorso di escalation, punto 1 (capacità del modello), per decisione
esplicita dell'utente (2026-08-27/28: "Aumenta capacità modello (Consigliato)").

**Fase A2 — escalation capacità del modello (avviata 2026-08-28, Sprint 10jj).**
Aggiunto supporto opzionale `ml.hidden_dims`/`ml.latent_dim` in
`src/core/autoencoder.py`/`src/ml/autoencoder_trainer.py`/`scripts/run_experiments.py` (default
`None`/`4` → architettura storica invariata (16, 8)/4, 570 parametri, per ogni config YAML che
non li imposta esplicitamente — **la campagna 5-seed in corso e ogni risultato già pubblicato non
sono affetti**). Nuovo config
`config/experiment_overfit_calibration_bigcap.yaml`: stesso setup della Fase A (office1, no-DP, 3
round) ma `hidden_dims: [32, 16]`, `latent_dim: 8` (~1870 parametri, ~3.3× la capacità storica),
`epochs: 1000` di default (il valore più alto già testato in Fase A, per isolare la variabile
capacità). Comando:
```
cd /Users/susetta/Documents/ChargeShield-FL
mkdir -p logs
nohup caffeinate -dimsu python3 scripts/run_experiments.py \
  --config config/experiment_overfit_calibration_bigcap.yaml \
  --no-dp --seed 42 --n-shadow 8 \
  --sweep-dir experiments/_calibration_overfit_bigcap \
  < /dev/null > logs/calibration_overfit_bigcap.log 2>&1 &
disown
tail -f logs/calibration_overfit_bigcap.log
```
`hidden_dims`/`latent_dim` vengono salvati nel `config` del JSON di output esattamente come
`epochs` (stessa motivazione di provenance) — verificabile leggendo il JSON, non solo il nome del
file di config o della sweep-dir.

**Risultato Fase A2 (completata 2026-08-28)**: `hidden_dims=[32, 16]`, `latent_dim=8` (~1870
parametri, ~3.3× la capacità storica), stesso setup di Fase A (office1, no-DP, 3 round,
epochs=1000). LiRA composto = **0.5294** (raw_mse_auc per round 0.531/0.527/0.540) — ancora
piatto, nessuna crescita rispetto alla capacità storica. **L'escalation di capacità non ha
prodotto il positive-control.** Su decisione esplicita dell'utente, si passa al punto 2
(feature-entropy), isolando la variabile invece di sommarla alla capacità già testata.

**Fase A3 — escalation entropia feature (avviata 2026-08-28, Sprint 10kk).**
Ipotesi: le 6 feature storiche sono aggregati a bassa entropia per sessione (kWh totali, potenza
media, ecc.) — potrebbero non contenere abbastanza informazione identificativa da memorizzare,
indipendentemente da capacità/epoche. Aggiunto supporto opzionale `ml.feature_names` in
`src/ml/autoencoder_trainer.py` e in tutti i 6 punti di `scripts/run_experiments.py` che
costruiscono tensori per gli attacchi MIA (FedMIA, Shadow MIA, LiRA — helper
`_mia_feature_names(cfg)`, stessa logica di `_autoencoder_arch_kwargs()` per capacità), più una
nuova feature derivata `start_time_epoch` (timestamp Unix in secondi dell'inizio sessione,
aggiunta in `enrich_sessions()`) — una feature reale, non un ID iniettato ad hoc, ma a
risoluzione abbastanza fine da essere quasi univoca per sessione. Default `None` → le 6 feature
storiche, invariato per ogni config esistente. Nuovo config
`config/experiment_overfit_calibration_richfeat.yaml`: stesso setup Fase A (office1, no-DP, 3
round, epochs=1000), capacità **tornata alla storica** (16, 8)/4 — non combinata con Fase A2, per
isolare l'effetto dell'entropia da quello della capacità — `feature_names` con le 6 storiche +
`start_time_epoch`, `input_dim: 7`. Comando:
```
cd /Users/susetta/Documents/ChargeShield-FL
mkdir -p logs
nohup caffeinate -dimsu python3 scripts/run_experiments.py \
  --config config/experiment_overfit_calibration_richfeat.yaml \
  --no-dp --seed 42 --n-shadow 8 \
  --sweep-dir experiments/_calibration_overfit_richfeat \
  < /dev/null > logs/calibration_overfit_richfeat.log 2>&1 &
disown
tail -f logs/calibration_overfit_richfeat.log
```
`feature_names` viene salvato nel `config` del JSON di output, stessa motivazione di provenance
già applicata a `epochs`/`hidden_dims`.

**Nota importante per l'interpretazione**: `start_time_epoch` è un controllo diagnostico
sull'harness, NON una feature che il paper propone di usare in produzione — se questo file mostra
overfitting rilevabile, non va aggiunta al run "ufficiale" (`config/experiment.yaml`) solo per far
salire l'AUC, perché cambierebbe cosa il modello osserva regolarmente, non solo se il
sanity-check funziona.

**Risultato Fase A3 (completata 2026-08-28)**: `feature_names` con le 6 storiche +
`start_time_epoch` (quasi univoca per sessione), capacità storica (16, 8)/4, stesso setup di Fase
A/A2 (office1, no-DP, 3 round, epochs=1000). LiRA composto = **0.5352** (raw_mse_auc per round
0.530/0.542/0.542) — ancora nel range di rumore osservato in TUTTE le calibrazioni precedenti
(0.48–0.54), nessuna crescita verso un vero positive-control. Anche con una feature quasi
identificativa per sessione, il modello non arriva a memorizzare abbastanza da essere rilevato da
LiRA. **Tre assi di escalation indipendenti (epoche fino a 1000, capacità ~3.3×, entropia feature
fino a un identificatore quasi univoco) danno tutti lo stesso esito: nessun positive-control.**

**Fase A4 (sito/dataset diverso) + Fase A2+A3 combinata (avviate 2026-08-28, Sprint 10mm)** — su
decisione esplicita dell'utente ("vorrei fare un quarto asse e poi combinare capacità e
feature"), invece di accettare subito il null triangolato:

- **A4 — sito diverso.** Ipotesi: office1 (il più piccolo, 8 EVSE, scelto in Fase A proprio
  perché "il più facile da overfittare") potrebbe avere una distribuzione dati che rende membri e
  non-membri poco separabili in feature space indipendentemente da modello/feature — un problema
  di separabilità del dataset, non di capacità di memorizzazione. `config/experiment_overfit_calibration_altsite.yaml`:
  stesso setup di Fase A (capacità storica, 6 feature, no-DP, 3 round, epochs=1000, n_shadow=8) ma
  sul sito **Caltech** (54 EVSE — popolazione molto più grande/diversa di office1), usando **solo
  l'anno 2021** (3034 sessioni — una slice reale del dataset, non un sottocampionamento
  artificiale, scelta per restare vicina in scala a office1 ed evitare il costo ~19× più alto di
  Caltech completo, che avrebbe reso la calibrazione economica non più economica). Comando:
  ```
  cd /Users/susetta/Documents/ChargeShield-FL
  mkdir -p logs
  nohup caffeinate -dimsu python3 scripts/run_experiments.py \
    --config config/experiment_overfit_calibration_altsite.yaml \
    --no-dp --seed 42 --n-shadow 8 \
    --sweep-dir experiments/_calibration_overfit_altsite \
    < /dev/null > logs/calibration_overfit_altsite.log 2>&1 &
  disown
  tail -f logs/calibration_overfit_altsite.log
  ```

- **Combinata — capacità + feature insieme.** Ipotesi: capacità (A2) ed entropia feature (A3)
  testate separatamente non bastano da sole, ma potrebbero bastare insieme (un modello più
  capiente potrebbe sfruttare l'informazione aggiuntiva di `start_time_epoch` meglio di uno a
  capacità storica). `config/experiment_overfit_calibration_combined.yaml`: stesso setup di Fase
  A (office1, no-DP, 3 round, epochs=1000, n_shadow=8) con `hidden_dims=[32,16]`/`latent_dim=8`
  (come A2) **e** `feature_names` con `start_time_epoch`/`input_dim=7` (come A3) insieme. Comando:
  ```
  cd /Users/susetta/Documents/ChargeShield-FL
  mkdir -p logs
  nohup caffeinate -dimsu python3 scripts/run_experiments.py \
    --config config/experiment_overfit_calibration_combined.yaml \
    --no-dp --seed 42 --n-shadow 8 \
    --sweep-dir experiments/_calibration_overfit_combined \
    < /dev/null > logs/calibration_overfit_combined.log 2>&1 &
  disown
  tail -f logs/calibration_overfit_combined.log
  ```

Nessuna modifica al codice necessaria per questi due file — riusano `hidden_dims`/`latent_dim`
(Sprint 10jj) e `feature_names` (Sprint 10kk) già implementati. Entrambi possono girare in
parallelo tra loro e con la campagna 5-seed (stessa nota di sempre — nessun conflitto, run CPU
locali indipendenti).

**Risultato test Combinata (completato 2026-08-28)**: `hidden_dims=[32,16]`/`latent_dim=8` +
`feature_names` con `start_time_epoch` insieme, stesso setup di Fase A (office1, no-DP, 3 round,
epochs=1000). LiRA composto = **0.5169** (raw_mse_auc per round 0.526/0.491/0.535) — ancora nel
range di rumore 0.48–0.54 osservato in OGNI singola calibrazione di questa catena. **Nemmeno
l'effetto congiunto capacità+feature produce un positive-control** — l'ipotesi che i due fattori
si sommassero non è confermata.

**Risultato Fase A4 (completata 2026-08-28)**: sito **Caltech 2021** (3034 sessioni, 2427
train/607 hold-out — popolazione molto più grande/diversa di office1), capacità e feature
storiche (come Fase A). LiRA composto = **0.5168** (raw_mse_auc per round 0.504/0.510/0.506) —
di nuovo nel range 0.48–0.54. **Anche un sito completamente diverso, con popolazione EVSE/utenti
molto più ampia, dà lo stesso esito nullo.** Nota diagnostica non allarmante: 2 campioni su 3
round hanno usato il fallback μ_in (`mu_in=...(FALLBACK)` nei log) invece della calibrazione IN
reale — atteso e già discusso nei fix Sprint 10bb/10cc (alcuni non-membri, per costruzione, non
ricadono mai nel subset IN di nessuno shadow in un dato round), non un nuovo bug.

**Sintesi (2026-08-28): 5 test di escalation indipendenti — epoche (50→1000), capacità (~3.3×),
entropia feature (fino a un identificatore quasi univoco), combinazione capacità+feature, sito
diverso (Caltech invece di office1) — convergono TUTTI su LiRA composto nel range 0.48–0.54,
nessuno sopra la soglia indicativa di positive-control (≥0.60).**

**DECISIONE FINALE (2026-08-28, esplicita dell'utente): accettare il null triangolato, chiudere
qui il sanity-check.** Non si procede con ulteriori assi di escalation (es. pool di tutti e 3 i
siti insieme, considerato e scartato) — la triangolazione a 5 assi indipendenti è considerata
evidenza sufficiente. **Conseguenza per il paper**: questo sanity-check non produce il
positive-control che si sperava (non possiamo scrivere "abbiamo dimostrato che l'attacco rileva
memorizzazione quando c'è"), quindi va scritto con l'onestà opposta — come un limite di
validazione dichiarato, non nascosto. Testo suggerito per la sezione limitazioni/discussione del
paper: *"Abbiamo condotto un sanity-check estensivo per verificare che l'harness LiRA/FedMIA/Yeom
sia in grado di rilevare memorizzazione quando presente, per rispondere alla preoccupazione che
AUC≈0.5 rifletta una metodologia d'attacco insensibile piuttosto che un'assenza genuina di
leakage. Abbiamo variato cinque fattori indipendenti — epoche di training (50-1000), capacità del
modello (fino a ~3.3× i parametri), informatività delle feature (aggiungendo un timestamp
per-sessione quasi univoco), la combinazione congiunta di capacità e feature, e la popolazione di
training (un sito diverso, molto più grande) — e in ogni configurazione il LiRA AUC è rimasto nel
range di rumore 0.48-0.54, ben sotto la soglia di 0.60 che avevamo fissato come evidenza di
memorizzazione rilevabile. Interpretiamo questo risultato come evidenza che l'overfitting
intenzionale è difficile da indurre in questo regime architettura/dati, non come prova che la
metodologia d'attacco sia inefficace di per sé (una prova definitiva richiederebbe, ad esempio,
un dataset costruito ad hoc con record duplicati) — lo segnaliamo esplicitamente come limite della
nostra validazione."* Questo testo va integrato in `docs/DSN2027_Positioning.md` (fatto, vedi
sotto) prima della stesura finale del paper.

Task #2 di questa roadmap è quindi **chiuso**. Prossimi passi: campagna 5-seed×8-config (#1, in
corso) resta il blocco principale; TPR@low-FPR (#4) e Wilcoxon (#5) restano aperti come da tabella
sopra.

**RIAPERTO E CONCLUSO POSITIVAMENTE (2026-08-31)**: il testo sopra dice esplicitamente che "una
prova definitiva richiederebbe... un dataset costruito ad hoc con record duplicati" — su domanda
esplicita dell'utente ("come facciamo a dimostrare l'efficacia di DP se non dimostriamo la
vulnerabilità senza DP?"), questo è stato fatto (Sprint 10vv-10yy, README): 5 sessioni reali del
client office1 duplicate 30× ciascuna nel training set ("canary insertion", tecnica standard in
letteratura DP/MIA — Carlini 2019, Jagielski et al. 2020), con 20 gemelle mai viste in training
come confronto pulito. **Risultato: positive control ottenuto, a livello di loss di ricostruzione
grezza del modello target (nessuna calibrazione shadow) — AUC 0.7040/0.8951/0.8526 nei 3 round,
sempre nella stessa direzione, sempre sopra la soglia 0.60.** Il modello memorizza inequivocabilmente
quando forzato a farlo. Il punteggio LiRA CALIBRATO sugli stessi canary resta invece instabile
(0.49/0.63/0.43) — spiegazione più probabile: gli shadow model si addestrano sulla stessa
popolazione che contiene le 150 copie duplicate, quindi il loro campionamento casuale del
sottoinsieme IN "vede" più copie di ogni canary indipendentemente da quale copia specifica è
nominalmente esclusa, diluendo il segnale differenziale. **Questo è un artefatto specifico a
record letteralmente quasi-duplicati in una popolazione condivisa target/shadow — le sessioni EV
reali del dataset naturale non hanno questa proprietà, quindi non si estende al risultato
principale.** Aggiorna la conclusione precedente: non più "nessun positive control ottenuto,
limite dichiarato senza risoluzione" ma "positive control ottenuto a livello di loss grezza, con
un limite specifico e meccanisticamente identificato (non solo sospettato) della calibrazione
shadow di LiRA in presenza di record quasi-duplicati" — una storia di validazione più precisa e
più difendibile. Testo per il paper aggiornato in `docs/DSN2027_Positioning.md` e nello scheletro
Word (`docs/paper/ChargeShield-FL_DSN2027_paper_skeleton.docx`, §6/Limitazione #7).

**AGGIORNAMENTO (2026-09-02, README Sprint 10zz+15/10zz+16/10zz+18) — la causa dell'instabilità
LiRA-calibrata è stata confermata (via lettura del codice di campionamento, non più solo
ipotizzata) e CORRETTA, con verifica su run reale.** La spiegazione sopra ("il campionamento
casuale vede comunque più copie di ogni canary") è stata precisata matematicamente: i 150
duplicati sono oggetti Python indipendenti in un pool comune, e con metà universo campionato per
shadow, la probabilità che uno shadow "OUT" per un duplicato specifico non veda NESSUNO dei suoi
149 gemelli è trascurabile — quasi certo che ogni shadow "OUT" si alleni comunque sul pattern del
canary. Fix: campionamento a livello di GRUPPO canary (atomico, mai spezzato tra IN e OUT),
`_sample_preserving_canary_groups()` in `scripts/run_experiments.py`, zero impatto su ogni run
reale/pubblicato (nessun `_canary_group` fuori dai run canary). **Verificato con un run reale
identico al precedente (stesso config, seed, n_shadow)**: `canary_auc_roc` per round passa da
0.4863/0.6265/0.4250 (instabile, un round sotto il caso) a **0.6375/0.6625/0.5875 — sempre sopra
0.5, composto 0.6875, ora esso stesso sopra la soglia 0.60** (prima ci arrivava solo il diagnostico
raw-loss). Resta più debole del segnale raw-loss (0.6625/0.85/0.8375) per un motivo separato e
già noto (σ_in/σ_out al floor nel 96-99% dei casi in questo run). **Conclusione aggiornata per il
paper**: il positive control ora regge anche sulla metrica calibrata primaria (LiRA), non solo sul
diagnostico raw-loss — da propagare allo scheletro del paper (non ancora fatto).

**Se anche questi due test restano a ~0.50**: a quel punto quattro spiegazioni alternative
indipendenti sono state escluse (poche epoche, poca capacità, poca informazione nelle feature,
sito/dataset specifico) più l'ipotesi congiunta capacità+feature — decisione da riportare
all'utente su come procedere, probabilmente accettando il null come conclusione robusta del
sanity-check a quel punto.

**Nota per la scrittura del paper (non un'azione da fare ora)**: se il risultato finale è "AUC≈0.50
a ogni ε, incluso no-DP", un revisore può fare la stessa domanda di fondo che una prima revisione
esterna ha fatto sui vecchi numeri (invertiti) — "come faccio a sapere che non è un bug, se il
meccanismo DP non produce nessun effetto misurabile a nessun livello?" Questo sanity-check
positivo è esattamente la risposta da citare nel paper a quella domanda: dimostra che l'harness
rileva memorizzazione quando c'è, quindi l'assenza di un effetto dose-risposta di ε è
un'osservazione reale (il modello non arriva a memorizzare abbastanza perché DP abbia qualcosa da
sopprimere), non un artefatto della misura.

### 3. Fix `check_significance.py` — ✅ completato 2026-08-27

Il vecchio script assumeva un'associazione fissa nome-cartella→epsilon (es. "dp-sweep2 = ε=0.1")
basata sull'ordine di lancio di sweep passati — lo stesso tipo di bug già trovato e corretto una
volta nel Makefile (README Sprint 10r). Con la nuova campagna l'ordine di lancio è diverso, quindi
la vecchia mappa avrebbe etichettato silenziosamente i risultati con l'epsilon sbagliato. Ora lo
script legge `dp_mode`/`epsilon` direttamente dal config di ogni JSON e raggruppa dinamicamente —
verificato funzionante contro i file diagnostici già presenti in `experiments/`.

---

## Should — rafforzano fortemente la submission

### 4. Metrica TPR@low-FPR — ✅ completato 2026-08-28

**Perché**: `docs/ReadingList_DSN2027.md` (nota 2026-08-14) documenta che LiRA (Carlini et al.
2022) stesso argomenta che AUC-ROC è una metrica MIA inadeguata e raccomanda TPR a FPR basso
(es. TPR@1%FPR) — ChargeShield-FL riportava solo AUC-ROC. Per un risultato nullo in AUC, questo
è particolarmente importante: un AUC~0.5 mediato su tutta la curva ROC potrebbe comunque
nascondere un segnale concentrato a bassissimo FPR (il regime che conta di più per un attacco
reale) — motivo per cui la metrica esiste.

**Implementato (Sprint 10pp)**: nuova `_tpr_at_fixed_fpr()` in `scripts/run_experiments.py`
(FPR target: 0.1%, 1%, 5% — `sklearn.metrics.roc_curve` + interpolazione lineare con
`np.interp`), calcolata sulle stesse coppie score/label già prodotte da `run_lira()` (confermato:
`round_member_scores`/`round_nonmember_scores` erano già calcolati in memoria ma solo l'AUC
aggregato veniva salvato — nessun nuovo esperimento necessario, solo una nuova aggregazione).
Aggiunta sia al livello per-round (`lira_results[round_num]`) sia al composto multi-round
(`composed_output` — la metrica "headline" citata nei Sprint-log), come campi
`tpr_at_fpr_0.001`/`tpr_at_fpr_0.01`/`tpr_at_fpr_0.05`. **Si applica solo ai run futuri**: i JSON
già prodotti (incluse le 5 fasi di calibrazione e la campagna 5-seed lanciata prima di questo
fix) non hanno questi campi — se serve per la tabella finale del paper, va ri-eseguito almeno il
run "ufficiale" dopo questo punto. Verificato: `py_compile` + 83 test non-torch passano;
verificata anche la logica di interpolazione con un array fpr/tpr simulato via numpy (disponibile
in questo sandbox, sklearn no — stesso vincolo di sempre).

### 5. Test di significatività Wilcoxon — ✅ completato 2026-08-28

La revisione esterna raccomanda esplicitamente un test di Wilcoxon oltre al bootstrap CI già
implementato.

**Implementato (Sprint 10pp)**: `scripts/check_significance.py` prova `scipy.stats.wilcoxon`
(vero test dei ranghi con segno) e, se scipy non è installato (come in questo sandbox — da
verificare sulla macchina reale), usa un **sign test** binomiale esatto pure-Python come
fallback dichiarato (mai silenzioso: il metodo usato è sempre riportato in output). **Nota
onesta da riportare se questi p-value finiscono nel paper**: con n=5 seed per gruppo, il p-value
minimo raggiungibile dal sign test è 2×(1/32)=0.0625 — non può MAI risultare significativo ad
α=0.05 con questa numerosità, un limite noto dei test non parametrici a campioni molto piccoli,
non un difetto dell'implementazione. Il bootstrap CI resta il test primario per questa
numerosità campionaria.

**Bug reale trovato eseguendo davvero lo script (non solo `py_compile`) contro
`experiments/` reale, durante questo lavoro**: il gruppo "no-DP baseline" risultava con n=20
invece di 5 — `discover_groups()` scansionava OGNI `experiment_*.json` in QUALUNQUE sottocartella
di `experiments/`, incluse le 5 cartelle di calibrazione LiRA appena prodotte
(`_calibration_overfit*`) e le cartelle diagnostiche (`_diag_*`/`_verify_*`/`_archive_*`) — non
solo le sweep-dir della campagna vera (`dp-sweep1`, `nodp-sweep1`). Prima dell'aggiunta dei file
di calibrazione questo restava quasi innocuo; ora conflava silenziosamente 5 seed reali (3 siti,
10 round) con 5+ run di calibrazione a 1 sito/3 round/architetture diverse mai pensati per essere
mediati insieme — stessa classe di bug del mapping cartella→epsilon già corretto una volta in
questo stesso script (2026-08-27). **Fix**: `discover_groups()` esclude ora di default le
cartelle il cui nome inizia con `_` (convenzione già in uso in questo progetto per
diagnostica/calibrazione/archivio, mai usata dalle sweep-dir reali) — `include_diagnostic=True`
le reintegra esplicitamente per chi vuole ispezionarle. Verificato **con esecuzione reale**
(sklearn/scipy assenti ma non richiesti da questo script) contro `experiments/` reale: dopo il
fix, "no-DP baseline" torna correttamente a n=5.

---

## Nice-to-have — solo se il tempo lo permette prima del 2026-11-25

### 6. Replica su secondo dataset EV (ChargePlace Scotland)

Task #89 (README, note 2026-08-07): dataset già scaricato (`datasets/alt/chargeplace_scotland/`).
**Aggiornamento 2026-09-09 (task #37): l'adapter è stato scritto e testato**
(`src/adapters/chargeplace_scotland_adapter.py`, `tests/test_chargeplace_scotland_adapter.py`,
30 test, tutti passanti). Conteggio reale verificato leggendo tutti e 18 i file mensili:
**3.120.526 sessioni totali** (NON ~3.9M come scritto qui finora — numero corretto). Decisioni di
design prese scrivendo l'adapter:
- site_id = local_authority (32 council area scozzesi, da `CPID_and_local_authority.xlsx`) —
  equivalente concettuale del siteID di ACN-Data (Caltech/JPL/Office1), ma con granularità molto
  più fine. Copertura del join verificata: 99.93%.
- user_id sempre `None` — ChargePlace Scotland non traccia l'utente, solo il CPID. **Limite da
  dichiarare esplicitamente se questo dataset verrà usato**: lo split entity-aware (task #27/#38)
  non è replicabile su questo dataset a livello utente.
- Bug reale trovato e corretto scrivendo i test (non dalla sola lettura del codice): le sessioni
  con durata >= 24h arrivano da Excel come stringa `"N day(s), H:MM:SS"` invece che
  `datetime.time` — un parsing ingenuo (`str.split(":")`) falliva su 993/162896 sessioni di un
  singolo file (quasi l'1%, non un caso limite). Non è un troncamento modulo-24h come temuto
  inizialmente: risolto con `pandas.Timedelta`, che interpreta nativamente entrambi i formati.
  Verificato: zero perdita di dati oltre alle 7 sessioni per file con Duration realmente mancante
  (NaN).

Rafforzerebbe il claim da "nessun leakage rilevabile su ACN-Data" a "...replicato su un secondo
dataset EV indipendentemente raccolto". **Aggiornamento 2026-09-09**: wiring completato —
`scripts/run_experiments.py::load_sessions()` sceglie l'adapter via `cfg["dataset_adapter"]`
(default "acn", retrocompatibile al 100% con ogni config esistente), nuovo
`config/experiment_chargeplace_scotland.yaml` con i 3 client scelti dall'utente (Glasgow City,
East Ayrshire, City of Edinburgh — top-3 per volume, 688.896 sessioni totali). 6 nuovi test in
`tests/test_run_experiments_integration.py` (non eseguibili in questo sandbox, richiedono torch —
da eseguire sulla macchina reale). Limite noto verificato: kwh_requested e minutes_available sono
sempre 0 in questo dataset (nessun equivalente di userInputs di ACN-Data) — 2 delle 6 feature di
input sono quindi costanti, gestito senza crash da compute_feature_stats() ma da menzionare se
questi risultati finiscono nel paper.

**Aggiornamento 2026-09-10 — smoke test no-DP baseline COMPLETATO**: lanciato 2026-09-09 16:24,
terminato 2026-09-10 04:41 (~12h17min totali) — training FL (10 round) e attacchi Yeom/Shadow
veloci (pochi minuti), ma LiRA da solo ha impiegato ~9h (10 round × ~54 min/round, contro pochi
minuti per round su ACN-Data — coerente con la scala ~10.3× più grande). Il tempo NON scala
linearmente con la dimensione dati come temuto: va tenuto in conto seriamente per pianificare
un'eventuale campagna DP completa su questo dataset (10 config × 5 seed a questa velocità sarebbe
comparabile o peggiore delle ~108 ore già spese su ACN-Data). **Risultato (no-DP baseline,
`experiments/experiment_20260910_044144.json`)**: mean_auc_roc=0.5008, mean_lira_auc_roc=0.5006,
privacy_risk=LOW — **nessun leakage rilevabile, primo risultato empirico sul secondo dataset**.
Rafforza già, anche solo con la baseline no-DP, il claim di generalizzabilità del progetto. Ancora
da fare se il tempo lo permette: configurazioni con DP attivo (dp-fedavg/central/local × ε) su
questo dataset, e più seed per un confronto statistico pari a quello di ACN-Data — non bloccante
per la submission, il claim su ACN-Data da solo resta pubblicabile.

### 7. Sweep IDS/Byzantine n=5 (task #50)

Mai eseguito in tutto il progetto (`experiments/ids_validation/` non esiste). Krum/CUSUM/Cosine
sono implementati e testati unitariamente, ma non validati end-to-end con un vero attacco
Byzantine iniettato a n=5 client. Rilevante solo se il paper vuole riportare anche le metriche
F1/Precision/Recall/AUROC dell'IDS stesso, non solo la parte privacy/MIA — verificare con l'utente
se rientra nello scope del paper DSN 2027 o è materiale per un lavoro futuro.

### 8. Validazione statistica sul deployment reale ContainerLab

**Aggiornamento 2026-09-09/10 — molto più avanti di quanto scritto sopra.** Step A (redeploy da
zero con codice aggiornato — ByzantineDetector rinominato, DP-sensitivity fix, ML Plane reale) ha
girato pulito, 10/10 round, zero errori. Step B — campagna statistica multi-seed:
- **dp-fedavg, ε=1.0, 5 seed (42/123/456/789/1234)**: tutti completi, 10/10 round ciascuno, zero
  `byzantine_detected`/`threats_detected`. 2 alert HIGH cosine-similarity su `office1` (seed 42
  round 4, seed 789 round 7) — stesso pattern già noto e documentato in `docs/IDS.md` §12
  (office1 è il sito più piccolo/rumoroso, mai caltech/jpl).
- **central, ε=1.0**: ora **5/5 seed completi** (42 pre-fix, 123/456/789/1234 post-fix — vedi fix
  round-1 sotto), tutti 10/10 round, zero alert/byzantine/threats. Il seed 42 confrontato
  round-per-round col dp-fedavg seed 42: round 1 identico su tutti i client, poi diverge
  genuinamente dal round 2 in poi (non un artefatto — vedi nota sotto e
  `docs/MetricsReference_DSN2027.md` per la spiegazione architetturale completa) — conferma che
  dp-fedavg e central esercitano davvero percorsi di codice diversi nel deployment reale, cosa che
  la sola simulazione single-process non può provare (lì dp-fedavg risulta identico a **local**,
  non a central — vedi nota dedicata in MetricsReference). `scripts/set_nvflare_seed.py` (nuovo,
  2026-09-11) scrive i due config JSON via `json.dump` invece di edit manuale, e salva uno snapshot
  dedicato `seed_snapshots/config_fed_client_seed<N>_<dp_mode>.json` per ogni run — evita la stessa
  classe di bug già vista con seed123/dp-fedavg (snapshot sbagliato usato per la rianalisi).
- **Fix round-1 (2026-09-10) VERIFICATO con dati reali (2026-09-11)**: sottomesso un run
  central/seed=999 (mai usato prima) e confrontato `raw_global_weights` del round 1 contro il
  vecchio run seed=42 pre-fix via `scripts/verify_seed_fix_round1.py` — tutti e 22 i tensori
  differiscono (max abs diff 0.006–1.04), l'inizializzazione del modello è ora genuinamente
  seed-dipendente. L'epsilon dell'audit JSON restava identico anche post-fix (falso allarme, non
  un fallimento del fix — è una funzione a forma chiusa di config statica per-client, non del
  contenuto reale dei pesi).
- **Rianalisi MIA offline** (`scripts/run_nvflare_mia.py`, n_shadow=16, stesso `run_lira()` della
  simulazione): seed 42 dp-fedavg completata — `mean_lira_auc_roc=0.499`, `privacy_risk=LOW`,
  nessuna fuga rilevata, coerente con la simulazione. Central 123/456 rianalizzati
  (`mean_lira_auc_roc` 0.5000/0.5015, `privacy_risk=LOW`, nessuna anomalia); **aggiornamento
  2026-09-11: 789/1234 ora rianalizzati anch'essi** (`mean_lira_auc_roc` 0.5003/0.5012,
  `privacy_risk=LOW` per entrambi, nessuna anomalia) — tutti e 5 i seed central (42 pre-fix +
  123/456/789/1234 post-fix) hanno ora sia il training sia la rianalisi MIA offline completi.
- **Canary positive control portato su NVFLARE (2026-09-11, mai esistito lì finora)**: l'utente ha
  chiesto se il null result NVFLARE fosse verificato contro un attacco silenziosamente rotto —
  `chargeshield_executor.py` accetta ora un blocco opzionale `canary`, `run_nvflare_mia.py`
  ricostruisce offline la stessa iniezione (stesso seed, stesso offset RNG) — verificato in
  isolamento (stessa selezione di template su entrambi i lati), non ancora lanciato per davvero.
  Vedi `docs/NVFlareIntegration.md` per il dettaglio completo.

**Finding metodologico trovato investigando questi dati (2026-09-10, non un bug che invalida i
risultati MIA, ma da documentare)**: i valori di round 1 in `nvflare_ids_audit_results_*.json`
(privacy_score/epsilon per client) sono **byte-identici in tutti e 5 i seed dp-fedavg E nel run
central** — perché il seed della campagna NVFLARE viene usato solo per lo split train/holdout
lato client (`chargeshield_executor.py`) e lo shuffle del DataLoader, MAI per l'inizializzazione
dei pesi del modello globale (il persistor NVFLARE crea sempre lo stesso checkpoint iniziale,
senza seeding esplicito) — a differenza della simulazione single-process, dove
`torch.manual_seed(seed)` viene chiamato in `main()` prima della creazione del modello. Inoltre al
round 1 `_compute_sensitivity` clippa la norma assoluta dei pesi (non un delta, perché
`reference_weights` è `None` al primo round) — dominata dal checkpoint condiviso e non seedato,
da cui l'identità esatta. Dal round 2 in poi (quando `reference_weights` esiste ed è
genuinamente diverso per seed/dp_mode) i valori divergono normalmente. **Non tocca le conclusioni
MIA**: LiRA/Yeom/Shadow operano su `raw_updates`/errore di ricostruzione (`nvflare_fl_results_*.pkl`),
non su questo specifico campo di audit — ma va tenuto a mente se in futuro si volesse citare la
varianza inter-seed del round 1 specificamente come evidenza di robustezza: al round 1 quella
varianza è artificialmente zero, non un segnale reale.

**Aggiornamento 2026-09-13**: anche `local` DP mode è ora completo a 5 seed su NVFLARE (42/123/456/
789/1234, ε=1.0, `run_nvflare_mia.py` con `n_shadow=16` per ognuno) — `mean_lira_auc_roc` per seed:
0.4993/0.5007/0.4991/0.5003/0.5007, tutti `privacy_risk=LOW`, nessuna anomalia. Bootstrap (10000
resample): mean=0.500000, std=0.000785, 95% CI=[0.499388, 0.500611] — contiene 0.5, coerente con
`central` (stesso null result) e con l'intera campagna single-process. `central` e `local` sono ora
entrambi completi a 5 seed su NVFLARE.

**Ancora da fare per un confronto pienamente allineato con la campagna single-process (10 config ×
5 seed)**: solo la variazione di epsilon (0.5/0.1 — solo ε=1.0 testato finora su NVFLARE, bassa
priorità esplicita) resta aperta. Non bloccante per la submission — il claim principale del paper
si basa sulla campagna single-process, già completa e statisticamente solida; questi run NVFLARE
sono una validazione supplementare "il risultato regge anche in un deployment reale
multi-container", non un sostituto. Costo/beneficio da valutare rispetto al tempo restante prima
della deadline (abstract 2026-11-25).

---

## Priorità immediate — aggiornato 2026-09-15 (Sprint 10zz+93)

Sostituisce, per l'ordine di priorità, tutte le sezioni sottostanti dove in conflitto
(restano come archivio storico di come si è arrivati qui). Riscritta su richiesta esplicita
dell'utente dopo la conversazione sul doppio consumatore dell'ML Plane (Privacy Auditor vs
ByzantineDetector/IDS) e sulla richiesta di unificare la superficie di attacco dei tre
attacchi. Per ognuno: la domanda di ricerca, il risultato atteso in entrambi i sensi, lo
stato reale, e — se pronto — il comando esatto.

1. **Blocker 2 (sotto, ora sbloccato)** — economico (3 round, 1 sito, nessun training
   pesante), nessun nuovo codice da scrivere per lanciarlo, e logicamente prioritario su
   tutto il resto: se il "vero leakage" non si stacca da 0.5 nemmeno con i tre attacchi
   estesi al canary, va rivista l'interpretazione di OGNI null result già pubblicato prima
   di investire altro tempo su di essi.
2. **RQ6 — rilevamento Byzantine sotto DP (nuovo, sotto)** — economico (10-15h stimate
   dall'utente), zero nuovo codice (il percorso `--byzantine` esiste ed è testato solo a
   livello unitario), e produce un risultato architetturale forte indipendentemente
   dall'esito. Va lanciato SUBITO DOPO Blocker 2, con lo stesso principio "prima il
   positive control, poi la soppressione": baseline no-DP per primo.
3. **GATE obbligatorio prima del rerun Strada B — risolvere l'interazione floor_mode ×
   member_scoring (aggiunto 2026-09-15, avvertimento esplicito dell'utente).**
   `floor_mode=symmetric` è ancora il default (mai validato con un A/B reale) e
   `member_scoring=matched_formula` (l'ancoraggio μ_in, errata punto #3) non è default —
   entrambi verificati nel codice in questo giro (`cfg.get("lira", {}).get("floor_mode",
   "symmetric")`, `cfg.get("lira", {}).get("member_scoring", "real")`). Le due variabili
   interagiscono: l'ablation cold-start (Sprint 10zz+91) mostra `matched_formula_auc` a
   0.49-0.51 sotto `shadow_init=cold` contro 0.34-0.44 sotto `shadow_init=warm` sullo
   STESSO config, mentre il floor-hit-rate di σ_in/σ_out sale a quasi il 100%. Se si lancia
   ora il rerun da ~33h di Strada B e poi si tocca lo scoring (floor o member_scoring),
   quelle ore vanno rifatte. **Comandi pronti, entrambi economici (10 round, 1 seed,
   nessun nuovo codice):**
   ```
   # A: member_scoring (errata #3, ancoraggio μ_in)
   python3 scripts/compare_floor_mode.py \
     --before experiments/<run_con_config/experiment.yaml> \
     --after experiments/<run_con_config/experiment_matched_formula_scoring.yaml> \
     --label-before "member_scoring=real (default)" \
     --label-after "member_scoring=matched_formula"

   # B: floor_mode (mai validato)
   python3 scripts/compare_floor_mode.py \
     --before experiments/<run_con_config/experiment.yaml> \
     --after experiments/<run_con_config/experiment_floor_independent.yaml> \
     --label-before "floor_mode=symmetric (default)" \
     --label-after "floor_mode=independent"
   ```
   Devono girare entrambi i run "after" (`config/experiment_matched_formula_scoring.yaml`,
   `config/experiment_floor_independent.yaml`) più un run "before" con
   `config/experiment.yaml` a parità di seed/ε/dp-mode, PRIMA del comando di rerun Strada B
   sotto — non in parallelo, in sequenza, perché il risultato di A/B può cambiare quale
   scoring usare per il rerun stesso.
4. **Rerun Strada B — terzo posizionamento DP reale (task #145, codice pronto) — BLOCCATO
   dal gate #3 sopra.** Costoso (~33h, invalida e sostituisce 15 celle dp-fedavg-con-DP già
   pubblicate). Non lanciare finché il gate #3 non è risolto: se cambia il default di
   `floor_mode`/`member_scoring` dopo aver già speso 33h con lo scoring vecchio, quelle ore
   sono da rifare integralmente.
5. **Blocker 3 — ablation filtro 8σ** — bloccato: serve prima il flag opt-in
   `cfg["lira"]["uncalibrated_z_threshold"]` (non ancora scritto).
6. **Sweep di utility largo** — bloccato: servono i 6 valori di ε da concordare con
   l'utente prima di poter dare un comando.
7. **Unificazione attack-surface Yeom/Shadow/LiRA (richiesta dall'utente 2026-09-15,
   sotto)** — implementazione in corso su istruzione esplicita dell'utente ("procedi con
   l'implementazione del design che hai scritto"). Vedi sezione dedicata per i dettagli.

## Blocker aperti dal feedback esterno verificato (errata 2026-09-14) — domanda e risultato atteso

Aggiunto 2026-09-15 su richiesta esplicita dell'utente ("nella roadmap degli esperimenti
scrivere quale domanda vogliamo rispondere e quali sono i risultati che ci aspettiamo").
Per ognuno: la domanda di ricerca precisa, cosa un risultato in un senso o nell'altro
significherebbe per il paper, e lo stato reale (non quello che i Sprint-log a volte lasciano
intendere — "fatto e verificato" per questi item ha sempre significato solo `py_compile` +
suite di test non-torch, MAI un run reale, perché questo sandbox non ha torch).

### Blocker 1 — ablation cold-start vs warm-start dello shadow model (task #120)

**Domanda.** Il retraining warm-started degli shadow ad ogni round (i pesi globali reali
del round precedente, non init casuale) è un adattamento nostro rispetto alla costruzione
originale di Carlini et al. 2022 (shadow indipendenti, init casuale, mai ri-addestrati). Il
null result (~0.50 ovunque) è in parte un artefatto di questo adattamento — il warm-start
potrebbe far convergere gli ensemble IN/OUT l'uno verso l'altro round dopo round, riducendo
la separazione che LiRA misura — o regge anche nella costruzione originale?

**Risultato atteso.** Se `shadow_init=cold` produce AUC ancora ~0.50 (banda 0.48-0.54, come
ogni altro test di questa roadmap): il null result NON dipende dal warm-start, rafforza la
sua robustezza. Se produce un AUC sistematicamente più alto: il warm-start sopprime segnale
reale — da riportare come limite importante, possibile necessità di rifare la campagna
principale con `shadow_init=cold`.

**Stato reale**: flag implementato e verificato (`py_compile` + suite non-torch), zero
impatto sul default. Prima gamba (`shadow_init=warm`, il default — quindi NON nuova
informazione, solo conferma che il comportamento invariato produce il pattern atteso)
eseguita dall'utente 2026-09-15 (`experiments/_blocker1_shadow_warm`, mean_lira_auc_roc=
0.5025). **Manca la seconda gamba** (`config/experiment_shadow_cold.yaml`) e il confronto
via `scripts/compare_floor_mode.py` — solo quel confronto risponde davvero alla domanda.

### Blocker 2 — regime con leakage reale come riferimento (task #121)

**Domanda.** Un AUC~0.50 è credibile come "nessun leakage" solo se sappiamo anche a cosa
assomiglia un AUC quando il leakage C'È — altrimenti non possiamo escludere che l'attacco
sia insensibile. Il canary positive control (office1, Sprint 10vv-10zz+18) risponde a
questo, ma con un solo seed, gemelli sbilanciati (120+ membri vs 19-20 non-membri per
costruzione — mai bilanciati) e solo su office1 (Caltech è invertito 0.37-0.42, JPL è al
caso 0.47-0.49, §6.2). È abbastanza per servire da "regime di riferimento" citabile nel
paper con lo stesso rigore statistico della campagna principale (5 seed, bootstrap CI)?

**Risultato atteso.** Se il canary AUC calibrato regge a 5 seed su office1 con CI che
esclude chiaramente 0.5: il regime di riferimento è stabilito con rigore pari alla campagna
principale, il contrasto "0.50 (nullo) vs X (con leakage reale iniettato)" diventa un
confronto statistico diretto, non un singolo run aneddotico. Se a 5 seed il segnale torna
instabile: il positive control stesso va rivisto (non solo il numero di seed) prima di
usarlo come riferimento.

**Stato reale**: SBLOCCATO 2026-09-15 (Sprint 10zz+93). L'utente ha chiesto esplicitamente
di verificare il leakage reale con "i tre attacchi già implementati", non solo LiRA — prima
di oggi solo `run_lira()` calcolava un `canary_auc_roc` (Sprint 10vv); Yeom e Shadow
attaccavano solo `global_weights` e non avevano alcuna vista ristretta ai canary. Aggiunto
`yeom_canary_auc_roc`/`yeom_canary_advantage`/`yeom_canary_confusion` a `run_fedmia()` e
`shadow_canary_auc_roc`/`shadow_canary_advantage`/`shadow_canary_confusion` a
`run_fedmia_shadow()` (stesso pattern già in uso per LiRA, chiavi prefissate per lo stesso
motivo del fix task #59 — altrimenti il merge yeom→shadow→lira nel dict per round le
sovrascrive silenziosamente). Per Shadow serviva anche una guardia nuova: lo split 50/50
shadow_train/eval_members è casuale e, senza correzione, poteva far finire dei duplicati
canary nello shadow_train — contaminando lo shadow model con gli stessi record che poi
valuta come "membro" (stessa classe di bug già corretta per LiRA in
`_sample_preserving_canary_groups`, Sprint 10zz+16). Ora i canary vengono sempre spostati in
`eval_members` prima dello split. Verificato con `py_compile` + suite di test non-torch
(297/297 passed) — NON ancora con un run reale (nessun torch in questo sandbox).

Comando pronto per la prima gamba (1 seed, no-DP, office1, come già previsto da Fase 0):

```
python3 scripts/run_experiments.py \
  --config config/experiment_canary_positive_control.yaml \
  --no-dp \
  --sweep-dir experiments/_blocker2_canary_nodp_3attacks
```

Se `yeom_canary_auc_roc`/`shadow_canary_auc_roc`/`canary_auc_roc` (LiRA) si staccano tutti e
tre da 0.5 in questa run: i tre attacchi hanno tutti un vero positive control, non solo LiRA
— il regime di riferimento del Blocker 2 può essere esteso a tutti e tre. Se solo LiRA si
stacca e Yeom/Shadow restano a ~0.5: significativo di per sé (i tre attacchi non sono
equivalenti in sensibilità nemmeno di fronte a un leakage iniettato aggressivamente), da
riportare esplicitamente. Il bilanciamento membri/non-membri, i "gemelli veri" e l'estensione
a 5 seed restano lavoro separato, non ancora iniziato.

**Risultato reale (2026-09-15, eseguito dall'utente sulla propria macchina)** —
`experiments/_blocker2_canary_nodp_3attacks/experiment_20260915_101915.json`, 1 seed (42),
no-DP, office1, 3 round. Esito **misto, non il caso "tutti e tre si staccano" previsto sopra**:

- **LiRA**: `canary_auc_roc` per round 0.6375/0.6625/0.5875, `canary_composed_auc_roc`
  0.6875 — riproduce quasi esattamente il risultato già documentato in Sprint 10zz+18
  (stesso composed 0.6875), coerente e stabile.
- **Yeom**: `yeom_canary_auc_roc` per round 0.70/0.86/0.86 — segnale FORTE, più alto di
  LiRA. Il positive control regge anche per un attacco che vede solo `global_weights`
  aggregati, non l'update del singolo client.
- **Shadow**: `shadow_canary_auc_roc` per round 0.26/0.32/0.32 — **sotto 0.5, cioè invertito
  rispetto all'atteso** (advantage quasi nullo: 0.0/0.1/0.1). Non rumore statistico da
  campione piccolo (n_member=150, n_nonmember=20, stesso ordine di grandezza di LiRA/Yeom
  che invece si staccano nettamente).

**Diagnosi (verificata leggendo il codice, non solo ipotizzata)**: `inject_canaries()`
(riga ~609) sceglie `n_templates` sessioni REALI già presenti in `site_train_sessions`
come template, poi **aggiunge** `n_duplicates` cloni taggati (`_canary_role="member"`)
SENZA rimuovere o taggare l'occorrenza originale — l'originale resta nel pool come sessione
membro ordinaria, indistinguibile da qualunque altra. La guardia anti-contaminazione di
Shadow (Sprint 10zz+93, sposta le sessioni con `_canary_role=="member"` da `shadow_train` a
`eval_members`) non la vede, perché l'originale non porta il tag. Se quell'occorrenza
originale finisce per caso in `shadow_train` (50% di probabilità), lo shadow model si
allena direttamente sullo stesso identico vettore di feature dei 30 duplicati canary — con
1000 epoche su un autoencoder piccolo, può arrivare a una loss bassa quanto o più bassa di
quella del target model (che ha comunque 30 copie a rinforzare il segnale, ma su un pool di
training più grande), producendo uno score calibrato (`shadow_loss - target_loss`) vicino a
zero o negativo per quel gruppo — invertendo l'AUC se questo capita per un numero
sufficiente dei 5 template. LiRA non mostra lo stesso collasso probabilmente perché la sua
calibrazione usa `n_shadow=8` modelli shadow distinti pescati dall'universo per-cluster
(`_sample_preserving_canary_groups`, che tratta i gruppi TAGGATI come atomici — ma
anch'essa non vede l'originale non taggato): l'effetto di un singolo shadow contaminato si
dilua nella media di 8, mentre Shadow ha UN SOLO modello shadow e quindi zero diluizione.

**Fix implementato (2026-09-15, Sprint 10zz+96, confermato dall'utente: "procedi con la
correzione del fix")**: taggata anche l'occorrenza originale del template con lo stesso
`_canary_group`/`_canary_role="member"` dei suoi cloni (sostituita dentro `injected_train`
con una copia, non mutando l'oggetto condiviso con `train_sessions`). Verificato che il
fix si applica correttamente: `n_members` nella confusion matrix canary passa da 150 a 155
(31 × 5, non più 30 × 5 + 5 invisibili). `py_compile` OK, suite non-torch 297/297 invariata.

**Risultato del rerun (`experiments/_blocker2_canary_nodp_3attacks_v2`) — il fix NON
risolve l'inversione, anzi la peggiora leggermente**: `shadow_canary_auc_roc`
0.21/0.24/0.25 (era 0.26/0.32/0.32 prima del fix). **La contaminazione dell'originale non
taggato NON era la causa (dominante) dell'inversione** — l'ipotesi iniziale era sbagliata.
Effetto collaterale reale del fix: `canary_composed_auc_roc` di LiRA scende da 0.6875 a
0.66 (resta comunque sopra 0.5 — il fix ha corretto un buco di isolamento analogo anche
nella calibrazione shadow di LiRA, non solo in quella di Shadow).

**Diagnostica aggiuntiva (Sprint 10zz+97)** — campi `shadow_canary_debug_group_means`/
`_nonmember_stats`, puramente additivi — isolano il punteggio calibrato medio per ciascuno
dei 5 gruppi canary. Risultato (`experiments/_blocker2_canary_nodp_3attacks_v3`, round 1):

```
group_means:      canary_m0=0.000098  canary_m1=0.000040  canary_m2=0.000041
                   canary_m3=0.000145  canary_m4=-0.000128
nonmember_stats:   mean=0.000527  std=0.000566  min=-0.000032  max=0.001746
```

Pattern identico in tutti e 3 i round. **Non è varianza di campione (1-2 outlier)**: TUTTI
e 5 i gruppi canary hanno una media calibrata sistematicamente più bassa della media
non-membro, in ogni round — un effetto sistematico, non rumore.

**Causa radice isolata, con dati incrociati**: `yeom_canary_auc_roc` nello stesso run è
0.70/0.86/0.86 — e Yeom legge la loss ASSOLUTA dello stesso identico `target_model`
(stessi `global_weights`, stesso round) usato dentro Shadow. Questo conferma che
`target_loss` sui canary è genuinamente molto più bassa che sui gemelli (memorizzazione
reale, non un artefatto) — quindi la spiegazione "il target non memorizza abbastanza" è
esclusa. L'unica spiegazione compatibile con i numeri è che `shadow_loss` sia ANCH'essa
bassa sugli stessi 5 campioni canary, quasi quanto `target_loss` — cancellando la
differenza che la calibrazione (`shadow_loss - target_loss`) dovrebbe rilevare. Ipotesi
più plausibile: con un modello a 6 feature/latent_dim=4, sia lo shadow (500 epoche, metà
dei dati reali) sia il target arrivano vicino al proprio "pavimento" di errore di
ricostruzione per sessioni tipiche di questo sito — la differenza assoluta shadow-vs-target
a questa scala (ordine 1e-4/1e-3) è dominata dalla difficoltà di ricostruzione intrinseca
delle singole sessioni scelte come template/gemelli, non dalla membership. Non è un bug di
codice individuabile — è una probabile LIMITAZIONE METODOLOGICA della calibrazione
shadow-singolo a differenza assoluta, specifica di questo regime (modello minuscolo, solo
5 template), che l'ensemble a 8 shadow Gaussiani di LiRA attutisce meglio (LiRA infatti
resta positivo, 0.66 composed) e che Yeom evita del tutto (nessuna calibrazione shadow,
solo loss assoluta sotto il target).

**Stato**: Blocker 2 risolto per LiRA e Yeom (positive control reale confermato con dati,
causa d'inversione isolata per Shadow con evidenza incrociata anche se non "corretta" nel
senso di codice). Raccomandazione: documentare la limitazione di Shadow nel paper (§6.2/§9,
accanto alla differenza di sensibilità già discussa tra i tre attacchi) invece di continuare
a rincorrere un fix di codice — ulteriori tentativi hanno rendimento calante e i dati
incrociati con Yeom rendono la spiegazione già solida. Decisione finale (chiudere qui o
investigare oltre con un dump dei valori assoluti shadow_loss/target_loss) rimandata
all'utente. Task #121 sostanzialmente risolto, in attesa di conferma per la chiusura.

### Blocker 3 — ablation del filtro 8σ (task #122)

**Domanda.** `_UNCALIBRATED_Z_THRESHOLD=8.0` esclude campioni troppo lontani da entrambe le
distribuzioni IN/OUT invece di forzarli con un segno arbitrario (Sprint 10aa). Il paper
(§3.5) riporta uno skip rate reale dello 0.0000% nella campagna finale, cioè il filtro non
scarta MAI nulla a questa scala — ma questo lo sappiamo dal contatore, non da un confronto
diretto. Disattivare il filtro del tutto cambia davvero zero, come il contatore suggerisce?

**Risultato atteso.** Se disattivare il filtro (soglia effettivamente infinita) produce un
AUC identico (atteso, dato skip_rate=0.0000% già misurato): conferma empiricamente, non solo
per inferenza dal contatore, che il filtro è dormiente per questi dati — la citazione nel
paper diventa più forte ("verificato disattivandolo" invece di "il contatore mostra zero
skip"). Se cambia qualcosa: il contatore stesso ha un bug, da investigare prima di
qualunque claim su §3.5.

**Stato reale**: non eseguito, e a differenza di Blocker 1 la soglia non è ancora
configurabile via `cfg["lira"]` — serve prima un flag opt-in analogo (es.
`cfg["lira"]["uncalibrated_z_threshold"]`, default 8.0) prima di poter dare un comando
eseguibile.

### RQ6 (nuova, 2026-09-15) — il rilevamento Byzantine sopravvive alla DP?

**Contesto.** L'ML Plane ha due consumatori con modelli di minaccia ortogonali sullo stesso
substrato di osservabilità: il Privacy Auditor (confidenzialità, server honest-but-curious)
e ByzantineDetector/IDS (integrità, client malevolo). `byzantine_attack.enabled` è `false`
in tutti e 13 i file di config esistenti e nessun `experiments/*.json` archiviato mostra
`byzantine_detected: true` — confermato via grep/find su tutta la repo. Il percorso di
codice esiste ed è testato SOLO a livello unitario (`gradient_scaling`, client sintetici
`synthetic_1`/`synthetic_2` per arrivare a n≥5 come richiede Krum), mai su un run FL reale.
Osservazione dell'utente: l'unica differenza tra `local` e `dp-fedavg` è se `raw_updates`
viene salvato — che è esattamente l'input dell'IDS. Sotto `local` il server non vede mai gli
aggiornamenti grezzi: protegge dal server curioso e nello stesso momento acceca il
rilevatore Byzantine. `ByzantineDetector` usa soglia cosine similarity 0.3 e soglia Krum 3.5
(verificato in `src/ids/charging_ids.py`, righe 519-541 — non citazione a memoria).

**Domanda.** Il rumore DP calibrato per la privacy (σ=4.84 a ε=1.0) rende inoperante il
rilevamento Byzantine sullo stesso flusso, indipendentemente dal posizionamento — e quanto?

**Verifica di plausibilità (fatta ora, non un run reale — una simulazione numerica pura
con `numpy`, nessun training, solo per stabilire se vale la pena spendere 10-15h)**: due
aggiornamenti onesti simulati (stesso obiettivo, piccola perturbazione locale, norma
0.1-1) hanno cosine similarity media 0.99 senza rumore. Con rumore gaussiano isotropo
σ=4.84 aggiunto a entrambi, la similarità media crolla a ~0.01 (range osservato
-0.05...+0.07) — sotto la soglia di 0.3 nel 100% delle 200 coppie oneste simulate.
L'ipotesi dell'utente è quantitativamente molto plausibile: a questo livello di rumore il
detector non solo perderebbe un vero attaccante (falso negativo), ma classificherebbe come
sospetti anche client completamente onesti (falsi positivi) — il segnale coseno diventa
puro rumore isotropo, la componente correlata (il vero gradiente) è sommersa. Questa è
un'indicazione di plausibilità su un modello giocattolo, non una misura sui pesi reali
dell'autoencoder — il run reale resta necessario.

**Risultato atteso.** Se `byzantine_detected` resta `false` (o il tasso di rilevamento crolla)
sotto DP attiva a qualunque ε testato, ma funziona nel baseline no-DP: risultato pratico
diretto per chi progetta un deployment — la DP calibrata per la privacy rende inoperante il
rilevamento di integrità sullo stesso flusso, e (ipotesi secondaria) `local` lo peggiora
ulteriormente rispetto a `dp-fedavg`/`central` perché nemmeno l'IDS vede l'update grezzo. Se
il rilevamento regge sotto DP: risultato comunque interessante (la soglia adattiva o la
normalizzazione esistente compensano meglio del previsto), da investigare perché.

**Metodologia (stesso principio già imposto ai canary — prima il positive control, poi la
soppressione)**: baseline no-DP PRIMA di ogni cella con DP attiva, altrimenti non possiamo
distinguere "la DP sopprime il rilevamento" da "il rilevamento non ha mai funzionato quaggiù".

**Stato reale**: non eseguito, nessun nuovo codice necessario (il percorso `--byzantine` è
già wired end-to-end, solo mai stato usato con dati reali). Comando per la baseline
(positive control, DA LANCIARE PER PRIMO):

```
python3 scripts/run_experiments.py \
  --config config/experiment.yaml --no-dp \
  --byzantine --byzantine-node synthetic_1 --scale-factor 10 \
  --sweep-dir experiments/_rq6_byzantine_nodp_scale10
```

Se `byzantine_detected: true` compare qui (come atteso, nessun rumore a disturbare la
similarità coseno): ripetere con `--scale-factor 3` (caso borderline) e poi con DP attiva
(`--dp-mode central|local|dp-fedavg --epsilon 1.0`, poi 0.5 e 0.1) per lo stesso confronto.
Se NON compare nemmeno qui: il problema è nel detector o nella sua soglia a questa scala di
pesi, non nella DP — da investigare prima di procedere con le celle DP.

### Unificazione attack-surface Yeom/Shadow/LiRA (richiesta dall'utente 2026-09-15) — design, non implementata

**Richiesta.** "Per tutti gli attacchi dobbiamo usare la stessa linea: Yeom e Shadow devono
poter attaccare anche gli update per-client (come LiRA/Strada B), e LiRA deve poter
attaccare anche i global_weights aggregati (come Yeom/Shadow), per chiudere il cerchio."

**Perché non è stata implementata in questo stesso giro.** Le due direzioni hanno un costo
molto diverso. LiRA-su-global_weights è relativamente contenuto: LiRA già calibra
per-campione (σ_in/σ_out via shadow), quindi sostituire "quale modello produce
target_loss" (un `client_model` per-sito → un unico modello globale condiviso per round,
come già fa `run_fedmia()`) è un cambiamento localizzato. Yeom/Shadow-su-updates è invece
strutturale: oggi caricano UN modello per round (`global_weights`) e valutano l'intero pool
membri/non-membri contro quello; per attaccare gli update per-client dovrebbero, per ogni
round, iterare sui singoli client (i tre siti reali), caricare il modello di ognuno da
`update.weights`/`raw_updates` (a seconda di `dp_mode`, stessa logica già scritta per LiRA
in Strada B), e restringere membri/non-membri al solo sito di quel client — un secondo
livello di loop che oggi non esiste in queste due funzioni, per giunta dentro una funzione
(`run_lira()`, ~2200 righe) già stratificata da anni di fix incrociati (floor_mode,
matched_formula, canary, calibrazione a cluster, diagnostica composed-score). Implementarlo
"alla cieca" in un solo giro, senza poter eseguire torch per verificarlo, rischia di
introdurre un bug sottile che emergerebbe solo dopo un run reale di ore — un rischio peggiore
di un breve rinvio per un design più accurato. Le correzioni "sicure" fatte oggi (Blocker 2
sopra, e la nota epsilon dell'auditor) erano invece additive/isolate e verificabili con
`py_compile` + test non-torch senza questo rischio.

**Piano proposto (da confermare con l'utente prima di implementare):**
- `cfg["lira"]["observation_surface"]`: `"client"` (default, invariato) | `"global"` (nuovo
  — LiRA valuta contro `round_data["global_weights"]`, un solo modello condiviso per round,
  stesso schema di calibrazione σ_in/σ_out già esistente per campione).
- `cfg["yeom"]["observation_surface"]`/`cfg["shadow"]["observation_surface"]`: `"global"`
  (default, invariato) | `"client"` (nuovo — richiede il secondo livello di loop per-sito
  descritto sopra, raddoppia il costo computazionale di quei due attacchi per un fattore
  pari al numero di siti).
- Tutti e tre gli opt-in, zero impatto sul default, stesso principio già rispettato per
  `member_scoring`/`floor_mode`/Strada B.

**Stato reale**: design proposto, non implementato. In coda dopo Blocker 2/RQ6/Strada
B-rerun nella lista di priorità sopra — non perché meno interessante, ma perché è la voce a
più alto rischio di introdurre un bug non rilevabile senza un run reale, e le prime tre sono
più economiche e più urgenti (soprattutto Blocker 2, che valida l'intero impianto di misura).

### Sweep di utility largo — trovare un ε operativo (non ancora nella roadmap prima di oggi)

**Domanda.** La tabella di utility già in §7 (Sprint 10zz+87/+88) mostra che ogni
configurazione DP *già testata* (ε∈{1.0,0.5,0.1}) aumenta la loss di ricostruzione di
~240-290× rispetto a no-DP — un modello quasi non funzionale. Ma copre solo i 3 livelli di
ε già scelti per la campagna principale, tutti a quanto pare oltre la soglia di rottura.
Esiste un ε più permissivo (es. 10, 50, 100) per cui il modello resta utilizzabile, così da
sapere se i 3 livelli della campagna sono stati scelti alla cieca dentro la sola zona
"il modello è comunque rotto"?

**Risultato atteso.** Se esiste un ε dove mean_loss torna vicino al baseline no-DP: quell'ε
diventa un punto operativo interessante da aggiungere alla campagna (nuova cella, nuovo
run completo) — cambia l'interpretazione del null result, perché a quel punto operativo
DP potrebbe finalmente "avere qualcosa da sopprimere" testando l'utility reale. Se anche a
ε=100 il modello resta rotto: rafforza ulteriormente la lettura già in §7 (il confondente
utility/leakage copre l'intero range testato, non solo i 3 punti scelti).

**Stato reale**: non eseguito, non era nella roadmap prima di oggi. Costo indicato
dall'utente: ~13 ore, un seed, sei punti di ε — comando ancora da preparare (dipende da
quali 6 valori di ε scegliere, da concordare).

### Diagnosi canary Caltech — amplificazione per record, non densità aggregata

**Nota (non un blocker, una correzione alla diagnosi già in tabella riga 2 sopra).** La
riga 2 di "Vista d'insieme" attribuisce il fallimento del canary a Caltech/JPL alla
scale-dependence (duplicati ~11% a office1 vs ~0.5% altrove) e cita una replica high-density
che avrebbe "appianato" la densità aggregata al 10% — ma l'ha fatta con 84 template × 30
copie invece di 5 template × 30 copie: la densità TOTALE coincide con office1, mentre
l'amplificazione per singolo record resta ~17× inferiore (è quest'ultima, non la densità
aggregata, a guidare la memorizzazione di un record specifico). L'esperimento che doveva
chiudere la domanda non l'ha testata.

**Stato (2026-09-15, Sprint 10zz+101 — su richiesta esplicita dell'utente "bisogna
allineare gli esperimenti anche a questo dataset")**: config pronto, calcolo verificato
contando le sessioni reali (non a memoria da un commento precedente, che citava sia 1680
sia 1344 per office1 senza specificare quale fosse quello giusto da usare qui):

```
office1: 3 file JSON = 1680 sessioni totali, split 80/20 → pool training = 1344
         amplificazione per-record = n_duplicates / pool = 30 / 1344 = 2.2321%
caltech: 4 file JSON = 31404 sessioni totali, split 80/20 → pool training = 25123
         n_duplicates necessario = 2.2321% × 25123 ≈ 560.85 → 561
```

Tenendo `n_templates=5` invariato (NON 84, a differenza della replica high-density) e
alzando solo `n_duplicates` a 561, densità aggregata E amplificazione per-record coincidono
entrambe con office1 per costruzione (5×561/25123 = 11.16% = 5×30/1344) — le due variabili
che la replica high-density aveva scollegato tornano ad essere la stessa cosa, senza dover
scegliere quale isolare. Nuovo config:
`config/experiment_canary_positive_control_caltech_amplification_matched.yaml`.

Include automaticamente l'estensione Blocker 2 (canary su Yeom/Shadow/LiRA insieme, attiva
di default dal Sprint 10zz+93) — un solo run copre sia la domanda sull'amplificazione
per-record sia l'allineamento di Caltech al lavoro già fatto su office1, senza bisogno di
comandi separati. Usa già il fix `inject_canaries()` del Sprint 10zz+96 (tag dell'originale),
quindi non è comparabile bit-per-bit con i 3 run Caltech precedenti (tutti pre-fix), ma è
comparabile con i run office1 più recenti (v2/v3 di Blocker 2).

**Comando**:
```
python3 scripts/run_experiments.py \
    --config config/experiment_canary_positive_control_caltech_amplification_matched.yaml \
    --no-dp --sweep-dir experiments/_canary_positive_control_caltech_amplification_matched
```

**Lettura attesa**: se `canary_auc_roc`/`yeom_canary_auc_roc` si avvicinano ai valori
office1 (LiRA composto ~0.6-0.7, Yeom ~0.7-0.9), l'amplificazione per-record è confermata
come variabile che conta, risolvendo la domanda aperta di §6.2. Se restano al livello del
caso anche qui, l'ipotesi densità/amplificazione va scartata a favore di una vera differenza
di sito — da riportare esplicitamente, non da assumere. **ATTENZIONE TEMPI**: 2805 record
duplicati aggiuntivi su un pool di 25123 sessioni — tempo atteso comparabile o lievemente
superiore alla replica high-density già eseguita (2520 record).

**Correzione minore trovata durante questa verifica (non ancora propagata al resto del
documento)**: la cifra "Caltech è invertito 0.37-0.42" citata altrove in questo file e nel
paper proviene dal run ad alta densità (84 template), non dalla replica diretta a 5 template
(che ha dato invece 0.4328/0.4383/0.3503) — entrambe sotto il caso, la conclusione
qualitativa non cambia, ma la citazione andrebbe resa precisa (quale run, quali numeri) la
prossima volta che si tocca questa sezione.

---

### Audit ChargePlace Scotland — stato e piano di allineamento (2026-09-15, Sprint 10zz+102)

**Su richiesta esplicita dell'utente** (dopo aver corretto un proprio refuso: si era
riferita a Caltech quando intendeva questo dataset), stato completo di ChargePlace Scotland
rispetto a quanto fatto su ACN-Data (caltech/jpl/office1):

**Fatto**: adapter (`src/adapters/chargeplace_scotland_adapter.py`, stesso contratto
`AbstractDataset`/6 feature di ACN), wiring in `run_experiments.py` (`dataset_adapter:
chargeplace_scotland`) e in NVFLARE, fix del bug UTC/ora-legale (sessioni erano salvate in
ora locale Europe/London grezza invece di UTC, sfasamento sistematico di +1h in BST). **Un
solo run reale**: no-DP, 3 client (Glasgow City/East Ayrshire/City of Edinburgh, 688896
sessioni), `experiments/experiment_20260910_044144.json`, `mean_auc_roc≈0.5008`,
`mean_shadow_auc_roc≈0.5007`, `mean_lira_auc_roc≈0.5006` (LOW) — **ma questo run precede il
fix UTC**, quindi non è verificato con il codice corretto.

**Non fatto**: nessun rerun post-fix, nessun canary positive control, nessuno sweep
DP-placement, nessuna replica multi-seed, nessuna integrazione nel paper. Limite strutturale
permanente (non risolvibile): `user_id` sempre `None` → nessuno split entity-aware possibile
su questo dataset; `kwh_requested`/`minutes_available` sempre 0.0/0 → solo 4 delle 6 feature
sono informative.

**Costo/rischio esplicito**: il sottoinsieme a 3 client (688896 sessioni) è ~10.3× l'intero
ACN-Data (66713 sessioni, 3 siti, tutti gli anni) — la campagna paper 5-seed×10-config su
quel volume ha impiegato ~108 ore. Il tempo NON scala necessariamente in modo lineare con la
dimensione dati (dipende da quanti shadow LiRA vengono riaddestrati e su quanti dati), va
misurato empiricamente. `config/experiment_chargeplace_scotland.yaml` lo segnala già da sé
(nota SMOKE TEST in fondo al file) raccomandando un test su 2-3 mesi prima di una campagna
piena, vista la deadline abstract del 2026-11-25.

**Compatibilità canary verificata (non assunta)**: `inject_canaries()` risolve il sito di
ogni sessione via `_resolved_site_name()` = `_SITE_ID_TO_NAME.get(raw, raw or "unknown")`.
Per Scotland `site_id` è già il nome leggibile della council area (es. "Glasgow City",
popolato da `self._local_authority.get(cpid, "")` nell'adapter) — non presente nella mappa
dei 3 codici ACN, quindi il fallback lo passa invariato: un `canary.site: "Glasgow City"`
funziona senza nessuna modifica al codice. `inject_canaries()` è inoltre invocato in modo
generico dopo il caricamento sessioni, indipendente da `cfg["dataset_adapter"]`.

**Piano approvato dall'utente, 3 componenti (in ordine di costo crescente)**:

1. **Re-run economico del baseline no-DP**, config invariato, ora che il fix UTC è in vigore
   — verifica minima che la conclusione AUC≈0.50/LOW regga col codice corretto:
   ```
   python3 scripts/run_experiments.py \
       --config config/experiment_chargeplace_scotland.yaml \
       --no-dp --sweep-dir experiments/_chargeplace_scotland_baseline_postfix
   ```

2. **Canary positive control**, single-site Glasgow City (il più grande dei 3 client,
   260347 sessioni totali). `n_duplicates=4650`, calcolato con la stessa amplificazione
   per-record di office1 (2.2321% × pool training 208277 = 260347×80% ≈ 4649.6 → 4650),
   `n_templates=5`/`n_nonmember_templates=20` invariati. **Deviazione deliberata:
   `epochs=50`, non 1000 come negli altri config canary** — a questa scala (Glasgow City da
   sola è ~155× office1 per volume) 1000 epoch sarebbe un costo enorme e incerto prima della
   deadline; il canary è quindi testato nelle stesse condizioni di training del run
   principale, con l'avvertenza che un esito negativo non distinguerebbe "amplificazione
   insufficiente a questa scala" da "50 epoch non bastano a far memorizzare il canary" — le
   due variabili non sono isolate in questo run. Nuovo config:
   `config/experiment_canary_positive_control_chargeplace_scotland.yaml`.
   ```
   python3 scripts/run_experiments.py \
       --config config/experiment_canary_positive_control_chargeplace_scotland.yaml \
       --no-dp --sweep-dir experiments/_canary_positive_control_chargeplace_scotland
   ```

3. **Sweep DP-placement** (dp-fedavg/central/local × ε∈{1.0,0.5,0.1}) — **la componente più
   costosa e rischiosa rispetto alla deadline**, non richiede nuovi file YAML (dp-mode/
   epsilon sono flag CLI, non nel config), quindi è la stessa matrice di comandi già usata
   per ACN-Data applicata a `config/experiment_chargeplace_scotland.yaml`:
   ```
   for dp_mode in dp-fedavg central local; do
     for eps in 1.0 0.5 0.1; do
       python3 scripts/run_experiments.py \
           --config config/experiment_chargeplace_scotland.yaml \
           --dp-mode $dp_mode --epsilon $eps \
           --sweep-dir experiments/_chargeplace_scotland_dp_sweep
     done
   done
   ```
   Da lanciare per ultima, solo dopo aver visto il tempo reale dei primi due run — 9
   run interi su un dataset ~10× più grande di ACN-Data rischiano concretamente di non
   stare nei tempi prima del 25/11.

Nessun run eseguito in questa voce — solo verifica di compatibilità (read-only) e
preparazione di config/comandi; i tre passi sopra restano da lanciare sulla macchina fisica
dell'utente, nell'ordine dato.

---

## Dipendenze tra i test

```
Fix #3 (fatto) ──┐
                  ├──> Analisi campagna #1 (bootstrap CI, Wilcoxon #5) ──> tabella risultati paper
Campagna #1 ──────┘

Sanity-check #2 ──> conferma che l'harness è sensibile ──> via libera a scrivere la tesi "nessun leakage"
                     (se fallisce: blocca tutto il resto, richiede nuova indagine)

TPR@low-FPR #4 ──> arricchisce la tabella risultati, non blocca la campagna #1

#6, #7, #8 ──> indipendenti, solo se c'è tempo

Rilancio campagna con metriche complete #13 (2026-09-03) ──> SOSTITUISCE #1 come blocco
    principale rimasto — P1 (central/local ε=0.5, mai eseguiti) prima di P2/P3/P4 (backfill
    metriche su config che hanno già dati preliminari) ──> tabella risultati finale paper
```
