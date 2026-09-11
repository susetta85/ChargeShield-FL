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
  (`mean_lira_auc_roc` 0.5000/0.5015, `privacy_risk=LOW`, nessuna anomalia); 789/1234 completi lato
  training, rianalisi in corso/da fare.
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

**Ancora da fare per un confronto pienamente allineato con la campagna single-process (10 config ×
5 seed)**: `local` DP mode (zero run finora su NVFLARE) e variazione di epsilon (0.5/0.1 — solo
ε=1.0 testato finora, bassa priorità esplicita). `central` è ora completo a 5 seed (vedi sopra). Non bloccante per la submission — il claim
principale del paper si basa sulla campagna single-process, già completa e statisticamente
solida; questi run NVFLARE sono una validazione supplementare "il risultato regge anche in un
deployment reale multi-container", non un sostituto. Costo/beneficio da valutare rispetto al tempo
restante prima della deadline (abstract 2026-11-25).

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
