# ChargeShield-FL — Esperimenti da eseguire

> **Stato: documento CANONICO.** Aggiornato il 2026-09-24 (revisione: costo, E-C, controlli (a) e (b) eseguiti, griglia del punto operativo, canary su più siti). Sostituisce,
> per gli esperimenti ancora da lanciare, il vecchio `TestRoadmap_DSN2027.md`, eliminato
> il 2026-09-22 e recuperabile dalla storia git. Ogni
> voce dice quale RQ serve, quale conclusione può cambiare, cosa deve essere vero
> prima di lanciarla, il comando, e come si legge l'esito. Le run vanno una alla
> volta sulla stessa macchina (lock anti-concorrenza nel Makefile, OOM del
> 2026-07-31). Ordine di priorità della guida: RQ1, poi RQ3, poi RQ2.

## 0. Decisioni da congelare PRIMA di leggere i risultati di E-A

La guida (sezioni 5 e 8) chiede di fissare questi campi prima della conferma, non
dopo. Proposta da confermare con il supervisore e registrare nella scheda
`risultati/decision_matrix_ACN_membership_DP.xlsx`:

| campo | proposta |
|---|---|
| metrica primaria di attacco | loss grezza (Yeom) sul modello globale e conteggio per record con z sui `composed_score` di LiRA; LiRA calibrato e Shadow come secondarie |
| superficie primaria | update del singolo client (A1-A3) per il per-record; modello globale per Yeom e Shadow, come nella campagna |
| FPR operativo | 1%, con TPR = FPR come livello di caso; 0.1% solo dove n lo sostiene |
| costo accettabile | loss finale entro 3 volte il riferimento no-DP (pilot: ε = 16 sta a 2.6, ε = 8 a 4.6) |
| margine di equivalenza | AUC entro 0.02 da 0.5 (TOST), come già in `check_significance.py` |
| unità di replica | il seed; 5 seed per cella; le copie di un canary non sono repliche |

**Deviazioni documentate il 2026-09-24.** (1) Costo: la definizione resta "loss finale
entro 3 volte il riferimento no-DP", intesa come loss sull'holdout del modello globale
rilasciato; `genera_matrici_faseA.py` usava invece la loss di addestramento locale
(segnalazione 46). Corretta l'implementazione, non la regola. (2) Metrica primaria per
record: il "conteggio per record con z" misura la stabilità del ranking e non
l'appartenenza (segnalazione 47). Proposta: sostituirlo con il test appaiato, tenendo il
conteggio come risultato secondario. Entrambi i risultati sono riportati in `STATO.md`
3.2. **Da confermare con il supervisore.**

## Regole che valgono per tutti

- **Scorer LiRA congelato.** Il riferimento worst-case no-DP (`nodp-sweep2`,
  z = 7.56) è calcolato sui `composed_score` dello scorer attuale. Nessuna modifica
  a floor, `member_scoring` o `observation_surface` prima che E-A ed E-B siano
  completi. Le indagini su LiRA (segnalazione 10) si fanno sulla cella canary.
- **Dump per campione sempre attivi** (`--per-sample-dump`).
- **Seed**: 42, 123, 456, 789, 1234, uno per run, nella stessa sweep-dir.
- **Dopo ogni sweep**: `python3 scripts/check_significance.py` e
  `python3 scripts/genera_matrici_faseA.py`; i numeri si leggono da `risultati/`.
- **Regime naturale e canary non si mescolano** (guida, Fase C).

## Prerequisiti di codice (dalla lista segnalazioni)

| prima di | correzione | segnalazione |
|---|---|---|
| E-C | unificare il percentile dei due script worst-case e rigenerare i JSON grezzi mancanti | 6 |
| **E-B, bloccante** | `check_significance.py` e `genera_matrici_faseA.py` devono distinguere le celle record-DP: oggi raggruppano per `(dp_mode, epsilon, no_dp, seed)` e una run record-DP lanciata con `--no-dp` finisce nel gruppo "no-DP baseline", dove, essendo più recente, **sostituisce** il seed corrispondente di `nodp-sweep2` | 37 |
| E-B | ~~accountant record-DP: `fl_rounds` e `delta` da `cfg["experiment"]`, n per client, dichiarare Poisson vs shuffle, `dp-accounting` in `pyproject.toml`~~ corretto il 2026-09-24 | 4 |
| E-B | il warning `[NO-DP BASELINE]` deve controllare `record_dp.enabled` | 5 |
| B1 | manca un flag clip-only (clipping attivo, σ = 0); va aggiunto o il braccio va dichiarato non eseguito | 38 |
| rianalisi NVFlare | `--client-config` con lo snapshot del seed, già nello script rigenerato | 1 |

## E-A — sweep di ε nella zona del ginocchio. PRIORITÀ MASSIMA

**RQ1, contrasto B0/B2 della guida.** A ε in {1, 0.5, 0.1} la loss è oltre 100
volte il riferimento; a ε = 16 è 2.6 volte, a ε = 8 4.6 volte, ma con un seed
solo. Servono celle in cui la DP è attiva e il modello funziona.

**Costo.** 20 run a circa 130 minuti, circa 2 giorni.

```bash
caffeinate -ims bash -c '
set -e
for e in 2 4 8 16; do
  for s in 42 123 456 789 1234; do
    python3 scripts/run_experiments.py \
      --config config/experiment_rq1_eps$e.yaml --rounds 10 --seed $s \
      --sweep-dir experiments/rq1-eps$e \
      --per-sample-dump experiments/rq1-eps$e/per_sample_seed$s.json
  done
done' 2>&1 | tee logs/rq1_eps_sweep.log
```

**Lettura, secondo la tabella della Fase C della guida.** Per cella: loss finale
in rapporto al no-DP, AUC dei tre attacchi, TPR a FPR 1%, e dopo E-C il conteggio
per record con z.

| esito | conclusione consentita | passo successivo |
|---|---|---|
| in una cella a costo accettabile l'eccesso per record scende sotto z = 3 | riduzione osservata dell'esposizione per record, per questi attacchi e questo accesso | quantificare il costo, fissare quell'ε come punto operativo per E-E ed E-D |
| l'eccesso resta e il costo è accettabile | la DP client-level a quell'ε non riduce l'esposizione misurata | riportarlo; E-B diventa il confronto decisivo |
| il costo supera la soglia in ogni cella | nessun punto operativo utile per il client-level in questo regime | è la risposta a RQ1 per questa unità; passare a E-B |
| risultati incoerenti fra seed | l'esperimento non quantifica bene il fenomeno | aumento mirato dei seed sulla sola cella ambigua, con limite di risorse |

**Stato (2026-09-24).** Eseguito dal 22 al 24 settembre, 20 run; numeri in `STATO.md`
sezione 3.1b. Da fare prima di chiudere la cella ε = 2: rilanciare il solo seed 789,
i cui attacchi sono falliti per un errore d'ambiente (segnalazione 42):

```bash
caffeinate -ims python3 scripts/run_experiments.py \
  --config config/experiment_rq1_eps2.yaml --rounds 10 --seed 789 \
  --sweep-dir experiments/rq1-eps2 \
  --per-sample-dump experiments/rq1-eps2/per_sample_seed789.json \
  2>&1 | tee logs/rq1_eps2_seed789_rerun.log
```

**Lettura (2026-09-24, dopo la segnalazione 46).** Seed 789 a ε = 2 rilanciato: 5 seed
validi per cella. Con il costo sul modello rilasciato nessuna cella sta sotto 3 volte
(ε = 16: 60 volte `nodp-sweep2`): terza riga della tabella, "nessun punto operativo
utile per il client-level in questo regime". E-C non trova segnale per record in
nessuna cella. ε = 8 non è più incoerente fra seed sulla misura corretta: niente seed
aggiuntivi. Prossimi passi: E-B e la ricerca di un punto operativo client-level, sezioni
sotto.

## B1 — braccio clipping senza rumore

**Fase B della guida.** Separa il contributo del clipping da quello del rumore.
Non esiste un flag: `central` clippa per client ma aggiunge rumore all'aggregato.
Serve un `--clip-only` (segnalazione 38), poi una cella a 5 seed sulla
configurazione ordinaria. Se il tempo non basta, dichiarare il braccio non
eseguito e attribuire l'effetto a "clipping più rumore" insieme.

## E-B — record-level DP su dati naturali

**RQ1, fattore unità protetta.** Le dieci run record-DP esistenti sono tutte in
regime canary, a σ = 5, su tre seed. Il confronto client contro record sulla stessa
configurazione ordinaria non è mai stato eseguito.

**Attenzione al flag.** `record_dp` non disattiva il meccanismo client-level: senza
`--no-dp` il config `rq1_recorddp_nm*` applica anche clip e rumore client-level a
ε = 1 e distrugge l'utility, rendendo la cella inutile. Le run record-DP canary
sono state tutte lanciate con `--no-dp`; qui va fatto lo stesso. Da qui la
segnalazione 37: senza la correzione al raggruppamento, queste run con
`no_dp=True` cancellerebbero il riferimento no-DP nelle statistiche.

**Prima di tutto: una run di prova a un seed** per misurare il tempo. DP-SGD a
microbatch 1 su tre siti, 50 epoche, 10 round non è mai stato cronometrato.
Primo tentativo del 2026-09-24: addestramento circa 12 minuti per round sul Mac,
ma FedMIA e Shadow saltavano ogni round per l'architettura sbagliata negli attacchi
(segnalazione 49, corretta; log in `logs/prova_recorddp_s42_attacchi_saltati.log`).
La run non si era fermata: è arrivata in fondo alle 14:47 e ha salvato un JSON in
`experiments/_prova_recorddp/`, con gli attacchi saltati, da non usare; negli ultimi 25
minuti si è sovrapposta ai controlli (a) e (b), e (a) riproduce comunque alla sesta cifra.
Durata misurata sul Mac principale: 3 ore e 55 minuti (addestramento circa 12 minuti per
round, 2 ore e 4 minuti; Shadow 5 minuti; LiRA circa 10 minuti per round). Da rifare in
una cartella nuova. Gli shadow non usano DP-SGD (segnalazione 51): LiRA sotto
record-level va letto con quella riserva.

```bash
python3 scripts/run_experiments.py \
  --config config/experiment_rq1_recorddp_nm1.yaml --rounds 10 --seed 42 --no-dp \
  --sweep-dir experiments/_prova_recorddp_v2 \
  --per-sample-dump experiments/_prova_recorddp_v2/per_sample_seed42.json
```

Poi, se il tempo lo consente e dopo le correzioni 4, 5, 37:

```bash
caffeinate -ims bash -c '
set -e
for nm in 0.5 1 2; do
  for s in 42 123 456 789 1234; do
    python3 scripts/run_experiments.py \
      --config config/experiment_rq1_recorddp_nm$nm.yaml --rounds 10 --seed $s --no-dp \
      --sweep-dir experiments/rq1-recorddp-nm$nm \
      --per-sample-dump experiments/rq1-recorddp-nm$nm/per_sample_seed$s.json
  done
done' 2>&1 | tee logs/rq1_recorddp.log
```

**Lettura.** Il budget della cella è `epsilon_record_dp`, non `epsilon`: è il massimo
sui client dell'ε di Poisson (`epsilon_record_dp_per_client`), di solito Office 1, che
ha pochi record. È un'approssimazione, perché il training usa shuffle con batch fissi:
riportarlo sempre insieme a `epsilon_record_dp_shuffle_bound` (adiacenza per
sostituzione, senza amplificazione), ciascuno con la propria adiacenza. Il confronto
è con la cella client-level allo stesso seed e a costo comparabile, non allo stesso ε:
sono unità diverse.

## E-C — analisi per record su tutte le celle nuove. Solo calcolo

**Stato (2026-09-24).** Eseguito sulle 8 celle esistenti (no-DP, dp-fedavg ε = 16, 8,
4, 2, 1, central e local ε = 1) con il test corretto della segnalazione 47: nessun
segnale di appartenenza per record, numeri in `STATO.md` 3.2. Le celle record-DP si
aggiungono dopo E-B. La segnalazione 6 riguarda ora solo il conteggio secondario.

```bash
python3 scripts/worst_case_livello_di_caso.py \
  --gruppo "no-DP=experiments/nodp-sweep2" \
  $(for e in 2 4 8 16; do echo --gruppo "eps$e=experiments/rq1-eps$e"; done) \
  $(for nm in 0.5 1 2; do echo --gruppo "recordDP_nm$nm=experiments/rq1-recorddp-nm$nm"; done) \
  --permutazioni 200 --output risultati/worst_case/livello_di_caso.json
python3 scripts/genera_matrici_faseA.py
```

Salvare anche il report grezzo di ogni gruppo con
`scripts/analyze_worst_case_vulnerability.py --output risultati/worst_case/<gruppo>.json`.

## E-E — RQ3, mu = 0 contro 0.01

**Fase D della guida, seconda priorità.** Il braccio senza DP si può lanciare
subito, perché riusa la configurazione ordinaria: serve un config
`experiment_rq3_mu0.yaml` identico a `config/experiment.yaml` con
`ml.proximal_mu: 0.0` (il campo vive in `cfg["ml"]`: messo altrove viene ignorato
in silenzio), 5 seed con `--no-dp`, confrontato con `nodp-sweep2`. Il braccio con
DP attende il punto operativo indicato da E-A. Nessuna ipotesi che FedProx sia più
privato; riportare insieme attacco e costo.

**Stato (2026-09-24).** `config/experiment_rq3_mu0.yaml` creato: differisce da
`config/experiment.yaml` solo per `ml.proximal_mu: 0.0` e per il nome. Il braccio
no-DP gira su una terza macchina (Windows, i9, commit 478d471): mu = 0 in
`experiments/rq3-mu0` e un braccio appaiato mu = 0.01 sulla stessa macchina in
`experiments/rq3-mu0.01` (`config/experiment.yaml`), perche' `nodp-sweep2` e' di
un'altra macchina e di un commit dell'8 settembre. Primo tentativo (08:39) fermato
dopo l'addestramento della prima run, nessun JSON: con i thread di default torch era
circa 8 volte piu' lento del Mac (`ENVIRONMENT.md` sezione 9). Rilanciato alle 10:59
con `OMP_NUM_THREADS=1`, **i due bracci in parallelo**, uno per finestra PowerShell,
log `logs/rq3_mu0.log` e `logs/rq3_mu0.01.log`. Il parallelo deroga alla regola "una
run alla volta" di `CLAUDE.md`: da confermare o da riportare a un braccio alla volta.
Il round 1 e' identico nei due bracci (il termine prossimale entra dal round 2),
controllo che i bracci differiscono solo per mu. Blocco di lancio, per ciascun
braccio (`$n = "mu0"; $c = "config\experiment_rq3_mu0.yaml"` oppure
`$n = "mu0.01"; $c = "config\experiment.yaml"`, con `PYTHONUTF8`,
`OMP_NUM_THREADS` e `MKL_NUM_THREADS` impostati nella stessa finestra):

```powershell
& {
  Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
  [W.P]::SetThreadExecutionState(2147483649) | Out-Null
  $dir = "experiments\rq3-$n"
  foreach ($s in 42,123,456,789,1234) {
    cmd /c ".venv\Scripts\python.exe scripts\run_experiments.py --config $c --rounds 10 --seed $s --no-dp --sweep-dir $dir --per-sample-dump $dir\per_sample_seed$s.json >> logs\rq3_$n.log 2>&1"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$dir\per_sample_seed$s.json")) { return }
  }
}
```

Al rientro delle cartelle vale la segnalazione 45: tenerle fuori da `experiments/`
finche' non e' corretta.

## E-D — RQ2, partizione IID contro per sito

**Fase E della guida, terza priorità.** Config appaiati pronti, differiscono per il
solo campo `partition.strategy`.

```bash
for strat in per_site iid; do
  for s in 42 123 456 789 1234; do
    python3 scripts/run_experiments.py \
      --config config/experiment_rq2_$strat.yaml --rounds 10 --seed $s --no-dp \
      --sweep-dir experiments/rq2-$strat \
      --per-sample-dump experiments/rq2-$strat/per_sample_seed$s.json
  done
done 2>&1 | tee logs/rq2_partizione.log
```

Poi la stessa coppia con la configurazione DP indicata da E-A. La partizione IID è
un riferimento sperimentale, non un deployment: la ground truth di appartenenza
per client cambia e va ricostruita.

**Stato (2026-09-24).** Braccio no-DP eseguito dal 22 al 23 settembre su una seconda
macchina (Mac, Python 3.13; E-A girava sulla prima con Python 3.14): 5 JSON e 5 dump
per campione in ciascuna di `experiments/rq2-per_site` e `experiments/rq2-iid`, zero
righe `[ERROR]` nel log. Prima di leggerlo vanno copiate le due cartelle e
`logs/rq2_partizione.log` in questo checkout, poi `check_significance.py` e
`genera_matrici_faseA.py`. La macchina diversa è una variabile non registrata nei JSON:
va dichiarata nel paper. Attenzione alla segnalazione 45 prima di copiarle in
`experiments/`.

## Controllo dell'inizializzazione comune (segnalazione 48)

**Validità della campagna, non una RQ.** Nelle run esistenti i tre client partono da
inizializzazioni diverse. Due passi sul Mac, uno alla volta.

(a) Riproduzione, solo addestramento, circa 15 minuti: stesso config di `nodp-sweep2`,
codice attuale. Si ferma a `Round 10 — loss globale` e si confronta round per round con
`per_round[r].fl.mean_loss` del JSON di `nodp-sweep2` seed 42. Il log contiene anche le
norme dei delta per client (`[NORMA DELTA]`), che servono alla sezione successiva.

```bash
nohup python3 scripts/run_experiments.py --config config/experiment.yaml \
  --rounds 10 --seed 42 --no-dp --sweep-dir experiments/_ctrl_riproduzione_s42 \
  > logs/ctrl_riproduzione_s42.log 2>&1 &
```

(b) Se (a) riproduce, la cella di controllo completa, circa 2 ore:

```bash
nohup python3 scripts/run_experiments.py --config config/experiment_ctrl_common_init.yaml \
  --rounds 10 --seed 42 --no-dp --sweep-dir experiments/_ctrl_common_init \
  --per-sample-dump experiments/_ctrl_common_init/per_sample_seed42.json \
  > logs/ctrl_common_init_s42.log 2>&1 &
```

**Lettura.** Confronto con `nodp-sweep2` seed 42 su loss sull'holdout del modello
rilasciato, Yeom, LiRA composto e test appaiato. Differenza dentro la variabilità fra
seed: le campagne restano valide e lo scostamento si dichiara. Altrimenti decisione col
supervisore.

**Esito, 2026-09-24.** (a) riproduce `nodp-sweep2` seed 42: la loss di tutti i 10 round
coincide alla sesta cifra (0.001200 al round 1, 0.002177 al round 10), con numpy 1.26.4
(segnalazione 52). (b), `experiments/_ctrl_common_init`: holdout del modello rilasciato
0.0080 contro 0.0651 al round 1, 0.00108 contro 0.00219 al round 10, dentro la
variabilita' fra seed di `nodp-sweep2` (0.00070-0.00219); Yeom 0.4989, Shadow medio
0.4997, LiRA composto 0.4991, TPR a FPR 1% 0.0099, tutti nel campo dei 5 seed senza
inizializzazione comune. Il test appaiato per record non si applica a un seed solo.
Lettura: le campagne restano valide, lo scostamento si dichiara; il protocollo delle
campagne future e' una decisione del supervisore. La griglia qui sotto resta sul
protocollo di E-A per restare confrontabile.

## Punto operativo client-level (dopo E-A)

**RQ1.** Nessuna cella client-level di E-A sta sotto 3 volte. Cambiare posizionamento
non basta: central aggiunge circa 0.50σ sul modello globale contro 0.69σ di dp-fedavg e
local, un fattore 1.4 contro un eccesso di 60 volte. Il limite è strutturale: tre client,
uno con metà dei dati, e rumore per client che la media non ammortizza. Due leve, da
scegliere dopo la misura delle norme del passo (a) sopra:

- **C più basso a parità di ε** (`experiment.max_grad_norm`): il rumore è proporzionale
  a C; utile se le norme reali dei delta stanno molto sotto 1.
- **ε più alti** (per esempio 64 e 256): estrapolando E-A la soglia cade fra circa 100 e
  500 per round; è un livello di rumore, non una garanzia (la calibrazione gaussiana vale
  per ε < 1).

Prima un seed per valore per trovare il ginocchio, poi 5 seed sul punto scelto; un config
nuovo per cella. Nel regime naturale non c'è segnale da ridurre neppure senza DP: il punto
operativo dice quanto costa un rumore che lascia il modello utile, l'effetto sulla privacy
si misura solo dove c'è segnale (canary su più siti, sotto).

**Norme misurate** (controllo (a), `logs/ctrl_riproduzione_s42.log`, no-DP, C = 1). Round 1:
8.84, 8.47, 4.64 (caltech, jpl, office1; ogni client parte dalla sua inizializzazione).
Round 2: 0.78-0.92. Round 3-10: caltech 0.16-0.29, jpl 0.16-0.21, office1 0.51-0.59. Con
C = 1 il clipping non agisce dopo il round 1. Due riserve: sotto DP le norme saranno
piu' grandi (i client correggono il rumore), e con C piccolo il modello si sposta al
massimo di C per round. Le run nuove scrivono le norme nel log e nel JSON.

**Griglia, decisa il 2026-09-24.** C in {0.25, 0.5, 1} per ε in {16, 64, 256}, un seed
(42), `dp-fedavg`; 8 run nuove (C = 1, ε = 16 è E-A), circa 130 minuti ciascuna. A
parita' di C/ε il rumore e' lo stesso: C = 0.25 con ε = 16 contro C = 1 con ε = 64, e
C = 0.25 con ε = 64 contro C = 1 con ε = 256, separano rumore e distorsione del
clipping. Cartelle `_op_*`: prove a un seed, escluse dalle celle delle matrici.
Ordine: prima le due coppie a pari rumore. Config: `experiment_rq1_C{0.25,0.5}_eps*.yaml`
ed `experiment_rq1_eps{64,256}.yaml`.

```bash
nohup caffeinate -ims bash -c '
for c in C0.25_eps16 eps64 C0.25_eps64 eps256 C0.5_eps16 C0.5_eps64 C0.25_eps256 C0.5_eps256; do
  python3 scripts/run_experiments.py --config config/experiment_rq1_$c.yaml \
    --rounds 10 --seed 42 --sweep-dir experiments/_op_$c > logs/op_$c.log 2>&1
done' > /dev/null 2>&1 & disown
```

**Lettura.** Per ogni run: holdout del modello rilasciato contro `nodp-sweep2` (media
0.00158), norme dei delta sotto DP, attacchi. Il ginocchio e' il primo punto sotto 3
volte; li' 5 seed, in cartelle senza `_`, che formano una cella propria con C
nell'etichetta (segnalazione 53, corretta).

## Canary su più siti

**Validazione dello strumento in federazione, prerequisito per leggere la DP
client-level.** Tutte le run canary finora hanno un solo client (Office 1), dove la DP
client-level non ha senso. Proposta: i tre siti reali, canary iniettati solo in Office 1
con il protocollo bilanciato già validato (k = 20, 30 duplicati, `paired_split`, braccio
scambiato, 5 seed, regime canary). Superficie primaria l'update di Office 1: nel modello
globale Office 1 pesa circa il 3.6%, e la diluizione va misurata. Condizioni: senza DP,
client-level al punto operativo, record-level. Serve un config nuovo, mai eseguito in
questa combinazione: prima una run di prova a un seed per verificare che iniezione e
punteggi funzionino con tre client e per misurare il tempo (1000 epoche su tre siti).
Da definire col supervisore: numero di epoche, LiRA completo o solo loss grezza.

## Canary bilanciato su un secondo sito

**Validazione dello strumento, non una RQ.** Il protocollo che regge (k = 20,
scambio dei ruoli, baseline a init casuale, 5 seed) esiste solo su Office 1. Su
Caltech e JPL ci sono solo run a seed 42 senza blocco canary nel JSON.

Serve un config `experiment_canary_balanced_caltech.yaml`, oggi assente, derivato
da `experiment_canary_balanced.yaml` con `site: caltech` e amplificazione per
record pari a Office 1: `n_duplicates` circa 561 su un pool di 25 123 sessioni,
cioè 20 template per 561 copie, 11 220 record. Tempo atteso ben superiore a
Office 1: misurare prima con un seed. Procedura identica alla sezione 10 di
`CanaryPositiveControl.md`, bracci A e B con lo stesso seed, baseline con
`check_canary_init_confound.py --seed`.

## Rianalisi NVFlare

I 25 dump a ε ≤ 1 (`experiments/_dump_nvflare_provenienza.csv`) sono nel regime
con utility distrutta e confermerebbero il nullo già noto in simulazione. Si
rianalizzano dopo E-A, solo 2-3 celle al punto operativo indicato, e per quelle
servono prima job NVFlare nuovi a quell'ε. Lo script `analizza_dump_nvflare.sh`
passa già lo snapshot del seed; senza, l'AUC tende a 0.5 per costruzione.

## Rinviato o escluso

RQ4 (rilevamento della MIA passiva) esclusa per threat model. ByzantineDetector
end-to-end, secondo dataset, adaptive clipping: lavoro futuro dichiarato. Le
indagini su LiRA (floor per record, scoring simmetrico) si fanno sulla cella
canary bilanciata a Office 1, dove la verità è nota, e non toccano la pipeline
finché E-A ed E-B non sono chiusi.
