# ChargeShield-FL — Esperimenti da eseguire

> **Stato: documento CANONICO.** Aggiornato il 2026-09-30 (revisione: costo, E-C, controlli (a) e (b) eseguiti, griglia del punto operativo, canary su più siti, E-D con DP, campagna E-B, braccio con DP di E-E; il 2026-09-30 esito dello screening di C e del solo clipping, config e run di prova del canary su più siti). Sostituisce,
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
| B1 | manca un flag clip-only per il client-level (per il record-level basta σ = 0, eseguito il 2026-09-30); va aggiunto o il braccio client-level va dichiarato non eseguito | 38 |
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

**Stato (2026-09-29).** Per il record-level il braccio non richiede un flag:
`noise_multiplier: 0` con `record_dp` attivo fa solo clipping per esempio (trainer,
`src/ml/autoencoder_trainer.py` righe 300-301: rumore solo se σ > 0; accountant,
`src/ml/record_dp_accounting.py` righe 156-171: ε `None` con la nota "nessuna garanzia").
Cella `config/experiment_rq1_recorddp_nm0.yaml` (commit 8dceb39) completa il 2026-09-30 a 5
seed sulla quarta macchina: 1.04 volte il riferimento GroupNorm, quindi il costo della DP per
record viene dal rumore (sezione E-B, `STATO.md` 3.5). Per il client-level il flag manca ancora (segnalazione 38).

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

**Esito della prova, 2026-09-24** (`_prova_recorddp_v2`, secondo Mac, commit 2c4a0f0, σ = 1,
seed 42; JSON in `experiments_altre_macchine/_prova_recorddp_v2`, valori riverificati sul
JSON il 2026-09-26). Tutti gli attacchi girano. Loss sull'holdout del
modello rilasciato 0.00240: 1.29 volte il braccio per sito di E-D allo stesso seed e sulla
stessa macchina, 1.51 volte la media di `nodp-sweep2`. Attacchi al caso: Yeom 0.498, Shadow
medio 0.497, LiRA composto 0.504, TPR a FPR 1% 0.0104. ε per record, Poisson: 30.2 per
Office 1 (1341 record), 5.0 Caltech, 4.8 JPL; limite valido con lo shuffle 1212. Durata
sul secondo Mac: 3 ore e 24 minuti.

Due conseguenze per la campagna. (1) A σ = 1 il costo è quasi nullo e l'ε di Office 1 è
già 30: σ = 0.5 aggiungerebbe poco, σ = 2 e 5 descrivono meglio il compromesso. Proposta:
σ ∈ {1, 2, 5}, config già presenti. (2) La cella record-DP cambia anche l'architettura
(GroupNorm al posto di BatchNorm, clipping per esempio): il rapporto sul no-DP mescola
l'effetto del rumore con quello della normalizzazione. Per separarli serve una cella
no-DP con `ml.norm: group`, oggi assente. Entrambe le scelte sono da confermare.

**Stato della campagna (2026-09-27).** σ = 1 completa sul Mac principale, 5 seed in
`experiments/rq1-recorddp-nm1` (log `logs/rq1_recorddp_nm1.log`, circa 3 ore e 20 minuti per
seed, ultimo salvataggio il 27 settembre alle 01:12, zero righe `[ERROR]`). Loss sull'holdout
1.4 volte la media di `nodp-sweep2`, sotto la soglia; attacchi al caso; test per record senza
segnale (z = 0.18); ε per record di Poisson circa 30 a Office 1, limite con lo shuffle 1212.
Numeri in `STATO.md` 3.5. Il seed 456 ha `-dirty` per documenti modificati durante la run
(segnalazione 57): le modifiche al repository vanno fatte con almeno un'ora di margine sul
salvataggio stimato.

**σ = 2 e cella no-DP con GroupNorm, completi il 2026-09-28** (Mac principale, 5 seed ciascuna,
`experiments/rq1-recorddp-nm2` e `rq1-nodp-groupnorm`). GroupNorm senza DP non costa (0.73
volte BatchNorm, non significativo): il costo della DP per record si legge su quel riferimento
ed è 1.87 volte a σ = 1 e 2.10 a σ = 2, sotto la soglia; da σ = 1 a 2 cresce di 1.12 volte
mentre l'ε per record scende da 30 a 9.7. Attacchi al caso, test per record senza segnale
(`STATO.md` 3.5). Da decidere: σ = 5, che dice quanto pesa il rumore, e una cella con clipping
per esempio e σ = 0, che isolerebbe il clipping (richiede di verificare che il codice accetti
`noise_multiplier: 0` con `record_dp` attivo).

Coda sul Mac principale, una run alla volta: σ = 2 parte da sola quando finisce il ciclo in
corso. Prima si legge il PID del ciclo (deve uscire un solo numero), poi si lancia l'attesa,
che controlla ogni 5 minuti se quel processo esiste ancora. Con la macchina libera basta il
ciclo `for`, senza l'attesa:

```bash
pgrep -f "caffeinate -ims bash"
P=<PID letto sopra> nohup caffeinate -ims bash -c '
while kill -0 $P 2>/dev/null; do sleep 300; done
set -e
for s in 42 123 456 789 1234; do
  python3 scripts/run_experiments.py --config config/experiment_rq1_recorddp_nm2.yaml \
    --rounds 10 --seed $s --no-dp --sweep-dir experiments/rq1-recorddp-nm2 \
    --per-sample-dump experiments/rq1-recorddp-nm2/per_sample_seed$s.json
done' > logs/rq1_recorddp_nm2.log 2>&1 & disown
```

Il `git_commit` si legge al salvataggio: l'albero va tenuto pulito anche durante la coda. σ = 2 lanciato il 2026-09-27 alle 09:32, con la macchina libera.

**σ = 5, completo il 2026-09-29** (quarta macchina, Mac di Domenico, 5 seed, commit 64757fd,
circa 3 ore e mezza per seed; copiato in `experiments_altre_macchine/rq1-recorddp-nm5`, con il
controllo `_ctrl_recorddp_nm1_s42_r2`). 2.32 volte il riferimento GroupNorm, ε per record di
Poisson 3.1 (shuffle 81), attacchi al caso: numeri in `STATO.md` 3.5.

**Solo clipping (σ = 0), lanciato il 2026-09-29** sulla quarta macchina, 5 seed, una run alla
volta, dopo `git fetch susetta && git merge --ff-only susetta/master` (commit 8dceb39):

```bash
cd ~/ChargeShield-FL && source .venv/bin/activate && nohup caffeinate -ims bash -c 'for s in 42 123 456 789 1234; do python3 scripts/run_experiments.py --config config/experiment_rq1_recorddp_nm0.yaml --rounds 10 --seed $s --no-dp --sweep-dir experiments/rq1-recorddp-nm0 --per-sample-dump experiments/rq1-recorddp-nm0/per_sample_seed$s.json; done' >> logs/rq1_recorddp_nm0.log 2>&1 &
```

Lanciato dal terminale di VS Code, il ciclo ha perso lo standard input quando VS Code e' stato
chiuso: il seed 42 era gia' in corso e ha salvato alle 12:35, i seed 123-1234 sono falliti
all'avvio nello stesso secondo (`Fatal Python error: init_sys_streams`, `Errno 9`, segnalazione
62). Rilancio del 2026-09-29, dall'app Terminale, con lo standard input da `/dev/null` e i seed
gia' salvati saltati:

```bash
cd ~/ChargeShield-FL && source .venv/bin/activate && echo "=== rilancio $(date) ===" >> logs/rq1_recorddp_nm0.log && nohup caffeinate -ims bash -c 'for s in 42 123 456 789 1234; do [ -e experiments/rq1-recorddp-nm0/per_sample_seed$s.json ] && continue; python3 scripts/run_experiments.py --config config/experiment_rq1_recorddp_nm0.yaml --rounds 10 --seed $s --no-dp --sweep-dir experiments/rq1-recorddp-nm0 --per-sample-dump experiments/rq1-recorddp-nm0/per_sample_seed$s.json; done' < /dev/null >> logs/rq1_recorddp_nm0.log 2>&1 &
disown
```

Con σ = 0 l'analisi IDS a fine run segnala GRADIENT_EXPLOSION, perché la soglia scende a C: è
un'analisi a posteriori, non tocca addestramento né attacchi. Il JSON ha `epsilon_record_dp`
`None`: la cella è diagnostica, non di privacy. Lettura: il costo sul riferimento GroupNorm
separa il contributo del clipping da quello del rumore (σ = 1, 2, 5).

**Esito, 2026-09-30** (5 seed, 0 errori, importati in `experiments_altre_macchine/rq1-recorddp-nm0`
con provenienza): 1.04 volte il riferimento GroupNorm (da 0.72 a 1.40, t = 0.35); rispetto al
solo clipping il rumore costa 1.80 volte a σ = 1. Numeri in `STATO.md` 3.5.

**Cella no-DP con GroupNorm, preparata il 2026-09-27.** Riferimento per leggere il costo delle
celle record-DP: `config/experiment_rq1_nodp_groupnorm.yaml` differisce da `experiment.yaml`
solo per `ml.norm: group`. Costo della DP per record = holdout record-DP / holdout di questa
cella, stesso seed; costo della normalizzazione = questa cella / `nodp-sweep2`. Etichetta
"no-DP baseline, norm=group": `scripts/etichetta_cella.py` registra ora `norm` fra i campi
del trattamento, altrimenti la cella avrebbe sostituito i seed di `nodp-sweep2` (stessa classe
della segnalazione 45). Sul Mac principale, dopo σ = 2, circa 2 ore per seed come
`nodp-sweep2`:

```bash
nohup caffeinate -ims bash -c '
set -e
for s in 42 123 456 789 1234; do
  python3 scripts/run_experiments.py --config config/experiment_rq1_nodp_groupnorm.yaml \
    --rounds 10 --seed $s --no-dp --sweep-dir experiments/rq1-nodp-groupnorm \
    --per-sample-dump experiments/rq1-nodp-groupnorm/per_sample_seed$s.json
done' > logs/rq1_nodp_groupnorm.log 2>&1 & disown
```

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
segnale di appartenenza per record, numeri in `STATO.md` 3.2. Aggiunti poi ε = 64 e i
quattro bracci di E-D (il 2026-09-26 i due con DP, con lo stesso comando ristretto ai due
gruppi e unito al file: il test è deterministico per gruppo) e, il 2026-09-27, la cella
record-DP a σ = 1 (`recordDP_nm1`), allo stesso modo. La segnalazione 6 riguarda ora solo il conteggio secondario.

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
DP usa il punto operativo scelto dopo E-A (sotto). Nessuna ipotesi che FedProx sia più
privato; riportare insieme attacco e costo.

**Braccio con DP, preparato il 2026-09-26.** Al punto operativo client-level scelto prima
di vedere RQ3 con DP, lo stesso di E-D: dp-fedavg, ε = 64, C = 1. Config
`experiment_rq3_mu0_eps64.yaml` (FedAvg, differisce da `experiment_rq1_eps64.yaml` solo per
`ml.proximal_mu: 0.0`) ed `experiment_rq1_eps64.yaml` (FedProx), 10 run senza `--no-dp`,
sulla stessa macchina dei due bracci senza DP (segnalazione 55), un braccio alla volta.
Poi le cartelle vanno in `experiments_altre_macchine/` come quelle di E-D. Blocco per
Windows, con `PYTHONUTF8`, `OMP_NUM_THREADS` e `MKL_NUM_THREADS` impostati nella finestra:

```powershell
& {
  Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
  [W.P]::SetThreadExecutionState(2147483649) | Out-Null
  foreach ($b in @(@("mu0", "config\experiment_rq3_mu0_eps64.yaml"), @("mu0.01", "config\experiment_rq1_eps64.yaml"))) {
    $n = $b[0]; $c = $b[1]; $dir = "experiments\rq3-$n-eps64"
    foreach ($s in 42,123,456,789,1234) {
      cmd /c ".venv\Scripts\python.exe scripts\run_experiments.py --config $c --rounds 10 --seed $s --sweep-dir $dir --per-sample-dump $dir\per_sample_seed$s.json >> logs\rq3_${n}_eps64.log 2>&1"
      if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$dir\per_sample_seed$s.json")) { return }
    }
  }
}
```

**Lettura.** Come E-D: le due coppie appaiate per seed, attacchi e costo sul modello
rilasciato, e il costo della DP per algoritmo come rapporto con DP / senza DP allo stesso
seed e sulla stessa macchina.

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

**Cambio di piano, 2026-09-28: E-E completo sul Mac principale.** Su Windows una run con DP dura
5-6 ore, quasi tutte di attacchi, e le prove su mu sarebbero partite dopo 9 run. Sul Mac
principale E-E ha gia' meta' dei bracci sulla stessa macchina: FedProx senza DP e' `nodp-sweep2`
(riprodotto alla sesta cifra dal codice attuale, controllo (a)), FedProx a eps = 64 e'
`rq1-eps64`. Mancano i due bracci FedAvg, in coda dietro la cella GroupNorm dal 28 settembre
alle 10:03 (log `logs/rq3_mac.log`), una run alla volta: FedAvg senza DP seed 42 (riferimento
delle prove e primo seed del braccio), prove su mu (0.001 e 0.1 a 10 round, 0.01 a 30 round,
seed 42, cartelle `_rq3_*`), FedAvg senza DP seed 123-1234 (`experiments/rq3-mu0`), FedAvg a
eps = 64, 5 seed (`experiments/rq3-mu0-eps64`). Le etichette di cella ("no-DP baseline,
mu=0.0", "dp-fedavg, eps=64.0, mu=0.0") non si sovrappongono a quelle di RQ1. Windows continua
E-E con DP come replica su un'altra macchina; il blocco delle prove su mu, se attivo, la
estende dopo i 10 JSON.

```bash
P=<PID del ciclo precedente> nohup caffeinate -ims bash -c '
while kill -0 $P 2>/dev/null; do sleep 300; done
set -e
corri() { python3 scripts/run_experiments.py --config config/$1 --rounds $2 --seed $3 $4 \
  --sweep-dir experiments/$5 --per-sample-dump experiments/$5/per_sample_seed$3.json; }
corri experiment_rq3_mu0.yaml 10 42 --no-dp rq3-mu0
corri experiment_rq3_mu0.001.yaml 10 42 --no-dp _rq3_mu0.001_s42
corri experiment_rq3_mu0.1.yaml 10 42 --no-dp _rq3_mu0.1_s42
corri experiment_rq3_mu0.01_r30.yaml 30 42 --no-dp _rq3_mu0.01_r30_s42
for s in 123 456 789 1234; do corri experiment_rq3_mu0.yaml 10 $s --no-dp rq3-mu0; done
for s in 42 123 456 789 1234; do corri experiment_rq3_mu0_eps64.yaml 10 $s "" rq3-mu0-eps64; done
' > logs/rq3_mac.log 2>&1 & disown
```

**Esito dei bracci senza DP (2026-09-27).** 10 run completate il 25 settembre, copiate in
`experiments_altre_macchine/rq3-mu0` e `rq3-mu0.01`, zero errori. FedAvg ha la loss
sull'holdout 3.9 volte piu' bassa di FedProx (5 seed su 5), attacchi al caso in entrambi
(`STATO.md` 3.10). Braccio con DP lanciato su Windows il 27 settembre alle 16:20.

**Prove su mu, decise il 2026-09-27.** Verificano l'ipotesi che con mu = 0.01 il termine
prossimale limiti lo spostamento per round e dieci round non bastino. Seed 42, senza DP, sulla
macchina del riferimento FedAvg (Windows), dopo il braccio con DP, una run alla volta:

| prova | config | round | cosa dice |
|---|---|---|---|
| FedAvg, ripetuta | `experiment_rq3_mu0.yaml` | 10 | norme degli update di FedAvg, che la run del 24 non registra; riproduzione del codice attuale (478d471 contro l'attuale: il percorso di training non cambia) |
| mu = 0.001 | `experiment_rq3_mu0.001.yaml` | 10 | dose-risposta |
| mu = 0.1 | `experiment_rq3_mu0.1.yaml` | 10 | dose-risposta |
| mu = 0.01, 30 round | `experiment_rq3_mu0.01_r30.yaml` | 30 | se FedProx arriva vicino a FedAvg e' solo piu' lento, non converge a un modello peggiore |

Lettura: loss sull'holdout per round e norme degli update per round. Se la loss al round 10
cresce con mu e le norme calano con mu, e se a 30 round FedProx si avvicina a FedAvg, l'ipotesi
regge. Round 1 identico in tutte le run: controllo che cambia solo mu o il numero di round.
Cartelle `_rq3_*`: prove a un seed, fuori dalle celle. In una seconda finestra PowerShell, con
le stesse variabili d'ambiente del braccio con DP; parte da sola quando il braccio con DP ha
salvato i suoi 10 JSON:

```powershell
$env:PYTHONUTF8 = "1"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
& {
  Add-Type -Namespace W2 -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
  [W2.P]::SetThreadExecutionState(2147483649) | Out-Null
  while (@(Select-String -Path logs\rq3_mu0.01_eps64.log -Pattern "Esperimento completato" -ErrorAction SilentlyContinue).Count -lt 5) { Start-Sleep 300 }
  $prove = @(
    @("mu0_s42_v2",     "config\experiment_rq3_mu0.yaml",        10),
    @("mu0.001_s42",    "config\experiment_rq3_mu0.001.yaml",    10),
    @("mu0.1_s42",      "config\experiment_rq3_mu0.1.yaml",      10),
    @("mu0.01_r30_s42", "config\experiment_rq3_mu0.01_r30.yaml", 30))
  foreach ($p in $prove) {
    $n = $p[0]; $c = $p[1]; $r = $p[2]; $dir = "experiments\_rq3_$n"
    cmd /c ".venv\Scripts\python.exe scripts\run_experiments.py --config $c --rounds $r --seed 42 --no-dp --sweep-dir $dir --per-sample-dump $dir\per_sample_seed42.json >> logs\rq3_prove_mu.log 2>&1"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$dir\per_sample_seed42.json")) { return }
  }
}
```

**Esito sul Mac principale, 2026-09-29.** La coda è finita alle 15:44 del 29 settembre: 13 run,
zero errori, commit 64757fd e 8dceb39. FedAvg senza DP a 5 seed, prove su mu, run a 30 round,
FedAvg con DP a 5 seed: numeri in `STATO.md` 3.10.

**Screening di C per FedAvg con DP, lanciato il 2026-09-29 alle 16:28** (Mac principale,
commit 7a21bb0). Motivo: C = 1 è stato scelto sulla griglia di FedProx (C in {0.25, 0.5, 1});
con DP gli update di FedAvg hanno norma 2.5-8.5 e C = 1 li taglia a ogni round, quelli di
FedProx no. Config `experiment_rq3_mu0_eps64_C{2,4,8}.yaml`, solo seed 42, circa 2 ore per run:

```bash
cd ~/Documents/ChargeShield-FL && nohup caffeinate -ims bash -c '
set -e
for C in 2 4 8; do
  d=_rq3_mu0_eps64_C${C}_s42
  python3 scripts/run_experiments.py --config config/experiment_rq3_mu0_eps64_C$C.yaml --rounds 10 --seed 42 --sweep-dir experiments/$d --per-sample-dump experiments/$d/per_sample_seed42.json
done
' > logs/rq3_screening_C.log 2>&1 & disown
```

Lanciato senza `< /dev/null`: la finestra del Terminale da cui e' partito va lasciata aperta
fino all'avvio dell'ultima run (segnalazione 62).

Lettura: loss sull'holdout contro C = 1 (0.01298 al seed 42). Il rumore è calibrato su C,
quindi a parità di ε un C più grande porta più rumore. Se un C batte C = 1, 5 seed a quel C e,
per simmetria, FedProx allo stesso C; la scelta sull'holdout è ottimistica come per FedProx e
va dichiarata.

**Esito, 2026-09-30** (dai JSON, numeri in `STATO.md` 3.10): nessun C batte C = 1 (C = 2: 1.01
volte, C = 4: 3.8, C = 8: 18.7, seed 42). Per la regola sopra niente 5 seed a un altro C né
FedProx allo stesso C: lo screening è chiuso e il braccio con DP di E-E resta a C = 1.

**Windows: standby e `-dirty` (2026-09-29).** (1) Con lo schermo che si spegne per inattività
il portatile entra in Modern Standby (S0 low power idle) e mette in pausa i processi, anche in
carica e con il coperchio aperto: il 28 settembre dalle 17:01 alle 20:38 (eventi Kernel-Power
506/507). `SetThreadExecutionState(2147483649)` dei blocchi sopra tiene sveglio il sistema ma
non lo schermo: nei prossimi lanci usare 2147483651 (anche lo schermo), oppure un ciclo in una
finestra separata con `SetThreadExecutionState(3)` ogni 30 secondi. La pausa non cambia i
risultati, solo i tempi. (2) Il clone Windows aveva due voci non tracciate (`logs/` con file
diversi da `*.log` e uno zip nella radice), che marcano le run `-dirty`; escluse con
`.git/info/exclude`, locale e non versionato (segnalazione 61).

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
righe `[ERROR]` nel log. **Analizzato il 2026-09-24**: le due cartelle e il log stanno in
`experiments_altre_macchine/` (non versionata; non in `experiments/`, segnalazione 45),
`genera_matrici_faseA.py` ne fa il foglio `RQ2_partizione`, il test per record è in
`livello_di_caso.json`. Esito in `STATO.md` 3.9: attacchi al caso in entrambi i bracci,
nessuna differenza significativa, loss sull'holdout più bassa con IID ma non
significativa. La macchina diversa è una variabile non registrata nei JSON e non è
trascurabile (segnalazione 55): il confronto usa i due bracci della stessa macchina.

**Braccio con DP, eseguito e analizzato il 2026-09-26.** 10 run sul secondo Mac dal 25 al
26 settembre (commit 3d4d318, zero righe `[ERROR]`), copiate in
`experiments_altre_macchine/rq2-per_site-eps64` e `rq2-iid-eps64`; foglio `RQ2_partizione`
(blocchi "CON DP" e "COSTO DELLA DP PER PARTIZIONE") e test per record in
`livello_di_caso.json`. Esito in `STATO.md` 3.9: attacchi al caso in entrambi i bracci,
nessun segnale per record, costo della DP più alto con IID (media geometrica 6.1 contro
3.9) ma interazione non significativa a 5 seed. Il disegno, fissato prima: il punto
operativo client-level scelto con la regola della griglia prima di guardare RQ2 con DP,
C = 1, ε = 64 (2.6 volte il no-DP, un seed). Config
`experiment_rq2_per_site_eps64.yaml` ed `experiment_rq2_iid_eps64.yaml`, 10 run, sulla
seconda macchina come il braccio senza DP (segnalazione 55); poi in
`experiments_altre_macchine/`. Da aggiungere: le statistiche delle feature per client
nelle due partizioni.

```bash
nohup caffeinate -ims bash -c '
set -e
for strat in per_site iid; do
  for s in 42 123 456 789 1234; do
    python3 scripts/run_experiments.py --config config/experiment_rq2_${strat}_eps64.yaml \
      --rounds 10 --seed $s --sweep-dir experiments/rq2-${strat}-eps64 \
      --per-sample-dump experiments/rq2-${strat}-eps64/per_sample_seed$s.json
  done
done' > logs/rq2_partizione_eps64.log 2>&1 & disown
```

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

**Esito, griglia completa il 2026-09-25 (seed 42, cartelle `_op_*`).** Loss sull'holdout del
modello rilasciato in rapporto alla media di `nodp-sweep2` (0.00158); per C = 1 ed ε = 16 la
run di E-A allo stesso seed.

| C (righe), ε (colonne) | 16 | 64 | 256 |
|---|---|---|---|
| 0.25 | 11.9 | 10.2 | 6.9 |
| 0.5 | 8.8 | 5.3 | 2.6 |
| 1 | 41.6 (E-A) | 2.6 | 2.9 |

A parità di rumore (stesso C/ε) il C più grande costa sempre meno: 0.25 con ε = 16 contro 1
con ε = 64, 11.9 contro 2.6; 0.25 con ε = 64 contro 1 con ε = 256, 10.2 contro 2.9. La
distorsione del clipping pesa più del rumore. A ε fisso il C migliore dipende da ε: a ε = 16
conviene 0.5, a ε = 64 conviene 1. Sotto le 3 volte ci sono tre celle, e con C = 1 ε = 256 non
costa meno di ε = 64: resta un costo di circa 2.6-2.9 volte che il rumore non spiega, a un seed
solo. Attacchi al caso in tutte le celle (Yeom 0.494-0.502, TPR a FPR 1% 0.009-0.012). Norme
dei delta al round 10: con C = 0.25 sempre sopra C (0.43-0.69), con C = 1 sotto (0.14-0.47).

**Decisione, per la regola fissata prima della griglia.** Punto operativo C = 1, ε = 64, il
più piccolo ε sotto 3 volte. Ora 5 seed in `rq1-eps64`, sul Mac principale come E-A; poi una
cella ε = 32 a un seed per descrivere il ginocchio fra 16 e 64. Circa 13 ore.

```bash
nohup caffeinate -ims bash -c '
set -e
for s in 123 456 789 1234 42; do
  python3 scripts/run_experiments.py --config config/experiment_rq1_eps64.yaml --rounds 10 \
    --seed $s --sweep-dir experiments/rq1-eps64 \
    --per-sample-dump experiments/rq1-eps64/per_sample_seed$s.json
done
python3 scripts/run_experiments.py --config config/experiment_rq1_eps32.yaml --rounds 10 \
  --seed 42 --sweep-dir experiments/_op_eps32' > logs/rq1_eps64.log 2>&1 & disown
```

**Esito dei 5 seed, 2026-09-26.** `rq1-eps64` (commit 8d44ae3): holdout 0.00712, 4.5 volte la
media del no-DP (per seed 2.6, 3.1, 6.5, 7.0, 3.2; rispetto al no-DP dello stesso seed da 1.9 a
14.7). Sopra la soglia: vale la terza riga della tabella di lettura, nessun punto operativo
utile per il client-level in questo regime, e si passa a E-B. ε = 32, seed 42: 7.3 volte.
Attacchi al caso (Yeom 0.492-0.508), test per record z = 0.69. E-D ed E-E con DP restano a
ε = 64, scelto prima di vederli: il loro confronto è fra partizioni e fra algoritmi a parità
di configurazione DP, non richiede che la configurazione sia sotto la soglia.

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

**Config e run di prova (2026-09-30).** `config/experiment_canary_multisite.yaml` (braccio A) e
`config/experiment_canary_multisite_swap.yaml` (braccio B): `experiment_canary_balanced.yaml`
con i tre siti reali, nient'altro cambia (header del file). LiRA legge il modello di ciascun
client (`lira.observation_surface` "client", default), quindi l'update di Office 1; Yeom e
Shadow valutano i canary solo sul modello globale ("global", default): la diluizione si legge
dal confronto fra le due superfici nello stesso JSON. Min e max della normalizzazione vengono
dal training dei tre siti, quindi le baseline a init casuale vanno rifatte con questi config.
Run di prova sul Mac principale: seed 42, senza DP, braccio A, precedute dalle baseline dei due
bracci nello stesso log (`set -e`: se la baseline non trova canary la run non parte).

```bash
cd ~/Documents/ChargeShield-FL && nohup caffeinate -ims bash -c '
set -e
for c in experiment_canary_multisite experiment_canary_multisite_swap; do
  python3 scripts/check_canary_init_confound.py --config config/$c.yaml --seed 42
done
python3 scripts/run_experiments.py --config config/experiment_canary_multisite.yaml --no-dp --seed 42 --sweep-dir experiments/_canary_multisite_s42
' < /dev/null > logs/canary_multisite_s42.log 2>&1 & disown
```

Tempo stimato dai log di `canary_balanced_s42` (Office 1: 44 s per 1000 epoche su 1924
sessioni, 3 min 19 s per gli 8 shadow di un round), scalato sulle dimensioni dei cluster: circa
20 minuti per round di FL, circa 90 per round di LiRA, circa 6 ore in tutto. Verifiche nel log:
`[CANARY] pool unificato: 40 template estratti da site_train_sessions(office1)`; `Client attivi
(3)`; `Cluster office1:` circa 1921 sessioni; nelle righe `[CANARY] AUC` di LiRA `DISTINTI 20x20`
(o 20x19, 20x18: `CanaryPositiveControl.md` 6.5); nel JSON `yeom_canary_auc_roc` e
`canary_raw_mse_auc_roc` per round. Se il tempo misurato conferma la stima, la campagna completa
(5 seed per 2 bracci per 3 condizioni) vale circa 30 run da 6 ore senza contare il costo in più
della DP per record: epoche e LiRA completo o solo loss grezza si decidono col supervisore con
il tempo misurato. Ordine deciso il 2026-09-30: questa prova, poi il canary bilanciato su
Caltech.

**Esito della prova, braccio A (2026-09-30, JSON `experiments/_canary_multisite_s42`, commit
b3feee4 pulito).** 6 ore e 5 minuti (07:25-13:30 sull'orologio del Mac), come stimato. Loss
grezza sull'update di Office 1 (20 x 20 coppie): 0.79, 0.5775, 0.65 nei tre round, contro una
baseline a init casuale di 0.7255 per il braccio A (0.2745 per il B, dal log della run). Al
round 1 il modello locale di Office 1 memorizza i canary; dal round 2 riparte dal modello
globale e la differenza sparisce (loss media dei membri 0.00452 contro 0.00443 dei non membri
al round 2, dal log). LiRA sull'update: 0.65, 0.3231, 0.4208, ma dal round 2 su pool ridotti
e non interpretabile (segnalazione 63). Modello globale, Yeom: 0.4625, 0.3825, 0.39; Shadow
identico (segnalazione 64). Da solo il braccio A non dice se il segnale è appartenenza: serve
la somma con il braccio B allo stesso seed.

**Braccio B, seed 42, lanciato il 2026-09-30 alle 16:27 (orologio del Mac)** sul Mac principale:

```bash
cd ~/Documents/ChargeShield-FL && nohup caffeinate -ims python3 scripts/run_experiments.py --config config/experiment_canary_multisite_swap.yaml --no-dp --seed 42 --sweep-dir experiments/_canary_multisite_swap_s42 < /dev/null > logs/canary_multisite_swap_s42.log 2>&1 & disown
```

**Esito del braccio B e somma (2026-10-01, JSON `experiments/_canary_multisite_swap_s42`, commit
567d32e, che rispetto a b3feee4 cambia solo `docs/`).** 6 ore (16:27-22:27 sull'orologio del
Mac). Loss grezza sull'update di Office 1: 0.7925, 0.5025, 0.5475. Somme A+B: 1.5825, 1.08,
1.1975, contro 1.00 delle baseline a init casuale. Al round 1 è appartenenza; ai round 2 e 3 il
segnale si riduce molto e con un solo seed non si separa dal rumore: la frase del braccio A "la
differenza sparisce" va letta come "si riduce". Media sui 3 round 0.6725 (A) e 0.6142 (B),
somma 1.29 contro 1.46 del sito singolo (`STATO.md` 3.3). Modello globale, Yeom: 0.54, 0.63,
0.605, somme 1.0025, 1.0125, 0.995, nessun segnale; Shadow quasi identico (0.54, 0.6325,
0.6075; segnalazione 64). LiRA sull'update: 0.6975, 0.1667, 0.2768, dal round 2 su pool 12x13 e
16x14; somme 1.3475, 0.49, 0.70, sotto 1 dal round 2, coerente con la segnalazione 63.
**Prossimo passo:** decidere col supervisore se fare la campagna (5 seed per 2 bracci per 3
condizioni, circa 30 run da 6 ore) e con quali epoche e metrica; per dire se il segnale
sopravvive all'aggregazione servono più seed.

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

**Config e run di prova (2026-09-30).** `config/experiment_canary_balanced_caltech.yaml` e
`_swap.yaml`: il riferimento `experiment_canary_balanced.yaml` con solo Caltech,
`canary.site: caltech` e `n_duplicates: 561`; con k = 20 coincidono con Office 1 sia
l'amplificazione per record (2.23% del pool di training) sia la densità aggregata (44.7%
contro 44.6%). Run di prova sulla quarta macchina, seed 42, senza DP, braccio A, con le
baseline dei due bracci nello stesso log, dopo `git fetch susetta && git merge --ff-only
susetta/master`:

```bash
cd ~/ChargeShield-FL && source .venv/bin/activate && nohup caffeinate -ims bash -c '
set -e
for c in experiment_canary_balanced_caltech experiment_canary_balanced_caltech_swap; do
  python3 scripts/check_canary_init_confound.py --config config/$c.yaml --seed 42
done
python3 scripts/run_experiments.py --config config/experiment_canary_balanced_caltech.yaml --no-dp --seed 42 --sweep-dir experiments/_canary_balanced_caltech_s42
' < /dev/null > logs/canary_balanced_caltech_s42.log 2>&1 & disown
```

Tempo stimato dal canary su più siti in corso (1000 epoche: circa 25 minuti per round di FL
su circa 54000 sessioni, circa 97 minuti per round di LiRA con tre cluster), scalato su circa
36300 sessioni di training: 4-5 ore. Correzione del 2026-09-30, dal log della run: sulla quarta
macchina un round di FL dura circa 40 minuti, circa tre volte il Mac principale a parità di
sessioni, quindi la run completa dura circa 12 ore (partita alle 10:44, LiRA dalle 13:50). Verifiche nel log: `pool unificato: 40 template estratti
da site_train_sessions(caltech)`; `Iniettati 20 template × 561 duplicati`; `Client attivi
(1)`; `DISTINTI 20x20` nelle righe `[CANARY] AUC` di LiRA.

**Braccio A finito sulla quarta macchina, bracci spostati sul Mac principale (2026-10-01).** Il
braccio A è finito il 30 settembre alle 22:26 (orologio della quarta macchina), senza errori, in
circa 11 ore e 40 minuti. Il braccio B non può girare lì perché la macchina serve a Domenico, e
i due bracci si confrontano solo sulla stessa macchina (segnalazione 55). Sul Mac principale
girano quindi, uno alla volta, le baseline a init casuale dei due config, il braccio B e poi il
braccio A: circa 4 ore per braccio, stimate dal rapporto di 3 a 1 fra le due macchine. Il
braccio B va per primo, così è pronto anche se la coda si interrompe. Il braccio A della quarta
macchina va in `experiments_altre_macchine/` come controllo di riproducibilità fra macchine; se
le baseline dei due Mac coincidono, l'inizializzazione non dipende dalla macchina.

```bash
cd ~/Documents/ChargeShield-FL && nohup caffeinate -ims bash -c '
set -e
for c in experiment_canary_balanced_caltech experiment_canary_balanced_caltech_swap; do
  python3 scripts/check_canary_init_confound.py --config config/$c.yaml --seed 42
done
python3 scripts/run_experiments.py --config config/experiment_canary_balanced_caltech_swap.yaml --no-dp --seed 42 --sweep-dir experiments/_canary_balanced_caltech_swap_s42
python3 scripts/run_experiments.py --config config/experiment_canary_balanced_caltech.yaml --no-dp --seed 42 --sweep-dir experiments/_canary_balanced_caltech_s42
' < /dev/null > logs/canary_balanced_caltech_mac_s42.log 2>&1 & disown
```

## Rianalisi NVFlare

I 25 dump a ε ≤ 1 (`experiments/_dump_nvflare_provenienza.csv`) sono nel regime
con utility distrutta e confermerebbero il nullo già noto in simulazione. Si
rianalizzano dopo E-A, solo 2-3 celle al punto operativo indicato, e per quelle
servono prima job NVFlare nuovi a quell'ε. Lo script `analizza_dump_nvflare.sh`
passa già lo snapshot del seed; senza, l'AUC tende a 0.5 per costruzione.

## Validazione sul deployment (NVFLARE e Containerlab)

**Scopo.** Mostrare che la simulazione (`scripts/run_experiments.py`) misura lo stesso sistema
del deployment reale e che l'auditor funziona nel punto di osservazione del ML Plane. Non si
replica ogni campagna: un gruppo minimo di celle, confrontato con le stesse celle in
simulazione. Deciso il 2026-09-30.

**Cosa esiste** (`NVFlareIntegration.md`). Deployment Containerlab a cinque nodi (server,
caltech, jpl, office1, fl-admin), riverificato il 2026-09-09 con un job pulito di 10 round e
una campagna a 5 seed con dp-fedavg a ε = 1, più una run central al seed 42. L'analisi degli
attacchi è offline, sui dump dell'aggregatore, con `scripts/run_nvflare_mia.py`; al 2026-09-10
era fatta solo sul seed 42. Il job (`nvflare/jobs/chargeshield_poc/app/custom/`) supporta la
DP per record (`record_dp` passato ad `AutoencoderTrainer`) e l'iniezione dei canary nel sito
indicato (`_inject_canary_members` in `chargeshield_executor.py`).

**Prerequisiti di codice, a campagne chiuse (codice DP).**
- Modalità senza DP: oggi `dp_mode` è sempre `dp-fedavg`, `central` o `local`
  (`chargeshield_aggregator.py` del job, riga 129). Senza di essa nel deployment manca il
  riferimento non protetto.
- Seed dell'aggregatore uguale a quello dei client (campo `seed` di `config_fed_server.json`,
  dal 2026-09-10, senza controllo automatico): da verificare in ogni job.

**Divergenze note dalla simulazione, da allineare o dichiarare.**
- Inizializzazione: nel deployment tutti i client ricevono lo stesso modello iniziale dal
  server, in simulazione ogni client parte dalla propria (segnalazione 48). Il riferimento in
  simulazione per questo confronto è quindi con `ml.common_init: true`.
- Normalizzazione min-max per sito sul client contro statistiche globali in simulazione.
- Clipping assoluto al round 1 in dp-fedavg (nessun modello di riferimento al primo round).
- In central gli update esportati come grezzi sono già clippati dal client.

**Celle, con gli stessi 5 seed della simulazione.**
1. Senza DP, dopo il prerequisito.
2. dp-fedavg al punto operativo (ε = 64, C = 1), più central al seed 42.
3. Facoltativa: DP per record a σ = 1, se il job lo regge.
4. Canary su più siti, solo dopo aver fissato il protocollo in simulazione (segnalazioni 63 e
   64).

Riferimento in simulazione: le stesse celle con inizializzazione comune; per la 1 esiste solo
il controllo a seed 42 (`experiment_ctrl_common_init.yaml`), per la 2 serve un config nuovo.

**Confronto, criterio fissato prima.** Per seed: loss sull'holdout del modello rilasciato e
AUC di Yeom, Shadow e LiRA. Tolleranza sul rapporto delle loss deployment/simulazione da
fissare tenendo conto dell'effetto macchina già misurato (segnalazioni 55 e 60); per gli AUC,
equivalenza a 0.5 con lo stesso margine della simulazione.

**Macchina e tempi.** Mac principale (Docker e Containerlab): il deployment occupa la macchina
e non va in parallelo con altre run. Tempo di un job da misurare al primo lancio.

**Dove va.** Tesi: Fase 1 e capitolo sul metodo, dove sono elencate le differenze fra
simulazione e deployment. Paper: una breve sottosezione di verifica.

## Rinviato o escluso

RQ4 (rilevamento della MIA passiva) esclusa per threat model. ByzantineDetector
end-to-end, secondo dataset, adaptive clipping: lavoro futuro dichiarato. Le
indagini su LiRA (floor per record, scoring simmetrico) si fanno sulla cella
canary bilanciata a Office 1, dove la verità è nota, e non toccano la pipeline
finché E-A ed E-B non sono chiusi.
