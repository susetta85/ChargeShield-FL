# ChargeShield-FL — Esperimenti da eseguire

> **Stato: documento CANONICO.** Aggiornato il 2026-09-24 (stato di E-A e del braccio no-DP di E-D). Sostituisce,
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
| E-B | accountant record-DP: `fl_rounds` e `delta` da `cfg["experiment"]`, n per client, dichiarare Poisson vs shuffle, `dp-accounting` in `pyproject.toml` | 4 |
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

Per applicare la tabella di lettura servono ancora: la scelta del riferimento di costo
(`STATO.md` sezione 3.8, decide se ε = 16 è a costo accettabile) ed E-C sulle quattro
celle, dopo la segnalazione 6. La cella ε = 8 è incoerente fra seed (quarta riga
della tabella).

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

```bash
python3 scripts/run_experiments.py \
  --config config/experiment_rq1_recorddp_nm1.yaml --rounds 10 --seed 42 --no-dp \
  --sweep-dir experiments/_prova_recorddp \
  --per-sample-dump experiments/_prova_recorddp/per_sample_seed42.json
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

**Lettura.** Il budget della cella è `epsilon_record_dp`, non `epsilon`. Riportare
l'ε per client e il massimo. Il confronto è con la cella client-level allo stesso
seed e a costo comparabile, non allo stesso ε: sono unità diverse.

## E-C — analisi per record su tutte le celle nuove. Solo calcolo

Dopo la segnalazione 6.

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
no-DP si esegue su una terza macchina (Windows), alternando per seed mu = 0
(`experiments/rq3-mu0`) e un braccio appaiato mu = 0.01 sulla stessa macchina
(`experiments/rq3-mu0.01`, `config/experiment.yaml`), perche' `nodp-sweep2` e' di
un'altra macchina e di un commit dell'8 settembre. Al rientro delle cartelle vale
la segnalazione 45: tenerle fuori da `experiments/` finche' non e' corretta.

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
