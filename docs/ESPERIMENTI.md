# ChargeShield-FL — Esperimenti da eseguire

> **Stato: documento CANONICO.** Aggiornato il 2026-09-22. Sostituisce, per gli
> esperimenti ancora da lanciare, `docs/_storico/TestRoadmap_DSN2027.md`. Ogni
> voce dice quale RQ serve, quale conclusione può cambiare, cosa deve essere vero
> prima di lanciarla, il comando, e come si legge l'esito. Le run vanno una alla
> volta sulla stessa macchina (lock anti-concorrenza nel Makefile, OOM del
> 2026-07-31).

## Regole che valgono per tutti

- **Scorer LiRA congelato.** Il riferimento worst-case no-DP (`nodp-sweep2`,
  z = 7.56) è calcolato con lo scorer attuale. Nessuna modifica a floor,
  `member_scoring` o `observation_surface` prima che E-A ed E-B siano completi,
  altrimenti le celle nuove non sono confrontabili con il riferimento. Le indagini
  su LiRA (segnalazione 10) si fanno sulla cella canary, non sulla pipeline
  principale.
- **Dump per campione sempre attivi** (`--per-sample-dump`): senza, l'analisi per
  record di una cella non è possibile.
- **Seed**: 42, 123, 456, 789, 1234, uno per run, nella stessa sweep-dir.
- **Dopo ogni sweep**: `python3 scripts/check_significance.py` e
  `python3 scripts/genera_matrici_faseA.py`; i numeri si leggono da `risultati/`.

## Prerequisiti di codice (dalla lista segnalazioni)

| prima di | correzione | segnalazione |
|---|---|---|
| E-C | unificare il percentile dei due script worst-case e rigenerare i JSON grezzi mancanti | 6 |
| E-B | accountant record-DP: `fl_rounds` e `delta` da `cfg["experiment"]`, n per client, dichiarare Poisson vs shuffle, `dp-accounting` in `pyproject.toml` | 4 |
| E-B | warning no-DP e registro devono riconoscere `record_dp` | 5 |
| rianalisi NVFlare | `--client-config` con lo snapshot del seed, già nello script rigenerato | 1 |

## E-A — sweep di ε nella zona del ginocchio. PRIORITÀ MASSIMA

**RQ1.** A ε in {1, 0.5, 0.1} la loss è oltre 100 volte il riferimento; a ε = 16 è
2.6 volte, a ε = 8 4.6 volte, ma con un seed solo. Servono celle in cui la DP è
attiva e il modello funziona: solo lì il nullo aggregato e l'analisi per record
diventano una risposta.

**Conclusione che può cambiare.** Se in una cella a utility accettabile l'eccesso
di record esposti scende rispetto ai 412 del no-DP, la DP client-level ha un
effetto misurabile e un punto operativo. Se non scende, o l'utility crolla prima,
la conclusione è che in questo regime non esiste un punto operativo utile: è
comunque la risposta a RQ1.

**Costo.** 20 run a circa 130 minuti ciascuna, circa 2 giorni.

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

**Lettura.** Per cella: loss finale in rapporto al no-DP, AUC dei tre attacchi,
TPR a FPR 1%, e dopo E-C il conteggio per record con z. Un ε è "operativo" se la
loss resta sotto un fattore da dichiarare prima di guardare i risultati; la guida
chiede di fissarlo nel pilot. Proposta: 3 volte il riferimento.

## E-B — record-level DP su dati naturali

**RQ1, fattore unità protetta.** Le dieci run record-DP esistenti sono tutte in
regime canary, a σ = 5, su tre seed. Il confronto client contro record sulla stessa
configurazione ordinaria non è mai stato eseguito.

**Prima di tutto: una run di prova a un seed** per misurare il tempo. DP-SGD a
microbatch 1 su tre siti, 50 epoche, 10 round non è mai stato cronometrato.

```bash
python3 scripts/run_experiments.py \
  --config config/experiment_rq1_recorddp_nm1.yaml --rounds 10 --seed 42 \
  --sweep-dir experiments/_prova_recorddp \
  --per-sample-dump experiments/_prova_recorddp/per_sample_seed42.json
```

Poi, se il tempo lo consente:

```bash
caffeinate -ims bash -c '
set -e
for nm in 0.5 1 2; do
  for s in 42 123 456 789 1234; do
    python3 scripts/run_experiments.py \
      --config config/experiment_rq1_recorddp_nm$nm.yaml --rounds 10 --seed $s \
      --sweep-dir experiments/rq1-recorddp-nm$nm \
      --per-sample-dump experiments/rq1-recorddp-nm$nm/per_sample_seed$s.json
  done
done' 2>&1 | tee logs/rq1_recorddp.log
```

**Lettura.** Il budget della cella è `epsilon_record_dp`, non `epsilon`. Riportare
l'ε per client e il massimo. Il confronto è con la cella client-level allo stesso
seed, non allo stesso ε: sono unità diverse.

## E-C — analisi per record su tutte le celle nuove. Solo calcolo

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

## Canary bilanciato su un secondo sito

**Validazione dello strumento, non una RQ.** Il protocollo che regge (k = 20,
scambio dei ruoli, baseline a init casuale, 5 seed) esiste solo su Office 1. Su
Caltech e JPL ci sono solo run a seed 42 senza blocco canary nel JSON.

Serve un config `experiment_canary_balanced_caltech.yaml` derivato da
`experiment_canary_balanced.yaml` con `site: caltech` e amplificazione per record
pari a Office 1: `n_duplicates` circa 561 su un pool di 25 123 sessioni, cioè
20 template per 561 copie, 11 220 record. Tempo atteso ben superiore a Office 1:
misurare prima con un seed. Procedura identica alla sezione 10 di
`CanaryPositiveControl.md`, bracci A e B con lo stesso seed, baseline con
`check_canary_init_confound.py --seed`.

## E-D — RQ2, partizione IID contro per sito

Config appaiati pronti, differiscono per il solo campo `partition.strategy`.

```bash
for strat in per_site iid; do
  for s in 42 123 456 789 1234; do
    python3 scripts/run_experiments.py \
      --config config/experiment_rq2_$strat.yaml --rounds 10 --seed $s \
      --sweep-dir experiments/rq2-$strat \
      --per-sample-dump experiments/rq2-$strat/per_sample_seed$s.json
  done
done 2>&1 | tee logs/rq2_partizione.log
```

Confronto minimo: IID contro per sito, senza DP e con la configurazione DP
indicata da E-A. La partizione IID è un riferimento sperimentale, non un
deployment.

## E-E — RQ3, mu = 0 contro 0.01

Dopo E-A, sulla configurazione che E-A avrà indicato come operativa. Serve un
config con `ml.proximal_mu: 0.0` appaiato a quello a 0.01. Il campo vive in
`cfg["ml"]`: messo altrove viene ignorato in silenzio.

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
