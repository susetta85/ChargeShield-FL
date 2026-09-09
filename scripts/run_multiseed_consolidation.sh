#!/usr/bin/env bash
# scripts/run_multiseed_consolidation.sh
#
# FIX 2026-08-03 (root cause finally found for the "Fatal Python error:
# init_sys_streams: OSError: Bad file descriptor" crashes archived in
# experiments/_archive_20260731_stalled_or_crashed/central-sweep1..4 — every
# seed of every central-DP sweep run that way crashed instantly at Python
# startup, "0 min" duration, no FL round ever begun): the recommended launch
# command below was `nohup caffeinate ... &` with NO stdin redirection.
# `nohup` only guards against SIGHUP, it does NOT detach stdin — every
# `python3` subprocess this script later spawns (one per seed, potentially
# hours apart) inherits fd 0 from the terminal/pty that started it. If that
# terminal is later closed (or an SSH session drops), fd 0 becomes invalid;
# every subsequent Python interpreter fails at its own startup trying to wrap
# it into sys.stdin, with exactly this error — independent of anything in
# this project's own code, and not caught by the Sprint 10o Makefile exit-code
# fix (that fix made the failure get *reported* correctly instead of silently
# claiming success — it could not and did not fix the underlying crash).
# Fixed here defensively so it's safe regardless of how this script is
# invoked: redirect this script's own stdin from /dev/null immediately, so
# every child process inherits a valid, always-open fd 0 instead of the
# calling terminal's.
exec < /dev/null
#
# ChargeShield-FL — esecuzione automatica e sequenziale di più esperimenti,
# uno alla volta, con log per-step e riepilogo finale.
#
# Creato 2026-07-25 su richiesta esplicita dell'utente ("creiamo un ciclo
# for/while per l'esecuzione automatica... per le prossime volte li
# avviamo in automatico con un piccolo script ed un ciclo for"), per non
# dover lanciare a mano un comando alla volta durante il consolidamento
# multi-seed della pipeline membership (Yeom/Shadow/LiRA).
#
# COME FUNZIONA:
#   - STEPS sotto è un elenco di comandi (Makefile target o invocazione
#     diretta di run_experiments.py), eseguiti IN SEQUENZA — MAI in
#     parallelo: le run FL/LiRA sono CPU-bound e competerebbero per le
#     stesse risorse, oltre a essere il pattern già stabilito in questo
#     progetto per gli sweep multi-seed (vedi commenti nel Makefile).
#   - Un fallimento in uno step NON blocca gli step successivi (loggato e
#     si continua) — stessa filosofia già usata in
#     run_experiments.py::run_registered_attacks() (un attacco fallito non
#     blocca gli altri né il salvataggio). Pensato per esecuzione "lascia e
#     vai via": un problema isolato in una run non deve far perdere ore di
#     run successive valide.
#   - Ogni step scrive il proprio log in logs/consolidation/<n>_<slug>.log,
#     OLTRE al log che il Makefile/run_experiments.py scrive già da solo
#     dentro la propria sweep-dir (experiments/<sweep>/sweep_log.txt) —
#     nessuna duplicazione persa, solo un log aggiuntivo con anche eventuali
#     errori di shell/exit code non catturati nel log applicativo.
#   - Fix 2026-07-31 (l'utente ha segnalato che `tail -f logs/consolidation_master.log`
#     non mostrava alcun progresso): l'output di ogni step veniva scritto SOLO nel
#     suo file di log per-step (`> "$step_log" 2>&1`), mai su stdout — quindi il
#     master log (che è solo lo stdout dello script, catturato da `nohup ... >
#     logs/consolidation_master.log`) conteneva solo gli echo di intestazione, mai
#     l'avanzamento reale. Ora ogni step usa `tee` per scrivere ENTRAMBI i file in
#     tempo reale; l'exit code reale del comando (non quello di `tee`) viene letto
#     da `${PIPESTATUS[0]}`, sicuro qui perché lo script è bash (non lo sarebbe in
#     POSIX sh puro, dove non esiste `PIPESTATUS`).
#   - Alla fine stampa un riepilogo: quali step sono riusciti/falliti,
#     tempo totale e per-step.
#
# USO CONSIGLIATO (girare anche chiudendo il terminale/schermo, seguirlo dal vivo):
#   mkdir -p logs
#   nohup caffeinate -dimsu ./scripts/run_multiseed_consolidation.sh > logs/consolidation_master.log 2>&1 &
#   disown
#   tail -f logs/consolidation_master.log
# NOTA: `caffeinate` impedisce lo sleep automatico del Mac mentre lo script gira,
# ma NON impedisce lo sleep se chiudi il coperchio del portatile senza monitor/
# tastiera esterni collegati — quello è uno sleep hardware che nessun comando
# software evita. Lascia il coperchio aperto (schermo spento va bene) o collega
# un monitor esterno.
#
# PER LE PROSSIME VOLTE: modifica l'array STEPS sotto con i comandi che ti
# servono (nuovo sweep, nuovo epsilon, altri seed, ecc.) e rilancia lo
# script — non serve altro. Ogni step è indipendente e idempotente rispetto
# ai file che produce (nuove sweep-dir numerate o seed aggiuntivi in una
# sweep-dir esistente), quindi rilanciare lo script dopo aver rimosso gli
# step già completati dall'array è sicuro.

set -uo pipefail   # NON -e: un singolo step fallito non deve fermare gli altri
cd "$(dirname "$0")/.." || exit 1   # sempre dalla root del progetto, non da scripts/

LOG_DIR="logs/consolidation"
mkdir -p "$LOG_DIR"

# ── Step da eseguire, IN QUEST'ORDINE ───────────────────────────────────────
# Formato: "slug|comando completo"
#
# RISCRITTO 2026-08-26 (sostituisce l'elenco precedente, 2026-07-25):
# quell'elenco copriva un piano di consolidamento parziale, calcolato PRIMA
# della catena di 6 fix reali a run_lira() trovati e corretti tra il
# 2026-08-11 e il 2026-08-21 (pooling cross-cluster, floor sigma simmetrico,
# esclusione outlier >8σ, ancoraggio μ_in, universo shadow simmetrico — vedi
# README Sprint 10x-10cc). OGNI numero LiRA raccolto prima di questi fix è
# invalidato, inclusi tutti gli sweep che quell'elenco produceva — per
# questo experiments/ è stato ripulito (archiviato, non cancellato: vedi
# experiments/_archive_20260815_pre_sigma_fix/ e
# experiments/_archive_20260820_pre_outlier_fix/) e la numerazione delle
# sweep-dir riparte pulita da 1 per ognuna delle 7 configurazioni.
#
# Una verifica preliminare a 4 vie (10 round, no-DP + Central DP locale,
# più il deployment reale ContainerLab/NVFLARE, più due sweep brevi a
# ε=8/16) ha già confermato la direzione attesa (AUC≈0.50 ovunque, nessun
# leakage rilevabile — README Sprint 10dd): questa campagna è il passo di
# rigore statistico (5 seed × bootstrap CI) che porta quel risultato a
# livello pubblicabile, non una nuova ricerca esplorativa.
#
# RISCRITTO 2026-09-03 (Sprint 10zz+30, task #52/#55) — sostituisce l'elenco
# precedente (2026-08-26). Motivo: (1) la matrice di config del paper aveva
# due buchi genuini mai eseguiti (Central/Local DP ε=0.5 — vedi
# docs/TestRoadmap_DSN2027.md, nuova sezione "Prossimi esperimenti"); (2)
# ogni config va ora ripetuta comunque per le metriche aggiunte in questa
# sessione (MIA Advantage/Confusion Matrix, task #41/#49; curve ROC complete
# + dump per-campione worst-case, task #50/#54) — nessuna di queste è
# retroattiva sui JSON storici, serve un run reale per popolarle. `DUMP_EXTRAS=1`
# (nuovo, Makefile) aggiunge --per-sample-dump/--roc-curve-dump-dir per ogni
# seed automaticamente — zero costo aggiuntivo, dati che altrimenti
# servirebbe un secondo giro di run per raccogliere.
#
# ORDINE DI PRIORITÀ (vedi docs/TestRoadmap_DSN2027.md per il ragionamento
# completo): prima i due buchi genuini della matrice (mai eseguiti finora),
# poi DP-FedAvg (il meccanismo primario citato nel paper), poi Central/Local
# DP ε=1.0/0.1 (hanno già dati preliminari, qui solo backfill metriche), poi
# la baseline no-DP. Local ε=1.0/0.1 restano attesi identici ai
# corrispondenti DP-FedAvg in questa simulazione single-process (nota
# 2026-08-06) — inclusi per completezza/coerenza storica, non perché ci si
# aspetti un numero diverso.
STEPS=(
  "central-dp-eps0.5|make experiment-central-dp-sweep EPS=0.5 DUMP_EXTRAS=1"
  "local-dp-eps0.5|make experiment-local-dp-sweep EPS=0.5 DUMP_EXTRAS=1"
  "dp-sweep-eps1.0|make experiment-dp-sweep EPS=1.0 DUMP_EXTRAS=1"
  "dp-sweep-eps0.5|make experiment-dp-sweep EPS=0.5 DUMP_EXTRAS=1"
  "dp-sweep-eps0.1|make experiment-dp-sweep EPS=0.1 DUMP_EXTRAS=1"
  "central-dp-eps1.0|make experiment-central-dp-sweep EPS=1.0 DUMP_EXTRAS=1"
  "central-dp-eps0.1|make experiment-central-dp-sweep EPS=0.1 DUMP_EXTRAS=1"
  "local-dp-eps1.0|make experiment-local-dp-sweep EPS=1.0 DUMP_EXTRAS=1"
  "local-dp-eps0.1|make experiment-local-dp-sweep EPS=0.1 DUMP_EXTRAS=1"
  "nodp-sweep|make experiment-nodp-sweep DUMP_EXTRAS=1"
)

TOTAL=${#STEPS[@]}
RESULTS=()
START_ALL=$(date +%s)

echo "════════════════════════════════════════════════════════════"
echo " ChargeShield-FL — consolidamento multi-seed: $TOTAL step"
echo " Avviato: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

for i in "${!STEPS[@]}"; do
    idx=$((i + 1))
    slug="${STEPS[$i]%%|*}"
    cmd="${STEPS[$i]#*|}"
    step_log="$LOG_DIR/${idx}_${slug}.log"

    echo ""
    echo "── [$idx/$TOTAL] $slug ──────────────────────────────────"
    echo "  comando: $cmd"
    echo "  log:     $step_log"
    echo "  inizio:  $(date '+%Y-%m-%d %H:%M:%S')"

    step_start=$(date +%s)
    eval "$cmd" 2>&1 | tee "$step_log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        status="OK"
    else
        status="FALLITO (exit $rc)"
    fi
    step_end=$(date +%s)
    elapsed_min=$(( (step_end - step_start) / 60 ))

    echo "  fine:    $(date '+%Y-%m-%d %H:%M:%S') — $status — ${elapsed_min} min"
    RESULTS+=("[$idx/$TOTAL] $slug — $status — ${elapsed_min} min — log: $step_log")
done

END_ALL=$(date +%s)
TOTAL_MIN=$(( (END_ALL - START_ALL) / 60 ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo " RIEPILOGO — completato in ${TOTAL_MIN} min totali"
echo "════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Prossimo passo: controlla il foglio 'Seed Aggregation' negli Excel di"
echo "ogni sweep-dir (N Seed dovrebbe essere 5 ovunque) prima di considerare"
echo "il consolidamento completo."
