#!/usr/bin/env bash
# scripts/run_multiseed_consolidation.sh
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
# NOTA 2026-07-25: 'make experiment-nodp-sweep' è stato avviato a mano
# dall'utente PRIMA di questo script (in corso al momento della scrittura:
# experiments/nodp-sweep1/, round 2/10) — non ripetuto qui per non
# ricalcolare ore di lavoro già in corso. Gli step sotto coprono il resto
# del piano di consolidamento concordato (vedi task #78):
#   1-2. DP-FedAvg ε=1.0 e ε=0.5, 5 seed ciascuno, a n_shadow=16 (i vecchi
#        dp-sweep1/dp-sweep2 usavano n_shadow=8, instabile — archiviati in
#        experiments/_archive_invalid_n_shadow8/, non cancellati).
#   3-5. Completamento di dp-sweep3 (ε=0.1, DP-FedAvg) con i 3 seed mancanti
#        (42, 789, 1234) — i seed 123/456 sono già presenti e a n_shadow=16
#        corretto, quindi qui si aggiunge nella STESSA cartella invece di
#        crearne una nuova (il target Makefile creerebbe sempre una nuova
#        dp-sweepN, frammentando i 5 seed in due cartelle diverse).
#   6-7. Central DP ε=1.0 e ε=0.1, 5 seed ciascuno (oggi solo seed=42
#        esiste, come file sciolto in experiments/) — si ripete anche
#        seed=42 per avere una sweep-dir singola e coerente con
#        aggregazione Seed Aggregation N=5 pulita, invece di spostare a
#        mano il file sciolto (più semplice, meno rischio di errori manuali,
#        a costo di poche ore di ricomputo per un solo seed).
STEPS=(
  "dp-sweep-eps1.0|make experiment-dp-sweep EPS=1.0"
  "dp-sweep-eps0.5|make experiment-dp-sweep EPS=0.5"
  "dp-sweep3-seed42|python3 scripts/run_experiments.py --config config/experiment.yaml --epsilon 0.1 --rounds 10 --seed 42 --n-shadow 16 --sweep-dir experiments/dp-sweep3"
  "dp-sweep3-seed789|python3 scripts/run_experiments.py --config config/experiment.yaml --epsilon 0.1 --rounds 10 --seed 789 --n-shadow 16 --sweep-dir experiments/dp-sweep3"
  "dp-sweep3-seed1234|python3 scripts/run_experiments.py --config config/experiment.yaml --epsilon 0.1 --rounds 10 --seed 1234 --n-shadow 16 --sweep-dir experiments/dp-sweep3"
  "central-dp-eps1.0|make experiment-central-dp-sweep EPS=1.0"
  "central-dp-eps0.1|make experiment-central-dp-sweep EPS=0.1"
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
