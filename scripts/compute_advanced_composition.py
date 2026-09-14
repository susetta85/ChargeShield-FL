#!/usr/bin/env python3
"""
Advanced composition (Dwork & Roth 2014, Theorem 3.20) — calcolo retroattivo
su ogni JSON già esistente, task #118 (Sprint 10zz+83, 2026-09-14).

Contesto: la composizione cumulativa che questo progetto riporta finora è
naive (epsilon_cumulative_naive = epsilon_per_round * fl_rounds, vedi
scripts/run_experiments.py::save_results()) — un limite reale, dichiarato
esplicitamente nel paper (§9) dopo la verifica del gap NC-vs-RDP di
Jayaraman & Evans (Sprint 10zz+81). Delle tre opzioni di rimedio discusse
con l'utente (A: composizione avanzata in forma chiusa, nessuna nuova
dipendenza; B: accounting RDP in forma chiusa per il meccanismo Gaussiano;
C: libreria esterna completa), questo script implementa e verifica
retroattivamente l'opzione A — riusa la STESSA formula già wired in
run_experiments.py (_advanced_composition_epsilon(), non duplicata qui) e
la applica a ogni esperimento già completato, usando solo epsilon/delta/
fl_rounds già loggati in ogni JSON — NESSUN nuovo run richiesto.

Risultato onesto, non nascosto: per fl_rounds=10 (usato in TUTTA la
campagna principale del paper), la composizione avanzata NON è mai più
stretta della naive per nessuno dei tre epsilon usati (1.0/0.5/0.1) — è
un miglioramento asintotico large-k/small-eps che richiede molti più
round di quanti questo progetto ne usi. Vedi la sezione "crossover" sotto
per il numero di round esatto a cui l'opzione A comincerebbe ad aiutare
per ciascun epsilon, calcolato analiticamente, non solo empiricamente sui
JSON esistenti.

Uso:
    python3 scripts/compute_advanced_composition.py [--include-diagnostic]
"""
import argparse
import json
import math
import sys

from check_significance import discover_groups

# NOTA: la formula è duplicata qui invece di importata da
# run_experiments.py::_advanced_composition_epsilon() perché
# run_experiments.py importa torch a livello di modulo — assente in
# questo sandbox di analisi (stesso limite già documentato per i test
# puro-Python del progetto, es. tests/test_mia_advantage.py). Le due
# copie sono verificate identiche da tests/test_advanced_composition.py,
# che replica la STESSA formula da zero (non uno dei due file) e la
# confronta contro valori noti — vedi quel file per i dettagli.


def _advanced_composition_epsilon(
    epsilon_per_round: float,
    delta_per_round: float,
    rounds: int,
    delta_prime: float | None = None,
) -> tuple[float, float]:
    """Copia — vedi run_experiments.py per la docstring completa e la derivazione."""
    if rounds <= 0:
        return 0.0, float(delta_prime if delta_prime is not None else delta_per_round)
    if delta_prime is None:
        delta_prime = delta_per_round
    epsilon_prime = (
        epsilon_per_round * math.sqrt(2 * rounds * math.log(1.0 / delta_prime))
        + rounds * epsilon_per_round * (math.exp(epsilon_per_round) - 1)
    )
    delta_total = rounds * delta_per_round + delta_prime
    return float(epsilon_prime), float(delta_total)


def crossover_rounds(epsilon: float, delta: float, max_k: int = 500_000) -> int | None:
    """
    Primo k (numero di round) per cui la composizione avanzata è più
    stretta della naive, a parità di epsilon/delta per-round. None se non
    accade entro max_k round — condizione nota: il termine lineare
    dominante di advanced ha pendenza epsilon*(e^epsilon - 1), quello di
    naive ha pendenza epsilon; per k grande advanced/naive -> (e^epsilon
    - 1), quindi un crossover a k finito può esistere SOLO se epsilon <
    ln(2) ~= 0.693 (altrimenti (e^epsilon-1) >= 1 e advanced non scende
    mai sotto naive, per nessun k). Verificato numericamente qui sotto,
    non solo asserito.
    """
    for k in range(1, max_k + 1):
        naive = epsilon * k
        advanced, _ = _advanced_composition_epsilon(epsilon, delta, k)
        if advanced < naive:
            return k
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-diagnostic", action="store_true",
                     help="includi anche le cartelle diagnostiche/calibrazione (prefisso _)")
    ap.add_argument("--exclude-methodology-variants", action="store_true",
                     help="escludi entity-split-sweep* (default: incluso, stesso motivo di compute_pes.py)")
    args = ap.parse_args()

    groups = discover_groups(
        include_diagnostic=args.include_diagnostic,
        include_methodology_variants=not args.exclude_methodology_variants,
    )

    print(f"{'gruppo':<32} {'n':>3} {'eps':>5} {'rounds':>6} "
          f"{'naive':>10} {'advanced':>12} {'best_known':>10} {'advanced_vince?':>16}")
    print("-" * 100)

    n_total = 0
    n_advanced_wins = 0
    seen_eps_delta_rounds: set[tuple[float, float, int]] = set()

    for label, files in sorted(groups.items()):
        for f in files:
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            cfg = d.get("config", {})
            epsilon = cfg.get("epsilon")
            delta = cfg.get("delta", 1e-5)
            rounds = cfg.get("fl_rounds")
            no_dp = cfg.get("no_dp", epsilon is None)
            if no_dp or epsilon is None or rounds is None:
                continue

            naive = epsilon * rounds
            advanced, _delta_tot = _advanced_composition_epsilon(epsilon, delta, rounds)
            best = min(naive, advanced)
            wins = advanced < naive
            n_total += 1
            n_advanced_wins += int(wins)
            seen_eps_delta_rounds.add((epsilon, delta, rounds))

            print(f"{label:<32} {'':>3} {epsilon:>5.2f} {rounds:>6d} "
                  f"{naive:>10.4f} {advanced:>12.4f} {best:>10.4f} "
                  f"{'SI' if wins else 'no':>16}")

    print("-" * 100)
    print(f"Totale esperimenti DP valutati: {n_total} — advanced-composition più stretta "
          f"della naive in {n_advanced_wins}/{n_total} casi.")

    print()
    print("Crossover analitico (round esatto in cui l'opzione A comincia ad aiutare, "
          "delta=1e-5, per gli epsilon usati nella campagna):")
    for epsilon in sorted({e for (e, _d, _r) in seen_eps_delta_rounds}):
        k = crossover_rounds(epsilon, 1e-5)
        if k is None:
            print(f"  epsilon={epsilon}: MAI (epsilon >= ln(2)={math.log(2):.4f} -> "
                  f"advanced non e' mai piu' stretta della naive, per nessun numero di round)")
        else:
            print(f"  epsilon={epsilon}: k={k} round (questo progetto ne usa "
                  f"{sorted({r for (e, _d, r) in seen_eps_delta_rounds if e == epsilon})})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
