#!/usr/bin/env python3
"""
scripts/run_sweep.py
ChargeShield-FL — Sweep automatico multi-round

Esegue run_experiments.py sequenzialmente per 100, 200, 500, 1000 round.
Dopo ogni esperimento il report Excel viene aggiornato automaticamente
(6 sheet: Raw Data, Heat Map, Per Rounds, Per Epsilon, Comparison, AUC Progression).

Usage:
    python scripts/run_sweep.py                         # round default: 100 200 500 1000
    python scripts/run_sweep.py --rounds 100 200 500    # round personalizzati
    python scripts/run_sweep.py --epsilon 0.5 1.0 2.0   # sweep epsilon invece di rounds
    python scripts/run_sweep.py --rounds 100 200 --epsilon 0.5 1.0  # sweep 2D
    python scripts/run_sweep.py --skip-ids              # salta IDS (più veloce)
    python scripts/run_sweep.py --dry-run               # verifica config senza training
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep")

PROJECT_ROOT = Path(__file__).parent.parent
RUNNER      = Path(__file__).parent / "run_experiments.py"
CONFIG      = PROJECT_ROOT / "config" / "experiment.yaml"


def run_experiment(rounds: int, epsilon: float, extra_args: list[str]) -> bool:
    """Lancia un singolo esperimento. Restituisce True se termina con successo."""
    cmd = [
        sys.executable, str(RUNNER),
        "--config",  str(CONFIG),
        "--rounds",  str(rounds),
        "--epsilon", str(epsilon),
        *extra_args,
    ]
    label = f"rounds={rounds}, ε={epsilon}"
    logger.info("=" * 60)
    logger.info(f"AVVIO — {label}")
    logger.info("=" * 60)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0:
        logger.info(f"COMPLETATO — {label} in {elapsed/60:.1f} min")
        return True
    else:
        logger.error(f"FALLITO — {label} (exit code {result.returncode})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="ChargeShield-FL — Sweep multi-round/epsilon")
    parser.add_argument(
        "--rounds", type=int, nargs="+",
        default=[100, 200, 500, 1000],
        help="Sequenza di round da eseguire (default: 100 200 500 1000)",
    )
    parser.add_argument(
        "--epsilon", type=float, nargs="+",
        default=[1.0],
        help="Valori di epsilon da testare (default: 1.0)",
    )
    parser.add_argument("--skip-ids", action="store_true", help="Salta valutazione IDS")
    parser.add_argument("--dry-run",  action="store_true", help="Dry-run senza training")
    args = parser.parse_args()

    extra: list[str] = []
    if args.skip_ids:
        extra.append("--skip-ids")
    if args.dry_run:
        extra.append("--dry-run")

    # Costruisce la lista di (rounds, epsilon) da eseguire
    experiments: list[tuple[int, float]] = [
        (r, e) for e in args.epsilon for r in args.rounds
    ]

    logger.info(f"Sweep: {len(experiments)} esperimenti pianificati")
    for i, (r, e) in enumerate(experiments, 1):
        logger.info(f"  [{i}/{len(experiments)}] rounds={r}, ε={e}")

    successes = 0
    failures:  list[tuple[int, float]] = []

    for rounds, epsilon in experiments:
        ok = run_experiment(rounds, epsilon, extra)
        if ok:
            successes += 1
        else:
            failures.append((rounds, epsilon))

    logger.info("=" * 60)
    logger.info(f"SWEEP COMPLETATO — {successes}/{len(experiments)} esperimenti OK")
    if failures:
        logger.warning(f"Falliti: {failures}")
    logger.info("Report Excel aggiornato in experiments/ChargeShield_FL_Results.xlsx")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
