#!/usr/bin/env python3
"""
scripts/run_sweep.py
ChargeShield-FL — Sweep automatico multi-round

Esegue run_experiments.py sequenzialmente per ogni combinazione rounds × epsilon.
Dopo ogni esperimento il report Excel del sweep viene aggiornato automaticamente
(6 sheet: Raw Data, Heat Map, Per Rounds, Per Epsilon, Comparison, AUC Progression).

Ogni sweep crea una directory numerata (experiments/full-sweep1/,
experiments/full-sweep2/, ...) che contiene i JSON dei risultati e il file
Excel del sweep (es. full-sweep1.xlsx). Questo garantisce che sweep distinti
non si mescolino mai.

Nota (fix 2026-07-22): questo script esegue lo stesso tipo di sweep
rounds×epsilon del target `make experiment-full-sweep` (Makefile), che usa il
contatore dedicato `full-sweep{N}` per evitare l'ambiguità di un contatore
generico condiviso tra tipi di sweep diversi (no-DP, DP, rounds×epsilon —
vedi README.md "Engineering Fixes", 2026-07-21). Questo script usava ancora
il vecchio schema generico `exp{N}`, ormai abbandonato dappertutto nel
progetto; allineato a `full-sweep{N}` per coerenza con il target Makefile.

Usage:
    python scripts/run_sweep.py                          # round default: 100 200 500 1000
    python scripts/run_sweep.py --rounds 100 200 500     # round personalizzati
    python scripts/run_sweep.py --epsilon 0.5 1.0 2.0    # sweep epsilon
    python scripts/run_sweep.py --rounds 100 200 --epsilon 0.5 1.0  # sweep 2D
    python scripts/run_sweep.py --skip-ids               # salta IDS (più veloce)
    python scripts/run_sweep.py --dry-run                # verifica config senza training
    python scripts/run_sweep.py --sweep-dir experiments/full-sweep3  # directory sweep manuale
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

PROJECT_ROOT  = Path(__file__).parent.parent
RUNNER        = Path(__file__).parent / "run_experiments.py"
CONFIG        = PROJECT_ROOT / "config" / "experiment.yaml"
EXPERIMENTS   = PROJECT_ROOT / "experiments"


def _next_sweep_dir() -> Path:
    """
    Rileva il numero del prossimo sweep disponibile e restituisce la directory.
    Esempio: se esistono full-sweep1/ e full-sweep2/, restituisce
    experiments/full-sweep3/ — stesso contatore dedicato usato dal target
    Makefile `experiment-full-sweep` (fix 2026-07-22: prima usava il vecchio
    schema generico `exp{N}`, condiviso e ambiguo tra i diversi tipi di sweep).
    """
    _prefix = "full-sweep"
    existing = [
        d for d in EXPERIMENTS.iterdir()
        if d.is_dir() and d.name.startswith(_prefix) and d.name[len(_prefix):].isdigit()
    ] if EXPERIMENTS.exists() else []
    next_num = len(existing) + 1
    return EXPERIMENTS / f"{_prefix}{next_num}"


def run_experiment(
    rounds: int,
    epsilon: float,
    sweep_dir: Path,
    extra_args: list[str],
) -> bool:
    """Lancia un singolo esperimento nel sweep_dir. Restituisce True se OK."""
    cmd = [
        sys.executable, str(RUNNER),
        "--config",    str(CONFIG),
        "--rounds",    str(rounds),
        "--epsilon",   str(epsilon),
        "--sweep-dir", str(sweep_dir),
        *extra_args,
    ]
    label = f"rounds={rounds}, ε={epsilon}"
    logger.info("=" * 60)
    logger.info(f"AVVIO — {label} → {sweep_dir.name}")
    logger.info("=" * 60)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        logger.info(f"COMPLETATO — {label} in {elapsed/60:.1f} min")
        return True
    else:
        logger.error(f"FALLITO — {label} (exit code {result.returncode})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChargeShield-FL — Sweep multi-round/epsilon con directory numerata"
    )
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
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help=(
            "Directory del sweep (es. experiments/full-sweep3). "
            "Se non fornita, viene auto-rilevata la prossima disponibile "
            "(schema full-sweep{N})."
        ),
    )
    args = parser.parse_args()

    # Determina la directory del sweep: esplicita o auto-numerata
    if args.sweep_dir:
        sweep_dir = args.sweep_dir.resolve()
    else:
        sweep_dir = _next_sweep_dir()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Directory sweep: {sweep_dir} ({sweep_dir.name})")

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
        ok = run_experiment(rounds, epsilon, sweep_dir, extra)
        if ok:
            successes += 1
        else:
            failures.append((rounds, epsilon))

    logger.info("=" * 60)
    logger.info(f"SWEEP COMPLETATO — {successes}/{len(experiments)} esperimenti OK")
    if failures:
        logger.warning(f"Falliti: {failures}")
    logger.info(f"Report Excel: {sweep_dir}/{sweep_dir.name}.xlsx")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
