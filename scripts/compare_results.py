#!/usr/bin/env python3
# scripts/compare_results.py
# ChargeShield-FL — Confronto risultati esperimenti
#
# Legge tutti i JSON in experiments/ e produce:
#   - tabella rounds × epsilon → AUC-ROC (heat map)
#   - confronto FedAvg vs FedProx
#
# Usage:
#   python scripts/compare_results.py
#   python scripts/compare_results.py --output experiments/summary.csv

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def load_results() -> list[dict]:
    """Carica tutti i file JSON in experiments/."""
    results = []
    for f in sorted(EXPERIMENTS_DIR.glob("experiment_*.json")):
        with open(f) as fp:
            data = json.load(fp)
            data["_file"] = f.name
            results.append(data)
    return results


def print_heatmap(results: list[dict]) -> None:
    """Stampa heat map rounds × epsilon → AUC-ROC."""
    # Raccoglie valori unici
    rounds_set  = sorted({r["config"]["fl_rounds"] for r in results})
    epsilon_set = sorted({r["config"]["epsilon"]   for r in results})

    print("\n=== Heat Map: AUC-ROC (rounds × epsilon) ===\n")

    # Header
    header = f"{'rounds':>8} | " + " | ".join(f"ε={e:<5}" for e in epsilon_set)
    print(header)
    print("-" * len(header))

    for r in rounds_set:
        row = f"{r:>8} | "
        for e in epsilon_set:
            # Trova il risultato corrispondente
            match = [
                x for x in results
                if x["config"]["fl_rounds"] == r
                and abs(x["config"]["epsilon"] - e) < 1e-9
            ]
            if match:
                auc = match[-1]["summary"]["mean_auc_roc"]
                cell = f"{auc:.3f} " if auc is not None else "  N/A "
            else:
                cell = "  --- "
            row += f"{cell:<8} | "
        print(row)

    print()


def save_csv(results: list[dict], output: Path) -> None:
    """Salva i risultati in CSV per analisi esterna."""
    import csv
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "timestamp", "fl_rounds", "epsilon",
            "proximal_mu", "mean_auc_roc", "max_auc_roc",
            "min_auc_roc", "privacy_risk"
        ])
        for r in results:
            writer.writerow([
                r["_file"],
                r["timestamp"],
                r["config"]["fl_rounds"],
                r["config"]["epsilon"],
                r["config"].get("proximal_mu", 0.0),
                r["summary"]["mean_auc_roc"],
                r["summary"]["max_auc_roc"],
                r["summary"]["min_auc_roc"],
                r["summary"]["privacy_risk"],
            ])
    print(f"CSV salvato: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChargeShield-FL — Confronto risultati esperimenti"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Salva risultati in CSV (es. experiments/summary.csv)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results()

    if not results:
        print("Nessun risultato trovato in experiments/")
        print("Esegui prima: make experiment-full-sweep")
        return

    print(f"Esperimenti trovati: {len(results)}")
    print_heatmap(results)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
