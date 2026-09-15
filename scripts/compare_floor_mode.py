#!/usr/bin/env python3
"""Confronto diagnostico A/B: floor_mode "symmetric" (default, invariato) vs
"independent" (Sprint 10zz+87) su un run LiRA — nessuna dipendenza da torch,
legge solo i JSON di output già prodotti da scripts/run_experiments.py.

Nato da un feedback esterno verificato sui dati reali (2026-09-14):
lira_debug_matched_formula_auc mostra che su `central` (floor-hit-rate
95-99%) il punteggio LiRA calibrato collassa nella forma algebrica di
_sablayrolles_score() quando sigma_in=sigma_out=floor condiviso, e il
"null result" di central potrebbe essere la cancellazione tra
un'inversione reale (formula-matched, ~0.21-0.34) e l'ancoraggio mu_in
dei non-membri, non un'assenza genuina di segnale.

Uso (dopo aver lanciato lo STESSO config due volte, una con
cfg["lira"]["floor_mode"]="symmetric" (o assente, e' il default) e una con
"independent", stesso seed/dp-mode/epsilon, sweep-dir diverse):

    python3 scripts/compare_floor_mode.py \
        --before experiments/central_eps1_floor_symmetric/experiment_*.json \
        --after  experiments/central_eps1_floor_independent/experiment_*.json

Accetta anche una directory (prende il file experiment_*.json piu' recente).
Stampa, per ogni round: floor-hit-rate (in/out), matched_formula_auc, e
lira_auc_roc — prima vs dopo — cosi' si vede a colpo d'occhio se
disattivare il floor condiviso cambia il segnale su `central`, e quanto.
"""
import argparse
import glob
import json
import sys
from pathlib import Path


def _resolve_path(spec: str) -> Path:
    """Accetta un file .json diretto, un glob, o una directory (prende il
    file experiment_*.json piu' recente per mtime)."""
    p = Path(spec)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(p.glob("experiment_*.json"), key=lambda f: f.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"Nessun experiment_*.json in {spec}")
        return candidates[-1]
    matches = sorted(glob.glob(spec))
    if not matches:
        raise FileNotFoundError(f"Nessun file corrisponde a {spec}")
    return Path(sorted(matches, key=lambda f: Path(f).stat().st_mtime)[-1])


def _extract_by_round(path: Path) -> dict[str, dict]:
    d = json.load(open(path))
    per_round = d.get("per_round", {})
    out = {}
    for r, rd in per_round.items():
        mia = rd.get("mia", {})
        out[r] = {
            "floor_hit_in": mia.get("lira_debug_sigma_in_floor_hit_rate"),
            "floor_hit_out": mia.get("lira_debug_sigma_out_floor_hit_rate"),
            "matched_formula_auc": mia.get("lira_debug_matched_formula_auc"),
            "lira_auc_roc": mia.get("lira_auc_roc"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="File/dir/glob del run con floor_mode=symmetric (default)")
    ap.add_argument("--after", required=True, help="File/dir/glob del run con floor_mode=independent")
    args = ap.parse_args()

    before_path = _resolve_path(args.before)
    after_path = _resolve_path(args.after)
    print(f"BEFORE (symmetric, default): {before_path}")
    print(f"AFTER  (independent):       {after_path}")
    print()

    before = _extract_by_round(before_path)
    after = _extract_by_round(after_path)
    rounds = sorted(set(before) | set(after), key=int)

    if not rounds:
        print("Nessun round con diagnostica lira_debug_* in nessuno dei due file — "
              "verifica che siano run LiRA reali (non file diagnostici troncati).")
        sys.exit(1)

    header = (
        f"{'round':>5} | {'floor_hit_in':>12} {'floor_hit_out':>13} "
        f"{'matched_fauc':>12} {'lira_auc':>9} || "
        f"{'floor_hit_in':>12} {'floor_hit_out':>13} {'matched_fauc':>12} {'lira_auc':>9}"
    )
    print(header)
    print("-" * len(header))
    print(f"{'':>5} | {'--- BEFORE (symmetric) ---':^50} || {'--- AFTER (independent) ---':^50}")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

    for r in rounds:
        b = before.get(r, {})
        a = after.get(r, {})
        print(
            f"{r:>5} | {fmt(b.get('floor_hit_in')):>12} {fmt(b.get('floor_hit_out')):>13} "
            f"{fmt(b.get('matched_formula_auc')):>12} {fmt(b.get('lira_auc_roc')):>9} || "
            f"{fmt(a.get('floor_hit_in')):>12} {fmt(a.get('floor_hit_out')):>13} "
            f"{fmt(a.get('matched_formula_auc')):>12} {fmt(a.get('lira_auc_roc')):>9}"
        )

    print()
    print(
        "Lettura: se dopo (independent) matched_formula_auc torna verso 0.5 e/o "
        "lira_auc_roc si allontana da 0.5 in modo stabile su piu' round, il floor "
        "condiviso stava mascherando segnale reale su central — vale la pena alzare "
        "n_shadow e ripetere. Se matched_formula_auc resta lontano da 0.5 anche cosi', "
        "il problema e' altrove (ancoraggio mu_in dei non-membri, vedi errata punto 3)."
    )


if __name__ == "__main__":
    main()
