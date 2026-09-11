"""
Verifica indipendente del fix seed 2026-09-10 (ChargeShieldAggregator).

Confronta round 1 di due run NVFLARE reali, usando "raw_global_weights"
(la media FedAvg pulita, PRIMA di qualunque rumore central-DP) invece
dell'epsilon riportato nell'audit JSON — l'epsilon di round 1 e' derivato
dalla norma assoluta dei pesi (reference_weights=None al round 1) ed e'
troppo insensibile per essere una prova definitiva da solo.

Se il fix funziona: i due round 1 devono differire (init diverso -> anche
con dati leggermente diversi lato client, la baseline di partenza cambia).
Se il fix NON funziona: i due round 1 possono restare quasi identici,
esattamente come nel bug pre-fix.

Uso:
    python3 scripts/verify_seed_fix_round1.py <pkl_vecchio> <pkl_nuovo>

Esempio (run pre-fix/seed42 vs run appena fatto a seed=999):
    python3 scripts/verify_seed_fix_round1.py \
        experiments/nvflare_fl_results_20260910_134034_d35cbd.pkl \
        experiments/nvflare_fl_results_20260910_140131_1f25db.pkl
"""

import sys
import pickle

import numpy as np


def to_numpy(w):
    if hasattr(w, "detach"):  # torch tensor
        return w.detach().cpu().numpy()
    return np.asarray(w)


def load_round1(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    rounds = payload["rounds"]
    # round_num potrebbe essere chiave int o str a seconda della versione
    key = 1 if 1 in rounds else "1"
    r1 = rounds[key]
    raw = r1.get("raw_global_weights")
    glob = r1.get("global_weights")
    return raw, glob, payload.get("config", {})


def compare(name, list_a, list_b):
    if list_a is None or list_b is None:
        print(f"  [{name}] almeno uno dei due e' None — non confrontabile "
              f"(dp_mode='local'? raw_updates non esportati)")
        return
    print(f"  [{name}] {len(list_a)} layer/tensori")
    all_identical = True
    for i, (a, b) in enumerate(zip(list_a, list_b)):
        a = to_numpy(a)
        b = to_numpy(b)
        identical = np.array_equal(a, b)
        maxdiff = float(np.abs(a - b).max()) if a.shape == b.shape else float("nan")
        all_identical = all_identical and identical
        print(f"    layer {i}: shape={a.shape} identico={identical} "
              f"max_abs_diff={maxdiff:.10g}")
    print(f"  -> TUTTI I LAYER IDENTICI: {all_identical}")
    if all_identical:
        print("  !! Se questo e' vero per RAW_GLOBAL_WEIGHTS tra due run con "
              "seed dichiarati DIVERSI, il fix non ha avuto effetto reale "
              "(o uno dei due run non ha davvero usato il seed che pensi).")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    path_old, path_new = sys.argv[1], sys.argv[2]

    raw_old, glob_old, cfg_old = load_round1(path_old)
    raw_new, glob_new, cfg_new = load_round1(path_new)

    print(f"=== {path_old} ===\nconfig: {cfg_old}\n")
    print(f"=== {path_new} ===\nconfig: {cfg_new}\n")

    print("--- Confronto round 1: raw_global_weights (pre-rumore DP, test primario) ---")
    compare("raw_global_weights", raw_old, raw_new)

    print("\n--- Confronto round 1: global_weights (post-rumore DP, solo informativo) ---")
    compare("global_weights", glob_old, glob_new)


if __name__ == "__main__":
    main()
