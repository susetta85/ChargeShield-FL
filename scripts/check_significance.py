#!/usr/bin/env python3
"""Bootstrap CI sulla media di mean_lira_auc_roc tra seed, per gruppo sweep.

Risponde alla domanda: un AUC medio ~0.5 osservato su N=5 seed e' davvero
statisticamente indistinguibile dal caso (0.5), o e' solo un punto stimato
senza intervallo di confidenza intorno? Nessuna dipendenza da scipy (non
installato in questo sandbox e non in requirements.txt) - bootstrap con solo
`random`/`statistics` di libreria standard.
"""
import json
import glob
import random
import statistics
import sys

random.seed(0)  # riproducibilita' del bootstrap stesso (non del training)

GROUPS = {
    "central-sweep1 (central, eps=1.0)": "experiments/central-sweep1/experiment_*.json",
    "central-sweep2 (central, eps=0.1, IN CORSO)": "experiments/central-sweep2/experiment_*.json",
    "dp-sweep2 (dp-fedavg, eps=1.0)": "experiments/dp-sweep2/experiment_*.json",
    "dp-sweep3 (dp-fedavg, eps=0.1)": "experiments/dp-sweep3/experiment_*.json",
    "dp-sweep-eps0.5 (dp-fedavg, eps=0.5)": "experiments/dp-sweep-eps0.5/experiment_*.json",
    "local-sweep1 (local, eps=1.0)": "experiments/local-sweep1/experiment_*.json",
    "local-sweep2 (local, eps=0.1)": "experiments/local-sweep2/experiment_*.json",
}


def bootstrap_ci(values, n_resamples=10000, alpha=0.05):
    n = len(values)
    if n < 2:
        return None
    means = []
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return means[lo_idx], means[hi_idx]


def main():
    print(f"{'gruppo':<42} {'n':>3} {'mean':>8} {'std':>8} {'95% CI':>20} {'contiene 0.5?':>14}")
    print("-" * 100)
    for label, pattern in GROUPS.items():
        files = sorted(glob.glob(pattern))
        aucs = []
        for f in files:
            d = json.load(open(f))
            s = d.get("summary", {})
            v = s.get("mean_lira_auc_roc")
            if v is not None:
                aucs.append(v)
        if not aucs:
            print(f"{label:<42} nessun dato")
            continue
        mean = statistics.mean(aucs)
        std = statistics.stdev(aucs) if len(aucs) > 1 else float("nan")
        ci = bootstrap_ci(aucs)
        if ci is None:
            ci_str = "n<2, N/A"
            contains = "N/A"
        else:
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            contains = "SI" if ci[0] <= 0.5 <= ci[1] else "NO"
        print(f"{label:<42} {len(aucs):>3} {mean:>8.4f} {std:>8.4f} {ci_str:>20} {contains:>14}")


if __name__ == "__main__":
    main()
