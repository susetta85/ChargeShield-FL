#!/usr/bin/env python3
"""Data of Figure 2: released-model holdout MSE per round on ACN-Data (median over 10 seeds).

Reads risultati/campagna_v3/*/loss_per_round.csv and prints, for rounds 1-30, the median over seeds of
the released-model holdout MSE for FedAvg without DP, FedAvg with local DP (epsilon = 64 per round),
and FedProx (mu = 0.01) with local DP.

Usage: python scripts/analisi_paper/convergence.py risultati/campagna_v3 out.csv
"""
import csv, glob, re, statistics, sys

root, out_path = sys.argv[1], sys.argv[2]
series = {"fedavg_nodp": "v3_30r_mu0_nodp", "fedavg_dp": "v3_30r_mu0_eps64", "fedprox_0.01_dp": "v3_30r_mu0.01_eps64"}
values = {k: {} for k in series}
for f in glob.glob(root + "/*/loss_per_round.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        m = re.match(r"(.*)_s(\d+)$", r["id"])
        if not m or not r["loss_holdout_rilasciato"]:
            continue
        for k, cell in series.items():
            if m.group(1) == cell:
                values[k].setdefault(int(r["round"]), []).append(float(r["loss_holdout_rilasciato"]))
rows = []
for rnd in range(1, 31):
    rows.append({"round": rnd, **{k: round(statistics.median(values[k][rnd]), 4) for k in series},
                 "seeds": min(len(values[k][rnd]) for k in series)})
with open(out_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
