#!/usr/bin/env python3
"""Subject (user / charging-point) canary summary reported in the paper (Tables III and VI).

Reads every risultati/campagna_v3/*/canary_persona.csv. A configuration is the result
file name without its seed suffix (e.g. canary_persona_indotto_central_eps64_ri_c30_d0.5_r3_e1000_AB)
plus the evaluated round. For the aggregator ("client_*" columns) and the released model
("globale_*" columns) it reports:
  - paired AUC: mean over seeds of the controlled paired AUC (both arms), and the number
    of seeds with a one-sided Mann-Whitney p < 0.05;
  - single-federation AUC: mean over seeds of (sum of the two arm AUCs) / 2, i.e. candidates
    ranked by loss within one federation, and a one-sided exact sign-flip test across seeds
    on AUC - 0.5.

Usage:
    python scripts/analisi_paper/canary_summary.py risultati/campagna_v3 out.csv
"""
import csv, glob, itertools, re, sys
import numpy as np

root, out_path = sys.argv[1], sys.argv[2]
groups = {}
for f in glob.glob(root + "/*/canary_persona.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        cfg = re.sub(r"_s\d+\.json$", "", r["file"])
        groups.setdefault((cfg, int(r["round"])), []).append(r)

def across_seed_p(x):
    x = np.asarray(x, dtype=float); n = len(x)
    S = np.array(list(itertools.product([1, -1], repeat=n)), dtype=float)
    return float(np.mean(S @ x / n >= x.mean() - 1e-15))

rows = []
for (cfg, rnd), rs in sorted(groups.items()):
    rec = {"configuration": cfg, "round": rnd, "seeds": len({r["seed"] for r in rs})}
    for surface, col in [("aggregator", "client_training"), ("released", "globale_training")]:
        paired = [float(r[col + "_appaiata_auc"]) for r in rs if r[col + "_appaiata_auc"]]
        pvals = [float(r[col + "_appaiata_p"]) for r in rs if r[col + "_appaiata_p"]]
        single = [float(r[col + "_somma_AB"]) / 2 for r in rs if r[col + "_somma_AB"]]
        rec[f"{surface}_paired_auc"] = round(float(np.mean(paired)), 4) if paired else ""
        rec[f"{surface}_paired_significant_seeds"] = sum(p < 0.05 for p in pvals)
        rec[f"{surface}_single_federation_auc"] = round(float(np.mean(single)), 4) if single else ""
        rec[f"{surface}_single_federation_p"] = round(across_seed_p(np.array(single) - 0.5), 4) if 1 < len(single) <= 20 else ""
    rows.append(rec)

with open(out_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    if r["round"] == 3:
        print(f"{r['configuration'][:62]:62s} r{r['round']} n={r['seeds']:2d} "
              f"aggr paired={r['aggregator_paired_auc']} ({r['aggregator_paired_significant_seeds']}) "
              f"single={r['aggregator_single_federation_auc']} p={r['aggregator_single_federation_p']} | "
              f"released single={r['released_single_federation_auc']} p={r['released_single_federation_p']}")
