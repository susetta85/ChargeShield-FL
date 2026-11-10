#!/usr/bin/env python3
"""Paired utility tests reported in the paper (Tables IV-V, Section V-B).

Reads every risultati/campagna_v3/*/riepilogo_run.csv, pairs FedAvg and FedProx
runs by seed (same data split and initialization) and applies an exact sign-flip
permutation test on the per-seed loss differences (FedProx - FedAvg), enumerating
all 2^n sign assignments. Holm-Bonferroni correction is applied within the
families described in the paper.

Usage:
    python scripts/analisi_paper/utility_tests.py risultati/campagna_v3 out.json
"""
import csv, glob, itertools, json, re, sys
import numpy as np

root, out_path = sys.argv[1], sys.argv[2]
L = {}
for f in glob.glob(root + "/*/riepilogo_run.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        m = re.match(r"(.*)_s(\d+)$", r["id"])
        if m and r["loss_holdout_rilasciato"]:
            L.setdefault(m.group(1), {})[int(m.group(2))] = float(r["loss_holdout_rilasciato"])

# Released-model loss at round 5 of the replication runs (v3k, new seeds), from loss_per_round.csv.
R5 = {}
for f in glob.glob(root + "/*/loss_per_round.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["id"].startswith("v3k_") and r["round"] == "5" and r["loss_holdout_rilasciato"]:
            m = re.match(r"(.*)_s(\d+)$", r["id"])
            R5.setdefault(m.group(1) + "@r5", {})[int(m.group(2))] = float(r["loss_holdout_rilasciato"])
L.update(R5)

_signs = {}
def sign_flip_p(d, alternative="two-sided"):
    """Exact sign-flip test on the mean difference; 'less' = FedProx lower loss."""
    d = np.asarray(d, dtype=float); n = len(d)
    S = _signs.setdefault(n, np.array(list(itertools.product([1, -1], repeat=n)), dtype=float))
    means = S @ d / n; obs = d.mean()
    if alternative == "two-sided":
        return float(np.mean(np.abs(means) >= abs(obs) - 1e-15))
    return float(np.mean(means <= obs + 1e-15))

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj

def compare(base, prox, alternative="two-sided", extra_base=None, extra_prox=None):
    a = dict(L[base]); b = dict(L[prox])
    if extra_base: a.update(L[extra_base])
    if extra_prox: b.update(L[extra_prox])
    seeds = sorted(set(a) & set(b))
    A = np.array([a[s] for s in seeds]); B = np.array([b[s] for s in seeds])
    return dict(n=len(seeds), fedavg=A.mean(), fedprox=B.mean(), ratio_of_means=B.mean() / A.mean(),
                median_ratio=float(np.median(B / A)), seeds_fedprox_lower=int((B < A).sum()),
                p=sign_flip_p(B - A, alternative))

def family(name, items):
    ps = [x["p"] for _, x in items]
    for (key, x), a in zip(items, holm(ps)):
        x["p_holm"] = a; x["family"] = name; out[key] = x

out = {}
# Table IV, three-client federations: one family per configuration block (two-sided).
for block in ["nodp", "clip", "noise", "eps64"]:
    items = []
    for ds in ["acn", "sc3", "acn4f"]:
        for mu in ["0.01", "0.1"]:
            b, p = f"v3_{ds}_{block}_mu0", f"v3_{ds}_{block}_mu{mu}"
            if b in L and p in L:
                items.append((f"{block}|{ds}|mu{mu}", compare(b, p)))
    family(f"three-client {block}", items)
# Table IV, 32-client Scotland: no DP (10 seeds) and full DP (20 seeds, two batches).
family("sc32 nodp", [(f"nodp|sc32|mu{mu}", compare("v3m_sc32_nodp_mu0", f"v3m_sc32_nodp_mu{mu}")) for mu in ["0.01", "0.1"]])
family("sc32 eps64", [
    ("eps64|sc32|mu0.01", compare("v3m_sc32_eps64_mu0", "v3m_sc32_eps64_mu0.01", "two-sided", "v3r_sc32_eps64_mu0", "v3t_sc32_eps64_mu0.01")),
    ("eps64|sc32|mu0.1", compare("v3m_sc32_eps64_mu0", "v3m_sc32_eps64_mu0.1", "two-sided", "v3r_sc32_eps64_mu0", "v3r_sc32_eps64_mu0.1")),
])
# Round-5 replication on 10 new seeds: loss at round 5 (one-sided, direction fixed before the runs).
family("round-5 replication", [(f"r5|{ds}|mu{mu}", compare(f"v3k_{ds}_eps64_mu0@r5", f"v3k_{ds}_eps64_mu{mu}@r5", "less"))
                                for ds in ["acn", "sc3", "acn4f"] for mu in ["0.01", "0.1"]])
# Table V: clipping threshold at fixed absolute noise (one-sided).
family("clipping threshold", [(f"clip|C{C}", compare(f"v3p_acn_rC1_C{C}_mu0", f"v3p_acn_rC1_C{C}_mu0.01", "less"))
                               for C in ["0.25", "0.5", "2"]])
# Round 30 on ACN-Data (two-sided, single comparison).
out["r30|acn|mu0.01"] = compare("v3_30r_mu0_eps64", "v3_30r_mu0.01_eps64")

for k, x in out.items():
    print(f"{k:22s} n={x['n']:2d} FedAvg={x['fedavg']:.4g} FedProx={x['fedprox']:.4g} "
          f"ratio={x['ratio_of_means']:.2f} lower={x['seeds_fedprox_lower']}/{x['n']} "
          f"p={x['p']:.4f} holm={x.get('p_holm', float('nan')):.4f}")
json.dump(out, open(out_path, "w"), indent=1)
