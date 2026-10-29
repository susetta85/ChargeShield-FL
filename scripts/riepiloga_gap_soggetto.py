"""Riepilogo dei JSON di gap_soggetto.py: una riga per dataset e seed, piu' la media.

Uso: python scripts/riepiloga_gap_soggetto.py esiti/gap_soggetto [--csv uscita.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# AUC tutte_limite_superiore_grezzo delle code (Mac di Domenico per Scozia), per il confronto
# quando il JSON di mia_soggetto non e' disponibile sulla macchina del controllo.
RIF_SC3 = {42: 0.43069263691152065, 123: 0.45715783037819896, 456: 0.5337584068477597,
           789: 0.5627565726264303, 1234: 0.43802952222901564}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cartella")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    righe = []
    for f in sorted(Path(a.cartella).glob("gap_soggetto_*.json")):
        d = json.loads(f.read_text())
        fin = d["per_round"][str(d["rounds"])]
        base = d["per_round"]["0"]
        ls = fin["loss_sessioni"]
        dataset = "Scozia" if "sc3" in d["config"] else "ACN"
        conf = d.get("confronto_mia_soggetto")
        auc_coda = conf["auc_json"] if conf else (RIF_SC3.get(d["seed"]) if dataset == "Scozia" else None)
        u = fin["auc_unita_tutte_grezzo"]
        righe.append({
            "dataset": dataset, "seed": d["seed"],
            "mediana_membri": ls["membri"]["mediana"], "mediana_non_membri": ls["non_membri"]["mediana"],
            "mediana_server": ls["server"]["mediana"],
            "divario_mediane_rel": (ls["non_membri"]["mediana"] - ls["membri"]["mediana"])
                                   / ls["non_membri"]["mediana"],
            "divario_mediane_rel_server": (ls["server"]["mediana"] - ls["membri"]["mediana"])
                                          / ls["server"]["mediana"],
            "auc_sessione": fin["auc_sessione"]["auc"],
            "auc_sessione_ci": fin["auc_sessione"]["ci95"],
            "auc_unita": u["auc"], "auc_unita_ci": u["ci95"], "auc_unita_p": u.get("p"),
            "auc_unita_round0": base["auc_unita_tutte_grezzo"]["auc"],
            "auc_coda": auc_coda,
            "differenza_dalla_coda": (u["auc"] - auc_coda) if auc_coda is not None else None,
        })
    righe.sort(key=lambda r: (r["dataset"], r["seed"]))
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    print(f"{'dataset':7s} {'seed':>5s} {'med m':>9s} {'med nm':>9s} {'med srv':>9s} {'div nm':>7s} "
          f"{'div srv':>7s} {'AUC sess [IC]':>20s} {'AUC unita [IC]':>20s} {'r0':>6s} {'coda':>6s} {'diff':>7s}")
    for r in righe:
        print(f"{r['dataset']:7s} {r['seed']:5d} {r['mediana_membri']:9.2e} {r['mediana_non_membri']:9.2e} "
              f"{r['mediana_server']:9.2e} {r['divario_mediane_rel']:7.3f} {r['divario_mediane_rel_server']:7.3f} "
              f"{r['auc_sessione']:.3f} [{r['auc_sessione_ci'][0]:.3f},{r['auc_sessione_ci'][1]:.3f}] "
              f"{r['auc_unita']:.3f} [{r['auc_unita_ci'][0]:.3f},{r['auc_unita_ci'][1]:.3f}] "
              f"{f3(r['auc_unita_round0']):>6s} {f3(r['auc_coda']):>6s} {f3(r['differenza_dalla_coda']):>7s}")
    for ds in sorted({r["dataset"] for r in righe}):
        rr = [r for r in righe if r["dataset"] == ds]
        print(f"{ds:7s} media su {len(rr)} seed: divario mediane non membri {np.mean([r['divario_mediane_rel'] for r in rr]):.3f}, "
              f"server {np.mean([r['divario_mediane_rel_server'] for r in rr]):.3f}; "
              f"AUC sessione {np.mean([r['auc_sessione'] for r in rr]):.3f}; AUC unita' {np.mean([r['auc_unita'] for r in rr]):.3f}; "
              f"seed con IC dell'AUC per unita' sopra 0,5: {sum(r['auc_unita_ci'][0] > 0.5 for r in rr)}/{len(rr)}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(righe[0].keys()))
            w.writeheader()
            w.writerows(righe)


if __name__ == "__main__":
    main()
