#!/usr/bin/env python3
"""
Fase A — Registro delle run (guida scientifica rev. 3, sezione 7).

Ricostruisce dalle evidenze su disco quali esecuzioni esistono davvero,
con quale configurazione e quali metriche, e le mappa sulle RQ.

NON assegna lo stato di validita': quello richiede giudizio umano e va
compilato a mano nella colonna finale del CSV prodotto. Lo script
distingue solo cio' che e' leggibile dai file da cio' che manca.

Uso:
    python3 scripts/build_run_registry.py                 # tabella a schermo
    python3 scripts/build_run_registry.py --csv out.csv   # registro completo
"""
import argparse, csv, glob, json, os, re, sys
from collections import defaultdict

CAMPI_CONFIG = [
    "dp_mode", "epsilon", "delta", "no_dp", "seed", "fl_rounds",
    "epochs", "proximal_mu", "max_grad_norm", "batch_size",
    "hidden_dims", "latent_dim", "norm", "record_dp",
]

def rq_di(cfg, sweep):
    """Mappa una run sulle RQ della guida. Una run puo' servirne piu' di una."""
    rq = []
    # RQ1: effetto DP e durata del training
    if cfg.get("fl_rounds", 0) and (cfg.get("no_dp") or cfg.get("dp_mode")):
        rq.append("RQ1")
    # RQ3: FedAvg vs FedProx — serve sapere mu
    if cfg.get("proximal_mu") is not None:
        rq.append("RQ3?")
    # canary: metodologia, non RQ
    if "canary" in sweep.lower() or cfg.get("canary", {}).get("enabled"):
        rq.append("CTRL")
    return "+".join(rq) if rq else "?"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--csv")
    args = ap.parse_args()

    righe = []
    for f in sorted(glob.glob(os.path.join(args.root, "*", "experiment_*.json"))):
        sweep = os.path.basename(os.path.dirname(f))
        try:
            d = json.load(open(f))
        except Exception as e:
            righe.append({"sweep": sweep, "file": os.path.basename(f),
                          "errore": type(e).__name__})
            continue
        cfg = d.get("config", {}) or {}
        ml  = cfg.get("ml", {}) if isinstance(cfg.get("ml"), dict) else {}
        summ = d.get("summary", {}) or {}
        pr = d.get("per_round", {}) or {}
        ultimo = max(pr, key=int) if pr else None
        mia = pr.get(ultimo, {}).get("mia", {}) if ultimo else {}
        fl  = pr.get(ultimo, {}).get("fl", {}) if ultimo else {}

        r = {"sweep": sweep, "file": os.path.basename(f),
             "mtime": os.path.getmtime(f)}
        for k in CAMPI_CONFIG:
            r[k] = cfg.get(k, ml.get(k))
        # seed dal nome cartella se il config lo riporta male o non lo riporta
        m = re.search(r"seed(\d+)", sweep)
        r["seed_da_nome"] = m.group(1) if m else None
        r["seed_discorde"] = (r["seed_da_nome"] is not None
                              and str(r["seed"]) != r["seed_da_nome"])
        r["n_round"] = len(pr)
        r["lira_composed"] = mia.get("composed_lira_auc_roc")
        r["lira_mean"]     = summ.get("mean_lira_auc_roc")
        r["yeom_mean"]     = summ.get("mean_auc_roc")
        r["shadow_mean"]   = summ.get("mean_shadow_auc_roc")
        r["tpr1"]          = mia.get("composed_tpr_at_fpr_0.01")
        r["loss_finale"]   = fl.get("mean_loss")
        r["canary_raw"]    = mia.get("canary_raw_mse_auc_roc")
        r["rq"] = rq_di(cfg, sweep)
        r["validita"] = ""   # da compilare a mano
        righe.append(r)

    print(f"run trovate: {len(righe)}\n")

    # 1. copertura per RQ1: dp_mode x epsilon x seed
    cov = defaultdict(set)
    for r in righe:
        if "RQ1" not in str(r.get("rq")): continue
        if r.get("canary_raw") is not None: continue      # regime canary a parte
        lbl = "no-DP" if r.get("no_dp") else f"{r.get('dp_mode')},eps={r.get('epsilon')}"
        cov[lbl].add(r.get("seed_da_nome") or r.get("seed"))
    print("RQ1 — copertura regime naturale (seed distinti per cella):")
    for k in sorted(cov, key=str):
        print(f"   {str(k):28s} {len(cov[k])} seed  {sorted(map(str, cov[k]))}")

    # 2. RQ3: esistono run appaiate mu=0 / mu=0.01?
    mu = defaultdict(int)
    for r in righe:
        if r.get("proximal_mu") is not None:
            mu[r["proximal_mu"]] += 1
    print(f"\nRQ3 — valori di proximal_mu osservati: {dict(mu) if mu else 'NESSUNO'}")
    if len(mu) < 2:
        print("   -> nessun confronto appaiato FedAvg/FedProx disponibile")

    # 3. RQ2: esiste una partizione di riferimento IID?
    print("\nRQ2 — partizioni: campo non presente nei config letti."
          "\n   -> verificare a mano se esista un riferimento IID eseguito")

    # 4. seed discordi
    disc = [r for r in righe if r.get("seed_discorde")]
    if disc:
        print(f"\nATTENZIONE: {len(disc)} run con seed nel config diverso dal nome cartella")
        for r in disc[:10]:
            print(f"   {r['sweep']:36s} config={r['seed']} nome={r['seed_da_nome']}")

    if args.csv:
        campi = list(righe[0].keys()) if righe else []
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campi)
            w.writeheader(); w.writerows(righe)
        print(f"\nregistro scritto in {args.csv} — compilare a mano la colonna 'validita'")

if __name__ == "__main__":
    main()
