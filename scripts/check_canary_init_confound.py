#!/usr/bin/env python3
"""
Controllo di confondente: i canary si separano dai non-membri ANCHE
prima di qualunque addestramento?

Se un modello a inizializzazione casuale li separa gia', il segnale
osservato nei run canary e' difficolta' intrinseca di ricostruzione
(magnitudine delle feature), non appartenenza.

Uso:
    python3 scripts/check_canary_init_confound.py \
        --config config/experiment_canary_positive_control.yaml \
        --seed 42

Criterio di lettura:
    AUC_init ~= 0.50  -> nessun confondente, il segnale post-training e' reale
    AUC_init >  0.60  -> confondente confermato, il positive control non e'
                         valido nella forma attuale
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.autoencoder import Autoencoder  # noqa: E402

# ---------------------------------------------------------------------
# DA VERIFICARE: questi due import devono puntare alle funzioni reali di
# run_experiments.py. Se i nomi differiscono, sostituirli qui: sono le
# uniche due dipendenze dal runner.
# ---------------------------------------------------------------------
from scripts.run_experiments import (  # noqa: E402
    load_sessions_for_sites,
    inject_canaries,
)


def auc_roc(scores_pos, scores_neg):
    """AUC via rank statistic (Mann-Whitney U). Alto score = predetto membro.

    Qui usiamo la loss NEGATA come score, perche' loss bassa => membro.
    """
    s = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # media dei ranghi sui ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def session_losses(model, sessions, feature_keys):
    """MSE di ricostruzione per sessione, modello in eval mode."""
    X = np.array(
        [[float(s[k]) for k in feature_keys] for s in sessions], dtype=np.float32
    )
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X)
        out = model(t)
        if isinstance(out, tuple):
            out = out[0]
        mse = ((out - t) ** 2).mean(dim=1).numpy()
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-init", type=int, default=20,
                    help="quante inizializzazioni casuali mediare")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    feature_keys = cfg["ml"]["features"]
    sites = cfg["data"]["sites"]

    train, holdout = load_sessions_for_sites(sites, seed=args.seed)
    train, holdout, canary_meta = inject_canaries(
        train, holdout, cfg["canary"], seed=args.seed
    )

    member_ids = set(canary_meta["member_ids"])
    nonmember_ids = set(canary_meta["nonmember_ids"])
    members = [s for s in train if id(s) in member_ids]
    nonmembers = [s for s in holdout if id(s) in nonmember_ids]

    print(f"canary membri     : {len(members)}")
    print(f"canary non-membri : {len(nonmembers)}")
    print(f"feature           : {feature_keys}")
    print()

    aucs = []
    for i in range(args.n_init):
        torch.manual_seed(args.seed + i)
        model = Autoencoder(input_dim=len(feature_keys))
        lm = session_losses(model, members, feature_keys)
        ln = session_losses(model, nonmembers, feature_keys)
        # loss bassa => membro, quindi score = -loss
        a = auc_roc(-lm, -ln)
        aucs.append(a)
        if i == 0:
            print(f"loss media membri     (init 0): {lm.mean():.6f}")
            print(f"loss media non-membri (init 0): {ln.mean():.6f}")
            print(f"rapporto                      : {ln.mean()/lm.mean():.2f}x")
            print()

    aucs = np.array(aucs)
    print(f"AUC a init casuale: media {aucs.mean():.4f}  "
          f"std {aucs.std():.4f}  min {aucs.min():.4f}  max {aucs.max():.4f}")
    print()
    if aucs.mean() > 0.60 or aucs.mean() < 0.40:
        print(">>> CONFONDENTE CONFERMATO. Un modello mai addestrato separa i")
        print(">>> canary dai non-membri: il positive control nella forma")
        print(">>> attuale misura difficolta' di ricostruzione, non")
        print(">>> appartenenza. Servono gemelli veri (copie esatte dei")
        print(">>> template tenute fuori dal training).")
    else:
        print(">>> Nessun confondente rilevabile a inizializzazione casuale.")
        print(">>> Il segnale post-training e' attribuibile all'addestramento.")


if __name__ == "__main__":
    main()
