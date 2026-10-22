#!/usr/bin/env python3
"""
Coda v2e (2026-10-19): da lanciare DOPO la coda v2d sul Mac principale.

Blocco A - RMIA offline con LiRA attiva (lira2.py, log_lr_rmia_offline): ACN, celle
dell'ascissa in epsilon scelte dalla v2c (eps 4, 16, 256 con --dp-mode local) piu' senza DP
mu 0.01, 5 seed della v2c. Stessi seed e stessi config della v2c: il punteggio online e
quello offline si confrontano sulle stesse federazioni. Shadow v2 saltato (gia' in v2d).
Le loss degli shadow non sono salvate, quindi l'offline si ricalcola solo rieseguendo.

Blocco B - 30 round per la questione RQ3 (utilita' sola, --skip-attacks lira,shadow):
quattro bracci (FedAvg e FedProx, con e senza DP per client a epsilon 64, C = 1,
dp-fedavg) su 10 seed: i 5 della v2c piu' 5 nuovi. Stessi config dei 30 round gia'
eseguiti, senza --common-init (come quelle run). Rerun sul Mac principale: stesso
codice dell'utilita'; lira2.py cambia solo nella parte di attacco.

Ore: stime da verificare con la prima run cronometrata (blocco B: 30 round senza LiRA).

Uso: python3 scripts/genera_coda_v2e.py --out <cartella>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
from genera_coda_v2 import SEEDS5, Coda  # noqa: E402

TEST = ("{PY} -m pytest tests/test_rmia_offline.py tests/test_lira2.py tests/test_rmia.py -q")
SEEDS_NUOVI = [2025, 2026, 3141, 2718, 1618]
SEEDS10 = list(SEEDS5) + SEEDS_NUOVI
ORE_LIRA_1W = 9      # come la curva in epsilon della v2c (ore con 1 worker, dopo divisione 0.9 h)
ORE_30R_1W = 1.0     # stima: 30 round senza LiRA; da misurare
FLAG_30R = "--rounds 30 --threads 1 --skip-attacks lira,shadow"

# Blocco B: 30 round, utilita', quattro bracci
BRACCI_30R = [
    ("mu0_nodp", "experiment_rq3_mu0", "--no-dp"),
    ("mu0.01_nodp", "experiment", "--no-dp"),
    ("mu0_eps64", "experiment_rq3_mu0_eps64", ""),
    ("mu0.01_eps64", "experiment_rq1_eps64", ""),
]


def costruisci() -> Coda:
    c = Coda("coda_v2e_mac.csv", 10)
    c.cmd("test_codice", 0, 0.2, 2, TEST)
    # Blocco A
    for e in (4, 16, 256):
        for seed in SEEDS5:
            c.run(f"v2e_rmia_off_acn_eps{e}_s{seed}", 1, ORE_LIRA_1W, 4,
                  f"experiment_rq1_eps{e}", "--dp-mode local", seed, skip="shadow", dopo="test_codice")
    for seed in SEEDS5:
        c.run(f"v2e_rmia_off_acn_nodp_s{seed}", 1, ORE_LIRA_1W, 4,
              "experiment", "--no-dp", seed, skip="shadow", dopo="test_codice")
    # Blocco B
    for seed in SEEDS10:
        for nome, cfg, flags in BRACCI_30R:
            id_ = f"v2e_30r_{nome}_s{seed}"
            comando = (f"{{PY}} scripts/run_experiments.py --config config/{cfg}.yaml {flags} "
                       f"{FLAG_30R} --seed {seed} --sweep-dir {{ESITI}}/{id_}").replace("  ", " ")
            c.cmd(id_, 2, ORE_30R_1W, 3, comando, dopo="test_codice")
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(QUI))
    a = ap.parse_args()
    c = costruisci()
    ids = [r["id"] for r in c.righe if r["id"] != "test_codice"]
    assert len(ids) == len(set(ids)), "id duplicati"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c.scrivi(out)
    print(f"run in coda v2e: {len(ids)} (A: {sum('rmia_off' in i for i in ids)}, B: {sum('30r' in i for i in ids)})")


if __name__ == "__main__":
    main()
