#!/usr/bin/env python3
# scripts/run_nvflare_mia.py
# ChargeShield-FL — analisi MIA/IDS post-hoc su un run NVFLARE reale (fase 5, 2026-07-22)
#
# STATO: scritto in un sandbox dove nvflare/torch non sono installabili — vedi
# docs/NVFlareIntegration.md. NON eseguito, NON testato (a differenza di
# run_experiments.py, che ha girato su dati reali). Verificare su una macchina
# con le dipendenze reali prima di fidarsi dell'output.
#
# COSA FA QUESTO SCRIPT E PERCHÉ ESISTE SEPARATO DA run_experiments.py:
#   ChargeShieldAggregator (nvflare/jobs/chargeshield_poc/app/custom/
#   chargeshield_aggregator.py) esporta, dopo ogni round di un vero job
#   NVFLARE, un dump pickle (default: experiments/nvflare_fl_results.pkl) con
#   ESATTAMENTE la stessa struttura dati che run_fl_rounds() produce in memoria
#   durante la simulazione single-process (mean_loss, n_participants, updates,
#   raw_updates, raw_global_weights, global_weights — un dict per round).
#
#   run_lira()/run_ids()/run_fedmia()/run_fedmia_shadow()/save_results() sono
#   già scritte per consumare esattamente quella struttura — e sono già state
#   validate empiricamente su nodp-sweep1/dp-sweep1 (5 round di fix per LiRA,
#   vedi la sua docstring). Riscriverle per girare "dal vivo" dentro
#   ChargeShieldAggregator.aggregate(), alla cieca (nessuna esecuzione
#   possibile in questo sandbox), sarebbe un secondo tentativo con altissima
#   probabilità di introdurre bug nuovi e silenziosi.
#
#   Questo script invece CARICA il dump prodotto da un job NVFLARE reale e
#   chiama quelle stesse funzioni, INVARIATE — zero rischio di regressione
#   sulla logica di attacco già validata. È l'equivalente, per NVFLARE, di
#   "main()" in run_experiments.py, con fl_results letto da disco invece che
#   prodotto da run_fl_rounds().
#
# LIMITE NOTO (non risolto qui): l'Aggregator/Executor NVFLARE non hanno un
# equivalente di --no-dp (bypass completo del rumore) — dp_mode è sempre uno
# dei 3 valori. Questo script chiama sempre run_ids()/run_lira() con
# no_dp=False; per un run NVFLARE "senza DP" servirebbe un epsilon molto
# grande in config_fed_server.json/config_fed_client.json (approssimazione,
# non equivalente esatto al bypass di run_fl_rounds()).
#
# Usage:
#   python scripts/run_nvflare_mia.py --config config/experiment.yaml \
#       --fl-results experiments/nvflare_fl_results.pkl
#   python scripts/run_nvflare_mia.py --fl-results experiments/nvflare_fl_results.pkl \
#       --n-shadow 8 --sweep-dir experiments/nvflare-run1

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── Path setup — identico a run_experiments.py ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # per "import run_experiments"

from ml.autoencoder_trainer import AutoencoderTrainer  # noqa: E402

from run_experiments import (  # noqa: E402
    compute_feature_stats,
    enrich_sessions,
    load_config,
    load_sessions,
    normalize_sessions,
    run_fedmia,
    run_fedmia_shadow,
    run_ids,
    run_lira,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_nvflare_mia")


def load_nvflare_fl_results(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """
    Carica il dump pickle prodotto da ChargeShieldAggregator._export_fl_results()
    (fase 5). Restituisce (fl_results, meta) dove fl_results ha la stessa forma
    prodotta da run_fl_rounds() nella simulazione — vedi quella funzione per il
    contratto esatto dei campi.

    Nota: manca sempre l'entry per il round 0 (l'Aggregator non vede mai i pesi
    di init casuale, generati da persistor/shareable_generator prima del round
    1) — run_ids()/run_lira() gestiscono già questo caso (fallback a None).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dump NVFLARE non trovato: {path}. "
            "Deve essere generato da un job NVFLARE reale con ChargeShieldAggregator "
            "(vedi nvflare/jobs/chargeshield_poc/app/config/config_fed_server.json, "
            "campo 'fl_results_export_path') — questo script non lo genera da solo."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)

    meta = payload.get("meta", {})
    rounds = payload.get("rounds", {})
    # Le chiavi sono già int (scritte così da ChargeShieldAggregator) — nessuna
    # conversione da stringa necessaria (a differenza del JSON di fase 4).
    fl_results = {int(r): data for r, data in rounds.items()}
    logger.info(
        f"Caricato dump NVFLARE: {len(fl_results)} round, "
        f"dp_mode={meta.get('dp_mode')!r}, epsilon={meta.get('epsilon')!r}"
    )
    return fl_results, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ChargeShield-FL — esegue LiRA/Shadow/Yeom/IDS su un dump prodotto "
            "da un vero job NVFLARE (fase 5), riusando run_experiments.py invariato."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument(
        "--fl-results", type=Path,
        default=Path("experiments/nvflare_fl_results.pkl"),
        help="Path al dump pickle prodotto da ChargeShieldAggregator (fase 5).",
    )
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help="Directory di output (come in run_experiments.py --sweep-dir).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-shadow", type=int, default=None)
    parser.add_argument("--shadow-epochs-cap", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ChargeShield-FL — Analisi post-hoc su dump NVFLARE (fase 5)")
    logger.info("=" * 60)

    cfg = load_config(args.config, {"epsilon": None, "rounds": None})
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    fl_results, meta = load_nvflare_fl_results(args.fl_results)

    # dp_mode effettivamente usato dal job NVFLARE (registrato nel dump da
    # ChargeShieldAggregator) — sovrascrive quello di config/experiment.yaml,
    # che descrive la simulazione single-process, non necessariamente lo
    # stesso run NVFLARE che ha prodotto questo dump.
    dp_mode = meta.get("dp_mode", cfg["experiment"].get("dp_mode", "dp-fedavg"))
    cfg["experiment"]["dp_mode"] = dp_mode
    cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_nvflare"
    if meta.get("epsilon") is not None:
        cfg["experiment"]["epsilon"] = meta["epsilon"]
    if meta.get("delta") is not None:
        cfg["experiment"]["delta"] = meta["delta"]
    if meta.get("max_grad_norm") is not None:
        cfg["experiment"]["max_grad_norm"] = meta["max_grad_norm"]

    # LIMITE NOTO (vedi docstring modulo): nessun bypass --no-dp lato NVFLARE.
    no_dp = False

    # ── Dataset: stessa pipeline di main() in run_experiments.py ────────────
    # VERIFY: deve combaciare con cosa hanno visto davvero i client NVFLARE.
    # Oggi (fase 1) ogni client carica lo stesso file condiviso e affetta per
    # indice — stesso schema fake-per-client di run_fl_rounds() — quindi
    # riusare load_sessions()/lo stesso split è corretto PER ORA. Se in futuro
    # ogni client avrà accesso reale al proprio dataset (vedi limitazioni in
    # docs/NVFlareIntegration.md), questa pipeline andrà rifatta per riflettere
    # quella realtà (niente più "un solo file, split unico").
    sessions = load_sessions(cfg)
    sessions = enrich_sessions(sessions)
    logger.info(f"Sessioni dopo enrichment: {len(sessions)}")

    seed = cfg["experiment"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(sessions)
    split = max(1, int(len(sessions) * 0.8))
    train_sessions = sessions[:split]
    holdout_sessions = sessions[split:]
    logger.info(f"Split — train: {len(train_sessions)}, hold-out: {len(holdout_sessions)}")

    _FEATURES = AutoencoderTrainer.CONTINUOUS_FEATURES
    feature_stats = compute_feature_stats(train_sessions, _FEATURES)
    train_sessions = normalize_sessions(train_sessions, feature_stats, _FEATURES)
    holdout_sessions = normalize_sessions(holdout_sessions, feature_stats, _FEATURES)

    # ── IDS — stessa funzione, stessa logica, usata dalla simulazione ───────
    ids_results: dict = {}
    try:
        ids_results = run_ids(cfg, fl_results, no_dp=no_dp)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run_ids() fallita: {exc}", exc_info=True)

    # ── Yeom (run_fedmia) + Shadow (run_fedmia_shadow) — sul modello globale ─
    mia_results: dict = {}
    try:
        mia_results = run_fedmia(cfg, train_sessions, holdout_sessions, fl_results)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run_fedmia() fallita: {exc}", exc_info=True)

    try:
        shadow_results = run_fedmia_shadow(cfg, train_sessions, holdout_sessions, fl_results)
        for rnd, data in shadow_results.items():
            mia_results.setdefault(rnd, {}).update(data)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run_fedmia_shadow() fallita: {exc}", exc_info=True)

    # ── LiRA — l'attacco primario, sul singolo update per-client ────────────
    n_shadow = args.n_shadow if args.n_shadow is not None else cfg.get("lira", {}).get("n_shadow", 8)
    try:
        lira_results = run_lira(
            cfg, train_sessions, holdout_sessions, fl_results,
            n_shadow=n_shadow, shadow_epochs_cap=args.shadow_epochs_cap,
            no_dp=no_dp, dp_mode=dp_mode,
        )
        for rnd, data in lira_results.items():
            mia_results.setdefault(rnd, {}).update(data)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run_lira() fallita: {exc}", exc_info=True)

    result_file = save_results(cfg, mia_results, ids_results, fl_results=fl_results, sweep_dir=args.sweep_dir)
    logger.info(f"Analisi NVFLARE completata — risultati in {result_file}")


if __name__ == "__main__":
    main()
