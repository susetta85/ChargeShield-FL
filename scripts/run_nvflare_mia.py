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

from adapters.acn_dataset import ACNDataset  # noqa: E402
from ml.autoencoder_trainer import AutoencoderTrainer  # noqa: E402

from run_experiments import (  # noqa: E402
    compute_feature_stats,
    enrich_sessions,
    load_config,
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


def load_client_sessions(client_config_path: Path) -> list[dict[str, Any]]:
    """
    Ricostruisce ESATTAMENTE le sessioni (enriched, NON shuffled, ordine
    originale del file) che i client NVFLARE reali hanno visto — leggendo
    "dataset_path" da config_fed_client.json invece di assumerlo.

    FIX 2026-07-22 (review indipendente): la prima stesura di questo script
    caricava le sessioni via load_sessions(cfg) da config/experiment.yaml
    (che combina jpl_2019+jpl_2020) E le shuffle-ava prima dello split
    train/hold-out — un dataset e un ordinamento DIVERSI da quelli che
    chargeshield_executor.py::_setup() usa davvero (SOLO il file indicato
    da "dataset_path" in config_fed_client.json, MAI shuffled, split
    contiguo per indice — vedi quella funzione). run_lira()/run_fedmia()
    ricostruiscono l'appartenenza ai cluster assumendo che train_sessions
    sia nello STESSO ordine/split usato dai client reali (stesso principio
    di run_fl_rounds() nella simulazione) — con un dataset/ordine diverso,
    la ground truth membership (chi ha allenato su cosa) sarebbe scorrelata
    da quella reale, rendendo gli AUC di LiRA/Shadow/Yeom non significativi
    SENZA generare alcun errore visibile (fallimento silenzioso, il più
    pericoloso). Fix: legge lo stesso "dataset_path" dal config del client
    reale e applica la stessa pipeline (ACNDataset → enrich, NESSUNO shuffle).
    """
    import json as _json

    with open(client_config_path) as f:
        client_cfg = _json.load(f)
    dataset_path_str = client_cfg["executors"][0]["executor"]["args"]["dataset_path"]
    dataset_path = PROJECT_ROOT / dataset_path_str

    ds = ACNDataset()
    ds.load(str(dataset_path))
    sessions = [ds.get_sample(i) for i in range(len(ds))]
    sessions = enrich_sessions(sessions)
    logger.info(
        f"Sessioni client ricostruite da {dataset_path.name} (stesso file, stesso "
        f"ordine, nessuno shuffle — mirroring chargeshield_executor.py): {len(sessions)}"
    )
    return sessions


def load_holdout_sessions(holdout_dataset_path: Path) -> list[dict[str, Any]]:
    """
    Sessioni hold-out (non-member) per MIA — devono provenire da un file MAI
    visto da nessun client NVFLARE reale (non da uno split random dello STESSO
    file, che qui non avrebbe senso: load_client_sessions() restituisce l'intero
    dataset del client, senza nessuna porzione riservata — a differenza della
    simulazione, dove main() fa un 80/20 split PRIMA di passare i dati a
    run_fl_rounds(), qui TUTTE le sessioni del file client vengono usate per
    il training reale, quindi un vero hold-out deve venire da un anno/file
    diverso, mai toccato da alcun client — es. acndata_sessions_2020.json
    quando i client hanno usato acndata_sessions_2019.json).
    """
    ds = ACNDataset()
    ds.load(str(holdout_dataset_path))
    sessions = [ds.get_sample(i) for i in range(len(ds))]
    sessions = enrich_sessions(sessions)
    logger.info(f"Sessioni hold-out da {holdout_dataset_path.name}: {len(sessions)}")
    return sessions


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
    parser.add_argument(
        "--client-config", type=Path,
        default=Path("nvflare/jobs/chargeshield_poc/app/config/config_fed_client.json"),
        help=(
            "config_fed_client.json del job NVFLARE reale — usato per leggere "
            "'dataset_path' e ricostruire ESATTAMENTE le sessioni viste dai "
            "client (stesso file, nessuno shuffle, vedi load_client_sessions())."
        ),
    )
    parser.add_argument(
        "--holdout-dataset", type=Path, default=None,
        help=(
            "Dataset MAI usato da alcun client NVFLARE, per il pool non-member "
            "di LiRA/Shadow/Yeom (es. datasets/acn/jpl/acndata_sessions_2020.json "
            "se i client hanno usato il file *_2019.json). Default: stesso nome "
            "file di dataset_path in --client-config con l'anno successivo, se "
            "individuabile automaticamente — altrimenti obbligatorio."
        ),
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

    # ── Dataset: ricostruisce ESATTAMENTE cosa hanno visto i client reali ───
    # FIX 2026-07-22 (review indipendente — vedi load_client_sessions() per il
    # dettaglio del bug originale): NON riusa load_sessions(cfg)/shuffle come
    # nella simulazione — quella pipeline combina un dataset diverso (2019+2020,
    # da config/experiment.yaml) e mischia l'ordine, disallineando la ground
    # truth membership da quella reale. Legge invece "dataset_path" da
    # --client-config (config_fed_client.json del job reale).
    seed = cfg["experiment"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_sessions = load_client_sessions(args.client_config)

    if args.holdout_dataset is not None:
        holdout_path = args.holdout_dataset
    else:
        import json as _json
        with open(args.client_config) as f:
            _client_cfg = _json.load(f)
        _client_dataset = _client_cfg["executors"][0]["executor"]["args"]["dataset_path"]
        # Tentativo automatico: sostituisce l'anno nel nome file con il
        # successivo (es. *_2019.json -> *_2020.json) — entrambi presenti nel
        # repo (datasets/acn/jpl/), mai toccati dallo stesso client.
        import re
        _m = re.search(r"(19|20)\d{2}", _client_dataset)
        if not _m:
            raise ValueError(
                f"Impossibile dedurre --holdout-dataset da {_client_dataset!r} "
                "(nessun anno a 4 cifre nel nome file) — specificalo esplicitamente."
            )
        _year = int(_m.group(0))
        holdout_path = PROJECT_ROOT / _client_dataset.replace(str(_year), str(_year + 1))
        logger.warning(
            f"--holdout-dataset non specificato — dedotto automaticamente: "
            f"{holdout_path} (verifica che sia corretto per il tuo run reale)."
        )
    holdout_sessions = load_holdout_sessions(holdout_path)

    logger.info(
        f"Train (client reali): {len(train_sessions)} sessioni — "
        f"Hold-out (non-member): {len(holdout_sessions)} sessioni"
    )

    _FEATURES = AutoencoderTrainer.CONTINUOUS_FEATURES
    # Stats calcolate su train_sessions (le sessioni realmente usate per il
    # training), NON su holdout — stessa regola anti-leakage di main() nella
    # simulazione, e stessa formula usata in chargeshield_executor.py::_setup()
    # (dove le stats sono calcolate su TUTTO il dataset condiviso prima dello
    # split per-cluster — equivalente, dato che train_sessions qui È l'intero
    # dataset condiviso, non una sua fetta).
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
