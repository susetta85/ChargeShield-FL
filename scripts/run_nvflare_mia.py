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
#   NVFLARE, un dump pickle (experiments/nvflare_fl_results_<timestamp>.pkl —
#   nome univoco per run dal 2026-07-24, per non sovrascrivere run precedenti;
#   prima di quel fix era un nome fisso) con ESATTAMENTE la stessa struttura
#   dati che run_fl_rounds() produce in memoria durante la simulazione
#   single-process (mean_loss, n_participants, updates, raw_updates,
#   raw_global_weights, global_weights — un dict per round).
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
#   python scripts/run_nvflare_mia.py --config config/experiment.yaml
#       # (senza --fl-results: usa il dump nvflare_fl_results_*.pkl più recente)
#   python scripts/run_nvflare_mia.py \
#       --fl-results experiments/nvflare_fl_results_20260724_162431.pkl \
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
    run_ids,
    run_registered_attacks,
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


def load_client_sessions(
    client_config_path: Path,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, list[int]] | None, list[dict[str, Any]]]:
    """
    Ricostruisce ESATTAMENTE le sessioni (enriched) che i client NVFLARE
    reali hanno visto in TRAINING — leggendo "dataset_path" da
    config_fed_client.json invece di assumerlo — e, separatamente, l'insieme
    hold-out riservato da ChargeShieldExecutor._setup() e MAI passato al
    training.

    FIX 2026-08-03 (bug reale trovato al primo tentativo di eseguire questo
    script su un dump NVFLARE reale completato: ogni sito aveva già usato
    TUTTI gli anni ACN-Data scaricati per il training — nessun hold-out
    genuino esisteva su disco, e scaricarne uno aggiuntivo si è rivelato non
    praticabile, vedi chargeshield_executor.py). Invece di richiedere un file
    esterno mai visto da alcun client, questa funzione ricostruisce lo STESSO
    split seed-based 80/20 che ChargeShieldExecutor._setup() applica ora
    prima del training (stesso ordine di caricamento file, stessa
    enrichment, stesso seed) — il 20% escluso dal training è per costruzione
    lo stesso hold-out che il client reale ha riservato, senza bisogno di
    alcun dato esterno. Richiede che il dump NVFLARE analizzato provenga da
    un job lanciato DOPO questo fix — un dump del vecchio comportamento
    (100% training, nessun hold-out riservato) non ha un hold-out genuino da
    ricostruire, qualunque split si applichi qui sarebbe usato anche nel
    training reale (data leakage silenzioso).

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
    reale e applica la stessa pipeline (ACNDataset → enrich). Nota
    (aggiornata 2026-08-03): all'epoca di questo fix non c'era ancora alcuno
    shuffle in nessuno dei due percorsi — lo split seed-based 80/20 introdotto
    sopra è arrivato dopo, e ha reso questa frase storica ("NESSUNO shuffle")
    non più accurata: oggi lo shuffle c'è, deliberatamente, in entrambi i
    file, con lo stesso seed.

    FIX 2026-07-22 (review indipendente pre-push, 3 siti reali): questa
    funzione assumeva ancora "dataset_path" = UN singolo file JSON condiviso
    da tutti i client (design pre-2026-07-22). Da quando
    chargeshield_executor.py carica invece TUTTI i file sotto la directory
    PADRE "datasets/acn/<cluster_id>/" (un cluster_id per sito reale), questa
    funzione andava in `IsADirectoryError` aprendo quella directory come se
    fosse un file — mai eseguita/testata da nessuno nel frattempo (nvflare/
    torch non disponibili in questo sandbox), quindi il bug non era stato
    notato. Fix: se "dataset_path" è una directory, la tratta come la radice
    multi-sito (stessa struttura di chargeshield_executor.py) e carica OGNI
    sottodirectory sito trovata (non solo quella del cluster_id di default
    nel config — quel valore è comunque solo un fallback per-sito, mai letto
    a runtime da un vero deployment NVFLARE, vedi commento in
    config_fed_client.json), costruendo anche un `cluster_membership`
    (site_name -> lista indici) — stesso schema di
    scripts/run_experiments.py::group_indices_by_site() — così run_lira()
    può raggruppare per sito reale invece di ricadere silenziosamente sul
    fallback a 4 fette fittizie (comportamento di default se
    cluster_membership=None). Se invece "dataset_path" è ancora un file
    singolo (design pre-2026-07-22, o un dump storico), il comportamento
    originale è preservato invariato e restituisce cluster_membership=None.
    """
    import json as _json

    with open(client_config_path) as f:
        client_cfg = _json.load(f)
    dataset_path_str = client_cfg["executors"][0]["executor"]["args"]["dataset_path"]
    dataset_path = PROJECT_ROOT / dataset_path_str

    if dataset_path.is_dir():
        site_dirs = sorted(p for p in dataset_path.iterdir() if p.is_dir())
        if not site_dirs:
            raise ValueError(
                f"'dataset_path' ({dataset_path}) è una directory ma non contiene "
                "nessuna sottodirectory sito (es. caltech/jpl/office1) — non è "
                "possibile ricostruire le sessioni dei client reali."
            )
        train_sessions: list[dict[str, Any]] = []
        holdout_sessions: list[dict[str, Any]] = []
        cluster_membership: dict[str, list[int]] = {}
        for site_dir in site_dirs:
            json_files = sorted(site_dir.glob("*.json"))
            if not json_files:
                logger.warning(f"Nessun file .json in {site_dir} — sito saltato.")
                continue
            site_sessions: list[dict[str, Any]] = []
            for jf in json_files:
                ds = ACNDataset()
                ds.load(str(jf))
                site_sessions.extend(ds.get_sample(i) for i in range(len(ds)))
            # Enrichment PRIMA dello split, per-sito — stesso ordine di
            # operazioni di chargeshield_executor.py::_setup() (2026-08-03).
            site_sessions = enrich_sessions(site_sessions)
            # FIX 2026-08-03: stesso split seed-based 80/20 di
            # ChargeShieldExecutor._setup() — vedi docstring della funzione.
            # Applicato PER SITO con lo stesso seed, perché ogni client reale
            # fa lo split indipendentemente sulle proprie sole sessioni.
            random.seed(seed)
            random.shuffle(site_sessions)
            split = max(1, int(len(site_sessions) * 0.8))
            site_train = site_sessions[:split]
            site_holdout = site_sessions[split:]

            start = len(train_sessions)
            train_sessions.extend(site_train)
            cluster_membership[site_dir.name] = list(range(start, len(train_sessions)))
            holdout_sessions.extend(site_holdout)

        logger.info(
            f"Sessioni multi-sito ricostruite da {dataset_path} — "
            f"{len(train_sessions)} train / {len(holdout_sessions)} hold-out "
            f"su {len(cluster_membership)} siti reali (seed={seed}): "
            + ", ".join(f"{k}={len(v)}" for k, v in cluster_membership.items())
        )
        return train_sessions, cluster_membership, holdout_sessions

    ds = ACNDataset()
    ds.load(str(dataset_path))
    sessions = [ds.get_sample(i) for i in range(len(ds))]
    sessions = enrich_sessions(sessions)
    logger.info(
        f"Sessioni client ricostruite da {dataset_path.name} (stesso file, stesso "
        f"ordine, nessuno shuffle — design a file singolo pre-2026-07-22, hold-out "
        f"da passare esplicitamente via --holdout-dataset): {len(sessions)}"
    )
    return sessions, None, []


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
        default=None,
        help=(
            "Path al dump pickle prodotto da ChargeShieldAggregator (fase 5). "
            "Se omesso, usa il file 'experiments/nvflare_fl_results_*.pkl' più "
            "recente (fix 2026-07-24: da quando ogni run NVFLARE genera un "
            "nome univoco con timestamp — vedi chargeshield_aggregator.py — "
            "non esiste più un unico 'nvflare_fl_results.pkl' fisso da usare "
            "come default)."
        ),
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
            "Override esplicito del pool non-member per LiRA/Shadow/Yeom. "
            "Di default (2026-08-03) NON serve passarlo: load_client_sessions() "
            "ricostruisce da sola l'hold-out 80/20 che ChargeShieldExecutor "
            "riserva ora prima del training (stesso seed, stesso ordine di "
            "caricamento). Passa questo argomento solo se vuoi usare un file "
            "diverso (es. un anno genuinamente mai scaricato)."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-shadow", type=int, default=None)
    parser.add_argument("--shadow-epochs-cap", type=int, default=None)
    return parser.parse_args()


def _round_count_or_reason(path: Path) -> str:
    """Legge un dump pickle NVFLARE e ritorna il numero di round contenuti
    (o il motivo per cui non è stato possibile leggerlo) — usato solo per
    logging diagnostico, non cambia quale file viene selezionato."""
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return f"{len(payload)} round"
    except Exception as e:  # file corrotto/troncato/formato inatteso
        return f"illeggibile ({e.__class__.__name__})"


def _resolve_fl_results_path(explicit: Path | None) -> Path:
    """Se --fl-results non è passato esplicitamente, sceglie il dump pickle
    NVFLARE più recente per data di modifica — fix 2026-07-24: da quando
    ChargeShieldAggregator inserisce un timestamp univoco nel nome (per non
    sovrascrivere i risultati di run precedenti, vedi il suo __init__), non
    esiste più un singolo 'nvflare_fl_results.pkl' fisso da usare come
    default.

    Fix 2026-07-24 (review indipendente, stesso giorno): la sola mtime NON
    distingue un run completato da uno interrotto a metà — ChargeShieldAggregator
    riscrive lo stesso file dopo OGNI round, quindi un run avviato dopo uno
    completato ma interrotto al round 2 ha comunque una mtime più recente, e
    verrebbe scelto in silenzio al posto del run completo precedente. Non
    risolto scegliendo un file diverso (quale euristica sostituire a "il più
    recente" è una scelta che cambia il comportamento di default, non ovvia
    da prendere alla cieca) — mitigato invece rendendo sempre visibile nel log
    il numero di round di ogni candidato, cosicché un run interrotto sia
    immediatamente riconoscibile prima di fidarsi dei risultati a valle."""
    if explicit is not None:
        return explicit
    candidates = sorted(
        Path("experiments").glob("nvflare_fl_results_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Nessun 'experiments/nvflare_fl_results_*.pkl' trovato — esegui prima "
            "'make nvflare-sim-smoke'/'make nvflare-sim', oppure passa --fl-results "
            "esplicitamente."
        )
    chosen = candidates[0]
    logger.info(
        f"Dump NVFLARE selezionato (più recente per mtime): {chosen.name} — "
        f"{_round_count_or_reason(chosen)}. Verifica che il conteggio round sia "
        "quello atteso prima di usare questi risultati — un run interrotto ha "
        "comunque mtime più recente di uno completato prima."
    )
    if len(candidates) > 1:
        others = ", ".join(
            f"{p.name} ({_round_count_or_reason(p)})" for p in candidates[1:]
        )
        logger.warning(
            f"Trovati {len(candidates)} dump NVFLARE — uso il più recente per mtime "
            f"(NON necessariamente il più completo): {chosen.name}. Altri: {others}"
        )
    return chosen


def main() -> None:
    args = parse_args()
    args.fl_results = _resolve_fl_results_path(args.fl_results)

    logger.info("=" * 60)
    logger.info("ChargeShield-FL — Analisi post-hoc su dump NVFLARE (fase 5)")
    logger.info(f"Dump NVFLARE: {args.fl_results}")
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

    train_sessions, cluster_membership, holdout_sessions = load_client_sessions(
        args.client_config, seed=seed,
    )

    if args.holdout_dataset is not None:
        # Override esplicito: ignora l'hold-out ricostruito sopra e usa un
        # file passato a mano (utile per confrontare contro un anno/periodo
        # genuinamente esterno, se in futuro ne esiste uno disponibile).
        logger.warning(
            "--holdout-dataset esplicito passato — ignoro l'hold-out 80/20 "
            "ricostruito da load_client_sessions() e uso questo file al suo posto."
        )
        holdout_sessions = load_holdout_sessions(args.holdout_dataset)
    elif not holdout_sessions:
        # Design a file singolo (pre-2026-07-22) o dump precedente al fix
        # 2026-08-03: nessun hold-out ricostruibile automaticamente da
        # load_client_sessions() — serve un file esplicito, non lo si deduce
        # più "alla cieca" (vedi FIX 2026-08-03 nel docstring di
        # load_client_sessions() per il perché).
        raise ValueError(
            "Nessun hold-out disponibile: load_client_sessions() non ne ha "
            "ricostruito uno (design a file singolo, o dump di un job NVFLARE "
            "precedente al fix 2026-08-03 che allenava sul 100% dei dati). "
            "Specifica --holdout-dataset esplicitamente con un file mai usato "
            "per il training di nessun client."
        )

    logger.info(
        f"Train (client reali): {len(train_sessions)} sessioni — "
        f"Hold-out (non-member): {len(holdout_sessions)} sessioni"
    )

    _FEATURES = AutoencoderTrainer.CONTINUOUS_FEATURES
    # Stats calcolate su train_sessions (le sessioni realmente usate per il
    # training), NON su holdout — stessa regola anti-leakage di main() nella
    # simulazione.
    #
    # ATTENZIONE — discrepanza nota, non "corretta" alla cieca (2026-07-24,
    # review indipendente round 3): questa riga calcola UNA sola feature_stats
    # globale sull'intero pool multi-sito combinato. Il commento precedente qui
    # affermava che questo fosse "equivalente" a chargeshield_executor.py::
    # _setup() — non è più vero nel design attuale (post-migrazione ai 3 siti
    # reali, Sprint 10): lì ogni client calcola le proprie stats SOLO sulle
    # sessioni del proprio sito (vedi il commento a chargeshield_executor.py
    # righe 364-373, che riconosce esplicitamente che min/max variano tra siti —
    # es. kWh massimo osservato a Caltech vs JPL — invece di un'unica scala
    # globale condivisa come nella simulazione single-process).
    #
    # Non correggiamo qui ricalcolando le stats per-sito "alla cieca": farlo
    # richiederebbe anche decidere con quali stats normalizzare holdout_sessions
    # (che non appartiene in modo univoco a nessun sito di training), una scelta
    # metodologica che non possiamo validare senza poter eseguire davvero
    # NVFLARE in questo sandbox (torch/nvflare non installabili). Finché questo
    # script resta "non eseguito, non testato" (vedi docstring in testa al
    # file), qualunque numero prodotto da questa pipeline offline contro un vero
    # dump NVFLARE va trattato come indicativo, non come replica esatta della
    # normalizzazione realmente vista da ciascun client in training.
    feature_stats = compute_feature_stats(train_sessions, _FEATURES)
    train_sessions = normalize_sessions(train_sessions, feature_stats, _FEATURES)
    holdout_sessions = normalize_sessions(holdout_sessions, feature_stats, _FEATURES)

    # ── IDS — stessa funzione, stessa logica, usata dalla simulazione ───────
    ids_results: dict = {}
    try:
        ids_results = run_ids(cfg, fl_results, no_dp=no_dp)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run_ids() fallita: {exc}", exc_info=True)

    # ── Yeom + Shadow + LiRA — tramite il registro pluggable ────────────────
    # Fix 2026-07-24: questo blocco chiamava run_fedmia()/run_fedmia_shadow()/
    # run_lira() direttamente, duplicando (quasi identica) la stessa logica di
    # dispatch/merge/error-handling di scripts/run_experiments.py::main().
    # Ora entrambi i punti chiamano run_registered_attacks() (definita in
    # run_experiments.py, itera src/plugins/attacks/ATTACK_REGISTRY) — un solo
    # punto di verità invece di due copie che potevano divergere. Stesso
    # comportamento di prima: yeom, poi shadow, poi lira, merge per round,
    # un attacco fallito non blocca gli altri né il salvataggio finale.
    n_shadow = args.n_shadow if args.n_shadow is not None else cfg.get("lira", {}).get("n_shadow", 8)
    mia_results = run_registered_attacks(
        cfg, train_sessions, holdout_sessions, fl_results,
        n_shadow=n_shadow, shadow_epochs_cap=args.shadow_epochs_cap,
        no_dp=no_dp, dp_mode=dp_mode, cluster_membership=cluster_membership,
    )

    result_file = save_results(cfg, mia_results, ids_results, fl_results=fl_results, sweep_dir=args.sweep_dir)
    logger.info(f"Analisi NVFLARE completata — risultati in {result_file}")


if __name__ == "__main__":
    main()
