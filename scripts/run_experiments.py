#!/usr/bin/env python3
# scripts/run_experiments.py
# ChargeShield-FL — Sprint 5: Experiment Runner
#
# Esegue il ciclo completo:
#   1. Carica ACN-Data JPL 2019 + 2020
#   2. Esegue FL rounds via ML Plane (AutoencoderTrainer + GradientManager + FedAvgAggregator)
#   3. Lancia FedMIA attack per ogni round
#   4. Valuta IDS come baseline defense
#   5. Misura AUC-ROC e privacy/utility trade-off (epsilon vs AUC-ROC)
#   6. Salva risultati in experiments/
#
# Usage:
#   python scripts/run_experiments.py --config config/experiment.yaml
#   python scripts/run_experiments.py --epsilon 0.5 --rounds 10
#   python scripts/run_experiments.py --dry-run

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapters.acn_dataset import ACNDataset
from auditor.privacy_auditor import PrivacyAuditor
from core.autoencoder import Autoencoder
from ids.charging_ids import ChargingIDS
from ml.autoencoder_trainer import AutoencoderTrainer
from ml.fedavg_aggregator import FedAvgAggregator
from ml.gradient_manager import GradientManager

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: Path | None, overrides: dict) -> dict:
    """Carica config da YAML e applica override da CLI."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        raise FileNotFoundError(
            f"Config non trovata: {config_path}. "
            "Specifica --config o crea config/experiment.yaml"
        )
    if overrides.get("epsilon") is not None:
        cfg["experiment"]["epsilon"] = overrides["epsilon"]
    if overrides.get("rounds") is not None:
        cfg["experiment"]["fl_rounds"] = overrides["rounds"]
    return cfg


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_sessions(cfg: dict) -> list[dict[str, Any]]:
    """Carica sessioni EV da ACN-Data JPL 2019 + 2020."""
    sessions: list[dict[str, Any]] = []
    for key, path_str in cfg["datasets"].items():
        p = PROJECT_ROOT / path_str
        if not p.exists():
            logger.warning(f"Dataset non trovato: {p} — skip {key}")
            continue
        dataset = ACNDataset()
        dataset.load(str(p))
        loaded = [dataset.get_sample(i) for i in range(len(dataset))]
        sessions.extend(loaded)
        logger.info(f"{key}: {len(loaded)} sessioni caricate")
    if not sessions:
        raise FileNotFoundError(
            "Nessun dataset trovato. Scarica ACN-Data JPL da "
            "https://ev.caltech.edu/dataset e posizionalo in datasets/"
        )
    logger.info(f"Totale sessioni: {len(sessions)}")
    return sessions

# ── Session enrichment ─────────────────────────────────────────────────────────
def enrich_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggiunge feature derivate dai timestamp ACN-Data.
    - hour_of_day: ora di connessione (0–23), pattern comportamentale
    - duration_hours: durata sessione in ore, correlata all'energia
    """
    enriched = []
    for s in sessions:
        try:
            start = datetime.fromisoformat(s["start_time"])
            end   = datetime.fromisoformat(s["end_time"])
            s["hour_of_day"]    = float(start.hour)
            s["duration_hours"] = max(0.0, (end - start).total_seconds() / 3600.0)
            enriched.append(s)
        except (KeyError, ValueError):
            pass  # scarta sessioni con timestamp malformati
    return enriched


# ── Feature Normalization ──────────────────────────────────────────────────────

def compute_feature_stats(
    sessions: list[dict[str, Any]],
    features: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Calcola min e max per ogni feature dalle sessioni di training.
    Chiamare SOLO su train_sessions per evitare data leakage dal hold-out.
    """
    stats: dict[str, tuple[float, float]] = {}
    for feat in features:
        vals = []
        for s in sessions:
            raw = s.get(feat)
            if raw is None:
                continue
            try:
                vals.append(float(raw))
            except (ValueError, TypeError):
                continue  # salta valori non numerici (corrotti o stringhe errate)
        if not vals:
            stats[feat] = (0.0, 1.0)
            continue
        fmin, fmax = min(vals), max(vals)
        stats[feat] = (fmin, fmax if fmax != fmin else fmin + 1.0)
    return stats


def normalize_sessions(
    sessions: list[dict[str, Any]],
    stats: dict[str, tuple[float, float]],
    features: list[str],
) -> list[dict[str, Any]]:
    """
    Applica min-max scaling [0,1] alle feature continue.
    stats deve provenire da compute_feature_stats(train_sessions, ...).
    """
    normalized = []
    for s in sessions:
        s = dict(s)  # shallow copy — non modificare l'originale
        for feat in features:
            val = s.get(feat)
            if val is None:
                continue
            fmin, fmax = stats[feat]
            s[feat] = (float(val) - fmin) / (fmax - fmin)
        normalized.append(s)
    return normalized

# ── FL Experiment ──────────────────────────────────────────────────────────────

def run_fl_rounds(
    cfg: dict,
    sessions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Esegue FL rounds via ML Plane.
    Ogni round: train locale → DP → FedAvg → global model.
    Restituisce gradient history per round.
    """
    exp_cfg  = cfg["experiment"]
    ml_cfg   = cfg["ml"]
    fl_rounds = exp_cfg["fl_rounds"]

    cluster_ids = ["highway", "urban", "residential", "corporate"]
    cluster_size = max(1, len(sessions) // len(cluster_ids))

    # Inizializza trainer per ogni cluster
    trainers: dict[str, AutoencoderTrainer] = {}
    cluster_sessions: dict[str, list] = {}
    for i, cid in enumerate(cluster_ids):
        # Propaga il seed sperimentale nella config ml così AutoencoderTrainer
        # lo usa per il DataLoader generator → shuffle deterministico per seed.
        trainer_cfg = {**ml_cfg, "seed": exp_cfg.get("seed", 42)}
        trainers[cid] = AutoencoderTrainer(
            config=trainer_cfg,
            node_id=f"{cid}-01",
            cluster_id=cid,
        )
        start = i * cluster_size
        end   = None if i == len(cluster_ids) - 1 else start + cluster_size
        cluster_sessions[cid] = sessions[start:end]
        logger.info(f"Cluster {cid}: {len(cluster_sessions[cid])} sessioni")

    gm = GradientManager({
        "epsilon":       exp_cfg["epsilon"],
        "delta":         exp_cfg["delta"],
        "max_grad_norm": exp_cfg["max_grad_norm"],
    })

    agg = FedAvgAggregator({"min_participants": len(cluster_ids)})

    results: dict[int, dict[str, Any]] = {}

    # ── Byzantine attack config ────────────────────────────────────────────────
    # Legge la sezione byzantine_attack dal config. Se assente o disabled, nessun attacco.
    _byz_cfg      = cfg.get("byzantine_attack", {})
    _byz_enabled  = _byz_cfg.get("enabled", False)
    _byz_node     = _byz_cfg.get("byzantine_node", "highway")   # cluster attaccante
    _byz_type     = _byz_cfg.get("attack_type", "gradient_scaling")
    _byz_scale    = float(_byz_cfg.get("scale_factor", 10.0))

    if _byz_enabled:
        logger.warning(
            f"[BYZANTINE ATTACK] abilitato — nodo={_byz_node}, "
            f"tipo={_byz_type}, scale={_byz_scale}"
        )

    # Baseline per IDS al round 1: pesi del modello inizializzato (prima del training).
    # Consente di calcolare il delta round 1 = post_training - init_model invece di
    # usare pesi assoluti (che causano GRADIENT_EXPLOSION falso per L2 >> max_grad_norm).
    _init_weights = trainers[cluster_ids[0]].get_weights() if cluster_ids else None
    results[0] = {"raw_global_weights": _init_weights}

    for round_num in range(1, fl_rounds + 1):
        logger.info(f"=== FL Round {round_num}/{fl_rounds} ===")

        round_updates = []
        raw_updates   = []  # pre-DP: usati da IDS per analisi non distorta dal rumore
        for cid, trainer in trainers.items():
            # Training locale
            update = trainer.train_local(cluster_sessions[cid], round_num)

            # ── Gradient scaling attack ──────────────────────────────────────
            # Se questo cluster è il nodo Byzantine e l'attacco è abilitato,
            # moltiplica tutti i pesi per scale_factor.
            # Effetto: l'update è geometricamente ~scale_factor× più distante
            # dagli altri nodi → Krum score >> 1.5 → alert reale.
            # L'attacco avviene sui pesi raw (pre-DP), che è ciò che l'IDS analizza;
            # poi passa anche nella privatizzazione → distorce FedAvg.
            if _byz_enabled and cid == _byz_node and _byz_type == "gradient_scaling":
                scaled_weights = [
                    (w if isinstance(w, torch.Tensor) else torch.tensor(float(w))) * _byz_scale
                    for w in (update.weights or [])
                ]
                from ml.base_ml import GradientUpdate as _GU
                update = _GU(
                    node_id=update.node_id,
                    cluster_id=update.cluster_id,
                    round_num=update.round_num,
                    weights=scaled_weights,
                    gradients=update.gradients,
                    loss=update.loss,
                    n_samples=update.n_samples,
                    metadata={**update.metadata, "byzantine_attack": True,
                               "scale_factor": _byz_scale},
                )
                logger.warning(
                    f"[BYZANTINE] Round {round_num}: {cid} — "
                    f"gradient scaling ×{_byz_scale} applicato"
                )

            raw_updates.append(update)          # conserva pre-DP per IDS
            # Applica DP — passa le chiavi per escludere buffer BatchNorm dal rumore
            weight_keys    = trainer.get_weight_keys()
            private_update = gm.privatize(update, weight_keys=weight_keys)
            agg.collect(private_update)
            round_updates.append(private_update)

        # Calcola raw_global_weights: media semplice dei pesi pre-DP.
        # Usato da IDS come riferimento per i delta (evita che il rumore DP
        # del global aggregato inquini l'analisi degli update locali).
        raw_global_weights: list[Any] | None = None
        if raw_updates and raw_updates[0].weights:
            n_w    = len(raw_updates[0].weights)
            total  = sum(u.n_samples for u in raw_updates) or len(raw_updates)
            raw_global_weights = []
            for i in range(n_w):
                wavg = sum(
                    (u.weights[i] if isinstance(u.weights[i], torch.Tensor)
                     else torch.tensor(float(u.weights[i])))
                    * (u.n_samples / total)
                    for u in raw_updates
                )
                raw_global_weights.append(wavg)

        # FedAvg
        aggregated = agg.aggregate(round_num)

        if aggregated is None:
            logger.warning(f"Round {round_num} saltato — partecipanti insufficienti")
            continue

        # Distribuisci modello globale ai trainer
        for trainer in trainers.values():
            trainer.apply_global_model(aggregated)

        loss_str = f"{aggregated.mean_loss:.6f}" if aggregated.mean_loss is not None else "N/A"
        logger.info(f"Round {round_num} — loss globale: {loss_str}")

        results[round_num] = {
            "mean_loss":         aggregated.mean_loss,
            "n_participants":    aggregated.n_participants,
            "updates":           round_updates,      # privatized — usati da FedMIA
            "raw_updates":       raw_updates,         # pre-DP — usati da IDS
            "raw_global_weights": raw_global_weights, # media raw — riferimento IDS
            "global_weights":    aggregated.global_weights,
        }

    return results

# ── FedMIA Attack ──────────────────────────────────────────────────────────────

# Feature ACN usate per FedMIA — allineate con AutoencoderTrainer
_MIA_FEATURES = [
    "total_energy_kwh", "max_power_kw", "kwh_requested",
    "minutes_available", "hour_of_day", "duration_hours",
]


def run_fedmia(
    cfg: dict,
    members: list[dict[str, Any]],
    non_members: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Loss-based Membership Inference Attack per FL con autoencoder.

    Per ogni round FL carica i pesi globali aggregati in un Autoencoder
    locale e misura l'errore di ricostruzione su membri e non-membri.
    Principio (Yeom et al., 2018): il modello FL produce errore basso sui
    campioni visti nel training (membri) e alto sui non visti (non-membri).
    La DP riduce questa gap → AUC-ROC → 0.5 (attacco non migliore del random).

    IMPORTANTE: members deve contenere sessioni effettivamente usate per il
    training FL; non_members deve essere un hold-out set mai visto da nessun
    FL node. Usare lo stesso pool per entrambi invalida la misura di AUC-ROC.

    L'AUC varia per round: round iniziali → modello non converge → AUC ≈ 0.5;
    round finali → modello memorizza → AUC cresce se DP insufficiente.

    Args:
        cfg:         configurazione esperimento
        members:     sessioni usate per FL training (vere member)
        non_members: sessioni hold-out mai viste durante training (vere non-member)
        fl_results:  dict round → {"global_weights": [...], ...}

    Returns:
        {round_num: {"auc_roc": float, "member_score_mean": float, ...}}
    """
    from sklearn.metrics import roc_auc_score

    logger.info(f"FedMIA — members: {len(members)}, non-members: {len(non_members)}")

    # ── Bilanciamento pool MIA ──────────────────────────────────────────────────
    # ACN-Data: 10,458 members vs 2,615 non-members (split 80/20).
    # Un pool sbilanciato 4:1 non invalida l'AUC-ROC (che è rank-based) ma produce
    # soglie di classificazione asimmetriche. Si usa un subsample fisso dei members
    # per garantire pool identici in dimensione e confrontabilità tra esperimenti.
    _pool_rng        = random.Random(cfg.get("experiment", {}).get("seed", 42))
    _n_bal           = min(len(members), len(non_members))
    members_balanced = _pool_rng.sample(members, _n_bal)
    logger.info(
        f"FedMIA pool bilanciato — members: {len(members_balanced)}, "
        f"non-members: {len(non_members)} "
        f"(members originali: {len(members)}, campionati con seed fisso)"
    )

    input_dim = cfg["ml"]["input_dim"]

    def _score_batch(model: Autoencoder, sess_list: list[dict]) -> list[float]:
        """Calcola membership scores (-MSE) sulle sessioni con feature complete.
        Coerente con _sessions_to_tensor: scarta sessioni con None nelle feature."""
        rows: list[list[float]] = []
        for s in sess_list:
            try:
                row = [float(s[f]) for f in _MIA_FEATURES]
                rows.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            return []
        tensor = torch.tensor(rows, dtype=torch.float32)
        results_: list[float] = []
        with torch.no_grad():
            for i in range(0, len(tensor), 256):
                batch = tensor[i : i + 256]
                recon  = model(batch)
                errors = torch.mean((recon - batch) ** 2, dim=1)
                # Score = -errore: basso errore → membro → score alto
                results_.extend(-e.item() for e in errors)
        return results_

    mia_results: dict[int, dict[str, Any]] = {}

    for round_num, round_data in sorted(fl_results.items()):
        global_weights = round_data.get("global_weights")
        if global_weights is None:
            logger.warning(f"Round {round_num}: global_weights assenti — skip FedMIA")
            continue

        # Carica pesi globali FL in un autoencoder locale (inference only).
        # load_state_dict trasferisce anche i buffer BatchNorm (running_mean/var).
        # global_weights è una lista con lo stesso ordine di state_dict().values():
        # sia AutoencoderTrainer.get_weights() che questo zip usano state_dict()
        # sulla stessa architettura Autoencoder, quindi l'ordine è garantito.
        model = Autoencoder(input_dim=input_dim)
        orig_state = model.state_dict()
        keys = list(orig_state.keys())
        if len(global_weights) != len(keys):
            logger.error(
                f"Round {round_num}: global_weights ha {len(global_weights)} elementi, "
                f"state_dict ne richiede {len(keys)} — skip FedMIA"
            )
            continue
        state = {
            k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(orig_state[k].dtype)
            for k, w in zip(keys, global_weights)
        }
        model.load_state_dict(state, strict=True)
        # Clamp BatchNorm running_var a valori positivi: il rumore DP con σ grande
        # (es. σ=48 per ε=0.1) può rendere running_var negativa, causando NaN in
        # sqrt(running_var + eps) durante la forward pass in eval mode.
        # Questo guard è difensivo; con il fix in GradientManager._add_noise() i
        # buffer BN non ricevono più rumore, quindi running_var sarà già positiva.
        for buf_name, buf in model.named_buffers():
            if "running_var" in buf_name:
                buf.clamp_(min=1e-8)
        model.eval()

        member_scores     = _score_batch(model, members_balanced)
        non_member_scores = _score_batch(model, non_members)

        if not member_scores or not non_member_scores:
            logger.warning(f"Round {round_num}: score batch vuoto — skip AUC")
            continue

        labels = [1] * len(member_scores) + [0] * len(non_member_scores)
        scores = member_scores + non_member_scores

        # Filtra NaN residui (belt-and-suspenders: senza il BN fix potrebbe ancora
        # verificarsi NaN per epsilon molto piccoli con sigma >> 1)
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        valid_mask = ~np.isnan(scores_arr) & ~np.isinf(scores_arr)
        if valid_mask.sum() < 10:
            logger.warning(
                f"Round {round_num}: troppi score NaN/Inf "
                f"({(~valid_mask).sum()}/{len(scores_arr)}) — probabile corruzione pesi DP "
                f"(σ grande >> norma pesi). AUC impostato a 0.5 (baseline random)."
            )
            mia_results[round_num] = {
                "auc_roc":               0.5,
                "member_score_mean":     float("nan"),
                "non_member_score_mean": float("nan"),
                "nan_fraction":          float((~valid_mask).sum()) / len(scores_arr),
            }
            continue
        if not valid_mask.all():
            logger.warning(
                f"Round {round_num}: {(~valid_mask).sum()} score NaN/Inf filtrati "
                f"su {len(scores_arr)} totali"
            )
        labels_arr = labels_arr[valid_mask]
        scores_arr = scores_arr[valid_mask]
        auc = roc_auc_score(labels_arr, scores_arr)
        logger.info(f"Round {round_num} — FedMIA AUC-ROC: {auc:.4f}")

        mia_results[round_num] = {
            "auc_roc":               auc,
            "member_score_mean":     float(np.nanmean(member_scores)),
            "non_member_score_mean": float(np.nanmean(non_member_scores)),
        }

    return mia_results


# ── Shadow Model MIA Attack ────────────────────────────────────────────────────

def run_fedmia_shadow(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Calibrated Shadow-Model MIA Attack (ispirato a LiRA, Carlini et al. 2022).

    Motivazione:
        L'attacco Yeom (loss-based) fallisce quando il modello FL generalizza bene:
        la loss su membro e non-membro è simile → AUC ≈ 0.5. Il shadow attack
        controlla per la "difficoltà intrinseca" di ogni campione confrontando
        il modello FL target con un modello shadow addestrato su dati simili ma
        diversi: il segnale di membership emerge dalla *differenza* di loss, non
        dal valore assoluto.

    Setup:
        - shadow_train (50% di train_sessions) → addestra shadow model (no FL, no DP)
        - eval_members (50% di train_sessions) → valutazione membro:
            * FL ha visto questi campioni → loss_target bassa
            * shadow NON li ha visti     → loss_shadow alta
            * score = loss_shadow - loss_target > 0  ✓
        - holdout_sessions → valutazione non-membro:
            * né FL né shadow li hanno visti
            * score = loss_shadow - loss_target ≈ 0  ✓

    Score di membership:
        score(x) = MSE(shadow_model, x) − MSE(target_model, x)
        Maggiore score → più probabile che x sia stato nel training FL.

    Riferimenti:
        Carlini et al., "Membership Inference Attacks From First Principles",
        IEEE S&P 2022. https://arxiv.org/abs/2112.03570

    Args:
        cfg:              configurazione esperimento
        train_sessions:   sessioni usate nel FL training (tutti i membri)
        holdout_sessions: sessioni mai viste durante FL (non-membri)
        fl_results:       dict round → {"global_weights": [...], ...}

    Returns:
        {round_num: {
            "shadow_auc_roc": float,
            "shadow_member_score_mean": float,
            "shadow_non_member_score_mean": float,
            "shadow_score_gap": float,   # differenza media membro - non-membro
            "n_eval_members": int,
            "n_non_members": int,
        }}
    """
    from sklearn.metrics import roc_auc_score

    seed       = cfg.get("experiment", {}).get("seed", 42)
    input_dim  = cfg["ml"]["input_dim"]
    ml_cfg     = cfg["ml"]
    local_epochs = ml_cfg.get("epochs", 3)
    total_rounds = cfg.get("experiment", {}).get("fl_rounds", 100)

    # ── Step 1: split train → shadow_train (50%) + eval_members (50%) ──────────
    rng = random.Random(seed + 999)       # seed diverso da quello del pool Yeom
    shuffled = list(train_sessions)
    rng.shuffle(shuffled)
    mid           = max(1, len(shuffled) // 2)
    shadow_train  = shuffled[:mid]
    eval_members  = shuffled[mid:]

    logger.info(
        f"Shadow MIA — shadow_train: {len(shadow_train)}, "
        f"eval_members: {len(eval_members)}, non-members: {len(holdout_sessions)}"
    )

    # ── Step 2: addestra il shadow model ──────────────────────────────────────
    # Autoencoder locale (non FL, no DP) addestrato sul shadow_train.
    # Epoche totali = local_epochs × total_rounds (equivalente al training FL),
    # capped a 500 per non rallentare troppo il sweep.
    shadow_epochs = min(local_epochs * total_rounds, 500)

    def _build_tensor(sess_list: list[dict]) -> torch.Tensor | None:
        rows = []
        for s in sess_list:
            try:
                rows.append([float(s[f]) for f in _MIA_FEATURES])
            except (KeyError, TypeError, ValueError):
                continue
        return torch.tensor(rows, dtype=torch.float32) if rows else None

    shadow_tensor = _build_tensor(shadow_train)
    if shadow_tensor is None or len(shadow_tensor) == 0:
        logger.warning("Shadow MIA: shadow_train vuoto dopo feature extraction — skip")
        return {}

    shadow_model = Autoencoder(input_dim=input_dim)
    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=ml_cfg.get("lr", 1e-3))
    shadow_criterion = torch.nn.MSELoss()
    batch_size = ml_cfg.get("batch_size", 32)

    if len(shadow_tensor) < batch_size:
        logger.warning(
            f"Shadow MIA: shadow_train ({len(shadow_tensor)} sessioni) < "
            f"batch_size ({batch_size}) — shadow model non addestrato con drop_last=True. "
            "shadow_auc_roc non significativo. Aumentare il dataset o ridurre batch_size."
        )
        return {}

    shadow_model.train()
    torch.manual_seed(seed + 999)
    shadow_ds = torch.utils.data.TensorDataset(shadow_tensor)
    shadow_loader_gen = torch.Generator()
    shadow_loader_gen.manual_seed(seed + 999)
    shadow_loader = torch.utils.data.DataLoader(
        shadow_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, generator=shadow_loader_gen,
    )

    for epoch in range(shadow_epochs):
        for (batch,) in shadow_loader:
            shadow_optimizer.zero_grad()
            recon = shadow_model(batch)
            loss  = shadow_criterion(recon, batch)
            loss.backward()
            shadow_optimizer.step()

    shadow_model.eval()
    logger.info(f"Shadow model addestrato — {shadow_epochs} epoche su {len(shadow_train)} sessioni")

    # ── Step 3: compute scores per-sample con entrambi i modelli ───────────────

    def _mse_batch(model: Autoencoder, sess_list: list[dict]) -> list[float]:
        """Calcola MSE per sessione (non negato: valore grezzo per calibrazione)."""
        tensor = _build_tensor(sess_list)
        if tensor is None:
            return []
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(tensor), 256):
                batch = tensor[i : i + 256]
                recon  = model(batch)
                errors = torch.mean((recon - batch) ** 2, dim=1)
                scores.extend(e.item() for e in errors)
        return scores

    # Pre-computa shadow scores una volta sola (non dipende dal round FL)
    shadow_scores_members     = _mse_batch(shadow_model, eval_members)
    shadow_scores_nonmembers  = _mse_batch(shadow_model, holdout_sessions)

    # Bilanciamento: stessa dimensione per eval_members e non-members
    _bal_rng       = random.Random(seed + 999)
    _n_bal         = min(len(shadow_scores_members), len(shadow_scores_nonmembers))
    shadow_scores_members    = _bal_rng.sample(shadow_scores_members, _n_bal)
    shadow_scores_nonmembers = _bal_rng.sample(shadow_scores_nonmembers, _n_bal)

    # ── Step 4: per ogni round FL, calcola score calibrato ─────────────────────
    shadow_results: dict[int, dict[str, Any]] = {}

    for round_num, round_data in sorted(fl_results.items()):
        global_weights = round_data.get("global_weights")
        if global_weights is None:
            continue

        # Carica pesi globali FL nel target model
        target_model = Autoencoder(input_dim=input_dim)
        orig_state   = target_model.state_dict()
        keys         = list(orig_state.keys())
        if len(global_weights) != len(keys):
            logger.error(
                f"Shadow MIA round {round_num}: global_weights {len(global_weights)} "
                f"!= state_dict {len(keys)} — skip"
            )
            continue
        state = {
            k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(orig_state[k].dtype)
            for k, w in zip(keys, global_weights)
        }
        target_model.load_state_dict(state, strict=True)
        for buf_name, buf in target_model.named_buffers():
            if "running_var" in buf_name:
                buf.clamp_(min=1e-8)
        target_model.eval()

        # Scores target model
        target_scores_members    = _mse_batch(target_model, eval_members)
        target_scores_nonmembers = _mse_batch(target_model, holdout_sessions)

        # Bilancia anche i target scores allo stesso indice del shadow
        _bal_rng2 = random.Random(seed + 999)
        target_scores_members    = _bal_rng2.sample(target_scores_members, _n_bal)
        target_scores_nonmembers = _bal_rng2.sample(target_scores_nonmembers, _n_bal)

        # score calibrato = loss_shadow − loss_target
        # Positivo → target conosce il campione meglio del shadow → membro
        calibrated_members    = [
            s - t for s, t in zip(shadow_scores_members,    target_scores_members)
        ]
        calibrated_nonmembers = [
            s - t for s, t in zip(shadow_scores_nonmembers, target_scores_nonmembers)
        ]

        labels = [1] * len(calibrated_members) + [0] * len(calibrated_nonmembers)
        scores = calibrated_members + calibrated_nonmembers

        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        valid_mask = ~np.isnan(scores_arr) & ~np.isinf(scores_arr)
        if valid_mask.sum() < 10:
            logger.warning(f"Shadow MIA round {round_num}: troppi NaN — skip")
            continue
        scores_arr = scores_arr[valid_mask]
        labels_arr = labels_arr[valid_mask]

        try:
            auc = roc_auc_score(labels_arr, scores_arr)
        except ValueError:
            auc = 0.5

        score_gap = float(np.nanmean(calibrated_members) - np.nanmean(calibrated_nonmembers))
        logger.info(
            f"Round {round_num} — Shadow MIA AUC: {auc:.4f} "
            f"(gap={score_gap:.6f})"
        )

        shadow_results[round_num] = {
            "shadow_auc_roc":               round(auc, 6),
            "shadow_member_score_mean":     round(float(np.nanmean(calibrated_members)), 6),
            "shadow_non_member_score_mean": round(float(np.nanmean(calibrated_nonmembers)), 6),
            "shadow_score_gap":             round(score_gap, 6),
            "n_eval_members":               _n_bal,
            "n_non_members":                _n_bal,
        }

    return shadow_results


# ── IDS Evaluation ─────────────────────────────────────────────────────────────

def run_ids(
    cfg: dict,
    fl_results: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Valuta ChargingIDS su ogni round FL.

    Usa PrivacyAuditor per generare AuditReport reali con threats_detected
    popolato (GRADIENT_EXPLOSION, PRIVACY_BUDGET_EXHAUSTED, ecc.).
    Un singolo auditor persiste tra i round per tracciare l'epsilon cumulativo.
    """
    config_path = str(PROJECT_ROOT / "config" / "auditor.yaml")

    ids = ChargingIDS(
        config_path=config_path,
        # byzantine_tolerance=0: con 4 cluster, la condizione Krum (n >= 2f+3) richiede
        # f=0 per essere soddisfatta (4 >= 3). Con f=0, Krum funziona come outlier
        # detector geometrico (non Byzantine-tolerant nel senso formale di Blanchard 2017).
        # Documentato nel paper come limitazione della topologia a 4 cluster.
        byzantine_tolerance=0,
        # cosine_threshold=0.3: soglia coerente con la variabilità naturale
        # dei gradienti nei cluster FL omogenei. 0.85 produceva falsi positivi sistematici.
        cosine_threshold=0.3,
        # krum_threshold=1.5: dopo fix normalizzazione (mean invece di max),
        # nodi legittimi hanno score≈1.0, Byzantine≈2.5–3.0.
        # Soglia 1.5 è conservativa: non spara per variazioni naturali (≤1.1)
        # ma rileva outlier reali (>1.5). La soglia default 0.8 dava FP sistematici
        # perché con normalizzazione per max tutti i nodi hanno score≈1.0.
        krum_threshold=1.5,
        # fedmia= non passato: il plugin shadow-model FedMIA (src/plugins/attacks/fedmia.py)
        # è disabilitato in questa configurazione sperimentale. Il MIA è valutato
        # separatamente tramite run_fedmia() che usa l'approccio loss-based (Yeom et al.,
        # 2018): membership score = -MSE(global_model, session).
        # Per attivare il plugin shadow-model nell'IDS, passare: fedmia=FedMIA(cfg)
    )
    # Un'unica istanza traccia l'epsilon cumulativo per nodo su tutti i round
    auditor = PrivacyAuditor(config_path=config_path, epsilon=cfg["experiment"]["epsilon"])

    ids_results: dict[int, dict[str, Any]] = {}

    # IDS usa pesi PRE-DP (raw_updates) e raw_global_weights come baseline.
    # Motivazione: con ε=0.1, σ ≈ 48×max_grad_norm. I pesi post-DP hanno
    # L2-norm >> max_grad_norm (rumore domina), causando GRADIENT_EXPLOSION
    # e BUDGET_EXHAUSTED falsi sistematici in ogni round.
    # In un sistema reale, il server/IDS vede gli update raw dai client PRIMA
    # che il rumore DP venga applicato → analisi corretta delle anomalie.
    # Delta = raw_local - raw_global_prev: rappresenta la deriva locale netta,
    # bounded dalla clipping norm × local epochs × lr (in pratica << max_grad_norm).

    # Inizializza prev_raw_global con i pesi del modello iniziale (round 0),
    # salvati in run_fl_rounds() prima dell'inizio del training loop.
    # Questo elimina il falso GRADIENT_EXPLOSION al round 1 dovuto ai pesi
    # assoluti del modello non ancora aggiornato.
    prev_raw_global: list[Any] | None = (fl_results.get(0) or {}).get("raw_global_weights")

    for round_num, round_data in sorted(
        (item for item in fl_results.items() if item[0] > 0), key=lambda x: x[0]
    ):
        # Preferisci raw_updates (pre-DP). Fallback su updates per retrocompatibilità.
        updates = round_data.get("raw_updates") or round_data.get("updates", [])
        # raw_global_weights di questo round (media raw, usato come prev al prossimo)
        current_raw_global = round_data.get("raw_global_weights")

        if not updates:
            prev_raw_global = current_raw_global
            ids_results[round_num] = {
                "alerts": [], "byzantine_detected": False, "drift_detected": False,
            }
            continue

        reports: dict[str, Any] = {}
        gradients: dict[str, dict[str, Any]] = {}

        for update in updates:
            if not update or not update.node_id:
                continue

            weights = update.weights or []

            # delta = raw_local_weights - raw_global_prev_round.
            # Al round 1 (prev_raw_global=None) usiamo pesi assoluti — GRADIENT_EXPLOSION
            # round 1 può essere legittimo (modello non ancora normalizzato).
            if prev_raw_global is not None and len(prev_raw_global) == len(weights):
                delta_weights = [
                    (w if isinstance(w, torch.Tensor) else torch.tensor(float(w)))
                    - (g if isinstance(g, torch.Tensor) else torch.tensor(float(g)))
                    for w, g in zip(weights, prev_raw_global)
                ]
            else:
                delta_weights = [
                    (w if isinstance(w, torch.Tensor) else torch.tensor(float(w)))
                    for w in weights
                ]

            model_update: dict[str, Any] = {
                f"layer_{i}": dw for i, dw in enumerate(delta_weights)
            }

            # AuditReport con threats_detected reali (GRADIENT_EXPLOSION, ecc.)
            reports[update.node_id] = auditor.audit(
                node_id=update.node_id,
                round_id=round_num,
                model_update=model_update,
            )

            # Gradient dict per Krum / cosine analysis dell'IDS
            gradients[update.node_id] = model_update

        prev_raw_global = current_raw_global

        if not reports:
            ids_results[round_num] = {
                "alerts": [], "byzantine_detected": False, "drift_detected": False,
            }
            continue

        analysis = ids.analyze_round(
            round_id=round_num,
            reports=reports,
            gradients=gradients,
        )

        ids_results[round_num] = {
            "alerts": [
                {
                    "node_id":            a.node_id,
                    "severity":           a.severity,
                    "reasons":            a.reasons,
                    "recommended_action": a.recommended_action,
                }
                for a in (analysis.alerts if analysis else [])
            ],
            "byzantine_detected":   len(analysis.byzantine_nodes) > 0 if analysis else False,
            "drift_detected":       False,
            "low_similarity_nodes": analysis.low_similarity_nodes if analysis else [],
        }

    return ids_results


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(
    cfg: dict,
    mia_results: dict[int, dict[str, Any]],
    ids_results: dict[int, dict[str, Any]],
    fl_results: dict[int, dict[str, Any]] | None = None,
    sweep_dir: Path | None = None,
) -> Path:
    """
    Salva risultati in experiments/ (o sweep_dir) con timestamp.

    Se sweep_dir è fornita, i JSON vengono salvati in quella directory
    e l'Excel verrà nominato come la directory (es. experiments/exp1/exp1.xlsx).
    Questo garantisce che ogni sweep abbia il proprio file Excel separato.
    """
    if sweep_dir is not None:
        output_dir = sweep_dir
    else:
        output_dir = PROJECT_ROOT / cfg["output"]["experiments_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"experiment_{timestamp}.json"

    auc_values = [
        r["auc_roc"]
        for r in mia_results.values()
        if r.get("auc_roc") is not None
    ]
    shadow_auc_values = [
        r["shadow_auc_roc"]
        for r in mia_results.values()
        if r.get("shadow_auc_roc") is not None
    ]

    # Privacy risk basato sull'attacco più forte disponibile:
    # se shadow AUC è disponibile usa quello, altrimenti Yeom.
    _primary_auc = shadow_auc_values if shadow_auc_values else auc_values
    _primary_mean = float(np.mean(_primary_auc)) if _primary_auc else None

    summary = {
        "experiment_name": cfg["experiment"]["name"],
        "timestamp":       timestamp,
        "config": {
            "epsilon":    cfg["experiment"]["epsilon"],
            "delta":      cfg["experiment"]["delta"],
            "fl_rounds":  cfg["experiment"]["fl_rounds"],
            "proximal_mu": cfg["ml"]["proximal_mu"],
        },
        "summary": {
            # Yeom 2018 loss-based MIA
            "mean_auc_roc": float(np.mean(auc_values)) if auc_values else None,
            "max_auc_roc":  float(np.max(auc_values))  if auc_values else None,
            "min_auc_roc":  float(np.min(auc_values))  if auc_values else None,
            # Carlini 2022 shadow/calibrated MIA (attacco più forte)
            "mean_shadow_auc_roc": float(np.mean(shadow_auc_values)) if shadow_auc_values else None,
            "max_shadow_auc_roc":  float(np.max(shadow_auc_values))  if shadow_auc_values else None,
            "min_shadow_auc_roc":  float(np.min(shadow_auc_values))  if shadow_auc_values else None,
            # Privacy risk basato sull'attacco primario (shadow se disponibile)
            "privacy_risk": (
                "HIGH"   if _primary_mean is not None and _primary_mean > 0.7 else
                "MEDIUM" if _primary_mean is not None and _primary_mean > 0.6 else
                "LOW"
            ),
        },
        "per_round": {
            # Itera sull'unione di tutti i round: FL, MIA e IDS.
            # round 0 è escluso: contiene solo raw_global_weights (init model) per IDS.
            str(r): {
                "fl":  {"mean_loss": (fl_results or {}).get(r, {}).get("mean_loss")},
                "mia": mia_results.get(r, {}),
                "ids": ids_results.get(r, {}),
            }
            for r in sorted(
                set(mia_results.keys())
                | {k for k in (fl_results or {}).keys() if k > 0}
                | set(ids_results.keys())
            )
        },
    }

    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    mean_str = f"{summary['summary']['mean_auc_roc']:.4f}" \
               if summary["summary"]["mean_auc_roc"] is not None else "N/A"
    logger.info(f"Risultati salvati: {result_file.name}")
    logger.info(
        f"AUC-ROC medio: {mean_str} — "
        f"Privacy risk: {summary['summary']['privacy_risk']}"
    )

    # Aggiorna automaticamente il report Excel del sweep corrente
    _update_excel_report(output_dir, named_sweep=(sweep_dir is not None))

    return result_file


def _update_excel_report(sweep_dir: Path, named_sweep: bool = False) -> None:
    """
    Rigenera il report Excel a 6 sheet per il sweep corrente.

    Il nome del file Excel dipende dalla modalità:
    - sweep_dir nominata (es. experiments/exp1) → exp1.xlsx
    - fallback (experiments/) → ChargeShield_FL_Results.xlsx (retro-compatibilità)

    Usa un import Python standard invece di exec_module() per evitare il rischio
    di arbitrary code execution se il file fosse modificato da un attacker con accesso
    al filesystem. Con import standard il modulo viene caricato una sola volta e
    cachato in sys.modules — sicuro e idempotente.

    Args:
        sweep_dir:   directory dove sono i JSON del sweep (e dove salvare l'Excel)
        named_sweep: True se sweep_dir è una directory nominata (es. exp1),
                     False per backward compatibility con experiments/
    """
    try:
        from openpyxl import Workbook

        # Import diretto: generate_excel_report.py è nella stessa directory di questo script.
        # Se il file non esiste, ImportError viene catturato sotto.
        _scripts_dir = str(Path(__file__).parent)
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        import generate_excel_report as gen  # noqa: PLC0415

        records = gen.load_experiments(sweep_dir)
        if not records:
            return

        wb = Workbook()
        wb.remove(wb.active)
        gen.build_raw_data(wb.create_sheet("Raw Data"),               records)
        gen.build_heat_map(wb.create_sheet("Heat Map"),               records)
        gen.build_per_rounds(wb.create_sheet("Per Rounds"),           records)
        gen.build_per_epsilon(wb.create_sheet("Per Epsilon"),         records)
        gen.build_comparison(wb.create_sheet("Comparison"),           records)
        gen.build_auc_progression(wb.create_sheet("AUC Progression"), records)
        wb.properties.title   = "ChargeShield-FL Experiment Results"
        wb.properties.subject = "FedMIA vs Differential Privacy — DSN 2027"

        # Nome file Excel: sweep nominato → "{nome}.xlsx", fallback → nome storico
        if named_sweep:
            output_path = sweep_dir / f"{sweep_dir.name}.xlsx"
        else:
            output_path = sweep_dir / "ChargeShield_FL_Results.xlsx"

        wb.save(output_path)
        logger.info(f"Report Excel aggiornato: {output_path.name}")
    except ImportError:
        logger.warning(
            "openpyxl non trovato — report Excel non generato. "
            "Installa con: pip install openpyxl"
        )
    except Exception as exc:
        logger.warning(f"Report Excel non generato: {exc}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChargeShield-FL — FedMIA Experiment Runner"
    )
    parser.add_argument("--config",   type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--epsilon",  type=float, default=None)
    parser.add_argument("--rounds",   type=int,   default=None)
    parser.add_argument("--skip-ids", action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help=(
            "Directory del sweep corrente (es. experiments/exp1). "
            "Se fornita, JSON e Excel vengono salvati qui con nome del sweep "
            "(es. exp1.xlsx). Permette di isolare i risultati di sweep distinti."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Seed riproducibilità (override experiment.yaml). "
            "Usato per: shuffle sessioni, DataLoader, init modello. "
            "Valori consigliati per sweep: 42 123 456 789 1234."
        ),
    )
    parser.add_argument(
        "--byzantine", action="store_true", default=False,
        help="Abilita Byzantine attack (gradient scaling) sul nodo configurato in experiment.yaml.",
    )
    parser.add_argument(
        "--byzantine-node", type=str, default=None,
        help="Override cluster attaccante (es. highway, urban). Default: valore da config.",
    )
    parser.add_argument(
        "--scale-factor", type=float, default=None,
        help="Override scale_factor dell'attacco. Default: valore da config (10.0).",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ChargeShield-FL — Sprint 5 Experiment")
    logger.info("=" * 60)

    cfg = load_config(args.config, {"epsilon": args.epsilon, "rounds": args.rounds})

    # Override seed da CLI (--seed)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    # Override Byzantine attack da CLI (--byzantine, --byzantine-node, --scale-factor)
    if args.byzantine:
        cfg.setdefault("byzantine_attack", {})["enabled"] = True
    if args.byzantine_node:
        cfg.setdefault("byzantine_attack", {})["byzantine_node"] = args.byzantine_node
    if args.scale_factor is not None:
        cfg.setdefault("byzantine_attack", {})["scale_factor"] = args.scale_factor
    exp_cfg = cfg["experiment"]
    logger.info(
        f"Config: epsilon={exp_cfg['epsilon']}, "
        f"rounds={exp_cfg['fl_rounds']}, "
        f"proximal_mu={cfg['ml']['proximal_mu']}"
    )

    sessions = load_sessions(cfg)
    sessions = enrich_sessions(sessions)
    logger.info(f"Sessioni dopo enrichment: {len(sessions)}")

    # Split hold-out PRIMA del training FL: 80% train, 20% hold-out (mai visti dai nodi FL).
    # Seed fisso per riproducibilità dei risultati — fondamentale per DSN 2027.
    seed = exp_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Determinismo GPU (no overhead su CPU-only, ignorato silenziosamente se no CUDA)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.shuffle(sessions)
    split = max(1, int(len(sessions) * 0.8))
    train_sessions   = sessions[:split]
    holdout_sessions = sessions[split:]
    logger.info(f"Split — train: {len(train_sessions)}, hold-out: {len(holdout_sessions)}")

    # Normalizzazione min-max: calcolata SOLO su train_sessions (no leakage dal hold-out).
    # Stessa trasformazione applicata a holdout_sessions per la FedMIA.
    _FEATURES = AutoencoderTrainer.CONTINUOUS_FEATURES  # importato a livello modulo (riga 41)
    feature_stats    = compute_feature_stats(train_sessions, _FEATURES)
    train_sessions   = normalize_sessions(train_sessions,   feature_stats, _FEATURES)
    holdout_sessions = normalize_sessions(holdout_sessions, feature_stats, _FEATURES)
    logger.info(f"Feature normalizzate [0,1]: {list(feature_stats.keys())}")
    if args.dry_run:
        logger.info("Dry run completato — uscita.")
        return

    fl_results = run_fl_rounds(cfg, train_sessions)

    # run_fedmia, run_fedmia_shadow e run_ids non devono impedire il salvataggio.
    # Con try/except, save_results() viene sempre chiamato anche in caso di errore.
    mia_results: dict = {}
    try:
        mia_results = run_fedmia(cfg, train_sessions, holdout_sessions, fl_results)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"run_fedmia() fallita: {exc}. "
            "I risultati FL vengono comunque salvati con mia_results={}. "
            "Controllare il log per la causa (es. NaN negli score MIA).",
            exc_info=True,
        )

    # Shadow MIA (Carlini 2022) — attacco calibrato più potente di Yeom 2018.
    # Affianca run_fedmia() senza sostituirlo: entrambe le metriche vengono salvate.
    shadow_mia_results: dict = {}
    try:
        shadow_mia_results = run_fedmia_shadow(
            cfg, train_sessions, holdout_sessions, fl_results
        )
        # Merge nei mia_results per round: aggiunge campi shadow_* al dict esistente
        for rnd, shadow_data in shadow_mia_results.items():
            if rnd in mia_results:
                mia_results[rnd].update(shadow_data)
            else:
                mia_results[rnd] = shadow_data
    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"run_fedmia_shadow() fallita: {exc}. Continuazione senza shadow MIA.",
            exc_info=True,
        )

    ids_results: dict = {}
    if not args.skip_ids:
        try:
            ids_results = run_ids(cfg, fl_results)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"run_ids() fallita: {exc}. Continuazione senza risultati IDS.", exc_info=True)

    # sweep_dir: se fornita via --sweep-dir, i risultati vanno in quella directory
    # con Excel nominato come il sweep (es. exp1.xlsx). Altrimenti usa experiments/.
    sweep_dir = args.sweep_dir.resolve() if args.sweep_dir else None
    save_results(cfg, mia_results, ids_results, fl_results, sweep_dir=sweep_dir)

    logger.info("=" * 60)
    logger.info("Esperimento completato.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
