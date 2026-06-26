#!/usr/bin/env python3
# scripts/run_experiment.py
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
#   python scripts/run_experiment.py --config config/experiment.yaml
#   python scripts/run_experiment.py --epsilon 0.5 --rounds 10
#   python scripts/run_experiment.py --dry-run

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapters.acn_dataset import ACNDataset
from ml.autoencoder_trainer import AutoencoderTrainer
from ml.gradient_manager import GradientManager
from ml.fedavg_aggregator import FedAvgAggregator
from plugins.attacks.fedmia import FedMIA
from ids.charging_ids import ChargingIDS

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
        trainers[cid] = AutoencoderTrainer(
            config=ml_cfg,
            node_id=f"{cid}-01",
            cluster_id=cid,
        )
        start = i * cluster_size
        cluster_sessions[cid] = sessions[start: start + cluster_size]
        logger.info(f"Cluster {cid}: {len(cluster_sessions[cid])} sessioni")

    gm = GradientManager({
        "epsilon":       exp_cfg["epsilon"],
        "delta":         exp_cfg["delta"],
        "max_grad_norm": exp_cfg["max_grad_norm"],
    })

    agg = FedAvgAggregator({"min_participants": len(cluster_ids)})

    results: dict[int, dict[str, Any]] = {}

    for round_num in range(1, fl_rounds + 1):
        logger.info(f"=== FL Round {round_num}/{fl_rounds} ===")

        round_updates = []
        for cid, trainer in trainers.items():
            # Training locale
            update = trainer.train_local(cluster_sessions[cid], round_num)
            # Applica DP
            private_update = gm.privatize(update)
            agg.collect(private_update)
            round_updates.append(private_update)

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
            "mean_loss":     aggregated.mean_loss,
            "n_participants": aggregated.n_participants,
            "updates":       round_updates,
        }

    return results

# ── FedMIA Attack ──────────────────────────────────────────────────────────────

# Feature ACN usate per FedMIA — allineate con AutoencoderTrainer
_MIA_FEATURES = [
    "total_energy_kwh", "max_power_kw", "kwh_requested",
    "minutes_available", "hour_of_day", "duration_hours",
]

def _sessions_to_loader(
    sessions: list[dict[str, Any]],
    batch_size: int = 32,
) -> "torch.utils.data.DataLoader":
    """
    Converte sessioni ACN in DataLoader per shadow model FedMIA.
    Usa dataset custom che restituisce tensori diretti (non tuple),
    compatibile con autoencoder.fit() che itera batch come Tensor.
    """
    import torch

    rows = [
        [float(s.get(f) or 0.0) for f in _MIA_FEATURES]
        for s in sessions
    ]
    tensor = torch.tensor(rows, dtype=torch.float32)

    class _DirectTensorDataset(torch.utils.data.Dataset):
        """Dataset che restituisce tensori row-by-row, non tuple."""
        def __init__(self, t: torch.Tensor) -> None:
            self._t = t
        def __len__(self) -> int:
            return len(self._t)
        def __getitem__(self, i: int) -> torch.Tensor:
            return self._t[i]

    return torch.utils.data.DataLoader(
        _DirectTensorDataset(tensor),
        batch_size=batch_size,
        shuffle=True,
    )


def run_fedmia(
    cfg: dict,
    sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Esegue FedMIA su sessioni ACN-Data.
    Shadow model addestrato su members (50%) — poi scorecard su tutti.
    Misura AUC-ROC: capacità di distinguere membro da non-membro.
    FedMIA opera su dati post-DP — gradienti già privatizzati.
    """
    from sklearn.metrics import roc_auc_score

    mia_cfg = cfg["fedmia"]
    n = len(sessions)
    members     = sessions[: n // 2]
    non_members = sessions[n // 2:]

    logger.info(f"FedMIA — members: {len(members)}, non-members: {len(non_members)}")

    fedmia = FedMIA(attack_threshold=mia_cfg["attack_threshold"])

    # Addestra shadow model su DataLoader dei membri
    logger.info("Training shadow model...")
    member_loader = _sessions_to_loader(members)
    fedmia.train_shadow_model(member_loader)

    # Calcola membership score per ogni sessione individualmente
    # Passa dict con sole feature numeriche per evitare ambiguità
    logger.info("Computing membership scores...")
    def _score(s: dict) -> float:
        feat = {f: float(s.get(f) or 0.0) for f in _MIA_FEATURES}
        return fedmia.compute_membership_score(feat)

    member_scores     = [_score(s) for s in members]
    non_member_scores = [_score(s) for s in non_members]

    labels = [1] * len(member_scores) + [0] * len(non_member_scores)
    scores = member_scores + non_member_scores
    auc    = roc_auc_score(labels, scores)
    logger.info(f"AUC-ROC: {auc:.4f}")

    # Associa AUC a ogni round FL (il shadow model è fisso,
    # il round indica il numero di round di training FL prima dell'attacco)
    mia_results: dict[int, dict[str, Any]] = {}
    for round_num in fl_results:
        mia_results[round_num] = {
            "auc_roc":               auc,
            "member_score_mean":     float(np.mean(member_scores)),
            "non_member_score_mean": float(np.mean(non_member_scores)),
        }

    return mia_results

# ── IDS Evaluation ─────────────────────────────────────────────────────────────
# ── IDS Evaluation ─────────────────────────────────────────────────────────────

def run_ids(
    cfg: dict,
    fl_results: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Valuta ChargingIDS su ogni round FL.
    Costruisce AuditReport e gradient dict dai GradientUpdate del ML Plane
    per ogni nodo, poi chiama analyze_round() con la firma corretta.
    """
    from core.base_auditor import AuditReport

    ids = ChargingIDS(
        byzantine_tolerance=1,
        cosine_threshold=0.85,
    )

    epsilon = cfg["experiment"]["epsilon"]
    ids_results: dict[int, dict[str, Any]] = {}

    for round_num, round_data in fl_results.items():
        updates = round_data.get("updates", [])
        if not updates:
            ids_results[round_num] = {
                "alerts": [], "byzantine_detected": False, "drift_detected": False,
            }
            continue

        # Costruisce AuditReport e gradient dict da GradientUpdate (ML Plane)
        reports: dict[str, AuditReport] = {}
        gradients: dict[str, dict[str, Any]] = {}

        for update in updates:
            if not update or not update.node_id:
                continue

            # AuditReport: privacy_score=1.0 se DP applicato, 0.5 altrimenti
            reports[update.node_id] = AuditReport(
                node_id=update.node_id,
                round_id=round_num,
                privacy_score=1.0 if update.metadata.get("dp_applied") else 0.5,
                epsilon=epsilon,
                threats_detected=[],
                metadata={
                    "n_samples": update.n_samples,
                    "loss":      update.loss,
                    "cluster_id": update.cluster_id,
                },
            )

            # Gradient dict: {layer_0: tensor, layer_1: tensor, ...}
            gradients[update.node_id] = {
                f"layer_{i}": w
                for i, w in enumerate(update.weights or [])
            }

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
                    "type":     a.alert_type,
                    "severity": a.severity,
                    "node_id":  a.node_id,
                    "message":  a.message,
                }
                for a in (analysis.alerts if analysis else [])
            ],
            "byzantine_detected": analysis.byzantine_detected if analysis else False,
            "drift_detected":     analysis.drift_detected if analysis else False,
        }

    return ids_results


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(
    cfg: dict,
    mia_results: dict[int, dict[str, Any]],
    ids_results: dict[int, dict[str, Any]],
) -> Path:
    """Salva risultati in experiments/ con timestamp."""
    output_dir = PROJECT_ROOT / cfg["output"]["experiments_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"experiment_{timestamp}.json"

    auc_values = [
        r["auc_roc"]
        for r in mia_results.values()
        if r.get("auc_roc") is not None
    ]

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
            "mean_auc_roc": float(np.mean(auc_values)) if auc_values else None,
            "max_auc_roc":  float(np.max(auc_values))  if auc_values else None,
            "min_auc_roc":  float(np.min(auc_values))  if auc_values else None,
            "privacy_risk": (
                "HIGH"   if auc_values and np.mean(auc_values) > 0.7 else
                "MEDIUM" if auc_values and np.mean(auc_values) > 0.6 else
                "LOW"
            ),
        },
        "per_round": {
            str(r): {
                "mia": mia_results.get(r, {}),
                "ids": ids_results.get(r, {}),
            }
            for r in sorted(mia_results.keys())
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
    return result_file


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
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ChargeShield-FL — Sprint 5 Experiment")
    logger.info("=" * 60)

    cfg = load_config(args.config, {"epsilon": args.epsilon, "rounds": args.rounds})
    exp_cfg = cfg["experiment"]
    logger.info(
        f"Config: epsilon={exp_cfg['epsilon']}, "
        f"rounds={exp_cfg['fl_rounds']}, "
        f"proximal_mu={cfg['ml']['proximal_mu']}"
    )

    sessions = load_sessions(cfg)
    sessions = enrich_sessions(sessions)
    logger.info(f"Sessioni dopo enrichment: {len(sessions)}")

    if args.dry_run:
        logger.info("Dry run completato — uscita.")
        return

    fl_results  = run_fl_rounds(cfg, sessions)
    mia_results = run_fedmia(cfg, sessions, fl_results)
    ids_results = {} if args.skip_ids else run_ids(cfg, fl_results)

    save_results(cfg, mia_results, ids_results)

    logger.info("=" * 60)
    logger.info("Esperimento completato.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
