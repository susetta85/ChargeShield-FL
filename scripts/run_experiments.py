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
            "mean_loss":      aggregated.mean_loss,
            "n_participants": aggregated.n_participants,
            "updates":        round_updates,
            "global_weights": aggregated.global_weights,
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
        model = Autoencoder(input_dim=input_dim)
        orig_state = model.state_dict()
        keys = list(orig_state.keys())
        state = {
            k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(orig_state[k].dtype)
            for k, w in zip(keys, global_weights)
        }
        model.load_state_dict(state, strict=True)
        model.eval()

        member_scores     = _score_batch(model, members)
        non_member_scores = _score_batch(model, non_members)

        if not member_scores or not non_member_scores:
            logger.warning(f"Round {round_num}: score batch vuoto — skip AUC")
            continue

        labels = [1] * len(member_scores) + [0] * len(non_member_scores)
        scores = member_scores + non_member_scores
        auc    = roc_auc_score(labels, scores)
        logger.info(f"Round {round_num} — FedMIA AUC-ROC: {auc:.4f}")

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

    Usa PrivacyAuditor per generare AuditReport reali con threats_detected
    popolato (GRADIENT_EXPLOSION, PRIVACY_BUDGET_EXHAUSTED, ecc.).
    Un singolo auditor persiste tra i round per tracciare l'epsilon cumulativo.
    """
    config_path = str(PROJECT_ROOT / "config" / "auditor.yaml")

    ids = ChargingIDS(
        config_path=config_path,
        byzantine_tolerance=1,
        cosine_threshold=0.85,
    )
    # Un'unica istanza traccia l'epsilon cumulativo per nodo su tutti i round
    auditor = PrivacyAuditor(config_path=config_path, epsilon=cfg["experiment"]["epsilon"])

    ids_results: dict[int, dict[str, Any]] = {}

    for round_num, round_data in fl_results.items():
        updates = round_data.get("updates", [])
        if not updates:
            ids_results[round_num] = {
                "alerts": [], "byzantine_detected": False, "drift_detected": False,
            }
            continue

        reports: dict[str, Any] = {}
        gradients: dict[str, dict[str, Any]] = {}

        for update in updates:
            if not update or not update.node_id:
                continue

            # Converti pesi in dict[layer → list[float]] per PrivacyAuditor
            model_update: dict[str, Any] = {}
            for i, w in enumerate(update.weights or []):
                if isinstance(w, torch.Tensor):
                    model_update[f"layer_{i}"] = w.flatten().tolist()
                else:
                    model_update[f"layer_{i}"] = float(w) if isinstance(w, (int, float)) else w

            # AuditReport con threats_detected reali (GRADIENT_EXPLOSION, ecc.)
            reports[update.node_id] = auditor.audit(
                node_id=update.node_id,
                round_id=round_num,
                model_update=model_update,
            )

            # Gradient dict per Krum / cosine analysis dell'IDS
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
                "fl":  {"mean_loss": (fl_results or {}).get(r, {}).get("mean_loss")},
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

    # Aggiorna automaticamente il report Excel con tutti i risultati accumulati
    _update_excel_report(output_dir)

    return result_file


def _update_excel_report(experiments_dir: Path) -> None:
    """
    Rigenera il report Excel a 6 sheet chiamando generate_excel_report.py.
    Produce: Raw Data, Heat Map, Per Rounds, Per Epsilon, Comparison, AUC Progression.
    """
    try:
        import importlib.util
        from openpyxl import Workbook

        script_path = Path(__file__).parent / "generate_excel_report.py"
        spec = importlib.util.spec_from_file_location("gen_xl", script_path)
        if spec is None or spec.loader is None:
            logger.warning(f"generate_excel_report.py non trovato o non caricabile: {script_path}")
            return
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        records = gen.load_experiments(experiments_dir)
        if not records:
            return

        wb = Workbook()
        wb.remove(wb.active)
        gen.build_raw_data(wb.create_sheet("Raw Data"),           records)
        gen.build_heat_map(wb.create_sheet("Heat Map"),           records)
        gen.build_per_rounds(wb.create_sheet("Per Rounds"),       records)
        gen.build_per_epsilon(wb.create_sheet("Per Epsilon"),     records)
        gen.build_comparison(wb.create_sheet("Comparison"),       records)
        gen.build_auc_progression(wb.create_sheet("AUC Progression"), records)
        wb.properties.title   = "ChargeShield-FL Experiment Results"
        wb.properties.subject = "FedMIA vs Differential Privacy — DSN 2027"
        output_path = experiments_dir / "ChargeShield_FL_Results.xlsx"
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

    # Split hold-out PRIMA del training FL: 80% train, 20% hold-out (mai visti dai nodi FL)
    random.shuffle(sessions)
    split = max(1, int(len(sessions) * 0.8))
    train_sessions  = sessions[:split]
    holdout_sessions = sessions[split:]
    logger.info(f"Split — train: {len(train_sessions)}, hold-out: {len(holdout_sessions)}")

    if args.dry_run:
        logger.info("Dry run completato — uscita.")
        return

    fl_results  = run_fl_rounds(cfg, train_sessions)
    mia_results = run_fedmia(cfg, train_sessions, holdout_sessions, fl_results)
    ids_results = {} if args.skip_ids else run_ids(cfg, fl_results)

    save_results(cfg, mia_results, ids_results, fl_results)

    logger.info("=" * 60)
    logger.info("Esperimento completato.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
