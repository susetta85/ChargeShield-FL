# tests/test_run_experiments_integration.py
"""
Integration tests for the core experiment pipeline in scripts/run_experiments.py:
run_fl_rounds(), run_fedmia(), run_lira(), run_ids().

These tests use small synthetic session data (no real ACN-Data files, no network
access) with tiny epoch/round/shadow counts so the full FL + attack pipeline runs
in a few seconds on CPU, while still exercising the real code paths.

Coverage rationale
-------------------
Beyond basic "does it run and return the right shape" checks, several tests are
regression tests for bugs found and fixed on 2026-07-21 in run_lira() and in the
IDS calibration (Sprint 9):

  - test_no_dp_updates_match_raw_updates / test_dp_perturbs_updates_vs_raw:
        guards the DP on/off switch in run_fl_rounds() itself.
  - test_lira_attacks_post_dp_updates_not_raw:
        regression test for "Fix c" (2026-07-21c) — LiRA used to attack
        raw_updates (captured before gm.privatize()), so DP noise never reached
        what LiRA attacked, by construction. This test builds fl_results by hand
        with deliberately different `updates` vs `raw_updates` per round and
        checks run_lira() reads `updates`, not `raw_updates`.
  - test_run_ids_no_false_positive_without_attack / test_run_ids_detects_byzantine_gradient_scaling:
        regression tests for the GRADIENT_EXPLOSION / Krum threshold calibration
        (Sprint 9) — no attack must not alert, a 10x gradient-scaling attack must.
  - test_run_ids_no_dp_disables_budget_exhausted:
        regression test for the no_dp=True / epsilon=1000 budget-exhaustion guard.

Caveats
-------
These tests require torch/scikit-learn to be installed (see pyproject.toml).
They were authored and syntax-checked (`python3 -m py_compile`) in a sandbox that
cannot install torch, so they have NOT been executed here — run them locally with:

    pytest tests/test_run_experiments_integration.py -v
"""

from __future__ import annotations

import copy
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiments as run_exp  # noqa: E402
from ml.base_ml import GradientUpdate  # noqa: E402


# ── Synthetic data ──────────────────────────────────────────────────────────────

_FEATURES = [
    "total_energy_kwh", "max_power_kw", "kwh_requested",
    "minutes_available", "hour_of_day", "duration_hours",
]


def _make_sessions(n: int, seed: int) -> list[dict[str, Any]]:
    """
    Genera n sessioni sintetiche con feature numeriche + start/end_time.
    enrich_sessions() calcola hour_of_day/duration_hours da start/end_time,
    quindi qui basta fornire timestamp ISO validi.
    """
    rng = random.Random(seed)
    base = datetime(2020, 1, 1)
    sessions = []
    for i in range(n):
        start = base + timedelta(hours=rng.randint(0, 24 * 60), minutes=rng.randint(0, 59))
        dur_h = rng.uniform(0.5, 8.0)
        end = start + timedelta(hours=dur_h)
        sessions.append({
            "session_id":        f"synthetic-{seed}-{i}",
            "start_time":        start.isoformat(),
            "end_time":          end.isoformat(),
            "total_energy_kwh":  rng.uniform(1.0, 30.0),
            "max_power_kw":      rng.uniform(3.0, 22.0),
            "kwh_requested":     rng.uniform(1.0, 30.0),
            "minutes_available": rng.uniform(30, 600),
        })
    return run_exp.enrich_sessions(sessions)


@pytest.fixture(scope="module")
def tiny_cfg() -> dict:
    """
    Config minima ma realistica: stessa struttura di config/experiment.yaml,
    con fl_rounds/epochs/n_shadow ridotti per rendere il test veloce (~secondi),
    non minuti come nei run reali (epochs=50, n_shadow=8-32).
    """
    return {
        "experiment": {
            "name": "pytest_integration",
            "fl_rounds": 3,
            "seed": 42,
            "epsilon": 1.0,
            "delta": 1.0e-5,
            "max_grad_norm": 1.0,
        },
        "ml": {
            "input_dim": 6,
            "lr": 0.01,
            "epochs": 2,
            "batch_size": 16,
            "proximal_mu": 0.0,
        },
        "lira": {"n_shadow": 2},
        "byzantine_attack": {"enabled": False},
    }


@pytest.fixture(scope="module")
def train_sessions() -> list[dict[str, Any]]:
    # 4 cluster × 40 sessioni = 160, sufficiente per split contiguo in run_fl_rounds.
    return _make_sessions(160, seed=1)


@pytest.fixture(scope="module")
def holdout_sessions() -> list[dict[str, Any]]:
    # Pool distinto (seed diverso) — non deve intersecare train_sessions.
    return _make_sessions(40, seed=2)


# ── run_fl_rounds() ──────────────────────────────────────────────────────────────

class TestRunFLRounds:
    def test_structure_and_participants(self, tiny_cfg, train_sessions):
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)

        fl_rounds = cfg["experiment"]["fl_rounds"]
        # Round 0 = init weights baseline (usato da IDS), round 1..N = training reale.
        assert 0 in fl_results
        assert fl_results[0]["raw_global_weights"] is not None
        for r in range(1, fl_rounds + 1):
            assert r in fl_results, f"round {r} mancante da fl_results"
            rd = fl_results[r]
            assert rd["n_participants"] == 4, "4 cluster (highway/urban/residential/corporate)"
            assert len(rd["updates"]) == 4
            assert len(rd["raw_updates"]) == 4
            assert rd["global_weights"] is not None
            assert rd["mean_loss"] is not None and rd["mean_loss"] >= 0.0

    def test_no_dp_updates_match_raw_updates(self, tiny_cfg, train_sessions):
        """Con no_dp=True, updates (post-privatize) devono coincidere con raw_updates
        (pre-privatize) — nessun rumore deve essere applicato."""
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
        rd = fl_results[1]
        for priv, raw in zip(rd["updates"], rd["raw_updates"]):
            for w_priv, w_raw in zip(priv.weights, raw.weights):
                assert (w_priv == w_raw).all(), (
                    "con no_dp=True, updates deve essere identico a raw_updates "
                    "(nessun rumore DP applicato)"
                )

    def test_dp_perturbs_updates_vs_raw(self, tiny_cfg, train_sessions):
        """Con no_dp=False, updates (post-privatize) deve DIFFERIRE da raw_updates:
        altrimenti il rumore DP non viene mai applicato (regressione del bug LiRA
        2026-07-21c, dove raw_updates non portava mai il rumore DP per costruzione)."""
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=False)
        rd = fl_results[1]
        any_diff = False
        for priv, raw in zip(rd["updates"], rd["raw_updates"]):
            for w_priv, w_raw in zip(priv.weights, raw.weights):
                if not (w_priv == w_raw).all():
                    any_diff = True
        assert any_diff, (
            "con no_dp=False (DP attivo), updates deve differire da raw_updates "
            "almeno in qualche peso — altrimenti il rumore DP non sta venendo applicato"
        )

    def test_central_dp_clips_per_client_but_does_not_noise_them(
        self, tiny_cfg, train_sessions
    ):
        """
        Regressione/verifica per dp_mode='central' (2026-07-22): ogni client deve
        CLIPPARE il proprio update (norma L2 <= max_grad_norm + tolleranza) ma NON
        rumorizzarlo individualmente — il rumore va solo sull'aggregato FedAvg
        (verificato separatamente in test_central_dp_noises_only_the_aggregate).

        Un update rumorizzato (dp_mode="dp-fedavg") avrebbe invece norma L2 ben
        superiore a max_grad_norm, perché sigma (~4.8 per epsilon=1.0 di default)
        domina rispetto al clipping a 1.0 — la differenza di scala è netta e
        rende questo un buon discriminante.
        """
        cfg = copy.deepcopy(tiny_cfg)
        max_grad_norm = cfg["experiment"]["max_grad_norm"]
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=False, dp_mode="central")
        rd = fl_results[1]
        for update in rd["updates"]:
            float_weights = [w for w in update.weights if w.is_floating_point()]
            l2 = sum(float(w.float().norm() ** 2) for w in float_weights) ** 0.5
            assert l2 <= max_grad_norm + 1e-3, (
                f"dp_mode='central': la norma L2 dell'update del client {update.node_id} "
                f"è {l2:.4f}, oltre max_grad_norm={max_grad_norm} + tolleranza — "
                "suggerisce che sia stato rumorizzato individualmente invece di "
                "solo clippato (clip_only())"
            )

    def test_central_dp_noises_only_the_aggregate(self, tiny_cfg, train_sessions):
        """dp_mode='central': il modello globale POST-aggregazione deve differire
        dalla media pesata pulita degli update raw — il rumore va sull'aggregato,
        non sui singoli update (verificato in test_central_dp_clips_per_client_...)."""
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=False, dp_mode="central")
        rd = fl_results[1]
        raw_avg = rd["raw_global_weights"]
        noised_global = rd["global_weights"]
        assert raw_avg is not None and noised_global is not None
        any_diff = any(
            not (torch_w_a == torch_w_b).all()
            for torch_w_a, torch_w_b in zip(raw_avg, noised_global)
            if torch_w_a.is_floating_point()
        )
        assert any_diff, (
            "dp_mode='central': il modello globale aggregato deve differire dalla "
            "media raw — altrimenti privatize_aggregate() non sta aggiungendo rumore"
        )

    def test_local_dp_hides_raw_updates_from_ids(self, tiny_cfg, train_sessions):
        """dp_mode='local' (2026-07-22): fl_results non deve contenere raw_updates/
        raw_global_weights per i round > 0 — il server non deve mai vedere il
        valore pulito, nemmeno transitoriamente (run_ids() userà updates via il
        fallback esistente `raw_updates or updates`)."""
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=False, dp_mode="local")
        for round_num, rd in fl_results.items():
            if round_num == 0:
                continue  # round 0 = baseline init, non soggetto a dp_mode
            assert rd.get("raw_updates") is None, (
                f"round {round_num}: dp_mode='local' non deve esporre raw_updates"
            )
            assert rd.get("raw_global_weights") is None, (
                f"round {round_num}: dp_mode='local' non deve esporre raw_global_weights"
            )
            # Ma gli update privatizzati (quelli che il client invia davvero) devono
            # comunque esistere e portare il rumore, come in dp-fedavg.
            assert rd.get("updates"), f"round {round_num}: updates mancanti"


# ── run_fedmia() (Yeom 2018, global model) ──────────────────────────────────────

class TestRunFedMIA:
    def test_auc_in_valid_range(self, tiny_cfg, train_sessions, holdout_sessions):
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
        mia_results = run_exp.run_fedmia(cfg, train_sessions, holdout_sessions, fl_results)

        assert mia_results, "run_fedmia non ha prodotto risultati per nessun round"
        for round_num, rd in mia_results.items():
            auc = rd["auc_roc"]
            assert 0.0 <= auc <= 1.0, f"round {round_num}: AUC {auc} fuori [0,1]"

    def test_members_and_non_members_must_differ(self, tiny_cfg, train_sessions, holdout_sessions):
        """members e non_members devono restare pool disgiunti — la docstring di
        run_fedmia() è esplicita: usare lo stesso pool per entrambi invalida l'AUC."""
        train_ids = {s["session_id"] for s in train_sessions}
        holdout_ids = {s["session_id"] for s in holdout_sessions}
        assert train_ids.isdisjoint(holdout_ids)


# ── run_lira() (Carlini 2022, per-client update) ────────────────────────────────

class TestRunLiRA:
    def test_auc_in_valid_range_and_covers_all_rounds(
        self, tiny_cfg, train_sessions, holdout_sessions
    ):
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
        lira_results = run_exp.run_lira(
            cfg, train_sessions, holdout_sessions, fl_results,
            n_shadow=cfg["lira"]["n_shadow"], shadow_epochs_cap=2, no_dp=True,
        )
        fl_rounds = cfg["experiment"]["fl_rounds"]
        for r in range(1, fl_rounds + 1):
            assert r in lira_results, f"LiRA non ha prodotto risultati per il round {r}"
            auc = lira_results[r]["lira_auc_roc"]
            assert 0.0 <= auc <= 1.0, f"round {r}: LiRA AUC {auc} fuori [0,1]"

    def test_scores_do_not_saturate_the_clip_ceiling(
        self, tiny_cfg, train_sessions, holdout_sessions
    ):
        """
        Regressione bug 2026-07-21e (varianza IN/OUT asimmetrica per i non-member):
        prima del fix, lira_non_member_score_mean saturava vicino al tetto di
        clipping (+20) subito dopo un round warm-started (osservato: +17.9 al
        round 2 su nodp-sweep2 reale), perché sigma_out per-campione (N=n_shadow,
        piccolo) poteva collassare quasi a zero mentre sigma_in usava il fallback
        pooled (molto più ampio) — il rapporto di verosimiglianza finiva dominato
        dal termine 1/sigma^2 invece che dal segnale di membership reale.
        Con il pavimento di varianza pooled (global_out_stats_per_cluster) né
        lira_member_score_mean né lira_non_member_score_mean dovrebbero avvicinarsi
        al tetto di clipping (±20) in un esperimento sintetico piccolo e pulito.
        """
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
        lira_results = run_exp.run_lira(
            cfg, train_sessions, holdout_sessions, fl_results,
            n_shadow=cfg["lira"]["n_shadow"], shadow_epochs_cap=2, no_dp=True,
        )
        for r, rd in lira_results.items():
            for key in ("lira_member_score_mean", "lira_non_member_score_mean"):
                val = rd[key]
                assert abs(val) < 15.0, (
                    f"round {r}: {key}={val} troppo vicino al tetto di clipping "
                    "(±20) — possibile regressione del fix 2026-07-21e "
                    "(collasso di varianza IN/OUT per i non-member)"
                )

    def test_lira_attacks_post_dp_updates_not_raw(
        self, tiny_cfg, train_sessions, holdout_sessions
    ):
        """
        Regressione bug 2026-07-21c: LiRA deve leggere round_data["updates"]
        (post-privatize), non round_data["raw_updates"] (pre-privatize).

        Costruisce fl_results con updates e raw_updates DELIBERATAMENTE diversi
        (raw_updates = pesi originali, updates = pesi scalati ×1000) e verifica
        che l'output di LiRA cambi se si scambiano i due campi — cioè che la
        funzione stia effettivamente leggendo "updates" e non "raw_updates".
        """
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)

        def _scaled_copy(fl_results_in: dict, factor: float) -> dict:
            """Copia fl_results sostituendo SOLO 'updates' con pesi scalati ×factor,
            lasciando 'raw_updates' invariato — così i due campi divergono
            deliberatamente e possiamo verificare quale dei due LiRA legge."""
            out = {}
            for r, rd in fl_results_in.items():
                rd2 = dict(rd)
                if rd2.get("updates"):
                    rd2["updates"] = [
                        GradientUpdate(
                            node_id=u.node_id, cluster_id=u.cluster_id,
                            round_num=u.round_num,
                            weights=[w * factor for w in u.weights],
                            gradients=u.gradients, loss=u.loss,
                            n_samples=u.n_samples, metadata=u.metadata,
                        )
                        for u in rd2["updates"]
                    ]
                out[r] = rd2
            return out

        fl_results_scaled = _scaled_copy(fl_results, 1000.0)

        lira_normal = run_exp.run_lira(
            cfg, train_sessions, holdout_sessions, fl_results,
            n_shadow=2, shadow_epochs_cap=2, no_dp=True,
        )
        lira_scaled = run_exp.run_lira(
            cfg, train_sessions, holdout_sessions, fl_results_scaled,
            n_shadow=2, shadow_epochs_cap=2, no_dp=True,
        )

        assert lira_normal[1]["lira_auc_roc"] != lira_scaled[1]["lira_auc_roc"], (
            "LiRA ha prodotto lo stesso AUC con updates normali e updates scalati "
            "×1000 — sospetto che stia ancora leggendo raw_updates invece di "
            "updates (regressione del bug 2026-07-21c)"
        )


# ── run_ids() (ChargingIDS + PrivacyAuditor) ────────────────────────────────────

class TestRunIDS:
    def test_no_false_positive_without_attack(self, tiny_cfg, train_sessions):
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
        ids_results = run_exp.run_ids(cfg, fl_results, no_dp=True)

        for round_num, rd in ids_results.items():
            assert rd["byzantine_detected"] is False, (
                f"round {round_num}: falso allarme Byzantine senza alcun attacco attivo "
                "(regressione calibrazione soglia Krum, Sprint 9)"
            )

    def test_detects_byzantine_gradient_scaling(self, tiny_cfg, train_sessions):
        cfg = copy.deepcopy(tiny_cfg)
        cfg["byzantine_attack"] = {
            "enabled": True,
            "byzantine_node": "highway",
            "attack_type": "gradient_scaling",
            "scale_factor": 10.0,
        }
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
        ids_results = run_exp.run_ids(cfg, fl_results, no_dp=True)

        detected_any = any(rd["byzantine_detected"] for rd in ids_results.values())
        assert detected_any, (
            "nessun round ha rilevato il nodo Byzantine (scale_factor=10) — "
            "regressione della soglia Krum/GRADIENT_EXPLOSION (Sprint 9)"
        )

    def test_no_dp_disables_budget_exhausted_alerts(self, tiny_cfg, train_sessions):
        """Regressione Fix 3a (Sprint 9): con no_dp=True l'auditor deve ricevere
        epsilon=1000.0 in modo che il budget non si esaurisca mai artificialmente."""
        cfg = copy.deepcopy(tiny_cfg)
        fl_results = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
        ids_results = run_exp.run_ids(cfg, fl_results, no_dp=True)

        for round_num, rd in ids_results.items():
            for alert in rd["alerts"]:
                reasons = " ".join(alert.get("reasons", []))
                assert "BUDGET_EXHAUSTED" not in reasons, (
                    f"round {round_num}: falso BUDGET_EXHAUSTED con no_dp=True "
                    "(regressione Fix 3a, Sprint 9)"
                )
