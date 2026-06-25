# tests/test_sprint5.py
# ChargeShield-FL — Sprint 5: ML Plane Unit Tests
#
# Testa: AbstractMLModel, AutoencoderTrainer, GradientManager, FedAvgAggregator
# Requisiti: pytest, torch

from __future__ import annotations

import math
import pytest
import torch

from ml.base_ml import (
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)
from ml.autoencoder_trainer import AutoencoderTrainer
from ml.gradient_manager import GradientManager
from ml.fedavg_aggregator import FedAvgAggregator


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def trainer_config():
    return {
        "input_dim": 7,
        "lr": 0.001,
        "epochs": 2,
        "batch_size": 4,
        "proximal_mu": 0.0,
    }

@pytest.fixture
def trainer(trainer_config):
    return AutoencoderTrainer(
        config=trainer_config,
        node_id="node_A_01",
        cluster_id="A",
    )

@pytest.fixture
def dp_config():
    return {
        "epsilon": 1.0,
        "delta": 1.0e-5,
        "max_grad_norm": 1.0,
    }

@pytest.fixture
def sessions():
    """Sessioni EV sintetiche con tutte le feature continue valide."""
    return [
        {
            "voltage_v": 230.0,
            "current_a": 16.0,
            "power_kw": 3.7,
            "energy_kwh": 10.0,
            "temperature_c": 22.0,
            "soc_percent": 80.0,
            "timestamp": 1.0,
        }
        for _ in range(20)
    ]

@pytest.fixture
def gradient_update(trainer, sessions):
    return trainer.train_local(sessions, round_num=1)


# ── AutoencoderTrainer ─────────────────────────────────────────────────────────

class TestAutoencoderTrainer:

    def test_init_fedavg(self, trainer):
        assert trainer.proximal_mu == 0.0
        assert trainer._global_weights is None

    def test_init_fedprox(self, trainer_config):
        cfg = {**trainer_config, "proximal_mu": 0.01}
        t = AutoencoderTrainer(cfg, "node_B_01", "B")
        assert t.proximal_mu == 0.01

    def test_missing_config_raises(self):
        with pytest.raises(ValueError):
            AutoencoderTrainer({"lr": 0.001, "epochs": 2, "batch_size": 4},
                               "n", "A")

    def test_get_weights_returns_list(self, trainer):
        weights = trainer.get_weights()
        assert isinstance(weights, list)
        assert len(weights) > 0
        assert all(isinstance(w, torch.Tensor) for w in weights)

    def test_set_weights_roundtrip(self, trainer):
        original = trainer.get_weights()
        # Modifica pesi
        for w in original:
            w.fill_(0.0)
        trainer.set_weights(original)
        restored = trainer.get_weights()
        for w in restored:
            assert torch.all(w == 0.0)

    def test_set_weights_saves_global(self, trainer):
        weights = trainer.get_weights()
        trainer.set_weights(weights)
        assert trainer._global_weights is not None

    def test_train_local_returns_update(self, trainer, sessions):
        update = trainer.train_local(sessions, round_num=1)
        assert isinstance(update, GradientUpdate)
        assert update.node_id == "node_A_01"
        assert update.cluster_id == "A"
        assert update.round_num == 1
        assert update.n_samples == len(sessions)
        assert update.loss is not None
        assert update.loss >= 0.0

    def test_train_local_empty_sessions(self, trainer):
        update = trainer.train_local([], round_num=1)
        assert update.n_samples == 0
        assert update.loss is None

    def test_train_local_none_feature_skipped(self, trainer):
        sessions = [
            {"voltage_v": None, "current_a": 16.0, "power_kw": 3.7,
             "energy_kwh": 10.0, "temperature_c": 22.0,
             "soc_percent": 80.0, "timestamp": 1.0},
        ]
        update = trainer.train_local(sessions, round_num=1)
        assert update.n_samples == 0

    def test_fedprox_term_applied(self, trainer_config, sessions):
        cfg = {**trainer_config, "proximal_mu": 0.1}
        t = AutoencoderTrainer(cfg, "node_C_01", "C")
        # Prima imposta global weights, poi train
        t.set_weights(t.get_weights())
        update = t.train_local(sessions, round_num=1)
        assert update.loss is not None

    def test_ml_plane_event_emitted(self, trainer, sessions):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        trainer.subscribe(Spy())
        trainer.train_local(sessions, round_num=1)
        assert len(events) == 1
        assert events[0].event_type == "gradient_upload"
        assert events[0].purdue_level == 1

    def test_apply_global_model_emits_event(self, trainer, gradient_update):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        trainer.subscribe(Spy())
        aggregated = AggregatedUpdate(
            round_num=1,
            global_weights=gradient_update.weights,
            n_participants=4,
            mean_loss=0.01,
        )
        trainer.apply_global_model(aggregated)
        assert any(e.event_type == "weight_download" for e in events)


# ── GradientManager ────────────────────────────────────────────────────────────

class TestGradientManager:

    def test_sigma_computed(self, dp_config):
        gm = GradientManager(dp_config)
        expected = (
            dp_config["max_grad_norm"]
            * math.sqrt(2 * math.log(1.25 / dp_config["delta"]))
            / dp_config["epsilon"]
        )
        assert abs(gm.sigma - expected) < 1e-6

    def test_missing_config_raises(self):
        with pytest.raises(ValueError):
            GradientManager({"epsilon": 1.0, "delta": 1e-5})

    def test_privatize_returns_update(self, dp_config, gradient_update):
        gm = GradientManager(dp_config)
        private = gm.privatize(gradient_update)
        assert isinstance(private, GradientUpdate)
        assert private.metadata.get("dp_applied") is True

    def test_privatize_changes_weights(self, dp_config, gradient_update):
        gm = GradientManager(dp_config)
        private = gm.privatize(gradient_update)
        # I pesi devono essere diversi dopo DP
        original_flat = torch.cat([w.flatten() for w in gradient_update.weights])
        private_flat  = torch.cat([w.flatten() for w in private.weights])
        assert not torch.allclose(original_flat, private_flat)

    def test_privatize_preserves_metadata(self, dp_config, gradient_update):
        gm = GradientManager(dp_config)
        private = gm.privatize(gradient_update)
        assert private.node_id    == gradient_update.node_id
        assert private.cluster_id == gradient_update.cluster_id
        assert private.round_num  == gradient_update.round_num
        assert private.n_samples  == gradient_update.n_samples

    def test_ml_plane_event_emitted(self, dp_config, gradient_update):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        gm = GradientManager(dp_config)
        gm.subscribe(Spy())
        gm.privatize(gradient_update)
        assert len(events) == 1
        assert events[0].purdue_level == 2

    def test_clipping_reduces_norm(self, dp_config, gradient_update):
        gm = GradientManager(dp_config)
        clipped = gm._clip_weights(gradient_update.weights)
        flat = torch.cat([w.flatten() for w in clipped])
        norm = float(torch.norm(flat, p=2))
        assert norm <= dp_config["max_grad_norm"] + 1e-5


# ── FedAvgAggregator ───────────────────────────────────────────────────────────

class TestFedAvgAggregator:

    def _make_update(self, node_id: str, cluster_id: str,
                     weights: list, n_samples: int, loss: float) -> GradientUpdate:
        return GradientUpdate(
            node_id=node_id,
            cluster_id=cluster_id,
            round_num=1,
            weights=weights,
            gradients=None,
            loss=loss,
            n_samples=n_samples,
        )

    def test_aggregate_returns_result(self, trainer):
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        result = agg.aggregate(round_num=1)
        assert result is not None
        assert result.n_participants == 2
        assert result.round_num == 1

    def test_aggregate_weighted_average(self, trainer):
        agg = FedAvgAggregator({"min_participants": 2})
        w1 = [torch.ones(p.shape) * 0.0 for p in trainer.model.parameters()]
        w2 = [torch.ones(p.shape) * 1.0 for p in trainer.model.parameters()]
        agg.collect(self._make_update("n1", "A", w1, 50, 0.1))
        agg.collect(self._make_update("n2", "B", w2, 50, 0.2))
        result = agg.aggregate(round_num=1)
        # Media pesata 50/50 → 0.5
        for w in result.global_weights:
            assert torch.allclose(w, torch.ones_like(w) * 0.5, atol=1e-5)

    def test_aggregate_below_min_returns_none(self, trainer):
        agg = FedAvgAggregator({"min_participants": 3})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        result = agg.aggregate(round_num=1)
        assert result is None

    def test_aggregate_clears_buffer(self, trainer):
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        agg.aggregate(round_num=1)
        # Secondo aggregate senza nuovi update → None
        result = agg.aggregate(round_num=2)
        assert result is None

    def test_mean_loss_weighted(self, trainer):
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.3))
        result = agg.aggregate(round_num=1)
        assert result.mean_loss is not None
        assert abs(result.mean_loss - 0.2) < 1e-5

    def test_ml_plane_event_emitted(self, trainer):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        agg = FedAvgAggregator({"min_participants": 2})
        agg.subscribe(Spy())
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        agg.aggregate(round_num=1)
        assert len(events) == 1
        assert events[0].event_type == "aggregation"
        assert events[0].purdue_level == 3
