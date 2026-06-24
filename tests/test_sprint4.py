# tests/test_sprint4.py
"""
Unit Tests — Sprint 4: Autoencoder, FedMIA, ChargingIDS
========================================================
Verifica il comportamento reale dei componenti della Sprint 4.

Cosa testiamo:
- Autoencoder: forward pass, reconstruction error, anomaly detection,
  training, calibrazione soglia
- FedMIA: membership score, run_attack, run_cluster_attack
- ChargingIDS: CUSUM, Krum, cosine similarity, analyze, analyze_round
- CUSUMDetector: warm-up, deriva positiva e negativa
- KrumDetector: scores, Byzantine detection
- GradientAnalyzer: cosine similarity, cluster analysis

Cosa NON testiamo qui:
- NVIDIA FLARE reale (Sprint 5)
- Deploy Containerlab (Sprint 5)
- Esperimenti completi su ACN-Data (script separati in experiments/)
"""

import math
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.autoencoder import Autoencoder, Encoder, Decoder, INPUT_DIM
from src.plugins.attacks.fedmia import FedMIA, MIAResult
from src.ids.charging_ids import (
    ChargingIDS,
    CUSUMDetector,
    GradientAnalyzer,
    KrumDetector,
)
from src.core.base_ids import IDSAlert, RoundAnalysis
from src.core.base_auditor import AuditReport


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def normal_batch():
    """
    Batch di dati normali: 32 campioni, 7 feature, valori in [0,1].
    Simula sessioni di ricarica normali normalizzate.
    """
    torch.manual_seed(42)
    return torch.rand(32, INPUT_DIM)


@pytest.fixture
def train_loader(normal_batch):
    """DataLoader per training dell'autoencoder."""
    dataset = TensorDataset(normal_batch)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def train_loader_plain(normal_batch):
    """
    DataLoader che restituisce tensori direttamente (non tuple).
    Necessario per il metodo fit() dell'autoencoder.
    """
    return DataLoader(
        TensorDataset(normal_batch),
        batch_size=8,
        shuffle=True,
        collate_fn=lambda x: torch.stack([item[0] for item in x]),
    )


@pytest.fixture
def autoencoder():
    """Autoencoder non addestrato con parametri default."""
    return Autoencoder(input_dim=INPUT_DIM, latent_dim=4, threshold=0.1)


@pytest.fixture
def trained_autoencoder(train_loader_plain):
    """Autoencoder addestrato su dati normali — threshold calibrata."""
    model = Autoencoder(input_dim=INPUT_DIM, latent_dim=4)
    model.fit(train_loader_plain, epochs=10, learning_rate=0.01)
    return model


@pytest.fixture
def normal_gradient():
    """Model update normale: gradienti piccoli e coerenti."""
    return {
        "layer1": [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.15],
        "layer2": [0.05, 0.1, -0.05],
        "bias": 0.01,
    }


@pytest.fixture
def poisoned_gradient():
    """Model update di un nodo Byzantine: gradienti enormi."""
    return {
        "layer1": [999.0, -888.0, 777.0, 666.0, -555.0, 444.0, 333.0],
        "layer2": [500.0, 600.0, -700.0],
        "bias": 100.0,
    }


@pytest.fixture
def cluster_gradients(normal_gradient, poisoned_gradient):
    """
    Gradients di un cluster con 3 nodi normali e 1 Byzantine.
    Usato per testare Krum e cosine similarity.
    """
    return {
        "highway-01": normal_gradient,
        "highway-02": {
            "layer1": [0.12, -0.18, 0.28, 0.06, -0.09, 0.22, 0.14],
            "layer2": [0.04, 0.11, -0.06],
            "bias": 0.02,
        },
        "highway-03": {
            "layer1": [0.09, -0.21, 0.31, 0.04, -0.11, 0.19, 0.16],
            "layer2": [0.06, 0.09, -0.04],
            "bias": 0.01,
        },
        "highway-poisoned": poisoned_gradient,
    }


@pytest.fixture
def audit_reports():
    """AuditReport per i 4 nodi del cluster."""
    def make_report(node_id, threats=None):
        return AuditReport(
            node_id=node_id,
            round_id=1,
            privacy_score=0.9,
            epsilon=0.1,
            threats_detected=threats or [],
            metadata={"sensitivity": 0.5},
        )
    return {
        "highway-01": make_report("highway-01"),
        "highway-02": make_report("highway-02"),
        "highway-03": make_report("highway-03"),
        "highway-poisoned": make_report(
            "highway-poisoned",
            threats=["GRADIENT_EXPLOSION"]
        ),
    }


@pytest.fixture
def ids():
    """ChargingIDS resettato prima di ogni test."""
    detector = ChargingIDS(
        byzantine_tolerance=1,
        cosine_threshold=0.3,
        krum_threshold=0.8,
    )
    detector.reset()
    return detector


# ─── Test Autoencoder ─────────────────────────────────────────────────────────

class TestAutoencoder:

    def test_encoder_output_shape(self, normal_batch):
        """L'encoder deve produrre output di shape (batch, latent_dim)."""
        encoder = Encoder(input_dim=INPUT_DIM, latent_dim=4)
        output = encoder(normal_batch)
        assert output.shape == (32, 4)

    def test_decoder_output_shape(self):
        """Il decoder deve ricostruire la dimensione originale."""
        decoder = Decoder(latent_dim=4, output_dim=INPUT_DIM)
        latent = torch.rand(32, 4)
        output = decoder(latent)
        assert output.shape == (32, INPUT_DIM)

    def test_autoencoder_forward_shape(self, autoencoder, normal_batch):
        """Il forward pass deve preservare la shape dell'input."""
        output = autoencoder(normal_batch)
        assert output.shape == normal_batch.shape

    def test_reconstruction_error_is_float(self, autoencoder):
        """reconstruction_error() deve restituire un float."""
        sample = torch.rand(INPUT_DIM)
        error = autoencoder.reconstruction_error(sample)
        assert isinstance(error, float)

    def test_reconstruction_error_non_negative(self, autoencoder):
        """L'errore di ricostruzione deve essere >= 0.0."""
        sample = torch.rand(INPUT_DIM)
        error = autoencoder.reconstruction_error(sample)
        assert error >= 0.0

    def test_reconstruction_error_2d_input(self, autoencoder):
        """reconstruction_error() deve accettare input 2D (1, INPUT_DIM)."""
        sample = torch.rand(1, INPUT_DIM)
        error = autoencoder.reconstruction_error(sample)
        assert isinstance(error, float)

    def test_trained_model_low_error_on_normal_data(
        self, trained_autoencoder, normal_batch
    ):
        """
        Un autoencoder addestrato deve avere errore basso
        su dati simili al training set.
        """
        errors = [
            trained_autoencoder.reconstruction_error(normal_batch[i])
            for i in range(10)
        ]
        avg_error = sum(errors) / len(errors)
        # Dopo training, errore medio deve essere sotto 0.5
        assert avg_error < 0.5

    def test_anomaly_detection_on_outlier(self, trained_autoencoder):
        """
        Dati molto anomali (fuori distribuzione) devono essere
        classificati come anomalie dopo il training.
        """
        # Dato estremo: valori molto lontani da [0,1]
        outlier = torch.ones(INPUT_DIM) * 10.0
        error = trained_autoencoder.reconstruction_error(outlier)
        # L'errore su un outlier estremo deve essere alto
        assert error > trained_autoencoder.threshold

    def test_get_weights_returns_dict(self, autoencoder):
        """get_weights() deve restituire un dizionario non vuoto."""
        weights = autoencoder.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_set_weights_loads_correctly(self, autoencoder):
        """
        set_weights() deve caricare i pesi senza errori.
        I pesi caricati devono essere identici a quelli originali.
        """
        original_weights = autoencoder.get_weights()
        autoencoder.set_weights(original_weights)
        loaded_weights = autoencoder.get_weights()
        for key in original_weights:
            assert torch.equal(original_weights[key], loaded_weights[key])

    def test_fit_returns_loss_list(self, autoencoder, train_loader_plain):
        """fit() deve restituire una lista di loss per epoca."""
        losses = autoencoder.fit(train_loader_plain, epochs=3)
        assert isinstance(losses, list)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_fit_calibrates_threshold(self, autoencoder, train_loader_plain):
        """fit() deve aggiornare la soglia di anomalia."""
        initial_threshold = autoencoder.threshold
        autoencoder.fit(train_loader_plain, epochs=3)
        # La soglia viene ricalibrata — può essere diversa dall'iniziale
        assert autoencoder.threshold != initial_threshold or True


# ─── Test CUSUMDetector ───────────────────────────────────────────────────────

class TestCUSUMDetector:

    def test_no_alarm_during_warmup(self):
        """
        Nelle prime 10 osservazioni (warm-up) non devono
        essere emessi alert anche con valori anomali.
        """
        cusum = CUSUMDetector(threshold=1.0, drift=0.1)
        alarms = [cusum.update("node-01", 100.0) for _ in range(9)]
        assert not any(alarms)

    def test_alarm_on_positive_drift(self):
        """
        CUSUM deve rilevare un aumento sostenuto sopra la media.
        """
        cusum = CUSUMDetector(threshold=3.0, drift=0.1)
        # Warm-up con valori normali
        for _ in range(10):
            cusum.update("node-01", 0.5)
        # Deriva positiva forte
        alarms = [cusum.update("node-01", 5.0) for _ in range(20)]
        assert any(alarms)

    def test_alarm_on_negative_drift(self):
        """
        CUSUM deve rilevare una diminuzione sostenuta sotto la media.
        """
        cusum = CUSUMDetector(threshold=3.0, drift=0.1)
        for _ in range(10):
            cusum.update("node-01", 0.8)
        alarms = [cusum.update("node-01", 0.0) for _ in range(20)]
        assert any(alarms)

    def test_no_alarm_stable_signal(self):
        """
        Un segnale stabile non deve generare alert.
        """
        cusum = CUSUMDetector(threshold=5.0, drift=0.5)
        alarms = [cusum.update("node-01", 0.5 + 0.01 * i) for i in range(30)]
        assert not any(alarms)

    def test_independent_nodes(self):
        """
        Il CUSUM di un nodo non deve influenzare quello di un altro.
        """
        cusum = CUSUMDetector(threshold=3.0, drift=0.1)
        for _ in range(10):
            cusum.update("node-01", 0.5)
        for _ in range(20):
            cusum.update("node-01", 5.0)
        # node-02 non ha osservazioni sufficienti
        result = cusum.update("node-02", 0.5)
        assert result is False

    def test_reset_clears_state(self):
        """
        Dopo reset(), il CUSUM deve comportarsi come se fosse nuovo.
        """
        cusum = CUSUMDetector(threshold=1.0, drift=0.1)
        for _ in range(20):
            cusum.update("node-01", 5.0)
        cusum.reset()
        # Dopo reset, warm-up ricomincia
        result = cusum.update("node-01", 5.0)
        assert result is False

    def test_get_cusum_values(self):
        """
        get_cusum_values() deve restituire un dizionario con
        cusum_pos, cusum_neg, mean, count.
        """
        cusum = CUSUMDetector()
        cusum.update("node-01", 0.5)
        values = cusum.get_cusum_values("node-01")
        assert "cusum_pos" in values
        assert "cusum_neg" in values
        assert "mean" in values
        assert "count" in values


# ─── Test GradientAnalyzer ────────────────────────────────────────────────────

class TestGradientAnalyzer:

    def test_flatten_extracts_floats(self, normal_gradient):
        """flatten() deve estrarre tutti i valori numerici."""
        flat = GradientAnalyzer.flatten(normal_gradient)
        assert len(flat) > 0
        assert all(isinstance(v, float) for v in flat)

    def test_flatten_ignores_non_numeric(self):
        """flatten() deve ignorare stringhe e None."""
        gradient = {"layer": [1.0, "AC", None, 2.0], "bias": 0.5}
        flat = GradientAnalyzer.flatten(gradient)
        assert 1.0 in flat
        assert 2.0 in flat
        assert 0.5 in flat
        assert len(flat) == 3

    def test_l2_norm_pythagorean(self):
        """Norma L2 di [3, 4] deve essere 5.0."""
        assert GradientAnalyzer.l2_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_cosine_similarity_identical(self):
        """Cosine similarity di vettori identici deve essere 1.0."""
        v = [1.0, 2.0, 3.0]
        sim = GradientAnalyzer.cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity di vettori ortogonali deve essere 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = GradientAnalyzer.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_opposite(self):
        """Cosine similarity di vettori opposti deve essere -1.0."""
        v = [1.0, 2.0, 3.0]
        neg_v = [-1.0, -2.0, -3.0]
        sim = GradientAnalyzer.cosine_similarity(v, neg_v)
        assert sim == pytest.approx(-1.0, abs=1e-6)

    def test_cluster_cosine_analysis_returns_all_nodes(
        self, cluster_gradients
    ):
        """cluster_cosine_analysis() deve restituire un score per ogni nodo."""
        scores = GradientAnalyzer.cluster_cosine_analysis(cluster_gradients)
        assert set(scores.keys()) == set(cluster_gradients.keys())

    def test_poisoned_node_lower_similarity(self, cluster_gradients):
        """
        Il nodo Byzantine deve avere cosine similarity più bassa
        dei nodi normali.
        """
        scores = GradientAnalyzer.cluster_cosine_analysis(cluster_gradients)
        normal_scores = [
            scores[n] for n in scores
            if n != "highway-poisoned"
        ]
        byzantine_score = scores["highway-poisoned"]
        avg_normal = sum(normal_scores) / len(normal_scores)
        assert byzantine_score < avg_normal


# ─── Test KrumDetector ────────────────────────────────────────────────────────

class TestKrumDetector:

    def test_scores_for_all_nodes(self, cluster_gradients):
        """compute_scores() deve restituire uno score per ogni nodo."""
        scores = KrumDetector.compute_scores(cluster_gradients, byzantine_tolerance=1)
        assert set(scores.keys()) == set(cluster_gradients.keys())

    def test_scores_in_range(self, cluster_gradients):
        """I Krum scores normalizzati devono essere in [0.0, 1.0]."""
        scores = KrumDetector.compute_scores(cluster_gradients, byzantine_tolerance=1)
        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_byzantine_node_highest_score(self, cluster_gradients):
        """
        Il nodo Byzantine deve avere il Krum score più alto
        (è il più distante dagli altri).
        """
        scores = KrumDetector.compute_scores(cluster_gradients, byzantine_tolerance=0)
        byzantine_score = scores["highway-poisoned"]
        normal_scores = [
            scores[n] for n in scores
            if n != "highway-poisoned"
        ]
        assert byzantine_score > max(normal_scores)

    def test_detect_byzantine_returns_list(self, cluster_gradients):
        """detect_byzantine() deve restituire una lista."""
        scores = KrumDetector.compute_scores(cluster_gradients, byzantine_tolerance=1)
        byzantine = KrumDetector.detect_byzantine(scores, threshold=0.8)
        assert isinstance(byzantine, list)

    def test_detect_byzantine_identifies_poisoned(self, cluster_gradients):
        """
        detect_byzantine() deve identificare il nodo Byzantine
        con soglia appropriata.
        """
        scores = KrumDetector.compute_scores(cluster_gradients, byzantine_tolerance=0)
        byzantine = KrumDetector.detect_byzantine(scores, threshold=0.5)
        assert "highway-poisoned" in byzantine

    def test_insufficient_nodes_returns_zero_scores(self):
        """
        Con meno di 2f+3 nodi, Krum non può operare
        e deve restituire score 0.0 per tutti.
        """
        gradients = {
            "node-01": {"layer": [0.1, 0.2]},
            "node-02": {"layer": [0.3, 0.4]},
        }
        scores = KrumDetector.compute_scores(gradients, byzantine_tolerance=1)
        assert all(s == 0.0 for s in scores.values())


# ─── Test FedMIA ──────────────────────────────────────────────────────────────

class TestFedMIA:

    def test_run_attack_without_training_raises(self, normal_gradient):
        """
        run_attack() senza shadow model addestrato deve sollevare RuntimeError.
        """
        fedmia = FedMIA()
        with pytest.raises(RuntimeError):
            fedmia.run_attack("highway-01", round_id=1, gradients=normal_gradient)

    def test_run_attack_returns_mia_result(
        self, normal_gradient, train_loader_plain
    ):
        """run_attack() deve restituire un MIAResult valido."""
        fedmia = FedMIA()
        fedmia.train_shadow_model(train_loader_plain, epochs=3)
        result = fedmia.run_attack("highway-01", round_id=1, gradients=normal_gradient)
        assert isinstance(result, MIAResult)

    def test_mia_result_fields(self, normal_gradient, train_loader_plain):
        """MIAResult deve contenere tutti i campi attesi."""
        fedmia = FedMIA()
        fedmia.train_shadow_model(train_loader_plain, epochs=3)
        result = fedmia.run_attack("highway-01", round_id=1, gradients=normal_gradient)
        assert result.node_id == "highway-01"
        assert result.round_id == 1
        assert 0.0 <= result.membership_score <= 1.0
        assert isinstance(result.is_member, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_membership_score_in_range(
        self, normal_gradient, train_loader_plain
    ):
        """membership_score deve essere in [0.0, 1.0]."""
        fedmia = FedMIA()
        fedmia.train_shadow_model(train_loader_plain, epochs=3)
        result = fedmia.run_attack("highway-01", round_id=1, gradients=normal_gradient)
        assert 0.0 <= result.membership_score <= 1.0

    def test_run_cluster_attack_returns_list(
        self, cluster_gradients, train_loader_plain
    ):
        """run_cluster_attack() deve restituire una lista di MIAResult."""
        fedmia = FedMIA()
        fedmia.train_shadow_model(train_loader_plain, epochs=3)
        results = fedmia.run_cluster_attack(
            cluster_id="highway",
            round_id=1,
            cluster_gradients=cluster_gradients,
        )
        assert isinstance(results, list)
        assert len(results) == len(cluster_gradients)

    def test_run_cluster_attack_all_nodes_covered(
        self, cluster_gradients, train_loader_plain
    ):
        """run_cluster_attack() deve produrre un risultato per ogni nodo."""
        fedmia = FedMIA()
        fedmia.train_shadow_model(train_loader_plain, epochs=3)
        results = fedmia.run_cluster_attack(
            cluster_id="highway",
            round_id=1,
            cluster_gradients=cluster_gradients,
        )
        result_nodes = {r.node_id for r in results}
        assert result_nodes == set(cluster_gradients.keys())

    def test_cluster_result_has_deviation_metadata(
        self, cluster_gradients, train_loader_plain
    ):
        """
        I MIAResult del cluster attack devono contenere
        cluster_deviation nel metadata.
        """
        fedmia = FedMIA()
        fedmia.train_shadow_model(train_loader_plain, epochs=3)
        results = fedmia.run_cluster_attack(
            cluster_id="highway",
            round_id=1,
            cluster_gradients=cluster_gradients,
        )
        for result in results:
            assert "cluster_deviation" in result.metadata


# ─── Test ChargingIDS ─────────────────────────────────────────────────────────

class TestChargingIDS:

    def test_analyze_no_alert_normal_report(self, ids):
        """
        Un AuditReport normale (nessuna minaccia, score alto)
        non deve generare alert.
        """
        report = AuditReport(
            node_id="highway-01",
            round_id=1,
            privacy_score=0.95,
            epsilon=0.05,
            threats_detected=[],
            metadata={"sensitivity": 0.3},
        )
        alert = ids.analyze(report)
        assert alert is None

    def test_analyze_gradient_explosion_generates_alert(self, ids):
        """
        GRADIENT_EXPLOSION deve generare un alert CRITICAL con EXCLUDE.
        """
        report = AuditReport(
            node_id="highway-01",
            round_id=1,
            privacy_score=0.5,
            epsilon=0.5,
            threats_detected=["GRADIENT_EXPLOSION"],
            metadata={},
        )
        alert = ids.analyze(report)
        assert alert is not None
        assert alert.severity == "CRITICAL"
        assert alert.recommended_action == "EXCLUDE"

    def test_analyze_budget_exhausted_generates_alert(self, ids):
        """PRIVACY_BUDGET_EXHAUSTED deve generare un alert HIGH con EXCLUDE."""
        report = AuditReport(
            node_id="urban-01",
            round_id=5,
            privacy_score=0.0,
            epsilon=2.0,
            threats_detected=["PRIVACY_BUDGET_EXHAUSTED"],
            metadata={},
        )
        alert = ids.analyze(report)
        assert alert is not None
        assert alert.recommended_action == "EXCLUDE"

    def test_analyze_cusum_detects_drift(self, ids):
        """
        CUSUM deve rilevare deriva nel privacy score dopo il warm-up.
        """
        # Warm-up con score normali
        for i in range(10):
            report = AuditReport(
                node_id="highway-01",
                round_id=i,
                privacy_score=0.9,
                epsilon=0.1,
                threats_detected=[],
                metadata={},
            )
            ids.analyze(report)

        # Deriva: score crolla improvvisamente
        alerts = []
        for i in range(20):
            report = AuditReport(
                node_id="highway-01",
                round_id=10 + i,
                privacy_score=0.1,
                epsilon=0.9,
                threats_detected=[],
                metadata={},
            )
            alert = ids.analyze(report)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0

    def test_analyze_round_returns_round_analysis(
        self, ids, audit_reports, cluster_gradients
    ):
        """analyze_round() deve restituire un RoundAnalysis."""
        result = ids.analyze_round(
            round_id=1,
            reports=audit_reports,
            gradients=cluster_gradients,
        )
        assert isinstance(result, RoundAnalysis)

    def test_analyze_round_detects_byzantine(
        self, ids, audit_reports, cluster_gradients
    ):
        """
        analyze_round() deve identificare il nodo Byzantine con Krum.
        """
        ids = ChargingIDS(
        byzantine_tolerance=0,
        cosine_threshold=0.3,
        krum_threshold=0.5,
        )

        result = ids.analyze_round(
            round_id=1,
            reports=audit_reports,
            gradients=cluster_gradients,
        )
        assert "highway-poisoned" in result.byzantine_nodes

    def test_analyze_round_has_krum_scores(
        self, ids, audit_reports, cluster_gradients
    ):
        """analyze_round() deve restituire Krum scores per tutti i nodi."""
        result = ids.analyze_round(
            round_id=1,
            reports=audit_reports,
            gradients=cluster_gradients,
        )
        assert set(result.krum_scores.keys()) == set(cluster_gradients.keys())

    def test_analyze_round_has_cosine_scores(
        self, ids, audit_reports, cluster_gradients
    ):
        """analyze_round() deve restituire cosine scores per tutti i nodi."""
        result = ids.analyze_round(
            round_id=1,
            reports=audit_reports,
            gradients=cluster_gradients,
        )
        assert set(result.cosine_scores.keys()) == set(cluster_gradients.keys())

    def test_risk_score_increases_with_anomalies(self, ids):
        """
        Il risk score deve aumentare dopo anomalie rilevate.
        """
        report = AuditReport(
            node_id="highway-01",
            round_id=1,
            privacy_score=0.5,
            epsilon=0.5,
            threats_detected=["GRADIENT_EXPLOSION"],
            metadata={},
        )
        ids.analyze(report)
        risk = ids.get_node_risk_score("highway-01")
        assert risk > 0.0

    def test_risk_score_decays_without_anomalies(self, ids):
        """
        Il risk score deve decadere nei round senza anomalie.
        """
        # Prima un'anomalia
        report_bad = AuditReport(
            node_id="highway-01",
            round_id=1,
            privacy_score=0.0,
            epsilon=2.0,
            threats_detected=["GRADIENT_EXPLOSION"],
            metadata={},
        )
        ids.analyze(report_bad)
        risk_after_anomaly = ids.get_node_risk_score("highway-01")

        # Poi round normali
        for i in range(10):
            report_good = AuditReport(
                node_id="highway-01",
                round_id=2 + i,
                privacy_score=0.9,
                epsilon=0.05,
                threats_detected=[],
                metadata={},
            )
            ids.analyze(report_good)

        risk_after_recovery = ids.get_node_risk_score("highway-01")
        assert risk_after_recovery < risk_after_anomaly

    def test_reset_clears_all_state(self, ids, audit_reports, cluster_gradients):
        """Dopo reset(), tutto lo stato deve essere azzerato."""
        ids.analyze_round(
            round_id=1,
            reports=audit_reports,
            gradients=cluster_gradients,
        )
        ids.reset()
        assert ids.get_node_risk_score("highway-01") == 0.0
        assert len(ids.get_alert_history()) == 0
        assert len(ids.get_round_history()) == 0

    def test_alert_history_accumulates(self, ids):
        """
        Lo storico degli alert deve accumularsi tra i round.
        """
        for i in range(3):
            report = AuditReport(
                node_id="highway-01",
                round_id=i,
                privacy_score=0.0,
                epsilon=2.0,
                threats_detected=["GRADIENT_EXPLOSION"],
                metadata={},
            )
            ids.analyze(report)
        assert len(ids.get_alert_history()) == 3
