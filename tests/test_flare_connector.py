# tests/test_flare_connector.py
"""
Unit Tests — FLAREConnector
===========================
Verifica il comportamento del connector FL simulato.

Cosa testiamo:
- Esecuzione corretta di un round FL
- Verifica quorum (min_clients)
- Applicazione DP ai gradienti
- Integrazione con PrivacyAuditor
- Esclusione nodi con budget esaurito o gradient explosion
- Aggregazione FedAvg
- Storico AuditReport
- Reset tra esperimenti

Cosa NON testiamo qui:
- NVIDIA FLARE reale (Sprint 4)
- Autoencoder (Sprint 4)
- ChargingIDS (Sprint 4)
"""

import pytest
from src.flare.flare_connector import FLAREConnector, _fedavg, _add_gaussian_noise
from src.auditor.privacy_auditor import PrivacyAuditor
from src.nodes.charging_node import ChargingNode
from src.adapters.ocpp16_adapter import OCPP16Adapter
from src.core.base_node import NodeConfig


# --- Fixture ---

@pytest.fixture
def adapter():
    """Adapter OCPP16 condiviso tra i nodi."""
    return OCPP16Adapter()


@pytest.fixture
def nodes(adapter):
    """
    Fixture: 4 nodi online (uno per cluster).
    Sufficienti a soddisfare il quorum minimo (min_clients=3).
    """
    configs = [
        NodeConfig(node_id="highway-01", cluster_id="highway", location="A1"),
        NodeConfig(node_id="urban-01", cluster_id="urban", location="B1"),
        NodeConfig(node_id="residential-01", cluster_id="residential", location="C1"),
        NodeConfig(node_id="corporate-01", cluster_id="corporate", location="D1"),
    ]
    return [ChargingNode(config=c, adapter=adapter) for c in configs]


@pytest.fixture
def auditor():
    """PrivacyAuditor resettato prima di ogni test."""
    pa = PrivacyAuditor(config_path="config/auditor.yaml")
    pa.reset()
    return pa


@pytest.fixture
def connector(nodes, auditor):
    """FLAREConnector con 4 nodi e auditor."""
    return FLAREConnector(nodes=nodes, auditor=auditor)


@pytest.fixture
def global_model():
    """
    Modello globale simulato — placeholder per Sprint 4.
    In Sprint 4 sarà sostituito dai pesi reali dell'autoencoder.
    """
    return {
        "layer1": [0.1, -0.2, 0.3, 0.4],
        "layer2": [0.05, 0.1, -0.05],
        "bias": 0.01,
    }


# --- Test run_round ---

def test_run_round_returns_dict(connector, global_model):
    """run_round() deve restituire un dizionario con le chiavi attese."""
    result = connector.run_round(round_id=1, global_model=global_model)
    assert "aggregated_update" in result
    assert "audit_reports" in result
    assert "excluded_nodes" in result
    assert "participating" in result


def test_run_round_participating_nodes(connector, global_model, nodes):
    """
    Tutti i nodi online devono partecipare al round.
    """
    result = connector.run_round(round_id=1, global_model=global_model)
    assert len(result["participating"]) == len(nodes)


def test_run_round_audit_reports_count(connector, global_model, nodes):
    """
    Deve essere prodotto un AuditReport per ogni nodo partecipante.
    """
    result = connector.run_round(round_id=1, global_model=global_model)
    assert len(result["audit_reports"]) == len(nodes)


def test_run_round_aggregated_update_keys(connector, global_model):
    """
    Il modello aggregato deve avere le stesse chiavi del modello globale.
    """
    result = connector.run_round(round_id=1, global_model=global_model)
    assert set(result["aggregated_update"].keys()) == set(global_model.keys())


# --- Test quorum ---

def test_quorum_not_met_skips_round(auditor, adapter, global_model):
    """
    Se i nodi online sono meno di min_clients, il round deve essere skippato.
    """
    # Solo 1 nodo — sotto il quorum minimo (min_clients=3)
    single_node = ChargingNode(
        config=NodeConfig(node_id="highway-01", cluster_id="highway", location="A1"),
        adapter=adapter,
    )
    connector = FLAREConnector(nodes=[single_node], auditor=auditor)
    result = connector.run_round(round_id=1, global_model=global_model)
    assert result.get("skipped") is True


def test_quorum_not_met_returns_global_model(auditor, adapter, global_model):
    """
    Se il round è skippato, il modello globale deve rimanere invariato.
    """
    single_node = ChargingNode(
        config=NodeConfig(node_id="highway-01", cluster_id="highway", location="A1"),
        adapter=adapter,
    )
    connector = FLAREConnector(nodes=[single_node], auditor=auditor)
    result = connector.run_round(round_id=1, global_model=global_model)
    assert result["aggregated_update"] == global_model


def test_offline_node_excluded_from_round(auditor, adapter, global_model):
    """
    Un nodo offline non deve partecipare al round FL.
    """
    online_nodes = [
        ChargingNode(
            config=NodeConfig(node_id=f"highway-0{i}", cluster_id="highway", location=f"A{i}"),
            adapter=adapter,
        )
        for i in range(1, 5)
    ]
    # Metti il primo nodo offline
    online_nodes[0].set_status("offline")
    connector = FLAREConnector(nodes=online_nodes, auditor=auditor)
    result = connector.run_round(round_id=1, global_model=global_model)
    assert "highway-01" not in result["participating"]


# --- Test FedAvg ---

def test_fedavg_average_correct():
    """
    FedAvg di due update identici deve restituire lo stesso update.
    """
    update = {"layer1": [1.0, 2.0], "bias": 0.5}
    result = _fedavg([update, update])
    assert result["layer1"] == [1.0, 2.0]
    assert result["bias"] == 0.5


def test_fedavg_average_values():
    """
    FedAvg di [0.0, 2.0] deve restituire 1.0.
    """
    u1 = {"bias": 0.0}
    u2 = {"bias": 2.0}
    result = _fedavg([u1, u2])
    assert result["bias"] == pytest.approx(1.0)


def test_fedavg_empty_raises():
    """
    FedAvg su lista vuota deve sollevare ValueError.
    """
    with pytest.raises(ValueError):
        _fedavg([])


# --- Test Differential Privacy ---

def test_gaussian_noise_changes_values():
    """
    _add_gaussian_noise deve modificare i valori numerici.
    La probabilità che il rumore sia esattamente 0 è trascurabile.
    """
    update = {"layer1": [1.0, 2.0, 3.0], "bias": 0.5}
    noisy = _add_gaussian_noise(update, noise_scale=0.1)
    assert noisy["layer1"] != update["layer1"]


def test_gaussian_noise_preserves_keys():
    """
    _add_gaussian_noise deve preservare tutte le chiavi del model update.
    """
    update = {"layer1": [1.0, 2.0], "bias": 0.5}
    noisy = _add_gaussian_noise(update, noise_scale=0.1)
    assert set(noisy.keys()) == set(update.keys())


def test_gaussian_noise_does_not_modify_original():
    """
    _add_gaussian_noise non deve modificare il dizionario originale.
    """
    update = {"layer1": [1.0, 2.0], "bias": 0.5}
    original_values = update["layer1"].copy()
    _add_gaussian_noise(update, noise_scale=0.1)
    assert update["layer1"] == original_values


# --- Test storico AuditReport ---

def test_audit_history_grows_with_rounds(connector, global_model, nodes):
    """
    Lo storico deve accumulare un AuditReport per nodo per round.
    """
    connector.run_round(round_id=1, global_model=global_model)
    connector.run_round(round_id=2, global_model=global_model)
    history = connector.get_audit_history()
    assert len(history) == len(nodes) * 2


def test_audit_history_reset(connector, global_model):
    """
    Dopo reset(), lo storico deve essere vuoto.
    """
    connector.run_round(round_id=1, global_model=global_model)
    connector.reset()
    assert len(connector.get_audit_history()) == 0
