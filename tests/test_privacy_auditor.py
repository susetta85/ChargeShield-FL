# tests/test_privacy_auditor.py
"""
Unit Tests — PrivacyAuditor (Membership Inference Attacker)
===========================================================
Verifica il comportamento del PrivacyAuditor come attaccante MIA.

Cosa testiamo:
- Produzione corretta di AuditReport
- Calcolo sensitivity (norma L2 dei gradienti)
- Accumulo epsilon cumulativo per nodo
- Rilevazione minacce (gradient explosion, budget esaurito)
- Reset dello stato tra esperimenti
- Campi None mai sostituiti con default silenzioso

Cosa NON testiamo qui:
- FL internals (FLARE)
- Dataset o protocolli
- ChargingIDS (test separati in Sprint 4)
"""

import pytest
from auditor.privacy_auditor import (
    PrivacyAuditor,
    _compute_l2_norm,
    _flatten_model_update,
)
from core.base_auditor import AuditReport


# --- Fixture ---

@pytest.fixture
def auditor():
    """
    Fixture: istanza di PrivacyAuditor con config reale da auditor.yaml.
    Resettata prima di ogni test per evitare contaminazione tra test.
    """
    pa = PrivacyAuditor(config_path="config/auditor.yaml")
    pa.reset()
    return pa


@pytest.fixture
def normal_update():
    """
    Model update normale: gradienti piccoli, nessuna anomalia attesa.
    Simula un round FL standard di un nodo onesto.
    """
    return {
        "layer1": [0.1, -0.2, 0.3],
        "layer2": [0.05, 0.1],
        "bias": 0.01,
    }


@pytest.fixture
def exploding_update():
    """
    Model update con gradient explosion: valori molto grandi.
    Simula un possibile attacco di model poisoning.
    """
    return {
        "layer1": [999.0, -888.0, 777.0],
        "layer2": [500.0, 600.0],
        "bias": 100.0,
    }


# --- Test AuditReport ---

def test_audit_returns_report(auditor, normal_update):
    """audit() deve restituire un AuditReport valido."""
    report = auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert isinstance(report, AuditReport)


def test_audit_report_fields(auditor, normal_update):
    """AuditReport deve contenere tutti i campi attesi."""
    report = auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert report.node_id == "highway-01"
    assert report.round_id == 1
    assert isinstance(report.privacy_score, float)
    assert isinstance(report.epsilon, float)
    assert isinstance(report.threats_detected, list)
    assert isinstance(report.metadata, dict)


def test_audit_privacy_score_range(auditor, normal_update):
    """privacy_score deve essere sempre tra 0.0 e 1.0."""
    report = auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert 0.0 <= report.privacy_score <= 1.0


def test_audit_epsilon_positive(auditor, normal_update):
    """epsilon consumato deve essere positivo."""
    report = auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert report.epsilon > 0.0


def test_audit_metadata_contains_sensitivity(auditor, normal_update):
    """metadata deve contenere il campo sensitivity."""
    report = auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert "sensitivity" in report.metadata
    assert report.metadata["sensitivity"] >= 0.0


def test_audit_empty_update(auditor):
    """
    Un model update vuoto deve produrre sensitivity 0.0
    senza sollevare eccezioni.
    """
    report = auditor.audit("highway-01", round_id=1, model_update={})
    assert report.metadata["sensitivity"] == 0.0


# --- Test epsilon cumulativo ---

def test_cumulative_epsilon_increases(auditor, normal_update):
    """
    Dopo ogni round, l'epsilon cumulativo del nodo deve aumentare.
    """
    auditor.audit("highway-01", round_id=1, model_update=normal_update)
    eps1 = auditor.get_cumulative_epsilon("highway-01")
    auditor.audit("highway-01", round_id=2, model_update=normal_update)
    eps2 = auditor.get_cumulative_epsilon("highway-01")
    assert eps2 > eps1


def test_cumulative_epsilon_independent_per_node(auditor, normal_update):
    """
    L'epsilon di un nodo non deve influenzare quello di un altro nodo.
    """
    auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert auditor.get_cumulative_epsilon("urban-01") == 0.0


def test_cumulative_epsilon_zero_before_audit(auditor):
    """
    Un nodo che non ha mai partecipato deve avere epsilon 0.0.
    """
    assert auditor.get_cumulative_epsilon("residential-01") == 0.0


# --- Test rilevazione minacce ---

def test_no_threats_for_normal_update(auditor, normal_update):
    """
    Un update normale non deve generare minacce.
    """
    report = auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert "GRADIENT_EXPLOSION" not in report.threats_detected


def test_gradient_explosion_detected(auditor, exploding_update):
    """
    Un update con gradienti enormi deve generare GRADIENT_EXPLOSION.
    """
    report = auditor.audit("highway-01", round_id=1, model_update=exploding_update)
    assert "GRADIENT_EXPLOSION" in report.threats_detected


def test_budget_exhaustion_detected(auditor, normal_update):
    """
    Dopo molti round, il budget epsilon deve esaurirsi
    e il threat PRIVACY_BUDGET_EXHAUSTED deve comparire.
    """
    threats_found = False
    for i in range(1000):
        report = auditor.audit("highway-01", round_id=i, model_update=normal_update)
        if "PRIVACY_BUDGET_EXHAUSTED" in report.threats_detected:
            threats_found = True
            break
    assert threats_found


# --- Test reset ---

def test_reset_clears_epsilon(auditor, normal_update):
    """
    Dopo reset(), l'epsilon cumulativo di tutti i nodi deve essere 0.0.
    """
    auditor.audit("highway-01", round_id=1, model_update=normal_update)
    auditor.reset()
    assert auditor.get_cumulative_epsilon("highway-01") == 0.0


def test_reset_allows_fresh_experiment(auditor, normal_update):
    """
    Dopo reset(), un nuovo esperimento non deve essere contaminato
    dai dati del precedente.
    """
    for i in range(5):
        auditor.audit("highway-01", round_id=i, model_update=normal_update)
    eps_before = auditor.get_cumulative_epsilon("highway-01")
    auditor.reset()
    auditor.audit("highway-01", round_id=1, model_update=normal_update)
    eps_after = auditor.get_cumulative_epsilon("highway-01")
    assert eps_after < eps_before


# --- Test utility functions ---

def test_compute_l2_norm_correct():
    """
    Norma L2 di [3.0, 4.0] deve essere 5.0 (teorema di Pitagora).
    """
    assert _compute_l2_norm([3.0, 4.0]) == 5.0


def test_compute_l2_norm_zero_vector():
    """Norma L2 di un vettore di zeri deve essere 0.0."""
    assert _compute_l2_norm([0.0, 0.0, 0.0]) == 0.0


def test_flatten_model_update_nested():
    """
    _flatten_model_update deve gestire dizionari annidati,
    liste e scalari senza perdere valori.
    """
    update = {
        "layer1": [1.0, 2.0],
        "layer2": {"weights": [3.0, 4.0], "bias": 0.5},
        "scalar": 1.5,
    }
    flat = _flatten_model_update(update)
    assert len(flat) == 6
    assert 1.0 in flat
    assert 0.5 in flat


def test_flatten_ignores_non_numeric():
    """
    _flatten_model_update deve ignorare valori non numerici
    senza sollevare eccezioni.
    """
    update = {"layer1": [1.0, "AC", None, 2.0]}
    flat = _flatten_model_update(update)
    assert 1.0 in flat
    assert 2.0 in flat
    assert len(flat) == 2
