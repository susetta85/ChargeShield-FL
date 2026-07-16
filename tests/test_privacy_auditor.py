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

    Soglia adattiva (Fix 3, Sprint 8):
        threshold = max_grad_norm + 3×sigma  (Gaussian mechanism, ε=1.0)
        sigma ≈ 4.84 → threshold ≈ 15.5

    exploding_update L2 ≈ 1735 >> 15.5 → GRADIENT_EXPLOSION atteso.
    La soglia adattiva sostituisce max_grad_norm×10 (=10.0) che era calibrata
    su pesi post-DP e causava falsi positivi sistematici con ε piccoli.
    """
    report = auditor.audit("highway-01", round_id=1, model_update=exploding_update)
    assert "GRADIENT_EXPLOSION" in report.threats_detected
    # Verifica che la soglia adattiva sia esposta nel metadata
    assert "explosion_threshold" in report.metadata
    assert report.metadata["explosion_threshold"] > 0.0


def test_gradient_explosion_threshold_adapts_to_epsilon():
    """
    La soglia GRADIENT_EXPLOSION deve scalare con epsilon (inversamente):
    epsilon più grande → sigma più piccola → soglia più stringente.
    epsilon più piccolo → sigma più grande → soglia più permissiva.
    """
    import math
    auditor_low_eps = PrivacyAuditor(config_path="config/auditor.yaml", epsilon=0.1)
    auditor_high_eps = PrivacyAuditor(config_path="config/auditor.yaml", epsilon=5.0)
    # epsilon=0.1: sigma≈48.4, threshold≈146
    # epsilon=5.0: sigma≈0.97, threshold≈3.9
    assert auditor_low_eps._explosion_threshold > auditor_high_eps._explosion_threshold, (
        "threshold deve essere maggiore per epsilon più piccolo "
        f"(eps=0.1: {auditor_low_eps._explosion_threshold:.2f}, "
        f"eps=5.0: {auditor_high_eps._explosion_threshold:.2f})"
    )


def test_metadata_contains_explosion_threshold(auditor, normal_update):
    """
    L'AuditReport deve esporre explosion_threshold nel metadata
    per permettere debug e confronto con la sensitivity nel JSON di output.
    """
    report = auditor.audit("highway-01", round_id=1, model_update=normal_update)
    assert "explosion_threshold" in report.metadata
    assert report.metadata["explosion_threshold"] > report.metadata["sensitivity"], (
        "Per un update normale, sensitivity deve essere < explosion_threshold "
        "(altrimenti GRADIENT_EXPLOSION sarebbe un falso positivo)"
    )


def test_budget_exhaustion_detected(auditor):
    """
    Dopo abbastanza round con un update ad alta sensitivity, PRIVACY_BUDGET_EXHAUSTED compare.

    Formula (post Fix H-03):
        budget_ratio = cumulative_epsilon / (epsilon_per_round × total_rounds_budget)
                     = n × (sensitivity / max_grad_norm) / total_rounds_budget
    Con max_grad_norm=1.0, total_rounds_budget=1000, sensitivity=1.0:
        budget_ratio = n / 1000 → esaurimento al round 1001.

    Nota: budget_ratio è epsilon-indipendente (epsilon si cancella tra
    round_epsilon e total_budget) — esaurimento dipende solo da
    sensitivity / max_grad_norm relativa.
    """
    # Update con sensitivity ≈ max_grad_norm=1.0 → budget_ratio = n/1000
    high_sensitivity_update = {"layer1": [1.0]}  # L2 = 1.0 = max_grad_norm
    threats_found = False
    for i in range(1, 1100):  # margine su 1000 round (= total_rounds_budget)
        report = auditor.audit("highway-01", round_id=i, model_update=high_sensitivity_update)
        if "PRIVACY_BUDGET_EXHAUSTED" in report.threats_detected:
            threats_found = True
            break
    assert threats_found, f"PRIVACY_BUDGET_EXHAUSTED non rilevato dopo 1100 round"


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


# --- Test epsilon override CLI ---

def test_epsilon_override_takes_priority():
    """
    PrivacyAuditor(epsilon=0.01) deve usare 0.01 come budget,
    non il valore da auditor.yaml (tipicamente > 0.01).
    Se la logica fosse 'epsilon or config_value', epsilon=0 verrebbe ignorato.
    """
    auditor_override = PrivacyAuditor(config_path="config/auditor.yaml", epsilon=0.01)
    assert auditor_override._epsilon_budget == 0.01, (
        f"_epsilon_budget deve essere 0.01, trovato {auditor_override._epsilon_budget}"
    )


def test_epsilon_override_exhausts_budget(normal_update):
    """
    Il budget DP deve esaurirsi dopo abbastanza round, indipendentemente da epsilon.

    Nota (post Fix H-03): budget_ratio = n × sensitivity / (max_grad_norm × total_rounds_budget),
    che è epsilon-indipendente (epsilon si cancella in round_epsilon / total_budget).
    L'esaurimento avviene quando n ≥ total_rounds_budget × max_grad_norm / sensitivity.
    Con normal_update (sensitivity≈0.39), max_grad_norm=1.0, total_rounds=1000:
        n_exhaust ≈ 1000 / 0.39 ≈ 2564 round.

    Questo test verifica che l'esaurimento avvenga entro un numero di round ragionevole,
    usando un update ad alta sensitivity per contenere il loop.
    """
    auditor_tight = PrivacyAuditor(config_path="config/auditor.yaml", epsilon=0.001)
    high_sensitivity_update = {"layer1": [1.0]}  # sensitivity = max_grad_norm → esaurimento in 1000 round
    exhausted_round = None
    for i in range(1, 1100):
        report = auditor_tight.audit("test-node", round_id=i, model_update=high_sensitivity_update)
        if "PRIVACY_BUDGET_EXHAUSTED" in report.threats_detected:
            exhausted_round = i
            break
    assert exhausted_round is not None, (
        "Budget non esaurito dopo 1100 round con sensitivity=max_grad_norm"
    )


def test_epsilon_none_uses_yaml_default():
    """
    Senza epsilon override, il budget deve corrispondere a config/auditor.yaml.
    """
    import yaml
    from pathlib import Path
    with open("config/auditor.yaml") as f:
        cfg = yaml.safe_load(f)
    expected_budget = cfg["auditor"]["dp"]["epsilon"]
    auditor_default = PrivacyAuditor(config_path="config/auditor.yaml")
    assert auditor_default._epsilon_budget == expected_budget
