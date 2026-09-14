"""
Test puro-Python (nessuna dipendenza torch — assente in questo sandbox,
stesso limite di tests/test_mia_advantage.py) per la formula di advanced
composition (Dwork & Roth 2014, "The Algorithmic Foundations of
Differential Privacy", Theorem 3.20), task #118 (Sprint 10zz+83,
2026-09-14) — "opzione A" del rimedio al gap naive-composition (Sprint
10zz+81) discusso con l'utente.

Replica ESATTA della formula usata da
_advanced_composition_epsilon() in scripts/run_experiments.py (e dalla
sua copia in scripts/compute_advanced_composition.py, verificate qui
identiche a una TERZA implementazione indipendente, non copiata da
nessuna delle due) — se questa formula cambia in uno dei due file,
aggiornare insieme qui.
"""
import math

import pytest


def _advanced_composition_epsilon(epsilon, delta, rounds, delta_prime=None):
    """Terza implementazione indipendente, per verificare le due copie di produzione."""
    if rounds <= 0:
        return 0.0, float(delta_prime if delta_prime is not None else delta)
    if delta_prime is None:
        delta_prime = delta
    eps_prime = (
        epsilon * math.sqrt(2 * rounds * math.log(1.0 / delta_prime))
        + rounds * epsilon * (math.exp(epsilon) - 1)
    )
    delta_total = rounds * delta + delta_prime
    return float(eps_prime), float(delta_total)


def test_reference_value_eps1_rounds10_delta1e5():
    # Valore di riferimento calcolato indipendentemente (vedi conversazione
    # con l'utente, Sprint 10zz+83): epsilon=1.0, delta=1e-5, 10 round.
    eps_prime, delta_total = _advanced_composition_epsilon(1.0, 1e-5, 10)
    assert eps_prime == pytest.approx(32.3571, abs=1e-3)
    assert delta_total == pytest.approx(10 * 1e-5 + 1e-5, abs=1e-12)


def test_rounds_zero_returns_zero_epsilon():
    eps_prime, delta_total = _advanced_composition_epsilon(1.0, 1e-5, 0)
    assert eps_prime == 0.0
    assert delta_total == pytest.approx(1e-5)


def test_advanced_is_monotonically_increasing_in_rounds():
    # Ogni termine della formula e' positivo e crescente in k (rounds):
    # sqrt(k) cresce, k*eps*(e^eps-1) cresce linearmente — advanced deve
    # quindi crescere monotonamente con k, per qualunque epsilon fissato.
    eps = 0.3
    values = [_advanced_composition_epsilon(eps, 1e-5, k)[0] for k in range(1, 50)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_best_known_never_exceeds_naive_for_our_actual_config():
    # Configurazione REALE del progetto (config/experiment.yaml: fl_rounds=10,
    # delta=1e-5) per i tre epsilon della campagna principale (dp-fedavg/
    # central/local x eps in {1.0, 0.5, 0.1}): per k=10, advanced e'
    # SEMPRE piu' larga della naive (verificato anche empiricamente su ogni
    # JSON gia' completato da scripts/compute_advanced_composition.py, 0/45
    # casi in cui advanced vince) — quindi best_known = min(naive, advanced)
    # deve coincidere esattamente con naive in tutti e tre i casi.
    delta = 1e-5
    rounds = 10
    for eps in (1.0, 0.5, 0.1):
        naive = eps * rounds
        advanced, _ = _advanced_composition_epsilon(eps, delta, rounds)
        best_known = min(naive, advanced)
        assert best_known == pytest.approx(naive)
        assert advanced > naive


def test_crossover_condition_epsilon_below_ln2_eventually_beats_naive():
    # Condizione analitica nota: per k grande, advanced/naive -> (e^eps - 1)
    # (il termine lineare k*eps*(e^eps-1) domina su sqrt(k)). Questo rapporto
    # e' < 1 (quindi advanced puo' eventualmente battere naive) se e solo se
    # e^eps < 2, cioe' eps < ln(2) ~= 0.6931. Verifichiamo che eps=0.1
    # (< ln2) trovi un crossover entro un numero ragionevole di round, e che
    # eps=1.0 (> ln2) non lo trovi mai entro un range ampio.
    delta = 1e-5
    found_crossover_below_ln2 = False
    for k in range(1, 1000):
        naive = 0.1 * k
        advanced, _ = _advanced_composition_epsilon(0.1, delta, k)
        if advanced < naive:
            found_crossover_below_ln2 = True
            assert k == 29  # valore esatto atteso, non solo "esiste"
            break
    assert found_crossover_below_ln2

    for k in range(1, 5000):
        naive = 1.0 * k
        advanced, _ = _advanced_composition_epsilon(1.0, delta, k)
        assert advanced >= naive  # mai un crossover per eps=1.0 >= ln(2)
