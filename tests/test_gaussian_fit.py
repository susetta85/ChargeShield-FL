"""
Test puro-Python (nessuna dipendenza numpy/scipy) per le formule di
skewness/curtosi/Jarque-Bera in scripts/check_gaussian_fit.py (task #57,
Sprint 10zz+32, 2026-09-03) — il controllo empirico dell'assunzione di
normalità richiesta dal fit parametrico Gaussiano di LiRA (§3 di
docs/MetricsReference_DSN2027.md).
"""
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_gaussian_fit import (  # noqa: E402
    _JB_CHI2_2_CRITICAL_005,
    excess_kurtosis,
    jarque_bera,
    log_transform,
    skewness,
    summarize_pool,
)


def test_skewness_symmetric_distribution_near_zero():
    # Distribuzione simmetrica per costruzione (valori a coppie +-d attorno
    # alla media) -> skewness esattamente 0.
    values = [5.0 - 3, 5.0 - 1, 5.0, 5.0 + 1, 5.0 + 3]
    assert skewness(values) == pytest.approx(0.0, abs=1e-9)


def test_skewness_right_skewed_distribution_is_positive():
    # Pochi valori enormi, molti piccoli -> coda a destra -> skewness > 0
    # (esattamente il tipo di forma atteso per una MSE di ricostruzione:
    # molti campioni con errore piccolo, pochi con errore grande).
    values = [1.0, 1.1, 1.0, 0.9, 1.0, 0.95, 1.05, 50.0]
    assert skewness(values) > 0


def test_excess_kurtosis_uniform_like_is_negative():
    # Una distribuzione "piatta" (platicurtica) ha curtosi in eccesso < 0.
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert excess_kurtosis(values) < 0


def test_jarque_bera_zero_for_few_samples():
    assert jarque_bera([1.0, 2.0]) == 0.0
    assert jarque_bera([]) == 0.0


def test_jarque_bera_low_for_approximately_gaussian_sample():
    # Campione grande generato da una vera Gaussiana (seed fisso,
    # riproducibile) -> JB atteso ben sotto la soglia critica (5.99) nella
    # stragrande maggioranza dei casi (proprietà statistica, non garanzia
    # assoluta -- ma con n=2000 e seed fisso il test è deterministico e
    # stabile).
    rng = random.Random(42)
    values = [rng.gauss(0, 1) for _ in range(2000)]
    jb = jarque_bera(values)
    assert jb < _JB_CHI2_2_CRITICAL_005


def test_jarque_bera_high_for_strongly_skewed_sample():
    # Campione da una distribuzione esponenziale (fortemente asimmetrica,
    # skewness teorica = 2) -> JB atteso ben sopra la soglia critica.
    rng = random.Random(42)
    values = [rng.expovariate(1.0) for _ in range(2000)]
    jb = jarque_bera(values)
    assert jb > _JB_CHI2_2_CRITICAL_005


def test_log_transform_reduces_skew_for_lognormal_like_data():
    # Dati generati come exp(gaussiana) (log-normali per costruzione) sono
    # fortemente asimmetrici a destra; il log-transform dovrebbe riportarli
    # vicino a una Gaussiana (per costruzione, essendo l'inverso esatto).
    rng = random.Random(7)
    raw = [math.exp(rng.gauss(0, 0.5)) for _ in range(2000)]
    log_values = log_transform(raw, eps=1e-8)
    assert abs(skewness(log_values)) < abs(skewness(raw))


def test_summarize_pool_flags_log_transform_as_better_for_lognormal_data():
    rng = random.Random(7)
    raw = [math.exp(rng.gauss(0, 0.5)) for _ in range(2000)]
    result = summarize_pool(raw)
    assert result["log_transform_improves_fit"] is True
    assert result["raw"]["jarque_bera"] > result["log"]["jarque_bera"]


def test_summarize_pool_handles_tiny_samples_gracefully():
    result = summarize_pool([1.0, 2.0])
    assert result["n"] == 2
    assert "note" in result


def test_log_transform_avoids_log_of_zero():
    # MSE esattamente 0 (ricostruzione perfetta) non deve far crashare —
    # eps la sposta appena sopra 0 prima del log.
    values = log_transform([0.0, 1.0], eps=1e-8)
    assert all(math.isfinite(v) for v in values)
