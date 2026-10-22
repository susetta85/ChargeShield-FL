"""Classificatore di tipo Shokri per lo shadow (regressione logistica su feature)."""
import numpy as np

from plugins.attacks.clf_logistica import punteggio_logistica, regressione_logistica


def test_separa_due_gaussiane():
    rng = np.random.default_rng(0)
    x1 = rng.normal(1.0, 1.0, size=(300, 3))
    x0 = rng.normal(-1.0, 1.0, size=(100, 3))
    m = regressione_logistica(x1, x0)
    tpr = (punteggio_logistica(m, x1) >= 0).mean()
    tnr = (punteggio_logistica(m, x0) < 0).mean()
    # distribuzioni a distanza 2 deviazioni: accuratezza bilanciata attesa circa 0.84
    assert (tpr + tnr) / 2 > 0.75


def test_deterministico():
    rng = np.random.default_rng(1)
    x1 = rng.normal(0.5, 1.0, size=(50, 2))
    x0 = rng.normal(-0.5, 1.0, size=(40, 2))
    a = punteggio_logistica(regressione_logistica(x1, x0), x1)
    b = punteggio_logistica(regressione_logistica(x1, x0), x1)
    assert np.array_equal(a, b)


def test_invariante_alla_scala():
    rng = np.random.default_rng(2)
    x1 = rng.normal(0.5, 1.0, size=(80, 3))
    x0 = rng.normal(-0.5, 1.0, size=(60, 3))
    s1 = punteggio_logistica(regressione_logistica(x1, x0), x1)
    s2 = punteggio_logistica(regressione_logistica(x1 * 1000.0, x0 * 1000.0), x1 * 1000.0)
    assert (np.sign(s1) == np.sign(s2)).mean() > 0.99


def test_insiemi_vuoti_rifiutati():
    try:
        regressione_logistica(np.zeros((0, 2)), np.ones((3, 2)))
    except ValueError:
        return
    raise AssertionError("insieme vuoto accettato")
