"""RMIA offline (log_lr_rmia_offline): Pr(x) dai soli shadow OUT. Nessun dato reale."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plugins.attacks.lira2 import log_lr_rmia, log_lr_rmia_offline  # noqa: E402


def test_offline_uguale_online_se_nessuno_shadow_vede_il_record():
    rng = np.random.default_rng(0)
    mt = rng.uniform(0.1, 1.0, size=6)
    ms = rng.uniform(0.1, 1.0, size=(4, 6))
    fuori = np.zeros((4, 6), dtype=bool)
    off = log_lr_rmia_offline(mt, ms, fuori, tau=0.5)
    on = log_lr_rmia(mt, ms, tau=0.5)
    assert np.allclose(off, on)


def test_offline_usa_solo_gli_shadow_out():
    mt = np.array([0.2, 0.7])
    ms = np.array([[0.1, 0.9],   # shadow 0: IN per il record 0
                   [0.3, 0.4],   # shadow 1: OUT per entrambi
                   [0.5, 0.2]])  # shadow 2: OUT per entrambi
    dentro = np.array([[True, False],
                       [False, False],
                       [False, False]])
    tau = 0.5
    off = log_lr_rmia_offline(mt, ms, dentro, tau)
    for j in range(2):
        out = [k for k in range(3) if not dentro[k, j]]
        ref = np.log(np.mean([np.exp(-ms[k, j] / tau) for k in out]))
        attesa = -mt[j] / tau - ref
        assert np.isclose(off[j], attesa)


def test_offline_nan_senza_shadow_out():
    mt = np.array([0.3])
    ms = np.array([[0.2], [0.4]])
    dentro = np.array([[True], [True]])
    assert np.isnan(log_lr_rmia_offline(mt, ms, dentro, tau=0.5)[0])
