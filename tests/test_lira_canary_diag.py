"""Diagnostica dei canary nella LiRA lato server (2026-10-26)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from plugins.attacks.lira2 import diagnostica_canary  # noqa: E402


def test_senza_canary_none():
    v = [{"session_id": "a"}, {"session_id": "b"}]
    assert diagnostica_canary(np.zeros(2), np.zeros((4, 2)), np.zeros((4, 2), bool),
                              np.array([True, False]), v) is None


def test_medie_in_out_per_gruppo():
    # 2 candidati canary (membro e non membro) e uno normale; 4 ombre, IN nelle prime 2
    v = [{"_canary_group": 0}, {"_canary_group": 1}, {"session_id": "x"}]
    stat = np.array([[-5.0, -1.0, 0.0], [-5.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]])
    msk = np.array([[True, True, True], [True, True, True], [False, False, False], [False, False, False]])
    d = diagnostica_canary(np.array([-2.0, -1.0, 0.0]), stat, msk, np.array([True, False, True]), v)
    assert d["membri"] == {"n": 1, "t_mediana": -2.0, "mu_in_mediana": -5.0, "mu_out_mediana": -1.0}
    assert d["non_membri"]["mu_in_mediana"] == -1.0 and d["non_membri"]["n"] == 1
