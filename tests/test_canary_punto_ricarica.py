"""Canary per punto di ricarica su Scotland (coda v3s, 2026-11-06): costruisci_pool con
chiave node_id e sito = council, come nell'adattatore di ChargePlace Scotland."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

cp = pytest.importorskip("canary_persona")


def _sessioni_scozia(n_punti=60, per_punto=40):
    ss = []
    for p in range(n_punti):
        for j in range(per_punto):
            ss.append({"user_id": None, "node_id": f"CP{p:04d}", "site_id": "Glasgow City",
                       "start_time": f"2023-09-{j % 28 + 1:02d}", "x": float(p + j)})
    for j in range(300):  # altri council
        ss.append({"user_id": None, "node_id": f"DD{j % 30:03d}", "site_id": "Dundee City",
                   "start_time": "2023-10-01", "x": float(j)})
    return ss


def test_pool_per_punto_di_ricarica():
    ss = _sessioni_scozia()
    p = cp.costruisci_pool(ss, "Glasgow City", 20, 30, 20, 10, 42, "casuale", chiave="node_id")
    assert len(p["g1"]) == len(p["g2"]) == 20 and not set(p["g1"]) & set(p["g2"])
    assert all(u.startswith("CP") for u in p["g1"] + p["g2"])
    for u in p["g1"] + p["g2"]:
        assert len(p["parti"][u]["training"]) == 20 and len(p["parti"][u]["fresche"]) == 10
    pool = set(p["g1"] + p["g2"])
    assert all(ss[i]["node_id"] not in pool for i in p["sfondo"])
    # lo sfondo tiene i punti di Glasgow non scelti e tutti gli altri council
    assert sum(1 for i in p["sfondo"] if ss[i]["site_id"] == "Dundee City") == 300
    assert sum(1 for i in p["sfondo"] if ss[i]["site_id"] == "Glasgow City") == 20 * 40


def test_senza_chiave_utente_scotland_non_ha_soggetti():
    # con la chiave di ACN (user_id) Scotland non ha soggetti: errore chiaro, non un pool vuoto
    with pytest.raises(SystemExit):
        cp.costruisci_pool(_sessioni_scozia(), "Glasgow City", 20, 30, 20, 10, 42, "casuale")


def test_default_resta_user_id():
    import inspect
    assert inspect.signature(cp.costruisci_pool).parameters["chiave"].default == "user_id"
