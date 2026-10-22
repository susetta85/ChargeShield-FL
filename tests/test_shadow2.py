"""Attacco shadow versione 2 (src/plugins/attacks/shadow2.py, 2026-10-19).

Il test centrale e' lo "specchio" della federazione: con gli stessi dati, lo
stesso seme e la stessa inizializzazione del bersaglio, e senza estrazioni di
rumore, la federazione shadow deve riprodurre round per round il modello
globale del bersaglio. Se lo shadow non segue la procedura federata del
bersaglio (client, FedProx, meccanismo DP, aggregazione), lo specchio si rompe.
"""

from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import run_experiments as rx  # noqa: E402
from plugins.attacks import shadow2  # noqa: E402
from test_lira2 import _cfg, _sessioni  # noqa: E402


# ── funzioni pure ──────────────────────────────────────────────────────────────

def test_soglia_vantaggio():
    li = np.array([0.1, 0.2, 0.3, 0.4])
    lo = np.array([0.35, 0.5, 0.6, 0.7])
    tau, adv = shadow2.soglia_vantaggio(li, lo)
    # a 0.3: TPR 0.75, FPR 0 -> 0.75; a 0.4: TPR 1, FPR 0.25 -> 0.75; vince la piu' bassa
    assert tau == pytest.approx(0.3) and adv == pytest.approx(0.75)
    # nessun segnale utile: nessun membro previsto
    tau, adv = shadow2.soglia_vantaggio(np.array([0.9, 1.0]), np.array([0.1, 0.2]))
    assert tau == -math.inf and adv == 0.0


def test_soglia_fpr_e_tassi():
    lo = np.arange(1, 101) / 100.0              # 0.01 ... 1.00
    tau = shadow2.soglia_fpr(lo, 0.05)
    assert tau == pytest.approx(0.05)
    assert np.mean(lo <= tau) <= 0.05
    assert shadow2.soglia_fpr(lo, 0.001) == -math.inf
    # valori ripetuti: la soglia non deve superare alpha
    rip = np.array([0.1] * 10 + [0.2] * 90)
    assert shadow2.soglia_fpr(rip, 0.05) == -math.inf
    t = shadow2.tassi(np.array([0.1, 0.2, 0.9]), np.array([0.15, 0.8, 0.9, 1.0]), 0.2)
    assert t["tpr"] == pytest.approx(2 / 3) and t["fpr"] == pytest.approx(0.25)
    assert t["vantaggio"] == pytest.approx(2 / 3 - 0.25)


def test_divisioni_disgiunte_e_canary_fuori_dallo_shadow():
    sess = [{"site_id": "a" if i < 60 else "b"} for i in range(100)]
    for i in (3, 70):
        sess[i]["_canary_role"] = "member"
    pools = {"a": list(range(60)), "b": list(range(60, 100))}
    sh, val = shadow2.dividi_membri(pools, sess, 0.5, 42)
    tutti = [i for v in sh.values() for i in v]
    assert set(tutti).isdisjoint(val) and sorted(tutti + val) == list(range(100))
    assert 3 not in tutti and 70 not in tutti and {3, 70} <= set(val)
    assert len(sh["a"]) == round(0.5 * 59) and len(sh["b"]) == round(0.5 * 39)
    # riproducibile
    assert shadow2.dividi_membri(pools, sess, 0.5, 42) == (sh, val)
    hold = [{"site_id": "a"} for _ in range(30)] + [{"site_id": "b", "_canary_group": 1}]
    cal, ev = shadow2.dividi_non_membri({"a": list(range(30)), "b": [30]}, hold, 0.5, 42)
    assert set(cal).isdisjoint(ev) and len(cal) == 15 and 30 in ev


# ── federazione shadow ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dati():
    return _sessioni(160, 1), _sessioni(48, 2)


def _client(train):
    return {"caltech": train[:90], "jpl": train[90:]}


def _bersaglio(cfg, train, no_dp=True, dp_mode="dp-fedavg"):
    shadow2._semina(int(cfg["experiment"]["seed"]))
    return rx.run_fl_rounds(copy.deepcopy(cfg), train, no_dp=no_dp, dp_mode=dp_mode,
                            cluster_sessions=_client(train))


@pytest.mark.parametrize("meccanismo", ["nessuna", "record", "soggetto"])
def test_specchio_federazione(dati, meccanismo):
    train, _ = dati
    cfg = _cfg(meccanismo)
    ris_t = _bersaglio(cfg, train)
    ris_s = shadow2.federazione_shadow(cfg, _client(train), cfg["experiment"]["seed"], True,
                                       "dp-fedavg", ris_t[0]["raw_global_weights"])
    for r in range(1, cfg["experiment"]["fl_rounds"] + 1):
        for a, b in zip(ris_t[r]["global_weights"], ris_s[r]["global_weights"]):
            assert torch.equal(torch.as_tensor(a), torch.as_tensor(b)), (meccanismo, r)


def test_seme_proprio_e_generatori_ripristinati(dati):
    train, _ = dati
    cfg = _cfg("nessuna")
    ris_t = _bersaglio(cfg, train, no_dp=False, dp_mode="local")
    prima = shadow2._salva_rng()
    ris_s = shadow2.federazione_shadow(cfg, _client(train), 42 + 500_009, False, "local",
                                       ris_t[0]["raw_global_weights"])
    dopo = shadow2._salva_rng()
    assert prima[0] == dopo[0] and torch.equal(prima[2], dopo[2])
    assert all(np.array_equal(x, y) for x, y in zip(prima[1][1:3], dopo[1][1:3]))
    # stessa inizializzazione, rumore diverso: modelli diversi gia' al round 1
    assert torch.equal(torch.as_tensor(ris_s[0]["raw_global_weights"][0]),
                       torch.as_tensor(ris_t[0]["raw_global_weights"][0]))
    assert not all(torch.equal(torch.as_tensor(a), torch.as_tensor(b))
                   for a, b in zip(ris_t[1]["global_weights"], ris_s[1]["global_weights"]))


def test_init_weights_richiede_common_init(dati):
    train, _ = dati
    cfg = _cfg("nessuna")
    cfg["ml"]["common_init"] = False
    with pytest.raises(ValueError, match="common_init"):
        rx.run_fl_rounds(cfg, train, no_dp=True, cluster_sessions=_client(train),
                         init_weights=[torch.zeros(1)])


# ── attacco completo ────────────────────────────────────────────────────────

def _membership(train):
    return {"caltech": list(range(90)), "jpl": list(range(90, len(train)))}


@pytest.mark.parametrize("no_dp,dp_mode", [(True, "dp-fedavg"), (False, "local"), (False, "central")])
def test_attacco_completo_deterministico(dati, no_dp, dp_mode):
    train, hold = dati
    for s in hold:
        s["site_id"] = "caltech"
    cfg = _cfg("nessuna")
    ris_t = _bersaglio(cfg, train, no_dp=no_dp, dp_mode=dp_mode)
    stato = shadow2._salva_rng()
    a = shadow2.run_shadow_v2(cfg, train, hold, ris_t, no_dp=no_dp, dp_mode=dp_mode,
                              cluster_membership=_membership(train))
    assert torch.equal(stato[2], shadow2._salva_rng()[2])
    b = shadow2.run_shadow_v2(cfg, train, hold, ris_t, no_dp=no_dp, dp_mode=dp_mode,
                              cluster_membership=_membership(train))
    assert sorted(a) == list(range(1, cfg["experiment"]["fl_rounds"] + 1))
    for r in a:
        ra = {k: v for k, v in a[r].items() if k != "shadow_fl"}
        rb = {k: v for k, v in b[r].items() if k != "shadow_fl"}
        assert ra == rb
        assert a[r]["shadow_version"] == 2 and 0.0 <= a[r]["shadow_auc_roc"] <= 1.0
        assert a[r]["shadow_n_member"] == a[r]["shadow_n_nonmember"]
        assert a[r]["shadow_cal_fpr"] <= 0.01 + 1e-9
    fl = a[max(a)]["shadow_fl"]
    assert fl["no_dp"] == no_dp and fl["dp_mode"] == dp_mode and fl["common_init"] is True


def test_superficie_client_rifiutata(dati):
    train, hold = dati
    cfg = _cfg("nessuna")
    cfg["shadow"] = {"observation_surface": "client"}
    with pytest.raises(ValueError, match="global"):
        shadow2.run_shadow_v2(cfg, train, hold, {1: {"global_weights": [0]}})
