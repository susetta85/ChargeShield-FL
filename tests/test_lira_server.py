"""LiRA e RMIA con ombre solo su dati del server (lira.fonte_ombre: server, 2026-10-20).

Regola della dottoranda (20 ottobre): nessun attacco usa dati che il server non
possiede. Il server ha i propri dati ausiliari (meta' dell'holdout per gli attacchi
di ogni sito, la stessa riserva dello shadow realistico) e le sessioni che interroga.
Questi test controllano che le ombre online si allenino solo su dati ausiliari piu'
candidati, che le ombre offline si allenino solo su dati ausiliari, che la
popolazione z di RMIA non entri mai in un training, e che l'attacco completo produca
le quattro metriche (LiRA e RMIA, online e offline) in modo deterministico.
"""

from __future__ import annotations

import copy
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
from plugins.attacks import lira2  # noqa: E402

ROUNDS = 2
SITI = ("0002", "0001")          # caltech, jpl


# ── funzioni pure ──────────────────────────────────────────────────────────────

def _campiona(rng, pool, n):
    return rng.sample(pool, n)


def _finte(n, prefisso):
    return [{"session_id": f"{prefisso}{i}"} for i in range(n)]


def test_insiemi_server_solo_dati_del_server():
    mb, nm, aux = _finte(100, "m"), _finte(20, "n"), _finte(30, "a")
    ins = lira2.insiemi_server(mb, nm, aux, 0.2, n_shadow=16, n_offline=8, seed=42, ci=0,
                               campiona=_campiona)
    valutati = {id(s) for s in ins["valutati"]}
    aux_tr = {id(s) for s in ins["aux_tr"]}
    z = {id(s) for s in ins["z"]}
    assert len(ins["z"]) == 6 and len(ins["aux_tr"]) == 24
    assert z.isdisjoint(aux_tr) and z <= {id(s) for s in aux}
    # membri valutati quanti i non membri; gli altri membri non compaiono mai
    assert int(ins["membro"].sum()) == 20 and len(ins["valutati"]) == 40
    for ss in ins["ins_on"]:
        ids = {id(s) for s in ss}
        assert ids <= (aux_tr | z | valutati)       # z nell'universo online, mai valutata
    for ss in ins["ins_off"]:
        ids = {id(s) for s in ss}
        assert ids <= aux_tr and ids.isdisjoint(z)
    assert z.isdisjoint(valutati)
    usati = {id(s) for ss in ins["ins_on"] + ins["ins_off"] for s in ss}
    membri_non_valutati = {id(s) for s in mb} - valutati
    assert usati.isdisjoint(membri_non_valutati)
    # ogni candidato e' IN in una parte delle ombre online, mai in tutte o in nessuna di solito
    quote = [sum(id(s) in {id(x) for x in ss} for ss in ins["ins_on"]) for s in ins["valutati"]]
    assert 0.3 < np.mean(quote) / 16 < 0.7


def test_insiemi_server_deterministico_e_canary_ai_candidati():
    mb = _finte(50, "m") + [{"session_id": "c1", "_canary_group": 0}] * 3
    nm = _finte(10, "n") + [{"session_id": "c2", "_canary_group": 1}]
    aux = _finte(12, "a")
    a = lira2.insiemi_server(mb, nm, aux, 0.2, 4, 4, 7, 1, _campiona)
    b = lira2.insiemi_server(mb, nm, aux, 0.2, 4, 4, 7, 1, _campiona)
    assert [s["session_id"] for s in a["valutati"]] == [s["session_id"] for s in b["valutati"]]
    assert all(s.get("_canary_group") is None for s in a["aux_tr"] + a["z"])
    assert sum(s.get("_canary_group") is not None for s in a["valutati"]) == 4


def test_punteggio_offline_segno_e_casi_limite():
    rng = np.random.default_rng(0)
    n, k = 400, 16
    base = rng.normal(0, 1, n)
    ref = base + rng.normal(0, 0.2, (k, n))           # ombre che non hanno visto nessuno
    membro = rng.random(n) < 0.5
    t = base + np.where(membro, -0.8, 0.0) + rng.normal(0, 0.2, n)
    score, diag = lira2.punteggio_lira_offline(t, ref)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(membro, score) > 0.95 and diag["n_valutati"] == n
    # una sola ombra: nessuna varianza, nessun punteggio
    score, diag = lira2.punteggio_lira_offline(t, ref[:1])
    assert np.isnan(score).all() and diag["n_valutati"] == 0


def test_punteggio_out_usa_solo_ombre_out():
    rng = np.random.default_rng(5)
    n, k = 400, 32
    in_mask = rng.random((k, n)) < 0.5
    base = rng.normal(0, 1, n)
    stat = base + np.where(in_mask, -0.8, 0.0) + rng.normal(0, 0.2, (k, n))
    membro = rng.random(n) < 0.5
    t = base + np.where(membro, -0.8, 0.0) + rng.normal(0, 0.2, n)
    score, diag = lira2.punteggio_lira_out(t, stat, in_mask)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(membro, score) > 0.95
    # se le ombre IN cambiano, il punteggio out non cambia
    stat2 = np.where(in_mask, stat + 5.0, stat)
    score2, _ = lira2.punteggio_lira_out(t, stat2, in_mask)
    np.testing.assert_allclose(score, score2)
    # senza ombre OUT il campione non si valuta
    in_mask[:, 0] = True
    score3, _ = lira2.punteggio_lira_out(t, stat, in_mask)
    assert np.isnan(score3[0])


def test_punteggio_offline_robusto_a_un_nan():
    rng = np.random.default_rng(6)
    ref = rng.normal(0, 1, (8, 50))
    ref[3, 7] = np.nan
    score, diag = lira2.punteggio_lira_offline(rng.normal(0, 1, 50), ref)
    assert np.isfinite(diag["sigma_out"]) and np.isnan(score[7]) and np.isfinite(score[:7]).all()


def test_punteggio_offline_nullo_se_target_indipendente():
    rng = np.random.default_rng(3)
    n, k = 3000, 16
    base = rng.normal(0, 1, n)
    ref = base + rng.normal(0, 0.3, (k, n))
    membro = rng.random(n) < 0.5
    t = base - 0.2 + rng.normal(0, 0.3, n)            # non dipende da membro
    score, _ = lira2.punteggio_lira_offline(t, ref)
    from sklearn.metrics import roc_auc_score
    assert abs(roc_auc_score(membro, score) - 0.5) < 0.04


# ── attacco completo su due siti ───────────────────────────────────────────────

def _sessioni(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = datetime(2020, 1, 1)
    out = []
    for i in range(n):
        start = base + timedelta(hours=rng.randint(0, 24 * 60), minutes=rng.randint(0, 59))
        end = start + timedelta(hours=rng.uniform(0.5, 8.0))
        out.append({
            "session_id": f"s-{seed}-{i}", "site_id": SITI[i % 2],
            "user_id": (f"u{rng.randrange(25)}" if i % 5 else None),
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "total_energy_kwh": rng.uniform(0.0, 1.0), "max_power_kw": rng.uniform(0.0, 1.0),
            "kwh_requested": rng.uniform(0.0, 1.0), "minutes_available": rng.uniform(0.0, 1.0),
        })
    sess = rx.enrich_sessions(out)
    for s in sess:
        s["hour_of_day"] = s["hour_of_day"] / 24.0
        s["duration_hours"] = s["duration_hours"] / 8.0
    return sess


def _cfg(meccanismo: str = "nessuna") -> dict:
    cfg = {
        "experiment": {"name": "t", "fl_rounds": ROUNDS, "seed": 42, "epsilon": 64.0,
                       "delta": 1e-5, "max_grad_norm": 1.0},
        "ml": {"input_dim": 6, "lr": 0.01, "epochs": 2, "batch_size": 16,
               "proximal_mu": 0.01, "common_init": True, "norm": "group"},
        "lira": {"n_shadow": 4, "n_shadow_offline": 4, "fonte_ombre": "server"},
        "byzantine_attack": {"enabled": False},
    }
    if meccanismo == "soggetto":
        cfg["ml"]["subject_dp"] = {"enabled": True, "subject_key": "user_id",
                                   "max_grad_norm": 0.5, "noise_multiplier": 1.0,
                                   "expected_subjects_per_step": 8,
                                   "max_records_per_subject": 16}
    return cfg


@pytest.fixture(scope="module")
def dati():
    return _sessioni(200, 1), _sessioni(480, 2)


def _fl(cfg, train, no_dp=True, dp_mode="dp-fedavg"):
    cm = rx.group_indices_by_site(train)
    cs = {c: [train[i] for i in idx] for c, idx in cm.items()}
    torch.manual_seed(42)
    return rx.run_fl_rounds(copy.deepcopy(cfg), train, no_dp=no_dp, dp_mode=dp_mode,
                            cluster_sessions=cs), cm


@pytest.mark.parametrize("meccanismo,no_dp,dp_mode", [
    ("nessuna", True, "dp-fedavg"), ("nessuna", False, "local"), ("soggetto", True, "dp-fedavg")])
def test_attacco_server_completo(dati, monkeypatch, meccanismo, no_dp, dp_mode):
    cfg = _cfg(meccanismo)
    train, hold = dati
    fl, cm = _fl(cfg, train, no_dp, dp_mode)
    visti: list[Any] = []
    originale = lira2.esegui_traiettorie

    def _registra(specs, workers=1):
        visti.extend(specs)
        return originale(specs, workers)

    monkeypatch.setattr(lira2, "esegui_traiettorie", _registra)
    comp: dict[str, Any] = {}
    res = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=no_dp, dp_mode=dp_mode,
                            cluster_membership=cm, composed_output=comp)
    for r in range(1, ROUNDS + 1):
        assert res[r]["lira_fonte_ombre"] == "server"
        for k in ("lira_auc_roc", "lira_out_auc_roc", "lira_off_auc_roc",
                  "rmia_auc_roc", "rmia_out_auc_roc", "rmia_off_auc_roc"):
            assert res[r][k] is not None and 0.0 <= res[r][k] <= 1.0, (r, k)
    for k in ("composed_lira_auc_roc", "composed_lira_out_auc_roc", "composed_lira_off_auc_roc",
              "composed_rmia_loglr_auc_roc", "composed_rmia_out_loglr_auc_roc",
              "composed_rmia_off_loglr_auc_roc"):
        assert k in comp, k
    # le ombre usano solo dati del server: dei membri del client, solo quelli valutati
    train_ids = {id(s) for s in train}
    online = visti[:len(visti) // 2]
    offline = visti[len(visti) // 2:]
    assert len(online) == len(offline) == 4 * len(cm)
    for sp in offline:
        assert not ({id(s) for s in sp.in_sessions} & train_ids)
    for c in cm:
        membri_usati = set().union(*[{id(s) for s in sp.in_sessions} & train_ids
                                     for sp in online if sp.cid == c])
        nm_val = res[1]["lira_per_client"][c]["n_valutati"]
        assert len(membri_usati) <= nm_val // 2 + 1


def test_attacco_server_deterministico_con_processi(dati):
    cfg = _cfg()
    train, hold = dati
    fl, cm = _fl(cfg, train)
    a = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm)
    b = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm, workers=2)
    for r in range(1, ROUNDS + 1):
        for k in ("lira_auc_roc", "lira_out_auc_roc", "lira_off_auc_roc",
                  "rmia_auc_roc", "rmia_out_auc_roc", "rmia_off_auc_roc"):
            assert a[r][k] == b[r][k], (r, k)


def test_server_rifiuta_partizione_iid(dati):
    cfg = _cfg()
    cfg["partition"] = {"strategy": "iid"}
    train, hold = dati
    fl, cm = _fl(_cfg(), train)
    with pytest.raises(ValueError, match="per_site"):
        lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm)


def test_senza_ombre_offline_restano_online_e_out(dati):
    cfg = _cfg()
    cfg["lira"]["n_shadow_offline"] = 0
    train, hold = dati
    fl, cm = _fl(cfg, train)
    res = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm)
    assert res[1]["lira_auc_roc"] is not None and res[1]["lira_out_auc_roc"] is not None
    assert res[1]["lira_off_auc_roc"] is None


def test_universo_resta_il_default(dati):
    """Senza fonte_ombre il comportamento e' quello della v2 (analisi di sensibilita')."""
    cfg = _cfg()
    del cfg["lira"]["fonte_ombre"]
    train, hold = dati
    fl, cm = _fl(cfg, train)
    res = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm)
    assert "lira_fonte_ombre" not in res[1] and "lira_off_auc_roc" not in res[1]


# ── avversario partecipante: solo modelli globali ──────────────────────────────

@pytest.mark.parametrize("no_dp,dp_mode", [(True, "dp-fedavg"), (False, "local"), (False, "central")])
def test_partecipante_usa_solo_globali_e_dati_del_server(dati, monkeypatch, no_dp, dp_mode):
    cfg = _cfg()
    cfg["lira"]["observation_surface"] = "partecipante"
    train, hold = dati
    fl, cm = _fl(cfg, train, no_dp, dp_mode)
    reali = {id(u) for r in fl if isinstance(r, int) and r >= 1
             for u in (fl[r].get("updates") or []) + (fl[r].get("raw_updates") or [])}
    visti, emul = [], []
    t_orig, e_orig = lira2.esegui_traiettorie, lira2.esegui_emulazioni

    def _t(specs, workers=1):
        visti.extend(specs)
        return t_orig(specs, workers)

    def _e(specs, workers=1):
        emul.extend(specs)
        return e_orig(specs, workers)

    monkeypatch.setattr(lira2, "esegui_traiettorie", _t)
    monkeypatch.setattr(lira2, "esegui_emulazioni", _e)
    res = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=no_dp, dp_mode=dp_mode,
                            cluster_membership=cm)
    train_ids = {id(s) for s in train}
    assert len(emul) == len(cm)
    for sp in emul:                                  # emulazioni: solo dati del server
        assert not ({id(s) for s in sp.in_sessions} & train_ids)
    for sp in visti:                                 # nessun aggiornamento reale nell'aggregazione
        for r, lista in sp.altri_update.items():
            assert all(u is None or id(u) not in reali for u in lista)
            assert sum(u is None for u in lista) == 1
    for r in range(1, ROUNDS + 1):
        assert res[r]["lira_observation_surface"] == "partecipante"
        for k in ("lira_auc_roc", "lira_out_auc_roc", "lira_off_auc_roc",
                  "rmia_auc_roc", "rmia_out_auc_roc", "rmia_off_auc_roc"):
            assert res[r][k] is not None and 0.0 <= res[r][k] <= 1.0, (r, k)


def test_partecipante_deterministico_e_solo_con_server(dati):
    cfg = _cfg()
    cfg["lira"]["observation_surface"] = "partecipante"
    train, hold = dati
    fl, cm = _fl(cfg, train)
    a = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm)
    b = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm, workers=2)
    for r in range(1, ROUNDS + 1):
        for k in ("lira_auc_roc", "lira_out_auc_roc", "lira_off_auc_roc", "rmia_auc_roc"):
            assert a[r][k] == b[r][k], (r, k)
    cfg2 = copy.deepcopy(cfg)
    del cfg2["lira"]["fonte_ombre"]
    with pytest.raises(ValueError, match="partecipante"):
        lira2.run_lira_v2(cfg2, train, hold, fl, n_shadow=4, no_dp=True, cluster_membership=cm)
