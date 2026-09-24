"""
Test delle correzioni del 2026-09-24 agli script di analisi (nessuna dipendenza
torch).

Segnalazione 46: il costo del foglio Utility_privacy_limite deve essere la loss
sull'holdout del modello globale RILASCIATO, non la loss di addestramento locale.

Segnalazione 47: il conteggio per record sui soli membri misura la stabilita' del
ranking; il test appaiato deve dare zero quando l'appartenenza non conta, anche se
i record hanno una difficolta' intrinseca stabile, e deve accorgersi di un effetto
di appartenenza vero.
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import genera_matrici_faseA as gm  # noqa: E402
import worst_case_livello_di_caso as wc  # noqa: E402


def _json(nm=-0.0945, fl=0.006, surface=None):
    cfg = {"seed": 42}
    if surface:
        cfg["yeom"] = {"observation_surface": surface}
    return {"config": cfg, "per_round": {
        "1": {"fl": {"mean_loss": 0.001}, "mia": {"non_member_score_mean": -0.06}},
        "10": {"fl": {"mean_loss": fl}, "mia": {"non_member_score_mean": nm}},
    }}


def test_loss_holdout_e_quella_del_modello_rilasciato_all_ultimo_round():
    assert gm.loss_holdout_modello_rilasciato(_json()) == pytest.approx(0.0945)
    assert gm.loss_addestramento_locale(_json()) == pytest.approx(0.006)


def test_loss_holdout_assente_o_non_valida():
    assert gm.loss_holdout_modello_rilasciato(_json(nm=None)) is None
    assert gm.loss_holdout_modello_rilasciato(_json(nm=float("nan"))) is None
    assert gm.loss_holdout_modello_rilasciato(_json(surface="client")) is None
    assert gm.loss_holdout_modello_rilasciato({"config": {}, "per_round": {}}) is None


def _scrivi_dump(cartella, effetto_membro, n=3000, seed_list=(42, 123, 456, 789, 1234)):
    """Dump sintetici: difficolta' intrinseca stabile per record, pool del seed =
    20% membri + 20% non-membri, punteggio = -difficolta' + rumore
    (+ effetto se membro)."""
    rng = random.Random(0)
    difficolta = [rng.gauss(0, 1) for _ in range(n)]
    for s in seed_list:
        r = random.Random(s)
        idx = list(range(n))
        r.shuffle(idx)
        membri, non_membri = idx[: n // 5], idx[n // 5: 2 * n // 5]
        rec = []
        for i in membri + non_membri:
            m = i in set(membri)
            score = -difficolta[i] + r.gauss(0, 0.3) + (effetto_membro if m else 0.0)
            rec.append({"session_id": f"s{i}", "is_member": m, "composed_score": score})
        (cartella / f"per_sample_seed{s}.json").write_text(json.dumps({"records": rec}))


def test_senza_effetto_di_appartenenza_il_test_appaiato_e_nullo(tmp_path):
    _scrivi_dump(tmp_path, effetto_membro=0.0)
    r = wc.analizza(str(tmp_path), n_perm=30, soglia=90.0, pavimento=75.0, min_seed=2)
    # la difficolta' stabile produce un eccesso sia fra i membri sia fra i non-membri
    assert r["z"] > 3 and r["controllo_non_membri"]["z"] > 3
    assert abs(r["appaiato"]["z"]) < 3
    assert r["lettura"].startswith("nessun segnale")


def test_con_effetto_di_appartenenza_il_test_appaiato_lo_vede(tmp_path):
    _scrivi_dump(tmp_path, effetto_membro=0.5)
    r = wc.analizza(str(tmp_path), n_perm=30, soglia=90.0, pavimento=75.0, min_seed=2)
    assert r["appaiato"]["z"] > 3
    assert r["appaiato"]["differenza_media"] > 0
    assert r["lettura"] == "segnale di appartenenza per record"


def test_main_con_piu_gruppi(tmp_path, monkeypatch, capsys):
    # regressione: la variabile del ciclo non deve oscurare gli argomenti
    for nome in ("g1", "g2"):
        (tmp_path / nome).mkdir()
        _scrivi_dump(tmp_path / nome, effetto_membro=0.0, n=600)
    out = tmp_path / "out" / "l.json"
    monkeypatch.setattr(sys, "argv", ["x", "--gruppo", f"a={tmp_path / 'g1'}",
                                      "--gruppo", f"b={tmp_path / 'g2'}",
                                      "--permutazioni", "5", "--output", str(out)])
    wc.main()
    d = json.loads(out.read_text())
    assert set(d["gruppi"]) == {"a", "b"}
    assert "appaiato" in d["gruppi"]["a"]
