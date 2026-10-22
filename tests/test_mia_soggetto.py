"""Attacco realistico al soggetto: divisione delle unita' e dati del riferimento del server."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

ms = pytest.importorskip("mia_soggetto")


def _sessioni():
    s = []
    for u in range(30):                      # 30 soggetti con 20 sessioni, su due siti
        for j in range(20):
            s.append({"user_id": f"u{u}", "site_id": "A" if u % 2 else "B",
                      "start_time": f"2020-01-{j + 1:02d}T10:00:00", "id": len(s)})
    for u in range(5):                       # 5 soggetti con 3 sessioni
        for j in range(3):
            s.append({"user_id": f"p{u}", "site_id": "A", "start_time": f"2020-02-{j + 1:02d}", "id": len(s)})
    for j in range(40):                      # 40 sessioni senza chiave
        s.append({"user_id": None, "site_id": "B", "start_time": f"2020-03-{j % 28 + 1:02d}", "id": len(s)})
    return s


@pytest.mark.parametrize("livello", ["entita", "gruppo"])
def test_riferimento_solo_dati_del_server(livello):
    s = _sessioni()
    ins = ms.insiemi(s, "user_id", livello, seed=42, min_sess=10, k=7)
    valutate = {i for u in ins["membri"] + ins["non_membri"] for i in ins["unita"][u]}
    rif, train = set(ins["riferimento"]), set(ins["train"])
    assert rif == {i for u in ins["server"] for i in ins["unita"][u]}
    assert not rif & train and not rif & valutate
    assert not set(ins["riserva"]) & train and not set(ins["riserva"]) & rif
    # le unita' membro sono tutte nel training, le non membro mai
    assert {i for u in ins["membri"] for i in ins["unita"][u]} <= train
    assert not {i for u in ins["non_membri"] for i in ins["unita"][u]} & train
    # terzi uguali
    assert len(ins["membri"]) == len(ins["non_membri"])


def test_entita_soggetti_piccoli_nello_sfondo():
    s = _sessioni()
    ins = ms.insiemi(s, "user_id", "entita", seed=1, min_sess=10, k=7)
    assert len(ins["unita"]) == 30
    piccoli = {i for i, x in enumerate(s) if str(x["user_id"]).startswith("p")}
    assert piccoli <= set(ins["train"])


def test_gruppo_consecutivi_stesso_sito_e_avanzi_nello_sfondo():
    s = _sessioni()
    unita, sfondo = ms.unita_gruppo(s, "user_id", 7)
    assert len(unita) == 30 * 2              # 20 sessioni -> 2 gruppi da 7, avanzano 6
    for u, idx in unita.items():
        assert len(idx) == 7
        assert len({s[i]["user_id"] for i in idx}) == 1 and len({s[i]["site_id"] for i in idx}) == 1
        tempi = [s[i]["start_time"] for i in idx]
        assert tempi == sorted(tempi)
    assert sum(1 for i in sfondo if str(s[i]["user_id"]).startswith("u")) == 30 * 6


def test_poche_unita_rifiutate():
    s = _sessioni()
    with pytest.raises(ValueError):
        ms.insiemi(s, "site_id", "entita", seed=1, min_sess=10, k=7)


def test_divisione_deterministica():
    s = _sessioni()
    a = ms.insiemi(s, "user_id", "gruppo", seed=7, min_sess=10, k=7)
    b = ms.insiemi(s, "user_id", "gruppo", seed=7, min_sess=10, k=7)
    c = ms.insiemi(s, "user_id", "gruppo", seed=8, min_sess=10, k=7)
    assert a["membri"] == b["membri"] and a["membri"] != c["membri"]
