"""Shadow con dati propri (shadow.fonte_membri: esterna).

I membri dello shadow devono venire solo dalla calibrazione dell'holdout: mai dai
client, mai dall'holdout di valutazione, mai canary. Calibrazione e membri disgiunti."""
import random

from plugins.attacks.shadow2 import dividi_esterni, e_canary


def _sessioni(n_sito):
    out = []
    for g, n in n_sito.items():
        for k in range(n):
            out.append({"site_id": g, "id": f"{g}-{k}", "_canary_role": "canary" if k % 7 == 0 else None})
    return out


def test_membri_solo_da_calibrazione_e_disgiunti():
    sess = _sessioni({"caltech": 60, "jpl": 40})
    gruppi = {"caltech": list(range(0, 60)), "jpl": list(range(60, 100))}
    idx_cal = [i for i in range(100) if i % 3 != 0]  # calibrazione dell'holdout (non tutta)
    membri, calib = dividi_esterni(gruppi, sess, idx_cal, 0.5, seed=42)
    tutti_membri = [i for v in membri.values() for i in v]
    assert set(tutti_membri).isdisjoint(calib)
    assert set(tutti_membri) <= set(idx_cal)
    assert set(calib) <= set(idx_cal)
    assert all(not e_canary(sess[i]) for i in tutti_membri + calib)


def test_stesso_seme_stesso_risultato():
    sess = _sessioni({"caltech": 50})
    gruppi = {"caltech": list(range(50))}
    idx_cal = list(range(50))
    a = dividi_esterni(gruppi, sess, idx_cal, 0.5, seed=7)
    b = dividi_esterni(gruppi, sess, idx_cal, 0.5, seed=7)
    assert a == b


def test_frazione_per_sito():
    sess = _sessioni({"caltech": 40, "jpl": 40})
    gruppi = {"caltech": list(range(40)), "jpl": list(range(40, 80))}
    idx_cal = list(range(80))
    membri, _ = dividi_esterni(gruppi, sess, idx_cal, 0.25, seed=1)
    for g in gruppi:
        liberi = [i for i in gruppi[g] if not e_canary(sess[i])]
        assert abs(len(membri[g]) - round(0.25 * len(liberi))) <= 1


def test_riduci_membri_taglia_e_sottoinsieme():
    from plugins.attacks.shadow2 import riduci_membri
    idx = {"caltech": list(range(0, 80)), "jpl": list(range(100, 140))}
    out = riduci_membri(idx, {"caltech": 10, "jpl": 5}, seed=3)
    assert [len(out["caltech"]), len(out["jpl"])] == [10, 5]
    assert set(out["caltech"]) <= set(idx["caltech"]) and set(out["jpl"]) <= set(idx["jpl"])
    assert out == riduci_membri(idx, {"caltech": 10, "jpl": 5}, seed=3)


def test_riduci_membri_non_supera_il_pool():
    from plugins.attacks.shadow2 import riduci_membri
    out = riduci_membri({"office1": [1, 2, 3]}, {"office1": 10}, seed=0)
    assert out["office1"] == [1, 2, 3]
