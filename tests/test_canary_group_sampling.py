"""
Test puro-Python (nessuna dipendenza torch, assente in questo sandbox —
stesso limite di tests/test_fedmia_gradient_pairing.py e
tests/test_mia_advantage.py) per _sample_preserving_canary_groups()
(Sprint 10zz+16, 2026-09-02, fix alla contaminazione degli shadow LiRA
confermata in Sprint 10zz+15).

Replica ESATTA della logica in scripts/run_experiments.py (non importabile
qui: `import torch` a livello di modulo) — se questa funzione cambia
laggiù, aggiornare insieme qui.
"""
import random

import pytest


def _sample_preserving_canary_groups(rng, pool, n):
    groups = {}
    units = []
    for s in pool:
        g = s.get("_canary_group")
        if g is None:
            units.append([s])
        else:
            groups.setdefault(g, []).append(s)
    for g_sessions in groups.values():
        units.append(g_sessions)

    if all(len(u) == 1 for u in units):
        return rng.sample(pool, n)

    shuffled_units = units[:]
    rng.shuffle(shuffled_units)
    sampled = []
    for unit in shuffled_units:
        if len(sampled) >= n:
            break
        sampled.extend(unit)
    return sampled


def _mk(idx, group=None):
    s = {"_id": idx}
    if group is not None:
        s["_canary_group"] = group
    return s


def test_no_canary_groups_matches_plain_sample_exactly():
    # Ogni run reale/pubblicato: nessuna sessione ha _canary_group -> deve
    # comportarsi ESATTAMENTE come rng.sample(pool, n), stessa sequenza RNG.
    pool = [_mk(i) for i in range(50)]
    n = 20
    a = _sample_preserving_canary_groups(random.Random(42), pool, n)
    b = random.Random(42).sample(pool, n)
    assert a == b


def test_canary_group_never_split_across_sample_boundary():
    # Un gruppo canary da 30 duplicati deve finire TUTTO dentro o TUTTO
    # fuori dal campione, mai spezzato a metà.
    pool = [_mk(i) for i in range(20)] + [_mk(f"c{i}", group="canary_m0") for i in range(30)]
    for seed in range(20):
        sampled = _sample_preserving_canary_groups(random.Random(seed), pool, 25)
        canary_in_sample = [s for s in sampled if s.get("_canary_group") == "canary_m0"]
        assert len(canary_in_sample) in (0, 30), (
            f"gruppo canary spezzato: {len(canary_in_sample)}/30 nel campione (seed={seed})"
        )


def test_singleton_groups_behave_as_individual_sessions():
    # I gemelli non-membro (canary_n{j}, un solo elemento per gruppo, vedi
    # inject_canaries()) non hanno il problema di frammentazione — ogni
    # gruppo è un singleton, quindi il fast-path si applica comunque.
    pool = [_mk(i) for i in range(10)] + [_mk(f"n{i}", group=f"canary_n{i}") for i in range(5)]
    n = 8
    a = _sample_preserving_canary_groups(random.Random(7), pool, n)
    b = random.Random(7).sample(pool, n)
    assert a == b


def test_sample_size_approximates_target_with_multi_element_groups():
    # Con gruppi multi-elemento la dimensione non è esatta (dichiarato nel
    # docstring) ma deve restare nell'intorno ragionevole di n, mai 0 e mai
    # l'intera pool se n < len(pool).
    pool = [_mk(i) for i in range(40)] + [_mk(f"c{i}", group="canary_m0") for i in range(30)]
    sampled = _sample_preserving_canary_groups(random.Random(3), pool, 35)
    assert 0 < len(sampled) < len(pool)


def test_multiple_independent_canary_groups_each_stay_atomic():
    pool = (
        [_mk(i) for i in range(10)]
        + [_mk(f"a{i}", group="canary_m0") for i in range(5)]
        + [_mk(f"b{i}", group="canary_m1") for i in range(5)]
    )
    for seed in range(20):
        sampled = _sample_preserving_canary_groups(random.Random(seed), pool, 12)
        g0 = sum(1 for s in sampled if s.get("_canary_group") == "canary_m0")
        g1 = sum(1 for s in sampled if s.get("_canary_group") == "canary_m1")
        assert g0 in (0, 5)
        assert g1 in (0, 5)
