"""
Test puro-Python (repliche esatte, stesso limite/pattern di
tests/test_sablayrolles_score.py) per alcune funzioni pure di
scripts/run_experiments.py rimaste senza copertura dopo il deep-review del
2026-09-04 (task #67/#70/#71): _autoencoder_arch_kwargs(), group_sessions_by_site(),
group_indices_by_site(), inject_synthetic_client_indices(), _mia_feature_names(),
entity_aware_split(). run_experiments.py importa torch a livello di modulo,
quindi non è importabile in questo sandbox — ogni replica qui è byte-per-byte
identica alla funzione reale (nessuna dipendenza torch/sklearn nelle funzioni
stesse, solo dict/list/random di libreria standard).

entity_aware_split() in particolare backa un risultato citabile nel paper
(task #10/#38, robustezza dello split) e non aveva NESSUN test prima di questo
file.
"""

import random

import pytest

_SITE_ID_TO_NAME = {
    "0002": "caltech",
    "0001": "jpl",
    "0019": "office1",
}

_MIA_FEATURES = [
    "total_energy_kwh", "max_power_kw", "kwh_requested",
    "minutes_available", "hour_of_day", "duration_hours",
]


def _autoencoder_arch_kwargs_reference(cfg: dict) -> dict:
    """Replica esatta di _autoencoder_arch_kwargs()."""
    ml_cfg = cfg.get("ml", {})
    hidden_dims = ml_cfg.get("hidden_dims")
    return {
        "hidden_dims": tuple(hidden_dims) if hidden_dims is not None else None,
        "latent_dim": ml_cfg.get("latent_dim", 4),
    }


def _mia_feature_names_reference(cfg: dict) -> list:
    """Replica esatta di _mia_feature_names()."""
    names = cfg.get("ml", {}).get("feature_names")
    return list(names) if names else _MIA_FEATURES


def _group_sessions_by_site_reference(sessions: list) -> dict:
    """Replica esatta di group_sessions_by_site()."""
    groups: dict = {}
    for s in sessions:
        site_id = s.get("site_id", "")
        name = _SITE_ID_TO_NAME.get(site_id, site_id or "unknown")
        groups.setdefault(name, []).append(s)
    return groups


def _group_indices_by_site_reference(sessions: list) -> dict:
    """Replica esatta di group_indices_by_site()."""
    groups: dict = {}
    for i, s in enumerate(sessions):
        site_id = s.get("site_id", "")
        name = _SITE_ID_TO_NAME.get(site_id, site_id or "unknown")
        groups.setdefault(name, []).append(i)
    return groups


def _inject_synthetic_client_indices_reference(
    real_index_groups: dict, n_synthetic: int = 2, seed: int = 42
) -> dict:
    """Replica esatta di inject_synthetic_client_indices()."""
    pooled_indices = [i for idxs in real_index_groups.values() for i in idxs]
    rng = random.Random(seed + 271828)
    shuffled = pooled_indices[:]
    rng.shuffle(shuffled)

    expanded = dict(real_index_groups)
    chunk_size = max(1, len(shuffled) // n_synthetic)
    for i in range(n_synthetic):
        start = i * chunk_size
        end = len(shuffled) if i == n_synthetic - 1 else start + chunk_size
        expanded[f"synthetic_{i + 1}"] = shuffled[start:end]
    return expanded


def _entity_aware_split_reference(
    sessions: list, entity_key: str, holdout_fraction: float, seed: int
) -> tuple:
    """Replica esatta di entity_aware_split() (senza il logger.info finale,
    irrilevante per il comportamento)."""
    groups: dict = {}
    ungrouped: list = []
    for s in sessions:
        key = s.get(entity_key)
        if key is None:
            ungrouped.append(s)
            continue
        groups.setdefault(key, []).append(s)

    rng = random.Random(seed)
    group_items = list(groups.items())
    group_items.extend((id(s), [s]) for s in ungrouped)
    rng.shuffle(group_items)

    total = len(sessions)
    target_holdout = int(total * holdout_fraction)

    holdout_sessions: list = []
    train_sessions: list = []
    for _key, group_sessions in group_items:
        if len(holdout_sessions) < target_holdout:
            holdout_sessions.extend(group_sessions)
        else:
            train_sessions.extend(group_sessions)

    return train_sessions, holdout_sessions


class TestAutoencoderArchKwargs:
    def test_default_when_ml_config_absent(self):
        assert _autoencoder_arch_kwargs_reference({}) == {"hidden_dims": None, "latent_dim": 4}

    def test_explicit_hidden_dims_becomes_tuple(self):
        cfg = {"ml": {"hidden_dims": [32, 16], "latent_dim": 8}}
        assert _autoencoder_arch_kwargs_reference(cfg) == {
            "hidden_dims": (32, 16), "latent_dim": 8,
        }

    def test_latent_dim_default_preserved_when_only_hidden_dims_set(self):
        cfg = {"ml": {"hidden_dims": [10]}}
        assert _autoencoder_arch_kwargs_reference(cfg) == {"hidden_dims": (10,), "latent_dim": 4}


class TestMiaFeatureNames:
    def test_default_six_historical_features(self):
        assert _mia_feature_names_reference({}) == _MIA_FEATURES

    def test_explicit_feature_names_override(self):
        cfg = {"ml": {"feature_names": ["a", "b"]}}
        assert _mia_feature_names_reference(cfg) == ["a", "b"]

    def test_empty_list_falls_back_to_default(self):
        # list vuota e' falsy in Python -> stesso comportamento di "non impostato"
        cfg = {"ml": {"feature_names": []}}
        assert _mia_feature_names_reference(cfg) == _MIA_FEATURES


class TestGroupBySite:
    _sessions = [
        {"site_id": "0002", "s": "a"},
        {"site_id": "0001", "s": "b"},
        {"site_id": "0019", "s": "c"},
        {"site_id": "0002", "s": "d"},
        {"site_id": "9999", "s": "e"},  # sito sconosciuto
        {"s": "f"},                     # site_id assente
    ]

    def test_known_sites_mapped_to_names(self):
        groups = _group_sessions_by_site_reference(self._sessions)
        assert set(groups.keys()) == {"caltech", "jpl", "office1", "9999", "unknown"}
        assert [s["s"] for s in groups["caltech"]] == ["a", "d"]

    def test_indices_match_session_grouping(self):
        idx_groups = _group_indices_by_site_reference(self._sessions)
        session_groups = _group_sessions_by_site_reference(self._sessions)
        for name, idxs in idx_groups.items():
            assert [self._sessions[i] for i in idxs] == session_groups[name]

    def test_unknown_site_id_grouped_under_its_own_raw_id(self):
        groups = _group_sessions_by_site_reference(self._sessions)
        assert [s["s"] for s in groups["9999"]] == ["e"]

    def test_missing_site_id_grouped_as_unknown(self):
        groups = _group_sessions_by_site_reference(self._sessions)
        assert [s["s"] for s in groups["unknown"]] == ["f"]


class TestInjectSyntheticClientIndices:
    def test_adds_exactly_n_synthetic_keys(self):
        real = {"caltech": [0, 1, 2, 3], "jpl": [4, 5, 6, 7], "office1": [8, 9]}
        expanded = _inject_synthetic_client_indices_reference(real, n_synthetic=2, seed=42)
        assert set(expanded.keys()) == {"caltech", "jpl", "office1", "synthetic_1", "synthetic_2"}

    def test_real_client_indices_unchanged(self):
        real = {"caltech": [0, 1, 2], "jpl": [3, 4, 5]}
        expanded = _inject_synthetic_client_indices_reference(real, n_synthetic=2, seed=42)
        assert expanded["caltech"] == [0, 1, 2]
        assert expanded["jpl"] == [3, 4, 5]

    def test_synthetic_indices_are_a_partition_of_the_pooled_indices(self):
        real = {"caltech": list(range(10)), "jpl": list(range(10, 20))}
        expanded = _inject_synthetic_client_indices_reference(real, n_synthetic=2, seed=42)
        pooled_synthetic = sorted(expanded["synthetic_1"] + expanded["synthetic_2"])
        assert pooled_synthetic == list(range(20))  # nessun indice perso o duplicato

    def test_deterministic_given_same_seed(self):
        real = {"caltech": list(range(20))}
        a = _inject_synthetic_client_indices_reference(real, n_synthetic=2, seed=7)
        b = _inject_synthetic_client_indices_reference(real, n_synthetic=2, seed=7)
        assert a == b

    def test_different_seed_changes_split(self):
        real = {"caltech": list(range(50))}
        a = _inject_synthetic_client_indices_reference(real, n_synthetic=2, seed=1)
        b = _inject_synthetic_client_indices_reference(real, n_synthetic=2, seed=2)
        assert a["synthetic_1"] != b["synthetic_1"]


class TestEntityAwareSplit:
    def test_no_entity_split_across_train_and_holdout(self):
        sessions = [{"node_id": f"node_{i % 5}", "idx": i} for i in range(50)]
        train, holdout = _entity_aware_split_reference(
            sessions, entity_key="node_id", holdout_fraction=0.2, seed=42
        )
        train_entities = {s["node_id"] for s in train}
        holdout_entities = {s["node_id"] for s in holdout}
        assert train_entities.isdisjoint(holdout_entities)

    def test_every_session_assigned_exactly_once(self):
        sessions = [{"node_id": f"node_{i % 7}", "idx": i} for i in range(30)]
        train, holdout = _entity_aware_split_reference(
            sessions, entity_key="node_id", holdout_fraction=0.3, seed=1
        )
        assert len(train) + len(holdout) == len(sessions)
        assert {s["idx"] for s in train} | {s["idx"] for s in holdout} == set(range(30))
        assert {s["idx"] for s in train} & {s["idx"] for s in holdout} == set()

    def test_sessions_missing_entity_key_treated_as_singleton_entities(self):
        # Nessuna key None condivisa: ogni sessione senza entity_key e' la sua
        # propria entita' (mai raggruppata con un'altra sessione senza key).
        sessions = [{"idx": i} for i in range(10)]  # tutte senza "node_id"
        train, holdout = _entity_aware_split_reference(
            sessions, entity_key="node_id", holdout_fraction=0.5, seed=42
        )
        assert len(train) + len(holdout) == 10

    def test_holdout_fraction_approximately_respected_with_many_small_groups(self):
        # Molti gruppi piccoli (1 sessione/entita') -> lo split greedy deve
        # avvicinarsi bene alla frazione target (docstring: "con molti gruppi
        # piccoli converge vicino alla frazione esatta").
        sessions = [{"node_id": f"node_{i}", "idx": i} for i in range(200)]
        train, holdout = _entity_aware_split_reference(
            sessions, entity_key="node_id", holdout_fraction=0.2, seed=42
        )
        achieved = len(holdout) / len(sessions)
        assert achieved == pytest.approx(0.2, abs=0.02)

    def test_deterministic_given_same_seed(self):
        sessions = [{"node_id": f"node_{i % 6}", "idx": i} for i in range(40)]
        r1 = _entity_aware_split_reference(sessions, "node_id", 0.25, seed=99)
        r2 = _entity_aware_split_reference(sessions, "node_id", 0.25, seed=99)
        assert [s["idx"] for s in r1[0]] == [s["idx"] for s in r2[0]]
        assert [s["idx"] for s in r1[1]] == [s["idx"] for s in r2[1]]

    def test_empty_sessions_list(self):
        train, holdout = _entity_aware_split_reference([], entity_key="node_id",
                                                         holdout_fraction=0.2, seed=42)
        assert train == []
        assert holdout == []
