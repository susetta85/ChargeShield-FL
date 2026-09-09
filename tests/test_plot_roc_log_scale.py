"""
Test puro-Python (nessuna dipendenza matplotlib/torch/sklearn) per la logica
di caricamento/selezione/clip di scripts/plot_roc_log_scale.py (task #54,
Sprint 10zz+29, 2026-09-03) — il plot vero e proprio richiede matplotlib
(import lazy, non testato qui, coerente col resto del progetto), ma
load_roc_dump()/select_curve()/clip_for_log() sono puro Python e vanno
testate direttamente.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plot_roc_log_scale import (  # noqa: E402
    _LOG_FLOOR,
    clip_for_log,
    load_roc_dump,
    select_curve,
)


def test_clip_for_log_replaces_zero_with_floor():
    assert clip_for_log([0.0, 0.5, 1.0]) == [_LOG_FLOOR, 0.5, 1.0]


def test_clip_for_log_custom_floor():
    assert clip_for_log([0.0, 1e-5, 0.1], floor=1e-3) == [1e-3, 1e-3, 0.1]


def test_clip_for_log_never_lowers_values_above_floor():
    values = [0.2, 0.5, 0.9]
    assert clip_for_log(values) == values


def test_load_roc_dump_rejects_malformed_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(ValueError, match="non sembra un dump"):
        load_roc_dump(str(bad))


def test_load_roc_dump_accepts_valid_file(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"attack": "yeom", "per_round": {}}))
    dump = load_roc_dump(str(good))
    assert dump["attack"] == "yeom"


def test_select_curve_last_round_yeom_shaped():
    dump = {
        "attack": "yeom",
        "per_round": {
            "1": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
            "3": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.6, 1.0]},
            "2": {"fpr": [0.0, 0.3, 1.0], "tpr": [0.0, 0.4, 1.0]},
        },
    }
    # "last" deve scegliere il round numericamente più alto (3), non
    # l'ultimo per ordine di inserimento nel dict (2) — le chiavi sono
    # stringhe dopo un round-trip JSON, il confronto numerico è essenziale.
    curve = select_curve(dump, which="last")
    assert curve == {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.6, 1.0]}


def test_select_curve_lira_shaped_defaults_to_lira_subkey():
    dump = {
        "attack": "lira",
        "per_round": {
            "1": {
                "lira": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
                "canary": {"fpr": [0.0, 1.0], "tpr": [0.0, 0.5]},
            },
        },
    }
    curve = select_curve(dump, which="last")
    assert curve == {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}


def test_select_curve_lira_shaped_explicit_canary_subkey():
    dump = {
        "attack": "lira",
        "per_round": {
            "1": {
                "lira": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
                "canary": {"fpr": [0.0, 1.0], "tpr": [0.0, 0.5]},
            },
        },
    }
    curve = select_curve(dump, which="last", subkey="canary")
    assert curve == {"fpr": [0.0, 1.0], "tpr": [0.0, 0.5]}


def test_select_curve_composed_returns_none_when_absent():
    dump = {"attack": "yeom", "per_round": {"1": {"fpr": [0.0], "tpr": [0.0]}}}
    assert select_curve(dump, which="composed") is None


def test_select_curve_composed_lira_shaped():
    dump = {
        "attack": "lira",
        "per_round": {},
        "composed": {"lira": {"fpr": [0.0, 1.0], "tpr": [0.0, 0.9]}},
    }
    curve = select_curve(dump, which="composed")
    assert curve == {"fpr": [0.0, 1.0], "tpr": [0.0, 0.9]}


def test_select_curve_empty_per_round_returns_none():
    dump = {"attack": "yeom", "per_round": {}}
    assert select_curve(dump, which="last") is None


def test_select_curve_invalid_which_raises():
    dump = {"attack": "yeom", "per_round": {}}
    with pytest.raises(ValueError, match="which deve essere"):
        select_curve(dump, which="bogus")
