"""Rumore DP da un generatore dedicato (seme_rumore_dp, 2026-10-26).

Nel canary persona i bracci A e B usano lo stesso seed. Con il rumore preso dal
generatore globale di torch ricevevano anche lo stesso rumore, che si
cancellava nella differenza appaiata L_B(u) - L_A(u): l'AUC appaiata restava
circa 0,72 anche con sigma = 1000. Con seme_rumore_dp il rumore viene da un
generatore proprio, diverso fra i bracci e indipendente dal resto della
casualita'. Senza la chiave il comportamento e' quello di prima.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from ml.autoencoder_trainer import AutoencoderTrainer  # noqa: E402
from ml.gradient_manager import GradientManager  # noqa: E402
from test_subject_dp import _make_sessions, _ml  # noqa: E402


def _addestra(ml: dict, sessioni, consuma_globale: bool = False) -> list[torch.Tensor]:
    torch.manual_seed(7)
    t = AutoencoderTrainer(config=ml, node_id="n", cluster_id="c")
    if consuma_globale:
        torch.rand(1000)  # sposta il generatore globale dopo l'inizializzazione
    return [w.clone() for w in t.train_local(sessioni, round_num=1).weights]


def _uguali(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _ml_record(**extra) -> dict:
    return {"input_dim": 6, "lr": 0.01, "epochs": 1, "batch_size": 8, "proximal_mu": 0.0,
            "seed": 42, "record_dp": {"enabled": True, "max_grad_norm": 1.0, "noise_multiplier": 1.0},
            **extra}


class TestTrainer:
    @pytest.mark.parametrize("fabbrica", [_ml, _ml_record], ids=["soggetto", "record"])
    def test_semi_diversi_rumore_diverso(self, fabbrica):
        ss = _make_sessions(60, seed=3)
        a = _addestra({**fabbrica(), "seme_rumore_dp": 101}, ss)
        b = _addestra({**fabbrica(), "seme_rumore_dp": 202}, ss)
        assert not _uguali(a, b)

    @pytest.mark.parametrize("fabbrica", [_ml, _ml_record], ids=["soggetto", "record"])
    def test_stesso_seme_non_dipende_dal_generatore_globale(self, fabbrica):
        ss = _make_sessions(60, seed=3)
        a = _addestra({**fabbrica(), "seme_rumore_dp": 101}, ss)
        b = _addestra({**fabbrica(), "seme_rumore_dp": 101}, ss, consuma_globale=True)
        assert _uguali(a, b)

    @pytest.mark.parametrize("fabbrica", [_ml, _ml_record], ids=["soggetto", "record"])
    def test_senza_seme_come_prima(self, fabbrica):
        """Senza la chiave (o con None) il rumore viene dal generatore globale:
        stessi pesi a parita' di seed globale, pesi diversi se lo si sposta."""
        ss = _make_sessions(60, seed=3)
        a = _addestra(fabbrica(), ss)
        b = _addestra({**fabbrica(), "seme_rumore_dp": None}, ss)
        c = _addestra(fabbrica(), ss, consuma_globale=True)
        assert _uguali(a, b)
        assert not _uguali(a, c)


class TestGradientManager:
    CFG = {"epsilon": 64.0, "delta": 1e-5, "max_grad_norm": 1.0}

    def test_semi(self):
        w = [torch.zeros(5, 3), torch.zeros(4)]
        a = GradientManager({**self.CFG, "seme_rumore_dp": 1})._add_noise(w)
        torch.rand(100)
        b = GradientManager({**self.CFG, "seme_rumore_dp": 1})._add_noise(w)
        c = GradientManager({**self.CFG, "seme_rumore_dp": 2})._add_noise(w)
        assert _uguali(a, b)
        assert not _uguali(a, c)

    def test_senza_seme_come_prima(self):
        gm = GradientManager(dict(self.CFG))
        w = [torch.zeros(5, 3)]
        torch.manual_seed(0)
        a = gm._add_noise(w)
        torch.manual_seed(0)
        atteso = torch.randn(5, 3) * gm.sigma
        assert torch.equal(a[0], atteso)


class TestCanaryPersona:
    def test_seme_per_braccio(self):
        cp = pytest.importorskip("canary_persona")
        assert cp.seme_rumore_braccio(42, "A") != cp.seme_rumore_braccio(42, "B")
        assert cp.seme_rumore_braccio(42, "A") == cp.seme_rumore_braccio(42, "A")
        assert cp.seme_rumore_braccio(42, "A") != cp.seme_rumore_braccio(123, "A")


class TestCombinaConRumore:
    """2026-10-27: l'hash del config non deve includere il seme del rumore, che e' diverso
    fra i bracci; le run del 2026-10-26 (hash con il seme) devono potersi combinare."""

    def _coppia(self, tmp_path, cp, **diff):
        import json as _json
        from test_canary_persona import _braccio
        g1 = [f"a{i}" for i in range(5)]
        g2 = [f"b{i}" for i in range(5)]
        pn = {p: {u: float(i) for i, u in enumerate(g1 + g2)} for p in ("training", "fresche")}
        for br in ("A", "B"):
            _braccio(tmp_path / f"{br}.json", br, g1, g2, pn)
            d = _json.loads((tmp_path / f"{br}.json").read_text())
            d.update({k: (v[0] if br == "A" else v[1]) for k, v in diff.items()})
            (tmp_path / f"{br}.json").write_text(_json.dumps(d))
        return tmp_path / "A.json", tmp_path / "B.json", tmp_path / "AB.json"

    def test_sha_config_ignora_il_seme(self):
        cp = pytest.importorskip("canary_persona")
        a = {"ml": {"x": 1, "seme_rumore_dp": 1}}
        b = {"ml": {"x": 1, "seme_rumore_dp": 2}}
        assert cp.sha_config(a) == cp.sha_config(b)
        assert cp.sha_config(a) != cp.sha_config({"ml": {"x": 2, "seme_rumore_dp": 1}})
        assert a["ml"]["seme_rumore_dp"] == 1   # il config originale non viene toccato

    def test_run_del_26_ottobre_si_combinano(self, tmp_path):
        cp = pytest.importorskip("canary_persona")
        fa, fb, fo = self._coppia(tmp_path, cp, seme_rumore_dp=(11, 22), config_sha=("aaa", "bbb"))
        assert "1" in cp.combina(fa, fb, fo)["per_round"]

    def test_hash_nuovo_diverso_rifiutato(self, tmp_path):
        cp = pytest.importorskip("canary_persona")
        fa, fb, fo = self._coppia(tmp_path, cp, seme_rumore_dp=(11, 22), config_sha=("aaa", "bbb"),
                                  sha_senza_rumore=(True, True))
        with pytest.raises(cp.BracciIncoerenti):
            cp.combina(fa, fb, fo)

    def test_stesso_seme_rifiutato(self, tmp_path):
        cp = pytest.importorskip("canary_persona")
        fa, fb, fo = self._coppia(tmp_path, cp, seme_rumore_dp=(11, 11))
        with pytest.raises(cp.BracciIncoerenti):
            cp.combina(fa, fb, fo)
