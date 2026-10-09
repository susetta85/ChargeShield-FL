"""Test per l'attaccante che riscala l'update osservato (Sprint 10zz+170).

Copre `_rescale_update_weights()` in scripts/run_experiments.py: con s = 1
restituisce il modello osservato, con s generico ref + s*(osservato - ref) sui
parametri float, e lascia invariati i buffer BatchNorm e i tensori interi
(stessa convenzione di GradientManager._clip_weights()). Copre anche il caso
che motiva l'attaccante: un update tagliato a norma C e poi riscalato del
fattore di taglio torna esattamente l'update originale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiments as run_exp  # noqa: E402

_KEYS = ["enc.0.weight", "enc.0.bias", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"]


def _pesi(seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(4, 3, generator=g),
        torch.randn(4, generator=g),
        torch.randn(4, generator=g),
        torch.rand(4, generator=g) + 0.5,
        torch.tensor(7 + seed, dtype=torch.int64),
    ]


def test_s1_restituisce_il_modello_osservato():
    obs, ref = _pesi(1), _pesi(2)
    out = run_exp._rescale_update_weights(obs, ref, 1.0, _KEYS)
    for o, w in zip(out, obs):
        assert torch.allclose(o.float(), w.float(), atol=1e-6)


def test_formula_sui_parametri_e_buffer_invariati():
    obs, ref = _pesi(3), _pesi(4)
    s = 8.0
    out = run_exp._rescale_update_weights(obs, ref, s, _KEYS)
    for i in (0, 1):
        assert torch.allclose(out[i], ref[i] + s * (obs[i] - ref[i]), atol=1e-5)
    # buffer BatchNorm e contatore intero: valore osservato, non riscalato
    for i in (2, 3, 4):
        assert torch.equal(out[i], obs[i])
    assert out[4].dtype == torch.int64


def test_riscalare_del_fattore_di_taglio_recupera_l_update():
    ref = _pesi(5)
    vero = _pesi(6)
    # update tagliato a norma C = 1 sui soli parametri (come _clip_weights)
    delta = [vero[i] - ref[i] for i in (0, 1)]
    norma = float(torch.sqrt(sum((d ** 2).sum() for d in delta)))
    C = 1.0
    f = min(1.0, C / norma)
    osservato = [ref[0] + f * delta[0], ref[1] + f * delta[1], vero[2], vero[3], vero[4]]
    out = run_exp._rescale_update_weights(osservato, ref, 1.0 / f, _KEYS)
    for i in (0, 1):
        assert torch.allclose(out[i], vero[i], atol=1e-4)


def test_lunghezze_diverse_restituiscono_none():
    obs, ref = _pesi(7), _pesi(8)
    assert run_exp._rescale_update_weights(obs[:4], ref, 2.0, _KEYS) is None
    assert run_exp._rescale_update_weights(obs, ref, 2.0, _KEYS[:4]) is None


_CFG = str(PROJECT_ROOT / "config" / "experiment_canary_multisite.yaml")


@pytest.mark.skipif(not Path(_CFG).exists(), reason="config del repository non trovato")
class TestFlagCanaryRescale:
    @pytest.mark.parametrize("extra", [
        ["--canary-rescale", "0,2"],
        ["--canary-rescale", "-1"],
        ["--canary-rescale", "a,b"],
        ["--canary-rescale", "1,16", "--skip-attacks", "lira"],
    ])
    def test_valori_non_validi_escono_prima_dei_dati(self, monkeypatch, extra):
        monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--config", _CFG, *extra])
        monkeypatch.setattr(run_exp, "load_sessions",
                            lambda *a, **k: pytest.fail("non deve caricare i dati"))
        with pytest.raises(SystemExit) as exc:
            run_exp.main()
        assert exc.value.code == 1

    def test_valori_validi_arrivano_nel_config(self, monkeypatch):
        class _Stop(Exception):
            pass

        seen: dict = {}

        def _load(cfg, *a, **k):
            seen.update(cfg.get("lira", {}))
            raise _Stop

        monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--config", _CFG,
                                          "--canary-rescale", "1,4,16"])
        monkeypatch.setattr(run_exp, "load_sessions", _load)
        with pytest.raises(_Stop):
            run_exp.main()
        assert seen["canary_rescale"] == [1.0, 4.0, 16.0]
