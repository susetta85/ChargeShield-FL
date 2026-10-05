"""
Ablazioni del meccanismo DP per client e attacchi saltati (2026-10-05).

Copre i flag `--dp-ablation {full,clip-only,noise-only}` e `--skip-attacks`
(linea per il paper DSN, punto 1 della coda; segnalazione 38):

- `GradientManager.clip_no_noise()`: taglia il delta a max_grad_norm e non
  aggiunge rumore (deterministico), buffer BatchNorm intatti;
- `GradientManager.noise_no_clip()`: aggiunge lo stesso rumore di privatize()
  (sigma tarato su max_grad_norm) senza tagliare;
- `run_fl_rounds()` con `cfg["experiment"]["dp_ablation"]`: stesso
  comportamento sui client reali, combinazioni non valide rifiutate;
- `run_registered_attacks(skip_attacks=...)`: gli attacchi indicati non girano;
- `main()`: le combinazioni non valide escono con codice 1 prima di caricare
  i dati.
Nessun test legge i dataset reali.
"""
from __future__ import annotations

import copy
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiments as run_exp  # noqa: E402
from ml.base_ml import GradientUpdate  # noqa: E402
from ml.gradient_manager import GradientManager  # noqa: E402


# ── GradientManager ─────────────────────────────────────────────────────────────

_KEYS = ["enc.weight", "enc.bias", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"]


def _gm(epsilon: float = 64.0, max_grad_norm: float = 1.0) -> GradientManager:
    return GradientManager({"epsilon": epsilon, "delta": 1e-5, "max_grad_norm": max_grad_norm})


def _update_and_reference(scale: float = 1.0) -> tuple[GradientUpdate, list[torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    reference = [
        torch.zeros(100, 100), torch.zeros(100),
        torch.zeros(100), torch.ones(100), torch.tensor(5, dtype=torch.int64),
    ]
    weights = [
        torch.randn(100, 100, generator=g) * scale, torch.randn(100, generator=g) * scale,
        torch.rand(100, generator=g), torch.rand(100, generator=g) + 0.5,
        torch.tensor(7, dtype=torch.int64),
    ]
    upd = GradientUpdate(
        node_id="n1", cluster_id="c1", round_num=2, weights=weights,
        gradients=None, loss=0.1, n_samples=10, metadata={},
    )
    return upd, reference


def _delta_norm(weights: list[torch.Tensor], reference: list[torch.Tensor]) -> float:
    idx = [i for i, k in enumerate(_KEYS) if k.split(".")[-1] not in
           {"running_mean", "running_var", "num_batches_tracked"}]
    return torch.cat([(weights[i].float() - reference[i].float()).flatten() for i in idx]).norm().item()


class _Collector:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def on_ml_event(self, event: Any) -> None:
        self.events.append(event)


class TestGradientManagerAblations:
    def test_clip_no_noise_clips_and_is_deterministic(self):
        gm = _gm(max_grad_norm=1.0)
        upd, ref = _update_and_reference()
        assert _delta_norm(upd.weights, ref) > 50  # delta grande: il taglio deve agire
        a = gm.clip_no_noise(upd, weight_keys=_KEYS, reference_weights=ref)
        b = gm.clip_no_noise(upd, weight_keys=_KEYS, reference_weights=ref)
        assert _delta_norm(a.weights, ref) == pytest.approx(1.0, rel=1e-4)
        for wa, wb in zip(a.weights, b.weights):
            assert torch.equal(wa, wb), "solo taglio: nessun rumore, quindi deterministico"
        assert a.metadata["dp_ablation"] == "clip-only"
        assert a.metadata["noise_perturbation_applied"] is False

    def test_clip_no_noise_matches_clip_only_numerically(self):
        """Stesso taglio del clip_only() di central: cambia solo l'etichetta."""
        gm = _gm(max_grad_norm=1.0)
        upd, ref = _update_and_reference()
        a = gm.clip_no_noise(upd, weight_keys=_KEYS, reference_weights=ref)
        b = gm.clip_only(upd, weight_keys=_KEYS, reference_weights=ref)
        for wa, wb in zip(a.weights, b.weights):
            assert torch.equal(wa, wb)

    def test_noise_no_clip_adds_sigma_noise_without_clipping(self):
        gm = _gm(epsilon=64.0, max_grad_norm=1.0)
        upd, ref = _update_and_reference()
        torch.manual_seed(123)
        out = gm.noise_no_clip(upd, weight_keys=_KEYS)
        diff = out.weights[0] - upd.weights[0]
        assert diff.std().item() == pytest.approx(gm.sigma, rel=0.05)
        assert diff.mean().abs().item() < 3 * gm.sigma / 100  # media ~0 su 10^4 valori
        # nessun taglio: il delta resta dell'ordine di quello grezzo, non a C = 1
        assert _delta_norm(out.weights, ref) > 50
        # buffer BatchNorm e interi intatti
        for i in (2, 3, 4):
            assert torch.equal(out.weights[i], upd.weights[i])
        assert out.metadata["dp_ablation"] == "noise-only"
        assert out.metadata["sigma"] == gm.sigma

    def test_noise_no_clip_has_same_noise_as_privatize(self):
        """privatize() = taglio + rumore; con lo stesso seed il rumore e' lo stesso."""
        gm = _gm(epsilon=64.0, max_grad_norm=1.0)
        upd, ref = _update_and_reference()
        clipped = gm.clip_no_noise(upd, weight_keys=_KEYS, reference_weights=ref)
        torch.manual_seed(7)
        full = gm.privatize(upd, weight_keys=_KEYS, reference_weights=ref)
        torch.manual_seed(7)
        noise_on_clipped = gm.noise_no_clip(clipped, weight_keys=_KEYS)
        for wf, wn in zip(full.weights, noise_on_clipped.weights):
            assert torch.allclose(wf.float(), wn.float(), atol=1e-6)

    def test_ablations_emit_purdue_level_2_events(self):
        gm = _gm()
        col = _Collector()
        gm.subscribe(col)
        upd, ref = _update_and_reference()
        gm.clip_no_noise(upd, weight_keys=_KEYS, reference_weights=ref)
        gm.noise_no_clip(upd, weight_keys=_KEYS)
        levels = [(e.event_type, e.purdue_level, e.metadata.get("dp_ablation")) for e in col.events]
        assert levels == [("gradient_upload", 2, "clip-only"), ("gradient_upload", 2, "noise-only")]


# ── run_fl_rounds() ─────────────────────────────────────────────────────────────

def _make_sessions(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = datetime(2020, 1, 1)
    sessions = []
    for i in range(n):
        start = base + timedelta(hours=rng.randint(0, 24 * 60), minutes=rng.randint(0, 59))
        end = start + timedelta(hours=rng.uniform(0.5, 8.0))
        sessions.append({
            "session_id": f"synthetic-{seed}-{i}",
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "total_energy_kwh": rng.uniform(1.0, 30.0), "max_power_kw": rng.uniform(3.0, 22.0),
            "kwh_requested": rng.uniform(1.0, 30.0), "minutes_available": rng.uniform(30, 600),
        })
    return run_exp.enrich_sessions(sessions)


@pytest.fixture(scope="module")
def tiny_cfg() -> dict:
    return {
        "experiment": {
            "name": "pytest_dp_ablation", "fl_rounds": 2, "seed": 42,
            "epsilon": 1.0, "delta": 1.0e-5, "max_grad_norm": 0.01,
        },
        "ml": {"input_dim": 6, "lr": 0.01, "epochs": 2, "batch_size": 16, "proximal_mu": 0.0},
        "lira": {"n_shadow": 2},
        "byzantine_attack": {"enabled": False},
    }


@pytest.fixture(scope="module")
def train_sessions() -> list[dict[str, Any]]:
    return _make_sessions(160, seed=1)


def _weight_keys(cfg: dict) -> list[str]:
    from ml.autoencoder_trainer import AutoencoderTrainer
    t = AutoencoderTrainer(config={**cfg["ml"], "seed": 42}, node_id="dummy", cluster_id="dummy")
    return t.get_weight_keys()


_BN = {"running_mean", "running_var", "num_batches_tracked"}


def _seeded_run(cfg: dict, sessions: list, seed: int, **kw: Any) -> dict:
    """run_fl_rounds() con i generatori fissati: due chiamate uguali danno lo stesso
    risultato se il meccanismo non aggiunge rumore; il seed di torch governa anche
    il rumore DP, quindi con seed diversi il rumore cambia."""
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return run_exp.run_fl_rounds(copy.deepcopy(cfg), sessions, no_dp=kw.pop("no_dp", False), **kw)


class TestRunFLRoundsAblations:
    def test_clip_only_clips_round2_deltas_and_adds_no_noise(self, tiny_cfg, train_sessions):
        cfg = copy.deepcopy(tiny_cfg)
        cfg["experiment"]["dp_ablation"] = "clip-only"
        C = cfg["experiment"]["max_grad_norm"]
        res_a = _seeded_run(cfg, train_sessions, seed=0)
        res_b = _seeded_run(cfg, train_sessions, seed=0)
        keys = _weight_keys(cfg)
        idx = [i for i, k in enumerate(keys) if k.split(".")[-1] not in _BN]
        reference = res_a[1]["global_weights"]
        for upd in res_a[2]["updates"]:
            l2 = torch.cat([(upd.weights[i].float() - reference[i].float()).flatten() for i in idx]).norm().item()
            assert l2 <= C + 1e-4, f"clip-only: delta di {upd.node_id} = {l2:.5f} oltre C = {C}"
        # nessun rumore: l'update inviato e' esattamente il grezzo tagliato,
        # ref + (raw - ref) * min(1, C / ||raw - ref||)
        for priv, raw in zip(res_a[2]["updates"], res_a[2]["raw_updates"]):
            d = [raw.weights[i].float() - reference[i].float() for i in idx]
            n = torch.cat([x.flatten() for x in d]).norm().item()
            f = min(1.0, C / n)
            for j, i in enumerate(idx):
                expected = reference[i].float() + d[j] * f
                assert torch.allclose(priv.weights[i].float(), expected, atol=1e-6), (
                    "clip-only: l'update inviato deve essere il grezzo tagliato, senza rumore"
                )
        # e la run e' riproducibile a parita' di seed
        for wa, wb in zip(res_a[2]["global_weights"], res_b[2]["global_weights"]):
            assert torch.equal(wa, wb)

    def test_noise_only_adds_sigma_noise_to_raw_updates(self, tiny_cfg, train_sessions):
        cfg = copy.deepcopy(tiny_cfg)
        cfg["experiment"]["dp_ablation"] = "noise-only"
        res = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=False)
        gm = GradientManager({k: cfg["experiment"][k] for k in ("epsilon", "delta", "max_grad_norm")})
        keys = _weight_keys(cfg)
        idx = [i for i, k in enumerate(keys) if k.split(".")[-1] not in _BN]
        diffs = []
        for priv, raw in zip(res[2]["updates"], res[2]["raw_updates"]):
            diffs += [(priv.weights[i].float() - raw.weights[i].float()).flatten() for i in idx]
        d = torch.cat(diffs)
        # update inviato = update grezzo + rumore sigma, senza la parte di taglio
        assert d.std().item() == pytest.approx(gm.sigma, rel=0.15)
        assert d.mean().abs().item() < 0.2 * gm.sigma

    def test_full_is_the_default(self, tiny_cfg, train_sessions):
        """Senza la chiave dp_ablation il comportamento e' quello di sempre (privatize)."""
        cfg = copy.deepcopy(tiny_cfg)
        torch.manual_seed(0)
        res_default = run_exp.run_fl_rounds(copy.deepcopy(cfg), train_sessions, no_dp=False)
        cfg_full = copy.deepcopy(cfg)
        cfg_full["experiment"]["dp_ablation"] = "full"
        torch.manual_seed(0)
        res_full = run_exp.run_fl_rounds(cfg_full, train_sessions, no_dp=False)
        for wa, wb in zip(res_default[2]["global_weights"], res_full[2]["global_weights"]):
            assert torch.equal(wa, wb)

    @pytest.mark.parametrize("no_dp,dp_mode", [(True, "dp-fedavg"), (False, "central")])
    def test_invalid_combinations_are_rejected(self, tiny_cfg, train_sessions, no_dp, dp_mode):
        cfg = copy.deepcopy(tiny_cfg)
        cfg["experiment"]["dp_ablation"] = "clip-only"
        with pytest.raises(ValueError):
            run_exp.run_fl_rounds(cfg, train_sessions, no_dp=no_dp, dp_mode=dp_mode)

    def test_check_dp_ablation(self):
        run_exp._check_dp_ablation("full", no_dp=True, dp_mode="central")  # full: sempre valido
        run_exp._check_dp_ablation("noise-only", no_dp=False, dp_mode="local")
        with pytest.raises(ValueError):
            run_exp._check_dp_ablation("rumore", no_dp=False, dp_mode="dp-fedavg")


# ── run_registered_attacks(skip_attacks=...) ────────────────────────────────────

class TestSkipAttacks:
    def test_skipped_attacks_do_not_run(self, monkeypatch):
        import plugins.attacks as attacks_pkg

        calls: list[str] = []

        def _fake(name: str):
            class _A:
                def run(self, *a, **k):
                    calls.append(name)
                    return {1: {f"{name}_auc_roc": 0.5}}
            _A.name = name
            return _A

        monkeypatch.setattr(attacks_pkg, "ATTACK_REGISTRY",
                            {"yeom": _fake("yeom"), "shadow": _fake("shadow"), "lira": _fake("lira")})
        out = run_exp.run_registered_attacks({}, [], [], {}, skip_attacks={"lira"})
        assert calls == ["yeom", "shadow"]
        assert set(out[1]) == {"yeom_auc_roc", "shadow_auc_roc"}

    def test_default_runs_everything(self, monkeypatch):
        import plugins.attacks as attacks_pkg

        calls: list[str] = []

        class _Y:
            name = "yeom"

            def run(self, *a, **k):
                calls.append("yeom")
                return {}

        monkeypatch.setattr(attacks_pkg, "ATTACK_REGISTRY", {"yeom": _Y})
        run_exp.run_registered_attacks({}, [], [], {})
        assert calls == ["yeom"]


# ── main(): combinazioni non valide ──────────────────────────────────────────────

_CFG = str(PROJECT_ROOT / "config" / "experiment_rq3_mu0_eps64.yaml")


@pytest.mark.skipif(not Path(_CFG).exists(), reason="config del repository non trovato")
class TestMainValidation:
    @pytest.mark.parametrize("extra", [
        ["--dp-ablation", "clip-only", "--no-dp"],
        ["--dp-ablation", "noise-only", "--dp-mode", "central"],
        ["--skip-attacks", "lira,boh"],
    ])
    def test_invalid_flags_exit_before_loading_data(self, monkeypatch, extra):
        monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--config", _CFG, *extra])
        monkeypatch.setattr(run_exp, "load_sessions",
                            lambda *a, **k: pytest.fail("non deve caricare i dati"))
        with pytest.raises(SystemExit) as exc:
            run_exp.main()
        assert exc.value.code == 1

    def test_unknown_ablation_rejected_by_argparse(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--config", _CFG, "--dp-ablation", "rumore"])
        with pytest.raises(SystemExit) as exc:
            run_exp.parse_args()
        assert exc.value.code == 2

    def test_valid_flags_reach_data_loading(self, monkeypatch):
        """Con flag validi main() supera la validazione e arriva a load_sessions."""
        class _Stop(Exception):
            pass

        seen: dict[str, Any] = {}

        def _load(cfg, *a, **k):
            seen.update(cfg["experiment"])
            raise _Stop

        monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--config", _CFG,
                                          "--dp-ablation", "clip-only", "--skip-attacks", "lira"])
        monkeypatch.setattr(run_exp, "load_sessions", _load)
        with pytest.raises(_Stop):
            run_exp.main()
        assert seen["dp_ablation"] == "clip-only"
        assert seen["skipped_attacks"] == ["lira"]
        assert seen["name"].endswith("_clip-only")
