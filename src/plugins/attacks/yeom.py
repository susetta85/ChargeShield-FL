# src/plugins/attacks/yeom.py
"""
YeomAttack — wrapper BaseAttack sottile su run_fedmia() (Yeom et al. 2018,
loss-based MIA), invariato in scripts/run_experiments.py.

Import lazy di run_fedmia dentro run() (non a livello di modulo): questo file
vive sotto src/, che NON ha scripts/ nel proprio sys.path — solo
scripts/run_experiments.py, quando eseguito come script, aggiunge la propria
directory (auto, comportamento standard di Python) e può quindi importare
sé stesso sotto il nome "run_experiments" (stesso pattern già usato con
successo da scripts/run_nvflare_mia.py). Import lazy = questo modulo resta
importabile (e testabile) senza torch; solo la chiamata a run() lo richiede,
esattamente come richiedeva già run_fedmia() prima di questo wrapper.
"""

from __future__ import annotations

from typing import Any

from core.base_attack import BaseAttack


class YeomAttack(BaseAttack):
    """Loss-based MIA (Yeom et al., 2018) — baseline debole, vedi run_fedmia()."""

    name = "yeom"

    def run(
        self,
        cfg: dict,
        train_sessions: list[dict[str, Any]],
        holdout_sessions: list[dict[str, Any]],
        fl_results: dict[int, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        from run_experiments import run_fedmia  # noqa: PLC0415 (lazy, vedi docstring)

        # Sprint 10zz+29 (2026-09-03, task #54) — stesso meccanismo di
        # per_sample_dump_path in lira.py: la directory arriva da kwargs
        # (passata a TUTTI gli attacchi da main()), qui costruiamo il file
        # specifico per Yeom dentro quella directory. None se il flag CLI
        # --roc-curve-dump-dir non è stato passato (default, zero impatto).
        _roc_dir = kwargs.get("roc_curve_dump_dir")
        _roc_path = None
        if _roc_dir is not None:
            import os  # noqa: PLC0415 (lazy, coerente col resto del modulo)
            _roc_path = os.path.join(_roc_dir, "roc_curves_yeom.json")

        # Sprint 10zz+94 (2026-09-15, task #155) — no_dp/dp_mode servono a
        # run_fedmia() solo quando cfg["yeom"]["observation_surface"]=="client"
        # (Strada B, selezione raw_updates/updates dp_mode-aware, stesso
        # meccanismo già in uso da LiRA in lira.py). Default invariati:
        # observation_surface="global" non li usa, quindi non c'è impatto sul
        # comportamento pubblicato finché il flag opt-in resta al default.
        return run_fedmia(
            cfg, train_sessions, holdout_sessions, fl_results,
            roc_curve_dump_path=_roc_path,
            no_dp=kwargs.get("no_dp", False),
            dp_mode=kwargs.get("dp_mode", "dp-fedavg"),
        )
