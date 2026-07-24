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

        return run_fedmia(cfg, train_sessions, holdout_sessions, fl_results)
