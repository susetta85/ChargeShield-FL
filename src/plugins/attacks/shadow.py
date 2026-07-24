# src/plugins/attacks/shadow.py
"""
ShadowAttack — wrapper BaseAttack sottile su run_fedmia_shadow() (calibrated
shadow-model MIA, ispirato a Carlini et al. 2022), invariato in
scripts/run_experiments.py. Vedi yeom.py per la spiegazione dell'import lazy.
"""

from __future__ import annotations

from typing import Any

from core.base_attack import BaseAttack


class ShadowAttack(BaseAttack):
    """Shadow-model calibrated MIA — vedi run_fedmia_shadow(). Forza intermedia
    fra Yeom (debole) e LiRA (primario)."""

    name = "shadow"

    def run(
        self,
        cfg: dict,
        train_sessions: list[dict[str, Any]],
        holdout_sessions: list[dict[str, Any]],
        fl_results: dict[int, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        from run_experiments import run_fedmia_shadow  # noqa: PLC0415 (lazy, vedi yeom.py)

        return run_fedmia_shadow(cfg, train_sessions, holdout_sessions, fl_results)
