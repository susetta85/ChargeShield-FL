# src/plugins/attacks/shadow.py
"""
ShadowAttack — wrapper BaseAttack sottile su run_shadow() (calibrated
shadow-model MIA, ispirato a Carlini et al. 2022), in
scripts/run_experiments.py. Vedi yeom.py per la spiegazione dell'import lazy.

Rinominato da run_shadow() a run_shadow() (Sprint 10zz+107,
2026-09-15) — stesso motivo di yeom.py: "fedmia" nel nome era fuorviante,
non solo storico, ora che run_fedmia_gradient() è un attacco diverso nello
stesso file. Alias run_shadow=run_shadow resta in run_experiments.py
solo per compatibilità con codice non toccato.
"""

from __future__ import annotations

from typing import Any

from core.base_attack import BaseAttack


class ShadowAttack(BaseAttack):
    """Shadow-model calibrated MIA — vedi run_shadow(). Forza intermedia
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
        from run_experiments import run_shadow  # noqa: PLC0415 (lazy, vedi yeom.py)

        # Sprint 10zz+29 (2026-09-03, task #54) — vedi yeom.py per la
        # motivazione, stesso pattern qui per Shadow.
        _roc_dir = kwargs.get("roc_curve_dump_dir")
        _roc_path = None
        if _roc_dir is not None:
            import os  # noqa: PLC0415 (lazy, coerente col resto del modulo)
            _roc_path = os.path.join(_roc_dir, "roc_curves_shadow.json")

        # Sprint 10zz+94 (2026-09-15, task #155) — vedi yeom.py per la
        # motivazione: no_dp/dp_mode servono solo a
        # cfg["shadow"]["observation_surface"]=="client" (Strada B). Default
        # invariati, nessun impatto sul comportamento pubblicato.
        return run_shadow(
            cfg, train_sessions, holdout_sessions, fl_results,
            roc_curve_dump_path=_roc_path,
            no_dp=kwargs.get("no_dp", False),
            dp_mode=kwargs.get("dp_mode", "dp-fedavg"),
        )
