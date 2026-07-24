# src/plugins/attacks/lira.py
"""
LiRAAttack — wrapper BaseAttack sottile su run_lira() (Likelihood Ratio
Attack, Carlini et al. 2022 — ★ attacco primario del progetto), invariato in
scripts/run_experiments.py. Vedi yeom.py per la spiegazione dell'import lazy.

A differenza di Yeom/Shadow, run_lira() accetta parametri opzionali specifici
(n_shadow, shadow_epochs_cap, no_dp, dp_mode, cluster_membership) — questo
wrapper li estrae da **kwargs con gli stessi default che main() usava già
alla chiamata diretta, così main() può passare un unico kwargs dict a tutti
gli attacchi registrati (vedi src/plugins/attacks/__init__.py) senza if/else
per attacco, e YeomAttack/ShadowAttack semplicemente ignorano le chiavi che
non usano.
"""

from __future__ import annotations

from typing import Any

from core.base_attack import BaseAttack


class LiRAAttack(BaseAttack):
    """LiRA (Carlini et al., 2022) — ★ attacco primario. Vedi run_lira() per
    la storia completa dei 5 round di fix empirici dietro questa implementazione."""

    name = "lira"

    def run(
        self,
        cfg: dict,
        train_sessions: list[dict[str, Any]],
        holdout_sessions: list[dict[str, Any]],
        fl_results: dict[int, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        from run_experiments import run_lira  # noqa: PLC0415 (lazy, vedi yeom.py)

        return run_lira(
            cfg, train_sessions, holdout_sessions, fl_results,
            n_shadow=kwargs.get("n_shadow", 8),
            shadow_epochs_cap=kwargs.get("shadow_epochs_cap"),
            no_dp=kwargs.get("no_dp", False),
            dp_mode=kwargs.get("dp_mode", "dp-fedavg"),
            cluster_membership=kwargs.get("cluster_membership"),
        )
