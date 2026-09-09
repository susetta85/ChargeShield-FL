# src/plugins/attacks/fedmia_gradient.py
"""
FedMIAGradientAttack — wrapper BaseAttack sottile su run_fedmia_gradient()
(scripts/run_experiments.py), Sprint 10zz (2026-09-01).

NON registrato in ATTACK_REGISTRY (src/plugins/attacks/__init__.py) — resta
opt-in, invocabile solo passando --include-fedmia-gradient a
scripts/run_experiments.py (vedi main()). Motivo: run_fedmia_gradient() è un
PRIMO DRAFT mai eseguito con torch reale (vedi il suo docstring per il
dettaglio) — a differenza di Yeom/Shadow/LiRA, che sono nel registro di
default perché già validati da centinaia di run reali. Attivarlo di default
significherebbe far girare, in ogni esperimento futuro, un secondo run_lira()
completo (stesso costo computazionale) più questo attacco, per un numero
tuttora da interpretare — non accettabile senza conferma esplicita
dell'utente per ogni esecuzione, finché non viene promosso (stesso principio
già applicato a hour_of_day circular encoding, Sprint 10eee, e all'entity-aware
split, Sprint 10rr, prima di essere validati).

Import lazy di run_fedmia_gradient dentro run() (non a livello di modulo),
stesso motivo/pattern di yeom.py/shadow.py/lira.py: questo file vive sotto
src/, che non ha scripts/ nel proprio sys.path — solo
scripts/run_experiments.py, eseguito come script, può importare se stesso
sotto il nome "run_experiments".
"""

from __future__ import annotations

from typing import Any

from core.base_attack import BaseAttack


class FedMIAGradientAttack(BaseAttack):
    """FedMIA a granularità round+cluster (composizione membri/non-membri del
    subset IN degli shadow di LiRA) — vedi run_fedmia_gradient() per il
    disegno completo e i limiti noti dichiarati prima di qualunque run reale."""

    name = "fedmia_gradient"

    def run(
        self,
        cfg: dict,
        train_sessions: list[dict[str, Any]],
        holdout_sessions: list[dict[str, Any]],
        fl_results: dict[int, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        from run_experiments import run_fedmia_gradient  # noqa: PLC0415 (lazy, vedi docstring)

        # Sprint 10zz+21 (2026-09-02): stesso pattern by-reference di
        # LiRAAttack.run() (src/plugins/attacks/lira.py) — pooling cross-round
        # per cluster, merge nel round finale sotto, vedi run_fedmia_gradient()
        # Args (composed_output) per il razionale.
        _composed: dict = {}
        results = run_fedmia_gradient(
            cfg, train_sessions, holdout_sessions, fl_results,
            n_shadow=kwargs.get("n_shadow", 8),
            shadow_epochs_cap=kwargs.get("shadow_epochs_cap"),
            no_dp=kwargs.get("no_dp", False),
            dp_mode=kwargs.get("dp_mode", "dp-fedavg"),
            cluster_membership=kwargs.get("cluster_membership"),
            # Sprint 10zz+17 (2026-09-02): test diagnostico mirato scala-vs-
            # confondimento-strutturale, vedi run_fedmia_gradient() Args.
            normalize_vectors=kwargs.get("fedmia_gradient_normalize", False),
            composed_output=_composed,
        )
        if results and _composed:
            _final_round = max(results.keys())
            results[_final_round].update(_composed)
        return results
