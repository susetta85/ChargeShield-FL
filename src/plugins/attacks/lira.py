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
    la storia completa dei fix empirici dietro questa implementazione (5
    storici + 2 sul pooling cross-cluster, 2026-08-11/12).

    PROTOTIPO 2026-08-12 — "LiRA composto" (multi-round): oltre alle 10 AUC
    per-round indipendenti (poi mediate), somma il log-likelihood-ratio dello
    STESSO campione su tutti i round prima di calcolare un unico AUC finale —
    la combinazione ottima (Neyman-Pearson) per evidenza indipendente
    ripetuta, pensata per sfruttare il vero ε COMPOSTO su più round (vedi
    "epsilon_cumulative_naive" in config) invece del solo ε per-round che ogni
    AUC indipendente vede. Piggyback sulla stessa retraining degli shadow già
    fatta da run_lira() per il calcolo standard — NON una funzione/attacco
    separato che rifarebbe il training da capo (raddoppierebbe il costo
    computazionale per un beneficio nullo, dato che userebbe gli stessi
    shadow). Risultato attaccato al round finale con chiavi dedicate
    (composed_lira_*) per non entrare in conflitto con lira_auc_roc del round
    — NON ancora verificato con un'esecuzione reale (torch non disponibile in
    questo sandbox); il report Excel non ha ancora una colonna dedicata per
    questi campi, verrebbero solo salvati nel JSON grezzo per ora."""

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

        _composed: dict[str, Any] = {}
        results = run_lira(
            cfg, train_sessions, holdout_sessions, fl_results,
            n_shadow=kwargs.get("n_shadow", 8),
            shadow_epochs_cap=kwargs.get("shadow_epochs_cap"),
            no_dp=kwargs.get("no_dp", False),
            dp_mode=kwargs.get("dp_mode", "dp-fedavg"),
            cluster_membership=kwargs.get("cluster_membership"),
            composed_output=_composed,
        )
        if results and _composed:
            _final_round = max(results.keys())
            results[_final_round].update(_composed)
        return results
