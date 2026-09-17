#!/usr/bin/env python3
"""
scripts/plot_roc_log_scale.py — task #54, Sprint 10zz+29 (2026-09-03).

Plot delle curve ROC COMPLETE in scala log-log, su richiesta esplicita
dell'utente dopo due citazioni di Carlini et al. 2022 discusse in chat:
(1) "la bontà di un attacco di privacy si misura unicamente osservando cosa
accade quando il tasso di falsi positivi è prossimo allo zero" — un singolo
numero aggregato (AUC-ROC, §1 di docs/MetricsReference_DSN2027.md) o anche un
punto a soglia fissa (TPR@0.01, §2) mostra solo un campione discreto di
quella regione; (2) "per visualizzare efficacemente questo comportamento
[...] il paper introduce l'uso di scale logaritmiche sugli assi, permettendo
di 'zoomare' esattamente sulla regione critica in cui si consuma la
violazione reale della privacy" — da qui asse FPR e asse TPR entrambi in
scala log (stessa convenzione della Figure 3 del paper LiRA originale).

Input: uno o più file JSON scritti da run_yeom()/run_shadow()/
run_lira() con --roc-curve-dump-dir (vedi _full_roc_curve() e
_write_roc_curve_dump() in scripts/run_experiments.py) — non calcola nulla
da zero, plotta solo le curve già salvate.

Uso:
    python scripts/plot_roc_log_scale.py \\
        experiments/_smoke/roc_curves_lira.json \\
        experiments/_smoke/roc_curves_yeom.json \\
        --label "LiRA" --label "Yeom" \\
        --which last --output figures/roc_log_log.png

    # Curva composta (solo LiRA/Shadow/Yeom con "composed" popolato — solo
    # LiRA lo scrive oggi, vedi run_lira()):
    python scripts/plot_roc_log_scale.py \\
        experiments/_smoke/roc_curves_lira.json --which composed \\
        --subkey lira --output figures/roc_composed.png

Dipendenza: matplotlib (opzionale, vedi pyproject.toml extra "viz") — import
lazy qui sotto, coerente con torch/sklearn altrove nel progetto: questo
script resta importabile (e testabile: vedi tests/test_plot_roc_log_scale.py
per la logica di selezione/parsing, pura Python) anche senza matplotlib
installato; solo la chiamata a main()/genera il plot lo richiede.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Floor usato per evitare log(0) — sotto questo valore un punto della curva
# non è distinguibile in scala log comunque (vedi docstring di clip_for_log).
_LOG_FLOOR = 1e-4


def load_roc_dump(path: str) -> dict[str, Any]:
    """Carica un file scritto da _write_roc_curve_dump() — nessuna
    trasformazione, solo json.load con un errore leggibile se il file non
    ha la forma attesa (manca "attack" o "per_round")."""
    with open(path) as f:
        dump = json.load(f)
    if "attack" not in dump or "per_round" not in dump:
        raise ValueError(
            f"{path}: non sembra un dump di curve ROC valido (manca "
            f"'attack' o 'per_round') — atteso un file scritto da "
            f"run_yeom()/run_shadow()/run_lira() con "
            f"--roc-curve-dump-dir."
        )
    return dump


def select_curve(
    dump: dict[str, Any], which: str = "last", subkey: str | None = None
) -> dict[str, list[float]] | None:
    """
    Estrae UNA curva {"fpr": [...], "tpr": [...]} da un dump.

    which="last": ultimo round disponibile in "per_round" (il modello più
        allenato/convergente — di solito il più informativo per un plot).
    which="composed": la chiave "composed" del dump (solo LiRA la scrive
        oggi, run_lira() — None per Yeom/Shadow, che non hanno un concetto
        di "composto multi-round" salvato).

    subkey: per i dump "shaped" come LiRA (dump["attack"] == "lira"), ogni
        entry per-round/composed è essa stessa un dict con più curve
        possibili ({"lira": {...}, "canary": {...}, "canary_raw": {...},
        "sablayrolles": {...}, "lira_log": {...} — queste ultime due solo
        per-round, task #58/#61, non nel "composed"}) — subkey seleziona quale.
        Default "lira" quando applicabile. Per i dump Yeom/Shadow (attack in
        {"yeom","shadow"}) l'entry è già direttamente {"fpr":[...],
        "tpr":[...]} — subkey ignorato.

    Returns:
        {"fpr": [...], "tpr": [...]} o None se non trovata (round vuoto,
        composed mai popolato, subkey assente per quel round).
    """
    is_lira_shaped = dump.get("attack") == "lira"
    effective_subkey = subkey or ("lira" if is_lira_shaped else None)

    if which == "composed":
        entry = dump.get("composed")
    elif which == "last":
        per_round = dump.get("per_round", {})
        if not per_round:
            return None
        # Le chiavi round sono stringhe dopo un round-trip JSON (int come
        # chiave dict non è supportato da JSON) — riconvertite per il max().
        last_round = max(per_round.keys(), key=lambda k: int(k))
        entry = per_round[last_round]
    else:
        raise ValueError(f"which deve essere 'last' o 'composed', ricevuto: {which!r}")

    if entry is None:
        return None
    if effective_subkey is not None:
        return entry.get(effective_subkey)
    return entry


def clip_for_log(values: list[float], floor: float = _LOG_FLOOR) -> list[float]:
    """
    Sostituisce ogni valore < floor con floor (mai 0.0, log(0) non è
    definito) — sklearn.roc_curve() include sempre un primo punto a
    fpr=tpr=0.0 (nessun positivo predetto), che altrimenti non sarebbe
    rappresentabile in scala log. floor=1e-4 di default: sotto questa
    soglia un punto non sarebbe comunque distinguibile visivamente dagli
    altri punti vicino a zero nello stesso plot.
    """
    return [max(v, floor) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ROC in scala log-log (task #54) — su entrambi gli assi, "
            "per rendere visibile il comportamento a FPR/TPR vicino a zero "
            "che un AUC-ROC medio o un singolo punto TPR@fixed-FPR "
            "nascondono (Carlini et al. 2022)."
        )
    )
    parser.add_argument("dumps", nargs="+", help="1+ file JSON da --roc-curve-dump-dir.")
    parser.add_argument(
        "--label", action="append", default=[],
        help="Etichetta legenda per ogni dump, stesso ordine (default: nome file).",
    )
    parser.add_argument("--which", choices=["last", "composed"], default="last")
    parser.add_argument(
        "--subkey", default=None,
        help="Per dump LiRA-shaped: 'lira' (default), 'canary', o 'canary_raw'.",
    )
    parser.add_argument("--fpr-floor", type=float, default=_LOG_FLOOR)
    parser.add_argument("--output", type=str, required=True, help="Path PNG di output.")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")  # nessun display richiesto, solo file
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "ERRORE: matplotlib non installato. Installare con "
            "`pip install matplotlib` (o l'extra 'viz' del progetto: "
            "`pip install -e .[viz]`).",
            file=sys.stderr,
        )
        sys.exit(1)

    labels = list(args.label)
    if labels and len(labels) != len(args.dumps):
        print(
            f"ERRORE: {len(labels)} --label per {len(args.dumps)} dump — "
            f"servono zero label (uso i nomi file) o esattamente una per dump.",
            file=sys.stderr,
        )
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(6, 6))
    plotted = 0
    for i, path in enumerate(args.dumps):
        dump = load_roc_dump(path)
        curve = select_curve(dump, which=args.which, subkey=args.subkey)
        if curve is None:
            print(f"ATTENZIONE: {path} — nessuna curva trovata per which={args.which!r} "
                  f"subkey={args.subkey!r}, salto.", file=sys.stderr)
            continue
        fpr = clip_for_log(curve["fpr"], args.fpr_floor)
        tpr = clip_for_log(curve["tpr"], args.fpr_floor)
        label = labels[i] if labels else path
        ax.plot(fpr, tpr, label=label, linewidth=1.5)
        plotted += 1

    if plotted == 0:
        print("ERRORE: nessuna curva plottata (vedi warning sopra).", file=sys.stderr)
        sys.exit(1)

    # Diagonale di riferimento (attacco = caso random, TPR=FPR ad ogni
    # soglia) — resta una retta anche in scala log-log (log(y)=log(x)).
    ax.plot([args.fpr_floor, 1.0], [args.fpr_floor, 1.0], linestyle="--",
            color="gray", linewidth=1.0, label="caso random (TPR=FPR)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(args.fpr_floor, 1.0)
    ax.set_ylim(args.fpr_floor, 1.0)
    ax.set_xlabel("False Positive Rate (scala log)")
    ax.set_ylabel("True Positive Rate (scala log)")
    ax.set_title("Curva ROC — scala log-log (Carlini et al. 2022)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    # Fix (2026-09-04, Sprint 10zz+45 — trovato da un run reale dell'utente,
    # FileNotFoundError perché "figures/" non esisteva ancora): stessa classe
    # di bug del task #60 in run_experiments.py (dump diagnostici scritti
    # prima che la directory padre esistesse) — qui matplotlib/PIL non creano
    # la directory di destinazione da soli. Fix strutturale identico: mkdir
    # preventivo, così --output non richiede mai più una directory
    # pre-esistente o un `mkdir -p` manuale.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Plot salvato: {args.output} ({plotted} curve)")


if __name__ == "__main__":
    main()
