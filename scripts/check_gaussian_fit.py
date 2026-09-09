#!/usr/bin/env python3
"""
scripts/check_gaussian_fit.py — task #57, Sprint 10zz+32 (2026-09-03).

Verifica empirica dell'assunzione di normalità richiesta dal fit
PARAMETRICO di LiRA (μ_in/σ_in, μ_out/σ_out — vedi
docs/MetricsReference_DSN2027.md §3, "Modellazione parametrica vs non
parametrica"), su richiesta esplicita dell'utente dopo aver ricostruito il
ruolo del logit-scaling in Carlini et al. 2022: gli autori verificano
esplicitamente (loro Fig. 4/8) che la confidenza/loss grezza NON è
approssimativamente Gaussiana, e applicano un logit-scaling PRIMA del fit
Gaussiano proprio per questo motivo — solo il logit-transformato risulta
ben approssimato da una Gaussiana nel loro caso. ChargeShield-FL fitta la
Gaussiana direttamente sulla MSE di ricostruzione grezza, senza alcuna
trasformazione — non esiste un logit naturale per un errore di
ricostruzione (nessuna probabilità/confidenza in [0,1] da cui partire).
Questo script controlla se la MSE grezza è comunque ragionevolmente
Gaussiana, o se — come per la confidenza grezza nel paper originale — serve
una trasformazione (qui il candidato più naturale è il logaritmo, dato che
la MSE è un valore semi-illimitato positivo, non bounded in [0,1] come una
confidenza: log() mappa (0,∞) su (-∞,∞), stesso spirito del logit ma per un
dominio diverso).

Input: un file JSON scritto da run_lira() con --raw-loss-dump (vedi
run_lira(raw_loss_dump_path=...) in scripts/run_experiments.py) — non
calcola nulla da zero, analizza solo i dati già raccolti (le liste
_diag_raw_loss_members/_diag_raw_loss_nonmembers, finora salvate solo come
media in lira_debug_raw_loss_member_mean/..._nonmember_mean).

Metodo: skewness campionaria, curtosi in eccesso campionaria, e statistica
di Jarque-Bera (JB = n/6 * (S^2 + K^2/4), dove S=skewness, K=curtosi in
eccesso) — sotto l'ipotesi nulla di normalità, JB è asintoticamente
chi-quadro con 2 gradi di libertà; il valore critico al 5% è 5.99 (valore
noto, hardcoded qui per evitare una dipendenza scipy — sklearn/scipy sono
assenti in alcuni ambienti di sviluppo di questo progetto, vedi note
sparse nel resto del codebase). JB più basso = più vicino a una Gaussiana.
Confronta MSE grezza vs log(MSE + eps) — lo stesso tipo di confronto che
Carlini et al. fanno tra confidenza/log-loss/logit nella loro Fig. 4/8,
adattato al nostro dominio (nessun logit possibile).

Uso:
    python scripts/check_gaussian_fit.py experiments/_smoke/raw_loss.json
    python scripts/check_gaussian_fit.py experiments/_smoke/raw_loss.json --round 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

# Valore critico chi-quadro(2), p=0.05 — sorgente: tabelle standard di
# distribuzione chi-quadro (es. Wikipedia "Chi-squared distribution",
# voce "Table of χ² value vs p-value"), non ricalcolato qui.
_JB_CHI2_2_CRITICAL_005 = 5.99


def skewness(values: list[float]) -> float:
    """Skewness campionaria (momento standardizzato di ordine 3, g1 di
    Fisher-Pearson, senza correzione per bias — coerente con la
    definizione usata nella statistica di Jarque-Bera classica)."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((x - mean) ** 2 for x in values) / n
    m3 = sum((x - mean) ** 3 for x in values) / n
    if m2 == 0:
        return 0.0
    return m3 / (m2 ** 1.5)


def excess_kurtosis(values: list[float]) -> float:
    """Curtosi in eccesso campionaria (g2 di Fisher-Pearson — 3, cosi' che
    una Gaussiana perfetta dia 0, non 3)."""
    n = len(values)
    if n < 4:
        return 0.0
    mean = sum(values) / n
    m2 = sum((x - mean) ** 2 for x in values) / n
    m4 = sum((x - mean) ** 4 for x in values) / n
    if m2 == 0:
        return 0.0
    return (m4 / (m2 ** 2)) - 3.0


def jarque_bera(values: list[float]) -> float:
    """
    Statistica di Jarque-Bera: JB = n/6 * (S^2 + K^2/4). Sotto H0
    (normalità), JB ~ chi-quadro(2) asintoticamente — JB=0 sse skewness e
    curtosi in eccesso sono ENTRAMBE esattamente 0 (Gaussiana perfetta).
    Nessuna dipendenza scipy: solo la formula, il confronto con il valore
    critico è fatto dal chiamante usando _JB_CHI2_2_CRITICAL_005.
    """
    n = len(values)
    if n < 4:
        return 0.0
    s = skewness(values)
    k = excess_kurtosis(values)
    return (n / 6.0) * (s ** 2 + (k ** 2) / 4.0)


def log_transform(values: list[float], eps: float = 1e-8) -> list[float]:
    """log(x + eps) — eps evita log(0) per MSE esattamente nulla (raro ma
    possibile per una ricostruzione quasi perfetta)."""
    return [math.log(x + eps) for x in values]


def summarize_pool(values: list[float], eps: float = 1e-8) -> dict[str, Any]:
    """
    Confronto raw vs log-transform per un pool di valori (member O
    nonmember, non entrambi insieme — la forma della distribuzione va
    controllata separatamente per le due ipotesi IN/OUT, esattamente come
    fanno μ_in/σ_in vs μ_out/σ_out nel fit di LiRA).
    """
    n = len(values)
    if n < 4:
        return {"n": n, "note": "troppo pochi campioni (<4) per una stima affidabile"}
    log_values = log_transform(values, eps)
    raw_jb = jarque_bera(values)
    log_jb = jarque_bera(log_values)
    return {
        "n": n,
        "raw": {
            "skewness": round(skewness(values), 4),
            "excess_kurtosis": round(excess_kurtosis(values), 4),
            "jarque_bera": round(raw_jb, 2),
            "rejects_normality_at_5pct": raw_jb > _JB_CHI2_2_CRITICAL_005,
        },
        "log": {
            "skewness": round(skewness(log_values), 4),
            "excess_kurtosis": round(excess_kurtosis(log_values), 4),
            "jarque_bera": round(log_jb, 2),
            "rejects_normality_at_5pct": log_jb > _JB_CHI2_2_CRITICAL_005,
        },
        "log_transform_improves_fit": log_jb < raw_jb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verifica se la MSE grezza usata dal fit Gaussiano di LiRA è "
            "approssimativamente normale (skewness/curtosi/Jarque-Bera), "
            "confrontata con la sua trasformata logaritmica — stesso tipo "
            "di controllo che Carlini et al. 2022 fanno per giustificare il "
            "logit-scaling della confidenza, qui adattato a una MSE di "
            "ricostruzione (nessun logit applicabile)."
        )
    )
    parser.add_argument("dump", help="File JSON scritto da run_lira() con --raw-loss-dump.")
    parser.add_argument(
        "--round", type=int, default=None,
        help="Round da analizzare (default: l'ultimo disponibile nel dump).",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon per log(x + eps).")
    args = parser.parse_args()

    with open(args.dump) as f:
        dump = json.load(f)

    per_round = dump.get("per_round", {})
    if not per_round:
        print("ERRORE: nessun round nel dump.", file=sys.stderr)
        sys.exit(1)

    if args.round is not None:
        round_key = str(args.round)
        if round_key not in per_round:
            print(f"ERRORE: round {args.round} non presente nel dump (disponibili: "
                  f"{sorted(per_round.keys(), key=int)}).", file=sys.stderr)
            sys.exit(1)
    else:
        round_key = max(per_round.keys(), key=lambda k: int(k))

    entry = per_round[round_key]
    member = entry.get("member_losses", [])
    nonmember = entry.get("nonmember_losses", [])

    print(f"Dump: {args.dump}")
    print(f"Config: seed={dump.get('seed')} epsilon={dump.get('epsilon')} "
          f"no_dp={dump.get('no_dp')} dp_mode={dump.get('dp_mode')}")
    print(f"Round analizzato: {round_key} (di {len(per_round)} disponibili)")
    print(f"Soglia Jarque-Bera (chi2(2), p=0.05): {_JB_CHI2_2_CRITICAL_005} "
          f"— JB sopra questo valore rigetta la normalità al 5%")
    print()

    for label, values in (("MEMBER", member), ("NONMEMBER", nonmember)):
        print(f"── {label} (n={len(values)}) ──")
        result = summarize_pool(values, args.eps)
        if "note" in result:
            print(f"  {result['note']}")
            print()
            continue
        for kind in ("raw", "log"):
            r = result[kind]
            verdict = "RIGETTA normalità" if r["rejects_normality_at_5pct"] else "non rigetta normalità"
            print(f"  {kind:>4}: skew={r['skewness']:>8} excess_kurt={r['excess_kurtosis']:>8} "
                  f"JB={r['jarque_bera']:>10} → {verdict}")
        better = "log-transform" if result["log_transform_improves_fit"] else "MSE grezza"
        print(f"  → più vicino a una Gaussiana: {better}")
        print()

    print(
        "Nota: questo controllo è a livello di POOL (tutti i campioni scorati in un round,\n"
        "membri e non-membri separatamente) — analogo alla Fig. 4/8 di Carlini et al. 2022\n"
        "(shape della distribuzione aggregata), non un test per-campione (per cui servirebbero\n"
        "molti più shadow model per campione di quanti ne usa questo progetto, n_shadow=8-32)."
    )


if __name__ == "__main__":
    main()
