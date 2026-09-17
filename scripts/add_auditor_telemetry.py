#!/usr/bin/env python3
"""
Sprint 10zz+119 — serializza la telemetria del Privacy Auditor nel JSON.

Problema: il subscriber produce un AuditReport per nodo a ogni round
(privacy_auditor_subscriber.py:179) e lo passa a ids.analyze_round(), ma nel
JSON sopravvive solo auditor_overhead_seconds. Privacy score, epsilon
consumato, sensibilita' osservata e minacce rilevate vengono scartati.

Conseguenza per il paper: oggi si puo' solo affermare che l'Auditor non ha
sollevato allarmi. Con i report salvati si puo' mostrare che la sua telemetria
e' indistinguibile fra la cella con memorizzazione indotta (canary raw 0.72) e
quella con record-DP (0.54) — cioe' trasformare un'assenza di allarmi in una
misura di NON-CORRELAZIONE fra budget dichiarato e rischio misurato, che e' la
tesi centrale del lavoro.

Aggiunge un blocco "auditor" a ids_results[round_num] nei due rami dove
reports e' disponibile.

Uso:  python3 add_auditor_telemetry.py --dry-run
      python3 add_auditor_telemetry.py
"""
import re, sys
from pathlib import Path

P = Path("scripts/run_experiments.py")
DRY = "--dry-run" in sys.argv
lines = P.read_text().splitlines(keepends=True)

# ---- helper a livello di modulo ------------------------------------------
HELPER = '''

def _auditor_telemetry(reports: dict) -> dict:
    """
    Telemetria del Privacy Auditor per un round (Sprint 10zz+119).

    reports: node_id -> AuditReport (src/core/base_auditor.py), prodotto dal
    subscriber a ogni round. Prima di questo sprint veniva consumato da
    ids.analyze_round() e poi scartato: nel JSON restava solo l'overhead.

    NOTA INTERPRETATIVA, da riportare nel paper. privacy_score ed epsilon qui
    sono un PROXY calcolato dalla norma degli aggiornamenti, non una misura del
    rischio di appartenenza e non una contabilita' DP formale (epsilon per
    round e' scalato sulla mediana dei peer, quindi dipende dai dati, mentre la
    contabilita' DP e' data-independent per costruzione). Servono a verificare
    che un nodo resti dentro i parametri dichiarati, non a stimare quanta
    informazione esce. La distinzione e' il motivo per cui questo lavoro
    instanzia avversari invece di leggere il budget.
    """
    if not reports:
        return None
    per_node = {}
    for node_id, r in reports.items():
        per_node[node_id] = {
            "privacy_score":    getattr(r, "privacy_score", None),
            "epsilon":          getattr(r, "epsilon", None),
            "threats_detected": list(getattr(r, "threats_detected", []) or []),
            "metadata":         dict(getattr(r, "metadata", {}) or {}),
        }
    scores = [v["privacy_score"] for v in per_node.values() if v["privacy_score"] is not None]
    eps    = [v["epsilon"]       for v in per_node.values() if v["epsilon"]       is not None]
    return {
        "per_node":            per_node,
        "n_nodes":             len(per_node),
        "privacy_score_min":   min(scores) if scores else None,
        "privacy_score_mean":  (sum(scores) / len(scores)) if scores else None,
        "epsilon_round_max":   max(eps) if eps else None,
        "epsilon_round_mean":  (sum(eps) / len(eps)) if eps else None,
        "n_threats":           sum(len(v["threats_detected"]) for v in per_node.values()),
        "note": ("privacy_score/epsilon sono un proxy data-dependent dalla norma "
                 "degli update, NON una misura di rischio di membership ne' una "
                 "contabilita' DP formale."),
    }

'''

src = "".join(lines)
if "_auditor_telemetry" not in src:
    i_first_def = next(i for i, l in enumerate(lines) if re.match(r"^def \w+\(", l))
    lines.insert(i_first_def, HELPER.lstrip("\n"))
    print(f"helper inserito prima della riga {i_first_def+1}")
else:
    print("helper gia' presente")

# ---- inserisce la voce nei blocchi ids_results che hanno reports ---------
# ancoraggio: le righe 'auditor_overhead_seconds": _auditor_overhead,'
targets = [i for i, l in enumerate(lines)
           if '"auditor_overhead_seconds": _auditor_overhead,' in l]
print(f"punti di inserimento trovati: {[i+1 for i in targets]}")
added = 0
for i in reversed(targets):
    if "_auditor_telemetry(" in lines[i + 1] if i + 1 < len(lines) else False:
        continue
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i + 1,
        f'{ind}# Sprint 10zz+119: telemetria Auditor. Vedi _auditor_telemetry()\n'
        f'{ind}# per la nota sul fatto che privacy_score/epsilon sono un proxy.\n'
        f'{ind}"auditor": _auditor_telemetry(reports),\n')
    added += 1
print(f"voci inserite: {added}")

if DRY:
    for i, l in enumerate(lines):
        if '"auditor": _auditor_telemetry' in l:
            print("\n".join(x.rstrip() for x in lines[max(0,i-4):i+2])); print("---")
    sys.exit(0)

P.write_text("".join(lines))
print("\nfile scritto.")
