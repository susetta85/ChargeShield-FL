#!/usr/bin/env python3
"""
Sprint 10zz+117 — aggiunge al JSON risultato i campi che oggi mancano e
senza i quali una cella record-DP e' indistinguibile da una no-DP.

Problema: il dizionario config serializzato contiene "epsilon" (budget
CLIENT-level) ma NON contiene record_dp ne' norm. Una cella record-DP a
sigma=1.0 salva quindi `epsilon: 1.0` mentre il suo budget reale, calcolato
con l'accountant RDP, vale 88.2. Chi apre l'artifact legge un numero
sbagliato di quasi due ordini di grandezza.

Aggiunge: record_dp, norm, epsilon_record_dp, record_dp_accounting_note.

Uso:  python3 fix_json_recorddp_fields.py           # applica
      python3 fix_json_recorddp_fields.py --dry-run # mostra e basta
"""
import re
import sys
from pathlib import Path

P = Path("scripts/run_experiments.py")
DRY = "--dry-run" in sys.argv

src = P.read_text()
lines = src.splitlines(keepends=True)

# ---- 1. trova la riga di ancoraggio -------------------------------------
anchor = [i for i, l in enumerate(lines)
          if '"epsilon_cumulative_best_known"' in l and ":" in l and "#" not in l.split('"')[0]]
if len(anchor) != 1:
    sys.exit(f"ERRORE: attesa 1 riga di assegnazione, trovate {len(anchor)}: "
             f"{[i+1 for i in anchor]}")
i_anchor = anchor[0]
indent = lines[i_anchor][:len(lines[i_anchor]) - len(lines[i_anchor].lstrip())]
print(f"ancoraggio: riga {i_anchor+1}, indentazione {len(indent)} spazi")

# ---- 2. individua il nome della variabile config in scope ---------------
#      risale al def che contiene l'ancoraggio e ne legge i parametri
i_def = max(i for i in range(i_anchor) if re.match(r"^def \w+\(", lines[i]))
sig = "".join(lines[i_def:i_def + 30])
cfg_name = next((c for c in ("cfg", "config", "experiment_cfg")
                 if re.search(rf"\b{c}\b", sig)), None)
if cfg_name is None:
    sys.exit("ERRORE: non trovo la variabile config nella firma di "
             f"{lines[i_def].strip()[:60]}")
n_name = next((c for c in ("train_sessions", "sessions", "all_sessions")
               if re.search(rf"\b{c}\b", sig)), None)
print(f"funzione contenitrice: {lines[i_def].strip()[:60]}")
print(f"config -> {cfg_name!r}   sessioni -> {n_name!r}")

# ---- 3. helper a livello di modulo --------------------------------------
helper = '''

def _record_dp_fields(cfg: dict, n_sessions: int | None = None) -> dict:
    """
    Campi record-DP per il JSON risultato (Sprint 10zz+117).

    Senza questi, una cella con DP per-record e' indistinguibile da una
    no-DP: il campo "epsilon" del JSON e' il budget CLIENT-level e non si
    applica al meccanismo per-record, che ha un budget suo calcolato via
    accountant RDP (a sigma=1.0, 1000 epoche, 3 round vale ~88, non 1.0).

    epsilon_record_dp resta None se l'accountant non e' installato o se
    n_sessions non e' noto: in quel caso si ricalcola a posteriori dal JSON,
    perche' record_dp/epochs/fl_rounds/batch_size sono comunque salvati.
    """
    ml = cfg.get("ml", {}) if isinstance(cfg.get("ml"), dict) else cfg
    rdp_cfg = ml.get("record_dp") or {}
    out = {
        "record_dp": rdp_cfg or None,
        "norm": ml.get("norm", "batch"),
        "batch_size": ml.get("batch_size"),
        "n_train_sessions": n_sessions,
        "epsilon_record_dp": None,
        "record_dp_accounting_note": None,
    }
    if not rdp_cfg.get("enabled"):
        out["record_dp_accounting_note"] = (
            "record-DP disattivo: il campo 'epsilon' si riferisce al "
            "meccanismo client-level (weight perturbation)."
        )
        return out

    sigma = float(rdp_cfg.get("noise_multiplier", 0.0))
    batch = int(ml.get("batch_size", 32) or 32)
    epochs = int(ml.get("epochs", 1) or 1)
    rounds = int(cfg.get("fl_rounds", 1) or 1)
    delta = float(cfg.get("delta", 1e-5) or 1e-5)

    if not n_sessions or sigma <= 0.0:
        out["record_dp_accounting_note"] = (
            "record-DP attivo ma epsilon non calcolato "
            f"(n_sessions={n_sessions}, sigma={sigma}). "
            "ATTENZIONE: il campo 'epsilon' NON si applica a questa cella."
        )
        return out

    try:
        from dp_accounting import dp_event, rdp as _rdp
        steps = (n_sessions // batch) * epochs * rounds
        acc = _rdp.RdpAccountant()
        acc.compose(
            dp_event.PoissonSampledDpEvent(
                batch / n_sessions, dp_event.GaussianDpEvent(sigma)),
            steps,
        )
        out["epsilon_record_dp"] = float(acc.get_epsilon(target_delta=delta))
        out["record_dp_accounting_note"] = (
            f"epsilon_record_dp = {out['epsilon_record_dp']:.4g} "
            f"(RDP accountant, Poisson subsampling q={batch/n_sessions:.5f}, "
            f"{steps} passi, sigma={sigma}, delta={delta}). "
            "Questo, NON il campo 'epsilon', e' il budget di questa cella."
        )
    except ImportError:
        out["record_dp_accounting_note"] = (
            "record-DP attivo; 'dp-accounting' non installato, epsilon da "
            "calcolare a posteriori. Il campo 'epsilon' NON si applica."
        )
    except Exception as exc:  # pragma: no cover
        out["record_dp_accounting_note"] = (
            f"record-DP attivo; accountant fallito ({exc}). "
            "Il campo 'epsilon' NON si applica."
        )
    return out

'''

if "_record_dp_fields" not in src:
    i_first_def = next(i for i, l in enumerate(lines) if re.match(r"^def \w+\(", l))
    lines.insert(i_first_def, helper.lstrip("\n"))
    i_anchor += helper.lstrip("\n").count("\n")
    print(f"helper inserito prima della riga {i_first_def+1}")
else:
    print("helper gia' presente, non reinserito")

# ---- 4. inserisce i campi dopo l'ancoraggio -----------------------------
n_arg = f", {n_name}" if n_name else ""
call = (f"{indent}# Sprint 10zz+117: record_dp/norm/epsilon_record_dp — senza questi\n"
        f"{indent}# un JSON record-DP e' indistinguibile da uno no-DP e il campo\n"
        f"{indent}# 'epsilon' qui sopra (client-level) viene letto come budget\n"
        f"{indent}# della cella, che e' falso.\n"
        f"{indent}**_record_dp_fields({cfg_name}{f', len({n_name})' if n_name else ''}),\n")

if "_record_dp_fields(" in "".join(lines[i_anchor:i_anchor + 8]):
    print("campi gia' inseriti, nessuna modifica")
else:
    lines.insert(i_anchor + 1, call)
    print(f"campi inseriti dopo la riga {i_anchor+1}")

if DRY:
    print("\n--- anteprima ---")
    print("".join(lines[max(0, i_anchor - 3):i_anchor + 8]))
    sys.exit(0)

P.write_text("".join(lines))
print("\nfile scritto. Verificare con:")
print("  python3 -m py_compile scripts/run_experiments.py")
print("  grep -n '_record_dp_fields' scripts/run_experiments.py")
