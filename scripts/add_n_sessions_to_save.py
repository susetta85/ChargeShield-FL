#!/usr/bin/env python3
"""
Sprint 10zz+118 — passa n_train_sessions a save_results(), cosi'
epsilon_record_dp viene calcolato in linea invece di restare None.

Oggi save_results() non riceve la lista delle sessioni, quindi
_record_dp_fields() non puo' chiamare l'accountant e ogni cella record-DP
salva epsilon_record_dp: None con una nota che invita a calcolarlo a mano.

Aggiunge un parametro keyword opzionale con default None, quindi qualunque
chiamata esistente continua a funzionare invariata.

Uso:  python3 add_n_sessions_to_save.py --dry-run
      python3 add_n_sessions_to_save.py
"""
import re, sys
from pathlib import Path

P = Path("scripts/run_experiments.py")
DRY = "--dry-run" in sys.argv
lines = P.read_text().splitlines(keepends=True)


def block_end(i_open):
    """indice della riga che chiude la parentesi aperta a i_open"""
    depth = 0
    for j in range(i_open, len(lines)):
        depth += lines[j].count("(") - lines[j].count(")")
        if depth == 0:
            return j
    sys.exit(f"ERRORE: parentesi non chiusa da riga {i_open+1}")


# ---- 1. firma di save_results -------------------------------------------
i_def = next(i for i, l in enumerate(lines) if l.startswith("def save_results("))
i_end = block_end(i_def)
sig = "".join(lines[i_def:i_end + 1])
print(f"firma: righe {i_def+1}-{i_end+1}")

if "n_train_sessions" in sig:
    print("  parametro gia' presente, salto")
else:
    # inserisce prima della riga che chiude, con l'indentazione dei parametri
    i_last_param = i_end - 1
    ind = lines[i_last_param][:len(lines[i_last_param]) - len(lines[i_last_param].lstrip())]
    lines.insert(i_end,
        f"{ind}# Sprint 10zz+118: serve a _record_dp_fields() per chiamare\n"
        f"{ind}# l'accountant RDP. Opzionale: senza, epsilon_record_dp resta\n"
        f"{ind}# None e la nota nel JSON lo dichiara.\n"
        f"{ind}n_train_sessions: int | None = None,\n")
    print(f"  parametro aggiunto prima della riga {i_end+1}")

# ---- 2. la chiamata a _record_dp_fields ---------------------------------
i_call = next(i for i, l in enumerate(lines) if "**_record_dp_fields(" in l)
if "n_train_sessions" in lines[i_call]:
    print("chiamata gia' aggiornata, salto")
else:
    lines[i_call] = lines[i_call].replace(
        "**_record_dp_fields(cfg)", "**_record_dp_fields(cfg, n_train_sessions)")
    print(f"chiamata aggiornata alla riga {i_call+1}: {lines[i_call].strip()}")

# ---- 3. il call site in main() ------------------------------------------
i_use = [i for i, l in enumerate(lines)
         if re.search(r"(?<!def )\bsave_results\(", l) and not l.startswith("def ")]
if len(i_use) != 1:
    sys.exit(f"ERRORE: attesa 1 chiamata a save_results(), trovate {len(i_use)}: "
             f"{[i+1 for i in i_use]}")
i_use = i_use[0]
i_use_end = block_end(i_use)
call_src = "".join(lines[i_use:i_use_end + 1])
print(f"\nchiamata in main: righe {i_use+1}-{i_use_end+1}")
print("   " + call_src.strip().replace("\n", "\n   "))

if "n_train_sessions" in call_src:
    print("  gia' aggiornata, salto")
else:
    var = next((v for v in ("train_sessions", "sessions", "all_sessions")
                if re.search(rf"\b{v}\b\s*=", "".join(lines[:i_use]))), None)
    if var is None:
        print("\n  ATTENZIONE: non trovo la variabile con le sessioni di training.")
        print("  Aggiungi a mano 'n_train_sessions=len(<tua_var>),' alla chiamata.")
    else:
        ind = lines[i_use_end][:len(lines[i_use_end]) - len(lines[i_use_end].lstrip())] + "    "
        lines.insert(i_use_end, f"{ind}n_train_sessions=len({var}),\n")
        print(f"  aggiunto 'n_train_sessions=len({var}),' prima della riga {i_use_end+1}")

if DRY:
    print("\n--- firma ---");  print("".join(lines[i_def:block_end(i_def)+1]))
    i2 = next(i for i, l in enumerate(lines) if re.search(r"(?<!def )\bsave_results\(", l) and not l.startswith("def "))
    print("--- chiamata ---"); print("".join(lines[i2:block_end(i2)+1]))
    sys.exit(0)

P.write_text("".join(lines))
print("\nfile scritto.")
