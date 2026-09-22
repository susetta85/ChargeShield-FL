#!/usr/bin/env python3
"""
Ripara l'inserimento sbagliato di fix_json_recorddp_fields.py.

Bug: lo script calcolava lo spostamento dell'ancoraggio con
helper.count("\\n"), ma l'helper viene inserito nella lista come UN solo
elemento. L'indice andava spostato di 1. Il blocco e' quindi finito 72
righe piu' in basso, dentro un'altra espressione.

Questo script rimuove il blocco mal posizionato e lo reinserisce subito
dopo "epsilon_cumulative_best_known". L'helper _record_dp_fields resta
dov'e', che e' corretto.
"""
import re, sys
from pathlib import Path

P = Path("scripts/run_experiments.py")
DRY = "--dry-run" in sys.argv
lines = P.read_text().splitlines(keepends=True)

# ---- 1. rimuove il blocco mal posizionato -------------------------------
i_bad = [i for i, l in enumerate(lines) if "**_record_dp_fields(" in l]
if len(i_bad) != 1:
    sys.exit(f"ERRORE: attese 1 chiamata, trovate {len(i_bad)}")
i_bad = i_bad[0]

start = i_bad
while start > 0 and lines[start - 1].lstrip().startswith("# ") and \
      ("Sprint 10zz+117" in lines[start - 1] or "record-DP" in lines[start - 1]
       or "'epsilon'" in lines[start - 1] or "della cella" in lines[start - 1]):
    start -= 1
print(f"rimuovo righe {start+1}-{i_bad+1}:")
for l in lines[start:i_bad + 1]:
    print("   " + l.rstrip())
del lines[start:i_bad + 1]

# ---- 2. reinserisce dopo il vero ancoraggio -----------------------------
anchor = [i for i, l in enumerate(lines)
          if '"epsilon_cumulative_best_known"' in l and ":" in l
          and not l.lstrip().startswith("#")]
if len(anchor) != 1:
    sys.exit(f"ERRORE: attesa 1 riga di ancoraggio, trovate {len(anchor)}")
i = anchor[0]
ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
print(f"\nancoraggio reale: riga {i+1}: {lines[i].strip()}")

block = (
    f"{ind}# Sprint 10zz+117: record_dp/norm/epsilon_record_dp. Senza questi un\n"
    f"{ind}# JSON record-DP e' indistinguibile da uno no-DP, e il campo\n"
    f"{ind}# 'epsilon' qui sopra (client-level) viene letto come budget della\n"
    f"{ind}# cella, che e' falso: a sigma=1.0 il budget reale vale ~88, non 1.0.\n"
    f"{ind}**_record_dp_fields(cfg),\n"
)
lines.insert(i + 1, block)

if DRY:
    print("\n--- anteprima ---")
    print("".join(lines[max(0, i - 4):i + 7]))
    sys.exit(0)

P.write_text("".join(lines))
print("\nfile scritto.")
