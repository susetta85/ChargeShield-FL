#!/usr/bin/env python3
"""
Ricalcola il budget della DP a livello di record dai JSON risultato.

Correzione della segnalazione 4 (2026-09-24). Usa src/ml/record_dp_accounting.py,
lo stesso codice che run_experiments.py usa al salvataggio, e sostituisce per i
JSON gia' prodotti scripts/_applicati/fix_json_recorddp_fields.py, che aveva lo
stesso difetto (round e delta letti dal posto sbagliato, n globale).

Solo lettura: stampa i valori e, con --output, li scrive in un file a parte. Non
modifica i JSON in experiments/.

Le run salvate prima della correzione non registrano le sessioni per client:
vanno passate con --n, leggendole dal log della run (righe
"[<client>-01] Round 1 — loss=..., n=<N>"). Esempio, run canary record-DP:

    python3 scripts/ricalcola_epsilon_record_dp.py \\
        experiments/_final_rdp_base_s42/experiment_*.json --n office1=1924
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE / "src"))

from ml.record_dp_accounting import record_dp_fields  # noqa: E402

CAMPI = (
    "epsilon_record_dp", "epsilon_record_dp_per_client",
    "epsilon_record_dp_shuffle_bound", "n_train_per_client",
    "record_dp_accounting", "record_dp_accounting_note",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("json", nargs="+", type=Path, help="JSON risultato di run_experiments.py")
    ap.add_argument("--n", action="append", default=[], metavar="CLIENT=N",
                    help="sessioni di training di un client (ripetibile); "
                         "sostituisce n_train_per_client del JSON")
    ap.add_argument("--output", type=Path, help="file JSON in cui scrivere i risultati")
    args = ap.parse_args()

    override: dict[str, int] = {}
    for item in args.n:
        client, sep, valore = item.partition("=")
        if not sep:
            ap.error(f"--n vuole CLIENT=N, ricevuto {item!r}")
        override[client] = int(valore)

    risultati = []
    for f in args.json:
        cfg = json.loads(f.read_text(encoding="utf-8")).get("config") or {}
        n_pc = override or cfg.get("n_train_per_client")
        campi = record_dp_fields(cfg, n_sessions=cfg.get("n_train_sessions"),
                                 n_per_client=n_pc)
        riga = {"file": str(f), "epsilon_record_dp_salvato": cfg.get("epsilon_record_dp")}
        riga.update({k: campi[k] for k in CAMPI})
        risultati.append(riga)
        print(f"{f}\n  salvato: {riga['epsilon_record_dp_salvato']}\n"
              f"  {campi['record_dp_accounting_note']}")

    if args.output:
        args.output.write_text(json.dumps(risultati, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"scritto {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
