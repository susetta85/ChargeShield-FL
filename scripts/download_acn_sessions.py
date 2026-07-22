#!/usr/bin/env python3
# scripts/download_acn_sessions.py
# ChargeShield-FL — download paginato da ev.caltech.edu (2026-07-22)
#
# Perché serve: l'interfaccia web di ev.caltech.edu ("Web Interface" nella
# pagina https://ev.caltech.edu/dataset) genera il JSON completo lato server
# prima di farlo scaricare — per range con molte sessioni (es. Caltech 2018,
# JPL 2019) il server sembra interrompere l'export prima di finire, lasciando
# un file JSON troncato a metà di un record (mancano le parentesi di chiusura
# di "_items" e "_meta"). Verificato: entrambi i file troncati forniti
# finiscono esattamente così, non in un punto casuale.
#
# Questo script usa invece la REST API con paginazione esplicita (25 sessioni
# a richiesta, segue i link "_links.next" finché l'API non li omette più —
# vedi la sezione "HATEOAS" in https://ev.caltech.edu/dataset) e scrive il
# file solo alla fine, quando tutte le pagine sono state raccolte — nessun
# limite di tempo lato server sull'intero export, solo sulla singola pagina
# da 25 sessioni.
#
# Uso:
#   export ACN_TOKEN="il-tuo-token-da-ev.caltech.edu"
#   python scripts/download_acn_sessions.py --site caltech --year 2018 \
#       --out datasets/acn/caltech/acndata_sessions_2018.json
#   python scripts/download_acn_sessions.py --site jpl --year 2019 \
#       --out datasets/acn/jpl/acndata_sessions_2019.json
#
# Il token va SOLO in una variabile d'ambiente, mai passato come argomento da
# riga di comando (finirebbe nella cronologia della shell/nei log di processo).

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://ev.caltech.edu/api/v1/sessions"


def download_site_year(site: str, year: int, token: str, max_retries: int = 3) -> list[dict]:
    """
    Scarica TUTTE le sessioni di un sito per un anno, seguendo la paginazione
    dell'API (25 sessioni/pagina) invece di affidarsi a un singolo export
    bulk. Ritorna la lista di record grezzi ("_items"), stesso formato che
    ACNDataset.load() si aspetta di trovare sotto la chiave "_items".
    """
    where = (
        f'connectionTime>="Mon, 1 Jan {year} 00:00:00 GMT" and '
        f'connectionTime<="Wed, 31 Dec {year} 23:59:59 GMT"'
    )
    url = f"{API_BASE}/{site}?where={where}&sort=connectionTime"
    items: list[dict] = []
    page = 1

    while url:
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, auth=(token, ""), timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  Pagina {page}: errore ({exc}) — retry tra {wait}s...", file=sys.stderr)
                time.sleep(wait)

        data = resp.json()
        page_items = data.get("_items", [])
        items.extend(page_items)
        print(f"  Pagina {page}: +{len(page_items)} sessioni (totale finora: {len(items)})")

        next_link = (data.get("_links") or {}).get("next")
        url = f"https://ev.caltech.edu/api/v1/{next_link['href']}" if next_link else None
        page += 1
        time.sleep(0.2)  # non bombardare l'API — nessun rate limit documentato, ma prudenza

    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download paginato ACN-Data (evita i timeout dell'export bulk web)."
    )
    parser.add_argument("--site", required=True, choices=["caltech", "jpl", "office001"])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    token = os.environ.get("ACN_TOKEN")
    if not token:
        print(
            "Errore: variabile d'ambiente ACN_TOKEN non impostata.\n"
            "Esegui: export ACN_TOKEN=\"il-tuo-token\"  (ottenuto da "
            "https://ev.caltech.edu/register o https://ev.caltech.edu/login)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Download {args.site} {args.year}...")
    items = download_site_year(args.site, args.year, token)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_meta": {"page": 1, "max_results": len(items), "total": len(items)}, "_items": items}
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"Completato: {len(items)} sessioni scritte in {args.out}")


if __name__ == "__main__":
    main()
