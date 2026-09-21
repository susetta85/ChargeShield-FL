#!/usr/bin/env python3
"""
Vulnerabilita' per record: conteggio osservato CONTRO livello di caso.

PERCHE' SERVE. analyze_worst_case_vulnerability.py conta quante sessioni reali
finiscono nel decile alto del punteggio d'attacco in modo CONSISTENTE su piu'
seed indipendenti. Ma un conteggio da solo non dice nulla: anche sotto ipotesi
nulla qualche sessione finisce in alto piu' volte per caso. Senza il livello di
caso, "411 sessioni vulnerabili" non e' un risultato, e' un numero.

COME SI OTTIENE IL LIVELLO DI CASO. Si permutano i percentili DENTRO ogni seed.
Questo conserva esattamente la distribuzione marginale dei punteggi di quel
seed e distrugge solo la CORRISPONDENZA fra seed, che e' precisamente cio' che
il criterio di flagging misura. E' la permutazione giusta: qualunque altra
cambierebbe anche la scala dei punteggi.

CRITERIO DI FLAGGING (identico a analyze_worst_case_vulnerability.py):
    la sessione e' membro in almeno `min_seeds` seed
    E percentile medio >= soglia (default 90)
    E percentile minimo >= pavimento (default 75)

LETTURA. z alto = esiste un sottoinsieme di record sistematicamente esposto,
invisibile nell'AUC aggregata. z ~ 0 = il conteggio e' quello che il caso
produce, nessuna vulnerabilita' per record rilevabile con questo criterio.

ATTENZIONE, confondente da dichiarare sempre: un z basso sotto DP non prova che
la DP abbia protetto i record se in quella stessa cella il modello e' stato
distrutto dal rumore. Un modello che non impara non memorizza, e quindi non
espone: e' il caso delle celle eps <= 1 di questo progetto, dove la loss e'
oltre 100 volte quella senza DP. Confrontare solo celle con utility residua
comparabile.

Uso:
    python3 scripts/worst_case_livello_di_caso.py \
        --gruppo no-DP=experiments/nodp-sweep2 \
        --gruppo dp-fedavg-eps1=experiments/dp-sweep4 \
        --permutazioni 200 --output risultati/worst_case/livello_di_caso.json
"""
from __future__ import annotations
import argparse
import bisect
import collections
import glob
import json
import os
import random
import statistics as st


def percentili_per_seed(cartella: str) -> list[dict[str, float]]:
    """Percentile del punteggio composto, per seed, sui soli membri."""
    per_seed = []
    for f in sorted(glob.glob(os.path.join(cartella, "per_sample_seed*.json"))):
        d = json.load(open(f))
        rec = [r for r in d.get("records", [])
               if r.get("session_id") and r.get("is_member")]
        if not rec:
            continue
        ordinati = sorted(r["composed_score"] for r in rec)
        n = max(len(ordinati) - 1, 1)
        per_seed.append({
            r["session_id"]: 100.0 * bisect.bisect_left(ordinati, r["composed_score"]) / n
            for r in rec
        })
    return per_seed


def conta_flagged(per_seed, soglia=90.0, pavimento=75.0, min_seed=2):
    agg = collections.defaultdict(list)
    for m in per_seed:
        for k, v in m.items():
            agg[k].append(v)
    flagged = sum(1 for v in agg.values()
                  if len(v) >= min_seed and st.mean(v) >= soglia and min(v) >= pavimento)
    multi = sum(1 for v in agg.values() if len(v) >= min_seed)
    return flagged, multi


def analizza(cartella, n_perm, soglia, pavimento, min_seed, seed_rng=42):
    ps = percentili_per_seed(cartella)
    if not ps:
        return None
    osservati, multi = conta_flagged(ps, soglia, pavimento, min_seed)
    rng = random.Random(seed_rng)
    nulli = []
    for _ in range(n_perm):
        perm = []
        for m in ps:
            chiavi = list(m.keys())
            valori = list(m.values())
            rng.shuffle(valori)
            perm.append(dict(zip(chiavi, valori)))
        nulli.append(conta_flagged(perm, soglia, pavimento, min_seed)[0])
    mu = st.mean(nulli)
    sd = st.stdev(nulli) if len(set(nulli)) > 1 else 0.0
    return {
        "cartella": cartella, "n_seed": len(ps),
        "sessioni_multi_seed": multi,
        "osservati": osservati,
        "attesi_per_caso": round(mu, 2),
        "sd_nulla": round(sd, 2),
        "z": round((osservati - mu) / sd, 3) if sd else None,
        "eccesso_assoluto": round(osservati - mu, 1),
        "percentuale_osservata": round(100 * osservati / multi, 3) if multi else None,
        "percentuale_attesa": round(100 * mu / multi, 3) if multi else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gruppo", action="append", required=True,
                    help="etichetta=cartella (ripetibile)")
    ap.add_argument("--permutazioni", type=int, default=200)
    ap.add_argument("--soglia", type=float, default=90.0)
    ap.add_argument("--pavimento", type=float, default=75.0)
    ap.add_argument("--min-seed", type=int, default=2)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    risultati = {}
    print(f"{'gruppo':24s} {'oss':>6s} {'caso':>9s} {'sd':>6s} {'z':>8s} "
          f"{'% oss':>8s} {'% attesa':>9s}")
    for g in a.gruppo:
        et, _, cart = g.partition("=")
        r = analizza(cart, a.permutazioni, a.soglia, a.pavimento, a.min_seed)
        if r is None:
            print(f"{et:24s} nessun dump per-campione in {cart}")
            continue
        risultati[et] = r
        print(f"{et:24s} {r['osservati']:6d} {r['attesi_per_caso']:9.1f} "
              f"{r['sd_nulla']:6.1f} {str(r['z']):>8s} "
              f"{r['percentuale_osservata']:8.2f} {r['percentuale_attesa']:9.2f}")

    if a.output:
        os.makedirs(os.path.dirname(a.output), exist_ok=True)
        json.dump({"parametri": {"permutazioni": a.permutazioni, "soglia": a.soglia,
                                 "pavimento": a.pavimento, "min_seed": a.min_seed},
                   "gruppi": risultati}, open(a.output, "w"), indent=2)
        print(f"\nscritto {a.output}")


if __name__ == "__main__":
    main()
