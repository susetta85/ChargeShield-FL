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

LETTURA (corretta il 2026-09-24, vedi sotto). z alto sui membri = alcuni record
restano in alto in modo stabile fra seed. Questo e' un segnale di appartenenza
SOLO se il test appaiato e' positivo; se lo stesso eccesso compare anche fra i
non-membri, e' stabilita' del ranking dovuta al record, non alla sua
partecipazione al training.

CORREZIONE DEL 2026-09-24 (segnalazione 47). Il conteggio qui sopra misura la
STABILITA' del ranking di un record fra seed, non la sua appartenenza: un record
intrinsecamente facile da ricostruire finisce in alto in ogni seed che sia
membro o no, e la permutazione dentro il seed non lo sa. Verifica sui dump di
nodp-sweep2: stesso criterio applicato alle apparizioni da NON-membro, 406
segnalati contro 288.9 attesi, z = 6.85, contro i 412 (z = 7.56) dei membri.
Per questo lo script ora riporta, accanto al conteggio originale (tenuto per
tracciabilita' e come misura di stabilita' del ranking):
  - il CONTROLLO NON-MEMBRI: lo stesso criterio e la stessa permutazione sulle
    apparizioni da non-membro (percentile fra i soli non-membri del seed);
  - lo z dell'ECCESSO MEMBRI MENO NON-MEMBRI;
  - il TEST APPAIATO, che e' la lettura primaria: per ogni sessione che e'
    membro in almeno un seed e non-membro in almeno un altro, differenza fra il
    percentile medio (nel pool completo del seed) quando e' membro e quando non
    lo e'. Sotto ipotesi nulla di nessun effetto dell'appartenenza la media e'
    0; z = media / errore standard.

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


def percentili_per_seed(cartella: str, ruolo: str = "membri") -> list[dict[str, float]]:
    """Percentile del punteggio composto, per seed, dentro un ruolo: "membri"
    (comportamento storico) oppure "non_membri" (controllo, segnalazione 47)."""
    voglio_membri = ruolo == "membri"
    per_seed = []
    for f in sorted(glob.glob(os.path.join(cartella, "per_sample_seed*.json"))):
        d = json.load(open(f))
        rec = [r for r in d.get("records", [])
               if r.get("session_id") and bool(r.get("is_member")) == voglio_membri]
        if not rec:
            continue
        ordinati = sorted(r["composed_score"] for r in rec)
        n = max(len(ordinati) - 1, 1)
        per_seed.append({
            r["session_id"]: 100.0 * bisect.bisect_left(ordinati, r["composed_score"]) / n
            for r in rec
        })
    return per_seed


def percentili_pool(cartella: str) -> list[dict[str, tuple[float, bool]]]:
    """Per seed: {session_id: (percentile nel pool COMPLETO del seed, e' membro)}."""
    per_seed = []
    for f in sorted(glob.glob(os.path.join(cartella, "per_sample_seed*.json"))):
        d = json.load(open(f))
        rec = [r for r in d.get("records", []) if r.get("session_id")]
        if not rec:
            continue
        ordinati = sorted(r["composed_score"] for r in rec)
        n = max(len(ordinati) - 1, 1)
        per_seed.append({
            r["session_id"]: (100.0 * bisect.bisect_left(ordinati, r["composed_score"]) / n,
                              bool(r.get("is_member")))
            for r in rec
        })
    return per_seed


def test_appaiato(pool: list[dict[str, tuple[float, bool]]]) -> dict | None:
    """Stessa sessione quando e' membro contro quando non lo e' (segnalazione 47).

    Per ogni sessione presente come membro in almeno un seed e come non-membro in
    almeno un altro: media dei percentili da membro meno media da non-membro. La
    difficolta' intrinseca del record si cancella nella differenza. Sotto nulla
    la media e' 0; z = media / errore standard (le sessioni sono le unita')."""
    come_m, come_n = collections.defaultdict(list), collections.defaultdict(list)
    for seed in pool:
        for sid, (pc, membro) in seed.items():
            (come_m if membro else come_n)[sid].append(pc)
    diff = [st.mean(come_m[s]) - st.mean(come_n[s]) for s in come_m if s in come_n]
    if len(diff) < 2:
        return None
    media = st.mean(diff)
    se = st.stdev(diff) / len(diff) ** 0.5
    return {"n_sessioni": len(diff), "differenza_media": round(media, 4),
            "errore_standard": round(se, 4), "z": round(media / se, 3) if se else None}


def conta_flagged(per_seed, soglia=90.0, pavimento=75.0, min_seed=2):
    agg = collections.defaultdict(list)
    for m in per_seed:
        for k, v in m.items():
            agg[k].append(v)
    flagged = sum(1 for v in agg.values()
                  if len(v) >= min_seed and st.mean(v) >= soglia and min(v) >= pavimento)
    multi = sum(1 for v in agg.values() if len(v) >= min_seed)
    return flagged, multi


def _eccesso(ps, n_perm, soglia, pavimento, min_seed, seed_rng=42):
    """Conteggio osservato contro livello di caso per UN insieme di percentili."""
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
        "n_seed": len(ps),
        "sessioni_multi_seed": multi,
        "osservati": osservati,
        "attesi_per_caso": round(mu, 2),
        "sd_nulla": round(sd, 2),
        "z": round((osservati - mu) / sd, 3) if sd else None,
        "eccesso_assoluto": round(osservati - mu, 1),
        "percentuale_osservata": round(100 * osservati / multi, 3) if multi else None,
        "percentuale_attesa": round(100 * mu / multi, 3) if multi else None,
    }


def analizza(cartella, n_perm, soglia, pavimento, min_seed, seed_rng=42):
    ps = percentili_per_seed(cartella, "membri")
    if not ps:
        return None
    # Conteggio storico sui membri: stessi numeri di prima della correzione
    # (stesso RNG), ora letto come stabilita' del ranking (segnalazione 47).
    r = {"cartella": cartella,
         **_eccesso(ps, n_perm, soglia, pavimento, min_seed, seed_rng)}
    nm = percentili_per_seed(cartella, "non_membri")
    ctrl = _eccesso(nm, n_perm, soglia, pavimento, min_seed, seed_rng) if nm else None
    r["controllo_non_membri"] = ctrl
    r["z_eccesso_membri_meno_non_membri"] = None
    if ctrl and r["sd_nulla"] and ctrl["sd_nulla"]:
        diff = r["eccesso_assoluto"] - ctrl["eccesso_assoluto"]
        sd = (r["sd_nulla"] ** 2 + ctrl["sd_nulla"] ** 2) ** 0.5
        r["z_eccesso_membri_meno_non_membri"] = round(diff / sd, 3)
    r["appaiato"] = test_appaiato(percentili_pool(cartella))
    za = (r["appaiato"] or {}).get("z")
    r["lettura"] = (
        "segnale di appartenenza per record" if (za is not None and za > 3)
        else "nessun segnale di appartenenza per record (test appaiato)")
    return r


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
    print(f"{'gruppo':24s} {'oss':>6s} {'caso':>9s} {'z membri':>9s} "
          f"{'z non-m.':>9s} {'z diff':>7s} {'appaiato (diff ± es)':>22s} {'z app.':>7s}")
    for g in a.gruppo:
        et, _, cart = g.partition("=")
        r = analizza(cart, a.permutazioni, a.soglia, a.pavimento, a.min_seed)
        if r is None:
            print(f"{et:24s} nessun dump per-campione in {cart}")
            continue
        risultati[et] = r
        c = r.get("controllo_non_membri") or {}
        pa = r.get("appaiato") or {}
        app = (f"{pa['differenza_media']:+.3f} ± {pa['errore_standard']:.3f}"
               if pa else "n.d.")
        print(f"{et:24s} {r['osservati']:6d} {r['attesi_per_caso']:9.1f} "
              f"{str(r['z']):>9s} {str(c.get('z')):>9s} "
              f"{str(r['z_eccesso_membri_meno_non_membri']):>7s} {app:>22s} "
              f"{str(pa.get('z')):>7s}")

    if a.output:
        os.makedirs(os.path.dirname(a.output), exist_ok=True)
        json.dump({"parametri": {"permutazioni": a.permutazioni, "soglia": a.soglia,
                                 "pavimento": a.pavimento, "min_seed": a.min_seed},
                   "gruppi": risultati}, open(a.output, "w"), indent=2)
        print(f"\nscritto {a.output}")


if __name__ == "__main__":
    main()
