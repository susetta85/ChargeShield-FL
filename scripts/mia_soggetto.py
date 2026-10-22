"""
MIA realistica a livello di soggetto (2026-10-20): persona (ACN-Data, chiave
`user_id`) o punto di ricarica (ChargePlace Scotland, chiave `node_id`).

Regola (decisione della dottoranda, 20 ottobre): l'avversario e' il server
onesto ma curioso e usa solo cio' che possiede: il modello globale rilasciato,
i propri dati ausiliari e le sessioni su cui fa la domanda. A differenza di
`scripts/mia_persona.py` (analisi di sensibilita', Segnalazioni voce 101) il
modello di riferimento NON vede sessioni dei client e non c'e'
re-identificazione nel pool delle sessioni dei client.

Due livelli (`--livello`):
  - entita: l'unita' e' il soggetto intero. I soggetti con almeno MIN sessioni
    sono divisi a caso con il seed in tre terzi: membri (tutte le sessioni nel
    training), non membri (fuori) e soggetti del server (fuori dal training,
    dati ausiliari del server). Il server conosce le prime N sessioni (in
    ordine di tempo) di ogni soggetto valutato, N in `--note` (default 1, 3,
    10); "tutte" (tutte le sessioni del soggetto) e' un limite superiore,
    riportato a parte. Metrica primaria: 10 sessioni note, punteggio calibrato.
  - gruppo: l'unita' e' un gruppo di K sessioni consecutive dello stesso
    soggetto nello stesso client (K = `--k-gruppo`, default 14, la mediana delle
    sessioni per utente in ACN-Data): la vista di gruppo della DP per record.
    I gruppi sono divisi a caso in tre terzi come sopra; le sessioni che non
    formano un gruppo intero e quelle senza chiave restano nel training. Un
    soggetto puo' avere gruppi in piu' terzi: si chiede se QUEL periodo di
    ricariche era nel training. Il server conosce tutte le K sessioni del gruppo
    interrogato; si valutano anche le prime 1 e 5. Metrica primaria: gruppo
    intero, punteggio calibrato.

Training e meccanismo: quelli del config della cella (`--config`, con
`--no-dp` o `--dp-mode/--epsilon` come nelle code), inizializzazione comune.
Training del bersaglio = sfondo + unita' membro, dove lo sfondo sono le sessioni
dei soggetti non valutabili e quelle senza chiave, tolta una riserva del 30%
delle sessioni senza chiave per i profili di abitudine della DP per soggetto
(mai nel training). Riferimento del server: stessa federazione, stesso
meccanismo e stessa inizializzazione del bersaglio (il server la distribuisce),
addestrata sulle sole unita' del server. Punteggi per unita': grezzo
(- loss media del bersaglio sulle sessioni note) e calibrato (media di loss del
riferimento - loss del bersaglio). AUC sulle unita' con intervallo bootstrap al
95% (1000 ricampionamenti delle unita'), p unilaterale di Mann-Whitney, Holm
sulle metriche secondarie della run.

Uso:
    python scripts/mia_soggetto.py --config config/experiment.yaml --no-dp --seed 42 \
        --livello entita --etichetta nodp --out esiti/mia_soggetto
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
from mia_persona import _allena, _auc_ci, _commit, _holm, _p_auc, _perdite, _semina  # noqa: E402

_SEED_DIVISIONE = 7177   # come mia_persona: stessa divisione dei soggetti a parita' di seed e chiave
_SEED_RISERVA = 4242
_SEED_RIFERIMENTO = 1


def chiave_di(s: dict[str, Any], chiave: str) -> str | None:
    v = s.get(chiave)
    return None if v in (None, "") else str(v)


def unita_entita(sessions: list[dict], chiave: str, min_sess: int) -> tuple[dict[str, list[int]], list[int]]:
    """Unita' = soggetti con almeno min_sess sessioni (indici in ordine di tempo);
    il resto e' sfondo."""
    per: dict[str, list[int]] = {}
    sfondo: list[int] = []
    for i, s in enumerate(sessions):
        k = chiave_di(s, chiave)
        if k is None:
            sfondo.append(i)
        else:
            per.setdefault(k, []).append(i)
    unita: dict[str, list[int]] = {}
    for k, idx in per.items():
        if len(idx) >= min_sess:
            unita[k] = sorted(idx, key=lambda i: str(sessions[i].get("start_time")))
        else:
            sfondo.extend(idx)
    return unita, sorted(sfondo)


def unita_gruppo(sessions: list[dict], chiave: str, k: int) -> tuple[dict[str, list[int]], list[int]]:
    """Unita' = gruppi di k sessioni consecutive (ordine di tempo) dello stesso
    soggetto nello stesso client; le sessioni avanzate e quelle senza chiave sono
    sfondo."""
    per: dict[tuple[str, str], list[int]] = {}
    sfondo: list[int] = []
    for i, s in enumerate(sessions):
        c = chiave_di(s, chiave)
        if c is None:
            sfondo.append(i)
        else:
            per.setdefault((c, str(s.get("site_id"))), []).append(i)
    unita: dict[str, list[int]] = {}
    for (c, sito), idx in sorted(per.items()):
        idx = sorted(idx, key=lambda i: str(sessions[i].get("start_time")))
        n = len(idx) // k
        for g in range(n):
            unita[f"{c}|{sito}|{g}"] = idx[g * k:(g + 1) * k]
        sfondo.extend(idx[n * k:])
    return unita, sorted(sfondo)


def dividi(unita: dict[str, list[int]], seed: int) -> tuple[list[str], list[str], list[str]]:
    """Tre terzi a caso con il seed: membri, non membri, unita' del server."""
    ids = sorted(unita)
    random.Random(seed + _SEED_DIVISIONE).shuffle(ids)
    t = len(ids) // 3
    return sorted(ids[:t]), sorted(ids[t:2 * t]), sorted(ids[2 * t:])


def insiemi(sessions: list[dict], chiave: str, livello: str, seed: int, min_sess: int, k: int):
    """Indici di training del bersaglio, del riferimento del server, della riserva
    dei profili, e le unita' valutate. Il riferimento contiene SOLO unita' del
    server: nessuna sessione del training del bersaglio, nessuna valutata."""
    if livello == "entita":
        unita, sfondo = unita_entita(sessions, chiave, min_sess)
    elif livello == "gruppo":
        unita, sfondo = unita_gruppo(sessions, chiave, k)
    else:
        raise ValueError(f"livello non valido: {livello!r}")
    if len(unita) < 30:
        raise ValueError(f"solo {len(unita)} unita' con la chiave {chiave!r}: chiave sbagliata per il dataset?")
    membri, non_membri, server = dividi(unita, seed)
    senza = [i for i in sfondo if chiave_di(sessions[i], chiave) is None]
    random.Random(seed + _SEED_RISERVA).shuffle(senza)
    riserva = set(senza[:int(round(0.3 * len(senza)))])
    train = [i for i in sfondo if i not in riserva] + [i for u in membri for i in unita[u]]
    rif = [i for u in server for i in unita[u]]
    return {"unita": unita, "membri": membri, "non_membri": non_membri, "server": server,
            "train": sorted(train), "riferimento": sorted(rif), "riserva": sorted(riserva)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dp", action="store_true")
    ap.add_argument("--dp-mode", default="dp-fedavg", choices=["dp-fedavg", "local", "central"])
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--livello", required=True, choices=["entita", "gruppo"])
    ap.add_argument("--chiave", default=None, help="default: subject_key del config, altrimenti user_id")
    ap.add_argument("--min-sessioni", type=int, default=10)
    ap.add_argument("--k-gruppo", type=int, default=14)
    ap.add_argument("--note", default=None, help="sessioni note per unita' (default 1,3,10 o 1,5,K)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--etichetta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(int(a.threads))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = rx.load_config(Path(a.config), {"rounds": a.rounds, "epochs": a.epochs})
    cfg["experiment"]["seed"] = a.seed
    if a.epsilon is not None:
        cfg["experiment"]["epsilon"] = float(a.epsilon)
    cfg["ml"]["common_init"] = True
    sdp = rx._subject_dp_config(cfg["ml"])
    chiave = a.chiave or (str(sdp["subject_key"]) if sdp["enabled"] else "user_id")
    if sdp["enabled"] and str(sdp["subject_key"]) != chiave:
        raise ValueError(f"chiave {chiave!r} diversa dalla chiave della DP per soggetto {sdp['subject_key']!r}")
    k = int(a.k_gruppo)
    note = ([int(x) for x in a.note.split(",")] if a.note else
            ([1, 3, 10] if a.livello == "entita" else [1, 5, k]))
    if a.livello == "gruppo" and max(note) > k:
        raise ValueError("nel livello gruppo le sessioni note non possono superare K")
    if a.livello == "entita" and max(note) > a.min_sessioni:
        raise ValueError("sessioni note piu' di --min-sessioni: alcune unita' non le avrebbero")
    primaria = "note10_calibrato" if a.livello == "entita" else f"note{k}_calibrato"
    if (10 if a.livello == "entita" else k) not in note:
        raise ValueError(f"--note deve contenere {10 if a.livello == 'entita' else k} (metrica primaria)")

    _semina(a.seed)
    sessions = rx.enrich_sessions(rx.load_sessions(cfg))
    ins = insiemi(sessions, chiave, a.livello, a.seed, a.min_sessioni, k)
    feats = rx._mia_feature_names(cfg)
    train = [sessions[i] for i in ins["train"]]
    stats, clip, info_norm = rx.statistiche_normalizzazione(cfg, train, feats)
    if not clip:
        raise ValueError("normalizzazione dal training: l'attaccante userebbe statistiche dei client; "
                         "usare i limiti pubblici")
    norm = rx.normalize_sessions(sessions, stats, feats, clip=clip)
    profili = None
    if sdp["enabled"] and sdp["senza_chiave"] == "profili" and ins["riserva"]:
        from ml.profili_abitudini import addestra, assegna
        pp = sdp.get("profili") or {}
        n_sito: dict[str, int] = {}
        for i in ins["train"]:
            sid = str(norm[i].get("site_id") or "")
            n_sito[sid] = n_sito.get(sid, 0) + 1
        mod = addestra([norm[i] for i in ins["riserva"]],
                       dimensione_media=float(pp.get("dimensione_media", 10.0)),
                       peso_gruppo=float(pp.get("peso_gruppo", 1.0)), seed=a.seed,
                       fonte="riserva dello sfondo senza chiave", n_training_per_sito=n_sito)
        assegna(mod, norm)
        profili = mod.descrizione()

    # bersaglio
    _semina(a.seed)
    t1 = time.time()
    cm = rx.group_indices_by_site([norm[i] for i in ins["train"]])
    cs = {cid: [norm[ins["train"][j]] for j in idx] for cid, idx in cm.items()}
    res_t = rx.run_fl_rounds(copy.deepcopy(cfg), [norm[i] for i in ins["train"]], no_dp=a.no_dp,
                             dp_mode=a.dp_mode, cluster_sessions=cs)
    ultimo = max(r for r in res_t if isinstance(r, int) and r > 0)
    w_t, w0 = res_t[ultimo]["global_weights"], res_t[0]["raw_global_weights"]
    t_train = time.time() - t1

    # riferimento del server: sole unita' del server, stesso meccanismo e stessa inizializzazione
    _semina(a.seed + _SEED_RIFERIMENTO)
    rif = [norm[i] for i in ins["riferimento"]]
    cm_r = rx.group_indices_by_site(rif)
    cs_r = {cid: [rif[j] for j in idx] for cid, idx in cm_r.items()}
    res_r = rx.run_fl_rounds(copy.deepcopy(cfg), rif, no_dp=a.no_dp, dp_mode=a.dp_mode,
                             cluster_sessions=cs_r, init_weights=w0)
    w_r = res_r[max(r for r in res_r if isinstance(r, int) and r > 0)]["global_weights"]

    valutate = ins["membri"] + ins["non_membri"]
    idx_val = sorted({i for u in valutate for i in ins["unita"][u]})
    pos = {i: j for j, i in enumerate(idx_val)}
    x_val = [norm[i] for i in idx_val]
    lt = _perdite(cfg, w_t, x_val, feats)
    lr = _perdite(cfg, w_r, x_val, feats)
    membri = set(ins["membri"])
    lab = [1 if u in membri else 0 for u in valutate]
    costo = float(np.mean([lt[pos[i]] for u in ins["non_membri"] for i in ins["unita"][u]]))

    ris: dict[str, Any] = {}
    pval: dict[str, float] = {}

    def _due(nome: str, gruppi: list[list[int]]) -> None:
        for versione in ("grezzo", "calibrato"):
            sc = [(-float(np.mean([lt[pos[i]] for i in g])) if versione == "grezzo"
                   else float(np.mean([lr[pos[i]] - lt[pos[i]] for i in g]))) for g in gruppi]
            v = _auc_ci(lab, sc)
            ris[f"{nome}_{versione}"] = v
            pval[f"{nome}_{versione}"] = _p_auc(lab, sc)

    for n in note:
        _due(f"note{n}", [ins["unita"][u][:n] for u in valutate])
    if a.livello == "entita":
        _due("tutte_limite_superiore", [ins["unita"][u] for u in valutate])
    if primaria not in pval:
        raise RuntimeError(f"metrica primaria {primaria} non calcolata (note = {note})")
    holm = _holm({x: p for x, p in pval.items() if x != primaria})
    holm[primaria] = pval[primaria]

    n_per_client = {cid: len(v) for cid, v in cs.items()}
    contab: dict[str, Any] = {}
    contab.update(rx._record_dp_fields(cfg, len(ins["train"]), n_per_client))
    if sdp["enabled"]:
        contab.update(rx._subject_dp_fields(cfg, rx._subject_stats(cs, str(sdp["subject_key"]), feats)))
    contab.update(rx._client_dp_fields(cfg, a.dp_mode, a.no_dp, n_per_client))
    esito = {
        "attacco": "mia_soggetto", "realistico": True, "livello": a.livello, "chiave": chiave,
        "etichetta": a.etichetta, "seed": a.seed, "config": a.config, "commit": _commit(),
        "no_dp": bool(a.no_dp), "dp_mode": a.dp_mode, "epsilon": cfg["experiment"].get("epsilon"),
        "rounds": cfg["experiment"]["fl_rounds"], "epochs": cfg["ml"]["epochs"],
        "min_sessioni": a.min_sessioni, "k_gruppo": k if a.livello == "gruppo" else None, "note": note,
        "unita_membri": len(ins["membri"]), "unita_non_membri": len(ins["non_membri"]),
        "unita_server": len(ins["server"]), "sessioni_train": len(ins["train"]),
        "sessioni_riferimento_server": len(ins["riferimento"]), "sessioni_riserva_profili": len(ins["riserva"]),
        "soggetti_valutati": len({u.split("|")[0] for u in valutate}),
        "metrica_primaria": primaria, "costo_loss_non_membri": costo,
        "nota_limite_superiore": ("tutte_limite_superiore_*: tutte le sessioni del soggetto, non realistico"
                                  if a.livello == "entita" else None),
        "auc": {x: {"auc": v[0], "ci95": [v[1], v[2]], "p": pval[x], "p_holm": holm.get(x)}
                for x, v in ris.items()},
        "riferimento": {"dati": "sole unita' del server (nessuna sessione dei client)",
                        "meccanismo": "come il bersaglio", "inizializzazione": "come il bersaglio"},
        "normalizzazione": info_norm, "profili": profili, "contabilita": contab,
        "secondi_training": t_train, "secondi_totali": time.time() - t0,
    }
    f = out / (f"mia_soggetto_{a.livello}_{Path(a.config).stem}_{a.etichetta}_s{a.seed}"
               f"_r{cfg['experiment']['fl_rounds']}_e{cfg['ml']['epochs']}.json")
    f.write_text(json.dumps(esito, indent=2))
    print(json.dumps({x: esito[x] for x in ("livello", "etichetta", "seed", "metrica_primaria")}, indent=1))
    print(json.dumps(esito["auc"], indent=1))


if __name__ == "__main__":
    main()
