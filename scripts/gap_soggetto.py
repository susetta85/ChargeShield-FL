"""
Controllo di memorizzazione per l'attacco al soggetto (2026-10-22).

Domanda: se l'attacco al soggetto (`scripts/mia_soggetto.py`) da' AUC vicino a 0,5
anche nel limite superiore senza DP, e' perche' il modello globale non memorizza
le sessioni di training, o perche' c'e' un errore nella pipeline dell'attacco?

Lo script riproduce il bersaglio di `mia_soggetto.py` con gli stessi argomenti
(stessa divisione delle unita', stessa normalizzazione con limiti pubblici, stessi
semi, stesso training federato) e misura, round per round, sul modello globale:

  - loss media e mediana delle sessioni: unita' membro (nel training), unita' non
    membro (fuori), unita' del server (fuori), sfondo nel training;
  - divario di generalizzazione: loss dei non membri meno loss dei membri, in
    assoluto e relativo;
  - AUC per sessione (Yeom, punteggio -loss), sessioni dei membri contro sessioni
    dei non membri, con intervallo bootstrap sulle unita';
  - AUC per unita' con tutte le sessioni, punteggio grezzo: e' la stessa quantita'
    di `tutte_limite_superiore_grezzo` del JSON di `mia_soggetto.py`.

Il round 0 (inizializzazione comune, prima del training) e' il controllo di base:
la divisione in membri e non membri e' casuale, quindi li' l'AUC deve essere circa
0,5 e il divario circa 0. Se il divario cresce con i round il modello memorizza;
se resta nullo l'attacco non ha segnale da trovare, e il risultato nullo
dell'attacco e' spiegato dal modello, non da un errore.

Non addestra il riferimento del server (serve solo al punteggio calibrato).

Con `--confronta` legge il JSON di `mia_soggetto.py` della stessa cella e dello
stesso seed e riporta la differenza fra l'AUC per unita' ricalcolata e quella del
JSON. Sulla stessa macchina deve essere ~0; su una macchina diversa il training
non e' identico bit per bit, e una differenza di qualche centesimo e' attesa.

Uso (stessi argomenti della riga di coda, senza --etichetta):
    python scripts/gap_soggetto.py --config config/experiment_sc3.yaml --no-dp \
        --livello entita --chiave node_id --seed 42 --out esiti/gap_soggetto \
        --confronta esiti/mia_soggetto/mia_soggetto_entita_experiment_sc3_sc3nodp_s42_r10_e50.json
"""

from __future__ import annotations

import argparse
import copy
import json
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
from mia_persona import _auc_ci, _p_auc, _perdite, _semina  # noqa: E402
from mia_soggetto import insiemi  # noqa: E402


def _auc(lab, sc, gruppi=None) -> dict[str, Any]:
    """AUC con IC bootstrap. Con `gruppi` l'IC ricampiona le unita'; il p di
    Mann-Whitney tratta invece le sessioni come indipendenti e non e' valido
    quando sono raggruppate: in quel caso non si riporta, si legge l'IC."""
    v = _auc_ci(lab, sc, gruppi=gruppi)
    out: dict[str, Any] = {"auc": v[0], "ci95": [v[1], v[2]]}
    if gruppi is None:
        out["p"] = _p_auc(lab, sc)
    return out


def _riassunto(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "media": float(v.mean()), "mediana": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def misura(cfg, pesi, norm, ins, feats) -> dict[str, Any]:
    """Loss e AUC del modello con i pesi dati, sulle quattro popolazioni."""
    membri, non_membri, server = ins["membri"], ins["non_membri"], ins["server"]
    unita = ins["unita"]
    in_unita = {i for u in unita.values() for i in u}
    sfondo_train = [i for i in ins["train"] if i not in in_unita]

    idx_m = [i for u in membri for i in unita[u]]
    idx_n = [i for u in non_membri for i in unita[u]]
    idx_s = [i for u in server for i in unita[u]]
    tutti = sorted(set(idx_m) | set(idx_n) | set(idx_s) | set(sfondo_train))
    pos = {i: j for j, i in enumerate(tutti)}
    loss = _perdite(cfg, pesi, [norm[i] for i in tutti], feats)
    L = lambda idx: np.array([loss[pos[i]] for i in idx])  # noqa: E731

    lm, ln, ls, lb = L(idx_m), L(idx_n), L(idx_s), L(sfondo_train)

    # AUC per sessione, bootstrap sulle unita' (le sessioni della stessa unita' sono correlate)
    lab_s = [1] * len(idx_m) + [0] * len(idx_n)
    sc_s = list(-lm) + list(-ln)
    grp_s = [u for u in membri for _ in unita[u]] + [u for u in non_membri for _ in unita[u]]
    auc_sess = _auc(lab_s, sc_s, gruppi=grp_s)

    # AUC per unita', tutte le sessioni, punteggio grezzo (= tutte_limite_superiore_grezzo)
    valutate = membri + non_membri
    lab_u = [1] * len(membri) + [0] * len(non_membri)
    sc_u = [-float(np.mean(L(unita[u]))) for u in valutate]
    auc_unita = _auc(lab_u, sc_u)

    # divario per unita': distribuzione della loss media di unita' membro e non membro
    lu_m = np.array([float(np.mean(L(unita[u]))) for u in membri])
    lu_n = np.array([float(np.mean(L(unita[u]))) for u in non_membri])

    return {
        "loss_sessioni": {"membri": _riassunto(lm), "non_membri": _riassunto(ln),
                          "server": _riassunto(ls), "sfondo_train": _riassunto(lb)},
        "loss_unita": {"membri": _riassunto(lu_m), "non_membri": _riassunto(lu_n)},
        "divario_assoluto": float(ln.mean() - lm.mean()),
        "divario_relativo": float((ln.mean() - lm.mean()) / ln.mean()) if ln.mean() > 0 else None,
        "auc_sessione": auc_sess,
        "auc_unita_tutte_grezzo": auc_unita,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dp", action="store_true")
    ap.add_argument("--dp-mode", default="dp-fedavg", choices=["dp-fedavg", "local", "central"])
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--livello", default="entita", choices=["entita", "gruppo"])
    ap.add_argument("--chiave", default=None)
    ap.add_argument("--min-sessioni", type=int, default=10)
    ap.add_argument("--k-gruppo", type=int, default=14)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--solo-ultimo", action="store_true", help="misura solo round 0 e ultimo")
    ap.add_argument("--confronta", default=None, help="JSON di mia_soggetto.py della stessa cella")
    ap.add_argument("--riusa-pesi", action="store_true",
                    help="se esiste il file dei pesi di una run precedente, non riaddestra")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(int(a.threads))

    # --- identico a mia_soggetto.py fino al training del bersaglio ---
    cfg = rx.load_config(Path(a.config), {"rounds": a.rounds, "epochs": a.epochs})
    cfg["experiment"]["seed"] = a.seed
    if a.epsilon is not None:
        cfg["experiment"]["epsilon"] = float(a.epsilon)
    cfg["ml"]["common_init"] = True
    sdp = rx._subject_dp_config(cfg["ml"])
    chiave = a.chiave or (str(sdp["subject_key"]) if sdp["enabled"] else "user_id")

    _semina(a.seed)
    sessions = rx.enrich_sessions(rx.load_sessions(cfg))
    ins = insiemi(sessions, chiave, a.livello, a.seed, a.min_sessioni, int(a.k_gruppo))
    feats = rx._mia_feature_names(cfg)
    train = [sessions[i] for i in ins["train"]]
    stats, clip, info_norm = rx.statistiche_normalizzazione(cfg, train, feats)
    if not clip:
        raise ValueError("normalizzazione dal training: usare i limiti pubblici, come mia_soggetto.py")
    norm = rx.normalize_sessions(sessions, stats, feats, clip=clip)
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

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nome = (f"{a.livello}_{Path(a.config).stem}_{'nodp' if a.no_dp else a.dp_mode}_s{a.seed}"
            f"_r{cfg['experiment']['fl_rounds']}_e{cfg['ml']['epochs']}")
    f_pesi = out / f"pesi_{nome}.pt"
    t1 = time.time()
    if a.riusa_pesi and f_pesi.exists():
        res = torch.load(f_pesi, weights_only=False)
    else:
        _semina(a.seed)
        cm = rx.group_indices_by_site([norm[i] for i in ins["train"]])
        cs = {cid: [norm[ins["train"][j]] for j in idx] for cid, idx in cm.items()}
        res_full = rx.run_fl_rounds(copy.deepcopy(cfg), [norm[i] for i in ins["train"]], no_dp=a.no_dp,
                                    dp_mode=a.dp_mode, cluster_sessions=cs)
        res = {0: {"raw_global_weights": res_full[0]["raw_global_weights"]}}
        for r, v in res_full.items():
            if isinstance(r, int) and r > 0 and isinstance(v, dict) and v.get("global_weights") is not None:
                res[r] = {"global_weights": v["global_weights"]}
        torch.save(res, f_pesi)
    t_train = time.time() - t1
    # --- fine della parte identica ---

    rounds = sorted(r for r in res if isinstance(r, int) and r > 0)
    ultimo = rounds[-1]
    da_misurare = [ultimo] if a.solo_ultimo else rounds
    per_round: dict[str, Any] = {"0": misura(cfg, res[0]["raw_global_weights"], norm, ins, feats)}
    for r in da_misurare:
        w = res[r].get("global_weights")
        if w is not None:
            per_round[str(r)] = misura(cfg, w, norm, ins, feats)

    fin, base = per_round[str(ultimo)], per_round["0"]
    esito: dict[str, Any] = {
        "controllo": "gap_soggetto", "config": a.config, "seed": a.seed, "livello": a.livello,
        "chiave": chiave, "no_dp": a.no_dp, "dp_mode": a.dp_mode,
        "rounds": cfg["experiment"]["fl_rounds"], "epochs": cfg["ml"]["epochs"],
        "unita": {"membri": len(ins["membri"]), "non_membri": len(ins["non_membri"]),
                  "server": len(ins["server"])},
        "sessioni_train": len(ins["train"]),
        "quota_train_da_unita_membro": float(sum(len(ins["unita"][u]) for u in ins["membri"])
                                             / len(ins["train"])),
        "normalizzazione": info_norm.get("method") if isinstance(info_norm, dict) else None,
        "sintesi": {
            "divario_relativo_round0": base["divario_relativo"],
            "divario_relativo_ultimo": fin["divario_relativo"],
            "auc_sessione_round0": base["auc_sessione"]["auc"],
            "auc_sessione_ultimo": fin["auc_sessione"]["auc"],
            "auc_unita_tutte_grezzo_ultimo": fin["auc_unita_tutte_grezzo"]["auc"],
        },
        "per_round": per_round,
        "secondi_training": t_train, "secondi_totali": time.time() - t0,
    }
    if a.confronta:
        j = json.loads(Path(a.confronta).read_text())
        rif = j["auc"]["tutte_limite_superiore_grezzo"]["auc"]
        esito["confronto_mia_soggetto"] = {
            "json": a.confronta, "auc_json": rif,
            "auc_ricalcolata": fin["auc_unita_tutte_grezzo"]["auc"],
            "differenza": fin["auc_unita_tutte_grezzo"]["auc"] - rif,
            "unita_json": [j.get("unita_membri"), j.get("unita_non_membri"), j.get("unita_server")],
        }

    f = out / f"gap_soggetto_{nome}.json"
    f.write_text(json.dumps(esito, indent=2))
    print(json.dumps({"file": str(f), **esito["sintesi"],
                      **({"confronto": esito["confronto_mia_soggetto"]} if a.confronta else {})}, indent=1))


if __name__ == "__main__":
    main()
