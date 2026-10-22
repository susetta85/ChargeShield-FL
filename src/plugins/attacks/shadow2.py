"""
Attacco shadow, versione 2 (2026-10-19): shadow federato e calibrazione globale.

Perche' una versione nuova: la versione 1 (`run_shadow()` in
scripts/run_experiments.py) addestra un autoencoder centralizzato, senza FL e
senza DP, in ogni cella, e usa un punteggio per record (loss dello shadow meno
loss del bersaglio) che non e' il metodo descritto nel paper. Decisione della
dottoranda, 19 ottobre: tutti gli attacchi e tutti i modelli sono federati, e
ogni attacco gira in tutte le configurazioni DP, nessuna DP inclusa; lo shadow
e' quello del paper (sez. Preliminaries): "a shadow model f_s is trained on a
disjoint split, and its loss distribution supplies the threshold applied to
l(x, f)". `run_shadow()` resta per riprodurre le run vecchie
(`shadow.version: 1`).

Metodo.

1. Divisione. Per ogni client, meta' dei membri non canary va allo shadow
   (`shadow.member_fraction`, default 0.5, come scritto nel threat model:
   "the shadow attack trains its single shadow on half of the members"), il
   resto e tutti i canary alla valutazione. I non membri (holdout), divisi per
   sito, vanno meta' alla calibrazione dello shadow (`shadow.calibration_
   fraction`) e meta' alla valutazione. I quattro insiemi sono disgiunti.
2. Federazione shadow. `run_fl_rounds()`, lo stesso codice del bersaglio:
   stessi client, stessi round, stesso FedProx, stessa DP (per client nelle
   modalita' local/central/dp-fedavg, per record, per soggetto, ablazioni),
   stessa inizializzazione comune del bersaglio (il server la distribuisce a
   tutti). Seme proprio (seed + 500009), quindi campionamenti e rumore
   indipendenti da quelli del bersaglio. Nessun attacco bizantino. Lo stato
   dei generatori casuali (random, numpy, torch) viene salvato prima e
   ripristinato dopo: la presenza dello shadow non cambia gli attacchi
   successivi (LiRA).
3. Calibrazione globale, a ogni round r. Loss del modello globale shadow del
   round r sui suoi membri (IN) e sui non membri di calibrazione (OUT). Due
   soglie, fissate senza guardare il bersaglio (membro se loss <= soglia):
   quella che massimizza TPR - FPR sullo shadow, e la piu' alta con FPR sullo
   shadow <= `shadow.target_fpr` (default 0.01, l'FPR operativo del paper).
4. Valutazione. Le due soglie applicate alla loss del modello globale
   bersaglio del round r su membri e non membri di valutazione (bilanciati,
   stesso sottoinsieme a ogni round): TPR, FPR e vantaggio realizzati.

L'AUC del punteggio -loss sull'insieme di valutazione (`shadow_auc_roc`) e'
quella di Yeom su un sottoinsieme: la calibrazione globale e' una soglia sulla
stessa loss, quindi non cambia l'ordinamento. Il contenuto nuovo dell'attacco
sono i tassi alla soglia scelta dallo shadow, confrontati con il vantaggio
ottimo che si avrebbe conoscendo le etichette (`shadow_oracle_advantage`).

Superficie: solo il modello globale (paper, sez. "Two observation surfaces").
Canary: esclusi dal training dello shadow, dalla calibrazione e dalle metriche
principali. Dal 2026-10-20 (controllo positivo dello shadow realistico) i canary
membri e non membri sono valutati a parte sul modello globale bersaglio, con le
stesse due soglie e lo stesso classificatore: `shadow_canary_*` e
`shadow_clf_canary_*`. Senza canary nulla cambia.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

logger = logging.getLogger(__name__)

_SEED_FED = 500_009
_SEED_SPLIT = 600_011
_SEED_BAL = 700_001


# ── Funzioni pure (testate in tests/test_shadow2.py) ───────────────────────────

def e_canary(s: dict[str, Any]) -> bool:
    """Sessione canary (membro o non membro), in qualunque protocollo."""
    return s.get("_canary_group") is not None or s.get("_canary_role") is not None


def dividi_membri(
    pools: dict[str, list[int]], sessions: list[dict[str, Any]], frazione: float, seed: int,
) -> tuple[dict[str, list[int]], list[int]]:
    """Per ogni client, la frazione dei membri non canary allo shadow; il resto e
    tutti i canary alla valutazione. Indici in `sessions`."""
    rng = random.Random(seed + _SEED_SPLIT)
    shadow: dict[str, list[int]] = {}
    valuta: list[int] = []
    for c in pools:
        liberi = [i for i in pools[c] if not e_canary(sessions[i])]
        canary = [i for i in pools[c] if e_canary(sessions[i])]
        rng.shuffle(liberi)
        n = int(round(frazione * len(liberi)))
        shadow[c] = sorted(liberi[:n])
        valuta.extend(sorted(liberi[n:]) + canary)
    return shadow, valuta


def dividi_non_membri(
    gruppi: dict[str, list[int]], sessions: list[dict[str, Any]], frazione: float, seed: int,
) -> tuple[list[int], list[int]]:
    """Non membri divisi per sito: la frazione dei non canary alla calibrazione
    dello shadow, il resto e tutti i canary alla valutazione."""
    rng = random.Random(seed + _SEED_SPLIT + 1)
    cal: list[int] = []
    valuta: list[int] = []
    for g in sorted(gruppi):
        liberi = [i for i in gruppi[g] if not e_canary(sessions[i])]
        canary = [i for i in gruppi[g] if e_canary(sessions[i])]
        rng.shuffle(liberi)
        n = int(round(frazione * len(liberi)))
        cal.extend(sorted(liberi[:n]))
        valuta.extend(sorted(liberi[n:]) + canary)
    return cal, valuta


def dividi_esterni(
    gruppi: dict[str, list[int]], sessions: list[dict[str, Any]], idx_cal: list[int],
    frazione: float, seed: int,
) -> tuple[dict[str, list[int]], list[int]]:
    """Versione con DATI PROPRI dell'avversario (shadow.fonte_membri: esterna).

    Pesca i membri dello shadow solo dalla calibrazione dell'holdout (`idx_cal`),
    mai dai client e mai dall'holdout di valutazione. Per sito, la frazione
    `frazione` dei non canary di `idx_cal` diventa membro dello shadow; il resto
    diventa la calibrazione dello shadow. Restituisce (membri per sito, calibrazione),
    indici in `sessions`: i due insiemi sono disgiunti tra loro e da tutto il training."""
    rng = random.Random(seed + _SEED_SPLIT + 2)
    cal_set = set(idx_cal)
    membri: dict[str, list[int]] = {}
    calib: list[int] = []
    for g in sorted(gruppi):
        liberi = [i for i in gruppi[g] if i in cal_set and not e_canary(sessions[i])]
        rng.shuffle(liberi)
        n = int(round(frazione * len(liberi)))
        membri[g] = sorted(liberi[:n])
        calib.extend(sorted(liberi[n:]))
    return membri, sorted(calib)


def riduci_membri(
    idx_shadow: dict[str, list[int]], dimensioni: dict[str, int], seed: int,
) -> dict[str, list[int]]:
    """Controllo a dimensione uguale (shadow.fonte_membri: bersaglio_ridotto): per ogni
    client un sottoinsieme casuale dei membri dello shadow, di taglia dimensioni[c]
    (al piu' tutti). Deterministico dato il seme."""
    rng = random.Random(seed + _SEED_SPLIT + 3)
    out: dict[str, list[int]] = {}
    for c in sorted(idx_shadow):
        n = min(int(dimensioni.get(c, 0)), len(idx_shadow[c]))
        out[c] = sorted(rng.sample(list(idx_shadow[c]), n))
    return out


def _tasso_sotto(ordinati: np.ndarray, soglie: np.ndarray) -> np.ndarray:
    """Quota dei valori <= soglia, per ogni soglia."""
    return np.searchsorted(ordinati, soglie, side="right") / len(ordinati)


def soglia_vantaggio(loss_in: np.ndarray, loss_out: np.ndarray) -> tuple[float, float]:
    """Soglia (membro se loss <= soglia) che massimizza TPR - FPR sullo shadow.
    Fra soglie a pari vantaggio, la piu' bassa. -inf (nessun membro previsto)
    se nessuna soglia ha vantaggio positivo. Restituisce (soglia, vantaggio)."""
    li = np.sort(np.asarray(loss_in, dtype=float))
    lo = np.sort(np.asarray(loss_out, dtype=float))
    if len(li) == 0 or len(lo) == 0:
        raise ValueError("soglia_vantaggio: insiemi IN o OUT vuoti")
    cand = np.concatenate([[-np.inf], np.unique(np.concatenate([li, lo]))])
    diff = _tasso_sotto(li, cand) - _tasso_sotto(lo, cand)
    j = int(np.argmax(diff))
    return float(cand[j]), float(diff[j])


def soglia_fpr(loss_out: np.ndarray, alpha: float) -> float:
    """Soglia piu' alta con FPR sullo shadow (quota dei non membri con loss <=
    soglia) non superiore ad alpha; -inf se nessuna."""
    lo = np.sort(np.asarray(loss_out, dtype=float))
    if len(lo) == 0:
        raise ValueError("soglia_fpr: insieme OUT vuoto")
    cand = np.unique(lo)
    ok = cand[_tasso_sotto(lo, cand) <= alpha + 1e-12]
    return float(ok[-1]) if len(ok) else -math.inf


def tassi(loss_membri: np.ndarray, loss_non: np.ndarray, soglia: float) -> dict[str, float]:
    """TPR, FPR, vantaggio e accuratezza bilanciata alla soglia data."""
    tpr = float(np.mean(np.asarray(loss_membri, dtype=float) <= soglia))
    fpr = float(np.mean(np.asarray(loss_non, dtype=float) <= soglia))
    return {"tpr": tpr, "fpr": fpr, "vantaggio": tpr - fpr, "accuratezza": (tpr + 1.0 - fpr) / 2.0}


def _num(x: float) -> float | None:
    """Per il JSON: niente infiniti."""
    return round(float(x), 9) if math.isfinite(x) else None


# ── Stato dei generatori casuali ──────────────────────────────────────────────

def _salva_rng() -> tuple[Any, Any, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _ripristina_rng(stato: tuple[Any, Any, torch.Tensor]) -> None:
    random.setstate(stato[0])
    np.random.set_state(stato[1])
    torch.set_rng_state(stato[2])


def _semina(s: int) -> None:
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


# ── Federazione shadow ───────────────────────────────────────────────────────

def federazione_shadow(
    cfg: dict,
    cluster_sessions: dict[str, list[dict[str, Any]]],
    seed_s: int,
    no_dp: bool,
    dp_mode: str,
    init_weights: list[Any] | None,
) -> dict[int, dict[str, Any]]:
    """Allena la federazione shadow con `run_fl_rounds()`, lo stesso codice del
    bersaglio: stessi client, round, FedProx e DP; seme `seed_s`; nessun
    attacco bizantino. Lo stato dei generatori casuali viene ripristinato
    all'uscita. Con gli stessi dati, lo stesso seme e la stessa inizializzazione
    del bersaglio, e senza estrazioni di rumore, riproduce il bersaglio
    (tests/test_shadow2.py)."""
    from run_experiments import run_fl_rounds

    cfg_s = copy.deepcopy(cfg)
    cfg_s["experiment"]["seed"] = int(seed_s)
    cfg_s["byzantine_attack"] = {"enabled": False}
    stato = _salva_rng()
    try:
        _semina(int(seed_s))
        return run_fl_rounds(cfg_s, [s for v in cluster_sessions.values() for s in v], no_dp=no_dp,
                             dp_mode=dp_mode, cluster_sessions=cluster_sessions,
                             init_weights=init_weights)
    finally:
        _ripristina_rng(stato)


# ── Attacco ──────────────────────────────────────────────────────────────────

def run_shadow_v2(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
    cluster_membership: dict[str, list[int]] | None = None,
    roc_curve_dump_path: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Shadow federato con calibrazione globale su ogni round di `fl_results`.
    Vedi il docstring del modulo per il metodo. Lo stato dei generatori
    casuali e' lo stesso all'ingresso e all'uscita: anche la costruzione dei
    modelli per la valutazione estrae pesi iniziali, e senza ripristino gli
    attacchi successivi (LiRA) cambierebbero con la presenza dello shadow."""
    stato = _salva_rng()
    try:
        return _run_shadow_v2(cfg, train_sessions, holdout_sessions, fl_results, no_dp, dp_mode,
                              cluster_membership, roc_curve_dump_path)
    finally:
        _ripristina_rng(stato)


def errore_per_feature(arch: dict, input_dim: int, weights: list, x: np.ndarray) -> np.ndarray:
    """Errore di ricostruzione al quadrato per feature, (n_campioni, d). Input del
    classificatore di tipo Shokri (shadow.classificatore: logistica)."""
    import torch
    from plugins.attacks.lira2 import _modello
    m = _modello(arch, input_dim, weights)
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32)
        return ((m(xt) - xt) ** 2).double().numpy()


def _run_shadow_v2(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    no_dp: bool,
    dp_mode: str,
    cluster_membership: dict[str, list[int]] | None,
    roc_curve_dump_path: str | None,
) -> dict[int, dict[str, Any]]:
    from run_experiments import (
        _autoencoder_arch_kwargs, _full_roc_curve, _mia_feature_names, _record_dp_on,
        _subject_dp_config, _write_diagnostic_dump, group_indices_by_site,
    )

    from plugins.attacks.clf_logistica import punteggio_logistica, regressione_logistica
    from plugins.attacks.lira2 import _metriche, mse_per_campione

    sh_cfg = cfg.get("shadow", {}) or {}
    superficie = sh_cfg.get("observation_surface", "global")
    if superficie != "global":
        raise ValueError(
            f"shadow v2: observation_surface {superficie!r} non supportata, solo 'global' "
            "(paper, sez. attacchi); la superficie client resta in shadow.version: 1")
    frazione = float(sh_cfg.get("member_fraction", 0.5))
    fraz_cal = float(sh_cfg.get("calibration_fraction", 0.5))
    alpha = float(sh_cfg.get("target_fpr", 0.01))
    clf = str(sh_cfg.get("classificatore", "soglia"))
    if clf not in ("soglia", "logistica"):
        raise ValueError(f"shadow.classificatore deve essere 'soglia' o 'logistica', trovato {clf!r}")
    clf_on = clf == "logistica"
    for nome, v in (("member_fraction", frazione), ("calibration_fraction", fraz_cal), ("target_fpr", alpha)):
        if not 0.0 < v < 1.0:
            raise ValueError(f"shadow.{nome} deve stare in (0, 1), trovato {v}")

    exp_cfg = cfg["experiment"]
    ml_cfg = cfg["ml"]
    seed = int(exp_cfg.get("seed", 42))
    feats = list(_mia_feature_names(cfg))
    arch = _autoencoder_arch_kwargs(cfg)
    input_dim = int(ml_cfg["input_dim"])
    rounds = sorted(r for r in fl_results
                    if isinstance(r, int) and r >= 1 and fl_results[r].get("global_weights") is not None)
    if not rounds:
        logger.warning("Shadow v2: nessun round con global_weights, attacco saltato")
        return {}

    # Partizione dei membri: la stessa del bersaglio (run_fl_rounds, run_lira_v2)
    if cluster_membership is not None:
        pools = {c: list(v) for c, v in cluster_membership.items()}
    else:
        cids = ["highway", "urban", "residential", "corporate"]
        size = max(1, len(train_sessions) // len(cids))
        pools = {c: list(range(i * size, len(train_sessions) if i == len(cids) - 1 else (i + 1) * size))
                 for i, c in enumerate(cids)}

    idx_shadow, idx_mem = dividi_membri(pools, train_sessions, frazione, seed)
    vuoti = [c for c, v in idx_shadow.items() if not v]
    if vuoti:
        logger.warning("Shadow v2: client senza membri per lo shadow %s, attacco saltato", vuoti)
        return {}
    idx_cal, idx_non = dividi_non_membri(group_indices_by_site(holdout_sessions), holdout_sessions,
                                         fraz_cal, seed)
    fonte = str(sh_cfg.get("fonte_membri", "bersaglio"))
    if fonte not in ("bersaglio", "esterna", "bersaglio_ridotto"):
        raise ValueError("shadow.fonte_membri deve essere 'bersaglio', 'esterna' o "
                         f"'bersaglio_ridotto', trovato {fonte!r}")
    if fonte != "bersaglio" and (cfg.get("partition") or {}).get("strategy", "per_site") != "per_site":
        raise ValueError(f"shadow.fonte_membri: {fonte} richiede partition.strategy: per_site: i dati "
                         "ausiliari del server sono divisi per sito")
    if fonte == "bersaglio":
        sessioni_shadow = {c: [train_sessions[i] for i in idx_shadow[c]] for c in pools}
    else:
        # Dati propri: membri dello shadow dall'holdout di calibrazione, per sito.
        # Valutazione (membri del bersaglio idx_mem, non membri idx_non) invariata.
        fraz_est = float(sh_cfg.get("fraz_esterna", 0.5))
        if not 0.0 < fraz_est < 1.0:
            raise ValueError(f"shadow.fraz_esterna deve stare in (0, 1), trovato {fraz_est}")
        membri_est, idx_cal = dividi_esterni(group_indices_by_site(holdout_sessions),
                                             holdout_sessions, idx_cal, fraz_est, seed)
        mancanti = [c for c in pools if not membri_est.get(c)]
        if mancanti:
            logger.warning("Shadow v2 (%s): nessuna sessione di calibrazione per i siti %s, attacco saltato",
                           fonte, mancanti)
            return {}
        if fonte == "esterna":
            idx_shadow = {c: membri_est[c] for c in pools}
            sessioni_shadow = {c: [holdout_sessions[i] for i in idx_shadow[c]] for c in pools}
        else:
            # Controllo a dimensione uguale: membri del bersaglio, tanti quanti i dati
            # propri per sito, stessa calibrazione e stessa valutazione della versione
            # esterna. Le due differiscono solo per la fonte dei membri dello shadow.
            idx_shadow = riduci_membri(idx_shadow, {c: len(membri_est[c]) for c in pools}, seed)
            sessioni_shadow = {c: [train_sessions[i] for i in idx_shadow[c]] for c in pools}
    if any(e_canary(s) for s in train_sessions + holdout_sessions):
        logger.info("Shadow v2: canary esclusi da shadow, calibrazione e valutazione principale; "
                    "metriche canary separate (shadow_canary_*)")

    def _valida(s: dict[str, Any]) -> bool:
        try:
            return all(np.isfinite(float(s[f])) for f in feats)
        except (KeyError, TypeError, ValueError):
            return False

    def _x(sess: list[dict[str, Any]]) -> np.ndarray:
        return np.array([[float(s[f]) for f in feats] for s in sess if _valida(s)], dtype=np.float32)

    x_in = _x([s for c in sessioni_shadow for s in sessioni_shadow[c]])
    x_out = _x([holdout_sessions[i] for i in idx_cal])
    # canary fuori dalle metriche principali (2026-10-20): dividi_membri e dividi_non_membri
    # li mettono nella valutazione; hanno metriche proprie (shadow_canary_*, sotto)
    x_mem_all = _x([train_sessions[i] for i in idx_mem if not e_canary(train_sessions[i])])
    x_non_all = _x([holdout_sessions[i] for i in idx_non if not e_canary(holdout_sessions[i])])
    # Controllo positivo (2026-10-20): i canary membri (copie nel training) e i canary non
    # membri (nell'holdout) sono valutati a parte, con le stesse soglie e lo stesso
    # classificatore dello shadow. Non entrano in shadow, calibrazione o metriche principali.
    x_can_mem = _x([train_sessions[i] for i in idx_mem if e_canary(train_sessions[i])])
    x_can_non = _x([holdout_sessions[i] for i in idx_non if e_canary(holdout_sessions[i])])
    canary_on = len(x_can_mem) > 0 and len(x_can_non) > 0
    if min(len(x_in), len(x_out), len(x_mem_all), len(x_non_all)) == 0:
        logger.warning("Shadow v2: un insieme e' vuoto dopo il filtro delle feature, attacco saltato")
        return {}
    # valutazione bilanciata come Yeom, stesso sottoinsieme a ogni round
    rng_b = random.Random(seed + _SEED_BAL)
    n_b = min(len(x_mem_all), len(x_non_all))
    x_mem = x_mem_all[sorted(rng_b.sample(range(len(x_mem_all)), n_b))]
    x_non = x_non_all[sorted(rng_b.sample(range(len(x_non_all)), n_b))]

    # ── Federazione shadow: stesso codice, stessa DP, seme proprio ───────────
    common_init = bool(ml_cfg.get("common_init", False))
    init = (fl_results.get(0) or {}).get("raw_global_weights") if common_init else None
    if common_init and init is None:
        raise RuntimeError("shadow v2: ml.common_init senza pesi iniziali in fl_results[0]")
    seed_s = seed + _SEED_FED
    shadow_cs = sessioni_shadow
    meccanismi = {
        "per_client": None if no_dp else {"dp_mode": dp_mode, "epsilon": exp_cfg.get("epsilon"),
                                          "dp_ablation": exp_cfg.get("dp_ablation", "full")},
        "per_record": (float((ml_cfg.get("record_dp") or {}).get("noise_multiplier", 0.0))
                       if _record_dp_on(ml_cfg) else None),
        "per_soggetto": (float(_subject_dp_config(ml_cfg)["noise_multiplier"])
                         if _subject_dp_config(ml_cfg)["enabled"] else None),
    }
    logger.info(
        "[SHADOW-FL] inizio federazione shadow: %s, seme %d, %d round, DP per client %s, "
        "DP per record %s, DP per soggetto %s",
        {c: len(v) for c, v in shadow_cs.items()}, seed_s, int(exp_cfg["fl_rounds"]),
        "no" if no_dp else (f"{dp_mode} (epsilon {exp_cfg.get('epsilon')}"
                            + (f", ablazione {exp_cfg.get('dp_ablation')}"
                               if exp_cfg.get("dp_ablation", "full") != "full" else "") + ")"),
        "no" if meccanismi["per_record"] is None else f"sigma {meccanismi['per_record']}",
        "no" if meccanismi["per_soggetto"] is None else f"sigma {meccanismi['per_soggetto']}")
    t0 = time.time()
    res_s = federazione_shadow(cfg, shadow_cs, seed_s, no_dp, dp_mode, init)
    secondi = time.time() - t0
    logger.info("[SHADOW-FL] fine federazione shadow, %.0f s", secondi)

    risultati: dict[int, dict[str, Any]] = {}
    curve: dict[int, Any] = {}
    for r in rounds:
        w_s = (res_s.get(r) or {}).get("global_weights")
        if w_s is None:
            logger.warning("Shadow v2: round %d senza modello shadow, saltato", r)
            continue
        l_in = mse_per_campione(arch, input_dim, w_s, x_in)
        l_out = mse_per_campione(arch, input_dim, w_s, x_out)
        w_t = fl_results[r]["global_weights"]
        l_mem = mse_per_campione(arch, input_dim, w_t, x_mem)
        l_non = mse_per_campione(arch, input_dim, w_t, x_non)

        tau_a, adv_s = soglia_vantaggio(l_in, l_out)
        tau_f = soglia_fpr(l_out, alpha)
        ta = tassi(l_mem, l_non, tau_a)
        tf = tassi(l_mem, l_non, tau_f)
        cal_f = tassi(l_in, l_out, tau_f)

        labels = [1] * len(l_mem) + [0] * len(l_non)
        scores = [-float(v) for v in l_mem] + [-float(v) for v in l_non]
        met = _metriche(labels, scores, "shadow")
        met["shadow_oracle_advantage"] = met.pop("shadow_advantage", None)
        risultati[r] = {
            **met,
            "shadow_version": 2,
            "shadow_cal_n_in": int(len(l_in)),
            "shadow_cal_n_out": int(len(l_out)),
            "shadow_cal_threshold_adv": _num(tau_a),
            "shadow_cal_advantage": round(adv_s, 6),
            "shadow_tpr_cal_adv": round(ta["tpr"], 6),
            "shadow_fpr_cal_adv": round(ta["fpr"], 6),
            "shadow_advantage_cal_adv": round(ta["vantaggio"], 6),
            "shadow_target_fpr": alpha,
            "shadow_cal_threshold_fpr": _num(tau_f),
            "shadow_cal_fpr": round(cal_f["fpr"], 6),
            "shadow_tpr_cal_fpr": round(tf["tpr"], 6),
            "shadow_fpr_cal_fpr": round(tf["fpr"], 6),
            "shadow_advantage_cal_fpr": round(tf["vantaggio"], 6),
            "shadow_cal_loss_in_mean": round(float(np.mean(l_in)), 9),
            "shadow_cal_loss_out_mean": round(float(np.mean(l_out)), 9),
            "shadow_loss_member_mean": round(float(np.mean(l_mem)), 9),
            "shadow_loss_nonmember_mean": round(float(np.mean(l_non)), 9),
        }
        if canary_on:
            lc_mem = mse_per_campione(arch, input_dim, w_t, x_can_mem)
            lc_non = mse_per_campione(arch, input_dim, w_t, x_can_non)
            ca, cf = tassi(lc_mem, lc_non, tau_a), tassi(lc_mem, lc_non, tau_f)
            lab_c = [1] * len(lc_mem) + [0] * len(lc_non)
            risultati[r].update({
                "shadow_canary_auc_roc": round(float(roc_auc_score(
                    lab_c, [-float(v) for v in np.r_[lc_mem, lc_non]])), 6),
                "shadow_canary_n_membri": int(len(lc_mem)), "shadow_canary_n_non_membri": int(len(lc_non)),
                "shadow_canary_tpr_cal_adv": round(ca["tpr"], 6), "shadow_canary_fpr_cal_adv": round(ca["fpr"], 6),
                "shadow_canary_tpr_cal_fpr": round(cf["tpr"], 6), "shadow_canary_fpr_cal_fpr": round(cf["fpr"], 6),
            })
        if clf_on:
            # Classificatore di tipo Shokri: addestrato sullo shadow (membri S contro
            # calibrazione), applicato al modello globale bersaglio. Decisione: membro
            # se logit >= 0 (probabilita' >= 1/2). Tassi realizzati e AUC del logit.
            f_in = errore_per_feature(arch, input_dim, w_s, x_in)
            f_out = errore_per_feature(arch, input_dim, w_s, x_out)
            modello_clf = regressione_logistica(f_in, f_out)
            z_mem = punteggio_logistica(modello_clf, errore_per_feature(arch, input_dim, w_t, x_mem))
            z_non = punteggio_logistica(modello_clf, errore_per_feature(arch, input_dim, w_t, x_non))
            ta_clf = tassi(-z_mem, -z_non, 0.0)
            lab_clf = [1] * len(z_mem) + [0] * len(z_non)
            sc_clf = [float(v) for v in np.r_[z_mem, z_non]]
            met_clf = _metriche(lab_clf, sc_clf, "shadow_clf")
            met_clf["shadow_clf_oracle_advantage"] = met_clf.pop("shadow_clf_advantage", None)
            risultati[r].update(met_clf)
            if canary_on:
                zc_mem = punteggio_logistica(modello_clf, errore_per_feature(arch, input_dim, w_t, x_can_mem))
                zc_non = punteggio_logistica(modello_clf, errore_per_feature(arch, input_dim, w_t, x_can_non))
                tc = tassi(-zc_mem, -zc_non, 0.0)
                risultati[r].update({
                    "shadow_clf_canary_auc_roc": round(float(roc_auc_score(
                        [1] * len(zc_mem) + [0] * len(zc_non), [float(v) for v in np.r_[zc_mem, zc_non]])), 6),
                    "shadow_clf_canary_tpr": round(tc["tpr"], 6), "shadow_clf_canary_fpr": round(tc["fpr"], 6),
                })
            risultati[r].update({
                "shadow_clf_tpr": round(ta_clf["tpr"], 6),
                "shadow_clf_fpr": round(ta_clf["fpr"], 6),
                "shadow_clf_advantage": round(ta_clf["vantaggio"], 6),
                "shadow_clf_pesi": [round(float(v), 6) for v in modello_clf["w"]],
                "shadow_clf_bias": round(float(modello_clf["b"]), 6),
            })
        if roc_curve_dump_path:
            curve[r] = _full_roc_curve(labels, scores)
        logger.info(
            "Shadow v2 round %d: AUC %.4f; soglia dello shadow a FPR %.3g -> TPR %.4f FPR %.4f; "
            "soglia di vantaggio -> TPR %.4f FPR %.4f (vantaggio %.4f, ottimo %.4f)",
            r, met.get("shadow_auc_roc") or float("nan"), alpha, tf["tpr"], tf["fpr"],
            ta["tpr"], ta["fpr"], ta["vantaggio"], met["shadow_oracle_advantage"] or float("nan"))

    if risultati:
        risultati[max(risultati)]["shadow_fl"] = {
            "seed": seed_s, "rounds": int(exp_cfg["fl_rounds"]), "common_init": common_init,
            "no_dp": bool(no_dp), "dp_mode": dp_mode, "dp_ablation": exp_cfg.get("dp_ablation", "full"),
            "meccanismi": meccanismi,
            "member_fraction": frazione, "calibration_fraction": fraz_cal, "fonte_membri": fonte,
            "membri_shadow_per_client": {c: len(v) for c, v in idx_shadow.items()},
            "secondi": round(secondi, 1),
        }
    if roc_curve_dump_path and curve:
        _write_diagnostic_dump(roc_curve_dump_path, {"attack": "shadow", "version": 2, "per_round": curve})
    return risultati
