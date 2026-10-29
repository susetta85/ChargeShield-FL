"""
LiRA e RMIA, versione 2 (2026-10-16): revisione profonda, Segnalazioni R,
voci 65-73.

Perche' una versione nuova invece di correggere `run_lira()`: la versione 1
(scripts/run_experiments.py) ha accumulato in tre mesi decine di correzioni e
opzioni diagnostiche intrecciate; i difetti trovati il 16 ottobre stanno nella
costruzione degli shadow e nella stima della varianza, cioe' al centro della
funzione. Qui c'e' solo il metodo, scritto da capo e coperto da test:
`run_lira()` resta per riprodurre le run vecchie (`lira.version: 1`).

Metodo (Carlini et al., "Membership Inference Attacks From First Principles",
IEEE S&P 2022; RMIA: Zarifzadeh et al., "Low-Cost High-Power Membership
Inference Attacks", ICML 2024), adattato all'osservatore federato:

1. Shadow. Per ogni client c e ogni k = 1..n_shadow uno shadow con un proprio
   `AutoencoderTrainer`, PERSISTENTE fra i round come il client reale: stesso
   codice di training (`train_local`), quindi stesso FedProx, stesso stato di
   Adam fra i round, stessa normalizzazione, stessa DP per record o per
   soggetto. Dal round 2 lo shadow riparte dal modello globale del round
   precedente (`set_weights`, come `apply_global_model` del client), al round
   1 da un'inizializzazione propria o, con `ml.common_init`, da quella comune.
   L'insieme IN ha la stessa taglia dei dati del client (`lira.in_size:
   client`), estratto dall'universo membri + non membri del client.
2. Osservazione. Lo shadow subisce la stessa trasformazione dell'update che il
   server osserva per il bersaglio: update grezzo senza DP per client e con
   dp-fedavg (LiRA attacca `raw_updates`), update tagliato e rumorizzato con
   local, solo tagliato con central. Per la superficie "global" lo shadow
   prende il posto del client c nell'aggregazione del round, insieme agli
   update osservati degli altri client, con lo stesso peso del client e, in
   central, con il rumore sull'aggregato.
3. Statistica. Log della MSE per campione (piu' vicina a una gaussiana della
   MSE). Medie IN e OUT per campione dagli shadow; varianza FISSA per client e
   round (media delle varianze per campione, Carlini et al. sez. VI-B), al
   posto delle soglie della versione 1, che valevano quasi sempre
   (floor_hit circa 0,997). Punteggio = log N(t; mu_in, s_in) - log N(t;
   mu_out, s_out). Si valutano solo i campioni con almeno uno shadow IN e uno
   OUT; il loro numero e quelli scartati vanno nel JSON.
4. RMIA online: Pr(x|theta) = exp(-mse/tau), tau = mediana della MSE del
   bersaglio sulla popolazione z del client e del round; Pr(x) = media sugli
   shadow; punteggio = quota della popolazione z con LR(x)/LR(z) > gamma. La
   popolazione z e' un sottoinsieme dei non membri del client riservato e mai
   valutato, scelto senza guardare le etichette. Stessi campioni di LiRA.
5. Composizione sui round: somma dei log-rapporti dello stesso campione sui
   round in cui e' valutato in tutti (LiRA e RMIA).

Fonte dei dati delle ombre (`lira.fonte_ombre`, 2026-10-20, decisione della
dottoranda: niente attacchi con dati che il server non possiede).
  - "universo" (default, comportamento delle campagne v2 e v2c): ombre su
    meta' circa di membri + non membri del client. Il server non possiede quei
    dati: i risultati valgono solo come analisi di sensibilita'.
  - "server": ombre solo su dati del server, cioe' i suoi dati ausiliari (meta'
    dell'holdout per gli attacchi di ogni sito, la stessa riserva dello shadow
    realistico) e le sessioni che interroga. Sugli stessi candidati: LiRA e RMIA
    online (ombre su meta' di dati ausiliari + candidati); LiRA e RMIA offline
    nella forma di Carlini et al. (sez. IV-C), con le sole ombre online che non
    hanno visto il campione ("out"); LiRA e RMIA offline con ombre allenate solo
    sui dati ausiliari, che non vedono nessun candidato ("off"). Popolazione z di
    RMIA dai dati ausiliari; RMIA offline con a = 1 (media sulle ombre OUT, senza
    l'approssimazione lineare di Zarifzadeh et al.). Vedi `insiemi_server` e
    `_run_lira_server`.
  Con `lira.observation_surface: partecipante` (solo con "server") l'avversario
  e' un partecipante che vede solo la sequenza dei modelli globali: a ogni round
  le ombre del client c ripartono dal globale osservato e prendono il posto di c
  nell'aggregazione, gli aggiornamenti degli altri client sono emulati con i dati
  ausiliari del server (`aggiornamenti_emulati`), con i pesi di aggregazione
  pubblici; il bersaglio osservato e' il modello globale.

Parallelismo: ogni traiettoria di shadow (client, k) e' indipendente e
deterministica dato il seed; con `workers > 1` gira in processi separati, un
thread ciascuno, e i numeri coincidono con l'esecuzione sequenziale.
"""

from __future__ import annotations

import logging
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

LOG_EPS = 1e-12
_SEED_CLUSTER = 104729
_SEED_SHADOW = 31337
_SEED_ROUND = 1_000_003


# ── Funzioni pure (testate in tests/test_lira2.py) ─────────────────────────────

def statistica(mse: np.ndarray) -> np.ndarray:
    """Statistica di LiRA: log della MSE per campione."""
    return np.log(np.asarray(mse, dtype=float) + LOG_EPS)


def stime_in_out(
    stat_shadow: np.ndarray, in_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Medie IN/OUT per campione e varianza fissa.

    stat_shadow, in_mask: forma (n_shadow, n_campioni).
    Restituisce mu_in, mu_out (nan dove mancano shadow), n_in, n_out per
    campione, e le due deviazioni standard fisse: radice della media, sui
    campioni con almeno due shadow di quel lato, della varianza campionaria
    (ddof = 1). Senza alcun campione con due shadow di un lato, la deviazione
    di quel lato e' quella dell'altro lato; senza nessuno dei due, nan.
    """
    s = np.asarray(stat_shadow, dtype=float)
    m = np.asarray(in_mask, dtype=bool)
    n_in = m.sum(axis=0)
    n_out = (~m).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_in = np.where(n_in > 0, np.where(m, s, 0.0).sum(axis=0) / np.maximum(n_in, 1), np.nan)
        mu_out = np.where(n_out > 0, np.where(~m, s, 0.0).sum(axis=0) / np.maximum(n_out, 1), np.nan)
        dev_in = np.where(m, s - mu_in, 0.0)
        dev_out = np.where(~m, s - mu_out, 0.0)
        var_in = (dev_in ** 2).sum(axis=0) / np.maximum(n_in - 1, 1)
        var_out = (dev_out ** 2).sum(axis=0) / np.maximum(n_out - 1, 1)
    vi = var_in[n_in >= 2]
    vo = var_out[n_out >= 2]
    s_in = float(np.sqrt(vi.mean())) if vi.size else float("nan")
    s_out = float(np.sqrt(vo.mean())) if vo.size else float("nan")
    if not np.isfinite(s_in):
        s_in = s_out
    if not np.isfinite(s_out):
        s_out = s_in
    return mu_in, mu_out, n_in, n_out, s_in, s_out


def _log_norm(x: np.ndarray, mu: np.ndarray, s: float) -> np.ndarray:
    return -0.5 * ((x - mu) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)


def punteggio_lira(
    t: np.ndarray, stat_shadow: np.ndarray, in_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Log-rapporto di verosimiglianza di LiRA online con varianza fissa.

    t: statistica del bersaglio per campione. Restituisce il punteggio (nan
    per i campioni senza almeno uno shadow IN e uno OUT) e la diagnostica.
    """
    mu_in, mu_out, n_in, n_out, s_in, s_out = stime_in_out(stat_shadow, in_mask)
    t = np.asarray(t, dtype=float)
    ok = (n_in >= 1) & (n_out >= 1) & np.isfinite(t)
    score = np.full(t.shape, np.nan)
    if ok.any() and np.isfinite(s_in) and np.isfinite(s_out) and s_in > 0 and s_out > 0:
        score[ok] = _log_norm(t[ok], mu_in[ok], s_in) - _log_norm(t[ok], mu_out[ok], s_out)
    diag = {
        "sigma_in": s_in, "sigma_out": s_out,
        "n_valutati": int(np.isfinite(score).sum()),
        "n_senza_in": int((n_in == 0).sum()), "n_senza_out": int((n_out == 0).sum()),
        "mu_in_meno_out": float(np.nanmean(mu_in[ok] - mu_out[ok])) if ok.any() else None,
    }
    return score, diag


def _logmeanexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def log_lr_rmia(mse_target: np.ndarray, mse_shadow: np.ndarray, tau: float) -> np.ndarray:
    """log Pr(x|theta) - log Pr(x), con Pr(x|theta) = exp(-mse/tau) e Pr(x)
    media sugli shadow (RMIA online). mse_shadow: (n_shadow, n)."""
    tau = max(float(tau), LOG_EPS)
    lp_t = -np.asarray(mse_target, dtype=float) / tau
    lp_ref = _logmeanexp(-np.asarray(mse_shadow, dtype=float) / tau, axis=0)
    return lp_t - lp_ref


def log_lr_rmia_offline(mse_target: np.ndarray, mse_shadow: np.ndarray, in_mask: np.ndarray,
                        tau: float) -> np.ndarray:
    """RMIA offline (Zarifzadeh et al.): Pr(x) e' la media sui soli shadow che NON hanno visto
    il record (OUT), non su tutti come nella forma online. in_mask: (n_shadow, n), True se lo
    shadow ha il record nel training. Dove non c'e' nessuno shadow OUT il risultato e' NaN."""
    tau = max(float(tau), LOG_EPS)
    lp_t = -np.asarray(mse_target, dtype=float) / tau
    lv = -np.asarray(mse_shadow, dtype=float) / tau
    fuori = ~np.asarray(in_mask, dtype=bool)
    n_out = fuori.sum(axis=0)
    lv_m = np.where(fuori, lv, -np.inf)
    mx = np.max(lv_m, axis=0)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lse = mx + np.log(np.sum(np.exp(lv_m - mx), axis=0))
        lp_ref = lse - np.log(n_out)
    lp_ref = np.where(n_out > 0, lp_ref, np.nan)
    return lp_t - lp_ref


def punteggio_rmia(llr_x: np.ndarray, llr_z: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Quota della popolazione z con LR(x)/LR(z) > gamma (Zarifzadeh et al.,
    eq. 4). llr_z: popolazione riservata, mai fra i campioni valutati."""
    z = np.sort(np.asarray(llr_z, dtype=float)[np.isfinite(llr_z)])
    x = np.asarray(llr_x, dtype=float)
    out = np.full(x.shape, np.nan)
    if z.size == 0:
        return out
    ok = np.isfinite(x)
    # LR(x)/LR(z) > gamma  <=>  llr_z < llr_x - log(gamma)
    out[ok] = np.searchsorted(z, x[ok] - math.log(gamma), side="left") / z.size
    return out


# ── Traiettoria di uno shadow (eseguibile in un processo separato) ─────────────

@dataclass
class SpecShadow:
    """Tutto cio' che serve a una traiettoria; serializzabile per i processi."""
    cid: str
    k: int
    seed: int
    rounds: list[int]
    in_sessions: list[dict[str, Any]]
    eval_x: np.ndarray                      # (n_eval, n_feature), gia' normalizzato
    ml_cfg: dict[str, Any]
    arch: dict[str, Any]
    input_dim: int
    gm_cfg: dict[str, Any]
    no_dp: bool
    dp_mode: str
    dp_ablation: str
    init_weights: list[Any] | None          # common_init: pesi iniziali comuni
    global_prev: dict[int, list[Any]]        # round r -> pesi globali del round r-1 (r >= 2)
    subject_weights: dict[str, float] | None
    superficie: str                          # "client" o "global"
    altri_update: dict[int, list[Any]] = field(default_factory=dict)   # solo "global"; None = posto dello shadow
    n_samples_client: dict[int, int] = field(default_factory=dict)     # solo "global"
    n_partecipanti: int = 1


def _modello(arch: dict[str, Any], input_dim: int, weights: list[Any]):
    from core.autoencoder import Autoencoder
    m = Autoencoder(input_dim=input_dim, **arch)
    sd = m.state_dict()
    keys = list(sd.keys())
    if len(keys) != len(weights):
        raise RuntimeError(f"pesi: {len(weights)} tensori, modello: {len(keys)}")
    m.load_state_dict({
        k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(sd[k].dtype).reshape(sd[k].shape)
        for k, w in zip(keys, weights)
    }, strict=True)
    m.eval()
    return m


def mse_per_campione(arch: dict[str, Any], input_dim: int, weights: list[Any],
                     x: np.ndarray) -> np.ndarray:
    """MSE di ricostruzione per campione, modello in eval. Stessa funzione per
    bersaglio e shadow."""
    m = _modello(arch, input_dim, weights)
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32)
        r = m(xt)
        return ((r - xt) ** 2).mean(dim=1).double().numpy()


def _seme_round(seed: int, r: int, extra: int = 0) -> int:
    return (seed + _SEED_ROUND * r + 7 * extra) % (2 ** 31 - 1)


def traiettoria_shadow(spec: SpecShadow) -> dict[int, np.ndarray]:
    """Allena uno shadow su tutti i round e restituisce la MSE per campione del
    modello osservato a ogni round (superficie scelta)."""
    from ml.autoencoder_trainer import AutoencoderTrainer
    from ml.base_ml import GradientUpdate
    from ml.fedavg_aggregator import FedAvgAggregator
    from ml.gradient_manager import GradientManager

    torch.set_num_threads(1)
    torch.manual_seed(spec.seed)
    tr = AutoencoderTrainer(
        config={**spec.ml_cfg, "seed": spec.seed},
        node_id=f"shadow-{spec.cid}-{spec.k}", cluster_id=spec.cid,
    )
    if spec.subject_weights is not None:
        tr.set_subject_weights(spec.subject_weights)
    if spec.init_weights is not None:
        tr.set_weights(spec.init_weights)
        tr._global_weights = None          # come run_fl_rounds con common_init
    keys = tr.get_weight_keys()
    gm = GradientManager(spec.gm_cfg)
    privato_server = (not spec.no_dp)
    out: dict[int, np.ndarray] = {}
    for r in spec.rounds:
        if r > 1:
            tr.set_weights(spec.global_prev[r])  # = apply_global_model del client
        pre = tr.get_weights()
        torch.manual_seed(_seme_round(spec.seed, r))
        upd = tr.train_local(spec.in_sessions, r)
        # update che entra nell'aggregazione (come run_fl_rounds)
        torch.manual_seed(_seme_round(spec.seed, r, 1))
        if not privato_server:
            aggregabile = upd
        elif spec.dp_mode == "central":
            aggregabile = gm.clip_only(upd, weight_keys=keys, reference_weights=pre)
        elif spec.dp_ablation == "clip-only":       # come _privatize_per_client
            aggregabile = gm.clip_no_noise(upd, weight_keys=keys, reference_weights=pre)
        elif spec.dp_ablation == "noise-only":
            aggregabile = gm.noise_no_clip(upd, weight_keys=keys)
        else:
            aggregabile = gm.privatize(upd, weight_keys=keys, reference_weights=pre)
        if spec.superficie == "client":
            # Il server osserva il grezzo senza DP e con dp-fedavg (LiRA attacca
            # raw_updates), il privatizzato con local, il tagliato con central.
            osservato = upd if (not privato_server or spec.dp_mode == "dp-fedavg") else aggregabile
            pesi = osservato.weights
        else:
            agg = FedAvgAggregator({"min_participants": spec.n_partecipanti})
            proprio = GradientUpdate(
                node_id=f"shadow-{spec.cid}-{spec.k}", cluster_id=spec.cid, round_num=r,
                weights=aggregabile.weights, gradients=None, loss=aggregabile.loss,
                n_samples=spec.n_samples_client.get(r, aggregabile.n_samples), metadata={},
            )
            # stesso ordine di raccolta del round reale: lo shadow al posto del client
            for u in spec.altri_update.get(r, []):
                agg.collect(proprio if u is None else u)
            aggregato = agg.aggregate(r)
            if aggregato is None:
                raise RuntimeError(f"shadow {spec.cid}/{spec.k} round {r}: aggregazione vuota")
            pesi = aggregato.global_weights
            if privato_server and spec.dp_mode == "central":
                torch.manual_seed(_seme_round(spec.seed, r, 2))
                pesi = gm.privatize_aggregate(
                    pesi, weight_keys=keys,
                    n_participants=aggregato.n_participants or spec.n_partecipanti,
                    participant_n_samples=list(
                        aggregato.metadata.get("participant_n_samples", {}).values()),
                )
        out[r] = mse_per_campione(spec.arch, spec.input_dim, pesi, spec.eval_x)
    return out


def aggiornamenti_emulati(spec: SpecShadow) -> dict[int, list[Any]]:
    """Avversario partecipante (2026-10-20): emula l'aggiornamento di un client con
    i soli dati del server (`spec.in_sessions`). A ogni round riparte dal modello
    globale osservato del round precedente, si allena con il codice e la DP del
    client e restituisce i pesi nella forma che il server aggrega (privatizzati
    con local e dp-fedavg, solo tagliati con central). Non usa nessun dato o
    aggiornamento reale dei client."""
    from ml.autoencoder_trainer import AutoencoderTrainer
    from ml.gradient_manager import GradientManager

    torch.set_num_threads(1)
    torch.manual_seed(spec.seed)
    tr = AutoencoderTrainer(config={**spec.ml_cfg, "seed": spec.seed},
                            node_id=f"emulato-{spec.cid}", cluster_id=spec.cid)
    if spec.subject_weights is not None:
        tr.set_subject_weights(spec.subject_weights)
    if spec.init_weights is not None:
        tr.set_weights(spec.init_weights)
        tr._global_weights = None
    keys = tr.get_weight_keys()
    gm = GradientManager(spec.gm_cfg)
    out: dict[int, list[Any]] = {}
    for r in spec.rounds:
        if r > 1:
            tr.set_weights(spec.global_prev[r])
        pre = tr.get_weights()
        torch.manual_seed(_seme_round(spec.seed, r))
        upd = tr.train_local(spec.in_sessions, r)
        torch.manual_seed(_seme_round(spec.seed, r, 1))
        if spec.no_dp:
            agg = upd
        elif spec.dp_mode == "central":
            agg = gm.clip_only(upd, weight_keys=keys, reference_weights=pre)
        elif spec.dp_ablation == "clip-only":
            agg = gm.clip_no_noise(upd, weight_keys=keys, reference_weights=pre)
        elif spec.dp_ablation == "noise-only":
            agg = gm.noise_no_clip(upd, weight_keys=keys)
        else:
            agg = gm.privatize(upd, weight_keys=keys, reference_weights=pre)
        out[r] = agg.weights
    return out


def esegui_emulazioni(specs: list[SpecShadow], workers: int = 1) -> list[dict[int, list[Any]]]:
    if workers <= 1 or len(specs) <= 1:
        stato = torch.get_num_threads()
        rng = torch.get_rng_state()
        try:
            return [aggiornamenti_emulati(s) for s in specs]
        finally:
            torch.set_num_threads(stato)
            torch.set_rng_state(rng)
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(aggiornamenti_emulati, specs))


def esegui_traiettorie(specs: list[SpecShadow], workers: int = 1) -> list[dict[int, np.ndarray]]:
    """Esegue le traiettorie, in sequenza o in processi separati. L'ordine dei
    risultati segue quello di `specs` in entrambi i casi."""
    if workers <= 1 or len(specs) <= 1:
        # Nel processo principale: si ripristinano thread e generatore globale,
        # cosi' gli attacchi successivi non dipendono da LiRA (voce 73).
        stato = torch.get_num_threads()
        rng = torch.get_rng_state()
        try:
            return [traiettoria_shadow(s) for s in specs]
        finally:
            torch.set_num_threads(stato)
            torch.set_rng_state(rng)
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(traiettoria_shadow, specs))


# ── Attacco completo ───────────────────────────────────────────────────────────

def _metriche(labels: list[int], scores: list[float], pref: str) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    from run_experiments import _mia_advantage, _tpr_at_fixed_fpr
    res: dict[str, Any] = {f"{pref}_n_member": int(sum(labels)),
                           f"{pref}_n_nonmember": int(len(labels) - sum(labels))}
    if len(set(labels)) < 2:
        res[f"{pref}_auc_roc"] = None
        return res
    res[f"{pref}_auc_roc"] = round(float(roc_auc_score(labels, scores)), 6)
    for k, v in _tpr_at_fixed_fpr(labels, scores).items():
        res[k if pref == "lira" else f"{pref}_{k}"] = v
    res[f"{pref}_advantage"] = _mia_advantage(labels, scores)
    return res


def assegna_non_membri(
    holdout_sessions: list[dict[str, Any]], cluster_ids: list[str],
    cfg: dict, cluster_sizes: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Client di origine dei non membri, coerente con la partizione dei membri
    (Segnalazioni, voce 72): per sito con `per_site`, a caso con le stesse
    proporzioni dei membri con `iid` o quando i siti non sono noti."""
    from run_experiments import _SITE_ID_TO_NAME
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    strategia = (cfg.get("partition") or {}).get("strategy", "per_site")
    out: dict[str, list[dict[str, Any]]] = {c: [] for c in cluster_ids}
    per_sito = strategia == "per_site"
    if per_sito:
        nomi = [_SITE_ID_TO_NAME.get(s.get("site_id", ""), s.get("site_id", "")) for s in holdout_sessions]
        per_sito = any(n in out for n in nomi)
    if per_sito:
        for s, n in zip(holdout_sessions, nomi):
            if n in out:
                out[n].append(s)
        return out
    rng = random.Random(seed + 271828)
    idx = list(range(len(holdout_sessions)))
    rng.shuffle(idx)
    tot = sum(cluster_sizes.values()) or 1
    pos = 0
    for j, c in enumerate(cluster_ids):
        n = (len(idx) - pos) if j == len(cluster_ids) - 1 else round(len(idx) * cluster_sizes[c] / tot)
        out[c] = [holdout_sessions[i] for i in idx[pos:pos + n]]
        pos += n
    return out


def run_lira_v2(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    n_shadow: int = 16,
    shadow_epochs_cap: int | None = None,
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
    cluster_membership: dict[str, list[int]] | None = None,
    composed_output: dict[str, Any] | None = None,
    per_sample_dump_path: str | None = None,
    workers: int = 1,
) -> dict[int, dict[str, Any]]:
    """LiRA e RMIA versione 2 su ogni round di `fl_results`. Vedi il docstring
    del modulo per il metodo."""
    import json
    from pathlib import Path

    from run_experiments import (
        _autoencoder_arch_kwargs, _mia_feature_names, _sample_preserving_canary_groups,
        _subject_cross_client_weights, _subject_dp_config,
    )

    lira_cfg = cfg.get("lira", {}) or {}
    if lira_cfg.get("shadow_init", "warm") != "warm":
        raise ValueError("lira v2: shadow_init diverso da 'warm' non e' supportato")
    if lira_cfg.get("canary_rescale"):
        raise ValueError("lira v2: canary_rescale non e' supportato (usare lira.version: 1)")
    superficie = lira_cfg.get("observation_surface", "client")
    if superficie not in ("client", "global", "partecipante"):
        raise ValueError(f"observation_surface non valido: {superficie!r}")
    if superficie == "partecipante" and lira_cfg.get("fonte_ombre", "universo") != "server":
        raise ValueError("observation_surface: partecipante richiede lira.fonte_ombre: server")
    in_size = lira_cfg.get("in_size", "client")
    if in_size not in ("client", "half"):
        raise ValueError(f"lira.in_size non valido: {in_size!r} (atteso 'client' o 'half')")
    gamma = float(lira_cfg.get("rmia_gamma", 1.0))
    frazione_z = float(lira_cfg.get("rmia_population_fraction", 0.2))

    exp_cfg = cfg.get("experiment", {})
    ml_cfg = dict(cfg["ml"])
    seed = int(exp_cfg.get("seed", 42))
    feats = list(_mia_feature_names(cfg))
    arch = _autoencoder_arch_kwargs(cfg)
    input_dim = int(ml_cfg["input_dim"])
    if shadow_epochs_cap is not None and shadow_epochs_cap != ml_cfg.get("epochs"):
        logger.warning(
            "LiRA v2: --shadow-epochs-cap %s diverso dalle epoche del client (%s): "
            "LiRA RIDOTTA, numeri non validi (ESPERIMENTI.md)", shadow_epochs_cap, ml_cfg.get("epochs"))
        ml_cfg["epochs"] = int(shadow_epochs_cap)
    dp_ablation = exp_cfg.get("dp_ablation", "full")
    gm_cfg = {"epsilon": exp_cfg.get("epsilon", 1.0), "delta": exp_cfg.get("delta", 1e-5),
              "max_grad_norm": exp_cfg.get("max_grad_norm", 1.0),
              "sigma_calibration": exp_cfg.get("sigma_calibration", "analitica")}
    rounds = sorted(r for r in fl_results if r >= 1)

    # Partizione dei membri: la stessa di run_fl_rounds
    if cluster_membership is not None:
        cluster_ids = list(cluster_membership.keys())
        pools = cluster_membership
    else:
        cluster_ids = ["highway", "urban", "residential", "corporate"]
        size = max(1, len(train_sessions) // len(cluster_ids))
        pools = {c: list(range(i * size, len(train_sessions) if i == len(cluster_ids) - 1 else (i + 1) * size))
                 for i, c in enumerate(cluster_ids)}
    membri = {c: [train_sessions[i] for i in pools[c]] for c in cluster_ids}
    fonte_ombre = str(lira_cfg.get("fonte_ombre", "universo"))
    if fonte_ombre not in ("universo", "server"):
        raise ValueError(f"lira.fonte_ombre deve essere 'universo' o 'server', trovato {fonte_ombre!r}")
    if fonte_ombre == "server":
        return _run_lira_server({
            "cfg": cfg, "holdout_sessions": holdout_sessions, "fl_results": fl_results, "seed": seed,
            "rounds": rounds, "cluster_ids": cluster_ids, "membri": membri, "n_shadow": n_shadow,
            "n_offline": int(lira_cfg.get("n_shadow_offline", n_shadow)), "feats": feats, "arch": arch,
            "input_dim": input_dim, "ml_cfg": ml_cfg, "gm_cfg": gm_cfg, "no_dp": no_dp, "dp_mode": dp_mode,
            "dp_ablation": dp_ablation, "superficie": superficie, "gamma": gamma, "frazione_z": frazione_z,
            "composed_output": composed_output, "per_sample_dump_path": per_sample_dump_path,
            "workers": workers,
        })
    non_membri = assegna_non_membri(holdout_sessions, cluster_ids, cfg,
                                    {c: len(v) for c, v in membri.items()})

    sdp = _subject_dp_config(ml_cfg)
    pesi_soggetti = (_subject_cross_client_weights(membri, str(sdp["subject_key"]), feats)
                     if sdp["enabled"] else None)

    common_init = bool(ml_cfg.get("common_init", False))
    init_comune = (fl_results.get(0) or {}).get("raw_global_weights") if common_init else None
    global_prev = {r: fl_results[r - 1]["global_weights"] for r in rounds if r > 1}

    def _vettore(sess: list[dict[str, Any]]) -> np.ndarray:
        return np.array([[float(s[f]) for f in feats] for s in sess], dtype=np.float32)

    def _valida(s: dict[str, Any]) -> bool:
        try:
            return all(np.isfinite(float(s[f])) for f in feats)
        except (KeyError, TypeError, ValueError):
            return False

    risultati: dict[int, dict[str, Any]] = {r: {} for r in rounds}
    righe: dict[int, list[dict[str, Any]]] = {r: [] for r in rounds}
    specs: list[SpecShadow] = []
    info_client: dict[str, dict[str, Any]] = {}

    for ci, c in enumerate(cluster_ids):
        rng = random.Random(seed + 9001 + ci * _SEED_CLUSTER)
        nm = [s for s in non_membri[c] if _valida(s)]
        mb = [s for s in membri[c] if _valida(s)]
        if not nm or not mb:
            logger.warning("LiRA v2 [%s]: membri o non membri assenti, client saltato", c)
            continue
        # popolazione z di RMIA: non membri (non canary) riservati, mai valutati,
        # scelti a caso senza guardare punteggi o perdite
        canarini = [s for s in mb + nm if s.get("_canary_group") is not None]
        canarini_ids = {id(s) for s in canarini}
        nm_shuf = [s for s in nm if id(s) not in canarini_ids]
        rng.shuffle(nm_shuf)
        n_z = max(1, int(round(frazione_z * len(nm_shuf)))) if len(nm_shuf) > 1 else 0
        z = nm_shuf[:n_z]
        nm_eval = nm_shuf[n_z:]
        mb_libere = [s for s in mb if id(s) not in canarini_ids]
        mb_eval = rng.sample(mb_libere, min(len(nm_eval), len(mb_libere)))
        valutati = mb_eval + nm_eval + canarini
        mb_ids = {id(s) for s in mb}
        membro = np.array([id(s) in mb_ids for s in valutati], dtype=bool)
        tutti = valutati + z                                   # z in coda
        x = _vettore(tutti)
        # universo e insiemi IN degli shadow
        universo = mb + nm
        n_in = len(mb) if in_size == "client" else max(1, len(universo) // 2)
        if n_in >= len(universo):
            raise ValueError(f"LiRA v2 [{c}]: insieme IN ({n_in}) >= universo ({len(universo)})")
        in_mask = np.zeros((n_shadow, len(tutti)), dtype=bool)
        pos = {id(s): j for j, s in enumerate(tutti)}
        for k in range(n_shadow):
            sseed = seed + ci * _SEED_CLUSTER + k * _SEED_SHADOW
            ins = _sample_preserving_canary_groups(random.Random(sseed), universo, n_in)
            for s in ins:
                j = pos.get(id(s))
                if j is not None:
                    in_mask[k, j] = True
            altri: dict[int, list[Any]] = {}
            n_cli: dict[int, int] = {}
            if superficie == "global":
                for r in rounds:
                    upd = fl_results[r].get("updates") or []
                    altri[r] = [None if u.cluster_id == c else u for u in upd]
                    propri = [u for u in upd if u.cluster_id == c]
                    if propri:
                        n_cli[r] = propri[0].n_samples
            specs.append(SpecShadow(
                cid=c, k=k, seed=sseed, rounds=rounds, in_sessions=ins, eval_x=x,
                ml_cfg=ml_cfg, arch=arch, input_dim=input_dim, gm_cfg=gm_cfg,
                no_dp=no_dp, dp_mode=dp_mode, dp_ablation=dp_ablation,
                init_weights=init_comune, global_prev=global_prev,
                subject_weights=(pesi_soggetti or {}).get(c, {}) if pesi_soggetti is not None else None,
                superficie=superficie, altri_update=altri, n_samples_client=n_cli,
                n_partecipanti=len(cluster_ids),
            ))
        info_client[c] = {"valutati": valutati, "membro": membro, "n_z": len(z), "x": x,
                          "in_mask": in_mask, "n_in": n_in, "universo": len(universo)}

    logger.info("LiRA v2: %d traiettorie di shadow (%d per client), %s worker, superficie %s, IN = %s",
                len(specs), n_shadow, workers, superficie, in_size)
    esiti = esegui_traiettorie(specs, workers=workers)
    per_client: dict[str, list[dict[int, np.ndarray]]] = {}
    for spec, es in zip(specs, esiti):
        per_client.setdefault(spec.cid, []).append(es)

    cumul_lira: dict[int, float] = {}
    cumul_rmia: dict[int, float] = {}
    cumul_rmia_off: dict[int, float] = {}
    conta_off: dict[int, int] = {}
    conta: dict[int, int] = {}
    meta: dict[int, tuple[bool, Any, Any]] = {}
    for r in rounds:
        rd = fl_results[r]
        if superficie == "global":
            pesi_target = {c: rd["global_weights"] for c in info_client}
        else:
            fonte = (rd.get("raw_updates") if (not no_dp and dp_mode == "dp-fedavg") else rd.get("updates")) or []
            pesi_target = {u.cluster_id: u.weights for u in fonte}
        lab_l: list[int] = []
        sc_l: list[float] = []
        lab_r: list[int] = []
        sc_r: list[float] = []
        lab_o: list[int] = []
        sc_o: list[float] = []
        raw_lab: list[int] = []
        raw_sc: list[float] = []
        can = {"l": [], "r": [], "raw": []}
        diag_c: dict[str, Any] = {}
        for c, inf in info_client.items():
            if c not in pesi_target:
                logger.warning("LiRA v2 round %d: bersaglio del client %s assente", r, c)
                continue
            mse_t = mse_per_campione(arch, input_dim, pesi_target[c], inf["x"])
            mse_s = np.stack([es[r] for es in per_client[c]])          # (n_shadow, n)
            nv = len(inf["valutati"])
            t = statistica(mse_t[:nv])
            score, diag = punteggio_lira(t, statistica(mse_s[:, :nv]), inf["in_mask"][:, :nv])
            tau = float(np.median(mse_t[nv:])) if inf["n_z"] else float(np.median(mse_t))
            llr_all = log_lr_rmia(mse_t, mse_s, tau)
            rmia = punteggio_rmia(llr_all[:nv], llr_all[nv:], gamma)
            ok = np.isfinite(score) & np.isfinite(rmia)
            llr_off = log_lr_rmia_offline(mse_t, mse_s, inf["in_mask"][:, :mse_s.shape[1]], tau)
            rmia_off = punteggio_rmia(llr_off[:nv], llr_off[nv:], gamma)
            ok_off = ok & np.isfinite(rmia_off)
            m = inf["membro"]
            for j in np.flatnonzero(ok):
                s = inf["valutati"][j]
                lab = int(m[j])
                canary = s.get("_canary_group") is not None
                # i canary entrano solo nelle metriche canary (revisione 2026-10-16)
                if not canary:
                    lab_l.append(lab); sc_l.append(float(score[j]))
                    lab_r.append(lab); sc_r.append(float(rmia[j]))
                    if ok_off[j]:
                        lab_o.append(lab); sc_o.append(float(rmia_off[j]))
                key = id(s)
                cumul_lira[key] = cumul_lira.get(key, 0.0) + float(score[j])
                cumul_rmia[key] = cumul_rmia.get(key, 0.0) + float(llr_all[j])
                if np.isfinite(llr_off[j]):
                    cumul_rmia_off[key] = cumul_rmia_off.get(key, 0.0) + float(llr_off[j])
                    conta_off[key] = conta_off.get(key, 0) + 1
                conta[key] = conta.get(key, 0) + 1
                meta[key] = (bool(m[j]), s.get("_canary_group"), s.get("session_id"))
                if s.get("_canary_group") is not None:
                    can["l"].append((lab, float(score[j])))
                    can["r"].append((lab, float(rmia[j])))
            for j in range(nv):
                if inf["valutati"][j].get("_canary_group") is not None:
                    can["raw"].append((int(m[j]), -float(mse_t[j])))
                else:
                    raw_lab.append(int(m[j])); raw_sc.append(-float(mse_t[j]))
            ok_nc = [j for j in np.flatnonzero(ok) if inf["valutati"][j].get("_canary_group") is None]
            pc = _metriche([int(m[j]) for j in ok_nc], [float(score[j]) for j in ok_nc], "lira")
            diag_c[c] = {"auc": pc.get("lira_auc_roc"), **{k: v for k, v in diag.items()},
                         "n_scartati": int(nv - ok.sum()), "tau_rmia": tau,
                         "n_in_shadow": inf["n_in"], "universo": inf["universo"]}
        res = _metriche(lab_l, sc_l, "lira")
        res.update(_metriche(lab_r, sc_r, "rmia"))
        res.update(_metriche(lab_o, sc_o, "rmia_off"))
        from sklearn.metrics import roc_auc_score
        res["lira_debug_raw_mse_auc_roc"] = (round(float(roc_auc_score(raw_lab, raw_sc)), 6)
                                             if len(set(raw_lab)) == 2 else None)
        if lab_l:
            mem = [s for l, s in zip(lab_l, sc_l) if l]
            non = [s for l, s in zip(lab_l, sc_l) if not l]
            if mem and non:
                res["lira_member_score_mean"] = round(float(np.mean(mem)), 6)
                res["lira_non_member_score_mean"] = round(float(np.mean(non)), 6)
                res["lira_score_gap"] = round(float(np.mean(mem) - np.mean(non)), 6)
        for nome, coppie in (("canary_auc_roc", can["l"]), ("canary_rmia_auc_roc", can["r"]),
                             ("canary_raw_mse_auc_roc", can["raw"])):
            ll = [a for a, _ in coppie]
            res[nome] = (round(float(roc_auc_score(ll, [b for _, b in coppie])), 6)
                         if len(set(ll)) == 2 else None)
        res.update({
            "lira_version": 2, "n_shadow": n_shadow, "lira_observation_surface": superficie,
            "lira_in_size": in_size, "rmia_gamma": gamma,
            "lira_per_client": diag_c,
        })
        risultati[r] = res
        logger.info("Round %d — LiRA v2 AUC %s, RMIA AUC %s (n=%d)", r, res.get("lira_auc_roc"),
                    res.get("rmia_auc_roc"), len(lab_l))

    # composizione sui round: solo i campioni valutati in tutti i round
    completi = [k for k, n in conta.items() if n == len(rounds) and meta[k][1] is None]
    if composed_output is not None and completi:
        lab = [int(meta[k][0]) for k in completi]
        composed_output.update({f"composed_{k}": v for k, v in
                                _metriche(lab, [cumul_lira[k] for k in completi], "lira").items()})
        # somma dei log-rapporti di RMIA sui round (non il punteggio a coppie di RMIA)
        composed_output.update({f"composed_{k}": v for k, v in
                                _metriche(lab, [cumul_rmia[k] for k in completi], "rmia_loglr").items()})
        completi_off = [k for k in completi if conta_off.get(k, 0) == len(rounds)]
        if completi_off:
            lab_off = [int(meta[k][0]) for k in completi_off]
            composed_output.update({f"composed_{k}": v for k, v in
                                    _metriche(lab_off, [cumul_rmia_off[k] for k in completi_off],
                                              "rmia_off_loglr").items()})
        composed_output["composed_n_samples"] = len(completi)
    if per_sample_dump_path is not None and completi:
        Path(per_sample_dump_path).parent.mkdir(parents=True, exist_ok=True)
        with open(per_sample_dump_path, "w") as f:
            json.dump({
                "seed": seed, "epsilon": exp_cfg.get("epsilon"), "no_dp": no_dp, "dp_mode": dp_mode,
                "lira_version": 2, "n_rounds": len(rounds), "n_records": len(completi),
                "records": [{"session_id": meta[k][2], "is_member": meta[k][0],
                             "is_canary": meta[k][1] is not None, "composed_score": cumul_lira[k],
                             "composed_rmia_log_lr": cumul_rmia[k],
                             "composed_rmia_off_log_lr": cumul_rmia_off.get(k)} for k in completi],
            }, f)
    return risultati


# ── Versione realistica: ombre solo su dati che il server possiede (2026-10-20) ─

_SEED_OFFLINE = 424_243
_SEED_EMULATO = 818_181


def insiemi_server(
    mb: list[dict[str, Any]], nm_valuta: list[dict[str, Any]], aux: list[dict[str, Any]],
    frazione_z: float, n_shadow: int, n_offline: int, seed: int, ci: int, campiona,
) -> dict[str, Any]:
    """Insiemi di un client per LiRA e RMIA con `lira.fonte_ombre: server`.

    Il server possiede solo due cose: i propri dati ausiliari `aux` (sessioni della
    stessa distribuzione, fuori dal training e dalla valutazione) e le sessioni che
    interroga (i candidati valutati). Mai il resto dei dati del client.

      - popolazione z di RMIA: una quota `frazione_z` di `aux`, mai usata per
        addestrare ombre e mai valutata;
      - candidati: membri non canary estratti in numero pari ai non membri di
        valutazione, i non membri di valutazione, i canary;
      - ombre online: ciascuna su meta' di (aux + candidati), cosi' ogni candidato e'
        IN in circa meta' delle ombre (Carlini et al., forma online); anche z sta nell'
        universo delle ombre online, come la popolazione di RMIA (Zarifzadeh et al.),
        ma non e' mai valutata;
      - ombre offline: ciascuna su meta' di (aux senza z); nessun candidato e' mai IN.

    `campiona(rng, pool, n)` e' `_sample_preserving_canary_groups` (i gruppi canary
    restano interi). Restituisce le liste di sessioni; nessun accesso al resto dei
    membri del client oltre ai candidati estratti."""
    rng = random.Random(seed + 9001 + ci * _SEED_CLUSTER)
    e_can = lambda s: s.get("_canary_group") is not None or s.get("_canary_role") is not None  # noqa: E731
    canarini = [s for s in mb + nm_valuta if e_can(s)]
    nm_eval = [s for s in nm_valuta if not e_can(s)]
    aux_l = [s for s in aux if not e_can(s)]
    rng.shuffle(aux_l)
    n_z = max(1, int(round(frazione_z * len(aux_l)))) if len(aux_l) > 1 else 0
    z, aux_tr = aux_l[:n_z], aux_l[n_z:]
    mb_libere = [s for s in mb if not e_can(s)]
    mb_eval = rng.sample(mb_libere, min(len(nm_eval), len(mb_libere)))
    valutati = mb_eval + nm_eval + canarini
    mb_ids = {id(s) for s in mb}
    membro = np.array([id(s) in mb_ids for s in valutati], dtype=bool)
    universo_on = aux_tr + z + valutati
    n_in_on = len(universo_on) // 2
    ins_on = [campiona(random.Random(seed + ci * _SEED_CLUSTER + k * _SEED_SHADOW), universo_on, n_in_on)
              for k in range(n_shadow)]
    n_in_off = len(aux_tr) // 2
    ins_off = ([random.Random(seed + ci * _SEED_CLUSTER + k * _SEED_SHADOW + _SEED_OFFLINE).sample(aux_tr, n_in_off)
                for k in range(n_offline)] if n_in_off >= 1 else [])
    return {"valutati": valutati, "membro": membro, "z": z, "aux_tr": aux_tr, "universo_on": universo_on,
            "n_in_on": n_in_on, "n_in_off": n_in_off, "ins_on": ins_on, "ins_off": ins_off}



def diagnostica_canary(t: np.ndarray, stat_on: np.ndarray, in_mask: np.ndarray,
                       membro: np.ndarray, valutati: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Diagnostica dei canary (2026-10-26): per i canary membri e non membri, mediana della
    statistica del bersaglio t e delle medie per campione sulle ombre online IN e OUT.

    Serve a leggere l'inversione di LiRA e RMIA online sui canary: se le ombre IN
    memorizzano il canary piu' del bersaglio, t dei membri cade dalla parte OUT del punto
    medio fra mu_in e mu_out. t e stat_on sono gia' nella scala della statistica (log MSE);
    stat_on ha forma (ombre, candidati), in_mask la stessa forma. None se non ci sono canary."""
    stat_on = np.asarray(stat_on, dtype=float)
    msk = np.asarray(in_mask, dtype=bool)
    t = np.asarray(t, dtype=float)
    idx = [j for j, s in enumerate(valutati) if s.get("_canary_group") is not None]
    if not idx:
        return None
    out: dict[str, Any] = {}
    for nome, sel in (("membri", [j for j in idx if membro[j]]), ("non_membri", [j for j in idx if not membro[j]])):
        if not sel:
            continue
        mi = [float(stat_on[msk[:, j], j].mean()) for j in sel if msk[:, j].any()]
        mo = [float(stat_on[~msk[:, j], j].mean()) for j in sel if (~msk[:, j]).any()]
        out[nome] = {"n": len(sel), "t_mediana": float(np.median(t[sel])),
                     "mu_in_mediana": float(np.median(mi)) if mi else None,
                     "mu_out_mediana": float(np.median(mo)) if mo else None}
    return out

def punteggio_lira_offline(t: np.ndarray, stat_ref: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """LiRA offline (Carlini et al. 2022, sez. IV-C): solo ombre che non hanno mai
    visto il campione. Statistica = log della MSE, quindi un membro ha t piu' basso.
    Punteggio = (mu_out - t) / s_out, con mu_out media per campione sulle ombre e
    s_out deviazione fissa per client e round (radice della media delle varianze per
    campione, ddof = 1). Monotono nel p-value unilaterale gaussiano, quindi stessa
    AUC. NaN se le ombre sono meno di due o la deviazione non e' positiva."""
    s = np.asarray(stat_ref, dtype=float)
    t = np.asarray(t, dtype=float)
    score = np.full(t.shape, np.nan)
    if s.ndim != 2 or s.shape[0] < 2:
        return score, {"sigma_out": None, "n_valutati": 0}
    with np.errstate(invalid="ignore"):
        mu = s.mean(axis=0)
        var = s.var(axis=0, ddof=1)
    var = var[np.isfinite(var)]
    sd = float(np.sqrt(var.mean())) if var.size else float("nan")
    ok = np.isfinite(t) & np.isfinite(mu)
    if np.isfinite(sd) and sd > 0:
        score[ok] = (mu[ok] - t[ok]) / sd
    return score, {"sigma_out": sd, "n_valutati": int(np.isfinite(score).sum())}


def punteggio_lira_out(t: np.ndarray, stat_shadow: np.ndarray,
                       in_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """LiRA offline nella forma di Carlini et al. (sez. IV-C) con le ombre online:
    per ogni campione si usano solo le ombre che NON l'hanno visto (OUT), cioe'
    ombre della stessa taglia di quelle online, allenate su dati ausiliari e altri
    candidati. Punteggio = (mu_out - t) / s_out, s_out fissa per client e round."""
    _, mu_out, _, n_out, _, s_out = stime_in_out(stat_shadow, in_mask)
    t = np.asarray(t, dtype=float)
    score = np.full(t.shape, np.nan)
    ok = (n_out >= 1) & np.isfinite(t) & np.isfinite(mu_out)
    if ok.any() and np.isfinite(s_out) and s_out > 0:
        score[ok] = (mu_out[ok] - t[ok]) / s_out
    return score, {"sigma_out": s_out, "n_valutati": int(np.isfinite(score).sum())}


def _run_lira_server(ctx: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """LiRA e RMIA, online e offline, con ombre addestrate solo su dati del server
    (`lira.fonte_ombre: server`). Vedi `insiemi_server` per gli insiemi; ombre,
    osservazione e statistica come nella versione con l'universo del client."""
    import json
    from pathlib import Path

    from run_experiments import (
        _sample_preserving_canary_groups, _subject_cross_client_weights, _subject_dp_config,
        group_indices_by_site,
    )

    from plugins.attacks.shadow2 import dividi_non_membri

    cfg = ctx["cfg"]
    holdout_sessions = ctx["holdout_sessions"]
    fl_results = ctx["fl_results"]
    seed, rounds, cluster_ids, membri = ctx["seed"], ctx["rounds"], ctx["cluster_ids"], ctx["membri"]
    n_shadow, n_off = ctx["n_shadow"], ctx["n_offline"]
    feats, arch, input_dim, ml_cfg, gm_cfg = ctx["feats"], ctx["arch"], ctx["input_dim"], ctx["ml_cfg"], ctx["gm_cfg"]
    no_dp, dp_mode, dp_ablation = ctx["no_dp"], ctx["dp_mode"], ctx["dp_ablation"]
    superficie, gamma, frazione_z = ctx["superficie"], ctx["gamma"], ctx["frazione_z"]
    composed_output, per_sample_dump_path, workers = (
        ctx["composed_output"], ctx["per_sample_dump_path"], ctx["workers"])
    exp_cfg = cfg.get("experiment", {})

    strategia = (cfg.get("partition") or {}).get("strategy", "per_site")
    if strategia != "per_site":
        raise ValueError("lira.fonte_ombre: server richiede partition.strategy: per_site "
                         f"(trovato {strategia!r}): i dati ausiliari del server sono divisi per sito")
    # Stessa divisione dell'holdout dello shadow realistico (shadow2.dividi_non_membri,
    # stesso seme): meta' ai dati ausiliari del server, meta' ai non membri valutati.
    fraz_cal = float((cfg.get("shadow") or {}).get("calibration_fraction", 0.5))
    gruppi = group_indices_by_site(holdout_sessions)
    idx_cal, idx_non = dividi_non_membri(gruppi, holdout_sessions, fraz_cal, seed)
    sito = {i: g for g, v in gruppi.items() for i in v}
    aux = {c: [holdout_sessions[i] for i in idx_cal if sito.get(i) == c] for c in cluster_ids}
    nm_val = {c: [holdout_sessions[i] for i in idx_non if sito.get(i) == c] for c in cluster_ids}

    def _valida(s: dict[str, Any]) -> bool:
        try:
            return all(np.isfinite(float(s[f])) for f in feats)
        except (KeyError, TypeError, ValueError):
            return False

    def _vettore(sess: list[dict[str, Any]]) -> np.ndarray:
        return np.array([[float(s[f]) for f in feats] for s in sess], dtype=np.float32)

    common_init = bool(ml_cfg.get("common_init", False))
    init_comune = (fl_results.get(0) or {}).get("raw_global_weights") if common_init else None
    global_prev = {r: fl_results[r - 1]["global_weights"] for r in rounds if r > 1}

    info: dict[str, dict[str, Any]] = {}
    for ci, c in enumerate(cluster_ids):
        mb = [s for s in membri[c] if _valida(s)]
        nmv = [s for s in nm_val[c] if _valida(s)]
        ax = [s for s in aux[c] if _valida(s)]
        if not mb or not nmv or len(ax) < 3:
            logger.warning("LiRA server [%s]: membri, non membri o dati del server insufficienti, "
                           "client saltato", c)
            continue
        ins = insiemi_server(mb, nmv, ax, frazione_z, n_shadow, n_off, seed, ci,
                             _sample_preserving_canary_groups)
        tutti = ins["valutati"] + ins["z"]                      # z in coda
        pos = {id(s): j for j, s in enumerate(tutti)}
        m_on = np.zeros((n_shadow, len(tutti)), dtype=bool)
        for k, ss in enumerate(ins["ins_on"]):
            for s in ss:
                j = pos.get(id(s))
                if j is not None:
                    m_on[k, j] = True
        info[c] = {**ins, "ci": ci, "x": _vettore(tutti), "n_z": len(ins["z"]), "in_mask_on": m_on,
                   "n_aux": len(ax)}

    sdp = _subject_dp_config(ml_cfg)
    pesi_on = pesi_off = None
    if sdp["enabled"]:
        key = str(sdp["subject_key"])
        pesi_on = _subject_cross_client_weights({c: v["universo_on"] for c, v in info.items()}, key, feats)
        pesi_off = _subject_cross_client_weights({c: v["aux_tr"] for c, v in info.items()}, key, feats)

    emulati: dict[str, dict[int, list[Any]]] = {}
    if superficie == "partecipante":
        # L'avversario partecipante vede solo i modelli globali: gli aggiornamenti
        # degli altri client li emula con i propri dati ausiliari di quel sito.
        from ml.base_ml import GradientUpdate
        spec_em = []
        for ci, c in enumerate(cluster_ids):
            dati = info[c]["aux_tr"] if c in info else [s for s in aux[c] if _valida(s)]
            if not dati:
                raise RuntimeError(f"LiRA partecipante: nessun dato del server per emulare il client {c}")
            spec_em.append(SpecShadow(
                cid=c, k=-1, seed=seed + ci * _SEED_CLUSTER + _SEED_EMULATO, rounds=rounds, in_sessions=dati,
                eval_x=np.zeros((0, input_dim), dtype=np.float32), ml_cfg=ml_cfg, arch=arch,
                input_dim=input_dim, gm_cfg=gm_cfg, no_dp=no_dp, dp_mode=dp_mode, dp_ablation=dp_ablation,
                init_weights=init_comune, global_prev=global_prev,
                subject_weights=(pesi_off or {}).get(c, {}) if pesi_off is not None else None,
                superficie="client"))
        emulati = dict(zip(cluster_ids, esegui_emulazioni(spec_em, workers=workers)))

    specs_on: list[SpecShadow] = []
    specs_off: list[SpecShadow] = []
    for c, inf in info.items():
        ci = inf["ci"]
        altri: dict[int, list[Any]] = {}
        n_cli: dict[int, int] = {}
        if superficie == "partecipante":
            # pesi di aggregazione pubblici (numerosita' dei client): l'ordine e le
            # numerosita' sono quelli del protocollo, i pesi del modello sono emulati
            for r in rounds:
                upd = fl_results[r].get("updates") or []
                altri[r] = [None if u.cluster_id == c else GradientUpdate(
                    node_id=f"emulato-{u.cluster_id}", cluster_id=u.cluster_id, round_num=r,
                    weights=emulati[u.cluster_id][r], gradients=None, loss=None, n_samples=u.n_samples,
                    metadata={}) for u in upd]
                propri = [u for u in upd if u.cluster_id == c]
                if propri:
                    n_cli[r] = propri[0].n_samples
        elif superficie == "global":
            for r in rounds:
                upd = fl_results[r].get("updates") or []
                altri[r] = [None if u.cluster_id == c else u for u in upd]
                propri = [u for u in upd if u.cluster_id == c]
                if propri:
                    n_cli[r] = propri[0].n_samples
        comuni = dict(rounds=rounds, eval_x=inf["x"], ml_cfg=ml_cfg, arch=arch, input_dim=input_dim,
                      gm_cfg=gm_cfg, no_dp=no_dp, dp_mode=dp_mode, dp_ablation=dp_ablation,
                      init_weights=init_comune, global_prev=global_prev,
                      superficie=("global" if superficie == "partecipante" else superficie),
                      altri_update=altri, n_samples_client=n_cli, n_partecipanti=len(cluster_ids))
        for k, ss in enumerate(inf["ins_on"]):
            specs_on.append(SpecShadow(
                cid=c, k=k, seed=seed + ci * _SEED_CLUSTER + k * _SEED_SHADOW, in_sessions=ss,
                subject_weights=(pesi_on or {}).get(c, {}) if pesi_on is not None else None, **comuni))
        for k, ss in enumerate(inf["ins_off"]):
            specs_off.append(SpecShadow(
                cid=c, k=k, seed=seed + ci * _SEED_CLUSTER + k * _SEED_SHADOW + _SEED_OFFLINE, in_sessions=ss,
                subject_weights=(pesi_off or {}).get(c, {}) if pesi_off is not None else None, **comuni))

    logger.info("LiRA server: %d ombre online e %d offline (%d e %d per client), %s worker, superficie %s",
                len(specs_on), len(specs_off), n_shadow, n_off, workers, superficie)
    esiti = esegui_traiettorie(specs_on + specs_off, workers=workers)
    per_on: dict[str, list[dict[int, np.ndarray]]] = {}
    per_off: dict[str, list[dict[int, np.ndarray]]] = {}
    for spec, es in zip(specs_on, esiti[:len(specs_on)]):
        per_on.setdefault(spec.cid, []).append(es)
    for spec, es in zip(specs_off, esiti[len(specs_on):]):
        per_off.setdefault(spec.cid, []).append(es)

    from sklearn.metrics import roc_auc_score
    acc = {n: {} for n in ("lira", "lira_out", "lira_off", "rmia", "rmia_out", "rmia_off")}
    conta = {n: {} for n in acc}
    meta: dict[int, tuple[bool, Any, Any]] = {}
    risultati: dict[int, dict[str, Any]] = {}
    for r in rounds:
        rd = fl_results[r]
        if superficie in ("global", "partecipante"):
            pesi_target = {c: rd["global_weights"] for c in info}
        else:
            fonte = (rd.get("raw_updates") if (not no_dp and dp_mode == "dp-fedavg") else rd.get("updates")) or []
            pesi_target = {u.cluster_id: u.weights for u in fonte}
        etich = {n: ([], []) for n in acc}
        can = {n: [] for n in list(acc) + ["raw"]}
        raw_lab: list[int] = []
        raw_sc: list[float] = []
        diag_c: dict[str, Any] = {}
        for c, inf in info.items():
            if c not in pesi_target or c not in per_on:
                logger.warning("LiRA server round %d: bersaglio o ombre del client %s assenti", r, c)
                continue
            mse_t = mse_per_campione(arch, input_dim, pesi_target[c], inf["x"])
            mse_on = np.stack([es[r] for es in per_on[c]])
            nv = len(inf["valutati"])
            t = statistica(mse_t[:nv])
            tau = float(np.median(mse_t[nv:])) if inf["n_z"] else float(np.median(mse_t))
            m_on = inf["in_mask_on"]
            s_on, d_on = punteggio_lira(t, statistica(mse_on[:, :nv]), m_on[:, :nv])
            s_out, d_out = punteggio_lira_out(t, statistica(mse_on[:, :nv]), m_on[:, :nv])
            llr_on = log_lr_rmia(mse_t, mse_on, tau)
            llr_out = log_lr_rmia_offline(mse_t, mse_on, m_on, tau)
            punt = {"lira": s_on, "lira_out": s_out,
                    "rmia": punteggio_rmia(llr_on[:nv], llr_on[nv:], gamma),
                    "rmia_out": punteggio_rmia(llr_out[:nv], llr_out[nv:], gamma)}
            somma = {"lira": s_on, "lira_out": s_out, "rmia": llr_on[:nv], "rmia_out": llr_out[:nv]}
            d_off = None
            if c in per_off:
                mse_off = np.stack([es[r] for es in per_off[c]])
                s_off, d_off = punteggio_lira_offline(t, statistica(mse_off[:, :nv]))
                llr_off = log_lr_rmia(mse_t, mse_off, tau)
                punt["lira_off"] = s_off
                punt["rmia_off"] = punteggio_rmia(llr_off[:nv], llr_off[nv:], gamma)
                somma["lira_off"] = s_off
                somma["rmia_off"] = llr_off[:nv]
            m = inf["membro"]
            for j in range(nv):
                s = inf["valutati"][j]
                lab = int(m[j])
                key = id(s)
                canary = s.get("_canary_group") is not None
                meta[key] = (bool(m[j]), s.get("_canary_group"), s.get("session_id"))
                for n, v in punt.items():
                    if not (np.isfinite(v[j]) and np.isfinite(somma[n][j])):
                        continue
                    if canary:
                        can[n].append((lab, float(v[j])))
                    else:
                        etich[n][0].append(lab); etich[n][1].append(float(v[j]))
                    acc[n][key] = acc[n].get(key, 0.0) + float(somma[n][j])
                    conta[n][key] = conta[n].get(key, 0) + 1
                if canary:
                    can["raw"].append((lab, -float(mse_t[j])))
                else:
                    raw_lab.append(lab); raw_sc.append(-float(mse_t[j]))
            canary_diag = diagnostica_canary(t, statistica(mse_on[:, :nv]), m_on[:, :nv],
                                             inf["membro"], inf["valutati"])
            diag_c[c] = {"lira_online": d_on, "lira_out": d_out, "lira_offline": d_off, "tau_rmia": tau,
                         "canary": canary_diag,
                         "n_valutati": nv, "n_z": inf["n_z"], "n_aux_server": inf["n_aux"],
                         "n_aux_ombre": len(inf["aux_tr"]), "n_in_online": inf["n_in_on"],
                         "n_in_offline": inf["n_in_off"]}
        res: dict[str, Any] = {}
        for n in acc:
            res.update(_metriche(etich[n][0], etich[n][1], n))
        res["lira_debug_raw_mse_auc_roc"] = (round(float(roc_auc_score(raw_lab, raw_sc)), 6)
                                             if len(set(raw_lab)) == 2 else None)
        for nome, coppie in (("canary_auc_roc", can["lira"]), ("canary_lira_out_auc_roc", can["lira_out"]),
                             ("canary_lira_off_auc_roc", can["lira_off"]),
                             ("canary_rmia_auc_roc", can["rmia"]), ("canary_rmia_out_auc_roc", can["rmia_out"]),
                             ("canary_rmia_off_auc_roc", can["rmia_off"]),
                             ("canary_raw_mse_auc_roc", can["raw"])):
            ll = [a for a, _ in coppie]
            res[nome] = (round(float(roc_auc_score(ll, [b for _, b in coppie])), 6)
                         if len(set(ll)) == 2 else None)
        res.update({"lira_version": 2, "lira_fonte_ombre": "server", "n_shadow": n_shadow,
                    "n_shadow_offline": n_off, "lira_observation_surface": superficie,
                    "rmia_gamma": gamma, "lira_per_client": diag_c})
        risultati[r] = res
        logger.info("Round %d — LiRA server online %s, offline OUT %s, offline aux %s; "
                    "RMIA online %s, offline OUT %s, offline aux %s",
                    r, res.get("lira_auc_roc"), res.get("lira_out_auc_roc"), res.get("lira_off_auc_roc"),
                    res.get("rmia_auc_roc"), res.get("rmia_out_auc_roc"), res.get("rmia_off_auc_roc"))

    # composizione sui round: somma per campione sui round in cui e' valutato in tutti
    # (log-rapporti per LiRA online e RMIA; punteggi z per LiRA offline, combinazione di Stouffer)
    if composed_output is not None:
        nomi = {"lira": "lira", "lira_out": "lira_out", "lira_off": "lira_off", "rmia": "rmia_loglr",
                "rmia_out": "rmia_out_loglr", "rmia_off": "rmia_off_loglr"}
        for n, pref in nomi.items():
            completi = [k for k, q in conta[n].items() if q == len(rounds) and meta[k][1] is None]
            if completi:
                lab = [int(meta[k][0]) for k in completi]
                composed_output.update({f"composed_{k}": v for k, v in
                                        _metriche(lab, [acc[n][k] for k in completi], pref).items()})
        composed_output["composed_n_samples"] = len(
            [k for k, q in conta["lira"].items() if q == len(rounds) and meta[k][1] is None])
        composed_output["lira_fonte_ombre"] = "server"
    if per_sample_dump_path is not None:
        completi = [k for k, q in conta["lira"].items() if q == len(rounds) and meta[k][1] is None]
        Path(per_sample_dump_path).parent.mkdir(parents=True, exist_ok=True)
        with open(per_sample_dump_path, "w") as f:
            json.dump({
                "seed": seed, "epsilon": exp_cfg.get("epsilon"), "no_dp": no_dp, "dp_mode": dp_mode,
                "lira_version": 2, "lira_fonte_ombre": "server", "n_rounds": len(rounds),
                "n_records": len(completi),
                "records": [{"session_id": meta[k][2], "is_member": meta[k][0],
                             "composed_score": acc["lira"].get(k), "composed_lira_out": acc["lira_out"].get(k),
                             "composed_lira_off": acc["lira_off"].get(k),
                             "composed_rmia_log_lr": acc["rmia"].get(k),
                             "composed_rmia_out_log_lr": acc["rmia_out"].get(k),
                             "composed_rmia_off_log_lr": acc["rmia_off"].get(k)} for k in completi],
            }, f)
    return risultati
