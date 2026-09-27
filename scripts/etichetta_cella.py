"""
Etichetta di cella comune a check_significance.py e genera_matrici_faseA.py.

Segnalazioni 44, 45 e 53 (corrette il 2026-09-24). Prima le celle si formavano
con (dp_mode, epsilon), "no-DP" o "record-DP, nm=...": due run con trattamenti
diversi ma stesso (dp_mode, epsilon) finivano nella stessa cella, e con la
deduplica per seed di check_significance.py la run piu' recente sostituiva
l'altra. Casi concreti: la soglia di clipping C (griglia del punto operativo),
l'inizializzazione comune (controllo della segnalazione 48), FedAvg contro
FedProx (E-E), la partizione IID (E-D).

REGOLA. Due run stanno nella stessa cella solo se coincidono tutti i campi del
config registrati nel JSON che definiscono il trattamento. L'etichetta parte da
quella storica e aggiunge un suffisso per ogni campo registrato che differisce
dal config di base (config/experiment.yaml). Un campo ASSENTE dal JSON non
aggiunge nulla: le run anteriori al 2026-09-24 non salvavano max_grad_norm ne'
common_init, e tutti i config del repository a quella data hanno C = 1.0 e
nessuna inizializzazione comune; partition_strategy e split_strategy sono
salvati dal 2026-09-21. Per le run gia' prodotte con i valori di base
l'etichetta resta quella di prima.

Il seed non fa parte dell'etichetta: e' la replica dentro la cella.
"""
from __future__ import annotations

from typing import Any

# Valori di config/experiment.yaml e dei default di run_experiments.py.
_FEATURE_DEFAULT = [
    "total_energy_kwh", "max_power_kw", "kwh_requested",
    "minutes_available", "hour_of_day", "duration_hours",
]
BASE: dict[str, Any] = {
    "max_grad_norm": 1.0,
    "common_init": False,
    "proximal_mu": 0.01,
    "epochs": 50,
    "fl_rounds": 10,
    "delta": 1e-5,
    "hidden_dims": None,          # None = (16, 8), l'architettura storica
    "latent_dim": 4,
    "feature_names": None,        # None = le 6 feature storiche
    "partition_strategy": "per_site",
    "split_strategy": "random",
}

# campo -> come compare nel suffisso
_NOMI = {
    "max_grad_norm": "C",
    "common_init": "init comune",
    "proximal_mu": "mu",
    "epochs": "epoche",
    "fl_rounds": "round",
    "delta": "delta",
    "hidden_dims": "hidden",
    "latent_dim": "latent",
    "feature_names": "feature",
    "partition_strategy": "partizione",
    "split_strategy": "split",
}


def _normalizza(campo: str, valore: Any) -> Any:
    if campo == "hidden_dims":
        return None if valore in (None, [], (16, 8), [16, 8]) else list(valore)
    if campo == "feature_names":
        return None if (not valore or list(valore) == _FEATURE_DEFAULT) else list(valore)
    if campo == "common_init":
        return bool(valore)
    if campo in ("max_grad_norm", "proximal_mu", "delta"):
        return float(valore)
    if campo in ("epochs", "fl_rounds", "latent_dim"):
        return int(valore)
    return valore


def etichetta_base(cfg: dict) -> str:
    """L'etichetta storica, identica a quella di check_significance.py prima
    del 2026-09-24."""
    rdp = cfg.get("record_dp") or {}
    eps = cfg.get("epsilon")
    no_dp = cfg.get("no_dp", eps is None)
    if rdp.get("enabled"):
        return f"record-DP, nm={rdp.get('noise_multiplier')}"
    if no_dp or eps is None:
        return "no-DP baseline"
    return f"{cfg.get('dp_mode', 'dp-fedavg')}, eps={eps}"


def scostamenti(cfg: dict) -> list[tuple[str, Any]]:
    """Campi registrati che differiscono dalla base, nell'ordine di BASE."""
    client_level = etichetta_base(cfg) != "no-DP baseline" and not (
        (cfg.get("record_dp") or {}).get("enabled"))
    out = []
    for campo, base in BASE.items():
        if campo not in cfg:
            continue                      # non registrato: nessun suffisso
        if cfg[campo] is None and campo not in ("hidden_dims", "feature_names"):
            continue                      # registrato vuoto: come non registrato
        if campo == "max_grad_norm" and not client_level:
            continue                      # C conta solo col clipping client-level
        try:
            v, b = _normalizza(campo, cfg[campo]), _normalizza(campo, base)
        except (TypeError, ValueError):
            v, b = cfg[campo], base
        if v != b:
            out.append((campo, v))
    return out


def etichetta_cella(cfg: dict) -> str:
    """Etichetta della cella di una run, dal blocco config del suo JSON."""
    et = etichetta_base(cfg)
    for campo, v in scostamenti(cfg):
        nome = _NOMI[campo]
        if campo == "common_init":
            et += f", {nome}"
        elif campo == "feature_names":
            et += f", {nome}={len(v)}"
        else:
            et += f", {nome}={v}"
    return et
