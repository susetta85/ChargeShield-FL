"""
Contabilita' del budget per la DP a livello di record (DP-SGD per esempio).

Correzione della segnalazione 4 (2026-09-24). Prima di questa correzione il
calcolo stava in `scripts/run_experiments.py::_record_dp_fields` e aveva tre
difetti: leggeva `fl_rounds` e `delta` dalla radice di `cfg`, dove non ci sono
(vivono in `cfg["experiment"]`), quindi usava sempre 1 round e il delta di
default; usava il numero GLOBALE di sessioni di training, mentre ogni client
addestra sul proprio sottoinsieme; e dichiarava un campionamento di Poisson che
il training non usa. Il modulo e' puro Python (dipende solo da `dp-accounting`,
importato dentro le funzioni) cosi' i test non richiedono torch.

Cosa si calcola, per ogni client i con n_i record di training, batch B, E epoche
locali per round e T round:

1. `epsilon` di Poisson (valore principale, APPROSSIMAZIONE). RDP accountant,
   adiacenza add/remove, campionamento di Poisson con q_i = B / n_i,
   passi = floor(n_i / B) * E * T. E' la contabilita' standard di DP-SGD ed e'
   il numero confrontabile con la letteratura. Ma il training qui non campiona
   alla Poisson: usa `DataLoader(shuffle=True, drop_last=True)`
   (`src/ml/autoencoder_trainer.py`), cioe' batch di dimensione fissa presi da
   una permutazione per epoca. Con lo shuffle l'accountant di Poisson non e' una
   garanzia e puo' sottostimare epsilon (Chua et al., ICML 2024).

2. Limite valido con lo shuffle, senza amplificazione da campionamento.
   Adiacenza per SOSTITUZIONE di un record (n fissato): in ogni epoca un record
   entra in al piu' un batch; sostituirlo cambia la somma dei gradienti clippati
   di al piu' 2C, e il rumore ha deviazione standard sigma*C, quindi ogni epoca
   e' al piu' un meccanismo gaussiano con moltiplicatore sigma/2. Composizione
   adattiva su E*T epoche (RDP). Non dipende da n_i ne' da B. Attenzione:
   l'adiacenza e' diversa da quella del punto 1, i due numeri non sono sulla
   stessa scala e vanno riportati ciascuno con la propria.

Budget della cella = massimo sui client: ogni client protegge i propri record,
e il client con meno record (tasso di campionamento piu' alto) e' il peggiore.

Non contabilizzati in nessuno dei due numeri: le statistiche min-max calcolate
sui dati di training prima del meccanismo e i pesi di aggregazione FedAvg
proporzionali a n_i. n_i e' il numero di sessioni assegnate al client; se
`_sessions_to_tensor` ne scartasse qualcuna (feature mancanti) n_i reale
sarebbe minore e epsilon leggermente maggiore.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _param(cfg: dict, section: str, key: str) -> Any:
    """Legge `key` da `cfg[section]` (config YAML annidato) o dalla radice
    (blocco `config` piatto dei JSON risultato). None se assente in entrambi.
    Nessun default silenzioso: il default 1 per i round era il bug."""
    sec = cfg.get(section)
    if isinstance(sec, dict) and sec.get(key) is not None:
        return sec[key]
    if cfg.get(key) is not None:
        return cfg[key]
    return None


def epsilon_poisson(
    n: int, batch_size: int, epochs: int, rounds: int,
    noise_multiplier: float, delta: float,
) -> tuple[float, int, float]:
    """(epsilon, passi, q) per UN client, RDP con campionamento di Poisson,
    adiacenza add/remove. Con n < batch_size il client non produce batch
    (drop_last=True): 0 passi, epsilon 0."""
    from dp_accounting import dp_event, rdp

    if n < batch_size:
        return 0.0, 0, float(batch_size) / float(n) if n else float("nan")
    steps = (n // batch_size) * epochs * rounds
    q = float(batch_size) / float(n)
    acc = rdp.RdpAccountant()
    acc.compose(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(noise_multiplier)),
        steps,
    )
    return float(acc.get_epsilon(target_delta=delta)), steps, q


def epsilon_shuffle_bound(
    epochs: int, rounds: int, noise_multiplier: float, delta: float,
) -> float:
    """Limite valido con lo shuffle, adiacenza per sostituzione, senza
    amplificazione: E*T meccanismi gaussiani con moltiplicatore sigma/2."""
    from dp_accounting import dp_event, rdp

    acc = rdp.RdpAccountant()
    acc.compose(dp_event.GaussianDpEvent(noise_multiplier / 2.0), epochs * rounds)
    return float(acc.get_epsilon(target_delta=delta))


def record_dp_accounting(
    n_per_client: dict[str, int], batch_size: int, epochs: int, rounds: int,
    noise_multiplier: float, delta: float,
) -> dict[str, Any]:
    """Entrambi i numeri, per client e massimo. Solleva ImportError se
    `dp-accounting` non e' installato."""
    per_client: dict[str, dict[str, Any]] = {}
    for cid, n in n_per_client.items():
        eps, steps, q = epsilon_poisson(
            int(n), batch_size, epochs, rounds, noise_multiplier, delta)
        per_client[str(cid)] = {"n": int(n), "q": q, "passi": steps, "epsilon": eps}
    worst = max(per_client, key=lambda c: per_client[c]["epsilon"])
    return {
        "per_client": per_client,
        "client_peggiore": worst,
        "epsilon_max": per_client[worst]["epsilon"],
        "epsilon_shuffle_bound": epsilon_shuffle_bound(
            epochs, rounds, noise_multiplier, delta),
    }


def record_dp_fields(
    cfg: dict,
    n_sessions: int | None = None,
    n_per_client: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Campi record-DP del JSON risultato.

    Accetta sia il config YAML annidato (`cfg["experiment"]`, `cfg["ml"]`) sia
    il blocco `config` piatto dei JSON gia' salvati. `fl_rounds` deve essere
    quello effettivo: `load_config` scrive `--rounds` in
    `cfg["experiment"]["fl_rounds"]`, quindi leggerlo da li' basta.

    `epsilon_record_dp` e' il massimo sui client dell'epsilon di Poisson. Il
    limite valido con lo shuffle e' in `epsilon_record_dp_shuffle_bound`.
    Nessun numero viene calcolato con parametri mancanti: in quel caso i campi
    restano None e la nota dice perche'.
    """
    ml = cfg.get("ml") if isinstance(cfg.get("ml"), dict) else cfg
    rdp_cfg = ml.get("record_dp") or cfg.get("record_dp") or {}
    out: dict[str, Any] = {
        "record_dp": rdp_cfg or None,
        "norm": ml.get("norm", "batch"),
        "batch_size": _param(cfg, "ml", "batch_size"),
        "n_train_sessions": n_sessions,
        "n_train_per_client": dict(n_per_client) if n_per_client else None,
        "epsilon_record_dp": None,
        "epsilon_record_dp_per_client": None,
        "epsilon_record_dp_shuffle_bound": None,
        "record_dp_accounting": None,
        "record_dp_accounting_note": None,
    }
    if not rdp_cfg.get("enabled"):
        out["record_dp_accounting_note"] = (
            "record-DP disattivo: il campo 'epsilon' si riferisce al "
            "meccanismo client-level (weight perturbation)."
        )
        return out

    sigma = float(rdp_cfg.get("noise_multiplier") or 0.0)
    batch = _param(cfg, "ml", "batch_size")
    epochs = _param(cfg, "ml", "epochs")
    rounds = _param(cfg, "experiment", "fl_rounds")
    delta = _param(cfg, "experiment", "delta")
    mancanti = [nome for nome, v in (
        ("batch_size", batch), ("epochs", epochs), ("fl_rounds", rounds),
        ("delta", delta), ("n per client", n_per_client or None),
    ) if v is None]
    if mancanti or sigma <= 0.0:
        motivo = (f"parametri mancanti: {', '.join(mancanti)}" if mancanti
                  else f"noise_multiplier={sigma}: nessuna garanzia")
        out["record_dp_accounting_note"] = (
            f"record-DP attivo ma epsilon non calcolato ({motivo}). "
            "ATTENZIONE: il campo 'epsilon' NON si applica a questa cella."
        )
        return out

    batch, epochs, rounds, delta = int(batch), int(epochs), int(rounds), float(delta)
    try:
        acc = record_dp_accounting(n_per_client, batch, epochs, rounds, sigma, delta)
    except ImportError:
        logger.warning(
            "[RECORD-DP] 'dp-accounting' non installato: epsilon_record_dp non "
            "calcolato. Installare con: pip install dp-accounting "
            "(dichiarato in pyproject.toml), poi ricalcolare con "
            "scripts/ricalcola_epsilon_record_dp.py."
        )
        out["record_dp_accounting_note"] = (
            "record-DP attivo; 'dp-accounting' non installato, epsilon da "
            "ricalcolare con scripts/ricalcola_epsilon_record_dp.py. "
            "Il campo 'epsilon' NON si applica."
        )
        return out
    except Exception as exc:  # pragma: no cover
        # Chiamato da save_results mentre costruisce il JSON: un'eccezione qui
        # farebbe perdere l'intera run. Si registra e si prosegue.
        logger.warning("[RECORD-DP] accountant fallito: %s", exc)
        out["record_dp_accounting_note"] = (
            f"record-DP attivo; accountant fallito ({exc}). "
            "Il campo 'epsilon' NON si applica."
        )
        return out

    try:
        from importlib.metadata import version
        _ver = version("dp-accounting")
    except Exception:  # pragma: no cover
        _ver = None
    out["epsilon_record_dp"] = acc["epsilon_max"]
    out["epsilon_record_dp_per_client"] = {
        c: v["epsilon"] for c, v in acc["per_client"].items()}
    out["epsilon_record_dp_shuffle_bound"] = acc["epsilon_shuffle_bound"]
    out["record_dp_accounting"] = {
        "accountant": "RDP (dp-accounting)",
        "dp_accounting_version": _ver,
        "principale": ("Poisson, adiacenza add/remove, q = B/n per client; "
                       "APPROSSIMAZIONE: il training usa shuffle con batch fissi"),
        "limite_shuffle": ("adiacenza per sostituzione, senza amplificazione: "
                           "E*T meccanismi gaussiani con moltiplicatore sigma/2"),
        "noise_multiplier": sigma, "batch_size": batch, "epochs": epochs,
        "fl_rounds": rounds, "delta": delta,
        "per_client": acc["per_client"],
        "client_peggiore": acc["client_peggiore"],
    }
    per_c = ", ".join(f"{c} {v['epsilon']:.4g}" for c, v in acc["per_client"].items())
    out["record_dp_accounting_note"] = (
        f"epsilon_record_dp = {acc['epsilon_max']:.4g} (massimo sui client, "
        f"{acc['client_peggiore']}; per client: {per_c}). RDP, Poisson "
        f"add/remove, sigma={sigma}, B={batch}, E={epochs}, T={rounds}, "
        f"delta={delta}: APPROSSIMAZIONE, il training usa shuffle. Limite valido "
        f"con lo shuffle (sostituzione, senza amplificazione): "
        f"{acc['epsilon_shuffle_bound']:.4g}. Questi, NON il campo 'epsilon', "
        "sono il budget di questa cella."
    )
    return out
