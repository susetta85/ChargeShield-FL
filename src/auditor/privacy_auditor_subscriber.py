# src/auditor/privacy_auditor_subscriber.py
# ChargeShield-FL — Privacy Auditor come vero subscriber ML Plane
#
# FASE 8 (2026-08-31, richiesta esplicita dell'autore dopo il feedback "il
# Privacy Auditor consuma i dati raccolti dal ML Plane ma viene invocato
# imperativamente, non tramite una vera subscription al ML Plane" — confermato
# leggendo il codice PRIMA di questo fix: PrivacyAuditor (privacy_auditor.py)
# non implementava MLPlaneListener; run_ids() (scripts/run_experiments.py) e
# _run_ids_analysis() (nvflare/.../chargeshield_aggregator.py, corretto nella
# Fase 7) calcolavano le delta peer-relative a mano e chiamavano
# auditor.audit(model_update=...) direttamente, mai come reazione a un evento.
#
# File SEPARATO da privacy_auditor.py (non la stessa classe nello stesso
# modulo): privacy_auditor.py è importato a livello di modulo da test
# torch-free (tests/test_privacy_auditor.py) e da altro codice che non
# richiede torch — un primo tentativo di aggiungere questa classe lì, con
# `import torch` a livello di modulo, ha rotto quei test
# (ModuleNotFoundError). Questo file dipende da torch (operazioni sui tensori
# dei pesi) ed è importato solo da chi ha torch disponibile:
# scripts/run_experiments.py (torch già importato incondizionatamente) e
# chargeshield_aggregator.py (import lazy dentro _ensure_components(), stesso
# principio già seguito da src/ml/gradient_manager.py).

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import torch

from ml.base_ml import GradientUpdate, MLPlaneEvent, MLPlaneListener
from ml.ml_plane import FLArtifactCollector

if TYPE_CHECKING:
    from auditor.privacy_auditor import PrivacyAuditor
    from core.base_auditor import AuditReport


class PrivacyAuditorSubscriber(MLPlaneListener):
    """
    Rende PrivacyAuditor un vero subscriber ML Plane.

    Design: reagisce SOLO all'evento "aggregation" (il segnale "round
    completo" — emesso da `FedAvgAggregator.aggregate()` quando è wired al
    ML Plane) e, a quel punto, legge gli update del round dal
    `FLArtifactCollector` che gli viene passato al costruttore — STESSA
    istanza già usata per raccogliere raw/privatized (mai una seconda fonte
    di verità: bufferizzare qui gli eventi "gradient_upload" in proprio
    avrebbe rischiato di perdere la semantica "raw preferito, privatized come
    fallback" già corretta nel collector, es. per dp-fedavg dove un evento
    purdue_level=2 arriva DOPO il purdue_level=1 nello stesso round —
    ribufferizzarli con un ingenuo "ultimo vince" avrebbe silenziosamente
    sovrascritto il raw col privatizzato, cambiando l'input reale
    dell'Auditor/IDS rispetto a prima).

    Applica poi ESATTAMENTE la stessa normalizzazione peer-relative (mediana
    → max_grad_norm, fix Sprint 9 per GRADIENT_EXPLOSION) già usata in
    run_ids()/_run_ids_analysis() — stessa formula, stessi input, stessi
    output: cambia SOLO il meccanismo di attivazione (subscribe/on_ml_event
    invece di una chiamata imperativa dopo un loop manuale), non un singolo
    numero già pubblicato (campagna 5-seed×8-config, Sprint 10tt).

    Idempotente per round: un'aggregazione può essere ri-emessa più volte
    nello stesso round (central DP, dopo privatize_aggregate() — vedi
    run_fl_rounds()/chargeshield_aggregator.py) — il secondo trigger per lo
    stesso round_num è un no-op esplicito (i report non dipendono dal rumore
    central-DP sull'aggregato, solo dagli update raw/privatizzati dei
    client, già invariati tra le due emissioni).
    """

    def __init__(
        self,
        auditor: "PrivacyAuditor",
        collector: FLArtifactCollector,
        max_grad_norm: float,
    ) -> None:
        self._auditor = auditor
        self._collector = collector
        self._max_grad_norm = max_grad_norm
        # Baseline pre-round per il delta — stesso ruolo di prev_raw_global
        # (run_ids()) / _prev_raw_global_weights (chargeshield_aggregator.py,
        # rimosso da lì in questa stessa fase, vedi commento lì).
        # Impostare esplicitamente via set_initial_baseline() prima del primo
        # round, se disponibile (nella simulazione: pesi random-init del
        # trainer, round 0; in NVFLARE: resta None, stesso KNOWN GAP già
        # documentato in chargeshield_aggregator.py per il round 1).
        self._prev_baseline: list[Any] | None = None
        self._reports_by_round: dict[int, dict[str, "AuditReport"]] = {}
        self._gradients_by_round: dict[int, dict[str, dict[str, Any]]] = {}
        # Overhead ML Plane attribuibile all'Auditor (Sprint 10zz+106, richiesto
        # esplicitamente dall'utente per il D1 corretto — "overhead con e senza
        # Auditor"). Misurato qui, non con due run separati: il rumore fra run
        # (variabilità di training/I-O) sommergerebbe il segnale, che è
        # tipicamente sotto il millisecondo per round su questi modelli piccoli.
        # Timer intorno all'UNICA sezione di lavoro reale in questa classe
        # (normalizzazione peer-relative + N chiamate a self._auditor.audit()),
        # sempre attivo — costo del timer stesso trascurabile (time.perf_counter()),
        # nessun flag opt-in necessario. Stessa istanza usata sia dal replay
        # post-hoc in scripts/run_experiments.py::run_ids() sia dal wiring dal
        # vivo in nvflare/.../chargeshield_aggregator.py — un solo punto di
        # strumentazione copre entrambi gli ambienti.
        self._overhead_seconds_by_round: dict[int, float] = {}

    def set_initial_baseline(self, weights: list[Any] | None) -> None:
        self._prev_baseline = weights

    def on_ml_event(self, event: MLPlaneEvent) -> None:
        if event.event_type != "aggregation":
            return
        self._handle_round_complete(event.round_num)

    def _handle_round_complete(self, round_num: int) -> None:
        if round_num in self._reports_by_round:
            return  # già processato — vedi nota idempotenza sopra

        # Stessa selezione "raw preferito, privatizzato come fallback" già
        # usata ovunque in questo progetto (run_ids(): `raw_updates or
        # updates`; chargeshield_aggregator.py: FLArtifactCollector.
        # privatized_updates() ricade già su raw_updates() se vuoto — qui
        # vogliamo l'inverso esplicito, la vista PIÙ raw disponibile).
        updates: list[GradientUpdate] = (
            self._collector.raw_updates(round_num)
            or self._collector.privatized_updates(round_num)
        )
        if not updates:
            self._reports_by_round[round_num] = {}
            self._gradients_by_round[round_num] = {}
            self._overhead_seconds_by_round[round_num] = 0.0
            return

        # Inizio finestra di misura overhead (Sprint 10zz+106) — copre TUTTO
        # il lavoro reale del componente per questo round: normalizzazione
        # peer-relative, le N chiamate a self._auditor.audit(), e
        # l'aggiornamento della baseline. Esclude solo il primo controllo di
        # idempotenza sopra (non è lavoro dell'Auditor, è un no-op).
        _overhead_t0 = time.perf_counter()

        prev_baseline = self._prev_baseline

        client_deltas: dict[str, list[Any]] = {}
        client_norms: dict[str, float] = {}
        for update in updates:
            if not update or not update.node_id:
                continue
            weights = update.weights or []
            if prev_baseline is not None and len(prev_baseline) == len(weights):
                delta = [
                    (w.float() if isinstance(w, torch.Tensor) else torch.tensor(float(w)))
                    - (g.float() if isinstance(g, torch.Tensor) else torch.tensor(float(g)))
                    for w, g in zip(weights, prev_baseline)
                ]
            else:
                delta = [
                    w.float() if isinstance(w, torch.Tensor) else torch.tensor(float(w))
                    for w in weights
                ]
            l2_sq = sum(
                float(dw.float().norm() ** 2) if isinstance(dw, torch.Tensor) else float(dw) ** 2
                for dw in delta
            )
            client_deltas[update.node_id] = delta
            client_norms[update.node_id] = float(math.sqrt(max(l2_sq, 1e-12)))

        if client_norms:
            sorted_norms = sorted(client_norms.values())
            median_norm = sorted_norms[(len(sorted_norms) - 1) // 2]
            scale = self._max_grad_norm / median_norm if median_norm >= 1e-4 else 1.0
        else:
            scale = 1.0

        reports: dict[str, "AuditReport"] = {}
        gradients: dict[str, dict[str, Any]] = {}
        for node_id, delta in client_deltas.items():
            model_update = {
                f"layer_{i}": (dw * scale if isinstance(dw, torch.Tensor) else torch.tensor(float(dw) * scale))
                for i, dw in enumerate(delta)
            }
            reports[node_id] = self._auditor.audit(
                node_id=node_id, round_id=round_num, model_update=model_update,
            )
            gradients[node_id] = {f"layer_{i}": dw for i, dw in enumerate(delta)}

        self._reports_by_round[round_num] = reports
        self._gradients_by_round[round_num] = gradients

        # Baseline per il prossimo round: SOLO dalla vista raw (mai dalla
        # privatizzata), a differenza di `updates` sopra (che accetta il
        # fallback privatizzato per il calcolo del delta DI QUESTO round).
        # Replica esattamente _raw_updates_for_export/_raw_global_weights_for_export
        # in chargeshield_aggregator.py e raw_global_weights in run_ids()/
        # run_fl_rounds(): sotto dp_mode="local", self._collector.raw_updates()
        # è SEMPRE vuoto (accept() non emette mai purdue_level=1 in quel modo,
        # vedi commento lì) → la baseline non avanza MAI, restando None per
        # tutta la durata del run — stessa "degradazione attesa e documentata"
        # di run_ids() ("IDS usa pesi PRE-DP... sotto local DP il calcolo del
        # delta peer-relative degrada a confronto sui pesi ASSOLUTI"), non un
        # bug di questo refactor. Se invece qui si usasse `updates` (che sotto
        # local DP è il fallback privatizzato), la baseline si popolerebbe
        # comunque e questa degradazione intenzionale sparirebbe silenziosamente.
        raw_only = self._collector.raw_updates(round_num)
        if raw_only:
            self._prev_baseline = self._weighted_average(raw_only)
        # else: baseline invariata rispetto al round precedente (None se mai
        # popolata — sotto local DP resta così per l'intero run, per design).

        self._overhead_seconds_by_round[round_num] = time.perf_counter() - _overhead_t0

    def reports_for_round(self, round_num: int) -> dict[str, "AuditReport"]:
        return self._reports_by_round.get(round_num, {})

    def gradients_for_round(self, round_num: int) -> dict[str, dict[str, Any]]:
        return self._gradients_by_round.get(round_num, {})

    def overhead_seconds_for_round(self, round_num: int) -> float:
        """Secondi di wall-clock spesi in questo round dentro
        _handle_round_complete() — normalizzazione peer-relative + audit() +
        aggiornamento baseline. 0.0 se il round non ha avuto update (nessun
        lavoro svolto) o non è ancora stato processato. Vedi commento in
        __init__ per il perché è misurato qui invece che con due run A/B."""
        return self._overhead_seconds_by_round.get(round_num, 0.0)

    @staticmethod
    def _weighted_average(updates: list[GradientUpdate]) -> list[Any] | None:
        """Stessa formula di chargeshield_aggregator.py::_weighted_average_weights()
        — media pesata per n_samples, non una re-implementazione indipendente."""
        if not updates or not updates[0].weights:
            return None
        n_w = len(updates[0].weights)
        total = sum(u.n_samples for u in updates) or len(updates)
        averaged = []
        for i in range(n_w):
            wavg = sum(
                (u.weights[i] if isinstance(u.weights[i], torch.Tensor)
                 else torch.tensor(float(u.weights[i])))
                * (u.n_samples / total)
                for u in updates
            )
            averaged.append(wavg)
        return averaged
