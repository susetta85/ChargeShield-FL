# src/ml/fedavg_aggregator.py
# ChargeShield-FL — ML Plane: FedAvg Aggregator
#
# Responsabilità (Purdue L2→L3):
#   - riceve GradientUpdate privatizzati da GradientManager
#   - esegue FedAvg (media pesata per n_samples)
#   - emette AggregatedUpdate verso il server NVFLARE
#   - non conosce AutoencoderTrainer, né Auditor, né IDS

from __future__ import annotations

import logging
from typing import Any

import torch

from ml.base_ml import (
    AbstractMLModel,
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)

logger = logging.getLogger(__name__)


class FedAvgAggregator(AbstractMLModel):
    """
    Aggrega i GradientUpdate dei client via FedAvg (McMahan et al., 2017).
    Media pesata per numero di campioni locali.

    Purdue Level: L2 (Supervisory) → L3 (Operations)
    ML Plane role: produce AggregatedUpdate, emette evento aggregation.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._listeners: list[MLPlaneListener] = []
        self._round_updates: list[GradientUpdate] = []
        config = config or {}
        # Minimo partecipanti per procedere con l'aggregazione
        self.min_participants = config.get("min_participants") or 2
        logger.info(f"FedAvgAggregator — min_participants={self.min_participants}")

    # ── AbstractMLModel ────────────────────────────────────────────────────────

    def get_weights(self) -> list[Any]:
        raise NotImplementedError("FedAvgAggregator non possiede un modello locale.")

    def set_weights(self, weights: list[Any]) -> None:
        raise NotImplementedError("FedAvgAggregator non possiede un modello locale.")

    def train_step(self, data: Any) -> float:
        raise NotImplementedError("FedAvgAggregator non esegue training.")

    def emit_event(self, event: MLPlaneEvent) -> None:
        for listener in self._listeners:
            listener.on_ml_event(event)

    def subscribe(self, listener: MLPlaneListener) -> None:
        self._listeners.append(listener)

    # ── Aggregation API ────────────────────────────────────────────────────────

    def collect(self, update: GradientUpdate) -> None:
        """
        Raccoglie un GradientUpdate da un client.
        Chiamato per ogni nodo partecipante al round.
        """
        self._round_updates.append(update)
        logger.debug(
            f"Raccolto update da [{update.node_id}] — "
            f"round={update.round_num}, n_samples={update.n_samples}"
        )

    def aggregate(self, round_num: int) -> AggregatedUpdate | None:
        """
        Esegue FedAvg sugli update raccolti per questo round.
        Svuota il buffer interno dopo l'aggregazione.

        Returns:
            AggregatedUpdate con pesi globali, o None se
            il numero di partecipanti è sotto min_participants.
        """
        updates = self._round_updates
        self._round_updates = []

        if len(updates) < self.min_participants:
            logger.warning(
                f"Round {round_num} — partecipanti insufficienti: "
                f"{len(updates)} < {self.min_participants}"
            )
            return None

        # Filtra update senza n_samples valido
        valid = [u for u in updates if u.n_samples and u.n_samples > 0]
        if not valid:
            logger.warning(f"Round {round_num} — nessun update con n_samples valido")
            return None

        total_samples = sum(u.n_samples for u in valid)

        # FedAvg: media pesata per n_samples
        global_weights = self._weighted_average(valid, total_samples)

        # Loss media pesata — pesa solo sui nodi che hanno effettivamente una loss
        loss_pairs = [(u.loss * u.n_samples, u.n_samples) for u in valid if u.loss is not None]
        mean_loss = (
            sum(l for l, _ in loss_pairs) / sum(n for _, n in loss_pairs)
            if loss_pairs else None
        )

        aggregated = AggregatedUpdate(
            round_num=round_num,
            global_weights=global_weights,
            n_participants=len(valid),
            mean_loss=mean_loss,
            metadata={
                "total_samples": total_samples,
                "participant_ids": [u.node_id for u in valid],
                "cluster_ids": list({u.cluster_id for u in valid}),
            },
        )

        # Emetti evento ML Plane → Auditor e IDS osservano l'aggregazione
        self.emit_event(MLPlaneEvent(
            event_type="aggregation",
            purdue_level=3,
            payload=aggregated,
            round_num=round_num,
        ))

        loss_str = f"{mean_loss:.6f}" if mean_loss is not None else "N/A"
        logger.info(
            f"Round {round_num} — FedAvg completato: "
            f"{len(valid)} partecipanti, loss={loss_str}"
        )
        return aggregated

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _weighted_average(
        self,
        updates: list[GradientUpdate],
        total_samples: int,
    ) -> list[torch.Tensor]:
        """
        Calcola la media pesata dei pesi per n_samples.
        Tutti gli update devono avere la stessa struttura di pesi (parametri + buffer BN).
        L'accumulazione avviene in float32 per supportare buffer int (num_batches_tracked);
        il dtype originale viene ripristinato al termine.
        """
        first_weights = updates[0].weights
        orig_dtypes: list[torch.dtype] = [
            (w if isinstance(w, torch.Tensor) else torch.tensor(w)).dtype
            for w in first_weights
        ]
        # Accumulatore in float32 per evitare errori su tensori int
        accumulator: list[torch.Tensor] = [
            torch.zeros_like(
                w if isinstance(w, torch.Tensor) else torch.tensor(w),
                dtype=torch.float32,
            )
            for w in first_weights
        ]

        for update in updates:
            weight = update.n_samples / total_samples
            for i, w in enumerate(update.weights):
                t = (w if isinstance(w, torch.Tensor) else torch.tensor(w)).float()
                accumulator[i] += t * weight

        # Ripristina dtype originali
        return [acc.to(dt) for acc, dt in zip(accumulator, orig_dtypes)]
