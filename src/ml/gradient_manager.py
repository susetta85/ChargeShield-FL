# src/ml/gradient_manager.py
# ChargeShield-FL — ML Plane: Gradient Manager
#
# Responsabilità (Purdue L1→L2):
#   - riceve GradientUpdate da AutoencoderTrainer
#   - applica gradient clipping (norma L2)
#   - applica Gaussian noise (Differential Privacy)
#   - emette GradientUpdate privatizzato verso FedAvgAggregator
#   - non conosce NVFLARE, né Auditor, né IDS

from __future__ import annotations

import logging
import math
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


class GradientManager(AbstractMLModel):
    """
    Applica Differential Privacy (Gaussian Mechanism) ai pesi locali
    prima che escano dal nodo verso l'aggregatore.

    Purdue Level: L1 (Control) → L2 (Supervisory)
    ML Plane role: privatizza GradientUpdate, emette evento DP.

    Parametri da config (nessun hardcoded):
        epsilon      : budget privacy
        delta        : probabilità fallimento DP
        max_grad_norm: soglia clipping norma L2
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: dizionario con epsilon, delta, max_grad_norm
        """
        epsilon       = config.get("epsilon")
        delta         = config.get("delta")
        max_grad_norm = config.get("max_grad_norm")

        if epsilon is None:
            raise ValueError("config['epsilon'] è obbligatorio")
        if delta is None:
            raise ValueError("config['delta'] è obbligatorio")
        if max_grad_norm is None:
            raise ValueError("config['max_grad_norm'] è obbligatorio")

        self.epsilon       = epsilon
        self.delta         = delta
        self.max_grad_norm = max_grad_norm

        # Sigma per Gaussian Mechanism: σ = max_grad_norm * sqrt(2*ln(1.25/δ)) / ε
        self.sigma = self._compute_sigma()

        self._listeners: list[MLPlaneListener] = []

        logger.info(
            f"GradientManager — ε={epsilon}, δ={delta}, "
            f"max_norm={max_grad_norm}, σ={self.sigma:.4f}"
        )

    # ── AbstractMLModel ────────────────────────────────────────────────────────

    def get_weights(self) -> list[Any]:
        raise NotImplementedError("GradientManager non possiede un modello diretto.")

    def set_weights(self, weights: list[Any]) -> None:
        raise NotImplementedError("GradientManager non possiede un modello diretto.")

    def train_step(self, data: Any) -> float:
        raise NotImplementedError("GradientManager non esegue training.")

    def emit_event(self, event: MLPlaneEvent) -> None:
        for listener in self._listeners:
            listener.on_ml_event(event)

    def subscribe(self, listener: MLPlaneListener) -> None:
        self._listeners.append(listener)

    # ── DP API ────────────────────────────────────────────────────────────────

    def privatize(self, update: GradientUpdate) -> GradientUpdate:
        """
        Applica gradient clipping + Gaussian noise ai pesi del GradientUpdate.

        Args:
            update: GradientUpdate grezzo da AutoencoderTrainer

        Returns:
            GradientUpdate con pesi privatizzati (DP garantita)
        """
        if not update.weights:
            logger.warning(f"[{update.node_id}] Pesi vuoti — skip DP")
            return update

        # Step 1: clip norma L2 globale
        clipped = self._clip_weights(update.weights)

        # Step 2: aggiungi rumore Gaussiano
        noised = self._add_noise(clipped)

        privatized = GradientUpdate(
            node_id=update.node_id,
            cluster_id=update.cluster_id,
            round_num=update.round_num,
            weights=noised,
            gradients=None,
            loss=update.loss,
            n_samples=update.n_samples,
            metadata={
                **update.metadata,
                "dp_applied": True,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "sigma": self.sigma,
                "max_grad_norm": self.max_grad_norm,
            },
        )

        # Emetti evento ML Plane
        self.emit_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=2,
            payload=privatized,
            round_num=update.round_num,
            metadata={"dp_applied": True},
        ))

        logger.debug(
            f"[{update.node_id}] DP applicato — "
            f"σ={self.sigma:.4f}, norm_clip={self.max_grad_norm}"
        )

        return privatized

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_sigma(self) -> float:
        """
        Calcola σ per il Gaussian Mechanism.
        σ = max_grad_norm * sqrt(2 * ln(1.25 / δ)) / ε
        """
        return (
            self.max_grad_norm
            * math.sqrt(2 * math.log(1.25 / self.delta))
            / self.epsilon
        )

    def _clip_weights(self, weights: list[Any]) -> list[torch.Tensor]:
        """
        Clippa la norma L2 globale dei pesi a max_grad_norm.
        Equivalente al gradient clipping per-sample di DP-SGD,
        applicato qui ai pesi aggregati del round locale.
        """
        tensors = [w if isinstance(w, torch.Tensor) else torch.tensor(w)
                   for w in weights]

        # Norma L2 globale su tutti i pesi concatenati
        flat   = torch.cat([t.flatten() for t in tensors])
        norm   = torch.norm(flat, p=2)
        factor = min(1.0, self.max_grad_norm / (float(norm) + 1e-8))

        return [t * factor for t in tensors]

    def _add_noise(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Aggiunge rumore Gaussiano N(0, σ²) a ogni tensore.
        """
        return [
            w + torch.randn_like(w) * self.sigma
            for w in weights
        ]
