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
        # NOTA: questo modulo implementa weight perturbation (rumore sui pesi
        # aggregati post-FedAvg), NON DP-SGD (rumore sui gradienti per-campione
        # durante il training). I bound di privacy sono diversi:
        # - DP-SGD: garanzia per ogni campione nel training set
        # - Weight perturbation: garanzia sul modello globale aggregato
        # Documentare questa distinzione nella sezione Privacy del paper.
        # NOTA composizione: sigma è calcolato per il Gaussian Mechanism a singolo round.
        # Con T round, la garanzia degrada. Usare RDP/zCDP per composizione formale.

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
        # ATTENZIONE: questo implementa weight perturbation, non DP-SGD.
# Con epochs > 1 per round, la sensitività del vettore pesi a un singolo
# campione NON è formalmente bounded da max_grad_norm.
# La garanzia (epsilon, delta)-DP formale vale solo per epochs=1.
# Per il paper: descrivere sigma come parametro di rumore sperimentale,
# non come garanzia DP formale. Il budget totale su T round è ~T*epsilon
# (composizione naive) — riportarlo nelle tabelle dei risultati.
        """
        Calcola σ per il Gaussian Mechanism.
        σ = max_grad_norm * sqrt(2 * ln(1.25 / δ)) / ε
        """
        if self.epsilon <= 0:
            raise ValueError(f"epsilon deve essere > 0, ricevuto: {self.epsilon}")
        if not (0 < self.delta < 1.25):
            raise ValueError(f"delta deve essere in (0, 1.25), ricevuto: {self.delta}")
        if self.delta > 1e-2:
            logger.warning(
                f"delta={self.delta} è insolitamente alto per DP — "
                "valori tipici sono 1e-5 o inferiori. Verificare la configurazione."
            )
        return (
            self.max_grad_norm
            * math.sqrt(2 * math.log(1.25 / self.delta))
            / self.epsilon
        )

    def _clip_weights(self, weights: list[Any]) -> list[torch.Tensor]:
        """
        Clippa la norma L2 globale dei pesi a max_grad_norm.
        Applicato ai pesi aggregati del round locale (weight perturbation).
        I buffer BatchNorm int64 (num_batches_tracked) vengono esclusi dal
        clipping e restituiti invariati: non sono parametri DP-perturbabili.
        """
        tensors = [w if isinstance(w, torch.Tensor) else torch.tensor(w)
                   for w in weights]

        # Separa floating-point (perturbabili) da int (buffer BN invariati)
        float_tensors = [t for t in tensors if t.is_floating_point()]
        int_masks      = [not t.is_floating_point() for t in tensors]

        if not float_tensors:
            return tensors

        # Norma L2 globale su tutti i pesi float concatenati
        flat   = torch.cat([t.flatten().float() for t in float_tensors])
        norm   = torch.norm(flat, p=2)
        factor = min(1.0, self.max_grad_norm / (float(norm) + 1e-8))

        result: list[torch.Tensor] = []
        float_iter = iter(t * factor for t in float_tensors)
        int_iter   = iter(t for t in tensors if not t.is_floating_point())
        for is_int in int_masks:
            result.append(next(int_iter) if is_int else next(float_iter))
        return result

    def _add_noise(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Aggiunge rumore Gaussiano N(0, σ²) solo ai tensori floating-point.
        I buffer BatchNorm int64 (num_batches_tracked) vengono restituiti invariati:
        torch.randn_like() su int64 solleva RuntimeError in PyTorch ≥2.0.
        """
        return [
            w + torch.randn_like(w) * self.sigma if w.is_floating_point() else w
            for w in weights
        ]
