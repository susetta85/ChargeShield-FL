# src/ml/autoencoder_trainer.py
# ChargeShield-FL — ML Plane: Autoencoder Trainer
#
# Responsabilità (Purdue L0→L1):
#   - riceve sessioni EV grezze dal layer Dataset
#   - esegue il training loop locale dell'Autoencoder
#   - supporta FedAvg (proximal_mu=0) e FedProx (proximal_mu>0)
#   - emette GradientUpdate verso il ML Plane
#   - non conosce NVFLARE, né Auditor, né IDS

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from core.autoencoder import Autoencoder
from ml.base_ml import (
    AbstractMLModel,
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)

logger = logging.getLogger(__name__)


class AutoencoderTrainer(AbstractMLModel):
    """
    Training locale dell'Autoencoder su sessioni EV (ACN-Data).

    Purdue Level: L0 (Field) → L1 (Control)
    ML Plane role: produce GradientUpdate dopo ogni training locale.

    Parametri da config (nessun hardcoded):
        input_dim     : dimensione input
        lr            : learning rate
        epochs        : epoche di training locale per round FL
        batch_size    : dimensione batch
        proximal_mu   : coefficiente FedProx (0.0 = FedAvg puro)
    """

    CONTINUOUS_FEATURES = [
        "total_energy_kwh",   # kWh erogati (ACN: kWhDelivered)
        "max_power_kw",       # potenza media stimata
        "kwh_requested",      # energia richiesta dall'utente
        "minutes_available",  # minuti disponibili dichiarati
        "hour_of_day",        # ora connessione (0–23) — pattern comportamentale
        "duration_hours",     # durata sessione in ore
    ]

    def __init__(self, config: dict[str, Any], node_id: str, cluster_id: str) -> None:
        self.node_id    = node_id
        self.cluster_id = cluster_id

        input_dim  = config.get("input_dim")
        lr         = config.get("lr")
        epochs     = config.get("epochs")
        batch_size = config.get("batch_size")

        if input_dim is None:
            raise ValueError("config['input_dim'] è obbligatorio")
        if lr is None:
            raise ValueError("config['lr'] è obbligatorio")
        if epochs is None:
            raise ValueError("config['epochs'] è obbligatorio")
        if batch_size is None:
            raise ValueError("config['batch_size'] è obbligatorio")

        self.epochs     = epochs
        self.batch_size = batch_size

        # FedProx: 0.0 = FedAvg puro
        self.proximal_mu: float = config.get("proximal_mu") or 0.0
        self._global_weights: list[Any] | None = None

        self.model     = Autoencoder(input_dim=input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._listeners: list[MLPlaneListener] = []
        self._current_round: int = 0

        logger.info(
            f"AutoencoderTrainer [{node_id}] — device={self.device}, "
            f"input_dim={input_dim}, lr={lr}, epochs={epochs}, "
            f"proximal_mu={self.proximal_mu} "
            f"({'FedProx' if self.proximal_mu > 0 else 'FedAvg'})"
        )

    # ── AbstractMLModel ────────────────────────────────────────────────────────

    def get_weights(self) -> list[Any]:
        """Restituisce i pesi del modello come lista di tensori CPU."""
        return [p.data.clone().cpu() for p in self.model.parameters()]

    def set_weights(self, weights: list[Any]) -> None:
        """
        Carica i pesi globali nel modello locale.
        Salva riferimento per il termine prossimale FedProx.
        """
        with torch.no_grad():
            for param, w in zip(self.model.parameters(), weights):
                param.data.copy_(w.to(self.device))
        # Copia frozen per proximal term
        self._global_weights = [
            w.clone().detach().to(self.device)
            if isinstance(w, torch.Tensor)
            else torch.tensor(w, device=self.device)
            for w in weights
        ]

    def train_step(self, data: Any) -> float:
        """
        Singolo passo di training su un batch.
        Se proximal_mu > 0, aggiunge termine prossimale FedProx:
            loss += (mu/2) * ||w - w_global||²
        """
        self.model.train()
        batch = data.to(self.device)
        self.optimizer.zero_grad()
        reconstructed = self.model(batch)
        loss = self.criterion(reconstructed, batch)

        if self.proximal_mu > 0.0 and self._global_weights is not None:
            proximal_term = torch.tensor(0.0, device=self.device)
            for param, w_global in zip(self.model.parameters(), self._global_weights):
                proximal_term += torch.norm(param - w_global) ** 2
            loss += (self.proximal_mu / 2.0) * proximal_term

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def emit_event(self, event: MLPlaneEvent) -> None:
        """Propaga l'evento a tutti i listener registrati."""
        for listener in self._listeners:
            listener.on_ml_event(event)

    def subscribe(self, listener: MLPlaneListener) -> None:
        """Registra un listener ML Plane (Auditor o IDS)."""
        self._listeners.append(listener)
        logger.debug(f"Listener registrato: {type(listener).__name__}")

    # ── Training API ───────────────────────────────────────────────────────────

    def train_local(
        self,
        sessions: list[dict[str, Any]],
        round_num: int,
    ) -> GradientUpdate:
        """
        Esegue il training locale completo su sessions.
        Restituisce GradientUpdate con pesi, loss, n_samples.
        """
        self._current_round = round_num

        tensor = self._sessions_to_tensor(sessions)
        if tensor is None:
            logger.warning(f"[{self.node_id}] Nessuna sessione valida — skip training")
            return GradientUpdate(
                node_id=self.node_id,
                cluster_id=self.cluster_id,
                round_num=round_num,
                weights=self.get_weights(),
                gradients=None,
                loss=None,
                n_samples=0,
            )

        dataset    = TensorDataset(tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        epoch_losses: list[float] = []
        for epoch in range(self.epochs):
            batch_losses = [self.train_step(batch) for (batch,) in dataloader]
            epoch_losses.append(sum(batch_losses) / len(batch_losses))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(
            f"[{self.node_id}] Round {round_num} — "
            f"loss={mean_loss:.6f}, n={len(sessions)}"
        )

        update = GradientUpdate(
            node_id=self.node_id,
            cluster_id=self.cluster_id,
            round_num=round_num,
            weights=self.get_weights(),
            gradients=None,
            loss=mean_loss,
            n_samples=len(sessions),
            metadata={
                "epochs": self.epochs,
                "device": str(self.device),
                "proximal_mu": self.proximal_mu,
            },
        )

        self.emit_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=1,
            payload=update,
            round_num=round_num,
        ))

        return update

    def apply_global_model(self, aggregated: AggregatedUpdate) -> None:
        """
        Carica il modello globale ricevuto dal server.
        Emette evento weight_download sul ML Plane.
        """
        self.set_weights(aggregated.global_weights)
        self.emit_event(MLPlaneEvent(
            event_type="weight_download",
            purdue_level=1,
            payload=aggregated,
            round_num=aggregated.round_num,
        ))
        logger.info(f"[{self.node_id}] Modello globale applicato — round {aggregated.round_num}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _sessions_to_tensor(
        self,
        sessions: list[dict[str, Any]],
    ) -> torch.Tensor | None:
        """
        Converte sessioni EV in tensore float32.
        Scarta sessioni con valori None nelle feature continue.
        """
        rows: list[list[float]] = []
        for s in sessions:
            row: list[float] = []
            valid = True
            for feat in self.CONTINUOUS_FEATURES:
                val = s.get(feat)
                if val is None:
                    valid = False
                    break
                try:
                    row.append(float(val))
                except (TypeError, ValueError):
                    valid = False
                    break
            if valid:
                rows.append(row)

        return torch.tensor(rows, dtype=torch.float32) if rows else None
