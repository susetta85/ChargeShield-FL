# src/ml/base_ml.py
# ChargeShield-FL — ML Plane: Abstract Base
#
# AbstractMLModel definisce l'interfaccia del ML Plane.
# È il contratto che AutoencoderTrainer, GradientManager
# e FedAvgAggregator devono rispettare.
#
# Il ML Plane è trasversale L0→L3/L4 del modello Purdue:
# cattura il traffico ML (gradienti, pesi, metadati) che
# il Purdue Model non contempla.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class GradientUpdate:
    """
    Rappresenta un aggiornamento gradiente da un singolo nodo FL.
    Catturato dal ML Plane al confine L1→L2 (Control→Supervisory).
    """
    node_id: str                        # identificatore nodo sorgente
    cluster_id: str                     # cluster di appartenenza
    round_num: int                      # round FL corrente
    weights: list[Any]                  # pesi del modello locale
    gradients: list[Any] | None         # gradienti (None se non disponibili)
    loss: float | None                  # loss locale dopo training
    n_samples: int | None               # numero campioni usati nel training
    metadata: dict[str, Any] = field(default_factory=dict)  # dati aggiuntivi


@dataclass
class AggregatedUpdate:
    """
    Rappresenta il modello globale aggregato dal server.
    Catturato dal ML Plane al confine L2→L3 (Supervisory→Operations).
    """
    round_num: int
    global_weights: list[Any]           # pesi globali post-FedAvg
    n_participants: int                 # nodi partecipanti al round
    mean_loss: float | None             # loss media tra i partecipanti
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MLPlaneEvent:
    """
    Evento emesso dal ML Plane ogni volta che cattura un'interazione.
    Consumato da PrivacyAuditor (FedMIA) e ChargingIDS.
    """
    event_type: str                     # "gradient_upload" | "aggregation" | "weight_download"
    purdue_level: int                   # 0=Field, 1=Control, 2=Supervisory, 3=Operations
    payload: GradientUpdate | AggregatedUpdate | None
    round_num: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Abstract Base ──────────────────────────────────────────────────────────────

class AbstractMLModel(ABC):
    """
    Interfaccia astratta del ML Plane.

    Ogni componente del piano ML (trainer, aggregatore, gradient manager)
    eredita da questa classe e implementa i metodi astratti.
    Non conosce NVFLARE, OCPP, né il layer Auditor/IDS.
    """

    @abstractmethod
    def get_weights(self) -> list[Any]:
        """
        Restituisce i pesi correnti del modello.
        Usato da FedAvg per raccogliere gli update dai client.
        """

    @abstractmethod
    def set_weights(self, weights: list[Any]) -> None:
        """
        Carica pesi nel modello (es. global model dal server).
        Usato dopo ogni round di aggregazione.
        """

    @abstractmethod
    def train_step(self, data: Any) -> float:
        """
        Esegue un passo di training locale.
        Restituisce la loss del batch.
        """

    @abstractmethod
    def emit_event(self, event: MLPlaneEvent) -> None:
        """
        Emette un evento ML Plane verso i listener (Auditor, IDS).
        Implementazione concreta registra i listener via subscribe().
        """

    def subscribe(self, listener: "MLPlaneListener") -> None:
        """
        Registra un listener per gli eventi ML Plane.
        Default: no-op (le sottoclassi possono sovrascrivere).
        """


class MLPlaneListener(ABC):
    """
    Interfaccia per chi vuole osservare il ML Plane.
    Implementata da PrivacyAuditor e ChargingIDS.
    """

    @abstractmethod
    def on_ml_event(self, event: MLPlaneEvent) -> None:
        """Chiamato dal ML Plane ad ogni evento catturato."""
