# src/core/base_node.py
#
# STATO (2026-07-24, review indipendente round 3): non importato da alcuna
# pipeline attiva — vedi la nota completa in src/adapters/ocpp16_adapter.py.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class NodeConfig:
    node_id: str
    cluster_id: str
    location: str


class AbstractChargingNode(ABC):
    """
    Base contract for any charging node in the network.
    Core knows nothing about protocols, datasets, or FL.
    """

    def __init__(self, config: NodeConfig):
        self.config = config

    @abstractmethod
    def collect_data(self) -> dict[str, Any]:
        """Collect raw telemetry from the charging station."""
        ...

    @abstractmethod
    def preprocess(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Apply local preprocessing before FL training."""
        ...

    @abstractmethod
    def get_status(self) -> str:
        """Return current node status (online/offline/error)."""
        ...
