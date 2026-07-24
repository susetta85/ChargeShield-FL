# src/core/base_adapter.py
#
# STATO (2026-07-24, review indipendente round 3): non importato da alcuna
# pipeline attiva — vedi la nota completa in src/adapters/ocpp16_adapter.py.
from abc import ABC, abstractmethod
from typing import Any


class AbstractProtocolAdapter(ABC):
    """
    Translates raw node data into a protocol-specific format.
    Core knows nothing about OCPP, MQTT, or any specific protocol.
    """

    @abstractmethod
    def encode(self, data: dict[str, Any]) -> bytes:
        """Serialize data into the protocol wire format."""
        ...

    @abstractmethod
    def decode(self, raw: bytes) -> dict[str, Any]:
        """Deserialize protocol bytes into a standard dict."""
        ...

    @abstractmethod
    def get_protocol_name(self) -> str:
        """Return protocol identifier, e.g. 'OCPP_16', 'MQTT_v5'."""
        ...
