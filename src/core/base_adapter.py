# src/core/base_adapter.py
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
