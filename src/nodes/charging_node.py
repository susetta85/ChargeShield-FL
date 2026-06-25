# src/nodes/charging_node.py
import yaml
from pathlib import Path
from typing import Any

from core.base_node import AbstractChargingNode, NodeConfig
from core.base_adapter import AbstractProtocolAdapter

import uuid 
from datetime import datetime, timezone

class ChargingNode(AbstractChargingNode):
    """
    Concrete charging node.
    Knows about protocol adapter, but not about datasets or FL.
    """

    def __init__(self, config: NodeConfig, adapter: AbstractProtocolAdapter):
        super().__init__(config)
        self.adapter = adapter
        self._status = "online"

    @classmethod
    def from_yaml(cls, node_id: str, adapter: AbstractProtocolAdapter,
                  config_path: str = "config/nodes.yaml") -> "ChargingNode":
        """Build a ChargingNode by reading config from nodes.yaml."""
        with open(config_path) as f:
            all_nodes = yaml.safe_load(f)["nodes"]

        node_data = next((n for n in all_nodes if n["id"] == node_id), None)
        if node_data is None:
            raise ValueError(f"Node '{node_id}' not found in {config_path}")

        config = NodeConfig(
            node_id=node_data["id"],
            cluster_id=node_data["cluster"],
            location=node_data["location"],
        )
        return cls(config, adapter)

    def collect_data(self) -> dict[str, Any]:
        """Simulate telemetry collection from the charging station."""
        return {
            "node_id": self.config.node_id,
            "cluster_id": self.config.cluster_id,
            "session_id": str(uuid.uuid4()),
            "transaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "voltage_v": 230.0,
            "current_a": 16.0,
            "power_kw": 3.68,
            "energy_kwh": 0.0,
            "temperature_c": 25.0,
            "soc_percent": 80.0,
            "charging_mode": "AC",
            "error_code": None,
            "status": self._status,
        }

    def preprocess(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize and clean raw telemetry."""
        return {
            "node_id": raw_data["node_id"],
            "cluster_id": raw_data["cluster_id"],
            "session_id": raw_data["session_id"],
            "transaction_id": raw_data["transaction_id"],
            "timestamp": raw_data["timestamp"],
            "voltage_v": float(raw_data["voltage_v"]),
            "current_a": float(raw_data["current_a"]),
            "power_kw": float(raw_data["power_kw"]),
            "energy_kwh": float(raw_data["energy_kwh"]),
            "temperature_c": float(raw_data["temperature_c"]),
            "soc_percent": float(raw_data["soc_percent"]),
            "charging_mode": str(raw_data["charging_mode"]),
            "error_code": raw_data.get("error_code"),
        }

    def get_status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        allowed = {"online", "offline", "error"}
        if status not in allowed:
            raise ValueError(f"Invalid status '{status}'. Allowed: {allowed}")
        self._status = status
