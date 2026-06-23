# src/adapters/ocpp16_adapter.py
import json
from datetime import datetime, timezone
from typing import Any

from src.core.base_adapter import AbstractProtocolAdapter


class OCPP16Adapter(AbstractProtocolAdapter):
    """
    Protocol adapter for OCPP 1.6 (JSON over WebSocket).
    Knows about OCPP, but nothing about FL or datasets.
    """

    PROTOCOL_NAME = "OCPP_16"

    def encode(self, data: dict[str, Any]) -> bytes:
        """
        Wraps telemetry into an OCPP 1.6 MeterValues message.
        Format: [MessageTypeId, UniqueId, Action, Payload]
        """
        message = [
            2,
            data.get("transaction_id", "unknown"),
            "MeterValues",
            {
                "connectorId": 1,
                "transactionId": data.get("transaction_id"),
                "nodeId": data.get("node_id"),
                "clusterId": data.get("cluster_id"),
                "sessionId": data.get("session_id"),
                "meterValue": [
                    {
                        "timestamp": data.get("timestamp",
                                    datetime.now(timezone.utc).isoformat()),
                        "sampledValue": [
                            {"value": str(data.get("voltage_v", 0)),
                             "measurand": "Voltage", "unit": "V"},
                            {"value": str(data.get("current_a", 0)),
                             "measurand": "Current.Import", "unit": "A"},
                            {"value": str(data.get("power_kw", 0)),
                             "measurand": "Power.Active.Import", "unit": "kW"},
                            {"value": str(data.get("energy_kwh", 0)),
                             "measurand": "Energy.Active.Import.Register",
                             "unit": "kWh"},
                            {"value": str(data.get("soc_percent", 0)),
                             "measurand": "SoC", "unit": "Percent"},
                            {"value": str(data.get("temperature_c", 0)),
                             "measurand": "Temperature", "unit": "Celsius"},
                            {"value": str(data.get("charging_mode", "AC")),
                             "measurand": "ChargingMode", "unit": ""},
                            {"value": str(data.get("error_code", "")),
                             "measurand": "ErrorCode", "unit": ""},
                        ],
                    }
                ],
            },
        ]
        return json.dumps(message).encode("utf-8")

    def decode(self, raw: bytes) -> dict[str, Any]:
        """
        Parse an OCPP 1.6 MeterValues message back into a standard dict.
        """
        message = json.loads(raw.decode("utf-8"))
        payload = message[3]
        sampled = payload["meterValue"][0]["sampledValue"]

        result: dict[str, Any] = {
            "transaction_id": payload.get("transactionId"),
            "timestamp": payload["meterValue"][0]["timestamp"],
            "node_id" :  payload.get("nodeId"),
            "cluster_id" :  payload.get("clusterId"),
            "session_id" : payload.get("sessionId")
        }

        measurand_map = {
             "Voltage": "voltage_v",
             "Current.Import": "current_a",
             "Power.Active.Import": "power_kw",
             "Energy.Active.Import.Register": "energy_kwh",
             "SoC": "soc_percent",
             "Temperature": "temperature_c",
             "ChargingMode": "charging_mode",
             "ErrorCode": "error_code",
        }
        for sv in sampled:
            key = measurand_map.get(sv["measurand"])
            if key:
                if sv["measurand"] in ("ChargingMode", "ErrorCode"):
                    result[key] = sv["value"] if sv["value"] != "None" else None
                else:
                    result[key] = float(sv["value"])      
        return result

    def get_protocol_name(self) -> str:
        return self.PROTOCOL_NAME
