# ChargeShield-FL — Protocol Adapters

## Ruolo

Gli adapter di protocollo traducono i dati standardizzati del framework
nel formato wire del protocollo specifico di ogni cluster.

Seguono il principio architetturale fondamentale:
**ChargingNode non conosce i protocolli** — delega all'adapter.

---

## Adapter implementati

### OCPP16Adapter
**File:** `src/adapters/ocpp16_adapter.py`
**Cluster:** Highway (OCPP 1.6), Urban (OCPP 1.6)

Open Charge Point Protocol 1.6 — standard de facto per
colonnine di ricarica EV su WebSocket con JSON.

**Feature trasmesse:**
voltage_v, current_a, power_kw, energy_kwh,

soc_percent, temperature_c, charging_mode,

error_code, node_id, cluster_id, session_id,

transaction_id, timestamp
**Formato messaggio:**
```json
[2, "transaction_id", "MeterValues", {
    "connectorId": 1,
    "nodeId": "highway-01",
    "clusterId": "highway",
    "sessionId": "...",
    "meterValue": [{
        "timestamp": "...",
        "sampledValue": [
            {"value": "230.0", "measurand": "Voltage", "unit": "V"},
            ...
        ]
    }]
}]
```

---

## Adapter pianificati (Sprint 6)

### OCPP20Adapter
**Cluster:** Corporate (OCPP 2.0.1)
OCPP 2.0.1 aggiunge supporto nativo per smart charging,
V2G (Vehicle-to-Grid) e sicurezza avanzata.

### MQTTAdapter
**Cluster:** Residential (MQTT v5)
MQTT v5 è usato nei contesti residenziali per la sua
leggerezza e il supporto a connessioni intermittenti.
Broker: Eclipse Mosquitto 2.0

---

## Aggiungere un nuovo adapter

1. Estendi `AbstractProtocolAdapter` in `src/core/base_adapter.py`
2. Implementa `encode()`, `decode()`, `get_protocol_name()`
3. Aggiungi il protocollo in `config/protocols.yaml`
4. Aggiorna `containerlab/topology.clab.yml` con il nuovo adapter
5. Scrivi i test in `tests/`

```python
class MyProtocolAdapter(AbstractProtocolAdapter):
    def encode(self, data: dict) -> bytes:
        ...
    def decode(self, raw: bytes) -> dict:
        ...
    def get_protocol_name(self) -> str:
        return "MY_PROTOCOL_v1"
```

---

## Riferimenti

- OCPP 1.6: https://www.openchargealliance.org
- OCPP 2.0.1: https://www.openchargealliance.org
- MQTT v5: https://mqtt.org
