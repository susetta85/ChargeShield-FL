# Protocol Adapters in ChargeShield-FL: Design Rationale, Implementation, and Security Architecture

**ChargeShield-FL Technical Documentation Series**
**Document Version:** 1.0
**Target Venue:** IEEE/IFIP International Conference on Dependable Systems and Networks (DSN 2027)
**Classification:** Public Research Documentation

---

## Abstract

ChargeShield-FL is a research framework designed to evaluate Membership Inference Attacks (MIA) against Federated Learning (FL) systems deployed within Electric Vehicle (EV) charging infrastructure. A foundational engineering challenge in this domain is the heterogeneity of industrial communication protocols employed across different deployment contexts — highway fast-charging corridors, urban charging clusters, corporate depot installations, and residential wallboxes operate under distinct protocol regimes shaped by hardware generations, bandwidth constraints, security requirements, and regulatory mandates. This document provides a comprehensive treatment of the protocol abstraction architecture adopted in ChargeShield-FL, covering the Adapter design pattern, the rationale for each supported protocol (OCPP 1.6, OCPP 2.0.1, MQTT v5), the VPN and mutual TLS infrastructure, the network emulation environment, and the NVFLARE-based federated learning communication plane. Every design decision is motivated by reference to real-world EV infrastructure constraints, industrial cybersecurity standards, and software engineering principles.

---

## Table of Contents

1. Introduction: Protocol Heterogeneity in Operational Technology Environments
2. The Adapter Pattern: AbstractProtocolAdapter
3. Why Multiple Protocols Reflect Industrial Reality
4. OCPP: History, Architecture, and Deployment Rationale
5. MQTT: Lightweight Messaging for Constrained Residential Devices
6. WireGuard VPN: Securing the Federated Learning Plane
7. Mutual TLS: Zero-Trust Node Authentication
8. Containerlab: Reproducible Network Topology Emulation
9. NVFLARE: Production-Grade Federated Learning Communications
10. Strategy Pattern: Interchangeable Attack Plugins
11. Observer Pattern: ML Plane Listener and Privacy Auditing
12. Adding a New Protocol Adapter: A Practitioner's Guide
13. References

---

## 1. Introduction: Protocol Heterogeneity in Operational Technology Environments

### 1.1 Defining the OT/IT Divide

The distinction between Operational Technology (OT) and Information Technology (IT) environments is foundational to understanding why protocol abstraction is non-trivial in industrial systems. IT environments — enterprise networks, cloud infrastructure, web services — are characterized by relatively homogeneous protocol stacks (TCP/IP, HTTP/S, REST, gRPC), short equipment replacement cycles (three to five years), and a primary concern with data confidentiality and integrity. OT environments, by contrast, encompass the hardware and software systems that monitor and control physical processes: industrial control systems (ICS), supervisory control and data acquisition (SCADA) platforms, programmable logic controllers (PLCs), and the communication buses that interconnect them.

OT environments exhibit properties that are qualitatively different from IT environments and that profoundly constrain system design:

**Equipment longevity.** OT hardware is often deployed for decades. A charging station installed in 2015 running OCPP 1.5 or 1.6 firmware may remain in service until 2030 or beyond. Unlike a web server that can be patched or replaced overnight, a physical charging unit at a highway rest stop requires a maintenance window, a field engineer, and potentially a firmware update that the vendor may no longer support.

**Real-time and safety constraints.** Many OT protocols were designed under assumptions of deterministic, low-latency communication. Introducing additional protocol layers or translation proxies risks adding jitter that is unacceptable in safety-critical control loops. While EV charging control loops are not as hard-real-time as, say, turbine control, the transaction integrity and metering accuracy requirements of EV charging are governed by legal metrology standards (e.g., OIML R 46) that impose their own timing and reliability constraints.

**Heterogeneous vendor ecosystems.** The EV charging market comprises hundreds of hardware manufacturers and dozens of software vendors, each with partial compliance to published standards and significant proprietary extensions. A charging network operator managing thousands of sites will inevitably encounter devices from multiple vendors across multiple protocol generations.

**Air-gapped and intermittently connected deployments.** Residential and rural charging locations may operate behind NAT, on cellular connections with intermittent availability, or in environments where persistent TCP connections are impractical. Protocols designed for always-on enterprise networks perform poorly or not at all in these conditions.

**Regulatory and certification requirements.** OT deployments in critical infrastructure sectors — of which EV charging is increasingly considered one — are subject to sector-specific cybersecurity regulations. In the European Union, the Network and Information Security Directive 2 (NIS2) and the forthcoming Cyber Resilience Act impose obligations on operators and manufacturers that shape protocol and security architecture choices.

### 1.2 EV Charging Infrastructure as a Multi-Protocol OT Environment

EV charging infrastructure is a particularly instructive example of OT heterogeneity because it spans multiple protocol generations within a single logical network. A national charging network operator today will simultaneously manage:

- **Legacy OCPP 1.5/1.6 stations** constituting the bulk of the installed base, procured between 2012 and 2022, communicating over WebSocket/JSON.
- **Newer OCPP 2.0.1 stations** procured from 2021 onward, incorporating native security profiles, ISO 15118 Plug & Charge integration, and advanced smart charging features.
- **Residential wallboxes** using MQTT or proprietary home automation protocols to integrate with smart home energy management systems (HEMS) and vehicle-to-home (V2H) applications.
- **Proprietary OEM-specific protocols** used by automotive manufacturers to communicate directly with their vehicle fleet's associated charging infrastructure.

This heterogeneity is not a temporary transitional state that will resolve itself when everyone upgrades to OCPP 2.0.1. The long asset lifetimes of OT equipment mean that protocol heterogeneity is a permanent feature of large-scale EV charging networks. Any research framework that claims to model realistic EV charging FL deployments must therefore accommodate multiple protocols simultaneously.

### 1.3 Why Protocol Abstraction Matters for FL Security Research

ChargeShield-FL is designed to evaluate MIA against FL systems operating within this heterogeneous OT environment. The choice of communication protocol has direct implications for the attack surface:

- **OCPP 1.6** lacks native transport security, making gradient updates transmitted over the same infrastructure vulnerable to interception if the transport layer is not separately secured.
- **MQTT** pub/sub semantics create a shared topic namespace that an adversarial broker or a compromised subscriber could exploit to observe gradient traffic alongside charging telemetry.
- **OCPP 2.0.1** Security Profile 3 (client certificate + TLS) significantly raises the bar for network-level eavesdropping, but introduces new attack surfaces around certificate management and revocation.

A research framework that hardcodes a single protocol cannot evaluate how these protocol-specific security properties interact with FL privacy guarantees. ChargeShield-FL therefore implements a clean protocol abstraction layer that allows the same FL experiment to be conducted across different protocol substrates, enabling controlled comparison of the privacy and security implications of each choice.

---

## 2. The Adapter Pattern: AbstractProtocolAdapter

### 2.1 Motivation: Why ChargingNode Must Be Protocol-Agnostic

A central software engineering principle in ChargeShield-FL is that the `ChargingNode` class — which encapsulates the behavior of a charging station participant in the FL system — must not contain any knowledge of the specific communication protocol it is using. This is a direct application of the Dependency Inversion Principle from SOLID object-oriented design: high-level modules (the FL participant logic in `ChargingNode`) should depend on abstractions, not on concretions (specific protocol implementations).

The reasons for this design choice are threefold:

**Separation of concerns.** The FL participation logic — receiving model updates, performing local training on charging data, computing gradients, and reporting metrics — is entirely independent of whether those activities are communicated over OCPP WebSocket frames, MQTT publish messages, or gRPC calls. Conflating these concerns would mean that a change in protocol (e.g., upgrading a cluster from OCPP 1.6 to OCPP 2.0.1) would require modifying the FL participation code, violating the Open/Closed Principle.

**Testability.** Protocol-specific code involves external dependencies: network sockets, broker connections, TLS certificates. These are difficult to unit-test in isolation. By depending on an abstract interface, `ChargingNode` can be tested with a mock adapter that simulates protocol behavior without requiring actual network infrastructure.

**Extensibility.** The EV charging protocol landscape is not static. ISO 15118-20 (supporting bidirectional power flow for V2G), OCPP 2.1 (currently in draft), and emerging smart grid protocols will require integration over time. By defining a stable adapter interface, ChargeShield-FL can incorporate new protocols by implementing a new adapter class, without modifying any existing code.

### 2.2 The AbstractProtocolAdapter Interface

The adapter interface is defined as a Python abstract base class using the `abc` module. This ensures that any concrete adapter class that does not implement all required methods will raise a `TypeError` at instantiation time, providing a compile-time-equivalent check in a dynamically typed language.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict


class AbstractProtocolAdapter(ABC):
    """
    Abstract base class for all protocol adapters in ChargeShield-FL.

    A protocol adapter encapsulates the serialization and deserialization
    logic for a specific charging communication protocol. Concrete subclasses
    implement encode() and decode() to translate between the internal
    ChargeShield-FL message format (a typed Python dictionary) and the
    wire format of the target protocol.

    All adapter implementations must be stateless with respect to FL round
    logic. Protocol-level state (e.g., WebSocket connection handles, MQTT
    client objects) may be maintained internally, but must not affect the
    semantics of encode() or decode().
    """

    @abstractmethod
    def encode(self, message: Dict[str, Any]) -> bytes:
        """
        Serialize an internal message dictionary to the protocol wire format.

        Parameters
        ----------
        message : Dict[str, Any]
            A dictionary conforming to the ChargeShield-FL internal message
            schema. Keys include 'message_type', 'node_id', 'payload', and
            optional 'timestamp' and 'sequence_number' fields.

        Returns
        -------
        bytes
            The serialized message in the protocol wire format, ready for
            transmission over the transport layer.

        Raises
        ------
        ProtocolEncodeError
            If the message cannot be encoded due to schema violations or
            protocol-specific constraints.
        """
        ...

    @abstractmethod
    def decode(self, raw: bytes) -> Dict[str, Any]:
        """
        Deserialize a raw wire-format message to the internal message schema.

        Parameters
        ----------
        raw : bytes
            Raw bytes received from the transport layer, in the protocol
            wire format.

        Returns
        -------
        Dict[str, Any]
            A dictionary conforming to the ChargeShield-FL internal message
            schema.

        Raises
        ------
        ProtocolDecodeError
            If the raw bytes cannot be decoded due to malformed messages,
            unknown message types, or schema violations.
        """
        ...

    @abstractmethod
    def get_protocol_name(self) -> str:
        """
        Return a canonical string identifier for this protocol adapter.

        The returned string is used for logging, metrics labeling, and
        experiment configuration. It must be unique across all registered
        adapters and must be stable across software versions to ensure
        reproducible experiment records.

        Returns
        -------
        str
            A canonical protocol identifier, e.g., 'ocpp16', 'ocpp201',
            'mqtt5', or 'iec61850'.
        """
        ...
```

### 2.3 How ChargingNode Consumes the Adapter

The `ChargingNode` class accepts an `AbstractProtocolAdapter` instance via constructor injection. This is the standard Dependency Injection pattern, which keeps the dependency graph explicit and testable.

```python
class ChargingNode:
    """
    Represents a single EV charging station participating in a FL round.
    """

    def __init__(
        self,
        node_id: str,
        adapter: AbstractProtocolAdapter,
        local_dataset: ChargingDataset,
        fl_client: NVFLAREClientExecutor,
    ) -> None:
        self.node_id = node_id
        self._adapter = adapter
        self._dataset = local_dataset
        self._fl_client = fl_client

    def send_gradient_report(self, gradient_summary: Dict[str, Any]) -> None:
        """
        Encode and transmit a gradient summary using the configured adapter.
        The ChargingNode has no knowledge of the underlying protocol.
        """
        message = {
            "message_type": "gradient_report",
            "node_id": self.node_id,
            "payload": gradient_summary,
            "timestamp": time.time(),
        }
        encoded = self._adapter.encode(message)
        self._transport.send(encoded)
```

The `ChargingNode` never calls `isinstance(self._adapter, OCPP16Adapter)` or branches on protocol type. All protocol-specific logic is entirely encapsulated within the adapter. This design ensures that the FL privacy analysis code — which is the primary research contribution of ChargeShield-FL — operates identically regardless of which protocol is in use, and that protocol-specific behavior can be varied experimentally without touching any FL code.

---

## 3. Why Multiple Protocols Reflect Industrial Reality

The decision to implement three distinct protocol adapters in ChargeShield-FL is not an academic exercise in software architecture. It reflects the empirical reality of large-scale EV charging network deployments.

According to data compiled by CharIN e.V. and the Open Charge Alliance (OCA), OCPP 1.6 remains the dominant charging protocol as of 2025, with an estimated 70–80% of publicly accessible charging stations globally communicating over OCPP 1.6 or its predecessor OCPP 1.5. OCPP 2.0.1 adoption is accelerating, particularly in commercial and fleet charging contexts where the Security Profile 3 mandate from major network operators is driving hardware procurement decisions. MQTT-based residential charging is growing rapidly as Home Energy Management Systems (HEMS), solar integration platforms, and vehicle-to-home (V2H) systems create demand for lightweight pub/sub integration between wallboxes and home automation controllers such as Home Assistant, openHAB, and proprietary OEM platforms.

The **heterogeneity problem** in EV charging FL is therefore not hypothetical. A charging network operator wishing to deploy an FL system to improve charging demand prediction across their entire asset base will face three distinct protocol environments simultaneously. A research framework that evaluates MIA only against a single homogeneous protocol substrate provides an incomplete and potentially misleading picture of the true attack surface. ChargeShield-FL's multi-protocol architecture directly addresses this gap.

Furthermore, the protocol choice influences the threat model in ways that are research-relevant:

- The absence of native payload encryption in OCPP 1.6 means that a network-layer adversary can observe charging telemetry and potentially correlate it with FL gradient updates in ways that amplify MIA effectiveness.
- MQTT's pub/sub semantics mean that an adversarial node with broker access can subscribe to gradient-adjacent topics without the FL server's awareness.
- OCPP 2.0.1 Security Profile 3, combined with the WireGuard overlay described in Section 6, creates a defense-in-depth posture that significantly constrains the adversary's observation capabilities.

Evaluating MIA effectiveness across these different protocol substrates is a primary experimental contribution of ChargeShield-FL.

---

## 4. OCPP: History, Architecture, and Deployment Rationale

### 4.1 History and Standardization

The Open Charge Point Protocol (OCPP) was developed by the E-Laad Foundation in the Netherlands, with the first publicly available version (OCPP 1.2) released in 2010. The protocol was designed to solve a specific and urgent problem: the lack of any standard interface between charging stations (charge points) and charging network management systems (central systems), which had produced a fragmented market where each network operator required proprietary hardware.

The Open Charge Alliance (OCA), a non-profit founded in 2014, assumed stewardship of OCPP and has since governed its development. OCPP 1.5 and 1.6 brought JSON/WebSocket as the primary transport, replacing the earlier SOAP/HTTP binding and dramatically simplifying implementation complexity. OCPP 2.0 (2018) and the corrected OCPP 2.0.1 (2020) introduced a comprehensive redesign incorporating native security profiles, advanced smart charging, ISO 15118 integration, and improved device management capabilities.

OCPP is today recognized as the de facto global standard for open charging station communication, with support mandated by regulation in the European Union (AFIR Regulation, Alternative Fuels Infrastructure Regulation, Annex II) and adopted by charging networks in North America, Asia, and Australia.

### 4.2 OCPP 1.6: Message Format and Transport Architecture

OCPP 1.6 operates over WebSocket (RFC 6455) using JSON message encoding. The choice of WebSocket as the transport was a significant architectural departure from the SOAP/HTTP binding of earlier versions and one that requires explanation.

**WebSocket vs. HTTP polling.** HTTP is a request-response protocol: the client initiates every interaction, and the server responds. In a charging station management context, this creates a fundamental problem: the central system (CSMS) must be able to push commands to the charging station asynchronously — for example, to initiate a remote start transaction, unlock a connector, or update configuration. Under HTTP polling, the charging station must periodically poll the CSMS to discover pending commands. This introduces latency proportional to the polling interval (typically 30–60 seconds in deployed systems), wastes bandwidth on empty poll responses, and creates scalability challenges when tens of thousands of stations are polling simultaneously. WebSocket solves this by establishing a persistent, full-duplex TCP connection over which either party can send messages at any time. The CSMS can push a `RemoteStartTransaction` command to a charging station within milliseconds of the operator action, without the station needing to poll.

**OCPP 1.6 message format.** OCPP 1.6 defines three message types, encoded as JSON arrays:

```json
// CALL: client-to-server or server-to-client request
[2, "unique-message-id", "ActionName", {"key": "value"}]

// CALLRESULT: successful response to a CALL
[3, "unique-message-id", {"key": "value"}]

// CALLERROR: error response to a CALL
[4, "unique-message-id", "ErrorCode", "ErrorDescription", {"details": "..."}]
```

A transaction lifecycle in OCPP 1.6 proceeds as follows:

1. **Authorize**: The charging station sends `[2, "id-001", "Authorize", {"idTag": "RFID-CARD-XYZ"}]` to the CSMS. The CSMS validates the RFID tag against its authorization list and responds `[3, "id-001", {"idTagInfo": {"status": "Accepted"}}]`.
2. **StartTransaction**: Upon connector plug-in and authorization, the station sends a `StartTransaction` CALL with energy meter reading, timestamp, and connector ID.
3. **MeterValues**: During the charging session, the station periodically sends `MeterValues` messages containing sampled measurements (energy, power, voltage, current, state of charge) according to a configurable sampling interval.
4. **StopTransaction**: On session end, the station sends a `StopTransaction` CALL with the final meter reading.

The `MeterValues` message is particularly significant for FL research because it contains the time-series charging behavior data that serves as the training dataset for the demand forecasting model in ChargeShield-FL. The structure and granularity of `MeterValues` data directly determines the information content available to a membership inference adversary.

### 4.3 Security Gap in OCPP 1.6

OCPP 1.6 does not mandate or provide any native transport security mechanism. The specification recommends TLS but leaves its configuration entirely to the implementer. In practice, a significant fraction of deployed OCPP 1.6 infrastructure operates without TLS, particularly in older installations or in environments where the CSMS and charging stations share a private network and operators have assumed (often incorrectly) that network-level isolation is sufficient.

The specific risk for ChargeShield-FL is that the FL gradient reports — which are attached to or correlated with the OCPP `MeterValues` messages in the simulation — could be observed by a passive network adversary. Without TLS, both the charging telemetry and the FL gradient summaries are transmitted in plaintext, providing an adversary with rich information for mounting passive MIA.

ChargeShield-FL addresses this by mandating mutual TLS (mTLS) at the transport layer for all OCPP 1.6 adapters, implemented using the `ssl` module in Python with client certificate verification. This compensates for OCPP 1.6's native security gap. The mTLS architecture is described in detail in Section 7.

### 4.4 OCPP 1.6 for Highway and Urban Charging Clusters

ChargeShield-FL assigns OCPP 1.6 adapters to the **Highway** (150 kW DC fast charging) and **Urban** (22 kW AC charging) cluster configurations. This assignment reflects the empirical reality of the installed base: the overwhelming majority of publicly accessible fast-charging and AC charging stations currently deployed in Europe and North America were procured between 2015 and 2022, when OCPP 1.6 was the current standard. Network operators have contractual and operational relationships with these stations that cannot be dissolved to enforce a protocol upgrade.

Furthermore, the hardware of many deployed OCPP 1.6 stations — particularly older DC fast chargers from manufacturers such as ABB, Tritium, and Efacec — runs embedded Linux or RTOS firmware with constrained computational resources that may not support the cryptographic operations required for OCPP 2.0.1 Security Profile 3 without firmware replacement. In many cases, firmware replacement is not commercially viable, making protocol upgrade effectively impossible without hardware replacement.

ChargeShield-FL models these operational constraints faithfully: the Highway and Urban clusters use OCPP 1.6 with mTLS enforcement at the framework level, simulating the realistic scenario where the transport is secured despite the protocol's native security gap.

### 4.5 OCPP 2.0.1 for Corporate Charging Clusters

ChargeShield-FL assigns an OCPP 2.0.1 adapter to the **Corporate** (50 kW DC) charging cluster configuration. Corporate depot charging — where a fleet operator manages tens to hundreds of charging points at a single site — represents a growing segment of the EV charging market with distinct characteristics that make OCPP 2.0.1 the appropriate choice.

**Native security profiles.** OCPP 2.0.1 defines three security profiles with increasing levels of protection:

- *Security Profile 1*: HTTP Basic Authentication over TLS. Provides server authentication and rudimentary client identity, suitable for low-risk deployments.
- *Security Profile 2*: HTTP Basic Authentication over TLS with client-side certificates for transport authentication. Provides stronger client identity assurance.
- *Security Profile 3*: Client certificate authentication over TLS, with no password-based authentication. Provides the strongest client identity assurance, prevents credential stuffing attacks, and is required by several major charging network operators for new hardware procurement.

ChargeShield-FL's Corporate cluster operates under Security Profile 3, which aligns with the mTLS architecture described in Section 7 and provides the strongest available protection against rogue node attacks in the FL context.

**Smart charging and ISO 15118 integration.** OCPP 2.0.1 includes a comprehensive Smart Charging feature set that allows the CSMS to issue charging schedules with minute-level granularity, enabling demand response integration with grid operators. The ISO 15118 Plug & Charge (PnC) integration allows vehicles to authenticate and authorize charging sessions using certificates stored in the vehicle's secure element, eliminating the need for RFID cards. For a corporate depot operator managing a known fleet of vehicles, PnC provides a seamless user experience and a richer data stream for the FL demand forecasting model.

**V2G support.** OCPP 2.0.1 in conjunction with ISO 15118-20 supports Vehicle-to-Grid (V2G) energy export from vehicle batteries to the grid. V2G sessions generate bidirectional energy flow data that significantly enriches the training dataset for the FL model and creates new privacy considerations: V2G data reveals not only when a vehicle is present and charging, but also its battery state of health and the operator's energy arbitrage strategy, both of which are commercially sensitive.

**Alternatives considered.** The ChargeShield-FL protocol selection considered several alternatives for the Corporate cluster:

- *ISO 15118 direct communication*: ISO 15118 defines a complete communication stack for EV-to-EVSE communication, including Plug & Charge and V2G. However, it is a vehicle-to-charger protocol, not a charger-to-CSMS protocol, and there is no existing FL ecosystem built around it. Implementing a ChargeShield-FL adapter for ISO 15118 would require implementing a full V2G Communication Controller, which is out of scope for the current framework version.
- *Proprietary REST APIs*: Several major charging hardware manufacturers expose proprietary REST APIs for station management. These APIs are not standardized, are subject to change without notice, and create vendor lock-in that is incompatible with ChargeShield-FL's goal of evaluating protocol-agnostic FL deployments.
- *REST/HTTP polling*: A REST-based adapter was prototyped and rejected due to the polling overhead described in Section 4.2. For a Corporate cluster with 50–100 charging stations executing synchronized FL rounds, the polling latency would introduce unacceptable synchronization delays in the FL protocol.

---

## 5. MQTT: Lightweight Messaging for Constrained Residential Devices

### 5.1 Why MQTT for Residential Charging

ChargeShield-FL assigns an MQTT v5 adapter to the **Residential** (7 kW AC) charging cluster. This assignment reflects the distinct operational context of residential wallboxes and the protocols that have organically emerged in the consumer IoT ecosystem in which they are embedded.

A residential wallbox operates in an environment characterized by:

- **Constrained hardware**: Entry-level residential wallboxes may run on microcontrollers with 256 KB of RAM and 1–2 MB of flash storage, leaving minimal headroom for a full WebSocket/JSON OCPP stack.
- **Intermittent connectivity**: Home broadband connections are subject to outages, NAT traversal issues, and dynamic IP assignment. A persistent WebSocket connection to a remote CSMS is fragile in this environment.
- **Smart home integration**: Residential EV charging increasingly needs to integrate with solar inverters, home energy storage systems, smart meters, and home automation controllers. These systems communicate via MQTT, Zigbee, Z-Wave, or Modbus — not OCPP.
- **Low data throughput**: A 7 kW AC wallbox generates significantly less telemetry per unit time than a 150 kW DC fast charger. The overhead of WebSocket framing and OCPP message structure is disproportionate to the payload content.

MQTT (Message Queuing Telemetry Transport) was designed by Andy Stanford-Clark (IBM) and Arlen Nipper (Arcom) in 1999 specifically for low-bandwidth, high-latency, unreliable networks — exactly the environment described above. Its publish/subscribe architecture, minimal packet overhead (fixed header as small as 2 bytes), and support for intermittent connectivity via persistent sessions make it the natural choice for residential charging integration.

### 5.2 MQTT v5 Features Relevant to ChargeShield-FL

ChargeShield-FL uses MQTT version 5.0 (OASIS standard, March 2019), which introduced several features over the widely deployed v3.1.1 that are directly relevant to the FL context:

**Topic aliasing.** In MQTT v5, a client can establish a mapping from a topic alias (a small integer) to a full topic string. Subsequent messages on the same topic can use the alias instead of the full string, reducing per-message overhead. In a residential cluster where a wallbox publishes telemetry at 15-second intervals over a cellular connection, topic aliasing meaningfully reduces bandwidth consumption over the session lifetime.

**Session expiry interval.** The `Session Expiry Interval` property in CONNECT allows a client to specify how long the broker should retain session state (subscriptions, queued messages) after disconnection. A value of 3600 seconds allows a wallbox that loses connectivity for up to one hour to reconnect and receive all queued messages — including any FL round initiation commands — without missing a round. This is critical for FL participation: a residential node that misses a round initiation due to a connectivity outage would produce a biased client selection if the CSMS does not account for it.

**Message expiry interval.** The `Message Expiry Interval` property in PUBLISH allows the sender to specify a maximum time-to-live for a message in the broker's queue. ChargeShield-FL uses this for FL round initiation commands: if a round initiation message is queued for a disconnected node longer than the FL round duration, it should be discarded rather than delivered to a node that is now in a different round context.

**Reason codes.** MQTT v5 provides detailed reason codes for all acknowledgment packets, enabling finer-grained error handling. This is used in ChargeShield-FL's MQTT adapter to distinguish between transient delivery failures (QoS 1 PUBACK with reason code 0x80 "Unspecified error") and permanent failures (QoS 1 PUBACK with reason code 0x87 "Not Authorized"), triggering different retry logic.

**User properties.** MQTT v5's `User Property` field allows arbitrary key-value metadata to be attached to any message. ChargeShield-FL uses User Properties to carry FL round metadata (round number, model version, aggregation server ID) alongside the charging telemetry payload, without requiring changes to the topic structure or payload schema.

### 5.3 MQTT Quality of Service Levels

MQTT defines three Quality of Service levels:

- **QoS 0 (At most once)**: Fire-and-forget delivery. No acknowledgment, no retransmission. Suitable for high-frequency telemetry where occasional loss is acceptable. ChargeShield-FL uses QoS 0 for intermediate MeterValues telemetry.
- **QoS 1 (At least once)**: The publisher retransmits until it receives a PUBACK from the broker. The subscriber may receive duplicates; the application must be idempotent. ChargeShield-FL uses QoS 1 for FL round initiation commands and gradient report acknowledgments.
- **QoS 2 (Exactly once)**: A four-way handshake ensures exactly-once delivery. ChargeShield-FL uses QoS 2 for final transaction records that feed the FL training dataset, where duplicate records would corrupt the local dataset and bias gradient computation.

### 5.4 Broker: Eclipse Mosquitto 2.0

ChargeShield-FL uses Eclipse Mosquitto 2.0 as the MQTT broker for the Residential cluster. The selection of Mosquitto over alternative brokers requires justification.

**Eclipse Mosquitto** is an open-source MQTT broker maintained by the Eclipse Foundation, written in C. It is designed for minimal resource consumption: the base installation requires approximately 1 MB of resident memory, and a typical residential deployment serving ten wallboxes requires less than 10 MB. This footprint is compatible with deployment on a Raspberry Pi or similar edge gateway co-located with a home energy management system.

**Why not EMQX?** EMQX is a high-performance MQTT broker written in Erlang/OTP, designed for large-scale deployments handling millions of concurrent connections. Its horizontal scalability and rich management API make it well-suited for cloud-hosted MQTT infrastructure serving thousands of customers. However, these capabilities come at a cost: EMQX requires a JVM-class hardware profile (4+ GB RAM, multi-core CPU) that is incompatible with the resource-constrained edge gateway scenario ChargeShield-FL models. Furthermore, EMQX's enterprise features (clustering, persistence, rule engine) introduce complexity that is unnecessary for the isolated residential cluster simulation.

**Why not HiveMQ?** HiveMQ is a commercial MQTT broker with open-core licensing. Its primary differentiation is enterprise features (clustering, persistence, compliance with industrial IoT standards) and commercial support. ChargeShield-FL is a research framework with an open-source mandate; dependence on a commercial broker would complicate reproducibility and licensing compliance.

**Why not VerneMQ?** VerneMQ (Erlang-based, maintained by Octavo Labs) shares EMQX's resource profile concerns and has experienced periods of reduced maintenance activity that create long-term reproducibility risk for a research framework.

Mosquitto 2.0 is mature, actively maintained by the Eclipse Foundation, embeds cleanly in Docker containers (the `eclipse-mosquitto:2.0` image is 9.6 MB), supports MQTT v5 fully, and has been the reference broker for MQTT protocol development and testing for over a decade.

### 5.5 MQTT Security Architecture

MQTT's native security model is deliberately minimal: the protocol defines username/password authentication in the CONNECT packet but does not mandate TLS, payload encryption, or message signing. This reflects MQTT's origins in resource-constrained environments where cryptographic operations were computationally prohibitive.

ChargeShield-FL's MQTT adapter enforces TLS 1.3 with client certificate authentication, providing:

- **Transport confidentiality**: All MQTT traffic is encrypted end-to-end between the wallbox and the broker.
- **Server authentication**: The wallbox verifies the broker's certificate against the local CA, preventing man-in-the-middle attacks.
- **Client authentication**: The broker verifies the wallbox's client certificate, preventing unauthorized nodes from connecting.

However, MQTT's lack of native payload encryption means that a compromised broker — whether through software vulnerability or insider threat — has access to all message payloads in plaintext. For ChargeShield-FL's FL gradient reports, this is mitigated by application-level HMAC signing using a per-node key derived during the provisioning process. A compromised broker can observe gradient reports but cannot forge them without access to the signing key.

This residual attack surface — broker compromise exposing gradient payloads — is an intentional design choice in ChargeShield-FL: it represents a realistic threat in residential deployments where the MQTT broker may be a consumer-grade home router or a third-party cloud service, and enables ChargeShield-FL to evaluate the MIA effectiveness of a broker-level adversary.

### 5.6 MQTT vs. OCPP: Architectural Comparison

The fundamental architectural difference between MQTT and OCPP is the communication paradigm:

- **OCPP is RPC-over-WebSocket**: The central system and charging station exchange structured request-response pairs with defined actions, payloads, and acknowledgment semantics. This is a transactional model suited to the stateful charging session lifecycle.
- **MQTT is publish/subscribe**: Publishers emit messages to named topics; subscribers receive messages from topics they have subscribed to, mediated by a broker. Publisher and subscriber are decoupled — they do not know each other's identity or address.

The pub/sub model is superior for residential smart home integration because:

1. A home energy management system, a solar monitoring app, and the ChargeShield-FL FL client can all subscribe to the wallbox's telemetry topic simultaneously, without the wallbox needing to be aware of or manage connections to all three consumers.
2. The wallbox can publish telemetry regardless of whether any subscriber is currently connected, and the broker will queue messages for offline subscribers according to their QoS and session persistence settings.
3. Adding a new consumer of wallbox data (e.g., a new home automation integration) requires only a new broker subscription, not a protocol change on the wallbox.

These properties make MQTT the natural fit for the heterogeneous, loosely coupled smart home environment in which residential wallboxes operate.

---

## 6. WireGuard VPN: Securing the Federated Learning Plane

### 6.1 Why WireGuard over IPSec or OpenVPN

ChargeShield-FL uses WireGuard as the VPN technology for its FL gradient communication overlay network. The selection of WireGuard over the more established alternatives of IPSec and OpenVPN is motivated by several converging technical arguments.

**WireGuard's cryptographic design.** WireGuard employs a fixed, modern cryptographic suite:

- **ChaCha20-Poly1305** for authenticated encryption with associated data (AEAD). ChaCha20 provides stream cipher confidentiality; Poly1305 provides message authentication. This combination is standardized in RFC 8439 and is preferred over AES-GCM in environments where hardware AES acceleration is unavailable, as is common in embedded and ARM-based gateway hardware deployed in EV charging edge infrastructure.
- **Curve25519** (X25519) for Elliptic Curve Diffie-Hellman (ECDH) key exchange. Curve25519 was designed by Bernstein et al. for resistance to side-channel attacks and does not rely on NIST curve parameters, avoiding the concerns raised about potential NIST curve weaknesses following the Dual_EC_DRBG controversy.
- **BLAKE2s** for hashing and key derivation functions. BLAKE2s is faster than SHA-256 on 32-bit processors while providing equivalent security guarantees.
- **SipHash-2-4** for hashtable keys to provide DoS resistance in the implementation.

This fixed cryptographic suite contrasts favorably with IPSec, which supports a large number of algorithm negotiation options (IKEv2 cipher suites include dozens of combinations). The complexity of IPSec negotiation has historically been a source of configuration errors and downgrade attacks; WireGuard eliminates negotiation entirely by fixing the algorithm suite at the protocol level.

**Codebase size.** WireGuard's kernel implementation comprises approximately 4,000 lines of code, compared to approximately 400,000 lines for OpenVPN and a comparable figure for the Linux IPSec stack (strongSwan IKEv2 daemon alone exceeds 300,000 lines). The smaller attack surface of WireGuard's codebase has a direct security implication: fewer lines of code means fewer potential vulnerabilities. Jason Donenfeld's original WireGuard paper (NDSS 2017) explicitly motivates this as a security property. The codebase has been formally verified in parts using the Tamarin prover for the handshake protocol.

**Kernel-space performance.** WireGuard operates as a Linux kernel module, processing packets in kernel space and avoiding the context switches and memory copies that accompany user-space VPN implementations such as OpenVPN. This reduces per-packet latency and CPU overhead, which is important for FL gradient communication: large gradient tensors transmitted over a high-latency VPN introduce round-trip time penalties that directly affect FL convergence speed in simulated experiments.

**Handshake latency and roaming support.** WireGuard's handshake protocol (based on the Noise Protocol Framework) completes in 1-RTT, meaning that a new connection is established in a single round trip. OpenVPN's TLS handshake requires 2-RTT or more. More importantly for the residential deployment scenario, WireGuard supports seamless roaming: if a node's IP address changes (e.g., due to a DHCP lease renewal or cellular network handoff), the WireGuard tunnel migrates transparently without requiring a new handshake. This is a critical property for residential nodes that may connect through consumer broadband with dynamic IP assignment.

**Comparison with IPSec.** IPSec (in tunnel mode with IKEv2) is the traditional enterprise VPN standard and provides equivalent security guarantees to WireGuard when correctly configured. However, its configuration complexity — IKEv2 daemon configuration, certificate or pre-shared key management, NAT traversal with MOBIKE, firewall rule management — makes it operationally challenging in the Containerlab emulation environment. WireGuard's configuration is a simple text file with peer public keys and allowed IP ranges, directly compatible with Containerlab's declarative topology model.

### 6.2 WireGuard's Role in ChargeShield-FL

WireGuard serves a specific and critical function in ChargeShield-FL's architecture: it creates an isolated overlay network — referred to as the "FL backbone" — over which FL gradient updates, model parameters, and round coordination messages are transmitted, completely separate from the OCPP/MQTT operational traffic.

Each charging cluster container (Highway, Urban, Corporate, Residential) runs a WireGuard interface (`wg0`) in addition to its operational network interface (`eth0`). The NVFLARE FL client in each container binds exclusively to the WireGuard interface, ensuring that all FL traffic traverses the encrypted overlay. The OCPP or MQTT adapter binds to the operational interface, ensuring that charging telemetry flows through the operational subnet.

This physical network separation at the interface level provides:

**Security isolation.** An adversary who compromises the operational OCPP network (e.g., by inserting a rogue CSMS or conducting a man-in-the-middle attack on unencrypted OCPP 1.6 traffic) gains no access to the FL gradient traffic, which flows over a different network path protected by WireGuard's cryptographic encapsulation.

**Traffic analysis resistance.** By routing FL traffic over a separate interface with a distinct IP range, ChargeShield-FL makes it structurally difficult for a network-level adversary to correlate FL gradient updates with specific charging sessions. The operational telemetry and the FL updates are visible on different networks and cannot be trivially correlated by traffic timing or volume analysis without access to both interfaces simultaneously.

**IEC 62443 zone separation compliance.** The IEC 62443 series of standards for industrial cybersecurity defines a zone and conduit model in which network segments with different security levels must be separated by controlled interfaces. The WireGuard overlay implements a logical zone separation between the operational technology zone (OCPP/MQTT charging traffic) and the research/ML zone (FL communications), consistent with IEC 62443-3-3 system security requirements for zone-to-zone communication conduits.

---

## 7. Mutual TLS: Zero-Trust Node Authentication

### 7.1 TLS vs. Mutual TLS: The Authentication Gap

Standard TLS (as used in HTTPS) provides one-way authentication: the server presents an X.509 certificate signed by a trusted Certificate Authority (CA), and the client verifies it. The client is not required to present any certificate; its identity is established (if at all) through application-layer mechanisms such as username/password or API keys.

In the context of a Federated Learning system deployed across a charging network, one-way TLS is insufficient. Consider the threat model: an adversary who wishes to inject poisoned gradients into the FL aggregation round need only establish a TLS connection to the NVFLARE server (which any client can do under standard TLS) and claim to be a legitimate charging node. Without client certificate verification, the server has no cryptographic basis for rejecting the rogue connection.

Mutual TLS (mTLS) requires both parties to present and verify X.509 certificates. The FL server presents its certificate (proving to clients that they are connected to the legitimate aggregation server), and each FL client presents a per-node certificate (proving to the server that the client is a provisioned, authorized participant). Connections from nodes that do not present a valid certificate signed by the framework's CA are rejected at the TLS handshake layer, before any application-level data is exchanged.

### 7.2 Certificate Management via make certs

ChargeShield-FL's `Makefile` includes a `make certs` target that generates the complete certificate infrastructure for a configured experiment topology. The process proceeds as follows:

1. **CA generation**: A self-signed root CA certificate and private key are generated using OpenSSL. The CA is local to the ChargeShield-FL experiment instance and is not trusted outside the experiment environment.
2. **Server certificate**: An NVFLARE aggregation server certificate is generated, signed by the local CA, with the server's IP address and hostname in the Subject Alternative Name (SAN) extension.
3. **Per-node certificates**: For each configured charging node (identified by `node_id`), a certificate and private key are generated, signed by the local CA. The certificate's Common Name is set to the `node_id`, providing a cryptographic binding between the X.509 identity and the ChargeShield-FL node identifier.
4. **Certificate bundle distribution**: Certificates are written to the `certs/` directory in a per-node subdirectory structure, ready for distribution to the corresponding Containerlab containers via Docker volume mounts.

This automated certificate generation ensures that each experiment instantiation has a unique, fresh PKI, preventing certificate reuse across experiments that could introduce cross-experiment correlations in security evaluations.

### 7.3 Integration with NVFLARE Provisioning

NVFLARE's provisioning system uses a `project.yml` file to define the FL participant topology. ChargeShield-FL's `project.yml` template maps directly onto the certificate infrastructure generated by `make certs`:

```yaml
api_version: 3
name: chargeshield_fl
participants:
  - name: server
    type: server
    org: chargeshield
    cn: fl-server
  - name: highway_cluster
    type: client
    org: chargeshield
    cn: highway_cluster
  - name: urban_cluster
    type: client
    org: chargeshield
    cn: urban_cluster
  - name: corporate_cluster
    type: client
    org: chargeshield
    cn: corporate_cluster
  - name: residential_cluster
    type: client
    org: chargeshield
    cn: residential_cluster
```

NVFLARE's `provision` command processes `project.yml` and generates a startup kit for each participant, incorporating the certificates generated by `make certs`. This declarative provisioning approach ensures that the certificate identity (`cn:`) exactly matches the `ChargingNode.node_id` used in the adapter layer, creating an end-to-end chain of identity from the X.509 certificate through the TLS handshake to the FL round participant record.

### 7.4 Attack Surface Reduction: Preventing the Rogue Aggregator Attack

A specific attack that mTLS prevents is the **rogue aggregator attack**: a scenario in which an adversary interposes a malicious FL server between the charging nodes and the legitimate aggregation server. Under standard TLS, the nodes would verify the server's certificate; under mTLS, the malicious server must also present a valid client certificate to any participant that challenges it — but the malicious server does not have a certificate signed by the experiment's local CA. The mTLS handshake fails, and the charging nodes refuse to connect to the rogue aggregator.

This attack is particularly relevant in MQTT-based residential deployments where the broker is potentially in a less trusted network position. Even if an adversary compromises the MQTT broker and uses it as a pivot point to attempt a connection to the NVFLARE server posing as a legitimate node, the absence of a valid client certificate causes the connection to be rejected at the transport layer.

---

## 8. Containerlab: Reproducible Network Topology Emulation

### 8.1 Why Containerlab over Alternatives

ChargeShield-FL uses Containerlab (developed by Nokia and the open-source community) to define, instantiate, and manage the emulated EV charging network topology. The selection of Containerlab over alternative network emulation platforms reflects specific requirements of the ChargeShield-FL research environment.

**Why not GNS3?** GNS3 is a mature network emulation platform widely used in network engineering education and certification preparation. It excels at emulating Cisco IOS, Juniper JunOS, and other vendor-specific network operating systems. However, GNS3's workflow is GUI-centric, making it poorly suited for reproducible, automated experiment execution in a CI/CD pipeline. GNS3's topology definitions are stored in proprietary `.gns3` project files that are not amenable to version control and diff review. Furthermore, GNS3's emulation of Linux hosts relies on QEMU virtual machines, which impose the startup time and memory overhead discussed below.

**Why not Mininet?** Mininet is widely used in SDN (Software-Defined Networking) research for emulating arbitrary network topologies using Linux namespaces and the Open vSwitch software switch. Mininet's strength is its Python API for programmatic topology construction and its tight integration with the OpenFlow protocol. However, Mininet's network stack is not fully Linux-native: it uses the `mn` virtual interface abstraction, which does not support all Linux networking features and has known limitations with certain routing protocols and tunnel interfaces. ChargeShield-FL requires full Linux network stack behavior (WireGuard kernel module, iptables rules, real routing table manipulation) that Mininet's abstraction layer does not reliably support.

**Why not bare metal?** A bare-metal testbed using physical charging station hardware would provide the most realistic experimental environment but is prohibitively expensive for a research framework (commercial 50 kW DC charging stations cost USD 20,000–50,000 each), requires physical installation, is not reproducible across research institutions, and cannot be automated for CI/CD testing. ChargeShield-FL's design prioritizes reproducibility and accessibility over hardware fidelity.

**Containerlab's advantages.** Containerlab addresses all of the above limitations:

- **Docker-native**: Containerlab deploys network nodes as Docker containers, leveraging the full Linux networking stack (real network namespaces, real kernel routing, real WireGuard support).
- **Declarative YAML topology**: The network topology is defined in a single YAML file that is version-controlled, human-readable, and diff-reviewable. A topology change is a one-line pull request.
- **Active development and community**: Containerlab is actively maintained with frequent releases, has a growing community of contributors, and is widely used in network automation and telemetry research.
- **CI/CD friendly**: Containerlab can be invoked from a shell script or Makefile (`containerlab deploy -t topology.yml`), making it trivial to integrate into GitHub Actions or GitLab CI workflows for automated experiment execution.

### 8.2 Why Docker Containers over Virtual Machines

The use of Docker containers (as opposed to QEMU or VirtualBox virtual machines) as the node implementation technology is motivated by several practical considerations:

**Startup time.** A Docker container starts in under one second (for containers with pre-pulled images). A QEMU virtual machine boots a full Linux kernel and requires 30–120 seconds to reach a usable state. For ChargeShield-FL experiments that may execute hundreds of FL rounds, the ability to rapidly restart nodes (e.g., to simulate node failure and recovery) is essential. Container-based node restart is operationally instantaneous compared to VM restart.

**Memory overhead.** A minimal Docker container running a Python FL client requires approximately 50–100 MB of resident memory. An equivalent QEMU VM running a minimal Linux distribution requires 256–512 MB, due to the overhead of the VM kernel, QEMU device emulation, and the guest OS init system. For a four-cluster ChargeShield-FL experiment, the difference is approximately 800 MB vs. 2+ GB of memory overhead for node emulation, leaving more memory available for the FL model parameters and training data.

**Reproducibility.** Docker images are content-addressed (image layers are identified by SHA-256 hashes of their content). A ChargeShield-FL experiment that specifies exact image versions (e.g., `chargeshield-fl-highway:1.0.0`) is bit-for-bit reproducible across any Docker-capable host. VM images, particularly those built with GUI-based tools, are difficult to version and reproduce consistently.

**CI/CD integration.** GitHub Actions, GitLab CI, and CircleCI all provide Docker-in-Docker or Docker-available runner environments as a standard service. Running Containerlab-based experiments in CI requires only a runner with Docker and Containerlab installed, both of which are available in standard CI environments.

### 8.3 Topology YAML and Its Mapping to the EV Charging Network

The ChargeShield-FL Containerlab topology YAML defines the following network elements:

```yaml
name: chargeshield-fl

topology:
  kinds:
    linux:
      image: chargeshield-fl-node:latest

  nodes:
    fl-server:
      kind: linux
      image: chargeshield-fl-server:latest
      binds:
        - certs/server:/certs:ro
        - configs/nvflare:/nvflare:ro

    highway-cluster:
      kind: linux
      binds:
        - certs/highway_cluster:/certs:ro
        - data/highway:/data:ro

    urban-cluster:
      kind: linux
      binds:
        - certs/urban_cluster:/certs:ro
        - data/urban:/data:ro

    corporate-cluster:
      kind: linux
      binds:
        - certs/corporate_cluster:/certs:ro
        - data/corporate:/data:ro

    residential-cluster:
      kind: linux
      binds:
        - certs/residential_cluster:/certs:ro
        - data/residential:/data:ro

    mqtt-broker:
      kind: linux
      image: eclipse-mosquitto:2.0
      binds:
        - configs/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro

  links:
    - endpoints: ["fl-server:eth1", "highway-cluster:eth1"]
    - endpoints: ["fl-server:eth1", "urban-cluster:eth1"]
    - endpoints: ["fl-server:eth1", "corporate-cluster:eth1"]
    - endpoints: ["fl-server:eth1", "residential-cluster:eth1"]
    - endpoints: ["mqtt-broker:eth1", "residential-cluster:eth2"]
```

In this topology:

- Each container corresponds to one charging cluster (not one charging station). Within the container, the ChargeShield-FL framework simulates the aggregated behavior of multiple charging stations within that cluster, using the local charging dataset (`/data`) as the training corpus.
- The `eth1` links connecting all clusters to the FL server form the operational subnets over which OCPP/MQTT traffic also notionally flows. In practice, the WireGuard overlay is configured within each container to route FL traffic over a logical `wg0` interface, separating it from OCPP/MQTT at the network layer.
- The `eth2` link between `mqtt-broker` and `residential-cluster` represents the dedicated MQTT subnet for residential charging communication.

### 8.4 Limitations of Emulation Relative to Hardware Testbed

ChargeShield-FL's Containerlab emulation provides a reproducible and accessible research environment, but several limitations relative to a physical hardware testbed should be acknowledged for scientific completeness:

**Network timing fidelity.** Docker bridge networking uses Linux software bridges, which introduce variable packet latency depending on host CPU load. Physical charging network deployments use dedicated switching hardware with deterministic forwarding behavior. For timing-sensitive FL synchronization experiments, this variability may affect measured round-trip times and should be accounted for in experimental analysis.

**Protocol stack completeness.** The OCPP adapter in ChargeShield-FL implements the message format and transaction lifecycle of OCPP 1.6/2.0.1 but does not emulate all edge cases of real charging station firmware (e.g., vendor-specific extensions, firmware update interruptions, power cycle recovery). Security evaluations that depend on specific firmware edge cases would require a hardware testbed.

**Physical attack surface.** Physical charging stations expose attack surfaces (hardware debug interfaces, physical connector manipulation, power side-channel analysis) that are entirely absent from a software emulation. ChargeShield-FL does not attempt to model physical attacks and explicitly scopes its threat model to network-layer and FL-layer adversaries.

---

## 9. NVFLARE: Production-Grade Federated Learning Communications

### 9.1 gRPC-Based FL Communication

NVFLARE (NVIDIA Federated Learning Application Runtime Environment) uses gRPC as its underlying communication protocol between the FL server (aggregator) and FL clients (participants). gRPC is a high-performance RPC framework developed by Google, using Protocol Buffers (protobuf) for message serialization and HTTP/2 as the transport.

The choice of gRPC for FL communication provides several advantages over REST/HTTP or custom socket protocols:

- **Efficient serialization**: Protocol Buffers produce compact binary encodings of structured data, significantly smaller than equivalent JSON representations. For FL gradient updates, which may comprise millions of floating-point values, the serialization efficiency difference between protobuf and JSON can be an order of magnitude.
- **Streaming support**: gRPC supports server-streaming, client-streaming, and bidirectional streaming RPCs over a single HTTP/2 connection. NVFLARE uses streaming to transmit large model parameter tensors without requiring multiple request-response cycles.
- **Built-in mTLS support**: gRPC's HTTP/2 transport integrates directly with TLS, and NVFLARE configures it with mutual TLS using the certificates generated by `make certs` and specified in the NVFLARE provisioning startup kit.

### 9.2 Declarative Provisioning via project.yml

NVFLARE's provisioning model is declarative: the experiment administrator defines the complete FL participant topology in `project.yml`, and NVFLARE's `provision` command generates all necessary configuration artifacts (startup scripts, certificates, channel definitions) automatically. This declarative approach is essential for ChargeShield-FL's reproducibility requirements.

The alternative — manual configuration of each NVFLARE participant — is error-prone and does not scale. In a ChargeShield-FL experiment with four charging clusters, manual configuration requires consistent certificate naming, consistent gRPC endpoint definitions, and consistent security profile settings across five configuration files (one per participant). Any inconsistency causes the FL round to fail in ways that are difficult to diagnose. The declarative `project.yml` approach ensures that all participants are configured from a single source of truth, eliminating configuration drift and enabling automated re-provisioning for each experiment run.

### 9.3 Why NVFLARE over Flower or PySyft

The selection of NVFLARE as the FL framework for ChargeShield-FL was made after evaluating the three most prominent open-source FL frameworks: NVFLARE, Flower (flwr), and PySyft.

**NVFLARE** (developed by NVIDIA, used extensively in healthcare FL research) offers:

- Production-grade reliability, with active development and commercial support from NVIDIA.
- Built-in support for Differential Privacy (DP) mechanisms, enabling ChargeShield-FL to evaluate the interaction between DP noise injection and MIA effectiveness.
- A flexible custom workflow executor API that allows ChargeShield-FL to hook into the FL round lifecycle at arbitrary points (before aggregation, after aggregation, before client update dispatch) for MIA evaluation.
- Native mTLS support with X.509 certificate-based authentication integrated at the framework level.
- Active maintenance with a strong development team and regular security patches.

**Flower (flwr)** is a research-focused FL framework designed for simplicity and flexibility. Its primary strength is the ease with which new FL strategies can be implemented in pure Python. However, Flower's production readiness is lower than NVFLARE's: it does not provide built-in certificate management, its security model relies on the user configuring TLS correctly, and its built-in DP support is limited. For a research framework that aims to evaluate security properties of FL deployments, these gaps in Flower's security infrastructure would require substantial custom implementation, reducing the experiment's fidelity as an evaluation of a realistic production FL system.

**PySyft** (developed by OpenMined) is designed for privacy-preserving ML with a focus on secure multi-party computation (SMPC) and homomorphic encryption. While PySyft's cryptographic capabilities are impressive, the framework has experienced significant API instability across major versions (from 0.2.x to 0.5.x to 0.8.x), making long-term experiment reproducibility challenging. Additionally, PySyft's SMPC-centric design is not representative of the FL deployments that ChargeShield-FL aims to evaluate, which use standard gradient aggregation (FedAvg) as the baseline — the same aggregation approach used in the vast majority of production FL deployments.

### 9.4 Custom Workflow Integration

ChargeShield-FL integrates with NVFLARE's round lifecycle through custom executor classes that inherit from NVFLARE's `Executor` base class. The integration points are:

- **Pre-aggregation hook**: Before the FL server aggregates client gradients, ChargeShield-FL's MIA evaluator receives the per-client gradient tensors and executes the configured attack strategy (see Section 10).
- **Post-aggregation hook**: After aggregation, the Privacy Auditor (see Section 11) records the round metrics (participation set, aggregated gradient norm, per-client contribution) to the experiment log.
- **Round completion notification**: The ML Plane Listener (see Section 11) dispatches round completion events to all registered observers.

This integration is implemented without modifying NVFLARE's core code, relying entirely on NVFLARE's public executor API. This ensures that ChargeShield-FL remains compatible with future NVFLARE versions and does not depend on internal implementation details that may change without notice.

---

## 10. Strategy Pattern: Interchangeable Attack Plugins

### 10.1 Motivation for the Strategy Pattern

ChargeShield-FL's primary research contribution is the evaluation of Membership Inference Attacks against FL systems in the EV charging domain. MIA is not a monolithic technique: the literature includes numerous distinct attack algorithms, including Shokri et al.'s original shadow model attack (S&P 2017), Nasr et al.'s gradient-based MIA (S&P 2019), Carlini et al.'s likelihood ratio attack (S&P 2022), and the FedMIA algorithm adapted for the federated setting. Implementing these attacks as hardcoded functions within the ChargeShield-FL core framework would violate the Open/Closed Principle: adding a new attack would require modifying existing, tested code.

The **Strategy Pattern** (Gamma et al., 1994) defines a family of algorithms, encapsulates each one, and makes them interchangeable. The framework depends on an abstract strategy interface, and the concrete attack algorithm is injected at experiment configuration time. This allows ChargeShield-FL to support an extensible library of MIA implementations without modifying core framework code.

### 10.2 AbstractAttack Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class AbstractAttack(ABC):
    """
    Abstract base class for Membership Inference Attack strategies.

    All MIA implementations in ChargeShield-FL must conform to this interface.
    The attack receives FL round artifacts (gradient tensors, round metadata)
    and produces a membership inference decision for a set of target records.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the attack with experiment-specific configuration.
        Called once before the first FL round begins.
        """
        ...

    @abstractmethod
    def observe_round(
        self,
        round_number: int,
        client_gradients: Dict[str, np.ndarray],
        aggregated_model: np.ndarray,
        round_metadata: Dict[str, Any],
    ) -> None:
        """
        Receive FL round artifacts for analysis.
        The attack may accumulate information across multiple rounds,
        building an attack model that improves in accuracy over time.
        """
        ...

    @abstractmethod
    def infer_membership(
        self,
        target_records: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Produce membership inference scores for a list of target records.

        Returns
        -------
        List[float]
            A score in [0, 1] for each target record, where a value
            approaching 1 indicates high confidence that the record was
            a member of the training dataset.
        """
        ...

    @abstractmethod
    def get_attack_name(self) -> str:
        """
        Return the canonical name of this attack algorithm.
        Used for experiment logging and result attribution.
        """
        ...
```

The `FedMIA` class implements `AbstractAttack` with the federated membership inference algorithm adapted to the EV charging telemetry domain. Future attacks (e.g., a gradient inversion attack, a property inference attack targeting charging behavior patterns) implement the same interface and can be substituted by changing a single configuration line:

```yaml
attack:
  strategy: FedMIA
  config:
    threshold: 0.5
    shadow_rounds: 10
```

Changing `strategy: FedMIA` to `strategy: LikelihoodRatioAttack` substitutes the attack algorithm entirely without touching any other framework component, demonstrating the power of the Strategy pattern for extensible experimental design.

---

## 11. Observer Pattern: ML Plane Listener and Privacy Auditing

### 11.1 Motivation for the Observer Pattern

The ML Plane Listener in ChargeShield-FL is responsible for observing FL round events — gradient updates received from charging nodes, round completion notifications, aggregation results — and dispatching these events to multiple downstream consumers: the Privacy Auditor, the Intrusion Detection System (IDS), and the experiment logger. If the ML Plane Listener maintained direct references to each of these consumers, adding a new consumer (e.g., a new anomaly detector for Byzantine fault detection) would require modifying the listener code.

The **Observer Pattern** (Gamma et al., 1994) defines a one-to-many dependency between objects: when the subject (ML Plane Listener) changes state, all registered observers are notified automatically. Observers subscribe to the subject independently, and the subject has no knowledge of observer implementation details.

### 11.2 MLPlaneListener Implementation

```python
from typing import List, Protocol
from dataclasses import dataclass
import numpy as np


@dataclass
class FLRoundEvent:
    """Immutable record of a completed FL round."""
    round_number: int
    participating_nodes: List[str]
    client_gradients: dict  # node_id -> gradient tensor
    aggregated_model: np.ndarray
    round_duration_seconds: float


class FLRoundObserver(Protocol):
    """Structural interface (Protocol) for FL round observers."""

    def on_round_complete(self, event: FLRoundEvent) -> None:
        """Called when an FL round completes successfully."""
        ...

    def on_gradient_received(
        self,
        node_id: str,
        gradient: np.ndarray,
        round_number: int,
    ) -> None:
        """Called when a gradient update is received from a specific node."""
        ...


class MLPlaneListener:
    """
    Observes FL round events from the NVFLARE runtime and dispatches
    them to all registered FLRoundObserver instances.

    This class implements the Subject role in the Observer pattern.
    It holds no direct references to observer implementation classes,
    depending only on the FLRoundObserver structural interface.
    """

    def __init__(self) -> None:
        self._observers: List[FLRoundObserver] = []

    def register_observer(self, observer: FLRoundObserver) -> None:
        """Register an observer to receive FL round events."""
        self._observers.append(observer)

    def deregister_observer(self, observer: FLRoundObserver) -> None:
        """Remove a previously registered observer."""
        self._observers = [o for o in self._observers if o is not observer]

    def notify_round_complete(self, event: FLRoundEvent) -> None:
        """Dispatch a round completion event to all registered observers."""
        for observer in self._observers:
            observer.on_round_complete(event)

    def notify_gradient_received(
        self,
        node_id: str,
        gradient: np.ndarray,
        round_number: int,
    ) -> None:
        """Dispatch a gradient received event to all registered observers."""
        for observer in self._observers:
            observer.on_gradient_received(node_id, gradient, round_number)
```

### 11.3 Observer Registration at Experiment Startup

At experiment startup, the relevant observers are instantiated and registered with the `MLPlaneListener`:

```python
listener = MLPlaneListener()

privacy_auditor = PrivacyAuditor(attack_strategy=FedMIA(config))
ids_module = ChargingIDS(anomaly_threshold=0.95)
experiment_logger = ExperimentLogger(output_path="results/")

listener.register_observer(privacy_auditor)
listener.register_observer(ids_module)
listener.register_observer(experiment_logger)
```

The `PrivacyAuditor` and `IDSModule` classes implement `FLRoundObserver` independently, using Python's structural subtyping (Protocol) rather than nominal inheritance. This means neither class needs to explicitly declare that it implements the observer interface — it is sufficient for the class to define `on_round_complete` and `on_gradient_received` methods with the correct signatures. This approach reduces coupling between the observer implementations and the subject.

### 11.4 Why Observer over Direct Method Calls

The alternative to the Observer pattern — the `MLPlaneListener` directly calling `privacy_auditor.on_round_complete(event)` and `ids_module.on_round_complete(event)` — would couple the listener to specific consumer implementations. The Observer pattern provides:

- **Decoupling**: The listener does not import or depend on consumer modules. This eliminates circular import risks in Python and simplifies unit testing (the listener can be tested with a mock observer that records events for assertion).
- **Dynamic subscription**: Observers can be registered and deregistered at runtime, enabling experiment configurations that activate or deactivate monitoring components without restarting the framework. For example, a warm-up phase that runs without MIA evaluation can register the `PrivacyAuditor` only after the initial rounds complete.
- **Multiple independent observers**: Each observer processes events independently, enabling parallel analysis pipelines where the Privacy Auditor and IDS operate concurrently without one blocking the other.
- **Extensibility**: A future contribution adding a new analysis component (e.g., a gradient norm monitor for Byzantine fault detection) requires only implementing the `FLRoundObserver` protocol and calling `listener.register_observer()`, with no modification to existing code.

---

## 12. Adding a New Protocol Adapter: A Practitioner's Guide

This section provides a step-by-step guide for implementing a new protocol adapter in ChargeShield-FL. As a running example, we use a hypothetical IEC 61850 MMS adapter, appropriate for a future scenario in which EV charging infrastructure in smart substation environments communicates via IEC 61850 MMS (Manufacturing Message Specification). This scenario is forward-looking but technically plausible given the increasing integration of EV charging with distribution grid infrastructure.

### 12.1 Step 1: Implement the AbstractProtocolAdapter

Create a new file `chargeshield_fl/adapters/iec61850_adapter.py`:

```python
import json
import struct
from typing import Any, Dict

from chargeshield_fl.adapters.base import AbstractProtocolAdapter
from chargeshield_fl.exceptions import ProtocolEncodeError, ProtocolDecodeError


# Simplified IEC 61850 MMS message type constants
# In a full implementation these would be ASN.1 BER-encoded PDU tags
MMS_READ_RESPONSE = 0x01
MMS_WRITE_REQUEST = 0x02
MMS_REPORT = 0x03


class IEC61850Adapter(AbstractProtocolAdapter):
    """
    Protocol adapter for IEC 61850 MMS (Manufacturing Message Specification).

    This adapter encodes ChargeShield-FL internal messages as simplified
    IEC 61850 MMS PDUs for research evaluation of FL in smart substation
    EV charging contexts. The encoding is a research approximation: it
    captures the structural properties of MMS (typed data objects, logical
    node addressing, report-based telemetry) without implementing the full
    ASN.1 BER encoding of the IEC 61850-8-1 standard, which would require
    an ASN.1 library dependency.

    In production integration with real IEC 61850 devices, a conformant
    ASN.1 BER encoder (e.g., libiec61850 via ctypes) would replace the
    simplified struct-based encoding used here.
    """

    # IEC 61850 Logical Node name prefix for EV charging (EVSE function)
    LN_PREFIX = "EVSE"

    def encode(self, message: Dict[str, Any]) -> bytes:
        """
        Encode an internal message as a simplified MMS Write Request PDU.

        Wire format:
          [1 byte]  message type code
          [1 byte]  node_id length (max 255 bytes)
          [2 bytes] payload length (big-endian uint16, max 65535 bytes)
          [N bytes] node_id as UTF-8
          [M bytes] payload as UTF-8 JSON
        """
        try:
            node_id_bytes = message["node_id"].encode("utf-8")
            payload_bytes = json.dumps(message["payload"]).encode("utf-8")

            if len(node_id_bytes) > 255:
                raise ProtocolEncodeError(
                    f"node_id exceeds 255 bytes after UTF-8 encoding: "
                    f"{len(node_id_bytes)} bytes"
                )
            if len(payload_bytes) > 65535:
                raise ProtocolEncodeError(
                    f"payload exceeds 65535 bytes after JSON serialization: "
                    f"{len(payload_bytes)} bytes"
                )

            msg_type = MMS_WRITE_REQUEST
            header = struct.pack(
                ">BBH",
                msg_type,
                len(node_id_bytes),
                len(payload_bytes),
            )
            return header + node_id_bytes + payload_bytes
        except (KeyError, struct.error) as exc:
            raise ProtocolEncodeError(
                f"IEC61850Adapter.encode failed: {exc}"
            ) from exc

    def decode(self, raw: bytes) -> Dict[str, Any]:
        """
        Decode a simplified MMS PDU to the internal message schema.
        """
        try:
            if len(raw) < 4:
                raise ProtocolDecodeError(
                    f"PDU too short: expected at least 4 bytes, got {len(raw)}"
                )

            msg_type, node_id_len, payload_len = struct.unpack_from(">BBH", raw, 0)
            offset = 4

            if len(raw) < offset + node_id_len + payload_len:
                raise ProtocolDecodeError(
                    f"PDU truncated: expected {offset + node_id_len + payload_len} "
                    f"bytes, got {len(raw)}"
                )

            node_id = raw[offset : offset + node_id_len].decode("utf-8")
            offset += node_id_len

            payload = json.loads(raw[offset : offset + payload_len])

            return {
                "message_type": self._resolve_message_type(msg_type),
                "node_id": node_id,
                "payload": payload,
            }
        except (struct.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolDecodeError(
                f"IEC61850Adapter.decode failed: {exc}"
            ) from exc

    def get_protocol_name(self) -> str:
        return "iec61850_mms"

    @staticmethod
    def _resolve_message_type(code: int) -> str:
        """Map a numeric MMS message type code to a string identifier."""
        mapping = {
            MMS_READ_RESPONSE: "read_response",
            MMS_WRITE_REQUEST: "write_request",
            MMS_REPORT: "report",
        }
        return mapping.get(code, f"unknown_{code:#04x}")
```

### 12.2 Step 2: Register the Adapter in the Factory

ChargeShield-FL uses an `AdapterFactory` to map protocol name strings from experiment YAML configuration to adapter instances. Adding the new adapter requires a single line:

```python
# chargeshield_fl/adapters/factory.py

from chargeshield_fl.adapters.ocpp16_adapter import OCPP16Adapter
from chargeshield_fl.adapters.ocpp201_adapter import OCPP201Adapter
from chargeshield_fl.adapters.mqtt5_adapter import MQTT5Adapter
from chargeshield_fl.adapters.iec61850_adapter import IEC61850Adapter
from chargeshield_fl.adapters.base import AbstractProtocolAdapter

_REGISTRY = {
    "ocpp16": OCPP16Adapter,
    "ocpp201": OCPP201Adapter,
    "mqtt5": MQTT5Adapter,
    "iec61850_mms": IEC61850Adapter,  # Add new adapter here
}


def create_adapter(protocol_name: str, **kwargs) -> AbstractProtocolAdapter:
    """
    Instantiate a protocol adapter by its canonical name.

    Parameters
    ----------
    protocol_name : str
        The canonical protocol identifier, as returned by get_protocol_name().
    **kwargs
        Protocol-specific configuration parameters passed to the adapter
        constructor (e.g., broker_url for MQTT5Adapter).

    Raises
    ------
    ValueError
        If protocol_name does not correspond to a registered adapter.
    """
    cls = _REGISTRY.get(protocol_name)
    if cls is None:
        raise ValueError(
            f"Unknown protocol '{protocol_name}'. "
            f"Registered protocols: {sorted(_REGISTRY.keys())}"
        )
    return cls(**kwargs)
```

### 12.3 Step 3: Write Unit Tests

Unit tests for the new adapter verify round-trip encoding correctness, boundary condition handling, and error behavior for malformed inputs. Thorough unit testing is essential because adapter correctness directly affects the fidelity of FL gradient data used in MIA evaluation.

```python
# tests/test_iec61850_adapter.py

import pytest
from chargeshield_fl.adapters.iec61850_adapter import IEC61850Adapter
from chargeshield_fl.exceptions import ProtocolDecodeError, ProtocolEncodeError


@pytest.fixture
def adapter():
    return IEC61850Adapter()


def test_round_trip_basic(adapter):
    message = {
        "message_type": "write_request",
        "node_id": "corporate_cluster",
        "payload": {"energy_kwh": 12.5, "session_id": "abc123"},
    }
    encoded = adapter.encode(message)
    decoded = adapter.decode(encoded)
    assert decoded["node_id"] == message["node_id"]
    assert decoded["payload"] == message["payload"]


def test_round_trip_unicode_node_id(adapter):
    """Verify that non-ASCII node IDs are correctly encoded and decoded."""
    message = {
        "message_type": "write_request",
        "node_id": "cluster_énergie",
        "payload": {"power_kw": 7.4},
    }
    encoded = adapter.encode(message)
    decoded = adapter.decode(encoded)
    assert decoded["node_id"] == message["node_id"]


def test_decode_truncated_pdu_raises(adapter):
    """PDUs shorter than 4 bytes must raise ProtocolDecodeError."""
    with pytest.raises(ProtocolDecodeError, match="too short"):
        adapter.decode(b"\x02\x05")


def test_decode_truncated_body_raises(adapter):
    """PDUs where declared length exceeds actual bytes must raise."""
    import struct
    # Header declares 10 bytes of node_id and 100 bytes of payload
    header = struct.pack(">BBH", 0x02, 10, 100)
    with pytest.raises(ProtocolDecodeError, match="truncated"):
        adapter.decode(header + b"short")


def test_get_protocol_name(adapter):
    assert adapter.get_protocol_name() == "iec61850_mms"


def test_node_id_too_long_raises(adapter):
    """Node IDs exceeding 255 bytes must raise ProtocolEncodeError."""
    message = {
        "message_type": "write_request",
        "node_id": "x" * 256,
        "payload": {},
    }
    with pytest.raises(ProtocolEncodeError, match="255 bytes"):
        adapter.encode(message)
```

### 12.4 Step 4: Configure the Adapter in an Experiment

To use the new adapter in a ChargeShield-FL experiment, specify it in the cluster configuration YAML:

```yaml
clusters:
  - id: substation_cluster
    protocol: iec61850_mms
    power_kw: 150
    connector_type: DC_CCS2
    node_count: 8
    dataset: data/substation_charging.parquet
```

The `AdapterFactory` will instantiate `IEC61850Adapter` for this cluster, and all `ChargingNode` instances in the cluster will use it transparently. No other framework component requires modification.

### 12.5 Checklist for New Adapter Contributors

For completeness, the following checklist summarizes all steps required to add a new adapter:

1. Implement `AbstractProtocolAdapter` in `chargeshield_fl/adapters/<name>_adapter.py`.
2. Ensure `get_protocol_name()` returns a unique, stable string identifier.
3. Register the adapter class in `chargeshield_fl/adapters/factory.py`.
4. Write unit tests in `tests/test_<name>_adapter.py`, achieving at minimum 90% line coverage.
5. Add the protocol name to the `protocol` enum in `chargeshield_fl/config/schema.py` to enable YAML validation.
6. Document the protocol's security properties and threat model implications in an entry within this document's Section 4 or 5 (as appropriate).
7. Open a pull request with the adapter, tests, and documentation update for peer review.

---

## 13. References

**Open Charge Alliance.** *OCPP 1.6 Specification, Edition 2.* Open Charge Alliance, 2019. Available: https://www.openchargealliance.org/protocols/ocpp-16/

**Open Charge Alliance.** *OCPP 2.0.1 Specification.* Open Charge Alliance, 2020. Available: https://www.openchargealliance.org/protocols/ocpp-201/

**OASIS MQTT Technical Committee.** *MQTT Version 5.0.* OASIS Standard, March 2019. Available: https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html

**Donenfeld, J. A.** WireGuard: Next Generation Kernel Network Tunnel. In *Proceedings of the Network and Distributed System Security Symposium (NDSS 2017)*. Internet Society, San Diego, CA, USA, 2017. DOI: 10.14722/ndss.2017.23160

**IEC.** *IEC 62443-1-1: Industrial communication networks — IT security for networks and systems — Part 1-1: Terminology, concepts and models.* International Electrotechnical Commission, Geneva, Switzerland, 2009.

**IEC.** *IEC 62443-3-3: Industrial communication networks — IT security for networks and systems — Part 3-3: System security requirements and security levels.* International Electrotechnical Commission, Geneva, Switzerland, 2013.

**IEC.** *IEC 62443-4-2: Security for industrial automation and control systems — Part 4-2: Technical security requirements for IACS components.* International Electrotechnical Commission, Geneva, Switzerland, 2019.

**Gamma, E., Helm, R., Johnson, R., and Vlissides, J.** *Design Patterns: Elements of Reusable Object-Oriented Software.* Addison-Wesley Professional, Reading, MA, USA, 1994. ISBN: 978-0201633610.

**NVIDIA Corporation.** *NVIDIA FLARE: Federated Learning Application Runtime Environment — Developer Guide, v2.4.* NVIDIA, Santa Clara, CA, USA, 2023. Available: https://nvflare.readthedocs.io/

**Containerlab Contributors.** *Containerlab: Container-based Networking Labs.* Open Source Project, 2021–2025. Available: https://containerlab.dev/

**Shokri, R., Stronati, M., Song, C., and Shmatikov, V.** Membership Inference Attacks Against Machine Learning Models. In *Proceedings of the 38th IEEE Symposium on Security and Privacy (S&P 2017)*. IEEE, San Jose, CA, USA, 2017. DOI: 10.1109/SP.2017.41

**Nasr, M., Shokri, R., and Houmansadr, A.** Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning. In *Proceedings of the 40th IEEE Symposium on Security and Privacy (S&P 2019)*. IEEE, San Francisco, CA, USA, 2019. DOI: 10.1109/SP.2019.00065

**Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F.** Membership Inference Attacks From First Principles. In *Proceedings of the 43rd IEEE Symposium on Security and Privacy (S&P 2022)*. IEEE, San Francisco, CA, USA, 2022. DOI: 10.1109/SP46214.2022.9833649

**McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and Agüera y Arcas, B.** Communication-Efficient Learning of Deep Networks from Decentralized Data. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*. PMLR, Fort Lauderdale, FL, USA, 2017. Available: https://arxiv.org/abs/1602.05629

**Bernstein, D. J., Duif, N., Lange, T., Schwabe, P., and Yang, B.-Y.** High-Speed High-Security Signatures. In *Proceedings of the 13th International Workshop on Cryptographic Hardware and Embedded Systems (CHES 2011)*. Springer, Nara, Japan, 2011. DOI: 10.1007/978-3-642-23951-9_9

**Rescorla, E.** *The Transport Layer Security (TLS) Protocol Version 1.3.* IETF RFC 8446, August 2018. DOI: 10.17487/RFC8446

**Postel, J., and Reynolds, J.** *Transmission Control Protocol.* IETF RFC 793, September 1981. DOI: 10.17487/RFC0793

**Fette, I., and Melnikov, A.** *The WebSocket Protocol.* IETF RFC 6455, December 2011. DOI: 10.17487/RFC6455

**ISO/IEC.** *ISO 15118-20: Road Vehicles — Vehicle to Grid Communication Interface — Part 20: 2nd Generation Network and Application Protocol Requirements.* International Organization for Standardization / International Electrotechnical Commission, Geneva, Switzerland, 2022.

**European Commission.** *Regulation (EU) 2023/1804 of the European Parliament and of the Council on the Deployment of Alternative Fuels Infrastructure (AFIR), and repealing Directive 2014/94/EU.* Official Journal of the European Union, L 234, September 2023.

**OASIS.** *Advanced Message Queuing Protocol (AMQP) Version 1.0.* OASIS Standard, October 2012. Available: https://www.amqp.org/

**Nance, C., Hay, S., and Bishop, M.** Analyzing Vulnerabilities in Industrial Control System Communication Protocols. In *Proceedings of the 46th Hawaii International Conference on System Sciences (HICSS 2013)*. IEEE, Wailea, HI, USA, 2013. DOI: 10.1109/HICSS.2013.133

**Perrig, A., Tygar, J. D., Song, D., and Canetti, R.** Efficient Authentication and Signing of Multicast Streams over Lossy Channels. In *Proceedings of the 21st IEEE Symposium on Security and Privacy (S&P 2000)*. IEEE, Berkeley, CA, USA, 2000. DOI: 10.1109/SECPRI.2000.848446

---

*This document was prepared as part of the ChargeShield-FL research project. All protocol specifications cited herein are publicly available standards documents. Python code examples are illustrative of the design intent and may differ from the production codebase in minor syntactic respects. For the authoritative implementation, refer to the ChargeShield-FL source repository and its accompanying test suite.*
