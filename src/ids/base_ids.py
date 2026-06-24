# src/ids/base_ids.py
"""
Abstract IDS — Intrusion Detection System Interface
====================================================
Definisce il contratto per qualsiasi sistema di rilevamento intrusioni
nel framework ChargeShield-FL.

Ruolo nel framework:
- È la DIFESA contro gli attacchi di membership inference (FedMIA)
  e altri attacchi federati (model poisoning, Byzantine fault)
- Monitora il comportamento dei nodi durante i round FL
- Decide se un nodo è sospetto e deve essere escluso dall'aggregazione

Cosa NON fa questo modulo:
- Non esegue attacchi (quello è PrivacyAuditor + FedMIA)
- Non conosce FL, protocolli, o dataset
- Non blocca direttamente i nodi — segnala, poi è il sistema chiamante a decidere

Relazione con gli altri moduli:
- PrivacyAuditor (src/auditor/) → produce AuditReport con il rischio MIA
- AbstractIDS (questo file) → riceve l'AuditReport e decide se il nodo è anomalo
- ChargingIDS (src/ids/charging_ids.py, Sprint 4) → implementazione concreta

Riferimenti:
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant SGD", NeurIPS 2017
- Fung et al., "Mitigating Sybils in Federated Learning Poisoning", 2020
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.base_auditor import AuditReport


@dataclass
class IDSAlert:
    """
    Rappresenta un alert generato dall'IDS per un nodo sospetto.

    Attributes:
        node_id:     nodo che ha generato l'alert
        round_id:    round FL in cui è stata rilevata l'anomalia
        severity:    livello di severità — LOW | MEDIUM | HIGH | CRITICAL
        reasons:     lista di motivazioni dell'alert
        recommended_action: azione consigliata — MONITOR | THROTTLE | EXCLUDE
        metadata:    informazioni aggiuntive per il debug
    """
    node_id: str
    round_id: int
    severity: str
    reasons: list[str]
    recommended_action: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractIDS(ABC):
    """
    Interfaccia astratta per l'Intrusion Detection System.

    Ogni implementazione concreta riceve gli AuditReport prodotti
    dal PrivacyAuditor e decide se il comportamento di un nodo
    è anomalo rispetto alla baseline del cluster.

    Il principio architetturale è rispettato:
    - AbstractIDS non conosce FL, protocolli, o dataset
    - Comunica solo tramite AuditReport (input) e IDSAlert (output)
    """

    @abstractmethod
    def analyze(self, report: AuditReport) -> IDSAlert | None:
        """
        Analizza un AuditReport e decide se generare un alert.

        Args:
            report: AuditReport prodotto dal PrivacyAuditor per un nodo

        Returns:
            IDSAlert se il nodo è sospetto, None se il comportamento è normale
        """
        ...

    @abstractmethod
    def update_baseline(self, node_id: str, report: AuditReport) -> None:
        """
        Aggiorna la baseline comportamentale di un nodo.

        La baseline è il comportamento atteso del nodo in condizioni normali.
        Viene aggiornata ad ogni round per adattarsi ai cambiamenti legittimi.

        Args:
            node_id: identificatore del nodo
            report:  AuditReport del round corrente
        """
        ...

    @abstractmethod
    def get_node_risk_score(self, node_id: str) -> float:
        """
        Restituisce il risk score corrente di un nodo.

        Args:
            node_id: identificatore del nodo

        Returns:
            float tra 0.0 (nessun rischio) e 1.0 (rischio massimo)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Resetta lo stato interno dell'IDS.
        Da chiamare tra esperimenti diversi.
        """
        ...
