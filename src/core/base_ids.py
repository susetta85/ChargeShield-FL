# src/core/base_ids.py
"""
Abstract IDS — Intrusion Detection System Interface
====================================================
Definisce il contratto per qualsiasi sistema di rilevamento intrusioni
nel framework ChargeShield-FL.

Ruolo nel framework:
- È la DIFESA contro attacchi MIA, model poisoning e Byzantine faults
- Monitora il comportamento dei nodi durante i round FL
- Opera sia a livello di singolo nodo che di cluster

Livelli di analisi:
1. analyze(report)       → singolo nodo, basato su AuditReport
2. analyze_round(...)    → intero round, usa gradienti reali di tutti i nodi
                           Krum, cosine similarity, CUSUM, FedMIA

Relazione con gli altri moduli:
- PrivacyAuditor → produce AuditReport (input per analyze)
- FedMIA         → produce MIAResult (input per analyze_round)
- ChargingIDS    → implementazione concreta (src/ids/charging_ids.py)

Riferimenti:
- Blanchard et al., "Machine Learning with Adversaries: Byzantine
  Tolerant SGD", NeurIPS 2017
- Fung et al., "Mitigating Sybils in FL Poisoning", 2020
- Page, "Continuous Inspection Schemes", Biometrika 1954 (CUSUM)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.base_auditor import AuditReport


@dataclass
class IDSAlert:
    """
    Alert generato dall'IDS per un nodo sospetto.

    Attributes:
        node_id:            nodo che ha generato l'alert
        round_id:           round FL in cui è stata rilevata l'anomalia
        severity:           LOW | MEDIUM | HIGH | CRITICAL
        reasons:            lista di motivazioni dell'alert
        recommended_action: MONITOR | THROTTLE | EXCLUDE
        metadata:           informazioni aggiuntive per debug e paper
    """
    node_id: str
    round_id: int
    severity: str
    reasons: list[str]
    recommended_action: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundAnalysis:
    """
    Risultato dell'analisi di un intero round FL.

    Attributes:
        round_id:         round analizzato
        alerts:           alert generati per i nodi sospetti
        byzantine_nodes:  nodi identificati come Byzantine (Krum)
        low_similarity_nodes: nodi con bassa cosine similarity nel cluster
        krum_scores:      {node_id: krum_score} per tutti i nodi
        cosine_scores:    {node_id: avg_cosine_similarity} per tutti i nodi
    """
    round_id: int
    alerts: list[IDSAlert]
    byzantine_nodes: list[str]
    low_similarity_nodes: list[str]
    krum_scores: dict[str, float] = field(default_factory=dict)
    cosine_scores: dict[str, float] = field(default_factory=dict)


class AbstractIDS(ABC):
    """
    Interfaccia astratta per l'Intrusion Detection System.

    Due livelli di analisi:
    1. analyze()       → analisi singolo nodo (rule-based + CUSUM)
    2. analyze_round() → analisi intero round (Krum + cosine + FedMIA)
    """

    @abstractmethod
    def analyze(self, report: AuditReport) -> IDSAlert | None:
        """
        Analizza un AuditReport e decide se generare un alert.

        Args:
            report: AuditReport del PrivacyAuditor per un nodo

        Returns:
            IDSAlert se sospetto, None se normale
        """
        ...

    @abstractmethod
    def analyze_round(
        self,
        round_id: int,
        reports: dict[str, AuditReport],
        gradients: dict[str, dict[str, Any]],
    ) -> RoundAnalysis:
        """
        Analizza un intero round FL con tutti i detector.

        Usa Krum, cosine similarity, CUSUM e FedMIA
        per rilevare comportamenti anomali a livello di cluster.

        Args:
            round_id:  numero del round FL
            reports:   {node_id: AuditReport} per tutti i nodi
            gradients: {node_id: model_update} per tutti i nodi

        Returns:
            RoundAnalysis con alert, Byzantine nodes, cosine scores
        """
        ...

    @abstractmethod
    def update_baseline(self, node_id: str, report: AuditReport) -> None:
        """
        Aggiorna la baseline comportamentale del nodo.

        Args:
            node_id: identificatore del nodo
            report:  AuditReport del round corrente
        """
        ...

    @abstractmethod
    def get_node_risk_score(self, node_id: str) -> float:
        """
        Restituisce il risk score corrente del nodo [0.0, 1.0].

        Args:
            node_id: identificatore del nodo

        Returns:
            risk score (0.0 = nessun rischio, 1.0 = rischio massimo)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Resetta lo stato interno — da chiamare tra esperimenti."""
        ...
