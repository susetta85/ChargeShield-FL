# src/core/base_auditor.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditReport:
    node_id: str
    round_id: int
    privacy_score: float          # 0.0 (no privacy) → 1.0 (full privacy)
    epsilon: float                # DP epsilon consumed
    threats_detected: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractPrivacyAuditor(ABC):
    """
    Contract for the privacy auditing component.
    Auditor knows nothing about FL internals or protocols.
    """

    @abstractmethod
    def audit(self, node_id: str, round_id: int, model_update: dict[str, Any]) -> AuditReport:
        """Analyze a model update and return a privacy audit report."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset auditor state between experiments."""
        ...

    @abstractmethod
    def get_cumulative_epsilon(self, node_id: str) -> float:
        """Return total DP epsilon consumed by a node so far."""
        ...
