# src/auditor/privacy_auditor.py
"""
Privacy Auditor — Membership Inference Attacker
================================================
Questo modulo implementa un attaccante di tipo Membership Inference Attack (MIA).

Ruolo nel framework:
- NON è una difesa
- È lo strumento con cui un avversario tenta di inferire se un dato campione
  è stato usato nel training di un nodo FL
- Misura il rischio di privacy esponendo quanto il modello "ricorda" i dati

Come funziona:
- Analizza i model update (gradienti) ricevuti da ogni nodo
- Calcola la sensitivity del gradiente come proxy del rischio MIA
- Accumula l'epsilon di privacy differenziale consumato per nodo
- Produce un AuditReport con il rischio stimato e le anomalie rilevate

Relazione con FedMIA (Sprint 4):
- FedMIA (src/plugins/attacks/fedmia.py) implementerà l'attacco completo
- Questo modulo è il punto di ingresso: raccoglie i gradienti PRIMA
  che vengano aggregati dal server FL e li passa a FedMIA

Relazione con IDS (Sprint 4):
- src/ids/charging_ids.py è la DIFESA contro questo attacco
- L'IDS monitora i comportamenti anomali e può bloccare nodi sospetti

Riferimenti:
- Shokri et al., "Membership Inference Attacks Against ML Models", IEEE S&P 2017
- Dwork & Roth, "Algorithmic Foundations of Differential Privacy", 2014
"""

import math
import yaml
from pathlib import Path
from typing import Any

from core.base_auditor import AbstractPrivacyAuditor, AuditReport


def _load_auditor_config(config_path: str = "config/auditor.yaml") -> dict:
    """
    Carica la configurazione del Privacy Auditor da YAML.
    Nessun valore hardcoded nel codice.

    Args:
        config_path: percorso al file auditor.yaml

    Raises:
        FileNotFoundError: se il file non esiste
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Auditor config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)["auditor"]


def _compute_l2_norm(values: list[float]) -> float:
    """
    Calcola la norma L2 di un vettore di float.

    Usata come proxy del rischio MIA: un gradiente con norma L2 alta
    indica che il modello ha memorizzato fortemente alcuni campioni,
    rendendoli più vulnerabili a membership inference.

    Args:
        values: lista di valori float (pesi o gradienti del modello)

    Returns:
        norma L2 del vettore
    """
    return math.sqrt(sum(v ** 2 for v in values))


def _flatten_model_update(model_update: dict[str, Any]) -> list[float]:
    """
    Appiattisce un model update in una lista di float.

    I model update contengono pesi del modello (sempre numerici).
    Le feature categoriche (charging_mode, error_code) vengono
    codificate numericamente PRIMA del training — quindi non
    compaiono come stringhe nel model update.

    Args:
        model_update: dizionario layer → pesi (float, list, o dict annidato)

    Returns:
        lista piatta di float
    """
    flat: list[float] = []
    for value in model_update.values():
        if isinstance(value, (int, float)):
            flat.append(float(value))
        elif isinstance(value, list):
            flat.extend(float(v) for v in value if isinstance(v, (int, float)))
        elif isinstance(value, dict):
            flat.extend(_flatten_model_update(value))
    return flat


class PrivacyAuditor(AbstractPrivacyAuditor):
    """
    Membership Inference Attacker per Federated Learning.

    Intercetta i model update dei nodi PRIMA dell'aggregazione
    e stima il rischio che un avversario possa inferire la membership
    dei dati di training.

    Il rischio viene misurato tramite:
    1. Sensitivity del gradiente (norma L2) — proxy della memorizzazione
    2. Epsilon di DP consumato — misura della privacy residua
    3. Rilevazione di pattern anomali nei gradienti

    Uso tipico nel framework:
        auditor = PrivacyAuditor()
        # Chiamato per ogni nodo, per ogni round FL, PRIMA dell'aggregazione
        report = auditor.audit("highway-01", round_id=1, model_update={...})
        if report.threats_detected:
            # passa i dati a FedMIA per l'attacco completo (Sprint 4)
            pass
    """

    def __init__(self, config_path: str = "config/auditor.yaml", epsilon: float | None = None):
        """
        Inizializza il Privacy Auditor dalla configurazione YAML.

        Args:
            config_path: percorso al file auditor.yaml
            epsilon:     se fornito, sovrascrive il budget DP da config (usato negli sweep)
        """
        config = _load_auditor_config(config_path)

        # Budget di privacy differenziale configurato (epsilon CLI ha priorità)
        self._epsilon_budget: float = epsilon if epsilon is not None else config["dp"]["epsilon"]
        self._delta: float = config["dp"]["delta"]
        self._max_grad_norm: float = config["dp"]["max_grad_norm"]

        # Soglia oltre cui il nodo è considerato a rischio MIA
        self._alert_threshold: float = config["alert_threshold"]

        # Tipi di attacco abilitati (es. FedMIA)
        self._attack_types: list[str] = config.get("attacks", [])

        # Epsilon accumulato per nodo attraverso i round FL
        # {node_id: epsilon_totale_consumato}
        self._cumulative_epsilon: dict[str, float] = {}

    def audit(
        self,
        node_id: str,
        round_id: int,
        model_update: dict[str, Any],
    ) -> AuditReport:
        """
        Analizza il model update di un nodo e stima il rischio MIA.

        Questo metodo va chiamato PRIMA che il server FL aggreghi
        gli update — è il punto di intercettazione dell'attaccante.

        Fasi:
        1. Calcola sensitivity (norma L2 del gradiente)
        2. Stima epsilon consumato in questo round
        3. Aggiorna il budget cumulativo del nodo
        4. Rileva pattern sospetti (gradient explosion, budget esaurito)
        5. Calcola il privacy score residuo

        Args:
            node_id:      identificatore del nodo (es. "highway-01")
            round_id:     numero del round FL corrente
            model_update: pesi/gradienti del modello locale del nodo

        Returns:
            AuditReport con privacy_score, epsilon consumato, e minacce rilevate
        """
        # Step 1: sensitivity come norma L2 del gradiente
        sensitivity = self._compute_sensitivity(model_update)

        # Step 2: epsilon consumato in questo round
        # Formula semplificata Gaussian Mechanism: sensitivity / max_grad_norm
        # La versione completa con composizione arriva nella Sprint 4
        round_epsilon = sensitivity / self._max_grad_norm

        # Step 3: aggiorna epsilon cumulativo per questo nodo
        prev_epsilon = self._cumulative_epsilon.get(node_id, 0.0)
        self._cumulative_epsilon[node_id] = prev_epsilon + round_epsilon

        # Step 4: rileva minacce
        threats = self._detect_threats(
            sensitivity,
            self._cumulative_epsilon[node_id],
        )

        # Step 5: privacy score residuo
        # 1.0 = privacy intatta, 0.0 = budget esaurito
        budget_ratio = self._cumulative_epsilon[node_id] / self._epsilon_budget
        privacy_score = max(0.0, 1.0 - budget_ratio)

        return AuditReport(
            node_id=node_id,
            round_id=round_id,
            privacy_score=round(privacy_score, 4),
            epsilon=round(round_epsilon, 6),
            threats_detected=threats,
            metadata={
                "sensitivity": round(sensitivity, 6),
                "cumulative_epsilon": round(
                    self._cumulative_epsilon[node_id], 6
                ),
                "epsilon_budget": self._epsilon_budget,
                "budget_exhausted": (
                    self._cumulative_epsilon[node_id] >= self._epsilon_budget
                ),
            },
        )

    def _compute_sensitivity(self, model_update: dict[str, Any]) -> float:
        """
        Calcola la sensitivity del model update come norma L2 dei pesi.

        Alta sensitivity → il modello ha memorizzato fortemente i dati
        → alta vulnerabilità a membership inference.

        Args:
            model_update: dizionario layer → pesi

        Returns:
            norma L2 del vettore appiattito (0.0 se update vuoto)
        """
        flat = _flatten_model_update(model_update)
        if not flat:
            return 0.0
        return _compute_l2_norm(flat)

    def _detect_threats(
        self,
        sensitivity: float,
        cumulative_epsilon: float,
    ) -> list[str]:
        """
        Rileva pattern nei gradienti che indicano vulnerabilità MIA.

        Minacce rilevate:
        - GRADIENT_EXPLOSION: sensitivity >> max_grad_norm → possibile poisoning
        - PRIVACY_BUDGET_NEAR_EXHAUSTION: epsilon vicino al limite
        - PRIVACY_BUDGET_EXHAUSTED: budget completamente consumato
        - FEDMIA_SUSPICIOUS_LOW_SENSITIVITY: sensitivity anomalmente bassa
          (possibile tentativo di evasione dell'auditor)

        Args:
            sensitivity:        norma L2 del model update corrente
            cumulative_epsilon: epsilon totale consumato dal nodo

        Returns:
            lista di minacce rilevate (vuota se nessuna)
        """
        threats: list[str] = []

        # Gradient explosion: possibile model poisoning attack
        if sensitivity > self._max_grad_norm * 10:
            threats.append("GRADIENT_EXPLOSION")

        # Budget di privacy quasi esaurito
        budget_ratio = cumulative_epsilon / self._epsilon_budget
        if budget_ratio >= self._alert_threshold:
            threats.append("PRIVACY_BUDGET_NEAR_EXHAUSTION")

        # Budget completamente esaurito → nodo ad alto rischio MIA
        if cumulative_epsilon >= self._epsilon_budget:
            threats.append("PRIVACY_BUDGET_EXHAUSTED")

        # Pattern FedMIA: sensitivity sospettamente bassa
        # Il rilevamento completo arriva nella Sprint 4
        if "FedMIA" in self._attack_types and sensitivity < 1e-6:
            threats.append("FEDMIA_SUSPICIOUS_LOW_SENSITIVITY")

        return threats

    def reset(self) -> None:
        """
        Resetta l'epsilon cumulativo di tutti i nodi.
        Da chiamare tra esperimenti diversi per non contaminare i risultati.
        """
        self._cumulative_epsilon.clear()

    def get_cumulative_epsilon(self, node_id: str) -> float:
        """
        Restituisce l'epsilon totale consumato da un nodo.

        Args:
            node_id: identificatore del nodo

        Returns:
            epsilon cumulativo (0.0 se il nodo non ha ancora partecipato)
        """
        return self._cumulative_epsilon.get(node_id, 0.0)
