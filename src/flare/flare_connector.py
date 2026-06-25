# src/flare/flare_connector.py
"""
FLARE Connector — NVIDIA FLARE Integration Layer
=================================================
Connette il framework ChargeShield-FL a NVIDIA FLARE
come sistema di Federated Learning.

Ruolo nel framework:
- Gestisce i round FL: raccolta update, aggregazione, distribuzione
- Applica Differential Privacy ai gradienti PRIMA dell'intercettazione
  del PrivacyAuditor (questo è l'ordine corretto: DP → Auditor → Aggregazione)
- NON conosce il dataset, il protocollo di rete, o l'IDS
- Comunica con i nodi tramite AbstractChargingNode
- Comunica con il PrivacyAuditor tramite AuditReport

Flusso di un round FL:
    1. Il server invia il modello globale a tutti i nodi
    2. Ogni nodo addestra localmente (local_epochs)
    3. Ogni nodo invia il model update al server
    4. DP aggiunge rumore al gradiente (contromisura MIA)
    5. PrivacyAuditor intercetta e analizza il gradiente
    6. FedAvg aggrega gli update approvati dall'IDS
    7. Il modello globale viene aggiornato

Nota Sprint 3:
    Questa è una implementazione simulata — non richiede
    un server FLARE attivo. L'integrazione reale con
    nvflare arriva nella Sprint 4 con Containerlab.

Riferimenti:
    - NVIDIA FLARE: https://nvflare.readthedocs.io
    - McMahan et al., "Communication-Efficient Learning of
      Deep Networks from Decentralized Data", AISTATS 2017 (FedAvg)
    - Dwork & Roth, "Algorithmic Foundations of DP", 2014
"""

import math
import random
import yaml
from pathlib import Path
from typing import Any

from core.base_node import AbstractChargingNode
from core.base_auditor import AbstractPrivacyAuditor, AuditReport


def _load_flare_config(config_path: str = "config/flare.yaml") -> dict:
    """
    Carica la configurazione FL da flare.yaml.
    Nessun valore hardcoded nel codice.

    Args:
        config_path: percorso al file flare.yaml

    Raises:
        FileNotFoundError: se il file non esiste
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"FLARE config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)["flare"]


def _add_gaussian_noise(
    model_update: dict[str, Any],
    noise_scale: float,
) -> dict[str, Any]:
    """
    Applica il Gaussian Mechanism di Differential Privacy.

    Aggiunge rumore gaussiano N(0, noise_scale^2) a tutti i valori
    numerici del model update PRIMA che vengano intercettati
    dal PrivacyAuditor. Questo è l'ordine corretto:
    DP protegge i dati → l'Auditor misura il rischio residuo.

    Args:
        model_update: dizionario layer → pesi/gradienti
        noise_scale:  deviazione standard del rumore gaussiano

    Returns:
        model update con rumore aggiunto (copia, non modifica in-place)
    """
    noisy: dict[str, Any] = {}
    for key, value in model_update.items():
        if isinstance(value, float):
            noisy[key] = value + random.gauss(0, noise_scale)
        elif isinstance(value, list):
            noisy[key] = [
                v + random.gauss(0, noise_scale)
                if isinstance(v, (int, float)) else v
                for v in value
            ]
        else:
            noisy[key] = value
    return noisy


def _fedavg(updates: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Implementa FedAvg: media pesata dei model update.

    Per semplicità, ogni nodo ha lo stesso peso (1/N).
    La versione pesata per dimensione del dataset arriva in Sprint 4.

    Args:
        updates: lista di model update dai nodi partecipanti

    Returns:
        model update aggregato (media aritmetica)

    Raises:
        ValueError: se la lista di update è vuota
    """
    if not updates:
        raise ValueError("Cannot aggregate empty list of updates.")

    n = len(updates)
    aggregated: dict[str, Any] = {}

    for key in updates[0]:
        values = [u[key] for u in updates if key in u]

        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = sum(values) / n

        elif all(isinstance(v, list) for v in values):
            # Media elemento per elemento tra le liste
            length = min(len(v) for v in values)
            aggregated[key] = [
                sum(v[i] for v in values) / n
                for i in range(length)
            ]
        else:
            # Campi non numerici: tieni il valore del primo nodo
            aggregated[key] = values[0]

    return aggregated


class FLAREConnector:
    """
    Connector per NVIDIA FLARE — gestisce i round FL simulati.

    In Sprint 3 questa classe simula il comportamento di FLARE
    senza richiedere un server attivo. In Sprint 4 verrà estesa
    per comunicare con il server FLARE reale via Containerlab.

    Uso tipico:
        connector = FLAREConnector(nodes, auditor)
        for round_id in range(connector.n_rounds):
            result = connector.run_round(round_id, global_model)
            global_model = result["aggregated_update"]
    """

    def __init__(
        self,
        nodes: list[AbstractChargingNode],
        auditor: AbstractPrivacyAuditor,
        config_path: str = "config/flare.yaml",
    ):
        """
        Inizializza il connector con i nodi partecipanti e l'auditor.

        Args:
            nodes:       lista di nodi FL partecipanti
            auditor:     istanza di PrivacyAuditor per intercettare i gradienti
            config_path: percorso al file flare.yaml
        """
        config = _load_flare_config(config_path)

        self.nodes = nodes
        self.auditor = auditor
        self.n_rounds: int = config["rounds"]
        self.min_clients: int = config["min_clients"]
        self.local_epochs: int = config["local_epochs"]
        self.learning_rate: float = config["learning_rate"]

        # Noise scale per Differential Privacy
        # Calibrata su max_grad_norm e epsilon dal config auditor
        self._noise_scale: float = 0.1  # placeholder — calibrazione in Sprint 4

        # Storico degli AuditReport per tutti i round
        self._audit_history: list[AuditReport] = []

    def run_round(
        self,
        round_id: int,
        global_model: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Esegue un singolo round FL.

        Fasi:
        1. Verifica quorum (min_clients nodi disponibili)
        2. Ogni nodo produce un model update locale
        3. DP aggiunge rumore ai gradienti
        4. PrivacyAuditor intercetta e analizza ogni update
        5. FedAvg aggrega gli update non esclusi
        6. Restituisce il risultato del round

        Args:
            round_id:     numero del round corrente
            global_model: modello globale corrente da distribuire ai nodi

        Returns:
            dizionario con:
            - aggregated_update: nuovo modello aggregato
            - audit_reports:     lista di AuditReport del round
            - excluded_nodes:    nodi esclusi dall'IDS
            - participating:     nodi che hanno partecipato
        """
        # Step 1: verifica nodi disponibili
        available = [
            n for n in self.nodes
            if n.get_status() == "online"
        ]

        if len(available) < self.min_clients:
            return {
                "aggregated_update": global_model,
                "audit_reports": [],
                "excluded_nodes": [],
                "participating": [],
                "skipped": True,
                "reason": f"Quorum non raggiunto: {len(available)}/{self.min_clients}",
            }

        # Step 2 + 3 + 4: raccolta update, DP, auditing
        approved_updates: list[dict[str, Any]] = []
        audit_reports: list[AuditReport] = []
        excluded_nodes: list[str] = []

        for node in available:
            # Simula training locale: collect + preprocess + update
            raw = node.collect_data()
            processed = node.preprocess(raw)

            # Model update simulato — in Sprint 4 sarà il gradiente reale
            local_update = self._simulate_local_update(global_model, processed)

            # Step 3: applica DP prima dell'intercettazione
            noisy_update = _add_gaussian_noise(local_update, self._noise_scale)

            # Step 4: PrivacyAuditor intercetta il gradiente rumoroso
            report = self.auditor.audit(
                node_id=node.config.node_id,
                round_id=round_id,
                model_update=noisy_update,
            )
            audit_reports.append(report)
            self._audit_history.append(report)

            # Escludi il nodo se il budget è esaurito o gradient explosion
            if "PRIVACY_BUDGET_EXHAUSTED" in report.threats_detected or \
               "GRADIENT_EXPLOSION" in report.threats_detected:
                excluded_nodes.append(node.config.node_id)
            else:
                approved_updates.append(noisy_update)

        # Step 5: FedAvg sugli update approvati
        if not approved_updates:
            aggregated = global_model
        else:
            aggregated = _fedavg(approved_updates)

        return {
            "aggregated_update": aggregated,
            "audit_reports": audit_reports,
            "excluded_nodes": excluded_nodes,
            "participating": [n.config.node_id for n in available],
        }

    def _simulate_local_update(
        self,
        global_model: dict[str, Any],
        node_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Simula un model update locale.

        In Sprint 4 questo metodo verrà sostituito dal training
        reale dell'autoencoder con PyTorch + FLARE.
        Per ora simula un gradiente piccolo attorno al modello globale.

        Args:
            global_model: pesi correnti del modello globale
            node_data:    dati preprocessati del nodo

        Returns:
            model update simulato (perturbazione del modello globale)
        """
        update: dict[str, Any] = {}
        for key, value in global_model.items():
            if isinstance(value, float):
                # Piccola perturbazione del peso globale
                update[key] = value + random.gauss(0, 0.01)
            elif isinstance(value, list):
                update[key] = [
                    v + random.gauss(0, 0.01)
                    if isinstance(v, (int, float)) else v
                    for v in value
                ]
            else:
                update[key] = value
        return update

    def get_audit_history(self) -> list[AuditReport]:
        """
        Restituisce lo storico di tutti gli AuditReport prodotti.
        Utile per analisi post-esperimento e per il paper.

        Returns:
            lista di AuditReport in ordine cronologico
        """
        return self._audit_history.copy()

    def reset(self) -> None:
        """
        Resetta il connector e l'auditor per un nuovo esperimento.
        Da chiamare tra esperimenti diversi.
        """
        self._audit_history.clear()
        self.auditor.reset()
