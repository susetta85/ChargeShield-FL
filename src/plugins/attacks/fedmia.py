# src/plugins/attacks/fedmia.py
"""
FedMIA — Federated Membership Inference Attack
===============================================
Implementa l'attacco di Membership Inference Attack (MIA)
nel contesto del Federated Learning per colonnine EV.

Ruolo nel framework:
- È un ATTACCO, non una difesa
- Tenta di inferire se un campione specifico è stato usato
  nel training di un nodo FL analizzando i gradienti
- Usa un shadow model (Autoencoder) addestrato su dati pubblici
  per distinguere gradienti di campioni "membri" da "non membri"

Come funziona:
1. Addestra un shadow model su un dataset pubblico (ACN-Data)
2. Genera gradienti di riferimento per campioni noti (membri e non membri)
3. Per ogni nodo target, confronta i gradienti con quelli di riferimento
4. Calcola un membership score: 0.0 = non membro, 1.0 = membro

Relazione con PrivacyAuditor:
- PrivacyAuditor intercetta i gradienti e misura il rischio
- FedMIA esegue l'attacco completo sui gradienti intercettati
- L'output (MIAResult) viene passato a ChargingIDS per la difesa

Riferimenti:
- Shokri et al., "Membership Inference Attacks Against ML Models",
  IEEE S&P 2017
- Nasr et al., "Comprehensive Privacy Analysis of Deep Learning",
  IEEE S&P 2019
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any

from core.autoencoder import Autoencoder


@dataclass
class MIAResult:
    """
    Risultato di un attacco FedMIA su un nodo.

    Attributes:
        node_id:          nodo analizzato
        round_id:         round FL in cui è stato eseguito l'attacco
        membership_score: probabilità che il campione sia membro [0.0, 1.0]
        is_member:        True se membership_score > attack_threshold
        confidence:       confidenza della predizione [0.0, 1.0]
        metadata:         informazioni aggiuntive per il paper
    """
    node_id: str
    round_id: int
    membership_score: float
    is_member: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FedMIA:
    """
    Federated Membership Inference Attacker.

    Usa un shadow model (Autoencoder) addestrato su dati pubblici
    per inferire la membership dei campioni nei nodi FL target.

    Il principio chiave:
    - Un modello addestrato su un campione produce un errore di
      ricostruzione BASSO su quel campione (lo ha "memorizzato")
    - Un modello che non ha visto il campione produce un errore ALTO
    - FedMIA sfrutta questa differenza per inferire la membership

    Uso tipico:
        fedmia = FedMIA()
        fedmia.train_shadow_model(public_dataset_loader)
        result = fedmia.run_attack("highway-01", round_id=1, gradients={...})
        if result.is_member:
            # il campione era probabilmente nel training set del nodo
    """

    def __init__(
        self,
        attack_threshold: float = 0.5,
        input_dim: int = 6,
        device: str | None = None,
    ):
        """
        Inizializza FedMIA con shadow model e device.

        Args:
            attack_threshold: soglia membership score per classificare
                              un campione come membro (default 0.5)
            device:           'cuda' | 'cpu' | None (auto-detect)
        """
        # Auto-detect device: usa GPU se disponibile
        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        # Shadow model: stesso autoencoder usato dai nodi target
        # Viene addestrato su dati pubblici (ACN-Data JPL)
        #self._shadow_model = Autoencoder().to(self._device)
        self._shadow_model = Autoencoder(input_dim=input_dim).to(self._device)
        # Soglia per classificare un campione come membro
        self._attack_threshold = attack_threshold

        # Flag: il shadow model deve essere addestrato prima dell'attacco
        self._shadow_trained = False

        # Errori di riferimento per calibrare il membership score
        # {member: lista errori, non_member: lista errori}
        self._reference_errors: dict[str, list[float]] = {
            "member": [],
            "non_member": [],
        }

    def train_shadow_model(
        self,
        public_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> list[float]:
        """
        Addestra il shadow model su dati pubblici.

        Il shadow model simula il comportamento del modello target.
        Deve essere addestrato su dati della stessa distribuzione
        del dataset privato dei nodi (es. ACN-Data JPL 2019).

        Dopo il training, calibra gli errori di riferimento
        per distinguere membri da non membri.

        Args:
            public_loader:  DataLoader con dati pubblici ACN-Data
            epochs:         epoche di training (default 10)
            learning_rate:  learning rate (default 0.01)

        Returns:
            lista delle loss per epoca
        """
        losses = self._shadow_model.fit(
            public_loader,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        # Calibra errori di riferimento sui dati pubblici
        # I dati su cui il shadow model è stato addestrato
        # sono considerati "membri" di riferimento
        self._calibrate_reference_errors(public_loader)
        self._shadow_trained = True

        return losses

    def _calibrate_reference_errors(
        self,
        data_loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Calibra gli errori di riferimento per membri e non membri.

        Usa il shadow model per calcolare:
        - Errori sui dati di training (membri): tipicamente bassi
        - Errori su dati rumorosi (non membri simulati): tipicamente alti

        Questa calibrazione permette di normalizzare il membership score
        tra 0.0 e 1.0 in modo significativo.

        Args:
            data_loader: DataLoader con i dati di training del shadow model
        """
        self._shadow_model.eval()
        member_errors: list[float] = []
        non_member_errors: list[float] = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self._device)
                reconstruction = self._shadow_model(batch)
                errors = torch.mean((reconstruction - batch) ** 2, dim=1)
                member_errors.extend(errors.tolist())

                # Simula non-membri aggiungendo rumore gaussiano
                noisy_batch = batch + torch.randn_like(batch) * 0.5
                noisy_reconstruction = self._shadow_model(noisy_batch)
                noisy_errors = torch.mean(
                    (noisy_reconstruction - noisy_batch) ** 2, dim=1
                )
                non_member_errors.extend(noisy_errors.tolist())

        self._reference_errors["member"] = member_errors
        self._reference_errors["non_member"] = non_member_errors

    def compute_membership_score(
        self,
        gradient: dict[str, Any],
    ) -> float:
        """
        Calcola il membership score di un gradiente.

        Il membership score misura quanto il gradiente somiglia
        a quello di un campione membro rispetto a un non membro.

        Algoritmo:
        1. Estrae i valori numerici dal gradiente
        2. Calcola l'errore di ricostruzione con il shadow model
        3. Normalizza l'errore rispetto agli errori di riferimento
        4. Restituisce un score in [0.0, 1.0]
           (0.0 = certamente non membro, 1.0 = certamente membro)

        Args:
            gradient: dizionario layer → pesi/gradienti del nodo target

        Returns:
            membership score in [0.0, 1.0]
        """
        if not self._shadow_trained:
            raise RuntimeError(
                "Shadow model non addestrato. "
                "Chiama train_shadow_model() prima di run_attack()."
            )

        # Estrai valori numerici dal gradiente e crea tensore
        flat_values = self._extract_numeric_values(gradient)
        if not flat_values:
            return 0.0

        # Adatta la dimensione al modello (padding o troncamento)
        tensor = self._prepare_tensor(flat_values)

        # Calcola errore di ricostruzione con il shadow model
        self._shadow_model.eval()
        with torch.no_grad():
            tensor = tensor.to(self._device)
            reconstruction = self._shadow_model(tensor)
            error = torch.mean((reconstruction - tensor) ** 2).item()

        # Normalizza rispetto agli errori di riferimento
        score = self._normalize_score(error)
        return score

    def _extract_numeric_values(
        self,
        gradient: dict[str, Any],
    ) -> list[float]:
        """
        Estrae tutti i valori numerici dal dizionario gradiente.

        Args:
            gradient: dizionario layer → pesi

        Returns:
            lista piatta di float
        """
        values: list[float] = []
        for value in gradient.values():
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, list):
                values.extend(
                    float(v) for v in value
                    if isinstance(v, (int, float))
                )
        return values

    def _prepare_tensor(self, values: list[float]) -> torch.Tensor:
        """
        Prepara un tensore di dimensione INPUT_DIM dal gradiente.

        Tronca se il gradiente è troppo lungo,
        fa padding con zeri se è troppo corto.

        Args:
            values: lista di float estratti dal gradiente

        Returns:
            tensore di shape (1, INPUT_DIM)
        """
        from core.autoencoder import INPUT_DIM
        if len(values) >= INPUT_DIM:
            tensor_values = values[:INPUT_DIM]
        else:
            # Padding con zeri
            tensor_values = values + [0.0] * (INPUT_DIM - len(values))
        return torch.tensor(tensor_values, dtype=torch.float32).unsqueeze(0)

    def _normalize_score(self, error: float) -> float:
        """
        Normalizza l'errore di ricostruzione in un membership score [0.0, 1.0].

        Un errore basso → score alto (probabile membro)
        Un errore alto → score basso (probabile non membro)

        Usa gli errori di riferimento calibrati sul shadow model
        per normalizzare in modo significativo.

        Args:
            error: errore di ricostruzione MSE

        Returns:
            membership score in [0.0, 1.0]
        """
        if not self._reference_errors["member"]:
            # Fallback se non calibrato
            return 1.0 / (1.0 + error)

        member_mean = sum(self._reference_errors["member"]) / \
            len(self._reference_errors["member"])
        non_member_mean = sum(self._reference_errors["non_member"]) / \
            len(self._reference_errors["non_member"])

        # Interpolazione lineare tra errore membro e non membro
        # error vicino a member_mean → score vicino a 1.0
        # error vicino a non_member_mean → score vicino a 0.0
        if non_member_mean == member_mean:
            return 0.5

        score = (non_member_mean - error) / (non_member_mean - member_mean)
        return max(0.0, min(1.0, score))

    def run_attack(
        self,
        node_id: str,
        round_id: int,
        gradients: dict[str, Any],
    ) -> MIAResult:
        """
        Esegue l'attacco MIA completo su un nodo.

        Intercetta i gradienti del nodo target e produce
        un MIAResult con il membership score e la classificazione.

        Args:
            node_id:   identificatore del nodo target
            round_id:  round FL corrente
            gradients: model update del nodo (gradienti/pesi)

        Returns:
            MIAResult con membership_score, is_member, confidence
        """
        membership_score = self.compute_membership_score(gradients)
        is_member = membership_score > self._attack_threshold

        # Confidenza: distanza dalla soglia normalizzata in [0.0, 1.0]
        confidence = abs(membership_score - self._attack_threshold) / \
            max(self._attack_threshold, 1.0 - self._attack_threshold)

        return MIAResult(
            node_id=node_id,
            round_id=round_id,
            membership_score=round(membership_score, 4),
            is_member=is_member,
            confidence=round(confidence, 4),
            metadata={
                "attack_threshold": self._attack_threshold,
                "shadow_trained": self._shadow_trained,
                "device": str(self._device),
            },
        )
    def run_cluster_attack(
        self,
        cluster_id: str,
        round_id: int,
        cluster_gradients: dict[str, dict[str, Any]],
    ) -> list[MIAResult]:
        """
        Esegue FedMIA su tutti i nodi di un cluster.

        Confronta ogni nodo con la media del cluster —
        un nodo con membership score molto diverso
        dagli altri è più sospetto.

        Args:
            cluster_id:        identificatore del cluster
            round_id:          round FL corrente
            cluster_gradients: {node_id: gradients} per ogni nodo del cluster

        Returns:
            lista di MIAResult, uno per nodo, con cluster_deviation
        """
        # Step 1: calcola membership score per ogni nodo
        results: list[MIAResult] = []
        scores: dict[str, float] = {}

        for node_id, gradients in cluster_gradients.items():
            score = self.compute_membership_score(gradients)
            scores[node_id] = score

        # Step 2: calcola media e deviazione standard del cluster
        cluster_mean = sum(scores.values()) / len(scores)
        cluster_std = (
            sum((s - cluster_mean) ** 2 for s in scores.values()) / len(scores)
        ) ** 0.5

        # Step 3: per ogni nodo, calcola la deviazione dal cluster
        for node_id, score in scores.items():
            is_member = score > self._attack_threshold

            # Deviazione dal comportamento del cluster
            # Alta deviazione → nodo anomalo rispetto al cluster
            cluster_deviation = (
                abs(score - cluster_mean) / cluster_std
                if cluster_std > 0 else 0.0
            )

            confidence = abs(score - self._attack_threshold) / \
                max(self._attack_threshold, 1.0 - self._attack_threshold)

            results.append(MIAResult(
                node_id=node_id,
                round_id=round_id,
                membership_score=round(score, 4),
                is_member=is_member,
                confidence=round(confidence, 4),
                metadata={
                    "cluster_id": cluster_id,
                    "cluster_mean_score": round(cluster_mean, 4),
                    "cluster_std": round(cluster_std, 4),
                    "cluster_deviation": round(cluster_deviation, 4),
                    "attack_threshold": self._attack_threshold,
                    "device": str(self._device),
                },
            ))

        return results
