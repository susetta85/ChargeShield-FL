# src/core/autoencoder.py
"""
Autoencoder — Modello FL per Anomaly Detection
===============================================
Implementa un autoencoder PyTorch per il rilevamento di anomalie
nelle sessioni di ricarica EV.

Ruolo nel framework:
- È il modello ML distribuito tramite FL tra i client (i 3 siti reali
  ACN-Data — Caltech/JPL/Office1 — negli esperimenti di privacy; +2 client
  sintetici solo nello sweep IDS/Byzantine separato — aggiornato 2026-07-24,
  era "12 nodi" dal vecchio schema fittizio a 4 cluster ormai superato)
- Ogni nodo addestra l'autoencoder sui propri dati locali
- FedAvg aggrega i pesi di encoder e decoder
- Le anomalie vengono rilevate tramite errore di ricostruzione (MSE)

Perché un autoencoder:
- I dati ACN-Data non hanno label di anomalia (anomaly_label = None)
- L'autoencoder apprende la distribuzione normale dei dati
- Sessioni anomale producono un MSE alto → rilevate come anomalie
- Si presta naturalmente a FL: pesi aggregabili con FedAvg

Architettura:
    Input  (6 feature numeriche normalizzate)
        ↓
    Encoder:  6 → 16 → 8 → 4
        ↓
    Latent space (4 dimensioni)
        ↓
    Decoder:  4 → 8 → 16 → 6
        ↓
    Output (ricostruzione)
        ↓
    MSE → soglia anomalia

Feature numeriche (da AutoencoderTrainer.CONTINUOUS_FEATURES):
    total_energy_kwh, max_power_kw, kwh_requested,
    minutes_available, hour_of_day, duration_hours

Riferimenti:
    - Hinton & Salakhutdinov, "Reducing Dimensionality with NNs", Science 2006
    - Chalapathy & Chawla, "Deep Learning for Anomaly Detection", 2019
"""

import torch
import torch.nn as nn
from torch import Tensor


# Numero di feature numeriche in input all'autoencoder.
# Corrisponde alle feature dell'ACNDataset dopo preprocessing.
INPUT_DIM = 6


class Encoder(nn.Module):
    """
    Encoder dell'autoencoder: comprime l'input in uno spazio latente.

    Architettura: 6 → 16 → 8 → 4
    Usa ReLU nei layer intermedi. Il layer finale non ha attivazione:
    ReLU sull'ultimo layer comprime lo spazio latente in [0,+∞),
    dimezzando la capacità espressiva senza benefici architetturali.
    BatchNorm1d stabilizza il training in FL dove i dati locali
    possono avere distribuzioni molto diverse tra i nodi.
    """

    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = 4):
        """
        Args:
            input_dim:  dimensione dell'input (default 6 feature)
            latent_dim: dimensione dello spazio latente (default 4)
        """
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: 6 → 16
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Layer 2: 16 → 8
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            # Layer 3: 8 → 4 (spazio latente — no ReLU: preserva segno)
            nn.Linear(8, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass dell'encoder.

        Args:
            x: tensore input di shape (batch_size, input_dim)

        Returns:
            rappresentazione latente di shape (batch_size, latent_dim)
        """
        return self.network(x)


class Decoder(nn.Module):
    """
    Decoder dell'autoencoder: ricostruisce l'input dallo spazio latente.

    Architettura: 4 → 8 → 16 → 6
    Usa Sigmoid nell'ultimo layer perché le feature sono normalizzate [0,1].
    """

    def __init__(self, latent_dim: int = 4, output_dim: int = INPUT_DIM):
        """
        Args:
            latent_dim: dimensione dello spazio latente (default 4)
            output_dim: dimensione dell'output (deve essere = input_dim)
        """
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: 4 → 8
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            # Layer 2: 8 → 16
            nn.Linear(8, 16),
            nn.ReLU(),
            # Layer 3: 16 → 6 (ricostruzione)
            # Sigmoid: output in [0,1] — coerente con feature normalizzate
            nn.Linear(16, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass del decoder.

        Args:
            x: rappresentazione latente di shape (batch_size, latent_dim)

        Returns:
            ricostruzione di shape (batch_size, output_dim)
        """
        return self.network(x)


class Autoencoder(nn.Module):
    """
    Autoencoder completo per anomaly detection su sessioni EV.

    Combina Encoder e Decoder. Durante il training FL:
    - Ogni nodo addestra l'autoencoder sui propri dati locali
    - FLAREConnector estrae i pesi con state_dict()
    - FedAvg aggrega i pesi di tutti i nodi
    - Il modello globale viene caricato con load_state_dict()

    La soglia di anomalia viene calibrata localmente su ogni nodo
    usando il 95° percentile degli errori di ricostruzione
    sul validation set locale.

    Uso tipico:
        model = Autoencoder()
        model.fit(train_loader, epochs=5)
        error = model.reconstruction_error(sample_tensor)
        if model.is_anomaly(sample_tensor):
            # segnala anomalia
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = 4,
        threshold: float = 0.1,
    ):
        """
        Args:
            input_dim:  numero di feature in input (default 6)
            latent_dim: dimensione spazio latente (default 4)
            threshold:  soglia MSE per anomalia (calibrata con fit())
        """
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

        # Soglia di anomalia: MSE > threshold → anomalia
        # Viene aggiornata durante fit() con il 95° percentile
        self.threshold = threshold

        # Loss function: MSE tra input e ricostruzione
        self._criterion = nn.MSELoss(reduction="mean")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass completo: input → latent → ricostruzione.

        Args:
            x: tensore input di shape (batch_size, input_dim)

        Returns:
            ricostruzione di shape (batch_size, input_dim)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def reconstruction_error(self, x: Tensor) -> float:
        """
        Calcola l'errore di ricostruzione MSE per un campione.

        Un MSE alto indica che il campione è anomalo rispetto
        alla distribuzione appresa durante il training.
        Il training mode originale viene ripristinato al termine.

        Args:
            x: tensore input di shape (1, input_dim) o (input_dim,)

        Returns:
            MSE tra input e ricostruzione (float)
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                reconstruction = self.forward(x)
                error = self._criterion(reconstruction, x)
            return error.item()
        finally:
            self.train(was_training)

    def is_anomaly(self, x: Tensor) -> bool:
        """
        Decide se un campione è anomalo confrontando MSE con la soglia.

        Args:
            x: tensore input di shape (1, input_dim) o (input_dim,)

        Returns:
            True se il campione è anomalo, False altrimenti
        """
        return self.reconstruction_error(x) > self.threshold

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        learning_rate: float = 0.01,
    ) -> list[float]:
        """
        Addestra l'autoencoder sui dati locali del nodo.

        Calibra automaticamente la soglia di anomalia al termine
        del training usando il 95° percentile degli errori
        sul training set (assunzione: training set = dati normali).

        Args:
            train_loader:  DataLoader con i dati di training locali
            epochs:        numero di epoche locali (da config/flare.yaml)
            learning_rate: learning rate (da config/flare.yaml)

        Returns:
            lista degli errori medi per epoca (per logging)
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            batch_losses: list[float] = []

            for batch in train_loader:
                # Unpack tuple: DataLoader da TensorDataset restituisce (tensor,)
                # invece di un tensore raw. Supporta entrambi i casi.
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                optimizer.zero_grad()
                reconstruction = self.forward(batch)
                loss = self._criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            if not batch_losses:
                continue  # DataLoader vuoto — salta l'epoca senza ZeroDivisionError
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)

        # Calibra soglia: 95° percentile degli errori sul training set
        self.threshold = self._calibrate_threshold(train_loader)

        return epoch_losses

    def _calibrate_threshold(
        self,
        data_loader: torch.utils.data.DataLoader,
        percentile: float = 95.0,
    ) -> float:
        """
        Calibra la soglia di anomalia sul training set.

        Usa il percentile specificato degli errori di ricostruzione.
        Il 95° percentile significa che il 5% dei dati normali
        sarà classificato come anomalo (false positive rate = 5%).

        Args:
            data_loader: DataLoader con i dati di calibrazione
            percentile:  percentile da usare come soglia (default 95)

        Returns:
            soglia calibrata (float)
        """
        self.eval()
        errors: list[float] = []

        with torch.no_grad():
            for batch in data_loader:
                # Unpack tuple: DataLoader da TensorDataset restituisce (tensor,)
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                reconstruction = self.forward(batch)
                # Errore per campione (non per batch)
                batch_errors = torch.mean(
                    (reconstruction - batch) ** 2, dim=1
                )
                errors.extend(batch_errors.tolist())

        if not errors:
            # data_loader interamente vuoto (0 batch in ogni epoca) — torch.quantile()
            # su un tensore vuoto solleva RuntimeError ("quantile() input tensor must
            # be non-empty"). Riportato nella review v4 (2026-07-09) come residuo del
            # fix ZeroDivisionError in fit(): quel guard evita il crash nel training
            # loop, ma _calibrate_threshold() veniva chiamata comunque subito dopo con
            # lo stesso DataLoader vuoto. Fallback: mantieni la soglia corrente
            # (invariata rispetto a prima della chiamata) invece di far crashare fit().
            return self.threshold

        # Calcola il percentile degli errori
        errors_tensor = torch.tensor(errors)
        threshold = float(torch.quantile(errors_tensor, percentile / 100.0))
        return threshold

    def get_weights(self) -> dict:
        """
        Restituisce i pesi del modello per la trasmissione a FedAvg.

        Returns:
            state_dict con tutti i pesi del modello
        """
        return self.state_dict()

    def set_weights(self, weights: dict) -> None:
        """
        Carica i pesi aggregati dal server FL.

        Args:
            weights: state_dict ricevuto dall'aggregatore dopo FedAvg
        """
        self.load_state_dict(weights)
