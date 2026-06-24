# ML Plane — Federated Learning in ChargeShield-FL

## Obiettivo

Addestrare un modello di anomaly detection distribuito
su 12 nodi di ricarica EV senza centralizzare i dati.

## Modello scelto: Autoencoder

### Motivazione
- I dati di training NON hanno label di anomalia (ACN-Data → `anomaly_label: None`)
- L'autoencoder apprende la distribuzione normale dei dati
- Le anomalie producono un errore di ricostruzione alto
- Si presta naturalmente a FL: ogni nodo addestra su dati locali

### Architettura (placeholder Sprint 4)
## Autoencoder — Implementazione Sprint 4

### Architettura
**File:** `src/core/autoencoder.py`

```python
Input  (7 feature numeriche normalizzate)
    ↓
Encoder:  7 → 16 → 8 → 4
    ↓
Latent space (4 dimensioni)
    ↓
Decoder:  4 → 8 → 16 → 7
    ↓
Output (ricostruzione)
    ↓
Reconstruction Error (MSE)
    ↓
Soglia anomalia (calibrata su validation set)
```

### Feature numeriche usate

| Feature | Preprocessing |
|---------|--------------|
| `total_energy_kwh` | MinMax [0,1] |
| `max_power_kw` | MinMax [0,1] |
| `kwh_requested` | MinMax [0,1] |
| `minutes_available` | MinMax [0,1] |
| `charging_mode` | one-hot (AC=0, DC=1) |
| `soc_percent` | MinMax [0,1] |
| `temperature_c` | MinMax [0,1] |

### Training FL

- Ogni nodo addestra l'autoencoder sui propri dati locali
- FedAvg aggrega i pesi dell'encoder e del decoder
- Il modello globale viene distribuito ai nodi ogni round
- La soglia di anomalia viene calibrata localmente

### Valutazione

- **Reconstruction Error** → MSE tra input e output
- **AUC-ROC** → per valutare la qualità del rilevamento
- **Privacy/Utility trade-off** → epsilon vs AUC-ROC
