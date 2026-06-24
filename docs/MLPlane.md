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
