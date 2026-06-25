# ChargeShield-FL — ML Plane

## Ruolo nel sistema

Il ML Plane è un layer **trasversale** all'architettura ChargeShield-FL.
Attraversa verticalmente tutti i livelli Purdue (L0→L3/L4) catturando
il traffico ML — gradienti, delta pesi, metadati di round — che il
modello Purdue non contempla.

### Gap del modello Purdue

Il Purdue Model definisce livelli per il traffico OT (SCADA, MODBUS,
OCPP, MQTT) ma non contempla il traffico FL. I gradient update che
i nodi inviano all'aggregatore sono invisibili al Purdue Model: non
sono dati di processo, non sono comandi di controllo.

ML Plane colma questo gap: rende il traffico FL visibile e analizzabile,
abilitando FedMIA (attacco) e IDS (difesa) su un canale altrimenti cieco.
Purdue L3-4  ── Aggregatore (FedAvg/FedProx)

↑ ↑

Purdue L2    ── [ML PLANE] GradientManager + FedAvgAggregator

↑ ↑

Purdue L1    ── [ML PLANE] AutoencoderTrainer

↑ ↑

Purdue L0    ── EV Charger (OCPP/MQTT) → sessioni grezze

---

## Componenti

### `AbstractMLModel` — `src/ml/base_ml.py`

Interfaccia astratta del ML Plane. Definisce il contratto per tutti
i componenti: `get_weights()`, `set_weights()`, `train_step()`,
`emit_event()`, `subscribe()`.

Definisce anche i dataclass scambiati sul piano:

| Dataclass | Descrizione |
|---|---|
| `GradientUpdate` | Pesi locali + loss + n_samples da un nodo dopo training |
| `AggregatedUpdate` | Pesi globali + loss media dopo FedAvg |
| `MLPlaneEvent` | Evento emesso dal piano (gradient_upload, aggregation, weight_download) |

I listener (`MLPlaneListener`) si registrano via `subscribe()` e
ricevono ogni evento tramite `on_ml_event()`. Implementato da
`PrivacyAuditor` e `ChargingIDS`.

---

### `AutoencoderTrainer` — `src/ml/autoencoder_trainer.py`

**Purdue L0 → L1**

Esegue il training locale dell'Autoencoder su sessioni EV (ACN-Data).

**Modello:** Encoder `7→16→8→4` + Decoder `4→8→16→7` (MSE loss)

**Feature continue usate (7):**

| Feature | Tipo |
|---|---|
| `voltage_v` | float |
| `current_a` | float |
| `power_kw` | float |
| `energy_kwh` | float |
| `temperature_c` | float (None → sessione scartata) |
| `soc_percent` | float (None → sessione scartata) |
| `timestamp` | float |

Le sessioni con valori `None` nelle feature continue vengono scartate
silenziosamente — mai silent default.

**FedAvg vs FedProx:**

Controllato da `proximal_mu` in config:

- `proximal_mu: 0.0` → FedAvg puro (McMahan et al., 2017)
- `proximal_mu: > 0.0` → FedProx (Li et al., 2020)

FedProx aggiunge il termine prossimale alla loss locale:
loss_fedprox = loss_reconstruction + (mu/2) * ||w - w_global||²
Raccomandato per ACN-Data (dati non-IID: pattern diversi per cluster
Highway/Urban/Residential/Corporate).

**Flusso:**
sessions → _sessions_to_tensor() → DataLoader

→ train_step() × epochs

→ GradientUpdate

→ emit_event(gradient_upload, purdue_level=1)

---

### `GradientManager` — `src/ml/gradient_manager.py`

**Purdue L1 → L2**

Applica Differential Privacy (Gaussian Mechanism) ai pesi locali
prima che escano dal nodo verso l'aggregatore.

**Algoritmo:**

1. **Gradient clipping** — clippa la norma L2 globale a `max_grad_norm`
2. **Gaussian noise** — aggiunge rumore `N(0, σ²)` a ogni tensore

**Calcolo σ (Gaussian Mechanism):**
σ = max_grad_norm × √(2 × ln(1.25 / δ)) / ε

**Parametri da config (nessun hardcoded):**

| Parametro | Descrizione |
|---|---|
| `epsilon` | Budget privacy (più basso = più privacy) |
| `delta` | Probabilità fallimento DP |
| `max_grad_norm` | Soglia clipping norma L2 |

Dopo privatizzazione emette `MLPlaneEvent(gradient_upload, purdue_level=2)`
con `dp_applied: True` nei metadata.

---

### `FedAvgAggregator` — `src/ml/fedavg_aggregator.py`

**Purdue L2 → L3**

Aggrega i `GradientUpdate` privatizzati via FedAvg (media pesata per
`n_samples`). Produce `AggregatedUpdate` con pesi globali.

**Algoritmo FedAvg:**
w_global = Σ (n_i / N) × w_i
dove `n_i` = campioni del client i, `N` = campioni totali.

**Flusso:**
collect(GradientUpdate) × n_clients

→ aggregate(round_num)

→ AggregatedUpdate

→ emit_event(aggregation, purdue_level=3)

Se `n_participants < min_participants` → restituisce `None` (round saltato).

---

## Flusso completo ML Plane per round FL

Server invia global model → AutoencoderTrainer.apply_global_model()

→ emit(weight_download, L1)
Ogni client:

AutoencoderTrainer.train_local(sessions, round_num)

→ emit(gradient_upload, L1)

→ GradientManager.privatize(update)

→ emit(gradient_upload, L2)  ← PrivacyAuditor e IDS osservano qui
Server:

FedAvgAggregator.collect(update) × 4 cluster

FedAvgAggregator.aggregate(round_num)

→ emit(aggregation, L3)      ← PrivacyAuditor e IDS osservano qui


---

## Relazione con Auditor e IDS

| Componente | Ruolo | Osserva |
|---|---|---|
| `PrivacyAuditor` (FedMIA) | ATTACCANTE | `gradient_upload` L2 + `aggregation` L3 |
| `ChargingIDS` | DIFESA | `gradient_upload` L2 (Krum, Cosine, CUSUM) |

Entrambi implementano `MLPlaneListener` e si registrano via `subscribe()`.
Non conoscono i componenti ML Plane — ricevono solo `MLPlaneEvent`.

---

## Configurazione

In `config/experiment.yaml` sezione `ml`:

```yaml
ml:
  input_dim: 7
  lr: 0.001
  epochs: 3           # round di sviluppo; 10+ per paper
  batch_size: 32
  proximal_mu: 0.01   # 0.0 = FedAvg, >0 = FedProx
```

In `config/experiment.yaml` sezione `experiment` per DP:

```yaml
experiment:
  epsilon: 1.0
  delta: 1.0e-5
  max_grad_norm: 1.0
```

---

## Riferimenti

- McMahan et al., *Communication-Efficient Learning of Deep Networks
  from Decentralized Data*, AISTATS 2017 — FedAvg
- Li et al., *Federated Optimization in Heterogeneous Networks*,
  MLSys 2020 — FedProx
- Dwork et al., *The Algorithmic Foundations of Differential Privacy*,
  2014 — Gaussian Mechanism

