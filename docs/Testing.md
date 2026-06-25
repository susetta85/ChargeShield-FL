# ChargeShield-FL — Test Documentation

Documentazione completa dei test unitari per tutti gli sprint.
Tutti i test si trovano in `tests/` e si eseguono con:

```bash
pytest tests/ -v --tb=short
```

---

## Sprint 4 — `tests/test_sprint4.py` (52 test)

### Autoencoder

| Test | Descrizione |
|---|---|
| `test_encoder_output_shape` | Verifica che Encoder produca output di dim 4 da input dim 7 |
| `test_decoder_output_shape` | Verifica che Decoder ricostruisca dim 7 da dim 4 |
| `test_autoencoder_forward` | Forward pass end-to-end: input e output stessa shape |
| `test_autoencoder_fit` | `fit()` su dati sintetici, loss decresce |
| `test_calibrate_threshold` | Soglia al 95° percentile calcolata correttamente |
| `test_get_set_weights` | Round-trip `get_weights()` → `set_weights()` conserva i valori |
| `test_device_agnostic` | Autoencoder funziona su CPU (e CUDA se disponibile) |

### FedMIA

| Test | Descrizione |
|---|---|
| `test_train_shadow_model` | Shadow model si addestra senza errori |
| `test_compute_membership_score` | Score in [0, 1] per ogni sessione |
| `test_run_attack_returns_result` | `run_attack()` restituisce `MIAResult` con auc_roc |
| `test_run_attack_auc_above_random` | AUC-ROC > 0.5 su dati members vs non-members |
| `test_run_cluster_attack` | Attacco per cluster restituisce risultati per ogni cluster_id |
| `test_cluster_deviation_in_metadata` | `cluster_deviation` presente nei metadata di `MIAResult` |
| `test_mia_result_dataclass` | Tutti i campi di `MIAResult` sono valorizzati correttamente |

### ChargingIDS

| Test | Descrizione |
|---|---|
| `test_ids_analyze_returns_alerts` | `analyze()` restituisce lista di `IDSAlert` |
| `test_ids_analyze_round` | `analyze_round()` restituisce `RoundAnalysis` |
| `test_ids_no_alert_clean_data` | Nessun alert su dati normali |
| `test_ids_alert_on_anomaly` | Alert generato su sessione anomala |

### CUSUMDetector

| Test | Descrizione |
|---|---|
| `test_cusum_no_drift_stable` | Nessuna drift su serie stabile |
| `test_cusum_detects_drift` | Drift rilevata su serie con shift improvviso |
| `test_cusum_warmup_period` | Nessun alert durante i primi 10 round (warmup) |
| `test_cusum_reset_after_detection` | Accumulatore resettato dopo detection |
| `test_cusum_threshold_respected` | Alert solo quando accumulatore supera threshold=5.0 |

### KrumDetector

| Test | Descrizione |
|---|---|
| `test_krum_scores_computed` | Score Krum calcolato per ogni nodo |
| `test_krum_selects_honest_node` | Con f=0 seleziona il nodo più vicino agli altri |
| `test_krum_detects_byzantine` | Nodo Byzantine (outlier) identificato correttamente |
| `test_krum_minimum_nodes` | Requisito n ≥ 2f+3 rispettato (f=0 con 4 nodi) |
| `test_krum_returns_alert` | `detect_byzantine()` restituisce `IDSAlert` su Byzantine |

### GradientAnalyzer

| Test | Descrizione |
|---|---|
| `test_cosine_similarity_identical` | Similarità coseno = 1.0 per vettori identici |
| `test_cosine_similarity_orthogonal` | Similarità coseno = 0.0 per vettori ortogonali |
| `test_cluster_cosine_analysis` | Analisi per cluster restituisce score per ogni cluster_id |
| `test_gradient_flatten` | Flatten corretto di pesi multi-layer |
| `test_low_cosine_triggers_alert` | Alert generato quando cosine < threshold=0.85 |

---

## Sprint 5 — `tests/test_sprint5.py`

### AutoencoderTrainer

| Test | Descrizione |
|---|---|
| `test_init_fedavg` | `proximal_mu=0.0` e `_global_weights=None` alla creazione |
| `test_init_fedprox` | `proximal_mu=0.01` impostato correttamente |
| `test_missing_config_raises` | `ValueError` se manca `input_dim` in config |
| `test_get_weights_returns_list` | Pesi restituiti come lista di `torch.Tensor` |
| `test_set_weights_roundtrip` | `set_weights()` carica correttamente i pesi nel modello |
| `test_set_weights_saves_global` | `_global_weights` salvato dopo `set_weights()` |
| `test_train_local_returns_update` | `train_local()` restituisce `GradientUpdate` con tutti i campi |
| `test_train_local_empty_sessions` | Sessioni vuote → `n_samples=0`, `loss=None` |
| `test_train_local_none_feature_skipped` | Sessioni con feature `None` scartate silenziosamente |
| `test_fedprox_term_applied` | Con `proximal_mu>0` il termine prossimale viene applicato |
| `test_ml_plane_event_emitted` | `gradient_upload` emesso sul ML Plane dopo training |
| `test_apply_global_model_emits_event` | `weight_download` emesso dopo `apply_global_model()` |

### GradientManager

| Test | Descrizione |
|---|---|
| `test_sigma_computed` | σ calcolato correttamente: `max_norm * sqrt(2*ln(1.25/δ)) / ε` |
| `test_missing_config_raises` | `ValueError` se manca `max_grad_norm` in config |
| `test_privatize_returns_update` | `privatize()` restituisce `GradientUpdate` con `dp_applied=True` |
| `test_privatize_changes_weights` | I pesi dopo DP sono diversi da quelli originali |
| `test_privatize_preserves_metadata` | `node_id`, `cluster_id`, `round_num`, `n_samples` invariati |
| `test_ml_plane_event_emitted` | Evento emesso a `purdue_level=2` dopo privatizzazione |
| `test_clipping_reduces_norm` | Norma L2 dei pesi clippati ≤ `max_grad_norm` |

### FedAvgAggregator

| Test | Descrizione |
|---|---|
| `test_aggregate_returns_result` | `aggregate()` restituisce `AggregatedUpdate` con campi corretti |
| `test_aggregate_weighted_average` | Media pesata 50/50 → pesi globali = 0.5 (verifica matematica) |
| `test_aggregate_below_min_returns_none` | `None` se partecipanti < `min_participants` |
| `test_aggregate_clears_buffer` | Buffer svuotato dopo `aggregate()` |
| `test_mean_loss_weighted` | Loss media pesata calcolata correttamente |
| `test_ml_plane_event_emitted` | Evento `aggregation` emesso a `purdue_level=3` |

---

## Esecuzione per sprint

```bash
# Solo Sprint 4
pytest tests/test_sprint4.py -v

# Solo Sprint 5
pytest tests/test_sprint5.py -v

# Tutti
pytest tests/ -v --tb=short
```

## Copertura attuale

| Sprint | File | Test |
|---|---|---|
| Sprint 4 | `test_sprint4.py` | 52 |
| Sprint 5 | `test_sprint5.py` | 25 |
| **Totale** | | **77** |

## Script di analisi

| Script | Descrizione |
|---|---|
| `scripts/run_experiment.py` | Esegue esperimento FedMIA + IDS, salva JSON in `experiments/` |
| `scripts/compare_results.py` | Confronta tutti i JSON in `experiments/`, produce heat map rounds×ε e CSV |

```bash
# Singolo esperimento
python scripts/run_experiment.py --config config/experiment.yaml

# Full sweep rounds × epsilon (Sprint 6)
make experiment-full-sweep

# Confronto risultati
python scripts/compare_results.py --output experiments/summary.csv
```
