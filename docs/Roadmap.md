# ChargeShield-FL — Roadmap

## Sprint 1 — v0.1.0 ✅
Repository open-source, struttura, interfacce astratte, YAML config, docs.

## Sprint 2 — v0.2.0 ✅
ChargingNode, OCPP16Adapter, ACNDataset (13,073 sessioni reali JPL 2019+2020).

## Sprint 3 — v0.3.0 ✅
PrivacyAuditor (MIA attacker), AbstractIDS, FLAREConnector,
Containerlab topology 12 nodi, mTLS, Dockerfile tutti i componenti.

## Sprint 4 — v0.4.0 ✅
Autoencoder (PyTorch, device-agnostic), FedMIA completo,
ChargingIDS reale (CUSUM + Krum + Cosine Similarity),
52 test unitari, documentazione completa.

## Sprint 5 — v0.5.0 🔄
- ML Plane: layer trasversale L0→L3 (Purdue Model gap)
  - AutoencoderTrainer (FedAvg/FedProx, proximal_mu)
  - GradientManager (Differential Privacy, Gaussian Mechanism)
  - FedAvgAggregator (media pesata per n_samples)
  - MLPlaneEvent/Listener (observer pattern per Auditor e IDS)
- NVFLARE 2.7.2: project.yml, provisioning, Dockerfile.flare
- Containerlab: topology aggiornata con FL client separati per cluster
- Threat Model aggiornato: aggregatore honest-but-curious (Scenario 1),
  FedMIA su pesi post-DP, WireGuard + mTLS per il canale
- Case Studies: CS1 (ε vs AUC-ROC), CS2 (multi-cluster), CS3 (DP vs no-DP)
- run_experiment.py: sweep epsilon, FedMIA + IDS evaluation
- 140 test unitari (77 Sprint 4+5 + regression)

## Sprint 6 — v0.6.0 (pianificato)
- Deploy reale su OrbStack + Containerlab (`make build` + `make provision` + `make deploy`)
- Full sweep esperimenti: rounds ∈ {100, 200, 500, 1000} × ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- Heat map AUC-ROC generata da `scripts/compare_results.py`
- CS1: curva ε vs AUC-ROC (contribuzione principale paper)
- CS2: membership score per cluster (non-IID analysis)
- CS3: DP vs no-DP, FedAvg vs FedProx
- OCPP 2.0 Adapter completo
- MQTT Adapter completo
- Scenario 2 MIA (client curioso) — future work
- Dataset ElaadNL (se disponibile)
- Paper: risultati sperimentali, heat map, analisi privacy/utility trade-off
