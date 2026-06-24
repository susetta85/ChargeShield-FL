# ChargeShield-FL

**Privacy Auditing Framework for Federated Learning in EV Charging Networks**

ChargeShield-FL is an open-source framework for privacy auditing and
intrusion detection in Federated Learning (FL) environments, designed
for Operational Technology (OT) contexts — specifically EV charging
station networks.

---

## Overview

ChargeShield-FL simulates a real-world FL deployment across 12 EV
charging nodes organized in 4 clusters (Highway, Urban, Residential,
Corporate). It implements both attack and defense mechanisms for
membership inference in FL, following a clean Adapter Pattern
architecture with zero hardcoded values.

---

## Architecture
Core (abstract interfaces)

↓ Adapter Pattern

Nodes → Protocol Adapters → FL Layer → Privacy Auditor → IDS

**Core principles:**
- Core knows nothing about protocols
- Nodes know nothing about datasets
- Datasets know nothing about FL
- FL knows nothing about the Privacy Auditor

---

## Components

| Module | Role |
|--------|------|
| `src/core/` | Abstract interfaces (contracts) |
| `src/nodes/` | ChargingNode implementation |
| `src/adapters/` | OCPP 1.6 adapter, ACNDataset adapter |
| `src/auditor/` | PrivacyAuditor — Membership Inference Attacker |
| `src/ids/` | ChargingIDS — Real IDS (CUSUM, Krum, Cosine Similarity) |
| `src/flare/` | NVIDIA FLARE connector (FL rounds, FedAvg, DP) |
| `src/plugins/attacks/` | FedMIA — Federated Membership Inference Attack |
| `src/core/autoencoder.py` | PyTorch Autoencoder for anomaly detection |

---

## Threat Model

| Attack | Detector | Status |
|--------|----------|--------|
| Membership Inference (MIA) | PrivacyAuditor + FedMIA | ✅ Sprint 4 |
| Model Poisoning | ChargingIDS (Krum + Cosine) | ✅ Sprint 4 |
| Byzantine Fault | ChargingIDS (Krum) | ✅ Sprint 4 |
| Statistical Drift | ChargingIDS (CUSUM) | ✅ Sprint 4 |
| Eavesdropping | mTLS (Containerlab) | ✅ Sprint 3 |

---

## Dataset

**ACN-Data JPL** (Adaptive Charging Network, Caltech)
- 13,073 real EV charging sessions (2019 + 2020)
- Source: https://ev.caltech.edu/dataset
- Adapter: `src/adapters/acn_dataset.py`

---

## Infrastructure

- **Containerlab** topology: 12 nodes + aggregator + auditor + IDS
- **mTLS** between all components (auto-generated via `make certs`)
- **NVIDIA FLARE** for FL orchestration (Sprint 5)
- **OrbStack** for local container runtime (Sprint 5)

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/susetta85/ChargeShield-FL.git
cd ChargeShield-FL

# Install dependencies
pip install -e ".[dev]"

# Run tests
make test

# Generate mTLS certificates
make certs

# Build Docker images
make build

# Deploy Containerlab topology (requires OrbStack)
make deploy
```

---

## Development

```bash
# Run tests with coverage
make test-coverage

# Lint
make lint

# Run FL experiment
make experiment

# Teardown
make destroy
```

---

## Sprint Roadmap

| Sprint | Tag | Content | Status |
|--------|-----|---------|--------|
| 1 | v0.1.0-sprint1 | Repository, interfaces, YAML config, docs | ✅ |
| 2 | v0.2.0-sprint2 | ChargingNode, OCPP16, ACNDataset (13k sessions) | ✅ |
| 3 | v0.3.0-sprint3 | PrivacyAuditor, AbstractIDS, FLARE, Containerlab, mTLS, Docker | ✅ |
| 4 | v0.4.0-sprint4 | FedMIA, ChargingIDS, Autoencoder, 52 tests | ✅ |
| 5 | v0.5.0-sprint5 | OrbStack deploy, real FLARE, experiments | 🔄 |

---

## References

- Shokri et al., *Membership Inference Attacks Against ML Models*, IEEE S&P 2017
- Blanchard et al., *Byzantine Tolerant SGD*, NeurIPS 2017
- McMahan et al., *Communication-Efficient Learning of Deep Networks*, AISTATS 2017
- Dwork & Roth, *Algorithmic Foundations of Differential Privacy*, 2014
- Page, *Continuous Inspection Schemes*, Biometrika 1954 (CUSUM)

---

## License

Apache 2.0
