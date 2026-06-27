# ChargeShield-FL Developer Guide

**Framework for Evaluating Membership Inference Attacks against Federated Learning in EV Charging Infrastructure**

*Target Venue: IEEE/IFIP International Conference on Dependable Systems and Networks (DSN 2027)*

---

> **Audience.** This guide targets contributors who extend, reproduce, or audit ChargeShield-FL. It assumes familiarity with Python 3.11+, federated learning fundamentals, and basic Docker/Linux tooling. It does *not* assume familiarity with OCPP, NVFLARE internals, or Containerlab.

---

## Table of Contents

1. [Philosophy and Design Principles](#1-philosophy-and-design-principles)
2. [Architectural Principles with Motivation](#2-architectural-principles-with-motivation)
3. [Repository Structure](#3-repository-structure)
4. [Dependency Injection and Configuration Flow](#4-dependency-injection-and-configuration-flow)
5. [Extension Points](#5-extension-points)
   - 5.1 [Adding a New Charging Node Type](#51-adding-a-new-charging-node-type)
   - 5.2 [Adding a New Protocol Adapter](#52-adding-a-new-protocol-adapter)
   - 5.3 [Adding a New Dataset](#53-adding-a-new-dataset)
   - 5.4 [Adding a New Attack Plugin](#54-adding-a-new-attack-plugin)
   - 5.5 [Adding a New IDS Detector](#55-adding-a-new-ids-detector)
6. [Configuration System](#6-configuration-system)
7. [Makefile Reference](#7-makefile-reference)
8. [Certificate Management](#8-certificate-management)
9. [Docker Infrastructure](#9-docker-infrastructure)
10. [Containerlab Topology](#10-containerlab-topology)
11. [NVFLARE Integration](#11-nvflare-integration)
12. [Testing Conventions](#12-testing-conventions)
13. [Commit Conventions](#13-commit-conventions)
14. [Sprint Workflow](#14-sprint-workflow)
15. [References](#15-references)

---

## 1. Philosophy and Design Principles

ChargeShield-FL is built on four guiding principles that collectively ensure the framework is suitable for rigorous, reproducible academic research.

### 1.1 Modularity

Every subsystem in ChargeShield-FL is expressed as an abstract base class whose concrete implementations are independently replaceable. The charging node, the communication protocol, the dataset loader, the federated learning algorithm, the differential privacy mechanism, and the membership inference attack plugin each live in separate Python modules with no circular dependencies. A researcher wishing to substitute FedAvg with a novel aggregation algorithm need touch exactly one file: the aggregator implementation. This property is not achieved by convention but enforced through the class hierarchy: concrete classes that violate the interface contract fail at construction time.

Modularity serves a dual purpose in the research context. First, it enables controlled experiments: when comparing two attack strategies, every variable except the attack plugin can be held constant by construction, because the rest of the stack is untouched. Second, it lowers the barrier for external reproducibility: a reviewer can audit the attack surface in isolation without needing to understand the entire codebase.

### 1.2 Reproducibility

All experimental parameters are externalised into YAML configuration files. No numerical constant, no file path, no network address, and no hyperparameter appears as a Python literal inside source code. Every experiment is identified by a deterministic hash of its configuration, and all outputs (model checkpoints, gradient logs, audit reports, attack results) are written to a subdirectory named after that hash. Re-running the same configuration on the same hardware must produce byte-identical results modulo floating-point non-determinism inherent to GPU execution.

Reproducibility is non-negotiable for DSN submission: referees increasingly require that experimental artefacts be deposited in a public archive and that results be replicable by running a single command. The `make experiment` target is designed to satisfy this requirement without modification.

### 1.3 Zero Hardcoded Values Policy

No value that could conceivably vary across experiments, deployments, or institutions may appear as a Python or YAML literal outside of the configuration layer. This policy extends to:

- IP addresses and port numbers (specified in `config/nodes/*.yaml`)
- Dataset file paths (specified in `config/experiment.yaml`)
- Differential privacy parameters epsilon, delta, and sigma (specified per-cluster)
- FedAvg/FedProx hyperparameters (rounds, local epochs, proximal_mu)
- MIA hyperparameters (shadow model architecture, attack threshold)
- Docker image tags and registry URLs

Violations of this policy are caught by the CI linter rule `no_hardcoded_config`, which greps for numeric literals in `src/` exceeding a configurable threshold. The sole exception is the integer `0` used as a default initialiser and the integer `1` used as a loop increment.

### 1.4 Separation of Concerns

The framework is organised into five concern layers, each ignorant of the layers above it:

```
Layer 0: Configuration   (YAML -> dataclasses)
Layer 1: Infrastructure  (Docker, Containerlab, mTLS, WireGuard)
Layer 2: Protocols       (OCPP 1.6, OCPP 2.0.1, MQTT v5)
Layer 3: Federated ML    (FedAvg, FedProx, gradient management)
Layer 4: Privacy Audit   (DP accounting, MIA plugins, IDS)
```

Each layer consumes the API of the layer directly beneath it and exposes an API to the layer directly above it. This strict layering is enforced by a dependency graph check run at import time: if any module in Layer N imports from Layer M where M > N, the framework raises `LayerViolationError` at startup.

---

## 2. Architectural Principles with Motivation

This section motivates the four primary architectural decisions that define the internal structure of ChargeShield-FL. Each decision imposes a constraint that initially appears restrictive; the motivation explains why that constraint serves the research agenda.

### 2.1 Core Knows Nothing About Protocols

**Principle.** The classes in `src/core/` — `BaseNode`, `BaseDataset`, `BaseAdapter`, `BaseAuditor`, `BaseIDS`, and `Autoencoder` — contain zero protocol-specific logic. They do not import `ocpp`, `paho.mqtt`, or any protocol library. They operate exclusively on abstract messages represented as Python dataclasses defined in `src/core/messages.py`.

**Motivation.** EV charging infrastructure is heterogeneous by design. A highway charging depot runs OCPP 1.6 over WebSocket. A residential cluster may run MQTT v5. A corporate fleet may run OCPP 2.0.1. If protocol-specific parsing were embedded in the core, every protocol change would require modifying classes that have already been validated through extensive testing. By keeping the core protocol-agnostic, a researcher can add support for ISO 15118-20 by writing a single new adapter class without touching a single line of validated core code. This is not merely an engineering convenience: it is a research validity guarantee. The membership inference attack, the differential privacy mechanism, and the IDS detector all operate on the same abstract message representation regardless of the underlying protocol, which ensures that experimental results are not confounded by protocol-layer artefacts.

**Enforcement.** The CI pipeline runs a post-import assertion verifying that no protocol library (ocpp, paho.mqtt) is transitively imported through the core package. Any protocol library imported through the core causes this assertion to fail.

### 2.2 Nodes Know Nothing About Datasets

**Principle.** The `ChargingNode` class in `src/nodes/charging_node.py` does not reference any dataset class. It receives data through the `BaseAdapter` interface, which abstracts over the origin of that data — whether it comes from the ACN-Data dataset, the ElaadNL dataset, a live OCPP connection, or a synthetic generator.

**Motivation.** In a membership inference attack evaluation framework, the choice of training dataset is a primary experimental variable. The adversarial strength of MIA varies significantly with dataset characteristics: session duration distribution, energy consumption variance, charging frequency per user, and temporal autocorrelation all affect the distinguishability of member from non-member samples. If the node class were coupled to a specific dataset, comparing attack efficacy across ACN-Data and ElaadNL would require duplicating the node implementation, introducing the risk that differences between the two node variants — rather than differences between the datasets — explain observed attack performance differences. The decoupling guarantees that the node's local training procedure is identical across datasets.

**Enforcement.** The module `src/nodes/charging_node.py` is subject to a static import analysis check that rejects any direct import from `src/adapters/`. The node may only import from `src/core/` and `src/ml/`.

### 2.3 Datasets Know Nothing About Federated Learning

**Principle.** The dataset adapter classes (`ACNDataset`, `ElaadNLDataset`) in `src/adapters/` return standard PyTorch `DataLoader` objects. They have no knowledge of federated learning rounds, aggregation protocols, or the NVFLARE runtime. They do not receive or emit gradient tensors.

**Motivation.** Federated learning research is a rapidly evolving field. FedAvg, introduced in 2017, has since been supplanted or augmented by FedProx, SCAFFOLD, FedNova, Mime, and dozens of other algorithms. If the dataset layer were coupled to the FL algorithm — for instance, if the dataset class managed the local training loop — then evaluating ChargeShield-FL with a new FL algorithm would require modifying the dataset class, risking the introduction of bugs into code that governs data loading and preprocessing. By keeping datasets as pure data providers, the FL algorithm is the sole locus of changes when comparing aggregation strategies. This cleanly isolates the FL algorithm as an independent variable in the experimental design.

**Enforcement.** The adapter modules must not import from `src/ml/` or reference any NVFLARE class. This is verified by the module-level import guard at the top of each adapter file.

### 2.4 FL Knows Nothing About the Privacy Auditor

**Principle.** The classes in `src/ml/` — `BaseMl`, `AutoencoderTrainer`, `GradientManager`, and `FedAvgAggregator` — do not import from `src/auditor/` or `src/plugins/`. They do not call the privacy auditor, do not invoke the MIA plugin, and do not inspect IDS output. They emit gradient tensors and model updates through a well-defined interface; the auditor observes this interface non-invasively.

**Motivation.** The privacy auditor is an experimental measurement instrument, not a component of the system under evaluation. Conflating the two would be analogous to an experimental physicist whose measurement apparatus modifies the system being measured. In the federated learning threat model, the MIA adversary is an external observer with access to model updates (in the passive gradient eavesdropping scenario) or a malicious aggregator (in the active scenario). If the FL code called into the auditor, the causal relationship between FL behaviour and audit outcome would be confounded: one could not determine whether the observed MIA success rate reflects a genuine privacy vulnerability in the FL algorithm or an artefact of the auditor's integration point. By enforcing strict separation, the auditor is free to be replaced by a stronger or weaker attack plugin without any modification to the FL code, enabling clean comparative evaluation of attack strategies.

**Enforcement.** The `src/ml/` package exposes gradient tensors through `GradientManager.snapshot()`, which writes to a versioned on-disk store. The auditor reads from this store independently. No direct Python call crosses the ML/auditor boundary at runtime.

---

## 3. Repository Structure

The following directory tree describes the complete layout of the ChargeShield-FL repository. Every directory and file is described in terms of its role, its dependencies, and the rationale for its placement.

```
chargeshield-fl/
├── config/
│   ├── experiment.yaml              # Global experiment parameters
│   └── nodes/
│       ├── cluster_a.yaml           # Highway cluster (4 nodes, OCPP 1.6, 150kW DC)
│       ├── cluster_b.yaml           # Urban cluster (3 nodes, OCPP 1.6, 22kW AC)
│       ├── cluster_c.yaml           # Residential cluster (3 nodes, MQTT v5, 7kW AC)
│       └── cluster_d.yaml           # Corporate cluster (2 nodes, OCPP 2.0.1, 50kW DC)
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_node.py             # Abstract base class for all charging nodes
│   │   ├── base_dataset.py          # Abstract base class for all dataset adapters
│   │   ├── base_adapter.py          # Abstract base class for protocol adapters
│   │   ├── base_auditor.py          # Abstract base class for the privacy auditor
│   │   ├── base_ids.py              # Abstract base class for intrusion detection
│   │   └── autoencoder.py           # Protocol-agnostic autoencoder model definition
│   ├── nodes/
│   │   ├── __init__.py
│   │   └── charging_node.py         # Concrete FL client node for EV charging
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── ocpp16_adapter.py        # OCPP 1.6 WebSocket protocol adapter
│   │   ├── acn_dataset.py           # ACN-Data (Caltech/JPL) dataset loader
│   │   └── elaadnl_dataset.py       # ElaadNL public dataset loader
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── base_ml.py               # Abstract local training interface
│   │   ├── autoencoder_trainer.py   # Concrete autoencoder local trainer
│   │   ├── gradient_manager.py      # Gradient snapshot/versioning/store
│   │   └── fedavg_aggregator.py     # FedAvg and FedProx server-side aggregation
│   ├── ids/
│   │   ├── __init__.py
│   │   └── charging_ids.py          # Autoencoder-based IDS for charging sessions
│   ├── plugins/
│   │   └── attacks/
│   │       ├── __init__.py
│   │       └── fedmia.py            # FedMIA membership inference attack plugin
│   ├── auditor/
│   │   ├── __init__.py
│   │   └── privacy_auditor.py       # Orchestrates DP accounting and MIA evaluation
│   └── flare/
│       ├── __init__.py
│       └── flare_connector.py       # NVFLARE runtime bridge and lifecycle hooks
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── test_sprint4.py              # 52 integration tests for Sprint 4 deliverables
│   └── test_sprint5.py              # 25 integration tests for Sprint 5 deliverables
├── infra/
│   ├── topology.clab.yml            # Containerlab network topology definition
│   ├── project.yml                  # NVFLARE provisioning manifest
│   ├── docker/
│   │   ├── Dockerfile.node          # FL client node image
│   │   ├── Dockerfile.server        # FL server/aggregator image
│   │   ├── Dockerfile.auditor       # Privacy auditor image
│   │   └── Dockerfile.ids           # IDS image
│   └── wireguard/
│       └── wg0.conf.template        # WireGuard VPN configuration template
├── certs/                           # Generated at runtime; gitignored
│   ├── ca/
│   ├── nodes/
│   └── server/
├── workspace/                       # NVFLARE provisioning output; gitignored
├── results/                         # Experiment outputs; gitignored
├── Makefile
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

### Directory Roles

**`config/`** — The sole location for all tunable parameters. The CI pipeline enforces that no file outside this directory defines experimental constants. The `experiment.yaml` file governs global parameters (FL rounds, DP budget, MIA configuration, output paths). Per-cluster YAML files govern node-specific parameters (IP address, protocol, power rating, dataset path, local training hyperparameters). The deliberate separation between global and per-cluster configuration reflects the physical reality of EV charging infrastructure: global FL policy (e.g., the number of aggregation rounds) is set by the fleet operator, while local parameters (e.g., the charging power rating) are fixed by hardware.

**`src/core/`** — The ontological foundation of the framework. Contains only abstract base classes and the autoencoder architecture definition. This directory has no dependencies on any external library except PyTorch (for tensor type annotations). Its stability is paramount: once a base class interface is published in a sprint, it must not be modified in a backward-incompatible way without a deprecation cycle.

**`src/nodes/`** — Contains exactly one concrete node class per EV charging scenario type. At the time of writing, `ChargingNode` is the single concrete implementation used for all four cluster types; the cluster-specific behaviour (protocol, power rating, local batch size) is injected through the configuration system rather than through separate subclasses. If a fundamentally different node behaviour is required — for example, a Vehicle-to-Grid (V2G) bidirectional node — a new concrete class would be added here.

**`src/adapters/`** — Protocol adapters and dataset loaders. This directory is the integration boundary between the physical world (real charging protocols and real datasets) and the abstract world (the core interfaces). Adapters are allowed to import protocol libraries (`ocpp`, `paho.mqtt`) and dataset libraries (`pandas`, `h5py`) that are forbidden in `src/core/` and `src/nodes/`.

**`src/ml/`** — Federated learning components: the local training loop, gradient management, and server-side aggregation. The `GradientManager` is architecturally critical: it maintains a versioned on-disk store of gradient snapshots indexed by (round, node_id, timestamp). This store is the only data pathway from the FL layer to the privacy audit layer, ensuring strict separation.

**`src/ids/`** — Intrusion detection subsystem. The `ChargingIDS` class wraps the autoencoder defined in `src/core/autoencoder.py` and adds threshold-based anomaly detection logic. It reads from the same gradient store as the auditor, but its operational role is detection rather than evaluation: it is intended to be deployed in production, whereas the auditor is an experimental instrument.

**`src/plugins/attacks/`** — Attack plugins. Each plugin implements the `BaseAttack` interface defined in `src/core/base_auditor.py`. The plugin directory is intentionally shallow: each attack is a single self-contained file. The `fedmia.py` plugin implements the FedMIA attack, which exploits gradient update magnitudes and direction cosines to distinguish members from non-members in the training set. New attacks are added by dropping a new file into this directory and registering the class name in `config/experiment.yaml`.

**`src/auditor/`** — The privacy auditor orchestrates the complete audit workflow: it invokes the configured attack plugin, computes differential privacy accounting (tracking (epsilon, delta) expenditure across rounds), and produces the structured audit report that constitutes a primary experimental output.

**`src/flare/`** — NVFLARE integration bridge. The `FlareConnector` wraps the NVFLARE client and server APIs, translating between NVFLARE's training task abstraction and ChargeShield-FL's internal interfaces. This isolation ensures that upgrading NVFLARE from version 2.7.2 to a future version requires changes only in this directory.

**`tests/`** — All pytest test files. The naming convention `test_sprint{N}.py` reflects the sprint-based development workflow: each sprint produces a corresponding test file that remains permanently in the repository. Tests are never deleted; if a sprint's deliverable is superseded, its tests are marked `xfail` with an explanatory message rather than removed.

**`infra/`** — All infrastructure-as-code artefacts: Containerlab topology, NVFLARE project manifest, Dockerfiles, and WireGuard templates.

**`certs/`**, **`workspace/`**, **`results/`** — Runtime-generated directories. All three are listed in `.gitignore`. The `certs/` directory contains mTLS certificates generated by `make certs`. The `workspace/` directory contains the NVFLARE provisioning output generated by `make provision`. The `results/` directory contains experiment outputs organised by configuration hash.

---

## 4. Dependency Injection and Configuration Flow

ChargeShield-FL uses a purely constructor-based dependency injection pattern. No global state, no singletons, no module-level configuration objects. This section traces the complete flow from YAML files on disk to instantiated runtime components.

### 4.1 Configuration Loading

Configuration is loaded in two phases. In the first phase, the `ConfigLoader` (located in `src/core/config.py`) reads `config/experiment.yaml` and all files matching `config/nodes/cluster_*.yaml`, merges them into a single validated configuration tree, and converts the result to a frozen Python dataclass hierarchy. Validation is performed using Pydantic v2: every field has a type annotation, and range constraints (e.g., `0 < epsilon <= 10.0`, `0 < delta < 1e-3`) are enforced at load time rather than at first use. A configuration that fails validation raises `ConfigValidationError` with a human-readable message identifying the offending field and its invalid value.

In the second phase, the top-level orchestrator reads the validated configuration tree and instantiates the component graph using constructor injection.

### 4.2 Configuration Hierarchy

```
ExperimentConfig
├── fl: FLConfig
│   ├── algorithm: str              # "fedavg" | "fedprox"
│   ├── rounds: int
│   ├── proximal_mu: float          # 0.0 for FedAvg, 0.01 for FedProx
│   └── aggregation_fraction: float
├── dp: DPConfig
│   ├── epsilon: float
│   ├── delta: float
│   ├── max_grad_norm: float
│   └── mechanism: str              # "gaussian"
├── mia: MIAConfig
│   ├── plugin: str                 # "fedmia" | custom
│   ├── shadow_model_epochs: int
│   └── attack_threshold: float
├── clusters: List[ClusterConfig]
│   └── ClusterConfig
│       ├── cluster_id: str
│       ├── protocol: str           # "ocpp16" | "ocpp201" | "mqtt5"
│       ├── power_kw: float
│       ├── nodes: List[NodeConfig]
│       └── dataset: DatasetConfig
│           ├── loader: str         # "acn" | "elaadnl"
│           ├── path: str
│           └── split: SplitConfig
└── infrastructure: InfraConfig
    ├── containerlab_topology: str
    ├── nvflare_project: str
    └── cert_dir: str
```

### 4.3 Component Instantiation Order

The orchestrator instantiates components in the following order, ensuring that every dependency is available before the component that requires it is constructed:

```python
# 1. Load and validate configuration
config = ConfigLoader.load("config/experiment.yaml", "config/nodes/")

# 2. Instantiate dataset adapters (no dependencies)
datasets = {
    cluster.cluster_id: DatasetFactory.create(cluster.dataset)
    for cluster in config.clusters
}

# 3. Instantiate protocol adapters (depend on cluster config)
adapters = {
    cluster.cluster_id: AdapterFactory.create(cluster.protocol, cluster)
    for cluster in config.clusters
}

# 4. Instantiate ML components (depend on config and dataset)
gradient_manager = GradientManager(config.fl, results_dir)
trainers = {
    node.node_id: AutoencoderTrainer(config.fl, datasets[cluster_id], gradient_manager)
    for cluster in config.clusters
    for node in cluster.nodes
}
aggregator = FedAvgAggregator(config.fl, gradient_manager)

# 5. Instantiate IDS (depends on autoencoder architecture from config)
ids_instances = {
    node.node_id: ChargingIDS(config, gradient_manager)
    for cluster in config.clusters
    for node in cluster.nodes
}

# 6. Instantiate attack plugin (depends on gradient_manager)
attack_plugin = AttackFactory.create(config.mia.plugin, config.mia, gradient_manager)

# 7. Instantiate auditor (depends on attack plugin and DP config)
auditor = PrivacyAuditor(config.dp, attack_plugin, results_dir)

# 8. Instantiate nodes (depend on adapter, trainer, IDS)
nodes = {
    node.node_id: ChargingNode(node, adapters[cluster_id], trainers[node.node_id],
                               ids_instances[node.node_id])
    for cluster in config.clusters
    for node in cluster.nodes
}

# 9. Instantiate FLARE connector (depends on all ML components)
connector = FlareConnector(config, nodes, aggregator, auditor)
```

This explicit instantiation order makes the dependency graph legible and eliminates any risk of initialisation-order bugs that afflict frameworks using lazy instantiation or service locators.

### 4.4 Differential Privacy Parameter Derivation

The Gaussian mechanism noise multiplier sigma is derived from the DP configuration parameters at instantiation time using the formula:

```
sigma = max_grad_norm * sqrt(2 * ln(1.25 / delta)) / epsilon
```

This derivation is performed once in `GradientManager.__init__()` and stored as `self._sigma`. The formula implements the standard Gaussian mechanism privacy guarantee: adding Gaussian noise with standard deviation `sigma * max_grad_norm` to the clipped gradient ensures (epsilon, delta)-differential privacy for a single round of gradient release. Across multiple rounds, privacy composition is tracked using the Renyi Differential Privacy accountant from the `autodp` library, which provides tighter bounds than the basic composition theorem.

**Example.** With `epsilon=1.0`, `delta=1e-5`, and `max_grad_norm=1.0`:

```
sigma = 1.0 * sqrt(2 * ln(1.25 / 1e-5)) / 1.0
      = sqrt(2 * ln(125000))
      = sqrt(2 * 11.736)
      ≈ sqrt(23.472)
      ≈ 4.845
```

The gradient noise standard deviation is therefore `4.845 * max_grad_norm = 4.845`.

---

## 5. Extension Points

ChargeShield-FL is designed to be extended without modifying existing code. This section provides step-by-step instructions and code skeletons for the five primary extension scenarios.

### 5.1 Adding a New Charging Node Type

**When to use this extension point.** A new node type is warranted when the node's local behaviour differs fundamentally from `ChargingNode` — for example, a Vehicle-to-Grid (V2G) node that participates in energy export, or a smart grid node that implements demand-response logic. If the difference is purely configurational (different power rating, different protocol), use the existing node type with a new cluster configuration file.

**Step 1: Create the node class.**

Create `src/nodes/<node_type>_node.py` inheriting from `BaseNode`:

```python
# src/nodes/v2g_node.py
from src.core.base_node import BaseNode
from src.core.base_adapter import BaseAdapter
from src.ml.base_ml import BaseMl
from src.core.base_ids import BaseIDS
from src.core.config import NodeConfig

class V2GNode(BaseNode):
    """
    Vehicle-to-Grid charging node.

    Extends ChargingNode with bidirectional energy flow logic.
    The local training procedure remains identical to ChargingNode;
    only the session feature extraction differs to include export_kwh.
    """

    def __init__(
        self,
        config: NodeConfig,
        adapter: BaseAdapter,
        trainer: BaseMl,
        ids: BaseIDS,
    ) -> None:
        super().__init__(config, adapter, trainer, ids)
        # Node-specific initialisation using config only; no literals.
        self._export_power_kw: float = config.export_power_kw

    def extract_features(self, raw_message: dict) -> dict:
        """
        Override to add the export_kwh feature to the standard feature vector.
        """
        features = super().extract_features(raw_message)
        features["export_kwh"] = raw_message.get("export_kwh", 0.0)
        return features

    # All abstract methods from BaseNode must be implemented.
    # Inherit local_train(), aggregate_update(), and anomaly_score()
    # from BaseNode's default implementations if they are sufficient.
```

**Step 2: Register the node type.**

Add the new node type to `src/nodes/__init__.py`:

```python
from src.nodes.charging_node import ChargingNode
from src.nodes.v2g_node import V2GNode

NODE_REGISTRY: dict[str, type] = {
    "charging": ChargingNode,
    "v2g": V2GNode,
}
```

**Step 3: Add a cluster configuration file.**

Create `config/nodes/cluster_e.yaml`:

```yaml
cluster_id: cluster_e
node_type: v2g          # Must match a key in NODE_REGISTRY
protocol: ocpp201
power_kw: 50.0
export_power_kw: 22.0   # V2G-specific parameter
nodes:
  - node_id: node_e1
    ip: 192.168.5.1
  - node_id: node_e2
    ip: 192.168.5.2
dataset:
  loader: acn
  path: /data/acn/v2g_sessions.h5
  split:
    train: 0.7
    val: 0.15
    test: 0.15
```

**Step 4: Add a Pydantic field for the new parameter.**

Extend `NodeConfig` in `src/core/config.py` with an `Optional` field:

```python
class NodeConfig(BaseModel):
    node_id: str
    ip: str
    export_power_kw: Optional[float] = None
```

**Step 5: Write tests.**

Add tests to a new `tests/test_v2g_node.py`:

```python
def test_v2g_node_feature_extraction_includes_export_kwh(v2g_node_fixture):
    features = v2g_node_fixture.extract_features({"export_kwh": 5.3})
    assert "export_kwh" in features
    assert features["export_kwh"] == pytest.approx(5.3)
```

**Step 6: Update Containerlab and NVFLARE.**

Add the new nodes to `infra/topology.clab.yml` and `infra/project.yml` following the naming conventions described in Sections 10 and 11.

---

### 5.2 Adding a New Protocol Adapter

**When to use this extension point.** A new protocol adapter is required when a charging cluster communicates using a protocol not yet supported: ISO 15118-20, EEBus, or a proprietary vendor protocol.

**Step 1: Create the adapter class.**

Create `src/adapters/<protocol>_adapter.py` inheriting from `BaseAdapter`:

```python
# src/adapters/iso15118_adapter.py
from src.core.base_adapter import BaseAdapter
from src.core.messages import ChargingMessage
from src.core.config import ClusterConfig

class ISO15118Adapter(BaseAdapter):
    """
    Protocol adapter for ISO 15118-20 (Vehicle-to-Grid Communication Interface).

    Translates ISO 15118-20 XML-based messages into the ChargeShield-FL
    abstract ChargingMessage format. Handles TLS 1.3 transport and
    plug-and-charge (PnC) certificate management internally.
    """

    def __init__(self, config: ClusterConfig) -> None:
        super().__init__(config)
        # Initialise protocol-specific state from config; no literals.
        self._pnc_enabled: bool = config.pnc_enabled
        self._tls_cert_path: str = config.tls_cert_path

    def connect(self) -> None:
        """Establish TLS 1.3 connection to the EVSE."""
        ...

    def disconnect(self) -> None:
        """Gracefully terminate the protocol session."""
        ...

    def send(self, message: ChargingMessage) -> None:
        """Translate ChargingMessage to ISO 15118-20 XML and send."""
        xml_payload = self._encode(message)
        self._transport.send(xml_payload)

    def receive(self) -> ChargingMessage:
        """Receive an ISO 15118-20 XML message and translate to ChargingMessage."""
        xml_payload = self._transport.recv()
        return self._decode(xml_payload)

    def _encode(self, message: ChargingMessage) -> bytes:
        # Protocol-specific encoding.
        ...

    def _decode(self, payload: bytes) -> ChargingMessage:
        # Protocol-specific decoding.
        ...
```

**Step 2: Register the adapter.**

Add to `src/adapters/__init__.py`:

```python
from src.adapters.ocpp16_adapter import OCPP16Adapter
from src.adapters.iso15118_adapter import ISO15118Adapter

ADAPTER_REGISTRY: dict[str, type] = {
    "ocpp16": OCPP16Adapter,
    "iso15118": ISO15118Adapter,
}
```

**Step 3: Reference the adapter in cluster configuration.**

In `config/nodes/cluster_f.yaml`:

```yaml
protocol: iso15118
pnc_enabled: true
tls_cert_path: /certs/cluster_f/client.pem
```

**Step 4: Add protocol-specific Pydantic fields to `ClusterConfig`.**

All protocol-specific fields must be `Optional` with sensible defaults so that existing cluster configurations remain valid.

**Step 5: Write tests covering encode/decode round-trip.**

```python
def test_iso15118_encode_decode_roundtrip(sample_charging_message):
    adapter = ISO15118Adapter(mock_cluster_config())
    encoded = adapter._encode(sample_charging_message)
    decoded = adapter._decode(encoded)
    assert decoded == sample_charging_message
```

---

### 5.3 Adding a New Dataset

**When to use this extension point.** A new dataset is required when evaluating MIA attack performance on a dataset with different statistical properties than ACN-Data or ElaadNL. Examples include the Pecan Street dataset, the NREL EV charging dataset, or a synthetic dataset generated by a simulation tool.

**Step 1: Create the dataset loader class.**

Create `src/adapters/<name>_dataset.py` inheriting from `BaseDataset`:

```python
# src/adapters/pecanstreet_dataset.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.core.base_dataset import BaseDataset
from src.core.config import DatasetConfig

class PecanStreetDataset(BaseDataset):
    """
    Dataset loader for the Pecan Street Dataport residential EV charging dataset.

    The Pecan Street dataset contains 1-minute interval smart meter data
    including EV charging sessions for approximately 1,000 residential
    customers in Austin, Texas and New York. The loader extracts per-session
    features: session_duration_min, energy_kwh, peak_kw, start_hour,
    day_of_week.

    Reference: Pecan Street Inc., Dataport, https://dataport.pecanstreet.org/
    """

    REQUIRED_COLUMNS = [
        "dataid", "localminute", "car1", "car2"
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._path: str = config.path
        self._split: dict = config.split.model_dump()
        self._df: pd.DataFrame | None = None

    def load(self) -> None:
        """Load raw CSV data from disk and validate schema."""
        self._df = pd.read_csv(self._path, parse_dates=["localminute"])
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self._df.columns]
        if missing:
            raise ValueError(f"Pecan Street dataset missing columns: {missing}")

    def preprocess(self) -> None:
        """Extract per-session features and normalise."""
        sessions = self._extract_sessions(self._df)
        self._features = self._normalise(sessions)

    def get_dataloader(self, split: str) -> DataLoader:
        """Return a PyTorch DataLoader for the specified split."""
        if self._features is None:
            raise RuntimeError("Call load() and preprocess() before get_dataloader().")
        tensor = torch.tensor(self._features[split], dtype=torch.float32)
        dataset = TensorDataset(tensor)
        return DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=(split == "train"),
            num_workers=self._config.num_workers,
            pin_memory=True,
        )

    def _extract_sessions(self, df: pd.DataFrame) -> dict:
        # Group by dataid, identify contiguous charging intervals,
        # compute session-level features.
        ...

    def _normalise(self, sessions: dict) -> dict:
        # z-score normalisation per feature column,
        # fit scaler on train split only.
        ...
```

**Step 2: Register the dataset loader.**

Add to `src/adapters/__init__.py`:

```python
DATASET_REGISTRY: dict[str, type] = {
    "acn": ACNDataset,
    "elaadnl": ElaadNLDataset,
    "pecanstreet": PecanStreetDataset,
}
```

**Step 3: Reference the dataset in configuration.**

```yaml
dataset:
  loader: pecanstreet
  path: /data/pecanstreet/15min_2022.csv
  batch_size: 64
  num_workers: 4
  split:
    train: 0.7
    val: 0.15
    test: 0.15
```

**Step 4: Document data provenance.**

Add a docstring citing the dataset's DOI or access URL, the version used, and the preprocessing steps applied. This is mandatory for DSN reproducibility requirements.

---

### 5.4 Adding a New Attack Plugin

**When to use this extension point.** A new attack plugin is required when evaluating a MIA strategy not yet implemented in the framework. This is the most common extension point for research contributions.

**Step 1: Create the attack plugin class.**

Create `src/plugins/attacks/<attack_name>.py` inheriting from `BaseAttack`:

```python
# src/plugins/attacks/nasr_mia.py
import numpy as np
import torch
from src.core.base_auditor import BaseAttack
from src.ml.gradient_manager import GradientManager
from src.core.config import MIAConfig

class NasrMIA(BaseAttack):
    """
    White-box membership inference attack after Nasr et al. (2019).

    This attack trains a binary classifier that receives as input the
    gradient norm, loss value, and per-layer gradient statistics of a
    target sample's forward-backward pass. It is more powerful than
    black-box attacks (which observe only model outputs) because it
    exploits gradient information available to a malicious aggregator
    or a passive gradient eavesdropper.

    Reference:
        Nasr, M., Shokri, R., & Houmansadr, A. (2019).
        Comprehensive privacy analysis of deep learning: Passive and
        active white-box inference attacks against centralized and
        federated learning.
        In IEEE S&P 2019.
    """

    def __init__(
        self,
        config: MIAConfig,
        gradient_manager: GradientManager,
    ) -> None:
        super().__init__(config, gradient_manager)
        self._shadow_epochs: int = config.shadow_model_epochs
        self._threshold: float = config.attack_threshold
        self._attack_model: torch.nn.Module | None = None

    def train_shadow_model(self, shadow_dataset) -> None:
        """
        Train the shadow model on a dataset with known membership labels.

        The shadow model must have the same architecture as the target FL model.
        """
        # 1. Train shadow FL model using shadow_dataset.
        # 2. Record gradient statistics for member and non-member samples.
        # 3. Train binary attack classifier on these statistics.
        ...

    def infer_membership(self, target_gradients: torch.Tensor) -> np.ndarray:
        """
        Infer membership for a batch of gradient tensors.

        Returns:
            Binary array of shape (batch_size,) where 1 indicates
            the attack predicts the corresponding sample is a training member.
        """
        if self._attack_model is None:
            raise RuntimeError("Call train_shadow_model() before infer_membership().")
        features = self._extract_attack_features(target_gradients)
        logits = self._attack_model(features)
        return (torch.sigmoid(logits) > self._threshold).numpy().astype(int)

    def compute_advantage(
        self,
        member_gradients: torch.Tensor,
        non_member_gradients: torch.Tensor,
    ) -> float:
        """
        Compute the MIA advantage: |TPR - FPR|.

        A value near 0 indicates the attack cannot distinguish members
        from non-members (strong privacy). A value near 1 indicates
        near-perfect membership inference (severe privacy violation).
        """
        member_preds = self.infer_membership(member_gradients)
        non_member_preds = self.infer_membership(non_member_gradients)
        tpr = member_preds.mean()
        fpr = non_member_preds.mean()
        return abs(tpr - fpr)

    def _extract_attack_features(
        self, gradients: torch.Tensor
    ) -> torch.Tensor:
        # Compute per-layer gradient norm, mean, variance, and L2 norm.
        ...
```

**Step 2: Register the attack plugin.**

Add to `src/plugins/attacks/__init__.py`:

```python
ATTACK_REGISTRY: dict[str, type] = {
    "fedmia": FedMIA,
    "nasr_mia": NasrMIA,
}
```

**Step 3: Reference the plugin in configuration.**

```yaml
mia:
  plugin: nasr_mia
  shadow_model_epochs: 50
  attack_threshold: 0.5
```

**Step 4: Implement the evaluation metrics.**

The `compute_advantage()` method is mandatory. Additionally implement `compute_auc()` and `compute_accuracy()` if the plugin supports them, as these are reported in the audit output.

---

### 5.5 Adding a New IDS Detector

**When to use this extension point.** A new IDS detector is required when the autoencoder-based anomaly detection in `ChargingIDS` is insufficient — for example, a one-class SVM detector, an isolation forest detector, or a sequence-based LSTM detector.

**Step 1: Create the IDS class.**

Create `src/ids/<detector_name>_ids.py` inheriting from `BaseIDS`:

```python
# src/ids/lstm_ids.py
import torch
import torch.nn as nn
from src.core.base_ids import BaseIDS
from src.core.config import IDSConfig

class LSTMChargeIDS(BaseIDS):
    """
    LSTM-based intrusion detection system for EV charging sessions.

    Models the temporal sequence of charging events to detect anomalies
    that are invisible to frame-level autoencoder detectors: for example,
    a slow-rate manipulation attack that alters energy readings by a small
    percentage over many sessions.

    The LSTM is trained on sequences of length config.sequence_length
    drawn from the normal training split of the dataset. At inference,
    reconstruction error of the last element conditioned on prior context
    is used as the anomaly score.
    """

    def __init__(self, config: IDSConfig) -> None:
        super().__init__(config)
        self._seq_len: int = config.sequence_length
        self._hidden_dim: int = config.hidden_dim
        self._threshold: float = config.anomaly_threshold
        self._model = self._build_model()

    def _build_model(self) -> nn.Module:
        return nn.LSTM(
            input_size=self._config.feature_dim,
            hidden_size=self._hidden_dim,
            num_layers=self._config.num_layers,
            batch_first=True,
        )

    def fit(self, dataloader) -> None:
        """Train the LSTM on normal charging sequences."""
        ...

    def anomaly_score(self, sequence: torch.Tensor) -> float:
        """Return the reconstruction error for the input sequence."""
        ...

    def is_anomalous(self, sequence: torch.Tensor) -> bool:
        """Return True if the anomaly score exceeds the configured threshold."""
        return self.anomaly_score(sequence) > self._threshold
```

**Step 2: Register the IDS.**

Add to `src/ids/__init__.py`:

```python
IDS_REGISTRY: dict[str, type] = {
    "autoencoder": ChargingIDS,
    "lstm": LSTMChargeIDS,
}
```

**Step 3: Reference the IDS in cluster configuration.**

```yaml
ids:
  detector: lstm
  sequence_length: 20
  hidden_dim: 64
  num_layers: 2
  anomaly_threshold: 0.035
  feature_dim: 5
```

---

## 6. Configuration System

### 6.1 Overview

ChargeShield-FL uses a two-level configuration system. The global configuration file `config/experiment.yaml` defines parameters that apply to the entire experiment. Per-cluster configuration files `config/nodes/cluster_{a,b,c,d}.yaml` define parameters specific to each cluster. All configuration files are validated against the Pydantic schema defined in `src/core/config.py` before any component is instantiated.

### 6.2 Global Configuration (`config/experiment.yaml`)

```yaml
# config/experiment.yaml
# Global experiment parameters for ChargeShield-FL.
# All values are validated against src/core/config.py:ExperimentConfig.

experiment:
  name: chargeshield_fl_dsn2027
  seed: 42                          # Global random seed for reproducibility
  results_dir: results/             # Relative to repository root; created at runtime
  log_level: INFO

fl:
  algorithm: fedavg                 # "fedavg" | "fedprox"
  rounds: 100
  proximal_mu: 0.0                  # 0.0 for FedAvg; set to 0.01 for FedProx
  aggregation_fraction: 1.0         # Fraction of nodes participating per round
  local_epochs: 5
  local_batch_size: 32

dp:
  mechanism: gaussian
  epsilon: 1.0
  delta: 1.0e-5
  max_grad_norm: 1.0
  # sigma is derived at runtime: max_grad_norm * sqrt(2*ln(1.25/delta)) / epsilon

mia:
  plugin: fedmia
  shadow_model_epochs: 100
  attack_threshold: 0.5
  evaluate_every_n_rounds: 10

ids:
  detector: autoencoder
  anomaly_threshold: 0.03

nvflare:
  project_yml: infra/project.yml
  workspace_dir: workspace/

containerlab:
  topology: infra/topology.clab.yml
```

### 6.3 Cluster Configuration (`config/nodes/cluster_a.yaml`)

```yaml
# config/nodes/cluster_a.yaml
# Highway charging cluster: 4 nodes, OCPP 1.6, 150kW DC fast charging.

cluster_id: cluster_a
cluster_name: highway
node_type: charging
protocol: ocpp16
power_kw: 150.0
charge_type: DC

nodes:
  - node_id: node_a1
    hostname: node-a1              # Must match NVFLARE project.yml and Containerlab
    ip: 192.168.1.1
    port: 9001
  - node_id: node_a2
    hostname: node-a2
    ip: 192.168.1.2
    port: 9002
  - node_id: node_a3
    hostname: node-a3
    ip: 192.168.1.3
    port: 9003
  - node_id: node_a4
    hostname: node-a4
    ip: 192.168.1.4
    port: 9004

dataset:
  loader: acn
  path: /data/acn/jpl_sessions.h5
  batch_size: 64
  num_workers: 4
  split:
    train: 0.7
    val: 0.15
    test: 0.15

local_training:
  epochs: 5
  learning_rate: 0.001
  weight_decay: 1.0e-4

ids:
  detector: autoencoder
  anomaly_threshold: 0.03
  feature_dim: 5
```

### 6.4 No Hardcoded Values Policy — Enforcement

The following practices are enforced by automated checks and code review policy:

1. **Numeric literals in `src/`.** The CI linter scans `src/` for floating-point literals and integer literals larger than 1 that are not used as loop bounds. Any match outside a test file causes the build to fail with a descriptive error message identifying the file and line.

2. **String literals that look like paths.** Any string literal matching the pattern of a Unix absolute path in `src/` is flagged. Paths must come from `config.*.path` attributes.

3. **IP address literals.** Any string matching the IPv4 pattern in `src/` is flagged.

4. **Port number literals.** Any integer in the range 1024-65535 in `src/` (excluding loop bounds) is flagged.

These checks are implemented in `scripts/lint_no_hardcoded.py` and run as a pre-commit hook and as a CI step before the test suite.

### 6.5 Configuration Validation

Configuration validation uses Pydantic v2 validators. Example validators:

```python
from pydantic import BaseModel, field_validator, model_validator
import math

class DPConfig(BaseModel):
    mechanism: str
    epsilon: float
    delta: float
    max_grad_norm: float

    @field_validator("epsilon")
    @classmethod
    def epsilon_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"epsilon must be positive; got {v}")
        return v

    @field_validator("delta")
    @classmethod
    def delta_must_be_valid_probability(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError(f"delta must be in (0, 1); got {v}")
        return v

    @model_validator(mode="after")
    def sigma_must_be_finite(self) -> "DPConfig":
        sigma = self.max_grad_norm * math.sqrt(
            2 * math.log(1.25 / self.delta)
        ) / self.epsilon
        if not math.isfinite(sigma):
            raise ValueError(
                f"Derived sigma is not finite: epsilon={self.epsilon}, "
                f"delta={self.delta}, max_grad_norm={self.max_grad_norm}"
            )
        return self
```

---

## 7. Makefile Reference

The `Makefile` provides a unified interface for all lifecycle operations. All targets that invoke external tools (Docker, Containerlab, NVFLARE) require that the corresponding tool be installed and in `PATH`. The `make help` target prints a formatted summary of all targets.

### 7.1 Infrastructure Lifecycle

| Target | Description |
|---|---|
| `make build` | Build all Docker images defined in `infra/docker/`. Images are tagged with the current Git commit hash. The build fails if any image fails to build cleanly. |
| `make provision` | Run the NVFLARE provisioning workflow: reads `infra/project.yml`, generates the `workspace/` directory with mTLS certificates, startup scripts, and configuration for all FL participants. Requires Python and `nvflare>=2.7.2`. |
| `make deploy` | Deploy the Containerlab topology defined in `infra/topology.clab.yml`. Creates Docker containers, virtual network interfaces, and routing tables as specified. Requires `containerlab>=0.50` and Docker. |
| `make destroy` | Destroy the Containerlab topology and stop all associated containers. Does not remove Docker images or provisioned workspace. |
| `make certs` | Generate the mTLS certificate authority and all node/server certificates. Outputs to `certs/`. This target is idempotent: if `certs/ca/ca.pem` already exists, it will not be regenerated unless `make certs FORCE=1` is specified. |
| `make status` | Display the current status of all Containerlab nodes (running/stopped/error) and the NVFLARE server status. |
| `make logs` | Tail the last 100 lines of logs from all running containers. Use `make logs NODE=node-a1` to tail a specific container. |

### 7.2 Experiment Operations

| Target | Description |
|---|---|
| `make experiment` | Run a single experiment using the configuration in `config/experiment.yaml`. Outputs are written to `results/<config_hash>/`. This target executes the complete pipeline: FL training, DP accounting, MIA evaluation, and audit report generation. |
| `make experiment-sweep` | Run a parameter sweep defined by the `sweep:` section of `config/experiment.yaml`. Each configuration combination is run sequentially, with results written to separate subdirectories. Use this target for ablation studies. |
| `make experiment-dry` | Validate configuration and print the complete instantiation plan without executing any FL training or infrastructure operations. Use this to verify that a new configuration is syntactically valid and that all referenced files exist. |

### 7.3 Testing

| Target | Description |
|---|---|
| `make test` | Run the complete pytest test suite. Exits with a non-zero code if any test fails. |
| `make test-sprint4` | Run only the tests in `tests/test_sprint4.py` (52 tests). Use during Sprint 4 development to get a fast feedback cycle. |
| `make test-sprint5` | Run only the tests in `tests/test_sprint5.py` (25 tests). |

### 7.4 Cleanup

| Target | Description |
|---|---|
| `make clean` | Remove all build artefacts: `__pycache__`, `.pyc` files, pytest cache, mypy cache. Does not remove `certs/`, `workspace/`, or `results/`. |
| `make clean-workspace` | Remove the `workspace/` directory generated by `make provision`. The next `make provision` will regenerate it from scratch. Use when changing `infra/project.yml`. |
| `make clean-experiments` | Remove the `results/` directory. **This operation is irreversible.** Experimental results that have not been backed up externally will be lost. The Makefile prompts for confirmation before executing. |

### 7.5 Help

| Target | Description |
|---|---|
| `make help` | Print a formatted table of all Makefile targets with one-line descriptions. This is the canonical reference for daily use; the table above provides the extended descriptions. |

---

## 8. Certificate Management

ChargeShield-FL uses mutual TLS (mTLS) for all inter-component communication: between FL clients and the FL server, between the auditor and the gradient store, and between the IDS and the charging nodes. This section describes the certificate authority hierarchy, the generation workflow, and the policies governing certificate storage and rotation.

### 8.1 Certificate Authority Hierarchy

ChargeShield-FL uses a two-tier certificate authority (CA) hierarchy:

```
Root CA (offline, 4096-bit RSA, validity: 10 years)
└── Intermediate CA (online, 2048-bit RSA, validity: 1 year)
    ├── FL Server certificate (2048-bit RSA, validity: 90 days)
    ├── node-a1 client certificate (2048-bit RSA, validity: 90 days)
    ├── node-a2 client certificate
    ├── ... (one per node, 12 total)
    └── auditor client certificate
```

The root CA private key is generated and used only during `make certs` and is not stored on any networked host. In production deployments, the root CA private key should be kept offline on an air-gapped machine or in a hardware security module (HSM). For the research prototype, it is stored at `certs/ca/ca.key` and listed in `.gitignore`.

### 8.2 Certificate Generation

`make certs` executes `scripts/generate_certs.sh`, which performs the following steps:

1. **Generate root CA key and self-signed certificate.**

```bash
openssl genrsa -out certs/ca/ca.key 4096
openssl req -new -x509 -days 3650 \
  -key certs/ca/ca.key \
  -out certs/ca/ca.pem \
  -subj "/C=IT/O=ChargeShield-FL/CN=ChargeShield-FL Root CA"
```

2. **Generate intermediate CA key, CSR, and certificate signed by root CA.**

```bash
openssl genrsa -out certs/ca/intermediate.key 2048
openssl req -new \
  -key certs/ca/intermediate.key \
  -out certs/ca/intermediate.csr \
  -subj "/C=IT/O=ChargeShield-FL/CN=ChargeShield-FL Intermediate CA"
openssl x509 -req -days 365 -CA certs/ca/ca.pem -CAkey certs/ca/ca.key \
  -in certs/ca/intermediate.csr -out certs/ca/intermediate.pem \
  -extensions v3_intermediate_ca -extfile scripts/openssl.cnf
```

3. **Generate per-node client certificates.** For each node hostname (read from `config/nodes/*.yaml`):

```bash
openssl genrsa -out certs/nodes/${NODE}.key 2048
openssl req -new \
  -key certs/nodes/${NODE}.key \
  -out certs/nodes/${NODE}.csr \
  -subj "/C=IT/O=ChargeShield-FL/CN=${NODE}"
openssl x509 -req -days 90 -CA certs/ca/intermediate.pem \
  -CAkey certs/ca/intermediate.key \
  -in certs/nodes/${NODE}.csr \
  -out certs/nodes/${NODE}.pem \
  -extensions v3_node_client -extfile scripts/openssl.cnf
```

4. **Generate FL server certificate with Subject Alternative Name entries.** The server certificate includes SAN entries for all hostnames and IP addresses at which the FL server is reachable, as required by modern TLS clients that do not fall back to Common Name matching.

### 8.3 Certificate Rotation

Certificates are valid for 90 days. The `make certs` target checks the expiry of all existing certificates and regenerates any certificate that will expire within 14 days. Regenerating a certificate invalidates all active FL sessions on that node; plan rotations during maintenance windows.

Automated rotation can be configured as a cron job:

```cron
0 2 * * * cd /path/to/chargeshield-fl && make certs >> /var/log/chargeshield-certs.log 2>&1
```

### 8.4 Gitignore Policy

The following certificate files are listed in `.gitignore` and must never be committed to the repository:

```gitignore
# Certificate Authority private keys
certs/ca/*.key

# Node private keys
certs/nodes/*.key

# Server private key
certs/server/*.key

# NVFLARE provisioning workspace (contains embedded certificates)
workspace/

# WireGuard private keys and rendered configs
infra/wireguard/*.key
infra/wireguard/wg*.conf
```

The public certificates (`.pem` files) may be committed for reference, but in practice they are regenerated by `make certs` on each deployment and are also excluded from the repository to avoid accumulating stale certificates.

### 8.5 WireGuard VPN

Inter-cluster communication traverses a WireGuard VPN overlay network. WireGuard keys are generated by `make provision` from the template `infra/wireguard/wg0.conf.template`. The template uses placeholders for all IP addresses and public keys, which are substituted from cluster configuration files during provisioning. WireGuard private keys are stored only on the corresponding node and are never logged or transmitted.

---

## 9. Docker Infrastructure

### 9.1 Image List

| Image | Dockerfile | Role |
|---|---|---|
| `chargeshield/node` | `infra/docker/Dockerfile.node` | FL client node. Runs the ChargingNode, AutoencoderTrainer, GradientManager, and ChargingIDS. One container per FL participant (12 total). |
| `chargeshield/server` | `infra/docker/Dockerfile.server` | FL aggregation server. Runs the NVFLARE server and FedAvgAggregator. One container per experiment. |
| `chargeshield/auditor` | `infra/docker/Dockerfile.auditor` | Privacy auditor. Runs the PrivacyAuditor and the configured attack plugin. One container per experiment. Reads the gradient store from a shared Docker volume. |
| `chargeshield/ids` | `infra/docker/Dockerfile.ids` | Standalone IDS evaluation container. Used for offline IDS performance evaluation; not deployed in federated training runs. |

### 9.2 Build

```bash
make build
```

This target builds all four images in dependency order. Each Dockerfile uses a multi-stage build: a `builder` stage installs build dependencies and compiles Python extensions, and a `runtime` stage copies only the compiled artefacts and runtime dependencies. This reduces the final image size and eliminates build tools from the attack surface.

All images are based on `python:3.11-slim-bookworm`. CUDA support is disabled by default; to enable GPU acceleration, set `GPU=1` in the Makefile invocation, which switches the base image to `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`.

Image tags follow the pattern `chargeshield/<role>:<git-commit-hash>`. The `latest` tag is also applied to the most recent successful build. All image names and tags are read from `config/experiment.yaml`; they are not hardcoded in the Makefile.

### 9.3 Deploy/Destroy Lifecycle

The deployment lifecycle is:

```
make certs        ->  Generate mTLS certificates
make provision    ->  Generate NVFLARE workspace
make build        ->  Build Docker images
make deploy       ->  Deploy Containerlab topology
make experiment   ->  Run FL experiment
make destroy      ->  Tear down Containerlab topology
```

`make destroy` is safe to run at any time. It calls `containerlab destroy -t infra/topology.clab.yml --cleanup`, which stops and removes all containers in the topology. It does not remove Docker images, provisioned workspace, or experiment results.

### 9.4 Volume and Network Architecture

Each node container mounts two volumes:

1. **Data volume** (`/data`): Read-only. Contains the dataset files for the node's cluster. Populated from the path specified in `config/nodes/cluster_*.yaml`.
2. **Results volume** (`/results`): Read-write. Shared between all containers in an experiment. The gradient store, model checkpoints, and audit reports are written here.

All containers are attached to a Docker network bridge created by Containerlab. Inter-container communication is mTLS-authenticated on port 8002 (NVFLARE default) with additional ports for protocol-specific traffic (OCPP WebSocket, MQTT), all specified in the cluster configuration files.

---

## 10. Containerlab Topology

### 10.1 Overview

Containerlab is used to create the virtual network environment in which the ChargeShield-FL experiment runs. It provisions Docker containers, connects them with virtual Ethernet interfaces, and configures IP addressing and routing. Containerlab was chosen over pure Docker Compose because it supports arbitrary L2/L3 topologies, which is necessary to simulate the geographic separation between charging clusters with different network latency characteristics.

### 10.2 Topology File Structure

The topology file `infra/topology.clab.yml` follows the Containerlab YAML schema. A representative excerpt for the four-cluster, twelve-node topology:

```yaml
# infra/topology.clab.yml
name: chargeshield-fl

topology:
  defaults:
    image: chargeshield/node:latest
    env:
      NVFLARE_WORKSPACE: /workspace
      LOG_LEVEL: INFO

  nodes:
    # FL Server
    fl-server:
      image: chargeshield/server:latest
      ports:
        - "8002:8002"   # NVFLARE gRPC
        - "8003:8003"   # NVFLARE admin
      env:
        ROLE: server

    # Privacy Auditor
    auditor:
      image: chargeshield/auditor:latest
      env:
        ROLE: auditor

    # Cluster A -- Highway (OCPP 1.6, 150kW DC)
    node-a1:
      image: chargeshield/node:latest
      env:
        CLUSTER_CONFIG: /config/nodes/cluster_a.yaml
        NODE_ID: node_a1
    node-a2:
      env:
        NODE_ID: node_a2
    node-a3:
      env:
        NODE_ID: node_a3
    node-a4:
      env:
        NODE_ID: node_a4

    # Cluster B -- Urban (OCPP 1.6, 22kW AC)
    node-b1:
      env:
        CLUSTER_CONFIG: /config/nodes/cluster_b.yaml
        NODE_ID: node_b1
    node-b2:
      env:
        NODE_ID: node_b2
    node-b3:
      env:
        NODE_ID: node_b3

    # Cluster C -- Residential (MQTT v5, 7kW AC)
    node-c1:
      env:
        CLUSTER_CONFIG: /config/nodes/cluster_c.yaml
        NODE_ID: node_c1
    node-c2:
      env:
        NODE_ID: node_c2
    node-c3:
      env:
        NODE_ID: node_c3

    # Cluster D -- Corporate (OCPP 2.0.1, 50kW DC)
    node-d1:
      env:
        CLUSTER_CONFIG: /config/nodes/cluster_d.yaml
        NODE_ID: node_d1
    node-d2:
      env:
        NODE_ID: node_d2

  links:
    # Each node connects to the FL server via a point-to-point link.
    - endpoints: ["fl-server:eth1", "node-a1:eth1"]
    - endpoints: ["fl-server:eth2", "node-a2:eth1"]
    - endpoints: ["fl-server:eth3", "node-a3:eth1"]
    - endpoints: ["fl-server:eth4", "node-a4:eth1"]
    - endpoints: ["fl-server:eth5", "node-b1:eth1"]
    - endpoints: ["fl-server:eth6", "node-b2:eth1"]
    - endpoints: ["fl-server:eth7", "node-b3:eth1"]
    - endpoints: ["fl-server:eth8", "node-c1:eth1"]
    - endpoints: ["fl-server:eth9", "node-c2:eth1"]
    - endpoints: ["fl-server:eth10", "node-c3:eth1"]
    - endpoints: ["fl-server:eth11", "node-d1:eth1"]
    - endpoints: ["fl-server:eth12", "node-d2:eth1"]
    - endpoints: ["fl-server:eth13", "auditor:eth1"]
```

### 10.3 Node Naming Convention

**Critical constraint.** The Containerlab node names (the keys under `topology.nodes`) must exactly match the hostnames used in the NVFLARE `project.yml` provisioning manifest and in the per-cluster YAML configuration files. This three-way consistency requirement is enforced by a pre-flight check in `scripts/validate_topology.py`, which is called by `make deploy` before Containerlab is invoked.

The naming convention is:

```
{role}-{cluster_letter}{index}
```

Where:
- `role` is always `node` for FL participants, `fl-server` for the aggregator, and `auditor` for the privacy auditor.
- `cluster_letter` is one of `a`, `b`, `c`, `d`.
- `index` is a 1-based integer with no zero-padding.

Examples: `node-a1`, `node-b3`, `node-d2`, `fl-server`, `auditor`.

This naming convention ensures that DNS resolution within the Containerlab network produces hostnames that NVFLARE can use as participant identifiers without any additional hostname mapping. Containerlab automatically creates DNS entries for all nodes in the topology using the node name as the hostname.

**The validation script `scripts/validate_topology.py` checks:**

1. Every hostname in `config/nodes/cluster_*.yaml` appears in `infra/topology.clab.yml`.
2. Every hostname in `infra/topology.clab.yml` (except `fl-server` and `auditor`) appears in `infra/project.yml`.
3. Every participant in `infra/project.yml` (except the server and admin) appears in `infra/topology.clab.yml`.
4. No hostname appears more than once across all three sources.

### 10.4 IP Addressing

| Subnet | Cluster | Purpose |
|---|---|---|
| `192.168.1.0/24` | Cluster A (Highway) | Node-to-server links for cluster A |
| `192.168.2.0/24` | Cluster B (Urban) | Node-to-server links for cluster B |
| `192.168.3.0/24` | Cluster C (Residential) | Node-to-server links for cluster C |
| `192.168.4.0/24` | Cluster D (Corporate) | Node-to-server links for cluster D |
| `192.168.0.0/24` | Management | FL server, auditor, admin console |

All IP addresses are specified in the per-cluster YAML files and in `topology.clab.yml`. No IP address appears more than once in the configuration; the validation script checks for conflicts.

---

## 11. NVFLARE Integration

### 11.1 Overview

NVFLARE 2.7.2 is the federated learning runtime used by ChargeShield-FL. NVFLARE handles the secure communication protocol between FL clients and the server, the coordination of training rounds, and the distribution of global model weights. ChargeShield-FL does not reimplement any of these functions; it integrates with NVFLARE through the `FlareConnector` class in `src/flare/flare_connector.py`.

### 11.2 Project Manifest (`infra/project.yml`)

The NVFLARE project manifest defines all participants in the FL experiment. It must be consistent with the Containerlab topology and the per-cluster YAML files. The `make provision` target reads this file and generates the `workspace/` directory.

```yaml
# infra/project.yml
api_version: 3
name: chargeshield_fl
description: ChargeShield-FL DSN 2027 Experiment

participants:
  # FL Server
  - name: fl-server
    type: server
    org: chargeshield
    protocol: grpc
    api_version: 3
    port: 8002
    admin_port: 8003

  # FL Clients -- must match Containerlab node names exactly
  - name: node-a1
    type: client
    org: chargeshield

  - name: node-a2
    type: client
    org: chargeshield

  - name: node-a3
    type: client
    org: chargeshield

  - name: node-a4
    type: client
    org: chargeshield

  - name: node-b1
    type: client
    org: chargeshield

  - name: node-b2
    type: client
    org: chargeshield

  - name: node-b3
    type: client
    org: chargeshield

  - name: node-c1
    type: client
    org: chargeshield

  - name: node-c2
    type: client
    org: chargeshield

  - name: node-c3
    type: client
    org: chargeshield

  - name: node-d1
    type: client
    org: chargeshield

  - name: node-d2
    type: client
    org: chargeshield

  # Admin console
  - name: admin@chargeshield
    type: admin
    org: chargeshield
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file: meta.json.template

  - path: nvflare.lighter.impl.cert.CertBuilder

  - path: nvflare.lighter.impl.signature.SignatureBuilder

  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
```

### 11.3 Provisioning Workflow

```bash
make provision
```

This target runs `nvflare provision -p infra/project.yml -w workspace/`. The provisioning process:

1. Reads `infra/project.yml` and generates one startup directory per participant under `workspace/`.
2. Generates mTLS certificates for each participant, signed by the NVFLARE internal CA.
3. Generates the `fed_server.json` and `fed_client.json` configuration files for each participant.
4. Generates the admin console startup script.

The `workspace/` directory structure after provisioning:

```
workspace/
├── fl-server/
│   ├── startup/
│   │   ├── start.sh
│   │   ├── fed_server.json
│   │   └── signature.pkl
│   └── local/
│       ├── server.crt
│       └── server.key
├── node-a1/
│   ├── startup/
│   │   ├── start.sh
│   │   ├── fed_client.json
│   │   └── signature.pkl
│   └── local/
│       ├── client.crt
│       └── client.key
├── ... (one directory per participant)
└── admin@chargeshield/
    └── startup/
        ├── fl_admin.sh
        └── signature.pkl
```

### 11.4 FedAvg and FedProx Configuration

ChargeShield-FL supports both FedAvg and FedProx through the same `FedAvgAggregator` class, which reads the `proximal_mu` parameter from configuration:

- **FedAvg** (`proximal_mu: 0.0`): Standard federated averaging. The local objective is the unmodified reconstruction loss on the local dataset. The aggregated model is the weighted average of client model updates, weighted by local dataset size.

- **FedProx** (`proximal_mu: 0.01`): Federated optimisation with proximal regularisation. The local objective adds a proximal term `(mu/2) * ||w - w_global||^2` that penalises the local model for diverging too far from the global model. This is particularly relevant for heterogeneous charging networks where data distributions differ significantly between clusters (e.g., highway fast-charging versus residential slow-charging).

The proximal term is implemented in `AutoencoderTrainer.local_step()`:

```python
def local_step(
    self,
    batch: torch.Tensor,
    global_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    reconstruction = self._model(batch)
    recon_loss = self._criterion(reconstruction, batch)
    if self._proximal_mu > 0.0:
        proximal_term = sum(
            torch.norm(p - global_params[name]) ** 2
            for name, p in self._model.named_parameters()
        )
        loss = recon_loss + (self._proximal_mu / 2.0) * proximal_term
    else:
        loss = recon_loss
    return loss
```

### 11.5 NVFLARE Admin Console

The NVFLARE admin console provides a command-line interface for monitoring and controlling a running FL experiment. After provisioning:

```bash
# Start the admin console
cd workspace/admin@chargeshield/startup && ./fl_admin.sh

# Example admin commands
> check_status server
> check_status client node-a1
> submit_job /path/to/job_config
> abort_job <job_id>
> shutdown server
```

The admin console communicates with the FL server over the admin port (configured in `project.yml`) using mTLS. The admin certificate is generated during `make provision` and stored in `workspace/admin@chargeshield/`.

---

## 12. Testing Conventions

### 12.1 Test Framework

ChargeShield-FL uses pytest as its test framework. All tests are located in the `tests/` directory. The test suite is designed to be runnable without any external infrastructure (no Docker, no Containerlab, no network connections) by using fixtures that mock all infrastructure dependencies.

### 12.2 Naming Conventions

| Convention | Rule | Example |
|---|---|---|
| Test file names | `test_sprint{N}.py` for sprint tests; `test_{module}.py` for unit tests | `test_sprint4.py`, `test_charging_node.py` |
| Test function names | `test_{what}_{condition}_{expected_outcome}` | `test_fedavg_aggregator_with_two_clients_returns_weighted_average` |
| Fixture names | `{component}_fixture` for component fixtures; `mock_{component}` for mocks | `charging_node_fixture`, `mock_gradient_manager` |
| Parametrised test IDs | `{parameter_name}={value}` | `algorithm=fedavg`, `epsilon=1.0` |

### 12.3 Fixture Strategy

All fixtures are defined in `tests/conftest.py` and organised by scope:

```python
# tests/conftest.py

import pytest
from unittest.mock import MagicMock
from src.core.config import ExperimentConfig, FLConfig, DPConfig, MIAConfig
from src.ml.gradient_manager import GradientManager


@pytest.fixture(scope="session")
def base_experiment_config() -> ExperimentConfig:
    """
    Minimal valid experiment configuration for testing.
    Values are chosen to be computationally inexpensive:
    - 2 FL rounds (not 100)
    - 1 local epoch (not 5)
    - Small model dimensions
    """
    return ExperimentConfig(
        fl=FLConfig(
            algorithm="fedavg",
            rounds=2,
            proximal_mu=0.0,
            aggregation_fraction=1.0,
            local_epochs=1,
            local_batch_size=4,
        ),
        dp=DPConfig(
            mechanism="gaussian",
            epsilon=1.0,
            delta=1.0e-5,
            max_grad_norm=1.0,
        ),
        mia=MIAConfig(
            plugin="fedmia",
            shadow_model_epochs=2,
            attack_threshold=0.5,
        ),
    )


@pytest.fixture(scope="function")
def mock_gradient_manager() -> MagicMock:
    """
    Mock GradientManager that records calls without performing disk I/O.
    """
    mgr = MagicMock(spec=GradientManager)
    mgr.snapshot.return_value = {"layer1.weight": __import__("torch").zeros(4, 4)}
    return mgr


@pytest.fixture(scope="function")
def tmp_results_dir(tmp_path) -> str:
    """Temporary directory for experiment results, cleaned up after each test."""
    results = tmp_path / "results"
    results.mkdir()
    return str(results)
```

The scope hierarchy is:
- `session` scope: Expensive-to-construct objects shared across all tests (e.g., base configuration, large test tensors).
- `module` scope: Objects that can be shared within a test file but should not persist across files.
- `function` scope (default): Objects that must be fresh for each test, typically mock objects and temporary directories.

### 12.4 Test Categories

Tests are categorised using pytest marks:

```python
@pytest.mark.unit        # Tests a single class or function in isolation
@pytest.mark.integration # Tests interaction between two or more components
@pytest.mark.slow        # Takes more than 5 seconds; excluded from quick CI runs
@pytest.mark.gpu         # Requires a CUDA-capable GPU; skipped in CPU-only CI
```

The default `make test` target runs all tests except those marked `gpu`. The CI pipeline runs `pytest -m "not gpu and not slow"` for every pull request and `pytest -m "not gpu"` for every merge to `main`.

### 12.5 Sprint Test Files

Each sprint produces a corresponding test file that remains permanently in the repository. Sprint test files test the deliverables of that sprint as specified in the sprint definition. They serve two purposes: verification that the sprint deliverables are met before the sprint is closed, and regression detection in future sprints.

- `tests/test_sprint4.py` — 52 tests covering FL training loop, gradient management, DP noise injection, FedAvg/FedProx aggregation, and NVFLARE integration.
- `tests/test_sprint5.py` — 25 tests covering FedMIA attack plugin, privacy auditor, audit report generation, and IDS evaluation metrics.

If a future sprint changes the behaviour of a Sprint 4 deliverable, the corresponding Sprint 4 test should be updated (not deleted) to reflect the new expected behaviour, with a comment explaining why the expectation changed.

### 12.6 Coverage Requirements

The minimum acceptable test coverage for `src/` is 80%, measured by `pytest --cov=src --cov-report=term-missing`. Coverage is checked in CI and the build fails if coverage drops below this threshold. Modules in `src/flare/` are excluded from coverage measurement because their integration tests require a live NVFLARE runtime.

---

## 13. Commit Conventions

ChargeShield-FL follows a structured commit message convention based on a subset of the Conventional Commits specification. All commits must be prefixed with one of the following types:

| Prefix | Meaning | Example |
|---|---|---|
| `feat` | A new feature or capability | `feat: add FedProx aggregation with proximal_mu support` |
| `fix` | A bug fix | `fix: correct sigma derivation in GradientManager init` |
| `config` | A change to configuration files only | `config: set FedProx proximal_mu to 0.01 in cluster_a.yaml` |
| `docs` | Documentation changes only | `docs: add LSTM IDS extension example to DeveloperGuide` |
| `test` | Adding or modifying tests | `test: add sprint5 tests for FedMIA advantage computation` |
| `refactor` | Code restructuring without behaviour change | `refactor: extract feature computation to BaseAttack mixin` |
| `infra` | Infrastructure changes (Dockerfile, Containerlab, NVFLARE) | `infra: add auditor container to topology.clab.yml` |
| `ci` | CI/CD pipeline changes | `ci: add GPU test job to GitHub Actions workflow` |

### 13.1 Commit Message Format

```
<type>: <short imperative description>

<Optional longer description. Explain WHY the change is needed,
not WHAT the change does -- the diff shows what. If the change
fixes a bug, describe the root cause and how the fix addresses it.>

<Optional: references to issues, papers, or sprint items>
Refs: #42, Nasr et al. (2019) S&P
```

### 13.2 Scope and Atomicity

Each commit must be atomic: it should implement exactly one logical change. A commit that adds a new dataset loader and also fixes a typo in a comment violates atomicity; split such changes into two commits.

Commits that touch `src/` must include or be paired with a corresponding test commit. The CI pipeline enforces that no Python source file in `src/` is modified by a commit unless the test coverage of that file is non-decreasing (measured by `pytest --cov`).

---

## 14. Sprint Workflow

ChargeShield-FL is developed in two-week sprints. Each sprint has a defined set of deliverables that must be demonstrated in a sprint review before the sprint is closed. This section defines the criteria that each sprint must satisfy before closing.

### 14.1 Sprint Closure Criteria

A sprint may be closed only when all of the following conditions are met:

1. **All sprint deliverables are implemented.** Every feature listed in the sprint backlog has a corresponding implementation in the codebase.

2. **All sprint tests pass.** The sprint's test file (`test_sprint{N}.py`) passes with zero failures and zero errors. Skipped tests are acceptable only if they are marked `@pytest.mark.skip(reason="...")` with a non-trivial reason.

3. **No regressions.** All tests from previous sprints continue to pass. The `make test` target reports zero failures.

4. **No hardcoded values.** The CI linter `scripts/lint_no_hardcoded.py` reports zero violations in `src/`.

5. **Configuration validated.** The `make experiment-dry` target completes without errors, confirming that the configuration system correctly validates the sprint's new configuration parameters.

6. **Documentation updated.** This Developer Guide, the inline docstrings of all new classes and methods, and the `CHANGELOG.md` have been updated to reflect the sprint's deliverables.

7. **Sprint review conducted.** The team has reviewed the sprint deliverables in a synchronous or asynchronous sprint review.

### 14.2 Sprint Deliverable Template

Each sprint's deliverables are specified using the following template in the sprint planning document:

```
Sprint N: <Sprint Title>

Goals:
  - <Primary goal>
  - <Secondary goal>

Deliverables:
  - [ ] <Concrete deliverable 1>
  - [ ] <Concrete deliverable 2>
  - ...

Test file: tests/test_sprint{N}.py
Expected test count: <N> tests

Definition of Done:
  All items above are checked, all tests pass, no regressions,
  no hardcoded values, documentation updated.
```

### 14.3 Sprint History Summary

| Sprint | Status | Primary Deliverable | Test File | Test Count |
|---|---|---|---|---|
| Sprint 1 | Complete | Repository scaffolding, base classes, configuration system | — | — |
| Sprint 2 | Complete | OCPP 1.6 adapter, ACN-Data loader, ChargingNode basic FL | — | — |
| Sprint 3 | Complete | FedAvg aggregator, GradientManager, NVFLARE integration | — | — |
| Sprint 4 | Complete | FedProx, DP Gaussian mechanism, multi-cluster topology | `test_sprint4.py` | 52 |
| Sprint 5 | Complete | FedMIA attack plugin, PrivacyAuditor, IDS evaluation | `test_sprint5.py` | 25 |
| Sprint 6 | In progress | ElaadNL dataset, OCPP 2.0.1 adapter, experiment sweep | `test_sprint6.py` | TBD |

### 14.4 Sprint 6 Deliverables (In Progress)

Sprint 6 targets the following deliverables, which must be complete before the DSN 2027 paper submission deadline:

1. **ElaadNL dataset adapter** (`src/adapters/elaadnl_dataset.py`): Full implementation with session extraction, z-score normalisation, and train/val/test split. The adapter must pass a round-trip consistency test demonstrating that loading the same file twice yields identical DataLoader outputs when the random seed is fixed.

2. **OCPP 2.0.1 adapter** (`src/adapters/ocpp201_adapter.py`): Implementation covering the Boot, Authorize, Transaction, and MeterValues message types. The adapter must be protocol-compatible with the Corporate cluster (Cluster D) configuration.

3. **Experiment sweep** (`make experiment-sweep`): The sweep target must correctly enumerate all parameter combinations from the `sweep:` section of `config/experiment.yaml`, run each combination sequentially, and write results to separate subdirectories named by configuration hash. A dry-run mode (`make experiment-dry SWEEP=1`) must print the full parameter grid without executing any training.

4. **Comparative MIA results**: Audit reports for both FedAvg and FedProx configurations, on both ACN-Data and ElaadNL datasets, at epsilon values {0.1, 0.5, 1.0, 5.0, 10.0}. This produces a 2 x 2 x 5 = 20 configuration matrix, which forms the core experimental results table for the DSN submission.

---

## 15. References

The following references motivate the design decisions, attack strategies, datasets, and infrastructure choices made in ChargeShield-FL.

**Federated Learning**

[1] McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*, PMLR 54, pp. 1273-1282.

[2] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. In *Proceedings of Machine Learning and Systems (MLSys 2020)*, Vol. 2, pp. 429-450.

**Membership Inference Attacks**

[3] Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. In *Proceedings of the IEEE Symposium on Security and Privacy (S&P 2017)*, pp. 3-18.

[4] Nasr, M., Shokri, R., & Houmansadr, A. (2019). Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In *Proceedings of the IEEE Symposium on Security and Privacy (S&P 2019)*, pp. 739-753.

[5] Hu, R., Guo, Y., Li, H., Pei, Q., & Gong, Y. (2022). Federated learning with GAN-based data synthesis for non-IID clients. *IEEE Transactions on Big Data*, 8(3), 601-610.

**Differential Privacy**

[6] Dwork, C., Roth, A., et al. (2014). The algorithmic foundations of differential privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4), 211-407.

[7] Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. In *Proceedings of the 23rd ACM Conference on Computer and Communications Security (CCS 2016)*, pp. 308-318.

[8] Mironov, I. (2017). Renyi differential privacy. In *Proceedings of the 30th IEEE Computer Security Foundations Symposium (CSF 2017)*, pp. 263-275.

**EV Charging Datasets**

[9] Lee, Z. J., Li, T., & Low, S. H. (2019). ACN-Data: Analysis and applications of an open EV charging dataset. In *Proceedings of the Tenth ACM International Conference on Future Energy Systems (ACM e-Energy 2019)*, pp. 139-149. Dataset available at: https://ev.caltech.edu/dataset

[10] ElaadNL. (2022). ElaadNL Open Data. EV Charging Sessions Dataset. https://www.elaad.nl/research/tools-en/data-science/open-data/

**EV Charging Protocols**

[11] Open Charge Alliance. (2019). OCPP 1.6J Specification. https://www.openchargealliance.org/protocols/ocpp-16/

[12] Open Charge Alliance. (2020). OCPP 2.0.1 Specification. https://www.openchargealliance.org/protocols/ocpp-201/

[13] International Electrotechnical Commission. (2022). IEC 15118-20: Road vehicles — Vehicle to grid communication interface — Part 20: 2nd generation network layer and application layer requirements. IEC Standard 15118-20:2022.

**Infrastructure**

[14] NVIDIA FLARE Team. (2023). NVIDIA FLARE 2.7 Documentation. https://nvflare.readthedocs.io/en/2.7/

[15] Containerlab Contributors. (2024). Containerlab: Container-based networking labs. https://containerlab.dev/

[16] WireGuard Project. (2024). WireGuard: Fast, Modern, Secure VPN Tunnel. https://www.wireguard.com/

**Anomaly Detection and IDS**

[17] Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.

[18] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

[19] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *Proceedings of the IEEE International Conference on Data Mining (ICDM 2008)*, pp. 413-422.

**Related Work on Privacy in Federated Learning and Smart Grid**

[20] Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., et al. (2021). Advances and open problems in federated learning. *Foundations and Trends in Machine Learning*, 14(1-2), 1-210.

[21] Vahidian, S., Morafah, M., Wang, C., Kungurtsev, V., Chen, C., Shah, M., & Lin, B. (2023). Efficient distribution similarity identification in clustered federated learning via principal angles between client data subspaces. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2023)*, 37(8), 10043-10052.

[22] Mothukuri, V., Parizi, R. M., Pouriyeh, S., Huang, Y., Dehghantanha, A., & Srivastava, G. (2021). A survey on security and privacy of federated learning. *Future Generation Computer Systems*, 115, 619-640.

---

*End of ChargeShield-FL Developer Guide. Maintained by the ChargeShield-FL research team. For questions, open an issue on the project repository. Document version: Sprint 6 draft, June 2026.*
