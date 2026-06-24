# Developer Guide — ChargeShield-FL

## Regola fondamentale

Ogni Sprint produce SEMPRE:
1. Documentazione aggiornata in `docs/`
2. UML Mermaid in `docs/Architecture.md`
3. Implementazione Python commentata in `src/`
4. Unit test in `tests/`
5. Git commit + tag (`vX.Y.0-sprintN`)

Solo dopo si passa allo Sprint successivo.

## Principio architetturale
Ecco il file completo — copia tutto e incolla in docs/DeveloperGuide.md:
markdown# Developer Guide — ChargeShield-FL

## Regola fondamentale

Ogni Sprint produce SEMPRE:
1. Documentazione aggiornata in `docs/`
2. UML Mermaid in `docs/Architecture.md`
3. Implementazione Python commentata in `src/`
4. Unit test in `tests/`
5. Git commit + tag (`vX.Y.0-sprintN`)

Solo dopo si passa allo Sprint successivo.

## Principio architetturale

Core non conosce protocolli.
Node non conosce dataset.
Dataset non conosce FL.
FL non conosce Privacy Auditor.

Tutto è collegato tramite interfacce (Adapter Pattern).

## Struttura repository
src/

├── core/          # Interfacce astratte (contratti)

├── nodes/         # Implementazioni ChargingNode

├── adapters/      # Protocol + Dataset adapters

├── auditor/       # Membership Inference Attacker

├── ids/           # Intrusion Detection System (difesa)

├── flare/         # NVIDIA FLARE connector

├── plugins/       # Attacchi (FedMIA), dataset aggiuntivi

└── utils/         # Utility condivise
config/            # YAML — zero valori hardcoded nel codice

docs/              # Documentazione e UML

tests/             # Unit test (pytest)

datasets/          # Dataset reali (ACN-Data JPL)

containerlab/      # Topologia infrastruttura

docker/            # Dockerfile per ogni componente

certs/             # Certificati mTLS (non committati)

## Aggiungere un nuovo nodo

1. Estendi `AbstractChargingNode` in `src/nodes/`
2. Aggiungi la configurazione in `config/nodes.yaml`
3. Scegli il `ProtocolAdapter` appropriato
4. Scrivi i test in `tests/`

## Aggiungere un nuovo dataset

1. Estendi `AbstractDataset` in `src/adapters/`
2. Aggiungi la configurazione in `config/datasets.yaml`
3. Scrivi i test con dati reali o sintetici

## Aggiungere un nuovo attacco

1. Crea il plugin in `src/plugins/attacks/`
2. Documenta in `docs/ThreatModel.md`
3. Aggiungi il nome in `config/auditor.yaml` → `attacks:`

## Makefile — Comandi disponibili

| Comando | Descrizione |
|---------|-------------|
| `make help` | Mostra tutti i comandi disponibili |
| `make test` | Esegui unit test con pytest |
| `make test-coverage` | Test con report coverage |
| `make lint` | Controllo qualità codice con ruff |
| `make certs` | Genera certificati mTLS per tutti i nodi |
| `make build` | Build tutte le immagini Docker |
| `make deploy` | Deploy topologia Containerlab |
| `make destroy` | Teardown topologia Containerlab |
| `make inspect` | Stato topologia attiva |
| `make experiment` | Lancia un round FL completo |
| `make clean` | Rimuovi artefatti temporanei |
| `make clean-all` | Pulizia completa |

## mTLS — Gestione certificati

I certificati vengono generati da `scripts/generate_certs.sh`.

Struttura:
certs/

├── ca/           → Certificate Authority (non committare ca.key)

├── highway-01/   → node.crt, node.key, ca.crt

├── urban-01/     → node.crt, node.key, ca.crt

└── ...

Regole:
- `certs/` è in `.gitignore` — non viene mai committato
- Le chiavi private (`*.key`) hanno permessi `600`
- I certificati hanno validità 365 giorni
- Per rigenerare: `make clean-certs && make certs`

## Docker — Componenti

| Immagine | Dockerfile | Ruolo |
|----------|------------|-------|
| `chargeshield/charging-node` | `docker/charging-node/` | Nodo FL client |
| `chargeshield/aggregator` | `docker/aggregator/` | Server FLARE |
| `chargeshield/auditor` | `docker/auditor/` | Privacy Auditor |
| `chargeshield/ids` | `docker/ids/` | IDS difesa |
| `chargeshield/fl-admin` | `docker/fl-admin/` | Admin FLARE |

## Convenzioni

- Ogni file `.py` ha docstring di modulo
- Ogni classe e metodo ha docstring
- Nessun valore hardcoded — tutto in `config/`
- `None` per campi assenti, mai default silenzioso
- Commit format: `feat:`, `fix:`, `config:`, `docs:`, `test:`

## Sprint roadmap

| Sprint | Tag | Contenuto |
|--------|-----|-----------|
| 1 | v0.1.0-sprint1 | Repository, interfacce, YAML, docs |
| 2 | v0.2.0-sprint2 | ChargingNode, OCPP16, ACNDataset |
| 3 | v0.3.0-sprint3 | PrivacyAuditor, AbstractIDS, FLAREConnector, Containerlab, mTLS, Docker, Makefile |
| 4 | v0.4.0-sprint4 | FedMIA, ChargingIDS, Autoencoder, Esperimenti |
