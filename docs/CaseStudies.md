# ChargeShield-FL — Case Studies

## Case Study 1 — JPL EV Charging Network

### Scenario
Rete di colonnine EV del Jet Propulsion Laboratory (Caltech).
Veicoli aziendali con pattern di ricarica regolari e prevedibili.

### Dataset
ACN-Data JPL 2019+2020 — 13,073 sessioni reali.

### Configurazione FL
- 12 nodi simulati sui dati JPL
- FedAvg, 100 round, 5 epoche locali
- PrivacyAuditor attivo su ogni round
- ChargingIDS con Krum (f=1) e CUSUM

### Domande di ricerca
1. Quanto è vulnerabile un nodo JPL a FedMIA?
2. Il CUSUM rileva deriva nel privacy score tra 2019 e 2020?
3. Krum identifica nodi anomali in un cluster omogeneo?

### Status
🔄 Pianificato Sprint 5

---

## Case Study 2 — Multi-Cluster Heterogeneous Network

### Scenario
Rete eterogenea con 4 cluster (Highway, Urban, Residential, Corporate)
con pattern di ricarica molto diversi tra i cluster.

### Dataset
ACN-Data JPL (proxy per tutti i cluster — dataset reali
per cluster specifici in Sprint 6 con ElaadNL).

### Configurazione FL
- 12 nodi, 4 cluster
- Protocolli eterogenei: OCPP 1.6, OCPP 2.0.1, MQTT v5
- FedMIA con shadow model addestrato su dati pubblici
- ChargingIDS con cosine similarity inter-cluster

### Domande di ricerca
1. La cosine similarity rileva nodi Byzantine in cluster eterogenei?
2. Il membership score di FedMIA varia tra cluster con dati diversi?
3. Qual è il trade-off privacy/utilità (epsilon vs AUC-ROC)?

### Status
🔄 Pianificato Sprint 5

---

## Case Study 3 — Adversarial Attack Simulation

### Scenario
Simulazione di un attacco coordinato su un cluster Highway:
un nodo invia gradienti manipolati (model poisoning)
mentre un avversario esterno tenta FedMIA sui gradienti.

### Configurazione
- 1 nodo Byzantine su 3 (highway-03 → poisoned)
- FedMIA attivo dall'esterno
- ChargingIDS in modalità difesa completa

### Domande di ricerca
1. Krum identifica highway-03 entro quanti round?
2. La DP riduce il membership score di FedMIA?
3. L'esclusione del nodo Byzantine impatta l'AUC-ROC del modello?

### Status
🔄 Pianificato Sprint 4/5
