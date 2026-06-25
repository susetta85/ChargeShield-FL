# ChargeShield-FL — Case Studies

I case study misurano la vulnerabilità MIA in scenari reali.
In tutti gli scenari:
- **Attaccante** = aggregatore honest-but-curious (Scenario 1)
- **FedMIA opera su pesi post-DP** — vede gradient update già
  privatizzati da `GradientManager` dopo decrypt mTLS/WireGuard
- **Metrica principale** = AUC-ROC (curva ε vs AUC-ROC per il paper)

---

## Case Study 1 — JPL EV Charging Network: ε vs AUC-ROC

### Scenario

Rete di colonnine EV del Jet Propulsion Laboratory (Caltech).
Veicoli aziendali con pattern di ricarica regolari e prevedibili.
L'aggregatore honest-but-curious analizza i gradient update
ricevuti dai 4 cluster FL per inferire membership.

### Dataset

ACN-Data JPL 2019+2020 — 13,073 sessioni reali.
Split: 50% members (in training), 50% non-members (attacco).

### Configurazione FL

- 4 cluster FL (highway, urban, residential, corporate)
- FedAvg con proximal_mu=0.0 (baseline) e FedProx con proximal_mu=0.01
- 100 round  →  sweep {100, 200, 500, 1000}
- GradientManager: ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, δ=1e-5
- FedMIA: shadow model su dati pubblici ACN-Data JPL
- Feature ACN (6): `total_energy_kwh`, `max_power_kw`, `kwh_requested`,
  `minutes_available`, `hour_of_day`, `duration_hours`
- Pipeline: `load_sessions()` → `enrich_sessions()` → trainer

### Domanda di ricerca

> A quale ε FedMIA diventa statisticamente non migliore del random
> (AUC-ROC → 0.5) anche quando opera su pesi già privatizzati da DP?

### Risultati attesi
| ε | Rumore DP | AUC-ROC atteso (100 round) | AUC-ROC atteso (1000 round) |
|---|---|---|---|
| 0.1 | alto | → 0.5 (MIA inefficace) | → 0.5 |
| 0.5 | medio | ~0.55–0.65 | ~0.60–0.70 |
| 1.0 | standard | ~0.65–0.75 | ~0.70–0.80 |
| 2.0 | basso | ~0.70–0.80 | ~0.75–0.85 |
| 5.0 | minimo | ~0.75–0.85 | ~0.80–0.90 |

**Sweep completo:** rounds ∈ {100, 200, 500, 1000} × ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
→ heat map 4×5 generata da `scripts/compare_results.py`
→ eseguibile con `make experiment-full-sweep` (stima: 8-12 ore su CPU)

### Status

🔄 In esecuzione Sprint 5/6 — primo run 100 round completato (ε=1.0)
Sweep completo: `make experiment-full-sweep` (stima: 8–12 ore CPU)
---

## Case Study 2 — Multi-Cluster Heterogeneous: membership score per cluster

### Scenario

Rete eterogenea con 4 cluster a pattern di ricarica molto diversi
(dati non-IID). L'aggregatore honest-but-curious analizza se il
membership score FedMIA varia tra cluster con distribuzioni diverse.

### Dataset

ACN-Data JPL (proxy per tutti i cluster).
Cluster simulati con split geografico/temporale dei dati JPL.

### Configurazione FL

- 4 cluster FL con protocolli eterogenei:
  - highway: OCPP 1.6, 150kW DC
  - urban: OCPP 1.6, 22kW AC
  - residential: MQTT v5, 7kW AC
  - corporate: OCPP 2.0.1, 50kW DC
- FedProx (proximal_mu=0.01) per gestire non-IID
- ε=1.0, δ=1e-5
- ChargingIDS: cosine similarity inter-cluster per anomalie

### Domanda di ricerca

> Il membership score FedMIA varia significativamente tra cluster
> con distribuzioni di dati molto diverse (highway DC 150kW vs
> residential AC 7kW)? FedProx riduce questa varianza rispetto a FedAvg?

### Risultati attesi

- Cluster con dati più omogenei (corporate, highway) →
  membership score più alto e stabile
- Cluster con alta varianza (residential) →
  membership score più basso e rumoroso
- FedProx riduce la varianza inter-cluster rispetto a FedAvg

### Status

🔄 Pianificato Sprint 5

---

## Case Study 3 — DP vs No-DP: impatto su FedMIA

### Scenario

Confronto diretto FedMIA con e senza Differential Privacy.
L'aggregatore honest-but-curious esegue l'attacco in due condizioni:
pesi con DP (GradientManager attivo) vs pesi senza DP (ε=∞).
Misura quanto DP riduce l'efficacia di FedMIA.

### Configurazione

- ε=1.0, δ=1e-5 (DP attiva) vs ε=∞ (DP disattiva, max_grad_norm=∞)
- FedAvg (proximal_mu=0.0) e FedProx (proximal_mu=0.01) a confronto
- ChargingIDS in modalità completa (CUSUM + Krum + Cosine)
- 100 round, 4 cluster

**Nota:** FedMIA opera sempre sui pesi post-decrypt mTLS.
La differenza è solo la presenza/assenza del rumore gaussiano
aggiunto da `GradientManager`.

### Domanda di ricerca

> Quanto DP riduce l'AUC-ROC di FedMIA (Δ AUC-ROC con/senza DP)?
> FedProx introduce variazioni rispetto a FedAvg nella vulnerabilità MIA?
> IDS rileva comportamenti anomali durante l'attacco MIA
> (risposta attesa: no — l'aggregatore è honest-but-curious)?

### Risultati attesi

- DP attiva → AUC-ROC significativamente più basso
- FedProx → membership score più uniforme tra cluster (meno varianza)
- IDS → nessun alert (attaccante honest-but-curious non disturba il protocollo)

### Status

🔄 Pianificato Sprint 5

---

## Scenario 2 — Client curioso (Future Work)

Non implementato in questa versione. Un FL client tenta di inferire
dati di altri client osservando solo il modello globale aggregato.
Attacco più difficile: meno informazioni disponibili rispetto
all'aggregatore curioso.

Da implementare in Sprint 6 con dataset ElaadNL (se disponibile).

---

## Esecuzione

```bash
# CS1 — sweep epsilon
make experiment-sweep

# CS2/CS3 — configurazione specifica
python scripts/run_experiments.py --config config/experiment.yaml --epsilon 1.0 --rounds 100

# Dry run — verifica config e dataset
make experiment-dry
```

I risultati vengono salvati in `experiments/` con timestamp.
