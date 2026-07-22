# ChargeShield-FL — Deep Code Review (Read-Only)
**Date:** 2026-07-09  
**Scope:** Bug analysis · Security · Model training correctness · Scalability · Log analysis  
**Note:** No files were modified during this review.

---

## 1. Bug Report

| Severity | File:Line | Description | Impact |
|---|---|---|---|
| CRASH | `charging_ids.py` | `ChargingIDS.__init__` chiama `_load_ids_config(config_path)` che lancia `FileNotFoundError` se `config/auditor.yaml` è assente. Nessun fallback. | Crash all'avvio dell'IDS se il file manca |
| WRONG\_RESULT | `fedavg_aggregator.py:98` | `valid = [u for u in updates if u.n_samples > 0]` filtra i nodi invalidi **dopo** il check `len(updates) >= min_participants`. Con 3 update di cui 2 invalidi, il check passa (3 ≥ 2) ma si aggrega su 1 solo nodo — la Byzantine-tolerance è silenziosamente annullata. | FedAvg con un solo partecipante; garanzia min_participants violata |
| WRONG\_RESULT | `fedmia.py:108` | Doppia istanziazione del shadow model: riga 107 crea `Autoencoder()` con `input_dim` default, riga 108 lo sovrascrive con `input_dim` corretto. Prima istanza è garbage-collected immediatamente. | Spreco di allocazione; latent bug se input_dim ≠ 6 |
| WRONG\_RESULT | `autoencoder.py:272` | `fit()` chiama `_calibrate_threshold(train_loader)` ma `_calibrate_threshold` itera il DataLoader aspettando tensori raw, non tuple. Un DataLoader standard da `TensorDataset` produce tuple `(tensor,)`. Causerebbe crash in produzione su qualsiasi caller che usi `fit()` con `TensorDataset`. | CRASH in `_calibrate_threshold` con DataLoader standard |
| WRONG\_RESULT | `run_experiments.py` | `members` (10,458 sessioni) è 4× più grande di `non_members` (2,615 sessioni). L'AUC-ROC è valido matematicamente ma la soglia ottimale è sbilanciata. Va documentato nel paper. | Asimmetria pool MIA non documentata |
| WRONG\_RESULT | `privacy_auditor.py:190` | `round_epsilon = sensitivity / self._max_grad_norm` non è un epsilon DP formale. I PRIVACY\_BUDGET alerts dell'IDS sono calibrati su una metrica proxy, non su un bound DP reale. | Tutti gli alert di budget DP non sono interpretabili come garanzie formali |
| SILENT | `run_experiments.py` | `_update_excel_report()` usa `importlib + exec_module` per caricare `generate_excel_report.py` dinamicamente. Errori swallowed da `except Exception: logger.warning(...)`. | Potenziale arbitrary code execution se il file è scrivibile da attacker |
| SILENT | `autoencoder_trainer.py` | Con `drop_last=True` e cluster con < 32 sessioni, il DataLoader produce 0 batch. Viene restituito un `GradientUpdate` con `n_samples=len(sessions)>0` ma weights non aggiornati (stale). Il FedAvg includerà questi weights stale. | Pesi stale contribuiscono al global model senza training |

---

## 2. Security Assessment

**Path traversal (LOW):** `ACNDataset.load()` non valida il path — un YAML modificato potrebbe caricare file arbitrari dal filesystem.

**Arbitrary code execution (MEDIUM):** `_update_excel_report()` carica `generate_excel_report.py` con `exec_module()`. Se il file è scrivibile da un attacker, si ha RCE. Errori swallowed silenziosamente.

**Secrets (NONE):** Nessuna credential hardcoded trovata.

**mTLS:** Gestito da NVFLARE. Nessun meccanismo di rotation visibile nel codice.

---

## 3. Model Training Analysis

### Architettura
6→16→8→4→8→16→6 con ~1,212 parametri. Architettura appropriata per 6 feature normalizzate. MSE come loss è corretto con output Sigmoid e input normalizzati in [0,1].

### BatchNorm in FL — Problema strutturale
`BatchNorm1d` nell'encoder accumula `running_mean` e `running_var` per cluster. FedAvg fa la media di questi buffer tra cluster con distribuzioni diverse. Dopo `apply_global_model()`, i BatchNorm stats locali vengono sovrascritti con la media globale — **sbagliata per quel cluster**. I primi batch di ogni round calcolano gradienti rispetto a statistiche scorrette. 

**Fix raccomandato:** sostituire `BatchNorm1d` con `LayerNorm` (normalizza per-sample, nessun running stats) — un cambiamento per layer nell'encoder.

### FedProx
Implementazione corretta: `loss += (mu/2) * ||w - w_global||²`. Il termine prossimale è attivo dal round 2 in poi (round 1: `_global_weights=None`, skip corretto).

### Data Flow della Normalizzazione
Il trace completo è corretto:
```
main() → compute_feature_stats(train_sessions) → normalize_sessions(train, stats)
                                               → normalize_sessions(holdout, stats)
       → run_fl_rounds(train_sessions)   # dati normalizzati
       → run_fedmia(train, holdout, ...)  # dati normalizzati
```
La normalizzazione raggiunge correttamente sia il training FL che la MIA scoring.

---

## 4. Scalability Analysis

| Dimensione | Attuale | Limite |
|---|---|---|
| Sessioni in RAM | 13k (~13 MB dict) | Praticabile fino a ~100k; 1M richiederebbe streaming |
| FedMIA per 1000 round | ~30 minuti extra di inferenza | Mitigabile eseguendo FedMIA ogni 10 round invece di ogni round |
| Training cluster (sequenziale) | ~4 ore per 1000 round | Parallelizzabile con `ProcessPoolExecutor` → ~1 ora su 4 core |
| GPU | Modello troppo piccolo (1212 param) per GPU | CPU è appropriato per questa dimensione |
| Sigma DP | Calcolato una volta in `__init__` | Corretto, nessun overhead |
| Scale a 20 cluster | Solo modifica lista `cluster_ids` | ~5x più lungo per round con training sequenziale |

---

## 5. Opportunità di Miglioramento (per impatto)

1. **Sostituire `BatchNorm1d` con `LayerNorm`** — elimina l'interferenza FL-BatchNorm, migliora la correttezza della convergenza. Cambiamento minimo, impatto alto.

2. **Parallelizzare il training dei cluster** — da sequenziale a `ProcessPoolExecutor(max_workers=4)`. Riduce il tempo per round di ~4x.

3. **Eseguire FedMIA ogni N round** (es. ogni 10) invece di ogni round — riduce la fase MIA da ~30 min a ~3 min per 1000 round, senza perdita significativa di informazione sulla curva AUC-ROC.

4. **Bilanciare il pool MIA** — usare lo stesso numero di members e non-members (es. 2,615 ciascuno) per una misurazione AUC-ROC pulita.

5. **Aumentare `epochs` a 5–10** per il baseline no-DP — più epoche → più memorizzazione → segnale MIA più forte → AUC-ROC misurabilmente > 0.5 senza DP.

6. **Adottare LiRA** (Carlini et al., 2022) come attacco MIA alternativo — più discriminante di Yeom et al. a bassi false-positive rate. Richiede shadow models multipli ma produce risultati più solidi per DSN.

7. **Fix `aggregate()` per controllare `len(valid) >= min_participants`** invece di `len(updates)`.

---

## 6. Analisi del Log di Training

**File:** `experiments/sweep_log.txt` (11,135 righe, fino a round 198/1000)

**Trovata: CONFERMA CHE GLI ESPERIMENTI ERANO SENZA NORMALIZZAZIONE**

| Round | Loss globale |
|---|---|
| Round 1 | 15,654.505316 |
| Round 10 | 15,708.823919 |
| Round 100 | ~15,500 (oscillante) |
| Round 191 | 15,468.242938 |
| Round 198 | ~15,350 |

**Riduzione totale su ~198 round: ~1.3%.** Il modello **non sta convergendo**.

**Causa:** Il decoder usa `Sigmoid` (output in [0,1]) ma i target erano nelle scale originali (`minutes_available` 0–600+, `total_energy_kwh` 0–80). Il MSE tra output bounded [0,1] e target illimitati è strutturalmente irriducibile — il modello non può mai minimizzare questa loss. La loss oscilla attorno a ~15,500 invece di scendere verso zero.

**Implicazione critica:** Tutti i risultati AUC-ROC prodotti da questi esperimenti (prima del fix della normalizzazione) sono **invalidi per il paper**. Un modello che non converge produce reconstruction errors dominati dal rumore → membership scores distribuiti uniformemente → AUC-ROC ≈ 0.5 per ragioni strutturali, non perché il DP sia efficace.

---

## 7. Verdetto Complessivo

**Gli esperimenti nei log esistenti non sono validi per DSN 2027.** Il log conferma training senza normalizzazione effettiva.

**Cosa funziona correttamente nel codice attuale:**
- ✅ Normalizzazione implementata e correttamente posizionata nel data flow
- ✅ FedAvg, FedProx, GradientManager corretti
- ✅ FedMIA loss-based correttamente cablato
- ✅ Test suite coprente (test_sprint4 + test_sprint5)

**Da fare prima che i nuovi esperimenti (in corso) siano validi:**

| # | Azione | Urgenza |
|---|---|---|
| 1 | Verificare che i nuovi esperimenti producano loss ∈ [0,1] (non 15,000) — controllare il log corrente | IMMEDIATO |
| 2 | Verificare che senza DP l'AUC-ROC > 0.5 (segnale MIA esiste) | Prima della submission |
| 3 | Sostituire `BatchNorm1d` con `LayerNorm` nel encoder | Alta priorità |
| 4 | Bilanciare pool MIA (members = non-members per dimensione) | Prima della submission |
| 5 | Fix `aggregate()`: `len(valid) >= min_participants` | Media priorità |
| 6 | Documentare nel paper: DP = weight perturbation, non DP-SGD | Prima della submission |

---

*Review condotto in modalità read-only. Nessun file modificato.*
