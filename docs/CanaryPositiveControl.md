# Canary Positive Control — protocollo, confondente, baseline

Stato: 2026-09-16, Sprint 10zz+118. **Controllo superato** — vedi §6.
Riferimenti codice: `inject_canaries()` e `_sample_preserving_canary_groups()`
in `scripts/run_experiments.py`; `scripts/check_canary_init_confound.py`.
Riferimenti paper: `sections/validation.tex` (Instrument Validation), §9
(limitazioni).

---

## 1. A cosa serve

Il sanity-check a cinque assi prova soltanto che la memorizzazione *naturale*
è difficile da indurre su questi dati. Non prova che l'harness sarebbe capace
di rilevare una violazione di privacy se ci fosse. Il canary positive control
risponde a quella seconda domanda: si forza la memorizzazione di record noti e
si verifica che gli attacchi la vedano.

Senza questo controllo, «non troviamo leakage» e «il nostro attacco non
funziona» non sono distinguibili, e nessun risultato nullo della campagna è
interpretabile.

---

## 2. Il confondente membership / difficoltà di ricostruzione

**Diagnosi (2026-09-15, review esterna su
`experiments/_d1_canary_realdp/experiment_20260915_155515.json`).**

Fino allo Sprint 10zz+108 i due lati del controllo venivano estratti da pool
diversi: i template membro da `site_train_sessions`, i gemelli non-membro da
`site_holdout_sessions`. Nessuna procedura garantiva che i due gruppi fossero
equiparabili per difficoltà intrinseca di ricostruzione, e nei dati reali non
lo erano.

L'evidenza è nel campo `shadow_canary_debug_group_raw` di quel run. Sotto
modelli shadow **puliti**, mai rumorizzati:

| gruppo | loss media sotto shadow |
|---|---|
| canary_m0 … canary_m4 (membri) | 0.000154 / 0.000079 / 0.000111 / 0.000286 / 0.000083 |
| non-membri (20 sessioni) | 0.000966 |

I cinque template scelti come canary si ricostruivano circa **sette volte
meglio** delle sessioni di confronto, indipendentemente dall'appartenenza.
Qualunque attacco che soglia sulla loss separava i due gruppi a prescindere
dalla memorizzazione, quindi `canary_auc_roc` misurava membership **confusa
con difficoltà della sessione**.

### Una soluzione scartata

Usare come gemelli **copie esatte** dei template membro, tenute fuori dal
training, non funziona. A parità di vettore di feature il modello produce la
stessa ricostruzione e quindi la stessa loss su entrambi i lati: l'AUC vale
0.5 per costruzione qualunque sia la memorizzazione, e il controllo diventa
incapace di rilevare alcunché.

### La correzione adottata

Pool unico e assegnazione casuale, che è il disegno standard nella
letteratura canary. Si estraggono `n_templates + n_nonmember_templates`
sessioni dal training set del sito con un'unica `rng.sample()`, si mescolano,
e si assegna la prima parte al gruppo membro (duplicato, resta in training) e
la seconda al gruppo non-membro, **rimosso da training e spostato in
holdout** perché sia genuinamente non-membro.

I due gruppi diventano campioni scambiabili della stessa distribuzione
prodotti dalla stessa procedura, quindi la difficoltà intrinseca è bilanciata
in aspettazione e si media via su più seed.

---

## 3. Parametri di configurazione

Blocco `canary:` nel YAML.

| chiave | default | significato |
|---|---|---|
| `enabled` | `false` | no-op se assente: zero impatto sui run storici |
| `site` | `office1` | sito in cui iniettare |
| `n_templates` | `5` | template assegnati al gruppo membro |
| `n_duplicates` | `30` | copie per template membro |
| `n_nonmember_templates` | `= n_templates` | template assegnati al gruppo non-membro |
| `nonmember_source` | `paired_split` | `paired_split` = pool unico (corretto); `holdout` = comportamento pre-fix, solo per riprodurre run storici, emette un warning |
| `swap_assignment` | `false` | inverte i due gruppi — vedi §4 |

Ogni gruppo membro produce `n_duplicates + 1` record taggati (i cloni più
l'occorrenza originale ri-taggata, Sprint 10zz+96). Ogni gruppo non-membro ne
produce 2 (l'originale ri-taggato più un clone, Sprint 10zz+108). Con i
default: 155 record membro e 40 non-membro.

---

## 4. Il controllo di scambio a gruppi bilanciati

### 4.1 Perché la versione sbilanciata non era un controllo

Fino allo Sprint 10zz+111 il controllo girava con `n_templates=5` e
`n_nonmember_templates=20`. Il codice (`run_experiments.py`, ramo
`paired_split`) fa:

```
drawn = rng.sample(site_train_sessions, n_templates + n_nonmember_templates)
rng.shuffle(drawn)
swap=False -> member = drawn[:n_t],   nonmember = drawn[n_t:]
swap=True  -> nonmember = drawn[:n_nm], member = drawn[n_nm:]
```

Con 5 e 20, i membri del braccio base sono `drawn[:5]` e i membri del braccio
swap sono `drawn[20:25]`: **insiemi disgiunti**. I due bracci erano due
esperimenti su record diversi, non lo stesso esperimento a ruoli invertiti. Un
esito discordante non dimostrava nulla.

### 4.2 La condizione che rende lo scambio un vero scambio

Con `n_templates == n_nonmember_templates == k` la stessa formula diventa uno
scambio esatto, senza modifiche al codice:

```
drawn ha 2k elementi (stesso seed -> stesso drawn nei due bracci)
braccio A: member = drawn[:k],   nonmember = drawn[k:2k]
braccio B: member = drawn[k:2k], nonmember = drawn[:k]
```

Config di riferimento: `config/experiment_canary_balanced.yaml` (braccio A) e
`config/experiment_canary_balanced_swap.yaml` (braccio B), identici tranne
`swap_assignment`. `k = 20` (non 5) perché il denominatore reale dell'AUC sono
le **coppie distinte**, non i record: i 30 duplicati di un template hanno
feature identiche, quindi loss e score identici (difetto A, §2). I due bracci
vanno lanciati con lo **stesso seed**: è ciò che garantisce che `drawn` sia
identico.

### 4.3 Criterio di lettura

- Δ sopra il proprio baseline in **entrambi** i versi → appartenenza.
- Δ sopra in un verso e sotto nell'altro → difficoltà intrinseca dei template
  sorteggiati.

Il confronto deve essere omogeneo: il baseline di
`check_canary_init_confound.py` è un'AUC su loss a pesi casuali, quindi va
confrontato con `canary_raw_mse_auc_roc` e **mai** con `canary_auc_roc`
(LiRA) — sarebbe il mismatch di metrica dello Sprint 10zz+110.

---

## 5. Baseline a inizializzazione casuale

`scripts/check_canary_init_confound.py` valuta gli stessi canary contro un
modello **mai addestrato**, replicando l'intera catena del runner
(`load_sessions` → `enrich_sessions` → split → `inject_canaries` →
`compute_feature_stats` → `normalize_sessions`) e mediando su 20
inizializzazioni.

**Misure a gruppi bilanciati, office1, `k=20`** (`logs/baselines_balanced.log`,
20 inizializzazioni per cella):

| seed | AUC init braccio A | AUC init braccio B | somma |
|---|---|---|---|
| 42 | 0.5274 (std 0.0085) | 0.4726 (std 0.0085) | 1.0000 |
| 123 | 0.3451 (std 0.0128) | 0.6549 (std 0.0128) | 1.0000 |
| 456 | 0.5829 (std 0.0078) | 0.4171 (std 0.0078) | 1.0000 |
| 789 | 0.3755 (std 0.0121) | 0.6245 (std 0.0121) | 1.0000 |
| 1234 | 0.3874 (std 0.0159) | 0.6126 (std 0.0159) | 1.0000 |

**I baseline non sono 0.5 e vanno usati come riferimento.** Con `k=20` il
sorteggio produce offset anche grandi (0.345–0.583): non è un confondente
sistematico, è quali template sono capitati da che parte.

### 5.1 Cosa dimostra la somma esatta a 1.0000 — e cosa non dimostra

⚠ **Non è un risultato sperimentale: è un'identità algebrica.**
$\mathrm{AUC}_B = 1 - \mathrm{AUC}_A$ vale **sempre**, per qualunque funzione
di score, ogni volta che si scambiano le etichette sugli stessi due gruppi
lasciando i punteggi invariati. Le `std` identiche dentro ogni coppia dicono
la stessa cosa: ogni singola inizializzazione produce $A$ e $1-A$.

Il suo valore è quindi **di controllo di correttezza, non di evidenza**:
conferma che i due bracci operano sugli stessi due gruppi a ruoli invertiti —
esattamente ciò che la configurazione sbilanciata (§4.1) non garantiva.
Presentarla come prova che il disegno «rileva la difficoltà intrinseca» è una
tautologia e va evitato nel paper.

**Corollario.** Poiché i baseline sommano esattamente a 1, vale
$\Delta_A + \Delta_B = \mathrm{somma}_{post} - 1$: il test sui Δ e il test
«somma post > 1» sono **lo stesso test**. I baseline non aggiungono potenza
statistica; aggiungono la scomposizione per braccio e il controllo di
correttezza.

### 5.2 Come NON misurare la significatività

La `std` dei baseline (0.008–0.016) misura la dispersione su 20
re-inizializzazioni **con il campione di template fissato**. Non è
l'incertezza che conta. L'incertezza dominante è il campionamento di *quali*
20+20 template sono stati sorteggiati, e con questi numeri è grande:
Hanley–McNeil dà **SE ≈ 0.077–0.087** per una singola AUC con 20 vs 20 item
distinti. Confrontare un Δ con la std dei baseline sovrastima la
significatività di circa un fattore 8.

Il test corretto è appaiato sui seed: vedi §6.3.

---

## 6. Risultato: campagna bilanciata a 5 seed (2026-09-16)

`k=20`, `n_duplicates=30`, office1, no-DP, 3 round, `shadow_init: cold`,
modello ad alta capacità (7 feature incl. `start_time_epoch`,
hidden `(32,16)`, latent 8, 1000 epoche). Dati letti da
`experiments/_canary_balanced{,_swap}_s{42,123,456,789,1234}/`.

### 6.1 Loss grezza (`canary_raw_mse_auc_roc`), medie sui 3 round

| seed | braccio A | braccio B | somma | baseline A | baseline B | Δ_A | Δ_B |
|---|---|---|---|---|---|---|---|
| 42 | 0.6926 | 0.6935 | 1.3861 | 0.5274 | 0.4726 | +0.165 | +0.221 |
| 123 | 0.8009 | 0.8333 | 1.6342 | 0.3451 | 0.6549 | +0.456 | +0.178 |
| 456 | 0.6947 | 0.7491 | 1.4439 | 0.5829 | 0.4171 | +0.112 | +0.332 |
| 789 | 0.7900 | 0.6542 | 1.4442 | 0.3755 | 0.6245 | +0.414 | +0.030 |
| 1234 | 0.8070 | 0.6518 | 1.4588 | 0.3874 | 0.6126 | +0.420 | +0.039 |
| **media** | **0.7570** | **0.7164** | **1.4734** | | | **+0.314** | **+0.160** |

**30 round su 30 sopra 0.5, minimo 0.6132. Dieci Δ su dieci positivi.**
Tutte e cinque le somme sono sopra 1: il segnale è simmetrico allo scambio dei
gruppi, quindi è appartenenza. Se fosse difficoltà intrinseca sarebbe
antisimmetrico e le somme starebbero a 1.

Effetto soffitto visibile e atteso: dove il baseline parte basso (s123, s789,
s1234) il Δ del braccio A è grande e quello di B piccolo. Mediando le due
direzioni dentro il seed il problema si annulla — è la ragione per cui il
disegno appaiato è quello giusto.

### 6.2 LiRA (`canary_auc_roc`), medie sui 3 round

| | braccio A | braccio B |
|---|---|---|
| media | **0.6506** | **0.6961** |
| minimo per run | 0.5950 | 0.5981 |

⚠ A livello di **singolo round** il LiRA scende sotto 0.5 una volta (braccio B,
seed 42, round 3: 0.4556). Quindi «30 round su 30 sopra 0.5» vale per la loss
grezza, **non** per il LiRA, dove sono 29 su 30. Le due metriche vanno
riportate distinte.

### 6.3 Statistica corretta

Δ medio per seed (media dei due bracci, che annulla il soffitto):
0.1931, 0.3171, 0.2219, 0.2221, 0.2294.

| | |
|---|---|
| media | **+0.2367** |
| sd | 0.0471 |
| errore standard | 0.0210 |
| **t (df=4)** | **11.25** → p < 0.001 |
| test dei segni | 5/5 → p = 0.031 |

Riportare questo, non il confronto con la std dei baseline (§5.2).

### 6.4 Due campagne su disco, e la non-riproducibilità del LiRA

Ogni cartella contiene **due** JSON (s42 ne ha tre): una tornata 12:19–14:45 e
una 15:00–16:56, stesso config e stesso seed. **La tornata di riferimento è la
seconda** (timestamp più recente); i numeri qui sopra vengono da lì.

Il confronto fra le due è però un risultato a sé:

- `canary_raw_mse_auc_roc` è **identica bit a bit** in tutte e dieci le celle;
- `canary_auc_roc` (LiRA) **cambia** su s42: 0.6157→0.6491 nel braccio A,
  0.6324→0.5981 nel braccio B.

Il modello target è deterministico sotto seed; la calibrazione shadow di LiRA
non lo è. È una prova indipendente della degenerazione documentata in §8 e va
citata nel paper.

### 6.5 Coppie distinte effettive

`canary_n_member_distinct × canary_n_nonmember_distinct` non è sempre 20×20:

| seed | coppie distinte |
|---|---|
| 42 | 20×18 = 360 |
| 123 | 20×19 = 380 |
| 456 | 19×20 = 380 |
| 789 | 20×20 = 400 |
| 1234 | 19×20 = 380 |

I baseline sono invece calcolati su 400 coppie piene. Differenza piccola, ma
la colonna va riportata per seed nella tabella del paper: è la dimensione
campionaria effettiva, cioè proprio il difetto A che questo disegno corregge.

---

## 7. Densità aggregata contro amplificazione per record

La diagnosi storica di scale-dependence fra office1 e caltech/jpl
(TestRoadmap riga 2, «duplicati ~11% a office1 vs ~0.5% a caltech/jpl»)
confondeva due quantità distinte.

| run | n_templates | n_duplicates | densità aggregata | amplificazione per record |
|---|---|---|---|---|
| office1 | 5 | 30 | ~10% di 1494 | 30/1494 = **2.0%** ciascuno |
| caltech high-density | 84 | 30 | ~10% di 25123 | 30/25123 = **0.12%** ciascuno |

La densità *aggregata* era equiparata, l'amplificazione del *singolo record*
differiva di circa 17×. La memorizzazione di un record dipende dalla seconda.
La replica a densità aggregata equiparata non testava quindi la variabile che
intendeva testare, e la conclusione «la densità non spiega il divario» non
segue da quell'esperimento.

Config corretta:
`config/experiment_canary_positive_control_caltech_amplification_matched.yaml`
(`n_templates=5`, `n_duplicates=561`).

---

## 8. Problema aperto: scala target/shadow sotto DP

Nel run con DP reale (`_d1_canary_realdp`, ε=1.0, dp-fedavg) il target è
rumorizzato e le shadow no. Le due popolazioni vivono su scale diverse di due
ordini e mezzo di grandezza: `target_mean` fra 0.306 e 0.510, `shadow_mean`
fra 0.0001 e 0.001.

LiRA valuta la loss del target sotto gaussiane stimate sulle shadow. Con
t ≈ 0.4, μ ≈ 0.0005 e σ ≈ 0.001, ogni campione sta a circa 400 deviazioni
standard da entrambe le medie, e il segno del log-likelihood-ratio diventa un
artefatto di quale delle due medie quasi identiche capiti a essere
marginalmente più vicina. Il filtro a 8σ non è attivo sul percorso di scoring
dei canary, altrimenti li escluderebbe tutti.

La firma diagnostica è l'andamento per round:

| round | loss modello | canary raw MSE AUC | canary LiRA AUC |
|---|---|---|---|
| 1 | 0.0015 | 0.689 | 0.711 |
| 2 | 0.448 | 0.489 | 0.856 |
| 3 | 0.379 | 0.522 | 0.622 |

Il rumore DP entra dal round 2. Da lì il segnale della loss grezza sparisce,
come atteso da un modello distrutto, mentre quello di LiRA cresce. Un attacco
il cui segnale si rafforza mentre il modello viene annientato sta misurando
l'annientamento, non l'appartenenza.

Ne segue che `canary_composed_auc_roc = 0.7889` e la media per-round 0.5685 di
quel run **non sono citabili**. L'unico punto potenzialmente valido è il round
1, dove il modello è ancora integro e raw e LiRA concordano a 0.69/0.71.

Opzioni non ancora valutate: rumorizzare anche le shadow, oppure escludere dal
calcolo canary i round in cui il modello è degradato oltre una soglia.

---

## 9. Cosa è invalidato

I numeri canary raccolti prima dello Sprint 10zz+109 vanno rilanciati:

- Sprint 10zz+18, LiRA composed 0.6875
- Sprint 10zz+95, Blocker 2 a tre attacchi
- `_d1_canary_realdp`, per il motivo separato del §8
- le repliche caltech e jpl, per il motivo del §7

Va inoltre considerato **superato** ogni run a gruppi sbilanciati
(`n_templates != n_nonmember_templates`), incluse le coppie
`_canary_maxmemo{,_swap}` e `_canary_pairedsplit_nodp{,_swap}`: il loro
braccio di scambio non è un controllo negativo (§4.1). I dati restano sul
disco perché il braccio base resta un positive control valido, ma la coppia
non va citata come evidenza di appartenenza. L'evidenza è la campagna
bilanciata del §6.

---

## 10. Procedura di riferimento

La campagna completa è cinque seed × due bracci, più i dieci baseline. I due
bracci di uno stesso seed **devono** usare lo stesso `--seed`.

```bash
cd ~/Documents/ChargeShield-FL

for s in 42 123 456 789 1234; do
  # baseline a inizializzazione casuale (20 init), entrambi i bracci
  python3 scripts/check_canary_init_confound.py \
      --config config/experiment_canary_balanced.yaml      --seed $s
  python3 scripts/check_canary_init_confound.py \
      --config config/experiment_canary_balanced_swap.yaml --seed $s

  # braccio A e braccio B, senza DP
  caffeinate -ims python3 scripts/run_experiments.py \
      --config config/experiment_canary_balanced.yaml --no-dp --seed $s \
      --sweep-dir experiments/_canary_balanced_s$s
  caffeinate -ims python3 scripts/run_experiments.py \
      --config config/experiment_canary_balanced_swap.yaml --no-dp --seed $s \
      --sweep-dir experiments/_canary_balanced_swap_s$s
done 2>&1 | tee logs/canary_balanced_campaign.log
```

Il `tee` sull'intero loop è deliberato: nella campagna del 2026-09-16
l'output di `check_canary_init_confound.py` finiva a schermo e i baseline
sono dovuti essere rifatti.

Verifiche nei log:

- `[CANARY] pool unificato:` presente in entrambi i bracci;
- `[SWAP ATTIVO]` solo nel braccio B;
- `DISTINTI 20x20 = 400 coppie reali` (o 360/380, vedi §6.5) — se compare un
  numero molto più grande, la config sta contando i duplicati e il
  denominatore dell'AUC è gonfiato;
- nessun warning `n_templates != n_nonmember_templates`, che segnalerebbe un
  ritorno alla configurazione non bilanciata del §4.1.

Lettura dei risultati:

```bash
python3 - <<'EOF'
import json, glob, statistics as st
for arm,lab in [("","A"),("_swap","B")]:
    for s in (42,123,456,789,1234):
        fs=sorted(glob.glob(f"experiments/_canary_balanced{arm}_s{s}/experiment_*.json"))
        pr=json.load(open(fs[-1]))["per_round"]      # [-1] = tornata piu' recente
        r=[pr[k]["mia"]["canary_raw_mse_auc_roc"] for k in sorted(pr,key=int)]
        l=[pr[k]["mia"]["canary_auc_roc"]          for k in sorted(pr,key=int)]
        print(f"{lab} s{s:<5d} raw={st.mean(r):.4f}  lira={st.mean(l):.4f}")
EOF
```
