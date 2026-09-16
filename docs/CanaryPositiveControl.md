# Canary Positive Control — protocollo, confondente, baseline

Stato: 2026-09-15, Sprint 10zz+109.
Riferimenti codice: `inject_canaries()` e `_sample_preserving_canary_groups()`
in `scripts/run_experiments.py`; `scripts/check_canary_init_confound.py`.
Riferimenti paper: §6.2 (positive control), §6.3 (simmetria dello scoring),
§9 (limitazioni).

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

## 4. Il controllo di scambio

Eseguire lo stesso seed due volte, con `swap_assignment` a `false` e a `true`,
è il controllo decisivo sul confondente residuo.

- AUC sopra il proprio baseline in **entrambi** i versi → appartenenza.
- AUC sopra in un verso e sotto nell'altro → resta difficoltà intrinseca non
  bilanciata: alzare `n_templates`.

Config di riferimento: `config/experiment_canary_positive_control.yaml` e
`config/experiment_canary_swap_control.yaml`, che differiscono solo per
`swap_assignment` (e per `nonmember_source` reso esplicito).

---

## 5. Baseline a inizializzazione casuale

`scripts/check_canary_init_confound.py` valuta gli stessi canary contro un
modello **mai addestrato**, replicando l'intera catena del runner
(`load_sessions` → `enrich_sessions` → split → `inject_canaries` →
`compute_feature_stats` → `normalize_sessions`) e mediando su 20
inizializzazioni.

**Misure post-fix, office1, seed 42, `n_templates=5`:**

| config | loss membri | loss non-membri | rapporto | AUC a init |
|---|---|---|---|---|
| base | 0.164551 | 0.157967 | 0.96× | **0.5285** (std 0.0124) |
| swap | 0.152828 | 0.160897 | 1.05× | **0.4810** (std 0.0164) |

Il rapporto fra le loss è passato da ~7× a ~1×: il confondente sistematico è
chiuso.

**I due baseline non sono 0.5 e vanno usati come riferimento.** L'offset
residuo (±0.03, quasi simmetrico attorno a 0.5) è combinatorio: con
`n_templates=5` il lato membro ha 155 record ma **5 valori distinti**, quindi
l'AUC è quantizzata a passi grossi e conserva uno sbilanciamento a seconda di
quali template il sorteggio ha assegnato. La quasi-simmetria fra i due valori
è la firma attesa di un residuo combinatorio e non di un confondente: un
confondente sistematico spingerebbe entrambi nella stessa direzione.

**Criterio di lettura dei run.** Non l'AUC assoluta, ma la differenza
`AUC_post_training − AUC_a_init`, calcolata separatamente per base e swap. Δ
positivi e di entità simile nei due versi indicano appartenenza.

**Riduzione della quantizzazione.** Portare `n_templates` a 10 raddoppia i
punti distinti sul lato membro lasciando l'amplificazione per record
sostanzialmente invariata (30/1644 ≈ 1.8% contro 30/1494 ≈ 2.0%). Consigliato
se i Δ misurati risultano dello stesso ordine della `std` dei baseline
(0.012–0.016).

---

## 6. Densità aggregata contro amplificazione per record

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

## 7. Problema aperto: scala target/shadow sotto DP

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

## 8. Cosa è invalidato

I numeri canary raccolti prima dello Sprint 10zz+109 vanno rilanciati:

- Sprint 10zz+18, LiRA composed 0.6875
- Sprint 10zz+95, Blocker 2 a tre attacchi
- `_d1_canary_realdp`, per il motivo separato del §7
- le repliche caltech e jpl, per il motivo del §6

---

## 9. Procedura di riferimento

```bash
# baseline a init, per entrambe le config
python3 scripts/check_canary_init_confound.py \
    --config config/experiment_canary_positive_control.yaml
python3 scripts/check_canary_init_confound.py \
    --config config/experiment_canary_swap_control.yaml

# run base e run di scambio, senza DP
python3 scripts/run_experiments.py \
    --config config/experiment_canary_positive_control.yaml \
    --no-dp --sweep-dir experiments/_canary_pairedsplit_nodp
python3 scripts/run_experiments.py \
    --config config/experiment_canary_swap_control.yaml \
    --no-dp --sweep-dir experiments/_canary_pairedsplit_nodp_swap
```

Verificare nei log la riga `[CANARY] pool unificato:` e, nel secondo run,
`[SWAP ATTIVO]`. Se compare il warning sulla modalità legacy, la config sta
passando `nonmember_source: holdout`.
