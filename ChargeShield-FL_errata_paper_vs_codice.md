# ChargeShield-FL — Errata: affermazioni del paper contro dati del repository

Documento di lavoro, non destinato alla submission. Mappa ogni affermazione
verificabile della bozza DSN sul codice e sui risultati reali contenuti
nell'archivio `ChargeShield-FL.zip` (1.7 GB, 7.596 file, 537 JSON di risultati).

**Metodo.** Codice letto direttamente; `scripts/check_significance.py` rieseguito
sui dati reali; diagnostiche `lira_debug_*` e metriche FL estratte da tutti i
file della campagna principale con la stessa logica di deduplica e di esclusione
cartelle usata dallo script del progetto. Non è stato possibile installare torch
nel sandbox di revisione, quindi il training non è stato rieseguito: tutto ciò
che segue riguarda il codice e i risultati già prodotti.

**Legenda verdetti.**
`CONFERMATO` — l'affermazione regge alla verifica.
`IMPRECISO` — sostanzialmente vero ma formulato in modo che un revisore
contesterebbe.
`SMENTITO` — i dati del repository dicono il contrario.
`NON VERIFICABILE` — l'informazione necessaria non è nei risultati.

---

## Indice dei problemi per gravità

| # | Affermazione | Verdetto | Sezione |
|---|---|---|---|
| 1 | Tre posizionamenti DP distinti | SMENTITO | §3.4, §5, §7 |
| 2 | Nessuna metrica di utility riportata | SMENTITO (omissione) | tutto |
| 3 | `central` non mostra leakage | SMENTITO | §7 |
| 4 | Table 2 riporta l'AUC composto | SMENTITO | §7 |
| 5 | Positive control sugli stessi canary | SMENTITO | §6.2 |
| 6 | Raw loss rileva il canary in ogni sito | SMENTITO | §6.2, §9 |
| 7 | Nessuna deviazione oltre 0.0005 da 0.5 | SMENTITO | §7 |
| 8 | Esclusione 8σ come adattamento sostanziale | IMPRECISO | §3.5 |
| 9 | Densità dei duplicati appianata | IMPRECISO | §6.2 |
| 10 | Soglia 0.60 "pre-dichiarata" | IMPRECISO | §6.1 |
| 11 | Range dell'asse epoche | IMPRECISO | §6.1 |
| 12 | Assunzione gaussiana "ereditata" | IMPRECISO (omissione) | §3.5 |
| 13 | ε come budget DP client-level | IMPRECISO | §3.4, §8 |
| 14 | Composizione avanzata e crossover | CONFERMATO | §9 |
| 15 | Quattro assi su cinque del sanity check | CONFERMATO | §6.1 |

---

## 1. "Three DP placements" — SMENTITO

**Paper, §3.4:** tre posizionamenti valutati, `dp-fedavg`, `central`, `local`,
dove `local` è descritto come *"per-client clip + noise, with the raw update
never transmitted to or observed by the server, not even transiently"*.

**Codice, `scripts/run_experiments.py:1009`:**

> `"dp-fedavg" (default) e "local" condividono lo stesso meccanismo per-client`
> `(clip+noise prima dell'aggregazione) — la differenza tra i due è SOLO nella`
> `visibilità di raw_updates per l'IDS`

L'unica divergenza è alle righe 1110–1112, dove sotto `local` la lista
`raw_updates` non viene salvata in `fl_results`. LiRA attacca l'update
sottomesso, identico nelle due modalità.

**Verifica sui dati.** Per ogni seed, ogni ε e tutti e tre gli attacchi, i
risultati sono identici bit per bit:

| Configurazione | seed | LiRA | Yeom | Shadow |
|---|---|---|---|---|
| dp-fedavg ε=0.1 | 123 | 0.500213 | 0.495281 | 0.498830 |
| local ε=0.1 | 123 | 0.500213 | 0.495281 | 0.498830 |
| dp-fedavg ε=1.0 | 789 | 0.499341 | 0.502059 | 0.502112 |
| local ε=1.0 | 789 | 0.499341 | 0.502059 | 0.502112 |

Vale per tutte e 15 le coppie (3 ε × 5 seed).

**Conseguenza.** Le 10 configurazioni della Table 2 sono 7. Tre righe su dieci
sono la ripetizione esatta di altre tre. L'affermazione *"This holds uniformly
across DP placement (dp-fedavg / central / local)"* (§7) copre due meccanismi,
non tre.

**Rimedio.** O si eliminano le righe `local` dalla tabella e si riformula il
contributo come "due posizionamenti", oppure si implementa davvero la
distinzione: sotto `dp-fedavg` l'avversario honest-but-curious deve vedere
l'update *pre-rumore* (è ciò che il §3.4 descrive a parole: *"the server
transiently observes the raw update, which is exactly what LiRA is built to
attack"*), sotto `local` quello post-rumore. Oggi in entrambi i casi vede il
post-rumore.

---

## 2. Utility mai riportata — omissione sostanziale

**Paper.** `U_cost` è nominato una volta nel §8 come "companion axis" e mai
quantificato. Nessuna tabella, nessuna cifra, in nessuna sezione.

**Dati.** `per_round[N].fl.mean_loss`, medie su 5 seed:

| Configurazione | round 1 | round 5 | round 10 |
|---|---|---|---|
| no-DP | 0.001160 | 0.002243 | 0.001677 |
| central ε=1.0 | 0.001160 | 0.287586 | 0.358641 |
| central ε=0.5 | 0.001160 | 0.395993 | 0.476242 |
| central ε=0.1 | 0.001160 | 0.405716 | 0.481271 |
| dp-fedavg / local ε=1.0 | 0.001160 | 0.397838 | 0.404192 |
| dp-fedavg / local ε=0.5 | 0.001160 | 0.473488 | 0.422867 |
| dp-fedavg / local ε=0.1 | 0.001160 | 0.478727 | 0.422351 |

L'errore di ricostruzione passa da 0.0012 a 0.36–0.48 in ogni configurazione DP:
degradazione di 200–400×. Su feature normalizzate un MSE di ~0.4 corrisponde
all'ordine di grandezza della varianza del dato, cioè a un modello che non
ricostruisce.

**Origine aritmetica.** `src/ml/gradient_manager.py:_compute_sigma()` calcola
σ = C·√(2·ln(1.25/δ))/ε. Con `max_grad_norm = 1.0` e `delta = 1e-5`
(`config/auditor.yaml`, `config/experiment.yaml`) il fattore √(2·ln(125000)) vale
4.836, quindi σ = 4.84 a ε=1.0, σ = 9.67 a ε=0.5, σ = 48.4 a ε=0.1 — applicato a
ogni peso di un modello da 570 parametri. La docstring del metodo cita già
"es. σ=48 per ε=0.1" a proposito dei buffer BatchNorm.

**Conseguenza.** Nessun punto operativo del sweep ha utility utilizzabile, ε=1.0
compreso. Il null result sotto DP non misura l'efficacia della privacy: misura un
modello distrutto. Il §10 conclude che il risultato è *"a practically relevant,
reportable answer for anyone deciding whether to deploy FL with DP over EV
charging infrastructure"*; con questi numeri la risposta pratica è che nessuna
delle configurazioni testate è deployabile.

**Rimedio.** Tabella utility × configurazione accanto alla Table 2, obbligatoria.
E un sweep su ε molto più ampio, o un clipping molto più aggressivo, per trovare
un punto in cui il modello sopravvive.

---

## 3. Il valore di `central` è la cancellazione di due artefatti — SMENTITO

**Paper, §7:** `central` ε=1.0 → 0.5005, CI [0.4996, 0.5014], letto come assenza
di segnale.

**Diagnostica del progetto.** `lira_debug_matched_formula_auc` esiste proprio per
questo controllo: forza i membri nella stessa formula di fallback usata per i
non-membri e ricalcola l'AUC. Il commento in `run_experiments.py` (righe
4068–4082) lo dichiara test decisivo: *"Se questo AUC resta vicino a 0.5, l'AUC
alto visto sopra è un artefatto di formule diverse, NON segnale reale."*

**Valori reali, media su tutta la campagna:**

| Gruppo | matched_formula_auc | lira_auc riportato |
|---|---|---|
| central ε=0.1 | **0.2087** | 0.5005 |
| central ε=0.5 | **0.2863** | 0.4999 |
| central ε=1.0 | **0.3352** | 0.5004 |
| dp-fedavg / local ε=0.1 | 0.5091 | 0.4999 |
| dp-fedavg / local ε=0.5 | 0.5075 | 0.4995 |
| dp-fedavg / local ε=1.0 | 0.5103 | 0.5000 |
| no-DP | 0.4304 | 0.5003 |

Minimo per-round osservato su `central`: **0.00031**.

**Lettura.** A formula identica sui due lati, la configurazione flagship del
paper è fortemente invertita, non neutra. Il 0.5005 riportato nasce dal fatto che
i membri usano `μ_in` stimato per campione mentre i non-membri usano
`μ_in = μ_out + (μ_in_fb − μ_out_fb)`, un ancoraggio costante per cluster
(ramo `else` alla riga ~3595). Due errori grandi che si annullano non sono
assenza di segnale, e la cancellazione non è garantita: cambia con ε
(0.21 → 0.34 passando da ε=0.1 a ε=1.0).

**Rimedio.** Simmetrizzare lo scoring tra le due classi prima di riportare
qualunque numero su `central`. Finché `matched_formula_auc` resta a 0.21, quella
riga della Table 2 non è interpretabile.

---

## 4. Table 2 non riporta la metrica dichiarata — SMENTITO

**Paper, §7:** *"Table 2 reports, for each configuration, the mean composed LiRA
AUC-ROC across 5 independent seeds"*.

**Codice.** `scripts/check_significance.py`, `main()`: legge
`summary["mean_lira_auc_roc"]`, cioè la **media degli AUC per-round**. La
riga TPR più sotto legge invece `composed_tpr_at_fpr_*`, quindi le due tabelle
del §7 usano metriche diverse mentre il testo le chiama entrambe "composed".

**Verifica.** Ho eseguito lo script sui dati reali. L'output riproduce la Table 2
esattamente, riga per riga, inclusi i p-value — confermando che la tabella
pubblicata è la media per-round.

**Perché conta.** Le due metriche differiscono, e la differenza è più grande
della deviazione che il paper rivendica:

| Gruppo | media per-round | composto | differenza |
|---|---|---|---|
| central ε=1.0 | 0.500472 | 0.502422 | +0.00195 |
| central ε=0.1 | 0.500461 | 0.501970 | +0.00151 |
| dp-fedavg ε=0.5 | 0.499458 | 0.497513 | −0.00195 |
| dp-fedavg ε=1.0 | 0.499960 | 0.498325 | −0.00164 |
| no-DP | 0.500302 | 0.500367 | +0.00007 |

**Rimedio.** Decidere quale metrica è la headline e usarla in entrambe le
tabelle. Se resta la composta, la Table 2 va rigenerata (`check_significance.py`
va modificato per leggere `composed_lira_auc_roc`) e le conclusioni del punto 7
qui sotto vanno riscritte.

---

## 5. "On the same canaries" — SMENTITO

**Paper, §6.2.** Una tabella con due righe presentate come misure sullo stesso
esperimento:

| Metrica (per round, 3 round) | Valori |
|---|---|
| Raw reconstruction-loss AUC | 0.7040 / 0.8951 / 0.8526 |
| Calibrated LiRA AUC *on the same canaries* | 0.6375 / 0.6625 / 0.5875 |

**Dati.** Le due righe vengono da run diversi, con pool diversi:

| File | n_member | n_nonmember | raw per round | LiRA per round |
|---|---|---|---|---|
| `_canary_positive_control/experiment_20260831_102118.json` | 146 | 19 | **0.7040 / 0.8951 / 0.8526** | 0.4863 / 0.6265 / 0.4250 |
| `_canary_positive_control/experiment_20260902_134618.json` | 120 | 20 | 0.6625 / 0.8500 / 0.8375 | **0.6375 / 0.6625 / 0.5875** |

Nel run da cui viene la riga raw, LiRA calibrato fa 0.43–0.63 e il composto 0.553
— sotto la soglia di 0.60. Nel run da cui viene la riga LiRA, il raw è
0.6625/0.85/0.8375, non 0.7040/0.8951/0.8526.

**Aggravanti.**
- Ogni run canary usa **seed 42, 3 round, singola esecuzione**. Nessuna replica.
- Il lato non-membro ha 19–20 campioni contro 120 membri, e i 120 membri sono
  duplicati di sole **5 sessioni template distinte** (`n_templates: 5`,
  `n_duplicates: 30` in `config/experiment_canary_positive_control.yaml`).
  L'AUC poggia quindi su ~5 valori di score distinti sul lato membro, ripetuti:
  ties massivi, intervallo di confidenza enorme e mai calcolato.
- I "gemelli" non sono gemelli. `inject_canaries()` costruisce i membri come
  `dict(template)` a partire dai template di training, ma i non-membri con
  `rng.sample(site_holdout_sessions, n_nonmember_templates)`: sessioni *diverse*
  pescate dall'holdout, non copie degli stessi template. Il confronto misura
  membership **più** difficoltà intrinseca di ricostruzione di quelle 5 sessioni,
  confuse insieme. Il paper le chiama *"20 held-out sibling sessions from the
  same site"*, che suggerisce il contrario.
- La docstring di `inject_canaries()` si contraddice: il primo paragrafo promette
  *"copie singole degli stessi template"*, il Fix del 2026-08-31 più sotto dice
  *"bastano sessioni reali distinte"*, che è ciò che il codice fa.

**Rimedio.** Rifare il positive control con gemelli veri (copie esatte dei
template tenute fuori dal training), lati bilanciati, ≥5 seed, e pubblicare
un'unica tabella da un unico run.

---

## 6. "Raw loss detects it at every site tested" — SMENTITO

**Paper, §6.2:** *"canary members reconstruct 4–7× more accurately at Office 1;
qualitatively similar separation at Caltech and JPL"*.
**Paper, §9, Limitazione #7:** *"the raw reconstruction loss detects it at every
site tested (AUC 0.70–0.90 at Office 1)"*.

**Dati, `canary_raw_mse_auc_roc` per round:**

| Sito | r1 | r2 | r3 |
|---|---|---|---|
| Office 1 | 0.7040 | 0.8951 | 0.8526 |
| Caltech (densità ~10%) | **0.3710** | **0.3805** | **0.4236** |
| JPL | **0.4737** | **0.4632** | **0.4947** |

A Caltech il segnale raw è **invertito** (sotto 0.5 in tutti e tre i round). A JPL
è al caso. Il positive control a livello di loss grezza funziona in **un sito su
tre**, non in tutti.

Questo è il punto più serio dell'intero §6, perché la difesa del null result
poggia interamente sull'affermazione che lo strumento è sensibile. Con raw
invertito a Caltech e LiRA calibrato sotto soglia a Caltech e JPL, la sensibilità
è dimostrata in una configurazione su tre, con un seed.

---

## 7. "No mean deviates from 0.5 by more than 0.0005" — SMENTITO per la metrica dichiarata

**Paper, §7:** *"no mean deviates from 0.5 by more than 0.0005 in absolute
terms"*.

Vero per la media per-round (massimo 0.00047, su `central` ε=1.0). **Falso per la
metrica composta che il paper dice di usare**: `central` ε=1.0 devia di 0.00242 e
`dp-fedavg` ε=0.5 di −0.00249, cioè circa cinque volte il limite dichiarato.

Non cambia la conclusione qualitativa, ma è il tipo di cifra che un revisore
ricalcola dall'artifact.

---

## 8. Esclusione 8σ — IMPRECISO

**Paper, §3.5:** presentata come il quarto dei quattro adattamenti dichiarati di
LiRA, *"we therefore exclude any sample whose score is more than 8 standard
deviations from both μ_in and μ_out from scoring... and log a per-run skip rate
for transparency"*.

**Dati.** `lira_debug_uncalibrated_skip_rate = 0.0000` in tutti e 10 i gruppi
della campagna finale, senza eccezioni. Il meccanismo non esclude nulla.

Il motivo è strutturale: la soglia è calcolata su σ già floorato al floor
simmetrico (vedi punto seguente), quindi gli z-score sono compressi e non
superano mai 8. Nella campagna storica pre-fix il tasso era 2.37%, citato nei
commenti del codice.

**Rimedio.** Va detto che nella campagna riportata il tasso è zero, altrimenti il
lettore attribuisce al filtro un effetto che non ha. In alternativa, spostarlo
dal §3.5 (adattamenti sostanziali) a una nota di implementazione.

---

## 9. "Density matched to Office 1's (≈10%)" — IMPRECISO

**Paper, §6.2:** la replica a Caltech con densità appianata al 10% è usata per
concludere che *"duplicate density alone does not fully explain the Office 1 /
Caltech gap"*.

**Configurazioni reali:**

| Run | n_templates | n_duplicates | record membro | densità aggregata | amplificazione per record |
|---|---|---|---|---|---|
| Office 1 | 5 | 30 | 150 | ~10% di 1.494 | 30/1.494 = **2,0%** ciascuno |
| Caltech high-density | 84 | 30 | 2.520 | ~10% di 25.123 | 30/25.123 = **0,12%** ciascuno |

La densità *aggregata* è appianata, l'amplificazione *per record* differisce di
circa 17×. La memorizzazione di un singolo record — che è ciò che la MIA attacca
— dipende dalla seconda, non dalla prima. I due esperimenti non testano quindi la
stessa variabile, e la conclusione che "la densità non spiega il divario" non
segue.

**Rimedio.** Replicare a Caltech con `n_templates: 5` e `n_duplicates` alzato fino
a eguagliare il 2% per record (circa 500 copie), oppure ritirare la conclusione
sulla densità.

---

## 10. Soglia 0.60 "pre-declared" — IMPRECISO

**Paper, §6.1:** *"well below the 0.60 threshold we pre-declared as evidence of
detectable memorization"*.

**Fonte nel repository.** `docs/TestRoadmap_DSN2027.md:292` la formula come
*"`lira_auc_roc` > 0.80 (o comunque nettamente sopra 0.5, indicativamente ≥0.60
per un pass più..."*. È una soglia indicativa in un documento di lavoro, non una
pre-registrazione. "Pre-declared" in un paper suggerisce un impegno formale
anteriore agli esperimenti, ed è esattamente il punto su cui un revisore attento
alle pratiche di ricerca chiede la data e il documento.

**Rimedio.** Riformulare come "soglia indicativa adottata durante la
progettazione del controllo", oppure documentare data e fonte.

---

## 11. Range dell'asse epoche — IMPRECISO (e internamente incoerente)

**Paper, §6.1, tabella:** riga "Training epochs", range `0.4767 – 0.5155`.

**Dati, `composed_lira_auc_roc` per `_calibration_overfit`:**

| epoche | composto |
|---|---|
| 50 | **0.5363** |
| 150 | 0.5162 |
| 300 | 0.4767 |
| 500 | 0.4792 |
| 1000 | 0.5155 |

Il massimo dell'asse è 0.5363, non 0.5155. Il valore riportato come estremo
superiore è quello a 1000 epoche, non il massimo della riga. Nota che il testo
del §6.1 poco sotto parla correttamente di *"a 0.48–0.54 noise band"*, che
copre 0.5363: la tabella e il testo si contraddicono.

**Rimedio.** Correggere il range in `0.4767 – 0.5363`, oppure escludere
esplicitamente la configurazione a 50 epoche come baseline dell'asse.

---

## 12. Assunzione gaussiana "inherited, not one we introduce" — IMPRECISO per omissione

**Paper, §3.5:** la terza deviazione dichiarata è che la formulazione eredita
*"unmodified, LiRA's own parametric assumption that the shadow loss distribution
is approximately Gaussian"*, presentata come assunzione altrui e lasciata lì.

**Il progetto ha misurato la violazione.**
`docs/MetricsReference_DSN2027.md` §3, run reale del 2026-09-03, no-DP, seed 42,
n_shadow=16, n_member=13.028:

| Pool | skewness | curtosi in eccesso | Jarque-Bera | soglia χ²(2) = 5.99 |
|---|---|---|---|---|
| Member, raw | 10.27 | 172.00 | 16.287.497 | rigetta di ~2,7 milioni di volte |
| Nonmember, raw | 11.36 | 210.90 | 24.406.312 | rigetta di ~4,1 milioni di volte |
| Member, log | 0.15 | 0.88 | 471.62 | rigetta di ~79× |
| Nonmember, log | 0.16 | 0.75 | 365.84 | rigetta di ~61× |

Il documento interno lo chiama giustamente *"non un dettaglio teorico, un fatto
misurato"*. Ereditare un'assunzione e averla misurata falsa di sei ordini di
grandezza sui propri dati sono due cose diverse, e la seconda va nel paper.

C'è anche del materiale positivo qui: la variante log-transform
(`_lira_log_score`, `_LIRA_LOG_SIGMA_FLOOR = 0.05`) riduce la non-normalità di
4–5 ordini di grandezza. È implementata ma non promossa a metrica primaria e non
compare nel paper. Se il null result sopravvive anche in scala logaritmica, è un
argomento molto più forte di quello attuale.

---

## 13. ε come budget DP — IMPRECISO

**Paper, §3.4:** i tre placement sono descritti come DP client-level nel senso di
McMahan et al., *"the unit of protection is a client's per-round contribution as
a whole"*. Il §8 costruisce PES_v1 su `strength(ε) = 1/(1+ε)` trattando ε come
budget nominale.

**Codice, `gradient_manager.py:_compute_sigma()`, docstring:**

> *"ATTENZIONE — weight perturbation, non DP-SGD: [...] Con epochs > 1 per round,
> la sensitività del vettore pesi a un singolo campione NON è formalmente bounded
> da max_grad_norm. La garanzia (ε,δ)-DP formale vale solo per epochs=1. Per il
> paper: descrivere ε come parametro di rumore sperimentale, non come garanzia DP
> formale."*

Gli esperimenti girano a **50 epoche locali** (`config/experiment.yaml`,
`ml.epochs: 50`). Il paper non riporta questo avvertimento da nessuna parte.

Nota che la docstring stessa confonde due livelli: per la DP *client-level* il
clipping del delta vincola la sensibilità indipendentemente dalle epoche locali,
quindi l'avvertimento vale per il livello record, non per quello client. Va
sistemata in entrambe le direzioni — nel codice e nel paper.

**Rimedio.** Nel paper: una frase esplicita che ε è calibrato al livello client e
che nessun placement vincola il contributo del singolo record, il che il §3.4 già
dice bene; e una che chiarisce la relazione con le 50 epoche locali.

---

## 14. Composizione avanzata — CONFERMATO

**Paper, §9:** *"at the T=10 rounds used throughout this campaign it is never
tighter than the naive bound [...] ≥29 rounds at ε=0.1, ≥187 at ε=0.5, and never
at ε=1.0, since ε ≥ ln(2)≈0.693 rules out any finite-round crossover"*.

**Verifica.** `_advanced_composition_epsilon()` implementa fedelmente Dwork &
Roth Thm 3.20: ε' = ε·√(2k·ln(1/δ')) + k·ε·(e^ε − 1). La condizione ε' < kε dà

```
k > 2·ln(1/δ') / (2 − e^ε)²,   valida solo per ε < ln 2
```

Con δ' = 1e-5: ε=0.1 → k > 28.76 (quindi k ≥ 29); ε=0.5 → k > 186.6 (k ≥ 187);
ε=1.0 → nessun crossover perché 2 − e^1 < 0. Tutte e tre le cifre del paper sono
corrette.

Questa parte è solida e va tenuta così com'è.

---

## 15. Sanity check a cinque assi — CONFERMATO (4 righe su 5)

| Asse | paper | dato reale | esito |
|---|---|---|---|
| Model capacity | 0.5294 | 0.529402 | corretto |
| Feature informativeness | 0.5352 | 0.535204 | corretto |
| Capacity + features | 0.5169 | 0.516885 | corretto |
| Training population | 0.5168 | 0.516771 | corretto |
| Training epochs | 0.4767–0.5155 | 0.4767–0.5363 | vedi punto 11 |

Da aggiungere nel paper: tutti e cinque i run sono **seed 42, 3 round, singola
esecuzione**, contro i 10 round e 5 seed della campagna principale. Il §6.1 non
lo dice e la differenza di protocollo è rilevante.

---

## Altri disallineamenti minori

**`n_shadow`.** `src/plugins/attacks/lira.py` passa `kwargs.get("n_shadow", 8)`;
tutti i JSON della campagna riportano `n_shadow=16`. Il paper non dichiara il
valore. Va dichiarato, ed è un parametro che un revisore chiede sempre.

**Nota datata del §7.** La nota interna dice che i numeri sono definitivi al
2026-09-08. `check_significance.py` contiene però un fix del **2026-09-14** che
esclude le cartelle `nvflare-*` dalla campagna principale, con impatto
documentato nel commento su `dp-fedavg` ε=1.0 e sulla sostituzione di tutti e 5 i
punti di `local` ε=1.0. I numeri attuali della Table 2 corrispondono alla versione
corretta, quindi la tabella è aggiornata, ma la nota che la accompagna descrive
uno stato precedente e va riscritta o rimossa.

**`scipy` non dichiarato.** I p-value di Wilcoxon del §7 richiedono scipy, assente
da `pyproject.toml`. Senza, `check_significance.py` cade sul sign test e produce
numeri diversi. Va aggiunto alle dipendenze, altrimenti l'artifact non riproduce
la tabella.

**Statistica con n=5.** La docstring di `_sign_test` dice già, correttamente, che
con n=5 il p minimo a due code è 0.0625 e che il test *"non puo' MAI risultare
significativo ad alpha=0.05 con questa numerosita', qualunque sia il dato
osservato"*. Il paper cita comunque quei p-value come supporto al null. Vanno
sostituiti da un test di equivalenza (TOST) contro un margine dichiarato, per
esempio |AUC − 0.5| < 0.02, che i CI bootstrap sostengono senza difficoltà.

**Config legacy.** `config/clusters.yaml` definisce ancora i quattro cluster
fittizi highway/urban/residential/corporate, che `src/core/autoencoder.py`
documenta come *"vecchio schema fittizio ormai superato"*.
`config/auditor.yaml` elenca `attacks: [FedMIA]` mentre l'attacco primario è LiRA.
Nessuno dei due influenza i risultati, entrambi confondono chi legge l'artifact.

**Commento rivelatore da citare, non da nascondere.**
`config/experiment.yaml`, sulla scelta di 50 epoche:

> *"Con 50 epoche/round il modello overfita sui dati locali, creando il segnale di
> membership necessario per dimostrare che DP lo sopprime. Il claim DSN 2027 'DP
> riduce AUC da X a 0.5' richiede prima X > 0.5 (Scenario A)."*

L'obiettivo di design è dichiarato per iscritto e non è mai stato raggiunto: X è
rimasto a 0.5. Questa è la cosa più onesta del progetto e, raccontata bene, è
anche la più interessante per la comunità dependability.

---

## Difetti tecnici che non sono errori del paper ma vanno corretti nel codice

**Floor simmetrico che degenera LiRA.** `σ_in` e `σ_out` sono entrambi floorati a
`_cluster_sigma_symmetric_floor = max(_cluster_sigma_in_fb, _cluster_sigma_out_fb)`.
Nei dati risultano identici a 5–6 decimali in ogni gruppo (central ε=0.1:
σ_in=0.095939, σ_out=0.095940), con floor-hit-rate **95–99% su central** e **68%
su dp-fedavg/local**. Quando σ_in = σ_out = σ, i termini −log σ si cancellano e il
log-likelihood-ratio si riduce a

```
score = (μ_in − μ_out)/σ² · [ t − (μ_in + μ_out)/2 ]
```

cioè una funzione monotona lineare della loss con soglia al punto medio: esattamente
`_sablayrolles_score()`, che il codice calcola già separatamente poche righe sotto.
La calibrazione della varianza per-esempio, l'unica cosa che distingue LiRA da
Yeom, è spenta nella grande maggioranza dei campioni. È la spiegazione meccanica
del perché LiRA calibrato non batte mai la loss grezza nei canary.

**Score composto sommato su un numero variabile di round.**
`_cumulative_scores[_sid] += lira_score` accumula, ma un campione può essere
saltato in alcuni round (guard cross-cluster, `len(out_losses) < 2`, NaN).
Campioni scorati in 10 round e in 3 round finiscono su scale diverse nella stessa
classifica. Va normalizzato per il conteggio, o va richiesta la presenza in tutti
i round.

**`id()` come chiave di identità.** `_cumulative_scores`, `_sample_is_member`,
`_sample_to_cluster`, `_sample_canary_group` usano tutti `id(sample)`. Qui è
corretto perché `eval_samples` è costruito alla riga 2710, prima del loop dei
round alla 2932, e resta vivo per tutta la funzione. Ma qualunque refactor che
ricostruisca la lista per round romperebbe l'accumulo in silenzio, senza errore,
producendo solo numeri diversi. Meriterebbe un `session_id` stabile.

**Commenti contraddittori sul fallback μ_in.** Il fix del 2026-08-21 (riga ~3485)
dice che l'universo shadow è ora combinato membri+non-membri, quindi un
non-membro campionato come IN finisce correttamente in `in_losses`. Il commento
del ramo `else` (riga ~3595), scritto lo stesso giorno, dice che i non-membri
*"non hanno MAI una calibrazione IN reale, per definizione"*. Uno dei due è
obsoleto, e la differenza determina quanto spesso scatta l'ancoraggio che produce
l'inversione del punto 3.

**Privacy Auditor senza contabilità DP.**
`round_epsilon = (sensitivity / max_grad_norm) * epsilon_budget`: il budget
consumato dipende dalla norma osservata del gradiente, cioè dai dati, mentre la
contabilità DP reale è data-independent per costruzione. Inoltre
`total_rounds_budget` vale 1000 per default mentre gli esperimenti girano 10
round, quindi `budget_ratio` resta attorno all'1% e nessuna soglia di allarme
scatta mai. Nella campagna l'auditor non rileva nulla e, non essendoci mai un
vero positivo, non è mai stato validato come rivelatore — il che è un problema
per il §3.2, che lo presenta come contributo.

---

## Quello che regge

Per equilibrio, e perché serve saperlo prima di riscrivere:

- La test suite passa: **283 test** superati nel sottoinsieme che non richiede
  torch (4 moduli esclusi per dipendenza mancante nel sandbox). Igiene di testing
  sopra la media per un artifact accademico.
- La deduplica per `(dp_mode, epsilon, no_dp, seed)` in `discover_groups()` è
  corretta, e le tre esclusioni successive (`_`, `entity-split`, `nvflare-`) sono
  metodologicamente giuste e ben motivate nei commenti.
- Il bound di composizione avanzata è implementato correttamente e le cifre di
  crossover del paper sono esatte (punto 14).
- La Table 2 è riproducibile: ho eseguito lo script e ottenuto esattamente i
  numeri pubblicati.
- Il livello di chance per TPR@FPR è implementato correttamente come `TPR = FPR`
  e non come 0.5, che è l'errore più comune in questa analisi.
- Il progetto documenta i propri bug e le proprie limitazioni con un livello di
  onestà raro. Diverse cose che segnalo qui sopra le ho trovate perché il codice
  stesso le aveva già scritte nei commenti.

---

## Ordine di intervento suggerito

1. Rimuovere `local` dai risultati, o implementarlo davvero come modalità
   distinta (punto 1).
2. Disattivare il floor simmetrico, alzare `n_shadow`, rimisurare. È l'intervento
   più economico e può cambiare tutto il resto.
3. Simmetrizzare lo scoring tra membri e non-membri, e verificare che
   `matched_formula_auc` risalga verso 0.5 prima di riportare qualunque numero su
   `central` (punto 3).
4. Trovare un punto operativo DP con utility non nulla e produrre la tabella
   utility × ε (punto 2).
5. Rifare il positive control con gemelli veri, lati bilanciati, ≥5 seed, tutti i
   siti (punti 5, 6, 9).
6. Aggiungere Yeom e Shadow alla Table 2: i numeri esistono già in
   `summary.mean_auc_roc` e `mean_shadow_auc_roc`, per ogni seed e configurazione,
   tutti attorno a 0.5.
7. Sostituire i p-value con un test di equivalenza e aggiungere scipy alle
   dipendenze.

Il punto 2 va fatto per primo in assoluto: non richiede rilanciare il training
completo e determina se quello che avete è un null result o un difetto di
calibrazione.
