# Reference metriche — DSN 2027

## Metriche aggiunte il 2026-09-21

| metrica | che cosa misura | documento |
|---|---|---|
| **z (vulnerabilita' per record)** | Quante deviazioni standard il numero di record segnalati dista dal livello di caso, ottenuto permutando i percentili dentro ogni seed. z~0 = il conteggio e' quello del caso; z>3 = eccesso reale. Dice quanto e' improbabile il caso, NON quanto grave sia l'esposizione. | `docs/VulnerabilitaPerRecord.md` sez. 3 |
| **record segnalati** | Sessioni membro in >=2 seed con percentile medio >=90 e minimo >=75. Da leggere SEMPRE accanto agli attesi per caso. | `docs/VulnerabilitaPerRecord.md` sez. 2 |
| **AUC max teorica** | Tetto che (eps,delta)-DP pone su qualunque attacco, calcolato sull'eps CUMULATIVO T*eps. Ai nostri valori e' vacuo (0.99996 a eps_tot=10): va usato per misurare quanto la garanzia sia lasca, non come conferma di sicurezza. | `docs/LimiteTeoricoDP.md` |

Perche' stanno qui e non solo nei documenti dedicati: sono le due quantita'
delle matrici che NON si leggono da sole e che, lette al contrario, portano a
conclusioni sbagliate in direzioni opposte — lo z sottostimando un segnale
reale, il bound teorico dichiarando una sicurezza che non c'e'.

---


Status: **documento vivo, verificato leggendo il codice reale (non a memoria), 2026-09-02.**
Per ogni metrica: dove è implementata, in quali lavori è usata/da cui deriva, perché la usiamo,
formula esatta. Le citazioni sono limitate a quelle già verificate (full-text o abstract-confirmed)
in `docs/ReadingList_DSN2027.md` / `docs/LiteratureReview.md` — dove la citazione non è ancora
verificata, è segnalato esplicitamente invece di essere affermato come certo.

---

## 1. AUC-ROC (attacco MIA, generico)

**Dove**: `sklearn.metrics.roc_auc_score`, usato per Yeom (`scripts/run_experiments.py` ~1214),
Shadow MIA (~1449), canary/canary composto (~2995/3009), LiRA composto (~3188), FedMIA-gradient
pooled/per-cluster (~3496/3519), controllo centralizzato (~3662).

**Perché**: è la metrica standard con cui l'intera letteratura MIA riporta la capacità di un
attaccante di distinguere membri da non-membri — probabilità che lo score assegnato a un membro
scelto a caso sia più alto dello score di un non-membro scelto a caso (equivalente alla statistica
di Mann-Whitney U normalizzata). 0.5 = livello del caso, 1.0 = separazione perfetta. Usata da
Shokri et al. 2017, Yeom et al. 2018 e Carlini et al. 2022 (i tre lavori che questo progetto
implementa direttamente — vedi §3) come una delle metriche di riferimento.

**Limite noto e dichiarato**: Carlini et al. 2022 (LiRA, letto full-text fino a inizio §V, vedi
`docs/ReadingList_DSN2027.md`) argomenta esplicitamente che l'AUC-ROC da sola è una metrica MIA
inadeguata, perché media su ogni soglia di decisione possibile — un attaccante realistico opera a
un FPR basso fissato, dove un AUC alto può nascondere un TPR trascurabile. Per questo il progetto
riporta anche TPR@low-FPR (§2) accanto ad AUC-ROC, non al suo posto.

**Formula**:
```
AUC = P( score(x_member) > score(x_non-member) )   per (x_member, x_non-member) scelti a caso
```

---

## 2. TPR@low-FPR

**Dove**: `scripts/run_experiments.py`, `_TPR_AT_FPR_TARGETS = (0.001, 0.01, 0.05)` (~riga 1007),
calcolato con `np.interp(target, fpr, tpr)` sulla curva ROC di sklearn (`roc_curve`, ~riga 1047).
Aggiunto Sprint 10pp (2026-08-28), roadmap #4.

**Perché**: risponde direttamente al limite di AUC-ROC segnalato sopra. Un attaccante realistico
fissa un tasso di falsi positivi basso (es. 1%) e chiede quanti membri veri riesce a recuperare a
quella soglia — è la metrica che Carlini et al. 2022 argomentano essere la più informativa per un
vero avversario, e che quindi il paper deve riportare per non limitarsi a un'AUC media su ogni
soglia.

**Formula**: dalla curva ROC (coppie (FPR(t), TPR(t)) al variare della soglia t), interpolazione
lineare del TPR ai punti FPR target:
```
TPR@FPR=α  =  interp(α, {FPR(t)}, {TPR(t)})     per α ∈ {0.001, 0.01, 0.05}
```

**Nota**: non presente nei 40 JSON della campagna 5-seed×8-config (Sprint 10tt) — aggiunto al
codice dopo che quella campagna era già in esecuzione. Da includere in una campagna futura.

---

## 3. LiRA — score log-likelihood-ratio Gaussiano

**Dove**: `run_lira()` in `scripts/run_experiments.py` (~righe 2886-2918).

**Fonte**: Carlini, Chien, Nasr, Song, Terzis, Tramèr, *"Membership Inference Attacks From First
Principles,"* IEEE S&P 2022 (arXiv:2112.03570) — **l'attacco primario del progetto**, letto
full-text fino a inizio §V (vedi Tier 0, `docs/ReadingList_DSN2027.md`). Attacca precisamente
l'artefatto che il modello di minaccia del progetto assume osservabile: l'update per-client prima
dell'aggregazione FedAvg (stesso threat model del paper QRS 2026 di riferimento, Imperatrice &
Romano — vedi README "Relation to Prior Work").

**Perché**: rispetto a Yeom (soglia fissa su una singola loss) e Shadow (classificatore addestrato
su shadow model), LiRA calibra due distribuzioni Gaussiane per-campione (IN = distribuzione della
loss quando il campione è membro di uno shadow, OUT = quando non lo è) usando un ensemble di shadow
model realmente riaddestrati per round (warm-start) — è l'attacco più potente dei tre disponibili
nel progetto, ed è per questo enfatizzato come "★ PRIMARY" nella narrativa (Sprint 10ccc).

**Deviazioni dichiarate rispetto alla costruzione originale di Carlini et al.** (documentate in
`docs/ReadingList_DSN2027.md`, da riportare nel paper): (1) i loro shadow model sono addestrati
indipendentemente su subset casuali, senza warm-start da una traiettoria round-su-round — la nostra
versione è un adattamento sostanziale; (2) LiRA originale è costruita per punteggi di confidenza di
classificazione (logit-trasformati), non per MSE di ricostruzione di un autoencoder non
supervisionato, come usiamo qui; (3) un quarto scostamento trovato debuggando (non da un
secondo read): quando `target_loss` è troppo lontano (>8σ) da entrambe le medie IN/OUT stimate,
il progetto esclude il campione dallo scoring (`_UNCALIBRATED_Z_THRESHOLD = 8.0`) invece di
assegnare uno score arbitrario — scelta metodologica non prevista dalla costruzione originale.

**Formula**:
```
log_p_in  = -0.5 * ((target_loss - μ_in)  / σ_in)  ** 2  -  log(σ_in)
log_p_out = -0.5 * ((target_loss - μ_out) / σ_out) ** 2  -  log(σ_out)
lira_score = clip( log_p_in - log_p_out, -20.0, 20.0 )
```
dove μ_in/σ_in (μ_out/σ_out) sono media e deviazione standard della loss del campione target sugli
shadow in cui era membro (rispettivamente non-membro). È la log-densità di una normale univariata
valutata al punto `target_loss`, confrontata tra le due ipotesi — un test del rapporto di
verosimiglianza, non una soglia arbitraria.

**Fondamento teorico (aggiunto 2026-09-03, su richiesta esplicita dell'utente — tre citazioni
consecutive di Carlini et al. 2022 §IV, §3.1 discusse in chat)**: la formula sopra non è
un'euristica — è la derivazione in tre passi che il paper LiRA fa esplicitamente, qui ricostruita
e collegata al codice riga per riga.

1. **Lemma di Neyman-Pearson**: il test di ipotesi ottimale a un tasso di falsi positivi FISSO è
   una soglia sul test del rapporto di verosimiglianza tra le due ipotesi (membro/non-membro):
   `Λ(f; x, y) = p(f | Q_in(x,y)) / p(f | Q_out(x,y))` — dove `Q_in`/`Q_out` sono le distribuzioni
   (sui possibili modelli allenati) rispettivamente CON e SENZA il campione (x,y) nel training set.
   Questo è il motivo per cui LiRA (un test del rapporto di verosimiglianza) è teoricamente il
   migliore attacco possibile a parità di FPR bersaglio — non solo empiricamente il più forte dei
   tre nel progetto (vedi "Perché" sopra), ma quello con la garanzia teorica più forte. Coerente
   con la scelta di riportare TPR@low-FPR/Advantage (§2/§10) invece del solo AUC: il lemma stesso
   parla di "FPR fisso", non di una media su tutte le soglie.
2. **Il test esatto è intrattabile**: calcolare `Q_in`/`Q_out` richiederebbe conoscere la
   distribuzione sui modelli allenabili con/senza (x,y) — impossibile in pratica (richiederebbe
   riaddestrare infiniti modelli e conoscerne la distribuzione esatta dei parametri).
3. **Riduzione a una statistica 1-dimensionale trattabile**: si sostituiscono le due probabilità
   con `p(ℓ(f(x),y) | Q̃_in/out(x,y))` — la distribuzione della SOLA loss del modello sul campione
   (non l'intero spazio dei parametri), stimabile per query al modello. Questo è esattamente
   `target_loss` nella formula sopra (nel nostro caso MSE di ricostruzione, non cross-entropy —
   vedi deviazione (2) già dichiarata sopra); Q̃_in/Q̃_out approssimate come Gaussiane
   (μ_in/σ_in, μ_out/σ_out) stimate empiricamente dagli shadow — l'approssimazione parametrica che
   rende il passo 3 calcolabile in pratica, al prezzo di assumere che la loss sia
   approssimativamente Gaussiana condizionata a IN/OUT (assunzione della LiRA originale, non
   nostra, ereditata insieme al resto della costruzione).

In sintesi: Neyman-Pearson dice qual è il test ottimale (irraggiungibile esattamente) → il paper
lo rende trattabile riducendolo a una statistica 1D (la loss) → noi stimiamo quella statistica con
un fit Gaussiano sugli shadow, esattamente il codice sopra. Citazione da aggiungere nel paper: gli
enunciati (1)-(3) sono la Sezione IV/3.1 di Carlini et al. 2022, non una parafrasi libera.

**Modellazione parametrica vs non parametrica (aggiunto 2026-09-03, su domanda esplicita
dell'utente: "usiamo una modellazione parametrica o non parametrica? è importante da citare?")**:
**il progetto usa una modellazione PARAMETRICA** — il fit Gaussiano (μ_in/σ_in, μ_out/σ_out) sopra
è esattamente questo, non un dettaglio implementativo neutro. È una scelta esplicita e giustificata
di Carlini et al. 2022 (§IV-C, intitolata letteralmente *"Estimating the likelihood-ratio with
parametric modeling"*), non l'unica opzione possibile — il paper stesso discute un'alternativa
**non parametrica** (stimare Q̃_in/Q̃_out empiricamente dai dati, senza assumere una forma
Gaussiana, es. Ye et al., lavoro concorrente citato nello stesso paper) e sceglie quella
parametrica per due motivi dichiarati, verificati sul testo (non a memoria):
1. **Efficienza campionaria**: "Parametric modeling requires training fewer shadow models to
   achieve the same generalization of nonparametric approaches" — Carlini et al. riportano un
   fattore **400×** in meno di shadow model necessari rispetto al confronto non parametrico citato.
   Rilevante per questo progetto: ogni shadow model richiede un retraining FL completo (costoso),
   quindi la scelta parametrica non è solo quella "storica" ereditata ma anche quella
   economicamente giustificata per continuare a usare `n_shadow` nell'ordine di 8-32 invece di
   centinaia.
2. **Estendibilità multivariata**: il paper nota che la modellazione parametrica si estende
   naturalmente a fit Gaussiani multivariati per query multiple sullo stesso campione — non
   sfruttato in questo progetto (nessuna query augmentation, coerente con un target
   autoencoder/non-immagine), ma un limite dichiarabile esplicitamente, non un'omissione silenziosa.

**Quanti shadow model, e come vengono generati i loro dati di training (aggiunto 2026-09-09, su
domanda esplicita dell'utente)**: `n_shadow` è impostato a **16 per cluster/sito** (default in
`config/experiment.yaml`, commentato esplicitamente come "buona qualità, consigliato per il paper
submission" — 8 resta disponibile solo per iterazione rapida/smoke test). Con i **3 siti reali**
del progetto (Caltech/JPL/Office1), questo significa **48 shadow model addestrati per round**, non
16 in totale — ognuno riaddestrato ogni round con warm-start (non da zero), come previsto dalla
costruzione LiRA di Carlini et al. 2022 citata sopra.

I dati di training di ciascuno shadow **non sono sintetici**: `run_lira()` (righe ~2774-2828 di
`scripts/run_experiments.py`) campiona ogni shadow da un pool di sessioni REALI specifico del suo
sito — l'unione delle sessioni membro (usate nel training FL reale di quel sito) e delle sessioni
hold-out (mai viste dal training FL, riservate come pool di non-membri) — poi partiziona
casualmente quel pool in IN/OUT per costruire le due distribuzioni Gaussiane per-shadow. Non viene
generato alcun dato sintetico (né tramite ricerca hill-climbing sullo spazio degli input, né tramite
statistiche di popolazione, né tramite perturbazione di dati reali con feature invertite): l'intero
pool di training degli shadow è dati ACN-Data realmente osservati. Questo riflette il modello di
minaccia del progetto — un aggregatore FL semi-onesto ha per costruzione accesso diretto a dati
reali della stessa distribuzione (le sessioni degli altri client/round), quindi non ha bisogno di
sintetizzarli.

**Sì, è importante da citare nel paper**: non è solo "abbiamo implementato LiRA", è "abbiamo
implementato la variante PARAMETRICA di LiRA, la stessa scelta e per le stesse ragioni degli
autori originali" — una frase in più nella sezione metodologia che preempta la domanda naturale di
un revisore ("perché Gaussiane e non una stima empirica diretta?").

**Deviazione da elevare esplicitamente (finora solo accennata dentro la deviazione (2) sopra, non
isolata come punto a sé)**: Carlini et al. NON applicano il fit Gaussiano alla confidenza/loss
grezza — prima trasformano il punteggio con un **logit scaling** (`φ(p) = log(p/(1-p))`)
specificamente perché, come mostrano loro stessi (Fig. 4/8 del paper), la confidenza/loss grezza
NON è ben approssimata da una Gaussiana, mentre il logit sì (verificato empiricamente da loro, non
assunto). **ChargeShield-FL fitta la Gaussiana direttamente sulla MSE di ricostruzione grezza,
senza alcuna trasformazione equivalente** — non esiste un logit naturale per un errore di
ricostruzione di un autoencoder non supervisionato (nessuna probabilità/confidenza da cui partire),
quindi non è un errore di implementazione, ma è una deviazione le cui conseguenze non sono state
verificate: **non sappiamo se la MSE grezza di questo progetto sia effettivamente
approssimativamente Gaussiana quanto lo è il logit-confidence nel paper originale.**

**Collegamento non ancora testato, da segnalare come domanda aperta (non una conclusione)**: il
progetto osserva da tempo (`lira_debug_sigma_in_floor_hit_rate`/`..._sigma_out_floor_hit_rate`,
`scripts/run_experiments.py`) che σ_in/σ_out toccano il valore-soglia minimo nel 96-99% dei casi in
diversi sweep — finora sempre spiegato con cause di dimensione/struttura del campione (pochi
shadow, popolazione IN/OUT sbilanciata per cluster, collasso numerico da warm-start; vedi
`docs/TestRoadmap_DSN2027.md`). **Una spiegazione alternativa, non ancora considerata né esclusa**:
se la MSE grezza non è approssimativamente Gaussiana (a differenza del logit-confidence per cui il
fit Gaussiano è stato validato dagli autori originali), il fit Gaussiano stesso potrebbe essere
sistematicamente povero indipendentemente dalla dimensione campionaria — un problema di forma
della distribuzione, non (solo) di numerosità. **Non è possibile verificarlo con i dati già
salvati**: i JSON storici e anche il nuovo dump per-campione (task #50) salvano solo lo score
COMPOSTO (post-fit), non la distribuzione grezza delle MSE per-shadow che servirebbe per un test di
normalità (es. Shapiro-Wilk o un QQ-plot). Verificarlo richiederebbe una nuova estrazione dei dati
grezzi, non ancora implementata — segnalato qui come possibile lavoro futuro/limite dichiarato nel
paper, non eseguito in questa voce.

**AGGIORNAMENTO (2026-09-03, task #57) — ora implementato, non più solo segnalato.** Su domanda
diretta dell'utente ("quando lo verifichiamo? cosa dobbiamo fare?"):
- Nuovo parametro opt-in `raw_loss_dump_path` su `run_lira()` — dumpa le liste COMPLETE (non solo
  la media) di `_diag_raw_loss_members`/`_diag_raw_loss_nonmembers`, già raccolte internamente ogni
  round ma finora mai salvate per intero (solo `lira_debug_raw_loss_member_mean`/`..._nonmember_mean`,
  §3 diagnostica esistente) — zero calcolo aggiuntivo, solo I/O. CLI: `--raw-loss-dump PATH`.
- Nuovo `scripts/check_gaussian_fit.py` — confronta MSE grezza vs `log(MSE + eps)` per il pool
  member/nonmember di un round, calcolando skewness, curtosi in eccesso, e statistica di
  **Jarque-Bera** (JB = n/6·(S²+K²/4), confrontata con la soglia χ²(2) al 5% = 5.99, valore
  hardcoded — nessuna dipendenza scipy, coerente col resto del progetto). Stesso tipo di confronto
  che Carlini et al. fanno tra confidenza/log-loss/logit nella loro Fig. 4/8, adattato al nostro
  dominio (nessun logit possibile per una MSE di ricostruzione — il log è il candidato più naturale,
  dato che la MSE è semi-illimitata positiva, non bounded in [0,1] come una confidenza).
- **Quando verificarlo**: NON serve aspettare la campagna 5-seed×10-config (task #52/#55) — è una
  domanda sulla FORMA della distribuzione, non una misura che richiede rigore multi-seed. Un run
  singolo, veloce, basta:
  ```bash
  cd /Users/susetta/Documents/ChargeShield-FL
  python3 scripts/run_experiments.py \
    --config config/experiment.yaml --no-dp --rounds 3 --seed 42 --n-shadow 16 \
    --sweep-dir experiments/_check_gaussian_fit \
    --raw-loss-dump experiments/_check_gaussian_fit/raw_loss.json
  python3 scripts/check_gaussian_fit.py experiments/_check_gaussian_fit/raw_loss.json
  ```
  (`--rounds 3` invece di 10 — la forma della distribuzione non richiede convergenza completa del
  modello, solo campioni scorati; se il tempo lo permette, ripetere anche con DP attivo, es.
  `--epsilon 1.0` al posto di `--no-dp`, per controllare se il rumore DP cambia la forma.)
- **Lettura del risultato**: se JB(log) << JB(raw) e JB(raw) supera 5.99, conferma l'ipotesi — la
  MSE grezza non è ben approssimata da una Gaussiana (come la confidenza grezza nel paper
  originale), e un log-transform prima del fit potrebbe migliorare la qualità della calibrazione
  σ_in/σ_out (e potenzialmente ridurre il floor-hit-rate, ipotesi ancora da testare separatamente
  con un run reale che confronti σ floor-hit-rate con/senza trasformazione). Se invece JB(raw) è già
  sotto 5.99, l'assunzione Gaussiana regge anche senza trasformazione, e il floor-hit-rate osservato
  ha una causa diversa (dimensione/struttura campionaria, come finora ipotizzato). **Non ancora
  eseguito con dati reali** — py_compile OK, 10 nuovi test puro-Python in `tests/test_gaussian_fit.py`
  (158 test non-torch totali), smoke test sintetico verificato (dati log-normali generati: JB
  raw=5148/4893, JB log=3.65/2.35 — il metodo distingue correttamente i due casi come atteso).

**RISULTATO REALE (2026-09-03, task #57, run dell'utente, no-DP, seed=42, n_shadow=16, round 3/3,
n_member=13028, n_nonmember=13018)**:

| Pool | | skewness | curtosi in eccesso | Jarque-Bera | vs soglia 5.99 |
|---|---|---|---|---|---|
| Member | raw | 10.27 | 172.00 | 16 287 496.97 | rigetta di ~2.7 milioni di volte |
| Member | log | 0.15 | 0.88 | 471.62 | rigetta, ma di ~79× (non ~2.7M×) |
| Nonmember | raw | 11.36 | 210.90 | 24 406 311.87 | rigetta di ~4.1 milioni di volte |
| Nonmember | log | 0.16 | 0.75 | 365.84 | rigetta, ma di ~61× (non ~4.1M×) |

**Ipotesi CONFERMATA per la parte grezza**: la MSE di ricostruzione non è minimamente approssimata
da una Gaussiana (skewness~10-11, un valore enorme — una Gaussiana ha skewness=0; curtosi in
eccesso~172-211, anch'esso enorme — una Gaussiana ha 0) — esattamente il tipo di violazione che
Carlini et al. documentano per la confidenza/loss grezza (loro Fig. 4), qui ancora più marcata. Il
fit Gaussiano di LiRA (§3 sopra) sta quindi operando su una distribuzione la cui forma viola
platealmente l'assunzione alla base del test parametrico — non un dettaglio teorico, un fatto
misurato.

**Il log-transform migliora enormemente ma NON "risolve" formalmente**: skewness scende da ~10-11 a
~0.15-0.16 (un fattore ~70×, verso una forma quasi simmetrica) e la curtosi in eccesso da ~172-211 a
~0.75-0.88 (verso quasi-Gaussiana) — un miglioramento di 4-5 ordini di grandezza nel punteggio JB.
**Ma il test JB rigetta comunque la normalità anche per il log-transform**, a p=0.05. **Nota
statistica importante, per non sopravvalutare questo secondo rigetto**: il test di Jarque-Bera (come
ogni test di normalità) ha potenza statistica che CRESCE con la dimensione campionaria — con
n≈13 000, anche deviazioni minime e praticamente irrilevanti da una Gaussiana perfetta vengono
dichiarate "statisticamente significative". Uno skewness di 0.15 e una curtosi in eccesso di 0.88 su
13 000 campioni sono deviazioni modeste in termini pratici (una Gaussiana con questi valori sarebbe
visivamente quasi indistinguibile da una perfetta in un istogramma), ma bastano a superare la soglia
fissa 5.99 a questa numerosità. Il confronto onesto non è "il log-transform ha fallito" ma "il
log-transform ha ridotto la non-normalità di ~4-5 ordini di grandezza, restando comunque misurabile
a questa dimensione campionaria" — una conclusione compatibile con quella di Carlini et al. (il
logit-transform della confidenza "migliora" senza garanzia di normalità perfetta) più che una
smentita.

**Implicazione pratica per LiRA**: dato che (a) la MSE grezza è drasticamente non-Gaussiana e (b) il
`σ_in`/`σ_out` floor-hit-rate osservato in questo stesso run è ~98-100% in ogni round (vedi log:
`floor_hit=0.9925/0.9981/0.9982`), l'ipotesi aperta in §"Modellazione parametrica vs non
parametrica" sopra — che la forma non-Gaussiana della MSE grezza contribuisca al floor-hit-rate,
non solo la dimensione campionaria — **è ora supportata da dati reali, non solo plausibile**.

**AGGIORNAMENTO (2026-09-03, task #61) — implementata la variante esplorativa per la prova
causale.** Su richiesta diretta dell'utente dopo il risultato reale sopra, `run_lira()` calcola ora
un terzo scorer opzionale (sempre attivo, zero shadow aggiuntivi) — `_lira_log_score()` — che fa lo
stesso fit log-likelihood-ratio Gaussiano di LiRA ma su `log(MSE+eps)` invece che su MSE grezza,
riusando gli stessi `in_losses`/`out_losses` già raccolti per il fit raw. **Semplificazioni
dichiarate rispetto al fit raw** (vedi docstring completa della funzione): fallback
`μ_in_log=μ_out_log` quando mancano ≥2 osservazioni IN reali (invece dell'ancoraggio per-cluster
raffinato del fit raw — un fallback più debole, non più forte: non può gonfiare artificialmente
l'AUC a favore della variante log) e floor fisso `0.05` invece di scale-adattivo (ragionevole perché
il log-transform comprime già la scala grezza in un intervallo comparabile). Questo è un ablation
esplorativo per rispondere alla domanda specifica "il log-transform riduce il floor-hit-rate e/o
migliora AUC/TPR?" — non ancora sottoposto allo stesso rigore (worst-case check §10c, canary
positive-control §6) del fit raw prima di essere promosso a metrica primaria.

**Campi risultato per round**: `lira_log_auc_roc`, `lira_log_advantage`, `lira_log_confusion`,
`lira_log_n_member`/`_n_nonmember`, `lira_log_tpr_at_fpr_{0.001,0.01,0.05}` — stesso pattern di
Sablayrolles (§10f) — più `lira_log_debug_sigma_in_mean`/`_floor_hit_rate` e
`lira_log_debug_sigma_out_mean`/`_floor_hit_rate`, da confrontare DIRETTAMENTE con
`lira_debug_sigma_in_floor_hit_rate`/`lira_debug_sigma_out_floor_hit_rate` del fit raw (sopra) — la
prova causale che mancava. Curva ROC completa inclusa nel dump `roc_curve_dump_path` (§10e) come
chiave `"lira_log"` — `scripts/plot_roc_log_scale.py --subkey lira_log` la plotta senza modifiche.
7 nuovi test puro-Python (`tests/test_lira_log_score.py`, stesso limite/pattern di
`tests/test_sablayrolles_score.py` — copre fallback, direzione del segno, floor, clipping,
separazione su popolazione giocattolo). Suite non-torch 165→171. **Non ancora eseguito con dati
reali** — il prossimo run con `--raw-loss-dump` (o qualunque run LiRA) produrrà questi campi
automaticamente, permettendo il confronto diretto floor-hit-rate raw vs log richiesto.

**RISULTATO REALE (2026-09-03, run dell'utente, stessa config del task #57 — no-DP, seed=42,
n_shadow=16, round 3/3, n_member=13028, n_nonmember=13018)**:

| Metrica | RAW (fit su MSE grezza) | LOG (fit su log(MSE)) |
|---|---|---|
| σ_in floor-hit-rate | **0.9982** (99.82%) | **0.0008** (0.08%) |
| σ_out floor-hit-rate | 0.9982 (99.82%) | 0.0005 (0.05%) |
| AUC-ROC | 0.5043 | 0.5057 |
| TPR@0.1%FPR | 0.0012 | 0.0011 |
| Advantage | 0.0122 | 0.0138 |

**Prova causale ottenuta**: il log-transform elimina quasi completamente il floor-hit-rate (da
~99.8% a <0.1%, un fattore ~2000×) — conferma diretta e inequivocabile che la forma non-Gaussiana
della MSE grezza (§ sopra: skewness~10-11, JB~16-24 milioni) era la causa del collasso del floor,
non (solo) la dimensione campionaria come ipotizzato inizialmente.

**Ma questo NON si traduce in un attacco più efficace**: AUC, TPR@low-FPR e Advantage restano
praticamente identici tra le due varianti (differenze nell'ordine del rumore statistico, non
sistematiche) — entrambe restano vicine al caso random (AUC~0.50). **Conclusione onesta e
importante per il paper**: il floor-hit-rate era un artefatto tecnico reale del fit (ora corretto
concettualmente da questa variante), ma NON era la causa dell'AUC vicino a 0.5 osservato in questi
esperimenti — l'assenza di segnale di membership persiste anche quando l'artefatto del floor viene
rimosso. Questo rafforza (non indebolisce) la conclusione "Scenario B" già documentata altrove
(nessuna leakage rilevabile in primo luogo, non un problema di misurazione mascherato da un σ
collassato) — un risultato negativo pulito, verificato con un ablation mirato, non solo assunto.

**REPLICA INDIPENDENTE (2026-09-04, seed=123 invece di 42 — stessa config, `--sweep-dir
experiments/_check_gaussian_fit_seed123`, `experiment_20260904_081146.json`, round 3/3)**: prima
conferma con un seed diverso (i run precedenti con seed=42 erano deterministici — stesso seed,
stesso identico risultato ad ogni rilancio, quindi non erano repliche indipendenti):

| Metrica | RAW | Sablayrolles | LOG |
|---|---|---|---|
| σ_in floor-hit-rate | 0.9985 (99.85%) | — | 0.0010 (0.10%) |
| σ_out floor-hit-rate | 0.9986 (99.86%) | — | 0.0009 (0.09%) |
| AUC-ROC | 0.4993 | 0.5006 | 0.5001 |
| TPR@0.1%FPR | 0.00075 | 0.0011 | 0.0010 |
| Advantage | 0.0095 | 0.0078 | 0.0067 |

Stesso quadro qualitativo del run seed=42: il log-transform riduce il floor-hit-rate di ~3 ordini
di grandezza (99.9%→0.1%), ma AUC/TPR/Advantage restano indistinguibili dal caso random su **tutti
e tre** gli scorer (raw, Sablayrolles, log) — nessuno mostra segnale di membership sopra il rumore.
Con due seed indipendenti che convergono sulla stessa conclusione, "Scenario B" (nessuna leakage
rilevabile) è ora supportato da evidenza replicata, non da un singolo run.

---

## 4. Yeom (baseline, loss-threshold)

**Dove**: `run_experiments.py` (~riga 1214, `roc_auc_score` su score = -loss).

**Fonte**: Yeom, Fredrikson, Jha, *"Privacy Risk in Machine Learning: Analyzing the Connection to
Overfitting,"* IEEE CSF 2018 — attacco più debole tra i tre, usato come baseline/lower-bound. Lo
stesso paper fornisce anche il bound teorico usato in §8 (MIA Advantage) sotto.

**Perché**: è l'attacco MIA più semplice riportabile onestamente — un membro tende ad avere loss
più bassa di un non-membro, senza alcuna calibrazione shadow. Serve da riferimento minimo: se
nemmeno Yeom rileva nulla, un attacco più sofisticato (LiRA) che rileva qualcosa è un segnale
tanto più significativo, e viceversa se anche Yeom fallisce ovunque (come nel progetto) rafforza
la conclusione di assenza di leakage misurabile.

**Formula**: score = -loss(x) sul modello target; AUC-ROC come in §1 su questo score.

**Aggiornamento (Sprint 10zz+28, 2026-09-03, task #53)**: `run_fedmia()` calcola ora anche
`tpr_at_fpr_0.001/0.01/0.05`, `advantage`, `confusion` sulle stesse coppie label/score — vedi §10c
per la motivazione (confrontare Yeom con Shadow/LiRA solo via AUC-ROC cadrebbe nella stessa
"fallacia delle medie" di Carlini et al. 2022, qui applicata al confronto TRA attacchi invece che
dentro un attacco). Non retroattivo sui JSON storici.

---

## 5. Shadow MIA (classificatore)

**Dove**: `run_experiments.py` (~riga 1449).

**Fonte**: Shokri, Stronati, Song, Shmatikov, *"Membership Inference Attacks Against Machine
Learning Models,"* IEEE S&P 2017 (arXiv:1610.05820) — origine dell'idea di "shadow model" (addestra
un modello attaccante a riconoscere differenze di comportamento membro/non-membro), diretto
antenato sia di questo attacco sia di LiRA. Citazione abstract-confirmed (non ancora full-text),
`docs/ReadingList_DSN2027.md` Tier 0.

**Perché**: rappresenta il livello intermedio dei tre attacchi (più sofisticato di Yeom, meno di
LiRA), utile come punto di confronto per isolare quanto del segnale di LiRA derivi dalla
calibrazione Gaussiana specifica piuttosto che dal solo uso di shadow model.

**Formula**: AUC-ROC (§1) sullo score prodotto dal classificatore attaccante addestrato sugli
output degli shadow model.

**Aggiornamento (Sprint 10zz+28, 2026-09-03, task #53)**: `run_fedmia_shadow()` calcola ora anche
`tpr_at_fpr_0.001/0.01/0.05`, `shadow_advantage`, `shadow_confusion` — stesso motivo di §4. Non
retroattivo sui JSON storici.

**Deviazioni dichiarate rispetto alla costruzione originale (Shokri et al. 2017, aggiunto
2026-09-10)**: come già fatto per Carlini et al. 2022 (§3), dichiariamo esplicitamente dove Shadow
MIA si discosta dalla costruzione originale di Shokri — non per minimizzare il debito, ma perché la
citazione è comunque motivata (origine diretta del concetto di shadow model, §"Fonte" sopra) e le
differenze vanno rese esplicite prima della submission:
- **Dati di training degli shadow model**: Shokri usa tre tecniche per sintetizzare dati shadow
  quando non è disponibile un dataset realistico (model-based synthesis via hill-climbing sul
  modello target, statistics-based synthesis, o dati reali rumorosi). ChargeShield-FL non ne ha
  bisogno: gli `n_shadow=16` modelli per sito sono addestrati su dati reali campionati dai pool
  member∪holdout dello stesso sito (`_sample_preserving_canary_groups()`, vedi nota su n_shadow più
  sopra) — un'assunzione di minaccia più forte (onest-ma-curioso con accesso ai dati locali reali),
  non un'approssimazione del caso in cui l'attaccante non ha dati realistici.
- **Metriche riportate**: Shokri riporta precision/recall (e F1 implicito). ChargeShield-FL riporta
  AUC-ROC, TPR@FPR fisso, Advantage e matrice di confusione (§1, §4, §10b) — scelta guidata dal
  fatto che precision/recall dipendono dalla soglia di decisione e dal bilanciamento member/non
  member del dataset di valutazione, mentre AUC-ROC/TPR@low-FPR sono invarianti alla soglia e
  standard nella letteratura DP-audit più recente (Carlini et al. 2022; Jagielski et al. 2020).
- **Incertezza della previsione**: Shokri usa anche l'entropia normalizzata del vettore di
  probabilità in output come feature/segnale aggiuntivo. Non applicabile qui: l'autoencoder di
  ChargeShield-FL è un modello ricostruttivo (MSE di ricostruzione come segnale, non un
  classificatore multi-classe con vettore di probabilità in output), quindi non esiste un
  equivalente diretto dell'entropia della softmax.
- **Regolarizzazione**: Shokri nota che i modelli target con overfitting più marcato (anche indotto
  con meno regolarizzazione) sono più vulnerabili al suo attacco. ChargeShield-FL non usa dropout
  (vedi `docs/MLPlane.md` §4.2, "Regolarizzazione — nessun dropout") — la difesa primaria contro la
  memorizzazione qui è il rumore DP (`GradientManager`), non una regolarizzazione anti-overfitting
  lato architettura, per cui il confronto diretto con la sensibilità di Shokri al livello di
  overfitting non è la leva principale studiata in questo lavoro.

---

## 6. Canary positive control (AUC su record iniettati)

> **Aggiornamento 2026-09-15 (Sprint 10zz+109).** Il protocollo, il
> confondente membership/difficolta' di ricostruzione che lo invalidava, i
> due parametri nuovi (`nonmember_source`, `swap_assignment`) e i baseline a
> inizializzazione casuale sono documentati per esteso in
> `docs/CanaryPositiveControl.md`. In sintesi: i due lati escono ora da un
> pool unico con assegnazione casuale; l'AUC va letta come differenza
> rispetto al baseline a init (0.5285 base, 0.4810 swap su office1 seed 42,
> non 0.5), e il controllo di scambio e' obbligatorio prima di citare
> qualunque valore. I numeri canary raccolti prima di questo sprint sono
> invalidati.

**Dove**: `inject_canaries()` + campi `canary_auc_roc`/`canary_composed_auc_roc`/
`canary_raw_mse_auc_roc` in `run_lira()` (Sprint 10vv-10yy).

**Fonte**: tecnica standard in letteratura DP/MIA — canary insertion (Carlini, *"The Secret
Sharer,"* USENIX Security 2019; Jagielski et al., *"Auditing Differentially Private Machine
Learning,"* NeurIPS 2020). Non ancora verificate full-text/abstract in questo giro di ricerca —
citazioni da confermare prima della submission.

**Perché**: il sanity-check a 5 assi (escalation naturale di memorizzazione) dimostra solo che la
memorizzazione *naturale* è difficile da indurre in questa architettura — non dimostra che
l'harness LiRA sarebbe capace di rilevare una violazione di privacy vera se esistesse. Il canary
positive control inserisce deliberatamente record duplicati amplificati per verificare la
sensibilità dell'harness stesso, indipendentemente dal risultato principale.

**Formula**: identica ad AUC-ROC (§1), ristretta al sottoinsieme di record canary iniettati
(membri = copie nel training set, non-membri = "gemelli" mai visti in training). `canary_raw_mse_auc_roc`
usa lo score raw (-loss di ricostruzione del modello target) invece del punteggio LiRA calibrato —
distinzione rilevante perché nel progetto il primo rileva il canary (0.70-0.90) mentre il secondo
resta instabile (0.43-0.63), un limite di calibrazione LiRA identificato meccanicamente (contaminazione
degli shadow model, che si addestrano sulla stessa popolazione contenente i duplicati) e documentato
in Sprint 10yy.

---

## 7. Bootstrap CI

**Dove**: `bootstrap_ci()` in `scripts/check_significance.py` (righe 163-174).

**Fonte**: tecnica statistica standard (bootstrap percentile), non specifica a un singolo paper MIA
citato dal progetto — nessuna citazione da verificare, è metodologia generica.

**Perché**: un AUC medio ~0.5 su N=5 seed da solo non dice se è statisticamente indistinguibile dal
caso o solo una stima puntuale senza intervallo attorno. Il bootstrap fornisce l'intervallo di
confidenza al 95% attorno alla media osservata senza assumere normalità.

**Formula**: percentile bootstrap, pura libreria standard (nessuna dipendenza da scipy, assente in
questo sandbox):
```
per i in 1..n_resamples (default 10000):
    campiona con reinserimento n valori da `values`
    calcola la media del campione
ordina le n_resamples medie
CI_95% = [ media al percentile 2.5%, media al percentile 97.5% ]
```

---

## 8. Test di significatività (Wilcoxon / sign test fallback)

**Dove**: `significance_test()` in `scripts/check_significance.py` (righe 83-108), aggiunto Sprint
10pp (roadmap #5) su richiesta esplicita di un revisore.

**Fonte**: test dei ranghi con segno di Wilcoxon — test statistico non parametrico standard,
nessuna citazione specifica al progetto necessaria.

**Perché**: verifica se gli AUC medi dei 5 seed di un gruppo sono significativamente diversi da
0.5 (livello del caso), oltre al bootstrap CI. Usa `scipy.stats.wilcoxon` se disponibile
(assente in questo sandbox, va verificato sulla macchina reale); fallback dichiarato — mai
silenzioso — su un sign test binomiale esatto puro-Python quando scipy manca.

**Formula (Wilcoxon)**: rango con segno delle differenze `v_i - 0.5`, statistica standard di
Wilcoxon (via `scipy.stats.wilcoxon`).

**Formula (sign test, fallback)**:
```
diffs = [v - 0.5 per v in values], nonzero = diffs ≠ 0, n = len(nonzero)
k = numero di differenze positive
p = 2 * min( P(X≤k), P(X≥n-k) )     con X ~ Binomiale(n, 0.5)
```
**Limite dichiarato**: con n=5 seed per gruppo, il p-value minimo raggiungibile è 2×(1/32)=0.0625 —
il sign test non può MAI risultare significativo ad α=0.05 con questa numerosità, qualunque sia il
dato osservato. Non è un difetto dell'implementazione, va riportato come limite noto dei test non
parametrici a campioni piccoli se questi p-value vengono citati nel paper.

**RISULTATO REALE (2026-09-03, run dell'utente sulla macchina reale, scipy disponibile — Wilcoxon
vero, non il sign-test fallback)**:

| Gruppo | n | mean | 95% CI bootstrap | CI contiene 0.5? | p (Wilcoxon) |
|---|---|---|---|---|---|
| central, ε=0.1 | 10 | 0.5005 | [0.5000, 0.5011] | SÌ | 0.1934 |
| central, ε=1.0 | 10 | 0.5004 | [0.4999, 0.5010] | SÌ | 0.1602 |
| dp-fedavg, ε=0.1 | 5 | 0.4999 | [0.4992, 0.5007] | SÌ | 1.0000 |
| dp-fedavg, ε=0.5 | 5 | 0.4995 | [0.4987, 0.5002] | SÌ | 0.3125 |
| dp-fedavg, ε=1.0 | 5 | 0.5000 | [0.4996, 0.5003] | SÌ | 0.8125 |
| local, ε=0.1 | 5 | 0.4999 | [0.4992, 0.5007] | SÌ | 1.0000 |
| local, ε=1.0 | 5 | 0.5000 | [0.4996, 0.5003] | SÌ | 0.8125 |
| no-DP baseline | 10 | 0.5009 | [0.5001, 0.5019] | **NO** | 0.1309 |

**Nota di rigore — i due test DISCORDANO sul gruppo no-DP baseline**: il CI bootstrap esclude 0.5
(suggerendo un'elevazione statisticamente rilevabile), ma il test di Wilcoxon sugli stessi 10 valori
NON è significativo (p=0.13, ben sopra α=0.05). Nessuno degli 8 gruppi raggiunge p<0.05 al test di
Wilcoxon. **Lettura onesta per il paper**: l'eventuale elevazione del no-DP baseline sopra 0.5 è, se
reale, minuscola (mean 0.5009, ~0.1 punti percentuali) e i due test non concordano sulla sua
significatività — non va riportata come "leakage rilevata sotto no-DP" senza questa cautela esplicita.
Il disaccordo è plausibilmente un problema di potenza statistica (n=10, effetto piccolo), non un
errore di uno dei due metodi — entrambi vanno riportati insieme, non scegliendo quello che conferma
la narrativa desiderata. Nessun gruppo (con o senza DP) mostra un AUC significativamente sopra 0.5
al test più conservativo dei due (Wilcoxon) — coerente con la conclusione "Scenario B" (nessuna
leakage rilevabile) già documentata altrove per questa campagna.

**AGGIORNAMENTO 2026-09-04 (Sprint 10zz+42, task #67/#70) — il n=10 sopra era un artefatto di
conflazione, ora corretto**: un audit del codice ha trovato che `discover_groups()` raggruppava
silenziosamente `entity-split-sweep1` (split train/holdout entity-aware, task #10/#38 — un
esperimento di ROBUSTEZZA, non parte della campagna standard) insieme a `nodp-sweep1` sotto
"no-DP baseline", perché i due hanno config JSON identici (`no_dp=True`, nessun campo registra la
metodologia di split) — la stessa classe di conflazione silenziosa già corretta due volte in questo
script (mapping cartella→epsilon 2026-08-27; cartelle diagnostiche 2026-08-28), semplicemente non
ancora chiusa per questo caso specifico. `docs/DSN2027_Positioning.md` dichiara esplicitamente che
entity-split-sweep1 "non è direttamente comparabile alla campagna 5-seed×8-config" — il pooling era
quindi anche metodologicamente scorretto, non solo un incidente tecnico. **Fix**: `entity-split-*`
escluso di default da `discover_groups()` (nuovo parametro `include_methodology_variants`, stesso
pattern di `include_diagnostic`). Con n=5 corretto (solo `nodp-sweep1`), il bootstrap CI diventa
**[0.4997, 0.5009] — ORA CONTIENE 0.5**. **Questo risolve la nota di rigore sopra**: il disaccordo
bootstrap-CI-vs-Wilcoxon sul gruppo no-DP baseline non era (solo) un problema di potenza statistica
a n=10 — era in parte un artefatto della conflazione con un campione di robustezza
metodologicamente diverso. La nota di rigore sopra resta per lo storico (non cancellata, era una
lettura onesta dei dati come allora disponibili), ma **il numero da citare nel paper è ora questo,
non quello sopra**. 12 nuovi test (`tests/test_check_significance.py`) coprono `discover_groups()`
in modo diretto (lo script è importabile in questo sandbox, a differenza di `run_experiments.py` —
nessuna replica necessaria) — prima di questo fix, `check_significance.py` non aveva NESSUN test
nonostante calcoli i numeri citati nel paper.

**RISULTATO REALE (2026-09-04, rilanciato dall'utente sulla macchina reale, scipy disponibile —
Wilcoxon vero, con l'n=5 corretto)**: gruppo "no-DP baseline" → n=5, mean=0.5003, 95% CI bootstrap
[0.4997, 0.5009] (contiene 0.5, SÌ), **p=0.6250 [wilcoxon]** — il disaccordo tra i due test
descritto sopra è ora COMPLETAMENTE RISOLTO, non solo attenuato: entrambi i test concordano
(nessuna evidenza di scostamento da 0.5). Tabella completa aggiornata degli altri 7 gruppi
(invariati nell'n, solo il metodo del p-value confermato reale): central ε=0.1 n=10 p=0.1934;
central ε=1.0 n=10 p=0.1602; dp-fedavg ε=0.1/0.5/1.0 n=5 ciascuno, p=1.0000/0.3125/0.8125; local
ε=0.1/1.0 n=5 ciascuno, p=1.0000/0.8125. Nessun gruppo raggiunge p<0.05. (central ε=0.5 mostra n=1
— `central-sweep5`, task #52, campagna in corso.)

**AGGIORNAMENTO FINALE 2026-09-08 (Sprint 10zz+46/+47, task #52 completato + task #73) —
TABELLA DEFINITIVA, sostituisce tutte quelle sopra.** Al completamento della campagna task #52
(10 configurazioni × 5 seed, 6485 minuti), un audit ha trovato un quarto bug reale di conflazione
in `discover_groups()`: i rerun della campagna riusano gli stessi 5 seed (42/123/456/789/1234) già
presenti nei run precedenti (per aggiungere metriche mancanti, non nuova potenza statistica), ma
venivano contati come osservazioni aggiuntive invece che come sostituzioni — n gonfiato a 10
(dp-fedavg/local/no-DP, rerun bit-per-bit identici, determinismo verificato per seed) o a 15
(central, dove sweep1/2 precedono il fix di sensibilità DP pesata di task #26/#36 e NON vanno
mediati insieme ai rerun corretti sweep3/4/6/7). **Fix**: `discover_groups()` deduplica ora per
(dp_mode, epsilon, no_dp, seed), tenendo solo il file più recente per seed. Risultato — n=5 per
TUTTI i 10 gruppi, nessuna esclusione manuale necessaria:

| Gruppo | n | mean | std | 95% CI bootstrap | CI contiene 0.5? | p (Wilcoxon, reale) |
|---|---|---|---|---|---|---|
| central, ε=0.1 | 5 | 0.5005 | 0.0009 | [0.4998, 0.5013] | SÌ | 0.4375 |
| central, ε=0.5 | 5 | 0.4999 | 0.0009 | [0.4992, 0.5006] | SÌ | 0.8125 |
| central, ε=1.0 | 5 | 0.5005 | 0.0012 | [0.4996, 0.5014] | SÌ | 0.3125 |
| dp-fedavg, ε=0.1 | 5 | 0.4999 | 0.0009 | [0.4992, 0.5007] | SÌ | 1.0000 |
| dp-fedavg, ε=0.5 | 5 | 0.4995 | 0.0010 | [0.4987, 0.5002] | SÌ | 0.3125 |
| dp-fedavg, ε=1.0 | 5 | 0.5000 | 0.0004 | [0.4996, 0.5003] | SÌ | 0.8125 |
| local, ε=0.1 | 5 | 0.4999 | 0.0009 | [0.4992, 0.5007] | SÌ | 1.0000 |
| local, ε=0.5 | 5 | 0.4995 | 0.0010 | [0.4987, 0.5002] | SÌ | 0.3125 |
| local, ε=1.0 | 5 | 0.5000 | 0.0004 | [0.4996, 0.5003] | SÌ | 0.8125 |
| no-DP baseline | 5 | 0.5003 | 0.0008 | [0.4997, 0.5009] | SÌ | 0.6250 |

**RISULTATO REALE CONFERMATO (2026-09-08, macchina dell'utente, scipy disponibile — Wilcoxon
vero, non sign-test fallback).** Tabella sopra aggiornata con i p-value reali: **tutti e 10 i
gruppi hanno CI che contiene 0.5 E p-value Wilcoxon ben sopra α=0.05 (range 0.3125–1.0000,
nessuno vicino alla significatività)** — nessuna configurazione (con o senza DP) mostra un
AUC-LiRA significativamente diverso dal caso. Confermato coerente con "Scenario B" (nessuna
leakage rilevabile) già documentato altrove. Questa è ora la tabella DEFINITIVA, verificata con
il test statistico reale — non serve più rilanciare. 6 nuovi test di regressione in
`tests/test_check_significance.py` coprono la deduplicazione (suite completa: 211 passed,
1 skipped).

**Il bug di pseudo-replicazione non ha compromesso la conclusione del paper**: per
dp-fedavg/local/no-DP i rerun erano bit-identici (nessuna informazione persa/alterata dal bug,
solo n gonfiato); per central il bug mescolava dati pre-fix (task #26, sbagliati) con dati
post-fix (corretti) nella stessa media — un errore reale se fosse rimasto nel paper, ma il CI
finale per central resta comunque dentro 0.5 dopo la correzione. Nessun risultato già pubblicato
altrove in questo progetto dipendeva dai numeri "central" pre-fix.

**Nota esplicita (aggiunta 2026-09-09, per anticipare la domanda di un revisore che confronti le
righe della tabella sopra): perché le righe `dp-fedavg` e `local` sono IDENTICHE, valore per
valore, ad ogni ε?** Non è un errore né una tabella duplicata per sbaglio — è un risultato atteso
e già verificato indipendentemente su dati reali (confermato di nuovo il 2026-09-09 confrontando
`experiments/local-sweep2` e `experiments/dp-sweep6`, stesso seed=42/ε=0.1: `mean_auc_roc`
identico bit-per-bit, 0.49887861868608124 in entrambi). Il motivo è architetturale, non
statistico: nella simulazione single-process, sia `dp-fedavg` che `local` applicano clip+rumore
tramite la STESSA funzione, `GradientManager.privatize()` (vedi `src/ml/gradient_manager.py`) —
l'unica differenza dichiarata fra i due modi è SE questa chiamata avviene concettualmente "lato
client, prima dell'invio" (`local`) o "lato server, subito dopo la ricezione, prima di ogni altra
elaborazione" (`dp-fedavg`). In una simulazione senza un vero confine di processo/rete fra client
e server, queste due collocazioni collassano nella stessa operazione sugli stessi dati — da cui
l'identità numerica. **`central` non è affetto da questa identità** perché usa un meccanismo
strutturalmente diverso, `GradientManager.privatize_aggregate()` (un solo draw di rumore
sull'aggregato pesato, non un draw indipendente per client) — e infatti le sue righe nella tabella
sopra sono genuinamente diverse dalle altre due, non un artefatto della stessa collisione. Questa
identità dp-fedavg≡local NON si è riprodotta nel deployment NVFLARE reale multi-container (dove
client e server sono processi/container separati con RNG indipendenti) — coerente con la
spiegazione data, non una contraddizione.

---

## 9. Privacy Exposure Score (PES) v1

**Dove**: `docs/PrivacyExposureScore_v1.md` (definizione), non ancora una funzione dedicata nel
codice — calcolato a mano dai JSON di output.

**Fonte**: metrica originale del progetto (non da un paper esterno), ma la sua parte teoricamente
fondata (`L(AUC)`) è concettualmente parallela al **membership advantage** di Yeom et al. 2018 (§4),
e la sezione "grounding teorico" del documento cita anche Humphries et al. 2020, *"Differentially
Private Learning Does Not Bound Membership Inference"* (bound più stretto, che tiene conto di δ).

**Perché**: un singolo AUC non distingue tra "attacco fallito perché la difesa funziona" e "attacco
fallito perché ε era già largo e non prometteva molto." PES è pensato per rendere evidente proprio
il caso più dannoso: un ε nominale piccolo (privacy dichiarata forte) che non impedisce comunque un
AUC sopra 0.5.

**Formula**:
```
L(AUC)      = clip( 2 * max(0, AUC_LiRA - 0.5), 0, 1 )     # 0 = caso, 1 = attacco perfetto
strength(ε) = 1 / (1 + ε)         per una config DP
              0                   per la baseline no-DP (nessuna promessa di privacy da violare)
U_cost      = (mean_loss_DP - mean_loss_noDP) / mean_loss_noDP   # costo di utilità, riportato SEPARATO, non moltiplicato

PES_v1 = L(AUC) * strength(ε)
```

**Cosa è teoricamente fondato e cosa no (dichiarato esplicitamente nel documento, 2026-08-27)**:
`L(AUC)` gioca lo stesso ruolo concettuale del *membership advantage* di Yeom (`Adv = |Pr[member|
member] - Pr[member|non-member]|`, con bound `Adv ≤ e^ε - 1`) e del bound più stretto di Humphries
et al. 2020 (`Adv ≤ (e^ε - 1 + 2δ) / (e^ε + 1)`, più corretto per questo progetto perché il
meccanismo Gaussiano usato ovunque dà (ε,δ)-DP, non ε-DP puro) — ma **non è dimostrato identico**
ad `Adv` (AUC integra su ogni soglia, `Adv` è a soglia fissa). `strength(ε) = 1/(1+ε)` è invece
esplicitamente **designed, non derivato** da nessun bound formale — ha la forma qualitativa giusta
(1 a ε→0, decade verso 0 per ε grande) ma non è "il vantaggio empirico normalizzato contro il
soffitto teorico." Un successore più fondato, proposto ma non implementato (PES v1.1):
`PES_v1.1 = L(AUC) / ((e^ε - 1 + 2δ) / (e^ε + 1))`, che richiede prima TPR@low-FPR (§2) per
sostituire un `L(AUC)` a soglia singola ben definito. Task #41 (pending).

---

## 10. MIA Advantage (Yeom) — implementata (task #41, Sprint 10zz+13, 2026-09-02)

**Aggiornamento 2026-09-02**: implementata per davvero, questa sezione era rimasta indietro
rispetto al codice. **Dove**: `_mia_advantage()` in `scripts/run_experiments.py`, wired in
`run_lira()` su `lira_advantage`/`canary_advantage` (per-round) e `composed_lira_advantage`/
`canary_composed_advantage` (composto). Disponibile SOLO per run eseguiti dopo il 2026-09-02 —
richiede la curva ROC completa, mai salvata nei JSON storici; per la campagna 5-seed×8-config già
completata, vedi il calcolo retroattivo di PES v1/v1.1 in `scripts/compute_pes.py` (che usa
`mean_lira_auc_roc`, non l'Advantage).

**Fonte**: Yeom, Fredrikson, Jha, IEEE CSF 2018 (stesso paper di §4).

**Formula**:
```
Adv = max_t( TPR(t) - FPR(t) )    (statistica J di Youden, sulla curva ROC)
```
equivalente, sulla definizione originale di Yeom et al., a
`Adv = | Pr[attaccante dice "membro" | è membro] - Pr[attaccante dice "membro" | non è membro] |`
alla soglia ottimale — con bound formale per un meccanismo ε-DP: `Adv ≤ e^ε - 1` (Yeom), o più
stretto per (ε,δ)-DP Gaussiano: `Adv ≤ (e^ε - 1 + 2δ) / (e^ε + 1)` (Humphries et al. 2020).

---

## 10b. MIA Confusion Matrix alla soglia ottimale — implementata (task #49, Sprint 10zz+25, 2026-09-03)

**Stato**: implementata su richiesta esplicita dell'utente ("calcoliamo il numero di veri
positivi e falsi negativi dell'attacco?"). **Dove**: `_mia_confusion_at_best_threshold()` in
`scripts/run_experiments.py`, wired in `run_lira()` accanto a `_mia_advantage()` — stessi 4 punti
(main pool per-round/composto, canary pool per-round/composto): `lira_confusion`,
`canary_confusion`, `composed_lira_confusion`, `canary_composed_confusion`. Stessa limitazione di
§10: disponibile solo per run futuri, non retroattiva sui JSON storici.

**Fonte**: stessa base di §10 (Yeom et al. 2018) — la soglia usata è esattamente quella che
massimizza l'Advantage (Youden J), qui letta come conteggi assoluti invece che come tasso
aggregato.

**Perché**: AUC-ROC, TPR@FPR fisso (§2) e Advantage (§10) sono tutte metriche "a tasso" — nessuna
riporta quanti campioni sono stati classificati correttamente/erroneamente in cifra assoluta.
Particolarmente utile per i pool piccoli (canary: n_member=150, n_nonmember~19-20/round, §6),
dove un conteggio concreto ("rilevati X canary su 150, mancati Y") è più leggibile e citabile
dell'AUC o del tasso nel testo del paper.

**Formula**:
```
t* = argmax_t( TPR(t) - FPR(t) )                    (stessa soglia di Adv, §10)
predetto "membro" se score >= t*
TP = |{ campioni membro con score >= t* }|
FN = n_membri - TP
FP = |{ campioni non-membro con score >= t* }|
TN = n_non_membri - FP
```
Il dict restituito include anche `threshold` e `advantage` (ricalcolato, per verifica incrociata
con §10) oltre ai quattro conteggi e ai totali di classe `n_members`/`n_nonmembers`.

**Nota (aggiunta 2026-09-09): perché non riportiamo precision/recall/F1 come metriche principali.**
Il progetto NON calcola/riporta precision (`TP/(TP+FP)`), recall (`TP/(TP+FN)`, equivalente al TPR
a quella soglia) o F1 come metriche a sé — ma sono banalmente derivabili dai quattro conteggi
TP/FN/FP/TN sopra, se servissero per un confronto diretto con lavori che le usano. La scelta di non
enfatizzarle come principali è deliberata, non un'omissione: precision/recall richiedono di fissare
UNA soglia operativa (qui quella che massimizza Youden J), mentre la valutazione MIA di questo
progetto segue esplicitamente la critica di Carlini et al. 2022 (citata in §2/§3) secondo cui una
metrica a soglia singola nasconde il comportamento nella coda a basso-FPR, che è la zona
rilevante per un attaccante realistico — da cui la scelta di riportare AUC-ROC, TPR@FPR-fisso (§2)
e Advantage (§10) come indicatori primari, con i conteggi assoluti di questa sezione come
supplemento leggibile, non come sostituto.

---

## 10c. Controllo worst-case per singolo campione — implementata (task #50, Sprint 10zz+27, 2026-09-03)

**Stato**: implementata su richiesta esplicita dell'utente, in seguito alla discussione sulla
critica di Carlini et al. 2022 ("la privacy non è una metrica del caso medio" — un sistema di
sicurezza non è tale se resiste "in media", ma solo se resiste nel caso peggiore). §1-§10b sono
tutte metriche **aggregate a livello di popolazione**: anche il null result verificato in
`docs/DSN2027_Positioning.md` (TPR@1%FPR medio=0.0098 su 300 round reali) non esclude che un
piccolo sottoinsieme di record REALI resti sistematicamente vulnerabile, mascherato dalla media.
Questo controllo verifica quella possibilità specifica, a livello di singolo campione.

**Dove**:
- `run_lira()` in `scripts/run_experiments.py`, nuovo parametro opzionale `per_sample_dump_path`
  (default `None`, doppio opt-in come `composed_output`: richiede sia `composed_output is not
  None` sia `per_sample_dump_path` non `None` — nessun impatto sui run esistenti). Se attivo,
  scrive un JSON con lo score composto di OGNI campione scorato (non un aggregato): `session_id`
  reale (da `sessionID` in ACN-Data, vedi `src/adapters/acn_dataset.py`), `is_member`,
  `is_canary`, `composed_score`.
- CLI: flag `--per-sample-dump PATH` in `scripts/run_experiments.py`.
- `src/plugins/attacks/lira.py` (`LiRAAttack.run()`) inoltra `per_sample_dump_path` da kwargs —
  letto solo da LiRA, no-op per Yeom/Shadow.
- `scripts/analyze_worst_case_vulnerability.py` — script di analisi post-hoc, puro Python
  (nessuna dipendenza torch/sklearn), che confronta 2+ dump (stessa config, seed diversi) e marca
  i `session_id` "worst_case_vulnerable".

**Perché serve un identificatore stabile cross-seed**: `run_lira()` traccia i campioni con
`id(sample)` (identità oggetto Python), valido solo dentro un singolo processo — non riconosce lo
stesso record reale tra run/seed diversi. Da qui l'uso di `session_id` (stabile, esterno) invece
di `id(sample)`.

**Perché il confronto usa un percentile, non lo score grezzo**: ogni seed usa uno split
train/holdout indipendente, quindi lo stesso `session_id` può essere membro in un seed e
non-membro in un altro — score grezzi non sono confrontabili tra run. Lo script calcola invece,
per ogni apparizione-membro, il percentile dello score del campione rispetto a TUTTI gli score
dello stesso run (membri+non-membri — la stessa vista che avrebbe un vero attaccante, che non
conosce le etichette reali).

**Criterio "worst-case vulnerabile"** (default, configurabile via CLI): un `session_id` è marcato
tale se (1) è membro in almeno `--min-seeds` seed indipendenti (default 2 — un solo seed non è
evidenza di consistenza), (2) percentile medio >= `--threshold-percentile` (default 90), (3)
percentile MINIMO >= `--floor-percentile` (default 75) — il punto (3) è ciò che distingue
"vulnerabile in modo consistente" da "alto in media grazie a un singolo seed anomalo".

**Formula**:
```
percentile(score, tutti_gli_score_del_seed) = 100 * (n_below + 0.5*n_equal) / n
worst_case_vulnerable(session_id) sse:
    n_seed_come_membro >= min_seeds
    AND media(percentili) >= threshold_percentile
    AND min(percentili)  >= floor_percentile
```

**Limitazione**: come §10/§10b, non retroattiva — richiede run futuri lanciati con
`--per-sample-dump`; nessun dump storico esiste per le campagne già completate (vedi task #52 in
`README.md` per il piano di backfill).

**Test**: `tests/test_worst_case_vulnerability.py` (8 test, import diretto delle funzioni reali
dello script — puro Python, non serve una replica come per §10b).

---

## 10d. Estensione di TPR@low-FPR/Advantage/Confusion a Yeom, Shadow, canary raw-loss — implementata (task #53, Sprint 10zz+28, 2026-09-03)

**Stato**: implementata su richiesta esplicita dell'utente, in seguito a un'altra citazione di
Carlini et al. 2022 discussa in chat — la "fallacia delle medie" nel confronto TRA attacchi (un
attacco chirurgico su un piccolo sottogruppo e un attacco uniformemente mediocre possono avere lo
stesso AUC-ROC aggregato). §2/§10/§10b erano cablate SOLO su LiRA — se il paper confrontasse Yeom
vs Shadow vs LiRA usando solo `auc_roc`, cadrebbe nella stessa fallacia, solo applicata al
confronto tra attacchi invece che tra soglie dentro un attacco.

**Dove**: stesso pattern additivo di §2/§10/§10b, sulle stesse coppie label/score già usate per
l'AUC di ciascun attacco — nessun costo computazionale aggiuntivo:
- `run_fedmia()` (Yeom, §4): aggiunge `tpr_at_fpr_*`, `advantage`, `confusion` al dict per-round.
- `run_fedmia_shadow()` (Shadow, §5): aggiunge `tpr_at_fpr_*`, `shadow_advantage`,
  `shadow_confusion`.
- `run_lira()`: aggiunge `canary_raw_advantage`/`canary_raw_confusion` sul diagnostico raw-loss
  del canary (§6) — LiRA calibrato aveva già Advantage/Confusion dal task #41/#49, mancava solo
  la versione raw-loss.

**Limitazione**: come §10/§10b/§10c, non retroattiva — disponibile solo per run futuri. py_compile
OK, 136/136 test non-torch invariati (nessun nuovo test dedicato: wiring puro di funzioni già
testate — `_mia_advantage`/`_mia_confusion_at_best_threshold`/`_tpr_at_fixed_fpr` — su nuovi punti
di chiamata, nessuna nuova formula/logica di soglia da verificare in isolamento, stesso principio
già applicato in Sprint 10zz+17/10zz+21).

**BUG scoperto e corretto (2026-09-03, task #59, Sprint 10zz+34)**, durante la verifica richiesta
esplicitamente dall'utente ("verifichiamo questa risposta per tutti e 4 gli attacchi implementati
MIA ed il FedMIA"): `_tpr_at_fixed_fpr()` restituisce sempre le stesse chiavi generiche
(`tpr_at_fpr_0.001` ecc.), identiche a quelle già usate da `run_lira()` (bare, da prima di questo
task, §2). Quando `run_registered_attacks()` fonde i risultati di yeom→shadow→lira nello STESSO
dict per round con `.update()` (ultimo scrittore vince), il TPR@low-FPR calcolato qui da Yeom e da
Shadow veniva **silenziosamente sovrascritto** da quello di LiRA prima di essere salvato nel JSON —
il campo `tpr_at_fpr_*` nei run reali era sempre quello di LiRA, mai quello di Yeom o Shadow,
nonostante entrambi lo calcolassero correttamente qui. `advantage`/`confusion` non erano affetti
(Yeom li salva bare per convenzione — nessuna collisione — Shadow/LiRA/canary li salvano già
prefissati). **Fix**: chiavi ora prefissate `yeom_tpr_at_fpr_*` (Yeom) e `shadow_tpr_at_fpr_*`
(Shadow); LiRA resta bare (`tpr_at_fpr_*`, invariato, è l'uso storico più citato in questo
progetto). **Impatto sui dati già pubblicati**: nessuno per gli AUC-ROC (indipendenti da questo
bug); il campo `tpr_at_fpr_*` in run con tutti e 3 gli attacchi eseguiti insieme (praticamente ogni
sweep del progetto — Yeom/Shadow/LiRA sono nel registro di default) va reinterpretato come
"TPR@low-FPR di LiRA", non di Yeom/Shadow, per ogni run esistente — nessun claim pubblicato finora
citava un TPR@low-FPR di Yeom o Shadow specificamente (solo di LiRA), quindi non risulta materiale
per conclusioni già scritte, ma va tenuto a mente leggendo JSON storici. 165/165 test non-torch
invariati (nessuna nuova formula, solo un prefisso sulle chiavi di un dict già testato).

---

## 10e. Curve ROC complete + plot log-log — implementata (task #54, Sprint 10zz+29, 2026-09-03)

**Stato**: implementata su richiesta esplicita dell'utente, dopo due citazioni di Carlini et al.
2022 discusse in chat: (1) "la bontà di un attacco di privacy si misura unicamente osservando cosa
accade quando il FPR è prossimo a zero" — un singolo numero (AUC, TPR@fixed-FPR, Advantage) mostra
solo un aggregato o un punto discreto di quella regione, mai il comportamento COMPLETO; (2) "il
paper introduce l'uso di scale logaritmiche sugli assi, permettendo di zoomare esattamente sulla
regione critica" — da cui il plot log-log, non solo il salvataggio dei dati.

**Dove**:
- `_full_roc_curve(labels, scores)` in `scripts/run_experiments.py` — fpr/tpr completi (via
  `sklearn.roc_curve`), stesso pattern try/except di `_mia_advantage()`. Nessuna nuova formula:
  stesse coppie label/score già usate per AUC/TPR@low-FPR/Advantage/Confusion.
- Nuovo parametro opt-in `roc_curve_dump_path` su `run_fedmia()`/`run_fedmia_shadow()`/`run_lira()`
  — se impostato, scrive un JSON con le curve per-round (+ `composed`/`canary`/`canary_raw` per
  LiRA, vedi §3/§6). Default `None`, zero impatto se omesso.
- CLI: `--roc-curve-dump-dir DIR` — un file per attacco eseguito (`roc_curves_yeom.json`/
  `_shadow.json`/`_lira.json`), thread in `main()` → `run_registered_attacks()` → i 3 wrapper
  (`yeom.py`/`shadow.py`/`lira.py`), che costruiscono il proprio path dentro `DIR`.
- Nuovo `scripts/plot_roc_log_scale.py` — carica 1+ dump, plotta in scala log-log (assi FPR e TPR
  entrambi log, floor 1e-4 per evitare log(0), diagonale di riferimento caso-random), salva PNG.
  matplotlib come dipendenza opzionale (`pyproject.toml`, extra `viz`) — import lazy, lo script
  resta importabile/testabile senza matplotlib installato (solo la generazione del plot lo
  richiede).

**Limitazione**: come §10/§10b/§10c/§10d, non retroattiva — disponibile solo per run futuri.
py_compile OK, 148/148 test non-torch (136→148: 12 nuovi in
`tests/test_plot_roc_log_scale.py`, sulla logica pura di caricamento/selezione/clip — non su
`_full_roc_curve()` stesso, che chiama `sklearn.roc_curve` come le altre funzioni di questa
famiglia, sklearn assente in questo sandbox). Smoke test sintetico eseguito end-to-end (dati
generati, non reali): il plot si genera correttamente sia per l'ultimo round sia per il composto,
con e senza subkey canary.

**Vedi anche**: §3 (LiRA) per il fondamento teorico Neyman-Pearson → test intrattabile → riduzione
a statistica 1D → approssimazione Gaussiana, che giustifica perché questa curva vada letta a
FPR basso, non mediata.

---

## 10f. Attacco di Sablayrolles et al. 2019 — soglia non-parametrica per-esempio — implementata (task #58, Sprint 10zz+33, 2026-09-03)

**Stato**: implementata su richiesta esplicita dell'utente, dopo una citazione di Carlini et al.
2022 (§V-C, Table I) discussa in chat: **"sorprendentemente, scopriamo che, nonostante sia stato
pubblicato nel 2019, l'attacco di Sablayrolles et al. [56] batte gli altri attacchi sotto la
nostra metrica"** (spesso di un ordine di grandezza), pur essendo "the most direct influence for
LiRA" secondo gli stessi autori — un precursore diretto e più semplice, non un attacco scorrelato.

**Formula (paper originale)**: `A'(x,y) = ℓ(f(x),y) - τ_{x,y}`, con soglia per-esempio
`τ_{x,y} = (μ_in(x,y) + μ_out(x,y))/2` stimata via shadow model — a differenza di LiRA, **non
fitta una Gaussiana**: usa solo le medie, non le varianze, e la soglia è il punto medio geometrico
tra le due, non un test di verosimiglianza.

**Perché è quasi gratis da aggiungere qui**: la nostra `run_lira()` è già la variante *online* di
Carlini — calcola per OGNI campione sia μ_in che μ_out reali (vedi `_diag_mu_in_values`/
`_diag_mu_out_values` in §3), esattamente gli input che Sablayrolles richiede. Implementarlo non
ha richiesto shadow model aggiuntivi né training aggiuntivo: `_sablayrolles_score(mu_in, mu_out,
target_loss)` in `scripts/run_experiments.py` è una seconda formula calcolata nello stesso ciclo
per-campione di `lira_score`, sugli stessi tre numeri già in memoria.

**Convenzione di segno**: qui restituiamo `τ_{x,y} - target_loss` (il NEGATIVO della formula
letterale del paper), per allinearci alla convenzione già usata in questo file — punteggio più
alto = più probabile membro, coerente con `lira_score` e con l'attacco LOSS di Yeom
(`-ℓ(x,y) > τ`).

**Dove**:
- `_sablayrolles_score()` — nuova funzione pura (nessuna dipendenza sklearn/numpy), estratta per
  testabilità (stesso motivo di `_mia_advantage()`/`_full_roc_curve()`).
- `run_lira()` — calcola `sablayrolles_score` per ogni campione già scorato da LiRA (stesso filtro
  8σ, stesso pool member/nonmember), sempre attivo (nessun parametro opt-in: costo aggiuntivo
  trascurabile, coerente con Advantage/Confusion/TPR@low-FPR, non con i dump diagnostici opt-in).
- Risultati per round: `sablayrolles_auc_roc`, `sablayrolles_advantage`, `sablayrolles_confusion`,
  `sablayrolles_n_member`/`_n_nonmember`, `sablayrolles_tpr_at_fpr_{0.001,0.01,0.05}` — stesse
  funzioni (`_mia_advantage()`/`_mia_confusion_at_best_threshold()`/`_tpr_at_fixed_fpr()`) già
  usate per LiRA/Yeom/Shadow/canary (§10d), solo su un pool di punteggi diverso.
- Curva ROC completa: se `roc_curve_dump_path` è impostato (§10e), il dump per-round include anche
  `"sablayrolles": {"fpr": [...], "tpr": [...]}` accanto a `"lira"`/`"canary"`/`"canary_raw"` —
  `scripts/plot_roc_log_scale.py --subkey sablayrolles` la plotta senza modifiche allo script.
- **Non estesa al "composed" multi-round** (§3, prototipo `_cumulative_scores`) — resterebbe un
  secondo accumulatore cumulativo parallelo, non richiesto e non essenziale per rispondere alla
  domanda "quanto guadagna il fit Gaussiano sopra la soglia non-parametrica", che è già visibile
  round per round.

**Come leggerlo per il paper**: confrontare `sablayrolles_auc_roc`/`_tpr_at_fpr_0.001` con
`lira_auc_roc`/`tpr_at_fpr_0.001` sullo stesso round è esattamente l'ablation che Carlini stesso
riporta in Table II ("+ Per-example thresholds [56]" → "+ Gaussian Likelihood") — se LiRA batte
Sablayrolles su dati EV reali (non solo su CIFAR-10 come nel paper originale), è un risultato
diretto da citare: quantifica il contributo specifico del fit Gaussiano+logit-scaling (o, nel
nostro caso, del fit Gaussiano sulla MSE grezza — vedi §3 "Modellazione parametrica vs non
parametrica") sopra la baseline non-parametrica più semplice, su un dominio diverso da quello del
paper originale.

**Limitazione**: come §10/§10b/§10c/§10d/§10e, non retroattiva — disponibile solo per run futuri
(il relaunch task #52 lo raccoglierà automaticamente, nessun flag aggiuntivo da passare). py_compile
OK, 165/165 test non-torch (158→165: 7 nuovi in `tests/test_sablayrolles_score.py`, sulla formula
pura — stesso limite/pattern di `tests/test_mia_advantage.py`, sklearn/torch assenti in questo
sandbox). Non ancora eseguito con dati reali.

**Vedi anche**: §3 (LiRA) per il fondamento Neyman-Pearson e la scelta parametrica vs non
parametrica — questa sezione ne è il confronto empirico diretto, non solo teorico.

---

## 11. Krum score (Byzantine detection)

**Dove**: `KrumDetector.compute_scores()` / `detect_byzantine()`, `src/ids/charging_ids.py`
(righe 349-474).

**Fonte**: Blanchard, El Mhamdi, Guerraoui, Stainer, *"Machine Learning with Adversaries:
Byzantine Tolerant Gradient Descent,"* NeurIPS 2017 — citato esplicitamente nel docstring della
classe.

**Perché**: rileva nodi con gradienti "isolati" (molto distanti da tutti gli altri) — segnale di
model poisoning o fault Byzantine in un aggregatore FL. Fa parte della difesa IDS (non
dell'attacco MIA) del progetto.

**Formula**:
```
per ogni coppia di nodi (i,j): dist(i,j) = || g_i - g_j ||²    (distanza euclidea al quadrato, sui gradienti appiattiti)
krum_score(i) = somma delle (n - f - 2) distanze più piccole di i verso gli altri nodi
score_normalizzato(i) = krum_score(i) / media(krum_score su tutti i nodi)
```
dove n = numero di nodi, f = tolleranza Byzantine dichiarata (default 1). Richiede n ≥ 2f+3 per la
garanzia teorica (Blanchard et al.); sotto questa soglia (tipico nei piccoli cluster EV del
progetto) la detection è disabilitata esplicitamente, non applicata con garanzie false. Soglia di
allerta calibrata empiricamente a 3.5 (non il default 0.8 originario — vedi commento in codice,
fix 2026-07-22) per un training a 50 epoche, dove la distribuzione naturale dei Krum score è più
alta e 0.8 produceva falsi positivi sistematici.

---

## 12. Cosine similarity (Byzantine detection, complementare a Krum)

**Dove**: `GradientAnalyzer.cosine_similarity()` / `cluster_cosine_analysis()`,
`src/ids/charging_ids.py` (righe 272-344).

**Fonte**: misura geometrica standard, nessuna citazione specifica a un paper — usata qui come
euristica complementare a Krum.

**Perché**: un nodo con similarità coseno media bassa (<0.5) verso tutti gli altri è
geometricamente isolato dal cluster — stesso tipo di segnale di Krum ma calcolato diversamente
(direzione del gradiente, non distanza euclidea), utile come conferma incrociata.

**Formula**:
```
cos_sim(a, b) = (a · b) / (||a|| * ||b||)      in [-1.0, 1.0]
avg_cos_sim(nodo_i) = media di cos_sim(nodo_i, nodo_j) su ogni altro nodo j
```
1.0 = gradienti identici, 0.0 = ortogonali, -1.0 = opposti (possibile poisoning). Con un solo nodo
nel cluster (similarità non calcolabile), restituisce `NaN` esplicitamente invece di essere
classificato erroneamente come "bassa similarità."

---

## 13. CUSUM (drift detection temporale)

**Dove**: `CUSUMDetector`, `src/ids/charging_ids.py` (righe ~100-160 dell'estratto letto).

**Fonte**: algoritmo standard di controllo statistico di processo (cumulative sum control chart),
nessuna citazione paper-specifica nel codice — tecnica generica, non originata nella letteratura
MIA/FL.

**Perché**: rileva deviazioni progressive (drift) nel comportamento di un nodo nel tempo — a
differenza di Krum/cosine similarity (istantanei, un solo round), CUSUM accumula evidenza
round-su-round, utile per attacchi Byzantine "lenti" che un singolo round non renderebbe visibili.

**Formula (two-sided CUSUM, con warm-up di 10 osservazioni)**:
```
S+_t = max(0, S+_{t-1} + (x_t - μ) - drift)
S-_t = max(0, S-_{t-1} + (μ - x_t) - drift)
allerta se S+_t > soglia  oppure  S-_t > soglia
```
dove μ è la media mobile del nodo (aggiornata solo durante il warm-up, poi fissata), `drift` è la
tolleranza minima di deviazione prima che conti (default 0.5), soglia di allerta default 5.0.

---

## 14. DP: σ del meccanismo Gaussiano e sensibilità pesata (Central DP)

**Dove**: `GradientManager._compute_sigma()` / `privatize_aggregate()`, `src/ml/gradient_manager.py`
(righe 44-80, 310-397).

**Fonte**: calibrazione analitica standard del meccanismo Gaussiano (Dwork & Roth, *"The
Algorithmic Foundations of Differential Privacy,"* 2014, Teorema 3.22) per la formula di σ; il
placement "client-side clip, noise-once server-side sull'aggregato" (mode `central`) è verificato
corrispondere esattamente a McMahan, Ramage, Talwar, Zhang, *"Learning Differentially Private
Recurrent Language Models,"* ICLR 2018 (arXiv:1710.06963) — letto full-text 2026-08-14, Algorithm 1:
`UserUpdateFedAvg` clippa lato client, il rumore Gaussiano `N(0, Iσ²)` è aggiunto una sola volta,
lato server, sull'aggregato dopo la media FedAvg. Nota importante già corretta nel README (Sprint
10ccc): questa citazione si attacca al mode `central`, non a `dp-fedavg` (che rumorizza per-client
prima dell'aggregazione — una variante più severa, non descritta letteralmente in McMahan et al.).

**Perché**: fornisce la garanzia formale (ε,δ)-DP sul peso aggregato che esce dal nodo verso
l'aggregatore — la difesa contro cui LiRA/Yeom/Shadow vengono testati.

**Formula (σ base, weight perturbation a singolo round)**:
```
σ = max_grad_norm * sqrt(2 * ln(1.25/δ)) / ε
```

**Formula (Central DP, sensibilità pesata — fix Sprint 10bbb, verificato attivo nel codice reale,
non solo dichiarato)**:
```
max_weight_fraction = max(n_i) / Σ(n_i)     se participant_n_samples è disponibile
                     = 1 / n_participants    fallback, con warning esplicito se non disponibile
σ_central = σ * max_weight_fraction
```
Fix rispetto alla versione precedente (`σ/n_participants`, che assumeva pesi uniformi non veri per
FedAvg pesato per campioni — con Office1 un ordine di grandezza più piccolo di Caltech/JPL). **Impatto
sui risultati già pubblicati**: cambia solo la magnitudo del rumore applicato (mai la misura di
leakage — LiRA/Yeom/Shadow AUC sono indipendenti da questa formula), quindi ogni AUC Central DP già
pubblicato resta valido come misura di leakage, ma l'ε nominale dichiarato per quei run specifici
era leggermente ottimistico rispetto al rumore realmente applicato (task #36, rilancio Central DP
ancora pending per numeri pienamente coerenti con l'ε dichiarato).

**Limite dichiarato esplicitamente nel codice**: questo è weight perturbation (rumore sui pesi
aggregati post-training), non DP-SGD (rumore per-campione durante il training) — la garanzia
formale del meccanismo Gaussiano vale esattamente solo per `epochs=1`, non per `epochs=50` usato
nella campagna principale; e σ è calcolato per un singolo round, la composizione su T round degrada
la garanzia e richiederebbe RDP/zCDP per un'analisi formale — non ancora fatta.

**Quanto è grande questo gap, in concreto (verificato 2026-09-14).** Jayaraman & Evans (USENIX
Security '19) — verificato via WebSearch + fetch diretto del testo del paper e del post del blog
dell'autore stesso, non accettato da un riassunto di seconda mano — misurano su un classificatore
a rete neurale a due strati che **RDP eguaglia l'utilità della naive composition con un budget
~50× più stretto** (53% di perdita di accuratezza a ε=10 sotto RDP contro ε=500 per la stessa
perdita sotto naive composition), e che anche il modello allenato con RDP resta attaccabile (0.399
di membership advantage a ε=1000 per un attacco white-box). Due claim associati a questo paper in
un feedback esterno non si sono verificati e non sono citati altrove in questo progetto: "RDP
mantiene perdita di accuratezza quasi zero" (falso — 53% a ε=10) e un 82% di PPV attribuito a "un
attaccante che osserva più round/modelli paralleli" (il vero esperimento riallena lo STESSO modello
5 volte indipendenti, non osserva più round di uno stesso training — framing diverso, non
direttamente applicabile alla nostra LiRA composta multi-round senza un argomento separato che non
abbiamo fatto). Il rapporto 50× è ora citato nel paper DSN 2027 (§2, §9) come motivazione concreta
per cui il nostro ε multi-round riportato va letto come limite superiore conservativo, non garanzia
stretta. **Rimedi possibili, dal più economico**: (A) composizione avanzata in forma chiusa (Dwork
& Roth 2014, Teorema 3.20); (B) accounting RDP in forma chiusa specifico per il meccanismo
Gaussiano — più stretto di (A), ancora nessuna libreria esterna; (C) libreria esterna completa di
DP-accounting (Opacus / TensorFlow Privacy / `dp-accounting`) con vero DP-SGD per-campione — il più
rigoroso, il più costoso, lavoro futuro vista la deadline abstract del 25/11.

**(A) implementata il 2026-09-14 (task #118, Sprint 10zz+83) — risultato nullo onesto, non un
fix.** `_advanced_composition_epsilon()` in `scripts/run_experiments.py` calcola il limite di
Dwork & Roth (ε' = ε·√(2k·ln(1/δ')) + k·ε·(e^ε−1), k=`fl_rounds`) e lo scrive in ogni nuovo
risultato come `epsilon_cumulative_advanced`/`epsilon_cumulative_best_known`;
`scripts/compute_advanced_composition.py` lo ricalcola retroattivamente su ogni JSON già
completato (nessun run necessario, basta epsilon/delta/fl_rounds già loggati);
`tests/test_advanced_composition.py` verifica la formula contro valori noti. Risultato: a
`fl_rounds`=10 (usato in tutto questo progetto), la composizione avanzata **non è mai più stretta**
della naive per nessuno dei tre epsilon usati (1.0/0.5/0.1) — confermato su tutti e 45 gli
esperimenti DP della campagna principale (0/45 casi in cui vince) e spiegato analiticamente: il
round di crossover (dove advanced comincia a battere naive) è k=29 a ε=0.1, k=187 a ε=0.5, e **mai
raggiungibile** a ε=1.0, perché ε≥ln(2)≈0.693 fa crescere il termine dominante di advanced almeno
quanto quello di naive per ogni k. (A) è implementata ma non cambia quale limite questo progetto
riporta (`epsilon_cumulative_naive` resta il più stretto per ogni risultato); (B) resta l'opzione
non implementata più promettente, perché non richiede lo stesso numero enorme di round.

---

## Sommario per lavoro citato

| Lavoro | Cosa cita da noi | Stato verifica |
|---|---|---|
| Shokri et al. 2017 (IEEE S&P) | Shadow MIA (§5), AUC-ROC generico (§1) | abstract-confirmed |
| Yeom et al. 2018 (IEEE CSF) | Yeom attack (§4), MIA Advantage (§10), MIA Confusion Matrix (§10b), grounding di PES (§9) | full-text non ancora letto, citazione usata da tempo nel progetto |
| Carlini et al. 2022 (IEEE S&P, arXiv:2112.03570) | LiRA (§3), TPR@low-FPR (§2), curve ROC log-log (§10e) | full-text letto (§I-VI, Table I/II incluse) |
| Sablayrolles et al. 2019 (ICML, "White-box vs Black-box: Bayes Optimal Strategies for Membership Inference") | soglia non-parametrica per-esempio (§10f) | **NON letto direttamente** — implementata dalla formula/descrizione riportata in Carlini et al. 2022 §V-C (citazione di seconda mano, esplicitamente dichiarata: prima della submission andrebbe letto il paper originale per verificare che la formula riportata da Carlini sia fedele, e per citarlo correttamente in bibliografia) |
| McMahan et al. 2018 (ICLR, arXiv:1710.06963) | Central DP placement (§14) | full-text letto |
| Blanchard et al. 2017 (NeurIPS) | Krum (§11) | citato nel codice, non ri-verificato in questo giro |
| Humphries et al. 2020 | grounding più stretto di PES/MIA Advantage (§9, §10) | citato in `PrivacyExposureScore_v1.md`, non ri-verificato in questo giro |
| Dwork & Roth 2014 | formula σ Gaussian Mechanism (§14) | formula standard, citazione da confermare nel testo del paper |
| Nasr et al. 2019 (IEEE S&P) | FedMIA (§ non incluso sopra — attacco NON attivo, vedi nota) | citato nel docstring di `fedmia.py`, non citabile come attacco eseguito |
| Carlini 2019 / Jagielski et al. 2020 | canary insertion (§6) | non ancora verificate in questo giro — da confermare prima della submission |

**Nota su FedMIA/Nasr et al.**: `src/plugins/attacks/fedmia.py` (classe `FedMIA`, cita Shokri 2017 +
Nasr et al. 2019 "Comprehensive Privacy Analysis of Deep Learning") non è nel `ATTACK_REGISTRY` e
non è mai passata a `ByzantineDetector` in nessun run che ha prodotto un risultato pubblicato — non va
citata come attacco eseguito. La stessa classe è riusata (metodi `calibrate_from_vectors()`/
`reconstruction_error()`) da `run_fedmia_gradient()` (Sprint 10zz), un attacco diverso, anch'esso
opt-in e non ancora validato con un run reale corretto (vedi README Sprint 10zz+9/10zz+10) — non
citabile nel paper allo stato attuale.

---

## Voci ancora da completare

- Verifica full-text di Yeom et al. 2018 (attualmente citato da tempo nel progetto ma senza il
  passaggio "letto full-text" registrato in `docs/ReadingList_DSN2027.md`).
- Verifica di Blanchard et al. 2017 (Krum) e Humphries et al. 2020 in questo stesso giro di ricerca
  citazioni (non ancora rifatta, solo riportata da versioni precedenti di `PrivacyExposureScore_v1.md`).
- Verifica di Carlini 2019 ("The Secret Sharer") e Jagielski et al. 2020 (canary insertion, §6) —
  citate nel README come riferimento standard ma non ancora passate per una verifica dedicata come
  fatto per i 3 paper containerized-FL (Sprint precedente).
