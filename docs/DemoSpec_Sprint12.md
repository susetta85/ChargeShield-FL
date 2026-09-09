# Demo Spec — Sprint 12 (ridimensionata): ChargeShield-FL Results Dashboard

**Stato:** design-only, nessun codice ancora scritto. Documento di specifica, non implementazione.
**Data:** 2026-08-28.
**Decisione di scoping (2026-08-27/28, confermata dall'utente):** demo ridimensionata da "visualizzazione
real-time del training FL" a **dashboard post-hoc su risultati già completati**, e ri-sequenziata per
**non bloccare/precedere** la stesura del paper — l'Artifact Evaluation di DSN 2027 è un track separato e
disaccoppiato ("the artifact does not need to be submitted at the same time of the paper", CFP DSN2027
Berlin), quindi non c'è più una dipendenza temporale stretta tra demo e paper. Le due attività (scheletro
paper, questa spec) sono state condotte in parallelo su esplicita richiesta dell'utente ("Entrambi in
parallelo").

---

## 1. Perché ridimensionare (motivazione, non solo decisione)

Il piano originale di Sprint 12 prevedeva una visualizzazione **real-time** del training FL via Streamlit:
curva AUC per round mentre il training è in corso, timeline degli alert IDS live, slider DP noise/utility
interattivo durante l'esecuzione. Questo avrebbe richiesto:

- Instrumentare `scripts/run_experiments.py` e/o il job NVFLARE con un canale di streaming (websocket, file
  polling, o simile) verso l'app Streamlit — lavoro di integrazione non banale, mai iniziato.
- Mantenere sincronizzati due percorsi di esecuzione (simulazione locale + deployment reale
  NVFLARE/Containerlab) con lo stesso meccanismo di streaming — rischio di introdurre una terza fonte di
  bug in un momento in cui il focus del progetto è la validazione statistica dei risultati esistenti.
- Essere pronta *prima* della scrittura del paper, sotto l'assunto (rivelatosi errato dopo verifica del CFP
  reale) che l'artifact demo dovesse accompagnare la submission.

Nessuna di queste tre condizioni regge più: (1) zero bozza di paper esisteva quando il piano originale fu
scritto — ora esiste uno scheletro completo (`docs/paper/ChargeShield-FL_DSN2027_paper_skeleton.docx`); (2)
la campagna 5-seed×8-config è l'unico lavoro realmente bloccante rimasto; (3) l'AE è disaccoppiata dalla
deadline del paper. La versione **post-hoc** qui specificata copre lo stesso obiettivo di fondo — un
artifact dimostrativo, ispezionabile, per la sottomissione AE — con una superficie di rischio molto minore:
legge JSON/Excel già scritti su disco, nessuna integrazione col training stesso.

---

## 2. Obiettivo e pubblico

**Obiettivo:** una dashboard interattiva che permetta a un revisore AE (o all'utente stesso) di esplorare i
risultati sperimentali già raccolti — senza dover aprire manualmente decine di file JSON o navigare fogli
Excel — e di verificare visivamente le affermazioni chiave del paper (null result su LiRA, triangolazione a
5 assi del sanity-check, confronto fra placement DP).

**Pubblico:** (a) revisori dell'Artifact Evaluation committee DSN 2027; (b) l'utente stesso, per ispezione
rapida durante lo sviluppo; (c) eventuali lettori del paper che vogliono verificare un claim specifico contro
il dato grezzo.

**Non è un obiettivo:** sostituire `scripts/generate_excel_report.py` (che resta la fonte primaria per le
tabelle del paper) o fornire training/inference live.

---

## 3. Framework: Streamlit (confermato)

Riconfermato rispetto alla valutazione già fatta in sede di discussione (non ripetuta qui in dettaglio):
Streamlit resta la scelta giusta perché (a) puro Python, si integra senza frizione con lo stack già in uso
nel progetto (pandas, openpyxl, numpy — nessuna nuova dipendenza di linguaggio); (b) `st.cache_data` copre
esattamente il caso d'uso "rileggi JSON/Excel statici, non rieseguire ogni volta"; (c) `st.plotly_chart`/
`st.altair_chart` sufficienti per tutti i grafici previsti sotto, senza bisogno di scrivere JS. Alternative
scartate: Gradio (pensato per demo di modelli inference-in/inference-out, non per esplorazione di dataset
tabellari multi-vista); Dash (più potente ma più verboso per un caso d'uso a sola lettura); notebook Jupyter
statico (non interattivo/filtrabile, meno adatto a un artifact "da cliccare" per un revisore AE).

---

## 4. Fonti dati (reali, non ipotetiche)

Tutte le fonti sono già presenti nel repository, generate da script esistenti — la dashboard è un consumatore
di sola lettura, non introduce un nuovo formato dati.

| Fonte | Formato | Prodotta da | Contenuto |
|---|---|---|---|
| `experiments/*/experiment_*.json` | JSON, un file per (config, seed) | `scripts/run_experiments.py` | `config` (dp_mode, epsilon, seed, hidden_dims, feature_names, ecc.), `summary` (mean/max/min AUC per Yeom/Shadow/LiRA), `per_round` (serie temporale per round) |
| `experiments/ChargeShield_FL_Results.xlsx` | Excel, 10 fogli | `scripts/generate_excel_report.py` | Raw Data, Heat Map, Per Rounds, Per Epsilon, Comparison, AUC Progression, Attack Comparison, Yeom/Shadow/LiRA Per Round |
| `experiments/dp-sweep*/`, `nodp-sweep1/`, ecc. | sottocartelle multi-seed | `run_experiments.py` (invocato da `scripts/run_multiseed_consolidation.sh`) | i veri dati N=5 seed per gruppo, con report `.xlsx` locale già aggregato per sweep |
| Output di `scripts/check_significance.py` | testo su stdout (da reindirizzare a file, es. `experiments/significance_report.txt`) | script esistente, esteso in Sprint 10pp | bootstrap CI, Wilcoxon/sign-test p-value per gruppo (dp_mode, epsilon) |
| `experiments/_calibration_*/` (5 cartelle) | stessa struttura JSON | run del sanity-check a 5 assi | i risultati della Sezione 6 del paper (capacità, feature, sito, combinato) |

Nota di provenienza (coerente con la disciplina già in uso nel progetto): la dashboard deve leggere
`dp_mode`/`epsilon`/`seed`/etc. **dal campo `config` di ogni JSON**, mai dal nome della cartella — esattamente
lo stesso principio che ha già causato due bug reali corretti in questa sessione
(`check_significance.py`, mapping cartella→epsilon; e la conflazione delle cartelle `_calibration_*`/`_diag_*`
nel gruppo "no-DP baseline"). La dashboard deve escludere di default le cartelle con prefisso `_`
(diagnostiche/calibrazione) dalla vista "Results" principale, e mostrarle solo nella vista dedicata
"Sanity-Check" (Tab 4 sotto).

---

## 5. Struttura proposta (tab/pagine)

### Tab 1 — Overview
- Riepilogo a colpo d'occhio: numero totale di esperimenti, range di epsilon testati, DP mode disponibili,
  data dell'ultimo run.
- Headline claim del paper in un box ben visibile: "AUC LiRA composito, tutte le configurazioni DP: 0.48–0.52
  — nessun segnale di membership rilevabile" con link diretto alla Tab 2 per i numeri esatti.

### Tab 2 — Results Explorer (il cuore della dashboard)
- Tabella filtrabile (dp_mode, epsilon, numero round) sui dati aggregati per seed, letta dai gruppi già
  scoperti da `check_significance.discover_groups()` (riuso diretto della logica esistente, non
  reimplementata).
- Per ogni gruppo selezionato: mean ± std AUC (Yeom/Shadow/LiRA), bootstrap 95% CI, p-value (Wilcoxon o
  sign-test con etichetta esplicita del metodo — mai nascosto, come richiesto nel codice sorgente).
  Il campo TPR@1%/0.1%/5%FPR va mostrato **solo per i run successivi a Sprint 10pp** (i run precedenti non
  hanno questo campo nel JSON) — la dashboard deve gestire `None`/campo mancante senza errore, mostrando
  "n/d (run pre-10pp)" invece di un traceback.
- Grafico AUC per round (convergenza FL) per la configurazione selezionata, sovrapponendo Yeom/Shadow/LiRA.
- Heat map epsilon × dp_mode × AUC composito (equivalente interattivo del foglio Excel "Heat Map").

### Tab 3 — DP Placement Comparison
- Confronto diretto fra le tre modalità (dp-fedavg, central, local) a parità di epsilon: AUC composito,
  utility residua (se disponibile nei JSON — verificare campo `utility`/`reconstruction_error` prima di
  assumerne l'esistenza), PES_v1 calcolato al volo con la formula di `docs/PrivacyExposureScore_v1.md`.

### Tab 4 — Sanity-Check a 5 Assi (Sezione 6 del paper)
- Vista dedicata sulle 5 cartelle `_calibration_overfit*`: tabella con Axis / Manipolazione / AUC composito,
  identica alla tabella già inserita nello scheletro del paper — così un revisore può verificare il numero
  del paper contro il JSON grezzo con due click.
- Nota esplicita nella UI stessa (non solo nel paper) sul significato del risultato: "nessun asse ha superato
  la soglia pre-dichiarata di 0.60 — interpretato come triangolazione di un null, non come prova conclusiva
  generale che l'attacco rilevi sempre la membership."

### Tab 5 — Raw Data / Export
- Tabella grezza, un JSON per riga, con possibilità di scaricare il subset filtrato come CSV — utile per un
  revisore AE che voglia ricalcolare qualcosa autonomamente senza dipendere dalla dashboard stessa.

---

## 6. Cosa NON include (esplicitamente fuori scope)

- Nessuna esecuzione di training, nessun avvio di `run_experiments.py` dalla UI — sola lettura di file già
  su disco.
- Nessuna connessione live a NVFLARE/Containerlab.
- Nessuna autenticazione/multi-utente — pensata per esecuzione locale (`streamlit run`) o come artifact
  scaricabile, non come servizio ospitato.
- Nessuna vista sugli alert ByzantineDetector/IDS timeline in questa prima versione — il dominio Integrity (§4.3
  del paper) è formalmente separato dal dominio Privacy che questa dashboard illustra; un'estensione futura
  potrebbe aggiungere una Tab 6 dedicata, ma non è nello scope minimo per l'AE.

---

## 7. Effort stimato e sequenza suggerita

Non bloccante rispetto al paper — da avviare solo dopo che lo scheletro del paper è stabile e la campagna
5-seed×8-config è conclusa (i numeri reali della Tab 2/3 dipendono da quei risultati). Stima approssimativa
(nessun codice scritto, quindi solo un ordine di grandezza):

1. Scaffold Streamlit + lettura/caching dei JSON esistenti (riuso di `discover_groups()` da
   `check_significance.py` per evitare di duplicare la logica di raggruppamento) — piccolo.
2. Tab 1+2 (Overview + Results Explorer) — il grosso del valore per un revisore AE — medio.
3. Tab 3+4 (DP comparison, sanity-check) — riuso diretto di dati/logica già calcolata altrove — piccolo.
4. Tab 5 (export) — piccolo.
5. QA: eseguire contro `experiments/` reale, verificare che nessun campo mancante (TPR pre-10pp, run
   diagnostici senza tutti i campi) causi un errore invece di un fallback controllato.

## 8. Prossimo passo

Nessuna azione richiesta ora — questa spec resta in attesa di conferma esplicita dell'utente prima di
iniziare l'implementazione, coerente con la decisione di ri-sequenziare la demo come attività non urgente.
