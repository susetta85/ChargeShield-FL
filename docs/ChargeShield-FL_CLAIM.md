# ChargeShield-FL — Claim scientifico

Spina dorsale del progetto e del paper. Le sezioni 1–5 sono una catena: ogni
passaggio è conseguenza del precedente. Se un'attività non serve quella catena,
non entra nel paper.

Versione 2, 19 settembre 2026.

> **Stato: IPOTESI, non linea del paper (deciso il 2026-09-22).** La linea canonica è
> `ChargeShield_FL_spina_dorsale_consolidata.md`, che alla sezione 9 fissa le
> condizioni per riaprire questa pista. Fino ad allora: i controlli di C1 entrano nel
> paper come metodologia della misura, non come contributo; la frase "la letteratura
> usa i canary, noi li validiamo" non va usata; C2 poggia su una cella record-DP a un
> seed con ε da ricalcolare (vedi `Segnalazioni_tecniche_2026-09-22.md`, punto 4).

---

## 1. La decisione da prendere

Un consorzio di operatori di ricarica per veicoli elettrici vuole addestrare
insieme un rilevatore di anomalie. Gli eventi rilevanti sono rari e non
etichettati, quindi nessun operatore ha dati sufficienti da solo; i record di
sessione sono però commercialmente sensibili e riconducibili a persone. Il
federated learning è la risposta naturale, e la differential privacy è la
protezione che si valuta di aggiungervi.

Prima di procedere, il consorzio deve rispondere a due domande operative:

1. **Quanta informazione sui singoli record esce dal processo di
   addestramento?**
2. **Quanto la riduce la differential privacy, e a quale costo per il modello?**

Sono domande quantitative. Le risposte determinano se il deployment si fa, con
quale meccanismo e con quale budget.

---

## 2. Perché la garanzia formale non risponde

La differential privacy fornisce un limite superiore alla quantità di
informazione che un avversario può ottenere su un'unità di dato. Il limite ha
tre caratteristiche che lo rendono inadatto a rispondere direttamente alle due
domande.

**È nel caso peggiore.** Vale per qualunque avversario, qualunque conoscenza
pregressa e qualunque coppia di dataset vicini. L'avversario reale, su dati
reali, estrae tipicamente molto meno.

**È relativo a un'unità che va scelta.** Proteggere una sessione, un operatore
o una persona sono garanzie diverse. Lo stesso valore di ε rispetto a unità
diverse non descrive la stessa protezione.

**Non si traduce in una metrica di rischio.** ε non è una percentuale di record
identificabili né predice il successo di un attacco specifico.

Ne segue che, per rispondere alle due domande, la garanzia formale va affiancata
da una **misura empirica**: si istanzia un avversario, lo si esegue sugli
artefatti che il protocollo gli consegna, e si misura quanto riesce a
distinguere i record usati nell'addestramento da quelli non usati.

È l'approccio adottato dalla letteratura di auditing empirico — Jayaraman &
Evans, CANIFE, Steinke et al., Nasr et al., Andrew et al. — e dagli studi
applicativi che valutano la DP in contesti federati specifici.

---

## 3. Perché la misura empirica è difficile da interpretare

Una misura empirica produce un numero: l'AUC dell'attacco, oppure il tasso di
veri positivi a un tasso di falsi positivi fissato. Per agire su quel numero
bisogna sapere che cosa significhi.

**Nella grande maggioranza dei casi il numero è prossimo al livello del caso.**
È l'esito tipico degli attacchi di appartenenza su modelli di dimensioni
contenute addestrati su dati tabellari, ed è l'esito che abbiamo ottenuto anche
noi su tre siti operativi reali. Non è un'eccezione da spiegare: è la
condizione ordinaria in cui il decisore si trova.

**Un valore prossimo al caso ammette due letture incompatibili.** Il sistema non
perde informazione rilevabile, oppure l'attacco non è in grado di rilevarla in
quel regime. Le due letture portano a decisioni opposte — nel primo caso la DP
è un costo evitabile, nel secondo la misura non dice nulla e la decisione resta
senza base — e nulla nel numero le distingue.

**Distinguerle richiede di stabilire che l'attacco avrebbe rilevato la perdita
se ci fosse stata.** È una proprietà dello strumento, non del sistema misurato,
e va accertata separatamente.

Lo strumento standard per accertarla è il **controllo positivo con canary**: si
inseriscono deliberatamente record memorizzabili nell'addestramento e si
verifica che l'attacco li riconosca. Se li riconosce, si conclude che lo
strumento è sensibile e che il valore misurato sui dati naturali è
interpretabile.

---

## 4. Dove la catena si rompe

Il controllo positivo è la giunzione su cui poggia l'intera interpretazione. Se
il controllo dà una risposta affermativa, tutto il resto segue; se la risposta
è affermativa ma sbagliata, l'errore si propaga a valle senza lasciare traccia.

Il controllo positivo è però **esso stesso un esperimento**, con un gruppo
trattato e un gruppo di confronto, e come ogni esperimento può essere
confuso. La letteratura lo impiega come verifica ma non ne verifica la validità
interna. Sono due i modi in cui può fallire, entrambi silenziosi.

### 4.1 Il controllo può confermare la sensibilità quando non c'è

I record inseriti come canary e quelli usati come confronto devono differire
*soltanto* per l'appartenenza al training. Se differiscono anche per altro,
l'attacco li separa comunque e il controllo restituisce un esito positivo falso.

Nel nostro caso è accaduto tre volte, per tre ragioni diverse.

I due gruppi erano estratti da pool distinti, e i record scelti come canary
risultavano intrinsecamente più facili da ricostruire: sotto modelli ombra
puliti la loro perdita era circa sette volte inferiore a quella dei record di
confronto, indipendentemente dall'appartenenza.

Lo scoring dell'attacco trattava le due classi con formule diverse, producendo
un valore complessivo neutro come somma di due effetti di segno opposto: la
diagnostica che forza entrambe le classi nella stessa formula restituiva 0.21
contro un valore riportato di 0.50.

Una soppressione apparente della perdita dopo l'applicazione della DP si
rivelava un effetto dell'assegnazione: invertendo i ruoli fra i due gruppi,
l'effetto cambiava segno.

In tutti e tre i casi il controllo positivo, eseguito secondo le procedure
pubblicate, avrebbe confermato la sensibilità dello strumento.

### 4.2 Il regime del controllo non è il regime della misura

Un canary è un record deliberatamente reso memorizzabile: duplicato molte volte,
spesso su un modello addestrato più a lungo o con maggiore capacità del
modello sotto esame. Rilevarlo dimostra che l'attacco funziona **in quel
regime**.

La misura da interpretare avviene invece su record visti una volta sola, in un
modello addestrato normalmente. La distanza fra i due regimi non è quantificata
in nessuno dei lavori che abbiamo esaminato, e senza quella quantificazione un
controllo positivo superato non autorizza a leggere il null come assenza di
perdita.

### 4.3 Che il problema sia reale, e non ipotetico

Tre indizi convergenti.

Diversi studi empirici riportano valori prossimi al caso **già in assenza di
protezione** e concludono comunque che la DP riduce il rischio: telemedicina
transfrontaliera con AUC ≈ 0.50 in tutte le modalità, classificazione marittima
AIS a 0.51 per il federated learning da solo, medicina di precisione con
baseline centralizzato a 0.5486. Quando il gruppo di controllo senza protezione
siede già al livello del caso, una misura post-protezione allo stesso livello
non può quantificare una riduzione.

Un lavoro sullo stesso dominio e sullo stesso dataset riporta un baseline senza
protezione di 0.781 dove noi misuriamo 0.5. Due misure della stessa grandezza,
sugli stessi dati e con la stessa classe di modello, differiscono di 0.28. Nulla
nei rispettivi protocolli permette di stabilire quale sia interpretabile.

E nella nostra esperienza diretta, tre risultati su altrettanti controlli
positivi si sono rivelati artefatti — individuati solo perché abbiamo verificato
il controllo invece di fidarcene.

---

## 5. Il claim

> **Interpretare una misura empirica di privacy richiede di validare lo
> strumento, e la validazione standard — il controllo positivo con canary — è a
> sua volta un esperimento che può dare esiti falsamente positivi e che opera in
> un regime distante da quello misurato. Definiamo i controlli che ne
> stabiliscono la validità interna, mostriamo che intercettano errori reali, e
> li usiamo per caratterizzare il confine di rilevabilità della memorizzazione
> su un deployment di ricarica elettrica.**

 una riga da non usare: **la letteratura usa i canary per validare la misura; noi validiamo
i canary.**

---

## 6. Contributi

### C1 — Controlli per la validità interna del controllo positivo

Tre verifiche applicabili a qualunque studio che usi canary.

**Baseline a inizializzazione casuale.** Gli stessi canary valutati contro un
modello mai addestrato. Se li separa già, la separazione precede
l'addestramento.

**Pool unificato con assegnazione casuale.** Entrambi i gruppi estratti da una
sola procedura, i non-membri rimossi dal training e spostati nell'holdout. I due
gruppi diventano campioni scambiabili della stessa distribuzione.

**Controllo di scambio dei ruoli.** Lo stesso seed eseguito nei due versi. Un
effetto simmetrico indica appartenenza, uno antisimmetrico indica una differenza
sistematica fra i gruppi. È il controllo senza precedenti a nostra conoscenza,
ed è quello che ha smentito due dei tre falsi positivi.

A questi si affianca una diagnostica dello scorer, che verifica che le due
classi siano valutate dalla stessa funzione di punteggio.

### C2 — Il confine di rilevabilità su un deployment reale

Applicando C1 a tre siti operativi, stabiliamo a quali condizioni la
memorizzazione diventa rilevabile e da quale parte del confine si collochi un
rilevatore realistico.

Sul modello impiegabile su un controllore di stazione — 570 parametri, sei
feature continue di sessione — nessuna delle sette manipolazioni tentate
produce un segnale rilevabile dagli attacchi valutati. Con capacità triplicata,
una feature quasi-univoca per sessione e duplicazione forzata dei record, il
segnale compare e i controlli di C1 ne certificano la validità.

In quel regime misuriamo l'effetto della protezione: a livello di record,
ε = 7.15 riduce l'AUC sui canary da circa 0.72 a circa 0.54, lasciando il
modello addestrabile.

### C3 — Il disallineamento fra unità protetta e soggetto

Su ACN-Data un utente contribuisce in mediana 14 sessioni e l'8,5% degli utenti
identificati ricarica presso più di un sito. Né la granularità di record né
quella di client limitano il contributo aggregato di una persona: la prima
perché quattordici contributi si compongono, la seconda perché nessun operatore
può limitare dati che non detiene.

La differenza rispetto ai lavori che confrontano le granularità su benchmark
federati: lì la distribuzione dei soggetti fra silos è costruita, qui è una
proprietà misurata del dato.

---

## 7. Limiti dichiarati

**Il risultato è «nessun segnale rilevato dagli attacchi valutati nelle
condizioni provate», non «nessuna perdita».** Il controllo positivo dimostra
sensibilità alla memorizzazione indotta; la distanza rispetto al regime naturale
è dichiarata, non azzerata.

**La granularità non è isolata come causa.** Il confronto fra record-level e
client-level varia insieme unità protetta, budget e regime. I risultati mostrano
che alcune configurazioni si comportano meglio; non dimostrano che sia la
granularità a determinarlo.

**L'utilità applicativa non è misurata.** Valutiamo l'errore di ricostruzione,
non la capacità di rilevare anomalie: ACN-Data non ha etichette di intrusione.
Esistono dataset di ricarica etichettati, ma sono raccolte di laboratorio su
una singola infrastruttura e non offrono la struttura multi-operatore che la
domanda richiede.

**I canary duplicati proteggono un gruppo, non un record.** Se l'appartenenza di
un template comporta trenta copie, l'esperimento manipola un gruppo, e il budget
dichiarato per il singolo record non ne descrive la protezione.

**Il riferimento all'ottimalità bayesiana della loss va contenuto.** Vale sotto
ipotesi specifiche sulla distribuzione dei parametri, e non autorizza a
concludere che gradienti e traiettorie non aggiungano informazione in questo
protocollo.

---

## 8. Perimetro

Restano fuori dal paper, come infrastruttura o lavoro futuro: l'ML Plane, già
pubblicato e qui strumento di raccolta; il Privacy Auditor, di cui riportiamo
soltanto il risultato negativo che la sua telemetria è indipendente dal rischio
misurato; il punteggio di esposizione euristico; il rilevatore Byzantine, mai
esercitato end-to-end; l'attacco basato su gradienti, non citabile per
numerosità; la replica su un secondo dataset di ricarica; il deployment
containerizzato, che resta una nota di riproducibilità.

---

## 9. Questioni aperte

**La discrepanza di baseline con il lavoro sullo stesso dominio.** 0.781 contro
0.5, stessi dati e stessa classe di modello. Le cause possibili — superficie di
osservazione, costruzione dei gruppi, architettura, feature — sono tutte
informative, e spiegarla rafforza il claim più di qualunque esperimento nuovo.

**L'isolamento della granularità**, che richiede un confronto a parità di budget
e regime.

**La quantificazione della distanza fra regime del canary e regime naturale**,
oggi dichiarata ma non misurata.

**La replica su altri domini**, ammessa solo per verificare un'ipotesi precisa:
se il compromesso osservato dipenda dalla struttura dei dati o sia generale.

---

## 10. Criterio di inclusione

Ogni attività va valutata contro questa domanda:

> **Quale conclusione scientifica potrebbe sbagliare un ricercatore usando le
> procedure oggi disponibili, e quale nostro elemento gli permette di
> correggerla?**

Risposta attuale: un ricercatore che esegua un controllo positivo con canary
secondo le procedure pubblicate può concludere che il proprio strumento è
sensibile quando non lo è, perché nessuna di quelle procedure ne verifica la
validità interna. Il controllo di scambio dei ruoli e la diagnostica di
simmetria dello scoring lo rilevano.
