# ChargeShield-FL — Guida scientifica e operativa

**Revisione 3 · 19 settembre 2026 · Per il dottorando e gli strumenti AI che lo assistono**

## 1. La direzione del progetto: leggere prima di iniziare qualsiasi attività

### Obiettivo centrale

> **In quali condizioni la differential privacy riduce la capacità degli attacchi di membership inference nel sistema federato studiato, e quale costo comporta sul modello?**

La direzione concordata è uno **studio empirico applicativo rigoroso**. Recuperiamo le domande iniziali su budget DP, durata del training, eterogeneità dei dati e FedAvg/FedProx. I controlli sperimentali rendono affidabile la misura. Il framework rende gli esperimenti riproducibili. Una nuova metodologia di validazione è una possibile estensione, subordinata a evidenze specifiche, e non l'obiettivo attuale.

**Catena logica da preservare:** esigenza applicativa → ciò che sappiamo dalla letteratura → domanda circoscritta → confronto sperimentale → risultato con incertezza → conclusione proporzionata alle evidenze.

Una difficoltà incontrata lungo la catena non cambia automaticamente la domanda iniziale. Se l'attacco non rileva un segnale, bisogna capire che cosa consente di concludere l'esperimento. Non bisogna necessariamente modificare il sistema finché l'attacco riesce, né trasformare la diagnostica nel contributo del paper.

### Decisioni già concordate

| Elemento | Ruolo attuale |
|---|---|
| Confronto DP/non-DP e costo sul modello | Centro del lavoro, RQ1. |
| Eterogeneità dei dati | Fattore da studiare mediante un confronto controllato, RQ2. |
| FedAvg/FedProx | Confronto da recuperare o completare, RQ3. |
| Controlli, canary e swap | Parte della metodologia; verifica della sensibilità e degli artefatti. |
| Rilevamento della MIA passiva mediante anomalie nei gradienti | RQ4 originaria esclusa dal piano corrente, per incompatibilità con l'attaccante passivo considerato. |
| OCPP/MQTT, container e topologia | Contesto o infrastruttura; fattori scientifici solo se se ne definisce e misura l'effetto sul training o sulle osservazioni. |
| Nuovo metodo di auditing | Estensione eventuale, da discutere con il supervisore su evidenze concrete. |
| Qualità applicativa del rilevatore | Non ancora dimostrata: con i dati descritti misuriamo costo in ricostruzione. |

Questo documento sostituisce come guida attiva l'impostazione metodologica della [versione precedente](ChargeShield_FL_spina_dorsale_consolidata_v1.md), che resta materiale storico. La [ricognizione bibliografica](Ricerca_DP_MIA_paper_dataset.md) fornisce ulteriori fonti e dataset. I risultati numerici presenti nei documenti del progetto sono **dichiarati, non verificati qui su codice e log**. L'inventario iniziale deve stabilire quali siano effettivamente utilizzabili.

## 2. Motivazione: quale decisione vogliamo informare

Le sessioni di ricarica contengono informazioni su energia, durata e orari; quando collegabili a utenti, possono rivelare aspetti delle loro abitudini. Il FL consente di apprendere su dati distribuiti senza riunire i record, ma gli artefatti condivisi devono essere valutati rispetto alle inferenze che consentono. Il problema applicativo è scegliere e valutare una protezione nel contesto concreto di dati, modello, attaccante e vincoli di qualità.

La DP offre una garanzia formale rispetto a un'unità protetta, a un meccanismo e ai suoi output. La valutazione empirica misura invece ciò che **attacchi specificati** riescono a inferire nelle condizioni osservate. Le due informazioni sono complementari. ε non è una percentuale di record identificabili; una MIA debole non dimostra che la protezione sia inutile. [Dwork e Roth](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

La motivazione del paper non è «la DP non risponde» né «nessuno sa interpretare la MIA». È questa:

> Per valutare una configurazione di protezione in questo sistema servono confronti empirici credibili, che mostrino insieme il successo degli attacchi e il costo sul modello e chiariscano quanto i risultati dipendano da training, distribuzione dei dati e algoritmo federato.

Non dare per dimostrati i presupposti del progetto iniziale. Tre siti reali non sono necessariamente tre operatori commerciali indipendenti; quattro cluster di una topologia simulata non sono quattro popolazioni osservate nel dataset. Non affermare che un singolo operatore abbia dati insufficienti senza evidenza. “Deployment realistico” richiede una corrispondenza documentata fra infrastruttura, dati e comportamento effettivamente esercitato.

### Perché questa domanda merita uno studio, anche se ha precedenti

Un risultato ottenuto con altro modello, distribuzione, unità protetta o accesso dell'attaccante non determina il risultato del nostro sistema. Uno studio applicativo può aggiungere conoscenza se isola fattori, produce confronti riproducibili e chiarisce condizioni e limiti del compromesso osservato. L'uso di un nuovo dominio, da solo, non garantisce originalità né forza scientifica: l'introduzione dovrà indicare esattamente quali confronti e risultati aggiungiamo ai precedenti più vicini.

## 3. Related work: da dove partiamo e che cosa non rivendicare

| Filone | Evidenza già disponibile | Implicazione per il progetto |
|---|---|---|
| **DP, attacchi e utilità** — [Jayaraman e Evans, USENIX 2019](https://www.cs.virginia.edu/~evans/pubs/usenix2019/); [Naseri et al., NDSS 2022](https://www.ndss-symposium.org/wp-content/uploads/2022-54-paper.pdf) | Il compromesso empirico fra protezione, MIA e qualità è già studiato, anche in FL. | Confrontare unità, osservazioni, dati e risultati; non rivendicare il primo studio DP/MIA. |
| **Qualità degli attacchi e metriche** — [Carlini et al., IEEE S&P 2022](https://arxiv.org/abs/2112.03570) | LiRA e la valutazione alle basse FPR affrontano limiti delle misure medie. | Usare metriche e calibrazione appropriate; AUC vicina al caso non descrive ogni regione della ROC. |
| **Validità della valutazione** — [Duan et al., COLM 2024](https://arxiv.org/abs/2402.07841); [Gomez, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39276); [Bai et al., MIE 2026](https://journals.sagepub.com/doi/10.3233/SHTI260258) | Sono documentati successi apparenti dovuti a differenze distributive e controlli per separazioni spurie. | La validità della misura è un problema noto; correggere un confondente non costituisce automaticamente una nuova metodologia. |
| **Canary e auditing** — [CANIFE, ICLR 2023](https://arxiv.org/abs/2210.02912); [Steinke et al., NeurIPS 2023](https://arxiv.org/html/2305.08846v1); [Nasr et al., USENIX 2023](https://www.usenix.org/conference/usenixsecurity23/presentation/nasr); [Andrew et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html) | Esistono canary progettati, inclusione randomizzata e audit con analisi statistiche e assunzioni esplicite. | Adottare e confrontare i metodi applicabili. Non presentare tutti gli audit come semplici controlli positivi privi di validazione. |
| **Unità del soggetto** — [Suri et al., preprint 2022/2023](https://arxiv.org/html/2206.03317v3) | La membership di un soggetto con record distribuiti fra silos è già studiata. | Distinguere record, template, persona e client; la presenza di utenti multisito motiva un limite, non dimostra un attacco subject-level. |
| **Dominio delle ricariche** — [PFAD-BC, 2026](https://link.springer.com/article/10.1007/s44443-026-01265-6) | Precedente vicino con autoencoder federati e confronto MIA/DP. | Confrontare protocolli. Il valore 0,781 riportato è accuracy: non sottrarlo alla nostra AUC circa 0,5. |

La revisione bibliografica deve produrre una **matrice di confrontabilità**, non una lista di citazioni: task/modello; dati e split; unità MIA/DP; osservazioni; algoritmo federato; budget e accounting; metriche; controlli; qualità del modello; codice disponibile; differenza rispetto al nostro studio. Indicare separatamente “non riportato”, “non estratto” e “non applicabile”. Le conclusioni dei paper vanno verificate sul testo originale, non ricostruite dalle sole sintesi AI.

Il possibile contributo applicativo è una caratterizzazione controllata del sistema studiato. Per descriverla come lacuna colmata, occorre prima verificare quali combinazioni e domande siano già affrontate dai lavori pertinenti. Non è ancora giustificata la formula «la letteratura usa i canary, noi li validiamo».

## 4. Research question: continuità con il progetto iniziale

Manteniamo la numerazione originale per rendere visibile che cosa è stato conservato, precisato o escluso. La priorità operativa è **RQ1 → RQ3 → RQ2**, perché il confronto fra algoritmi può riutilizzare più direttamente il protocollo centrale. RQ2 non viene cancellata: la sua fattibilità va decisa sulla base dell'inventario e delle risorse.

### RQ1 — Effetto della DP e della durata del training

> Come variano le prestazioni degli attacchi e l'errore di ricostruzione al variare della configurazione DP e dei round, a dati, partizione e algoritmo federato fissati?

La RQ originaria citava FedMIA e AUC. Nell'inventario verificare quale implementazione FedMIA sia stata realmente eseguita e con quali osservazioni. Se gli attacchi disponibili sono Yeom, Shadow o LiRA, documentare la modifica: non rinominarli FedMIA per allinearsi al progetto iniziale. Conservare AUC per continuità, affiancandola a metriche operative sostenibili dalla numerosità.

**Risposta richiesta:** curve o confronti appaiati che mostrino attacco, costo e incertezza. Un risultato negativo circoscritto è una risposta possibile; una baseline non informativa non quantifica automaticamente il beneficio della DP.

### RQ2 — Effetto dell'eterogeneità dei dati

> A parità di popolazione complessiva e configurazione di apprendimento, come cambiano MIA e costo della DP passando da partizioni circa IID a partizioni eterogenee?

Separare quantità di dati per client e distribuzioni delle feature. Un confronto tra siti diversi senza riferimento omogeneo non isola l'eterogeneità. Per inferenza a livello di client, ricostruire le etichette dopo ogni nuova partizione; per membership globale, mantenere quando possibile lo stesso insieme IN/OUT.

**Risposta richiesta:** effetto del tipo di partizione e, se supportato dal disegno, sua interazione con la protezione. Non attribuire l'effetto a OCPP/MQTT senza una variabile misurata che colleghi protocollo e training/osservazioni.

### RQ3 — FedAvg rispetto a FedProx

> A parità di dati, partizione, risorse di training e garanzia DP confrontabile, FedProx con μ=0,01 cambia il compromesso MIA–ricostruzione rispetto a FedAvg con μ=0?

**Risposta richiesta:** confronto appaiato nei regimi senza DP e con DP. La presenza del termine prossimale nel codice non risponde alla RQ. Il confronto riguarda questi valori di μ, non la superiorità universale di un algoritmo. Se cambia il budget di ottimizzazione, dichiararlo e separarne l'effetto.

### RQ4 originaria — Rilevamento dell'attaccante

**Esclusa dal piano corrente.** L'aggregatore honest-but-curious può eseguire la MIA offline senza modificare gli aggiornamenti. Un rilevatore che vede soltanto quegli aggiornamenti non dispone necessariamente di un segnale dell'avvenuta inferenza. Krum e filtri di similarità non costituiscono, per questo solo fatto, rilevatori di MIA passiva.

Una domanda su attacchi attivi richiederebbe azioni e sensori definiti, un diverso threat model e un piano specifico. È un possibile progetto successivo, non un esperimento mancante da aggiungere incidentalmente.

### Tabella da completare nell'inventario

| Domanda | Stato ricavabile dai documenti | Verifica necessaria |
|---|---|---|
| RQ1 | Confronti descritti, ma con cambiamenti di pipeline e regimi differenti. | Quali esecuzioni sono corrette, comparabili e complete? |
| RQ2 | Descritta eterogeneità fra siti. | Esiste un riferimento omogeneo realmente eseguito? |
| RQ3 | Descritto un termine prossimale. | Esistono run appaiati μ=0/0,01? |
| RQ4 | Componenti di rilevamento citati; threat model passivo. | Registrare l'esclusione e non rivendicare una risposta. |

“Non documentato” non significa “non eseguito”. Cercare prima configurazioni e risultati, poi programmare nuove run.

## 5. Specifica scientifica minima da congelare

Prima della campagna centrale il dottorando compila una scheda con questi campi. Gli strumenti AI devono leggerla e segnalare incoerenze prima di aggregare risultati.

| Campo | Decisione richiesta |
|---|---|
| Popolazione e dati | Versione ACN, siti, date, filtri, feature, duplicati e identificativi disponibili. |
| Unità di membership | Sessione, finestra, template, persona o client. Non cambiarla fra celle dello stesso confronto. |
| Partizioni | Training, validation, test naturale, dati ausiliari dell'attacco, IN/OUT e assegnazione ai client. |
| Modello | Architettura, parametri, normalizzazione, preprocessing; distinguere ordinario e modificato per i canary. |
| Training | Algoritmo, μ, round, epoche locali, batch, ottimizzatore, scheduling e partecipazione dei client. |
| Avversario | Che cosa osserva, quando lo osserva e quali dati ausiliari possiede. Scegliere una superficie primaria per il primo confronto. |
| DP | Punto del rumore, clipping, unità/adiacenza, campionamento, σ, δ, accountant, composizione ed ε effettivo. |
| Misura | Attacchi/versioni, calibrazione, soglie, metriche primarie e secondarie, incertezza e unità di replica. |
| Costo | Errore di ricostruzione su holdout naturale, per sito e aggregato; risorse se misurate. |

Un server che vede l'update grezzo non è protetto rispetto a quella osservazione da rumore applicato più tardi. Gli attacchi sul modello finale e sugli update individuali non hanno necessariamente lo stesso bersaglio. Le shadow addestrate con dati riservati al client richiedono accesso privilegiato: non presentarle come un attacco operativo se il threat model non concede quei dati.

Controllare preprocessing e normalizzazione: se usano dati non privatizzati e ne rilasciano statistiche, l'accounting del solo training potrebbe non descrivere tutto il meccanismo. Documentare anche le assunzioni di fiducia. Un accountant restituisce la garanzia del meccanismo modellato, non verifica da solo la correttezza dell'implementazione.

## 6. Che cosa intendiamo per “controllo”

Un **controllo sperimentale è una prova di confronto che aiuta a capire perché otteniamo un risultato**. Non è una difesa e non certifica la privacy.

Esempio: l'attacco distingue A da B. Vorremmo attribuirlo al fatto che A era nel training. Ma A potrebbe essere più facile da ricostruire anche senza essere stato incluso. Il controllo mette alla prova questa spiegazione alternativa.

| Controllo | Che cosa facciamo | Che cosa possiamo dedurne |
|---|---|---|
| **Negativo** | Confrontiamo pseudo-classi di candidati entrambi esclusi, o entrambi inclusi, secondo un disegno dichiarato. | Una separazione ripetibile non prova una differenza di membership fra quelle classi; occorre cercare altre cause. |
| **Positivo con canary** | Creiamo una condizione favorevole alla memorizzazione con assegnazione IN/OUT corretta. | Se emerge un segnale dopo i controlli, l'attacco lo rileva in quel regime; non è validato per ogni regime naturale. |
| **Modello non addestrato** | Valutiamo i candidati prima del training. | Un segnale preesistente mette in dubbio l'attribuzione all'addestramento; l'assenza di segnale non esclude ogni altro confondente. |
| **Swap** | Riaddestriamo con A-IN/B-OUT e poi B-IN/A-OUT. | Verifichiamo se l'effetto segue l'inclusione o favorisce sempre gli stessi candidati. È una diagnostica, non un certificato generale. |
| **Clipping senza rumore** | Inseriamo il clipping mantenendo σ=0. | Distinguiamo il cambiamento dovuto al clipping da quello associato all'aggiunta del rumore. |

Requisiti di correttezza: pool di candidati costruito con procedura comune e assegnazione dichiarata; scorer che non consulta la vera etichetta del candidato al test; calibrazione separata; preprocessing coerente; duplicati e sovrapposizioni gestiti secondo l'unità scelta.

Scambiare soltanto etichette su punteggi fissi produce AUC complementari: la somma 1 è un'identità, non dimostra sensibilità. Lo swap utile richiede nuovi training. Anche allora, “simmetrico=membership” non è un teorema generale: le interazioni fra record possono modificare l'effetto. Il cambio dell'intero gruppo non equivale all'add/remove isolato di una sessione.

I controlli sono **requisiti per fidarsi della misura**, non una promessa di novità. Non serve dimostrare che siano nuovi per rispondere alle RQ applicative.

## 7. Piano di lavoro: fasi, output e criteri di avanzamento

### Fase A — Inventario delle evidenze, prima di nuovi training

Ricostruire le run effettivamente completate. Una riga per run con ID, RQ, commit, configurazione, versione/hash degli split, seed, algoritmo/μ, regime naturale o canary, superficie dell'attacco, DP/accounting, checkpoint, metriche, log, artefatti e stato di validità. Distinguere **verificata**, **completata da verificare**, **invalidata**, **incompleta**, **pianificata**. Nessun valore mancante va riempito per analogia.

Produrre una matrice di confronti: quali coppie rispondono a RQ1, RQ2 e RQ3? Segnalare differenze non controllate, configurazioni duplicate e risultati prodotti dallo scorer precedente. Conservare i vecchi risultati con il motivo dell'invalidazione; correggere la pipeline non rende automaticamente valide le metriche già calcolate.

**Output:** registro run, matrice di confrontabilità, elenco breve delle lacune effettive e proposta del minimo rerun necessario. **Passaggio alla fase B:** esiste una configurazione ordinaria tracciabile e sono risolti i problemi di correttezza che impediscono il confronto centrale. Non occorre prima dimostrare una nuova metodologia.

### Fase B — Esperimento centrale per RQ1

Scegliere una sola configurazione ordinaria rappresentativa. Fissare dati, partizioni, architettura, algoritmo e superficie di osservazione. Conservare ogni modifica necessaria alla DP, per esempio la normalizzazione, anche nei riferimenti confrontabili.

| Condizione | Funzione |
|---|---|
| B0: nessun clipping, nessun rumore | Riferimento ordinario senza protezione DP. |
| B1: clipping, nessun rumore | Controllo del contributo del clipping; non attribuire una garanzia DP solo al clipping. |
| B2 e B3: stesso clipping con due livelli di rumore DP | Misura dell'effetto della protezione e della sua intensità. I livelli si scelgono nel pilot, poi si congelano. |

Valutare alcuni checkpoint prestabiliti fino al budget massimo di round. Non variare contemporaneamente architettura, feature, duplicazione e rumore. Usare almeno una baseline semplice e un attacco calibrato, purché entrambi applicabili alla superficie scelta; separarli dal confronto con avversari più privilegiati.

**Due confronti da non confondere.** A σ fissato, più round cambiano sia il modello sia ε cumulativo. Questa curva risponde alla domanda sull'evoluzione di una politica di training. Per confrontare durate diverse allo stesso ε finale occorre invece calibrare σ separatamente e ripetere il training: è un'altra domanda. Iniziare dalla prima, riportando ε a ogni checkpoint, senza chiamarla “effetto dei round a privacy costante”.

L'accounting deve coprire il transcript osservabile. Se l'avversario vede ogni update, la garanzia deve coprire quella sequenza. Checkpoint derivati da un transcript già contabilizzato non vanno automaticamente contati come meccanismi indipendenti; nuove release che non siano post-processing richiedono analisi. Se si pubblicano più modelli addestrati sugli stessi dati privati, distinguere anche la garanzia della singola run dalla composizione delle release.

**Output per checkpoint:** successo di ciascun attacco con intervalli, errore di ricostruzione su holdout naturale fisso, ε/δ effettivi, diagnostiche e costi misurati. Validare gli attacchi con controlli separati. Il holdout di utilità non deve essere dominato dalle copie canary.

### Fase C — Decisione dopo il confronto centrale

| Esito | Conclusione consentita | Azione successiva |
|---|---|---|
| Segnale naturale misurabile senza DP e riduzione con DP | Riduzione osservata per attacchi, dati e accesso specificati. | Quantificare costo ed estendere a RQ3/RQ2. |
| Risultato vicino al caso con intervalli sufficientemente precisi | Effetto piccolo entro margini dichiarati per la metrica studiata; non sicurezza generale. | Misurare comunque il costo e verificare se RQ3/RQ2 aggiungano un confronto informativo. |
| Risultato vicino al caso con incertezza ampia o attacco degenere | L'esperimento non quantifica bene il fenomeno. | Correzione o aumento mirato della precisione, con limite di risorse stabilito. |
| Ricalibrazione o attacco alternativo rivela un residuo sotto DP | La prima misura sottostimava il segnale rilevabile dagli attacchi esaminati. | Aggiornare i confronti senza dichiarare automaticamente DP inefficace. |
| Canary positivo soltanto in regime molto alterato | Sensibilità dimostrata in quel regime. | Tenere il risultato separato; non usarlo per certificare il null naturale. |

Non inseguire indefinitamente una MIA positiva. Stabilire prima quante diagnostiche e quali risorse dedicare all'incertezza. Se il protocollo resta inconcludente, riportare il limite. Se si esplora memorizzazione indotta, quella campagna riceve un'etichetta e un'ipotesi proprie.

### Fase D — Confronto FedAvg/FedProx per RQ3

Riutilizzare split, candidati, checkpoint e condizioni DP della fase B. Eseguire μ=0 e μ=0,01 con risorse e impostazioni comparabili. Appaiare le run per split/seed e verificare che il meccanismo di clipping/rumore e l'accounting consentano il confronto. Il solo uso dello stesso σ non basta se cambiano campionamento o passi.

Confrontare sia il risultato entro ogni livello di protezione sia la riduzione DP rispetto al riferimento dello stesso algoritmo. Mostrare anche la qualità del modello: una diminuzione della MIA accompagnata da forte peggioramento della ricostruzione non è automaticamente un migliore compromesso. Nessuna ipotesi impone che FedProx sia più privato.

**Output:** tabella/figura appaiata algoritmo × protezione, con intervalli per differenze di attacco e ricostruzione. Se tutti gli attacchi sono poco informativi, la parte privacy della RQ resta limitata anche se il confronto di ricostruzione è preciso.

### Fase E — Confronto di eterogeneità per RQ2

Usare lo stesso insieme globale di record e, per quanto possibile, lo stesso split globale train/test. Costruire un riferimento circa IID e una partizione eterogenea, inizialmente mantenendo le stesse numerosità per client: così la differenza riguarda principalmente la distribuzione delle feature. Studiare lo sbilanciamento quantitativo in un contrasto separato, se necessario. Ripetere la randomizzazione delle partizioni.

Documentare cosa la partizione preserva e cosa altera: tempo, sito, utenti condivisi e sovrapposizioni. Una partizione IID artificiale è un riferimento sperimentale, non un nuovo deployment reale. Per l'attacco per-client, la verità di membership cambia con la partizione; ricostruire correttamente pool e dati ausiliari. Mantenere comparabile la conoscenza dell'avversario.

**Confronto minimo:** IID/non-IID × senza DP/con una configurazione DP scelta in anticipo. Riportare quantità di dati e statistiche distributive, per verificare che il fattore sia stato effettivamente modificato. Non aggregare soltanto la media: mostrare i siti/client quando la numerosità consente conclusioni.

**Output:** differenze fra partizioni e confronto dell'effetto DP nelle due condizioni. Se quantità, distribuzione e task cambiano insieme, descrivere un confronto fra configurazioni, non l'effetto isolato dell'eterogeneità.

### Fase F — Sintesi, eventuale estensione e scrittura

Chiudere le risposte a RQ1–RQ3 secondo le evidenze. Una RQ non affrontabile va dichiarata rinviata, con ragione e conseguenza sul claim, e discussa con il supervisore: non deve scomparire dalla narrativa.

Solo dopo il nucleo, scegliere un'estensione motivata: altro dataset per verificare una precisa trasferibilità; regime indotto per esplorare la sensibilità; dati etichettati per la detection; oppure verifica di un contributo metodologico. Non avviare tutte le direzioni insieme. Ulteriori esperimenti devono poter cambiare una conclusione identificata.

## 8. Misure e criteri che rendono dimostrabili le conclusioni

### Prestazione dell'attacco e riduzione con DP

Definire prima della conferma una metrica primaria S e un FPR operativo sostenibile dai dati; conservare ROC e AUC come complementi. La baseline casuale per TPR@FPR è TPR=FPR. Definire il contrasto ΔS=S(non-DP)−S(DP) e stimarlo con intervalli appaiati. Separare il contrasto B0/B2 dal contrasto B1/B2, che isola l'aggiunta di rumore rispetto al clipping-only.

Se l'AUC è inferiore a 0,5, verificare orientamento e score su validation; non scegliere il verso sulla test per aumentare il risultato. Non confrontare accuracy e AUC come se fossero la stessa grandezza. I valori quasi casuali non escludono un residuo in una parte della ROC o in sottogruppi.

### Calibrazione, repliche e potenza

Separare sviluppo, calibrazione e test degli attacchi. Consentire una calibrazione adeguata a ogni regime DP con budget dichiarato. Selezionare attacchi e iperparametri su validation; non riportare il massimo sulla test come se fosse una scelta prefissata.

Il pilot serve a fissare livelli, variabilità, numero di repliche e risorse; il risultato finale richiede blocchi nuovi. Ogni blocco identifica split/pool, assegnazione e seed. Considerare le dipendenze fra sessioni, template e training; le copie di uno stesso canary non sono repliche indipendenti. Riportare anche i singoli risultati, non soltanto un p-value.

Fissare margini di rilevanza per S, ΔS e ricostruzione. Il margine AUC 0,02 citato nel progetto richiede motivazione e non vale automaticamente per TPR. Un intervallo che include zero non dimostra equivalenza: può indicare scarsa precisione. Usare criteri di equivalenza solo con margini e analisi appropriati. Le analisi esplorative e la molteplicità dei confronti vanno dichiarate.

Con 20 template OUT, un falso positivo vale il 5%: non è una base sufficiente per una stima solida a FPR 0,1%. Anche con molti candidati servono intervalli e gestione delle dipendenze. Verificare la numerosità effettiva prima di scegliere la metrica primaria, senza cambiare opportunisticamente metrica dopo i risultati.

Nel contesto precedente compaiono cinque seed, t(2) e un sign test 5/5 con p≈0,031. Riconciliare le unità: il valore 0,03125 è unilaterale, quello bilaterale è 0,0625. Nessuno di questi numeri deve essere riutilizzato senza risalire all'analisi originale.

### Qualità del modello e unità di protezione

Misurare ricostruzione su dati naturali non usati per il fitting, con trasformazioni fissate e scala dichiarata. Riportare variazione assoluta e, se sensata, relativa: rapporti contro una loss quasi zero possono essere fuorvianti. Una loss assoluta “piccola” non dimostra utilità applicativa. Senza etichette o un task validato, il paper non misura la capacità di rilevare intrusioni.

Se l'inclusione riguarda trenta copie, il bersaglio MIA è il gruppo/template; ε per record non ne descrive direttamente la protezione. Sessioni ripetute per persona richiedono garanzie derivate o meccanismi coerenti con quell'unità. Non confrontare record-DP e client-DP allo stesso ε come protezioni equivalenti.

La mediana dichiarata di 14 sessioni non è un limite massimo sul contributo individuale. L'8,5% di utenti multisito riguarda gli identificativi osservati, con copertura incompleta e diversa fra siti. Sono caratteristiche da verificare e limiti da discutere, non una valutazione già svolta della privacy delle persone.

## 9. Contributo e claim: cosa il paper può promettere

### Formulazione di lavoro, senza anticipare i risultati

> Conduciamo una valutazione riproducibile della membership inference e del costo di ricostruzione in autoencoder federati su sessioni reali di ricarica. Esaminiamo l'effetto della configurazione DP e della durata del training; mediante confronti controllati valutiamo inoltre il ruolo dell'algoritmo federato e della partizione dei dati. La valutazione distingue prestazioni degli attacchi, validità della misura e limiti delle conclusioni nei regimi con segnale debole.

Questa formulazione vale per un paper che completi tutti quei confronti. Se RQ2 o RQ3 non vengono completate, ridurre esplicitamente la promessa. Il claim finale dovrà contenere **risultati**, non soltanto l'elenco delle attività.

| Contributo candidato | Evidenza necessaria | Cosa non basta |
|---|---|---|
| Caratterizzazione del compromesso DP–MIA–ricostruzione | Fase B, intervalli e accounting confrontabile. | Due numeri di AUC senza riferimento e costo. |
| Effetto di FedAvg/FedProx e/o eterogeneità | Fasi D/E con fattori isolati e risultati tracciabili. | Aver usato FedProx o dati di tre siti. |
| Riproducibilità della valutazione | Configurazioni, split o procedure per ricrearli, codice, log, analisi e versioni. | Un container funzionante senza esperimenti ricostruibili. |
| Risultato negativo informativo | Protocollo corretto, precisione adeguata e perimetro esplicito. | “p>0,05, quindi nessun leakage”. |

Possibile conclusione circoscritta: «Nel regime X non quantifichiamo un beneficio empirico aggiuntivo della DP con gli attacchi considerati, entro la precisione indicata, mentre osserviamo il costo Y». Non tradurla in «DP non serve». Nel regime indotto, un calo circa 0,72→0,54 rimane un risultato dichiarato da verificare e separato dalle osservazioni naturali.

### Quando riaprire la pista metodologica

Solo se si identifica un problema riproducibile che resta dopo l'applicazione corretta dei metodi pertinenti. Confrontare P1, riferimento corretto, e P2, riferimento più il controllo proposto, anche a pari risorse. Mostrare errori evitati, segnali validi preservati e una conseguenza sostanziale sulla conclusione DP. La pipeline storica difettosa può documentare il problema, ma batterla non dimostra novità.

Se P1 risolve tutto, adottarlo e continuare le RQ applicative. Se P2 aggiunge evidenza, presentare al supervisore confronto bibliografico, risultati, costo e proposta di cambiamento. Non spostare automaticamente titolo, introduzione e campagna verso «validiamo i canary». Il cambio di direzione è una decisione scientifica esplicita.

## 10. Struttura del paper: ogni sezione serve una domanda

| Sezione | Contenuto | Evidenza o figura attesa |
|---|---|---|
| **1. Introduction** | Decisione applicativa, limiti dei risultati trasferibili dai precedenti, RQ e contributi effettivi. | Sintesi dei risultati finali, senza promesse non valutate. |
| **2. Background and Related Work** | DP, MIA, metriche, auditing, studi FL e dominio EV; differenze pertinenti. | Matrice di confrontabilità con i lavori più vicini. |
| **3. System, Data and Threat Model** | Corrispondenza fra siti e client, dati, modello, osservazioni e unità DP/MIA. | Schema piccolo del flusso osservato, tabella dei dati. |
| **4. Experimental Methodology** | Run appaiate, configurazioni, controlli, attacchi, accounting e statistica. | Matrice degli esperimenti collegata alle RQ. |
| **5. Results** | Prima RQ1; poi RQ3 e RQ2, motivando l'ordine. Separare dati naturali e canary. | Curve attacco/costo/round, confronto FedAvg–FedProx, confronto delle partizioni. |
| **6. Discussion and Limitations** | Quando il beneficio è misurabile, quando non lo è; persone, utilità applicativa e trasferibilità. | Tabella delle conclusioni consentite e dei limiti. |
| **7. Conclusion** | Risposta alle RQ effettivamente affrontate. | Nessuna nuova affermazione o estensione non valutata. |

I controlli occupano lo spazio necessario a giustificare la misura; casi dettagliati possono andare in appendice. ML Plane, monitor del budget, Byzantine detection e protocolli di rete non diventano contributi principali per il solo fatto di essere implementati. Canary e framework non devono far scomparire le RQ.

Titolo di lavoro: **“Differential Privacy and Membership Inference in Federated EV-Charging Autoencoders: An Empirical Study”**. Affinarlo sui risultati. Il nome ChargeShield-FL non deve implicare che sia stata proposta una nuova difesa.

## 11. Regole operative per il dottorando e gli assistenti AI

### Prima di proporre o eseguire un'attività

1. Leggere obiettivo e decisioni della sezione 1, scheda scientifica e registro delle run.
2. Dichiarare la RQ servita e quale conclusione potrebbe cambiare grazie all'attività.
3. Verificare se l'evidenza esiste già, se è comparabile e se la pipeline che l'ha prodotta è valida.
4. Distinguere correzione di un errore, completamento di un confronto ed estensione della ricerca. Procedere con le attività autorizzate nel piano; proporre esplicitamente i cambi di obiettivo o di perimetro al supervisore.

### Durante il lavoro

- Non adattare l'obiettivo al risultato che è più facile ottenere. Un attacco debole non autorizza a cambiare contemporaneamente modello, feature e dati per produrre una curva più convincente.
- Separare fatti verificati, risultati dichiarati, inferenze, ipotesi e attività proposte. Un job pianificato non è un esperimento completato; uno completato non è automaticamente valido.
- Non inventare numeri, bibliografia, stato dei job, metriche o risultati mancanti. Per una fonte citare testo originale e sezione/tabella pertinente; distinguere i limiti di estrazione AI dalle omissioni del paper.
- Non trattare un cambio di codice come scientificamente neutro: scorer, accesso ai dati, normalizzazione, split e accounting possono invalidare confronti precedenti. Registrare quali run vanno rianalizzate o ripetute.
- Non confrontare unità, metriche o superfici differenti senza dichiararlo. Conservare l'identità degli esperimenti anche quando i numeri coincidono.
- Non selezionare solo seed, attacchi o celle favorevoli. Congelare criteri dopo il pilot, riportare esiti contrari e distinguere esplorazione da conferma.
- Se una diagnostica risolve il problema, tornare alla RQ. Non espandere indefinitamente la diagnostica in un nuovo progetto.

### Dopo ogni blocco di lavoro

Aggiornare registro run e registro decisioni. Il resoconto deve indicare: RQ; lavoro svolto; evidenze con percorsi; risultato e incertezza; interpretazione consentita; limiti; stato della RQ; prossimo passo minimo. Spiegare se il nuovo risultato conferma o cambia una conclusione precedente.

Sono richieste decisioni del supervisore quando si propone di eliminare una RQ, cambiare threat model o unità protetta, promuovere i controlli a contributo principale o ampliare significativamente la campagna. Correzioni tecniche e analisi già comprese nel piano non richiedono continue riconferme: devono essere tracciate e motivate.

### Espressioni da non usare senza l'evidenza corrispondente

| Espressione | Correzione |
|---|---|
| «AUC≈0,5: sistema sicuro» | Nessun segnale rilevato da quell'attacco nella metrica e precisione riportate. |
| «DP ha eliminato il leakage» | Diminuzione del successo degli attacchi valutati; riportare anche il residuo. |
| «Tre siti provano l'effetto non-IID» | Servono confronto e fattori controllati di RQ2. |
| «FedProx migliora la privacy» | Serve RQ3; distinguere privacy misurata e perdita di qualità. |
| «Il canary valida tutti i null naturali» | Valida sensibilità nel regime verificato. |
| «Siamo i primi a validare la MIA» | Esistono precedenti diretti; delimitare e verificare la novità. |
| «Il framework realistico prova la rilevanza del protocollo» | Documentare quali proprietà dell'infrastruttura cambiano l'esperimento. |
| «Loss bassa: rilevatore efficace» | Ricostruzione e capacità di detection sono evidenze diverse. |

## 12. Modelli riutilizzabili per mantenere il filo

### Scheda da compilare prima di una nuova campagna

```text
ID attività:
RQ e fase del piano:
Domanda precisa / ipotesi verificabile:
Conclusione che questo confronto può cambiare:
Evidenze già disponibili e motivo per cui non bastano:
Fattore variato:
Condizioni mantenute uguali:
Unità di membership e unità DP:
Osservazioni e dati ausiliari dell'attaccante:
Configurazioni / split / versioni / checkpoint:
Controlli necessari e spiegazioni alternative esaminate:
Metrica primaria / costo sul modello / margine rilevante:
Calibrazione, repliche, intervalli e criterio di arresto:
Risorse stimate e output atteso:
Esito favorevole, contrario e inconcludente:
Decisione successiva per ciascun esito:
```

### Registro delle decisioni

```text
Data e decisione:
Motivo ed evidenze:
RQ interessate:
Impatto su esperimenti, claim e testo del paper:
Responsabile della decisione:
Condizione per riesaminarla:
```

### Istruzione breve da dare a un assistente AI

> Usa questa guida come riferimento scientifico del progetto. L'obiettivo è valutare empiricamente DP, membership inference e costo in ricostruzione, rispondendo a RQ1–RQ3. Parti dal registro degli esperimenti e distingui evidenze verificate da ipotesi. Collega ogni attività a una RQ e a una conclusione che potrebbe cambiare. Usa controlli appropriati per rendere affidabile la misura, senza trasformarli implicitamente nel contributo principale. Non inventare risultati o lacune di letteratura. Segnala i confronti non validi, conserva gli esiti negativi e proponi il prossimo passo minimo previsto dal piano. Per cambiamenti di obiettivo presenta motivazione ed evidenze al supervisore.

### Prossima azione concreta

**Completare la fase A e portare al supervisore la matrice degli esperimenti reali e un confronto DP/non-DP tracciabile.** Solo dopo questo controllo stabilire i rerun necessari per la fase B. Non iniziare una nuova campagna estesa sulla novità dei canary come primo passo.
