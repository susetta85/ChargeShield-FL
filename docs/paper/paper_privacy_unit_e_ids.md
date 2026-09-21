# Paragrafi aggiornati — unità di privacy, rilevatore non valutato, canary anonimi

Tutti i numeri sono misurati su ACN-Data con lo stesso codice del progetto e
riproducibili con seed 42. Italiano e inglese.

Fonti dei dati citati:
- copertura `user_id`: 48.404 / 66.713 sessioni; per sito 31.452/33.629 Caltech
  (94%), 16.368/31.404 JPL (52%), 584/1.680 Office 1 (35%)
- utenti: 1.028 distinti, mediana 14 sessioni, 87 multi-sito (8,5%), 7.618
  sessioni coinvolte
- canary (protocollo bilanciato, seed 42): 3/20 template membro e 2/20
  non-membro con `user_id`; utente `000006620` in 3 template
  (m18 / n10 / n13), stesso punto di connessione, aprile–settembre 2020
- `anomaly_label`: nullo su tutte le 66.713 sessioni

---

## 1. §7 — Il disallineamento dell'unità di privacy

### Italiano

> **L'unità protetta e l'unità che conta.** La differential privacy limita
> l'influenza di un'unità di dato definita dalla relazione di vicinanza scelta.
> Nei nostri due meccanismi quell'unità è il singolo record, sotto clipping
> per-esempio, oppure l'intero contributo di un client, sotto clipping del
> delta. Nessuna delle due coincide con l'unità che una persona riconoscerebbe
> come propria, e su ACN-Data la distanza è misurabile.
>
> L'identificativo utente è presente in 48.404 delle 66.713 sessioni, con
> copertura fortemente disomogenea fra i siti: 94% a Caltech, 52% a JPL, 35% a
> Office 1. Fra le sessioni identificate, 1.028 utenti distinti contribuiscono
> in mediana **14 sessioni ciascuno**, e 87 di essi — l'**8,5%** — ricaricano
> presso più di un sito, per complessive 7.618 sessioni.
>
> Ne discendono due lacune distinte. La granularità di record non protegge una
> persona perché 14 record compongono: il budget speso su ciascuno si somma, e
> la garanzia a livello di individuo è corrispondentemente più debole di quella
> nominale per record. La granularità di client non la protegge perché per
> l'8,5% degli utenti il contributo attraversa i confini organizzativi, e
> nessun operatore può clippare dati che non possiede.
>
> **Un caso osservabile nel nostro controllo positivo.** Il disallineamento non
> è ipotetico nemmeno all'interno dell'esperimento. Dei quaranta template
> estratti per il controllo bilanciato, cinque portano un identificativo utente,
> e tre appartengono alla stessa persona: una sessione assegnata al gruppo
> membro e due al gruppo non-membro, tutte registrate sullo stesso punto di
> connessione nell'arco di cinque mesi. Un attacco a livello di record risponde
> correttamente alla domanda che pone — quella sessione era nel training set? —
> ma per quel guidatore la risposta negativa su due record convive con quella
> affermativa su un terzo. Nessuna delle tre risposte dice ciò che la persona
> vorrebbe sapere, cioè se la propria attività di ricarica sia finita nel
> modello; la risposta a quella domanda è sì.
>
> Riportiamo questo caso come illustrazione e non come statistica: con cinque
> template identificati su quaranta è per costruzione aneddotico, e la misura
> resta quella sull'intero dataset riportata sopra.ChargeShield-FL, un framework per la valutazione e l’auditing della privacy in sistemi FL
>
> **Perché non valutiamo la DP a livello di soggetto.** Il rimedio naturale
> sarebbe limitare il contributo aggregato di ciascun soggetto, il che richiede
> di raggruppare i gradienti per persona anziché per record e porta la
> sensibilità al numero massimo di sessioni per utente. Oltre al costo
> implementativo, su questi dati la valutazione sarebbe parziale: con il 35% di
> copertura a Office 1, dove conduciamo il controllo positivo, l'unità soggetto
> non è definita per la maggioranza delle sessioni. La riportiamo quindi come
> lavoro futuro motivato da una misura, non come opzione trascurata.

### English

> \textbf{The protected unit and the unit that matters.} Differential privacy
> bounds the influence of a data unit fixed by the chosen neighbouring relation.
> In our two mechanisms that unit is the individual record, under per-example
> clipping, or a client's entire contribution, under delta clipping. Neither
> coincides with the unit a person would recognise as their own, and on ACN-Data
> the gap is measurable.
>
> A user identifier is present in 48,404 of 66,713 sessions, with markedly
> uneven coverage across sites: 94\% at Caltech, 52\% at JPL, 35\% at Office~1.
> Among identified sessions, 1,028 distinct users contribute a median of
> \textbf{14 sessions each}, and 87 of them --- \textbf{8.5\%} --- charge at more
> than one site, accounting for 7,618 sessions.
>
> Two distinct gaps follow. Record granularity does not protect a person because
> 14 records compose: the budget spent on each accumulates, and the
> individual-level guarantee is correspondingly weaker than the nominal
> per-record one. Client granularity does not protect them because for 8.5\% of
> users the contribution crosses organisational boundaries, and no operator can
> clip data it does not hold.
>
> \textbf{A case observable within our own positive control.} The mismatch is not
> hypothetical even inside the experiment. Of the forty templates drawn for the
> balanced control, five carry a user identifier, and three belong to the same
> person: one session assigned to the member group and two to the non-member
> group, all recorded at the same connection point over five months. A
> record-level attack answers correctly the question it poses --- was this
> session in the training set? --- yet for that driver a negative answer on two
> records coexists with an affirmative one on a third. None of the three answers
> tells the person what they would want to know, namely whether their charging
> activity entered the model; the answer to that is yes.
>
> We report this case as an illustration rather than a statistic: with five
> identified templates out of forty it is anecdotal by construction, and the
> measurement remains the dataset-wide one above.
>
> \textbf{Why we do not evaluate subject-level DP.} The natural remedy bounds
> each subject's aggregate contribution, which requires grouping gradients by
> person rather than by record and raises sensitivity to the maximum number of
> sessions per user. Beyond the implementation cost, evaluation on this data
> would be partial: at 35\% coverage in Office~1, where we run the positive
> control, the subject unit is undefined for most sessions. We therefore report
> it as future work motivated by a measurement, not as a neglected option.

---

## 2. §5 — Il rilevatore non è valutato come rilevatore

Va subito dopo la sottosezione sulla scelta del modello.

### Italiano

> **Cosa non misuriamo.** Non riportiamo l'accuratezza di rilevamento del
> modello, e la ragione è nel dato: ACN-Data non contiene etichette di
> intrusione, e il campo predisposto per esse è nullo su tutte le 66.713
> sessioni. A nostra conoscenza nessun dataset pubblico di ricarica per veicoli
> elettrici ne fornisce.
>
> È la stessa scarsità che motiva il federated learning in questo dominio. Se
> ogni operatore disponesse di incidenti annotati in quantità sufficiente,
> addestrerebbe un rilevatore per conto proprio e la collaborazione non
> servirebbe; è proprio perché le etichette mancano che si ricorre a un modello
> non supervisionato addestrato in comune, ed è quella collaborazione a
> sollevare la domanda di privacy che studiamo.
>
> Valutiamo quindi il rilevatore per l'unica proprietà che la nostra domanda
> richiede — cosa il suo addestramento rivela sui dati che lo hanno prodotto —
> e non per la sua capacità di rilevare attacchi. Abbiamo considerato e scartato
> l'iniezione di anomalie sintetiche: misurerebbe il rilevatore contro il
> generatore che le produce, e non direbbe nulla di trasferibile.
>
> La conseguenza da tenere presente nel leggere i risultati è che il modello è
> rappresentativo per architettura, scala e regime di addestramento di ciò che
> un operatore impiegherebbe, ma la sua efficacia operativa resta fuori
> dall'ambito di questo lavoro.

### English

> \textbf{What we do not measure.} We do not report the model's detection
> accuracy, and the reason lies in the data: ACN-Data carries no intrusion
> labels, and the field provided for them is null across all 66,713 sessions. To
> our knowledge no public EV-charging dataset supplies them.
>
> This is the same scarcity that motivates federated learning in this domain. If
> each operator held enough annotated incidents, it would train a detector on its
> own and collaboration would be unnecessary; it is precisely because labels are
> absent that an unsupervised model is trained jointly, and it is that
> collaboration which raises the privacy question we study.
>
> We therefore evaluate the detector for the one property our question requires
> --- what its training reveals about the data that produced it --- and not for
> its ability to detect attacks. We considered and rejected injecting synthetic
> anomalies: that would measure the detector against the generator producing
> them, and would not transfer.
>
> The consequence to bear in mind when reading our results is that the model is
> representative in architecture, scale and training regime of what an operator
> would deploy, while its operational effectiveness remains outside the scope of
> this work.

---

## 3. §6.2 — Nota sui canary e l'identificazione

Da inserire alla fine della descrizione del protocollo canary.

### Italiano

> **I canary sono in larga maggioranza sessioni anonime.** Dei venti template
> assegnati al gruppo membro, tre portano un identificativo utente; due su venti
> nel gruppo non-membro. La proporzione è coerente con la copertura del 35% di
> Office 1 e non è un artefatto della selezione, che è casuale e non filtra su
> quel campo.
>
> Ne segue un limite di portata che dichiariamo esplicitamente: il controllo
> positivo stabilisce che l'apparato rileva la memorizzazione **a livello di
> record**, e non consente di concludere nulla sul livello utente o soggetto nel
> sito in cui è condotto.

### English

> \textbf{Canaries are overwhelmingly anonymous sessions.} Of the twenty
> templates assigned to the member group, three carry a user identifier; two of
> twenty in the non-member group. The proportion is consistent with Office~1's
> 35\% coverage and is not an artefact of selection, which is random and does not
> filter on that field.
>
> A scope limit follows, which we state explicitly: the positive control
> establishes that the apparatus detects memorisation \textbf{at record level},
> and supports no conclusion about the user or subject level at the site where it
> is run.

---

## 4. §9 — Voci di Limitazioni da aggiornare

**Sostituire** la voce esistente sulla granularità con:

> - **Disallineamento dell'unità di privacy, misurato.** Nessuno dei due
>   meccanismi valutati limita il contributo aggregato di una persona. Su
>   ACN-Data un utente contribuisce in mediana 14 sessioni, e l'8,5% degli
>   utenti identificati ricarica presso più siti. La DP a livello di soggetto
>   resta lavoro futuro, anche perché la copertura parziale dell'identificativo
>   (35% a Office 1) ne renderebbe la valutazione incompleta su questi dati.

**Aggiungere:**

> - **Il rilevatore non è valutato come tale.** ACN-Data non contiene etichette
>   di intrusione. Riportiamo le proprietà di privacy del suo addestramento, non
>   la sua accuratezza operativa.

> - **Copertura disomogenea dell'identificativo utente** fra i siti (94% / 52% /
>   35%), che limita ogni analisi a livello di persona e in particolare la
>   rende parziale nel sito del controllo positivo.
