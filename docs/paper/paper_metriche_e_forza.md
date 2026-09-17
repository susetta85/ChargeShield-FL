# §5.x — Cosa misuriamo e cosa rende forte un attacco

Due sottosezioni da inserire nel §5, prima dei risultati. Italiano e inglese.
Sono il pezzo concettuale del paper: vanno scritte per essere capite da chi non
conosce LiRA, senza perdere precisione.

---

## A. Perché l'AUC non basta

### Italiano

> **Cosa significa davvero un attacco riuscito.** Un attacco di inferenza di
> appartenenza produce, per ogni campione, un punteggio che ordina i candidati
> dal più probabile membro al meno probabile. L'AUC-ROC misura la qualità di
> quell'ordinamento nel suo complesso: è la probabilità che, presi a caso un
> membro e un non-membro, il membro riceva il punteggio più alto. Un'AUC di 0.5
> significa ordinamento casuale, un'AUC di 1.0 separazione perfetta.
>
> Il problema è che questa domanda non corrisponde a nessun avversario reale.
> Carlini et al. lo argomentano così: un attaccante non deve classificare
> correttamente tutti i campioni, e non gli interessa la sua accuratezza media.
> Gli interessa **identificare con sicurezza almeno qualche membro**. Un
> attaccante che, su diecimila sessioni, riesce a dire con quasi certezza che
> cinquanta di esse appartenevano al training set di un operatore specifico ha
> violato la privacy di cinquanta persone, e il fatto che sulle restanti 9.950
> tiri a indovinare non lo consola né lo danneggia.
>
> L'AUC media quell'informazione via. È un integrale su tutte le soglie
> possibili, e pesa allo stesso modo il regime ad alto tasso di falsi positivi —
> dove l'attaccante accusa mezzo dataset per prenderne metà, il che non è un
> attacco — e il regime a basso tasso di falsi positivi, dove l'attaccante
> accusa pochi campioni e ha ragione. Un attacco può quindi identificare con
> certezza l'uno per cento dei membri, che è una violazione grave, e avere
> un'AUC praticamente indistinguibile da 0.5, perché quell'uno per cento
> contribuisce pochissimo all'integrale.
>
> **La metrica giusta.** Riportiamo quindi il tasso di veri positivi valutato a
> tassi di falsi positivi fissi e bassi: 0.1%, 1% e 5%. La domanda che pone è
> quella che conta: *se l'attaccante accetta di sbagliare al più una volta su
> cento, quanti membri riesce a identificare?* È una misura del comportamento
> nella coda della distribuzione dei punteggi, che è dove il danno si concentra.
>
> Due precisazioni che teniamo esplicite. Il livello di caso per questa metrica
> è TPR = FPR, non 0.5: un attaccante che indovina a caso e accusa l'1% dei
> campioni ne prende l'1% dei membri. E le due metriche possono divergere — è
> anzi nella regione in cui divergono che l'attacco è pericoloso — per cui
> riportarle entrambe non è ridondanza.
>
> Conserviamo comunque l'AUC per due ragioni pratiche: è la metrica con cui la
> letteratura empirica confronta i risultati, e il test di equivalenza che
> sostiene il nostro risultato nullo è formulato su di essa.

### English

> \textbf{What a successful attack actually means.} A membership-inference
> attack produces, for each sample, a score that ranks candidates from most to
> least likely member. AUC-ROC measures the quality of that ranking as a whole:
> it is the probability that a randomly chosen member outranks a randomly chosen
> non-member. An AUC of $0.5$ means random ordering; $1.0$ means perfect
> separation.
>
> The difficulty is that this question corresponds to no real adversary. Carlini
> et al.~\cite{carlini2022lira} put it this way: an attacker need not classify
> every sample correctly and does not care about average accuracy. What the
> attacker wants is to \textbf{identify at least some members with confidence}.
> An adversary who, across ten thousand sessions, can state with near certainty
> that fifty of them were in a specific operator's training set has violated
> fifty people's privacy, and guessing at random on the remaining 9,950 neither
> helps nor hurts that claim.
>
> AUC averages this away. It integrates over every threshold, weighting equally
> the high-false-positive regime --- where the attacker accuses half the dataset
> to catch half of it, which is not an attack --- and the low-false-positive
> regime, where the attacker accuses few samples and is right. An attack can
> therefore identify one per cent of members with certainty, a serious breach,
> while showing an AUC barely distinguishable from $0.5$, because that one per
> cent contributes almost nothing to the integral.
>
> \textbf{The right metric.} We therefore report true-positive rate at fixed low
> false-positive rates: $0.1\%$, $1\%$ and $5\%$. The question it asks is the one
> that matters: \emph{if the attacker accepts being wrong at most once in a
> hundred, how many members can be identified?} It measures behaviour in the tail
> of the score distribution, which is where the harm lives.
>
> Two points we keep explicit. The chance level for this metric is
> $\mathrm{TPR} = \mathrm{FPR}$, not $0.5$: an attacker guessing at random who
> accuses $1\%$ of samples catches $1\%$ of members. And the two metrics can
> diverge --- indeed the region where they diverge is where the attack is
> dangerous --- so reporting both is not redundancy.
>
> We nonetheless retain AUC for two practical reasons: it is the metric the
> empirical literature compares against, and the equivalence test supporting our
> null result is formulated on it.

---

## B. Perché white-box non vuol dire forte

### Italiano

> **Una gerarchia che non regge.** La letteratura sulla privacy ordina
> abitualmente gli attacchi per livello di accesso dell'avversario. Nel modello
> *black-box* l'attaccante vede solo l'output del modello; nel modello
> *white-box* vede i pesi, i gradienti, le attivazioni interne. L'intuizione è
> che il secondo sia più pericoloso, perché vede di più, e da questa intuizione
> discende una strategia di difesa diffusa: ridurre ciò che l'avversario può
> osservare, nascondere i gradienti, cifrare gli aggiornamenti.
>
> Il nostro apparato è costruito su una premessa diversa, e i risultati la
> sostengono. **Quando la loss di un campione è osservabile, la forza
> dell'attacco non dipende dal livello di accesso ma dalla qualità della
> funzione di punteggio.**
>
> **Perché la loss è speciale.** Un attacco di appartenenza deve rispondere a
> una domanda binaria: questo campione era nel training set? Tutto ciò che può
> aiutare a rispondere deve manifestarsi in qualche modo misurabile, e per un
> modello addestrato a minimizzare una funzione di perdita quella manifestazione
> è la perdita stessa. Un campione su cui il modello è stato addestrato è un
> campione su cui l'ottimizzatore ha spinto la loss verso il basso; un campione
> mai visto no. La memorizzazione, per definizione, è visibile nella loss.
>
> Sablayrolles et al. hanno reso questa intuizione un risultato formale.
> Mostrano che, sotto l'ipotesi che l'avversario conosca le distribuzioni della
> loss per i campioni dentro e fuori dal training, esiste una trasformazione
> della loss che è **Bayes-ottimale**: estrae tutta l'informazione disponibile
> sull'appartenenza. Non parte di essa, tutta. Ne segue che nessuna ulteriore
> osservazione — i gradienti per strato, le attivazioni intermedie, la
> traiettoria dell'ottimizzatore — può portare l'attaccante oltre quel limite.
>
> **Che ruolo ha allora l'accesso white-box?** Uno, e indiretto. L'attaccante
> non conosce davvero le due distribuzioni della loss: le stima, tipicamente
> addestrando modelli ombra. L'accesso privilegiato può renderlo migliore in
> quella stima. Ma sposta il limite solo nella misura in cui migliora la
> calibrazione, non perché i gradienti contengano un segnale di appartenenza
> aggiuntivo che la loss non contiene.
>
> **La conseguenza pratica, per chi progetta difese.** Nascondere i gradienti
> non protegge se la loss resta osservabile. Una difesa che riduce la visibilità
> dell'avversario senza ridurre la separabilità delle due distribuzioni di loss
> sposta il costo dell'attacco, non la sua possibilità: rende la stima più
> faticosa, non la conclusione meno raggiungibile. È una falsa sicurezza, e va
> distinta dalle difese che agiscono sulla separabilità — la differential
> privacy fra queste — che riducono il segnale invece di oscurarlo.
>
> **Come questo orienta la nostra misura.** Per questo concentriamo lo sforzo
> sperimentale sulla qualità dei valutatori anziché sulla moltiplicazione dei
> livelli di accesso: tre trasformazioni della loss, calibrazione per campione
> tramite modelli ombra, e controlli espliciti che lo scoring tratti membri e
> non-membri con la stessa formula. I nostri dati confermano che è la scelta
> giusta. LiRA è l'attacco con il privilegio maggiore fra quelli che eseguiamo,
> perché osserva l'aggiornamento del singolo client prima dell'aggregazione;
> eppure, quando la sua calibrazione degenera, non supera un semplice test di
> soglia sulla loss grezza. Il privilegio non basta se il punteggio è mal
> costruito.

### English

> \textbf{A hierarchy that does not hold.} The privacy literature routinely
> ranks attacks by adversary access level. Under \emph{black-box} the attacker
> sees only model outputs; under \emph{white-box} the attacker sees weights,
> gradients and internal activations. The intuition is that the latter is more
> dangerous because it sees more, and from that intuition follows a widespread
> defensive strategy: reduce what the adversary can observe, hide the gradients,
> encrypt the updates.
>
> Our apparatus rests on a different premise, and our results support it.
> \textbf{When a sample's loss is observable, attack strength is determined not
> by access level but by the quality of the scoring function.}
>
> \textbf{Why loss is special.} A membership attack must answer a binary
> question: was this sample in the training set? Anything that helps answer it
> must show up in some measurable quantity, and for a model trained to minimise
> a loss function that quantity is the loss itself. A sample the model was
> trained on is a sample whose loss the optimiser pushed down; an unseen sample
> is not. Memorisation, by definition, is visible in the loss.
>
> Sablayrolles et al. turned this intuition into a formal result. They show that,
> assuming the adversary knows the loss distributions for in-training and
> out-of-training samples, there exists a transformation of the loss that is
> \textbf{Bayes-optimal}: it extracts all available membership information. Not
> part of it --- all of it. It follows that no further observation --- per-layer
> gradients, intermediate activations, the optimiser trajectory --- can take the
> attacker past that bound.
>
> \textbf{What role does white-box access play, then?} One, and it is indirect.
> The attacker does not truly know the two loss distributions; they are
> estimated, typically by training shadow models. Privileged access can make
> that estimate better. But it moves the bound only insofar as it improves
> calibration, not because gradients carry membership signal that the loss does
> not.
>
> \textbf{The practical consequence, for defence designers.} Hiding gradients
> does not protect if the loss remains observable. A defence that reduces
> adversary visibility without reducing the separability of the two loss
> distributions shifts the cost of the attack rather than its feasibility: it
> makes estimation more laborious, not the conclusion less reachable. That is
> false assurance, and it must be distinguished from defences that act on
> separability --- differential privacy among them --- which reduce the signal
> rather than obscure it.
>
> \textbf{How this shapes our measurement.} We therefore concentrate
> experimental effort on evaluator quality rather than on multiplying access
> levels: three loss transformations, per-example calibration via shadow models,
> and explicit checks that scoring treats members and non-members under the same
> formula. Our data confirm the choice. LiRA is the most privileged attack we
> run, observing an individual client's update before aggregation; yet when its
> calibration degenerates it fails to beat a plain threshold test on raw loss.
> Privilege does not suffice when the score is poorly constructed.

---

## Nota su dove collocarle

La sottosezione A va in §5 accanto alla descrizione delle metriche, prima dei
risultati. La B va **subito dopo il threat model**, perché riformula cosa
significhi "avversario forte" e il lettore deve averla in mente prima di
leggere che LiRA è l'attacco primario.

Le due si richiamano: A dice che la misura giusta guarda la coda, B dice che
la coda dipende dalla calibrazione e non dall'accesso. Insieme giustificano
perché il §6 sulla validazione dello strumento occupa tanto spazio.

## Citazione mancante

Sablayrolles et al. non è ancora in `references.bib` e viene ora citato due
volte. Verificare autori e sede prima dell'uso: a memoria è *"White-box vs
Black-box: Bayes Optimal Strategies for Membership Inference"*, ICML 2019, di
Sablayrolles, Douze, Schmid, Ollivier, Jégou. Il titolo è notevole perché è
esattamente la tesi di questa sottosezione.
