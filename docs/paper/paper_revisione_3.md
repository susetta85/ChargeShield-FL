# Revisione 3 — scorer, asse di forza, threat model, figura, pulizia codice

---

## 1. Perché TPR@FPR e non AUC (§5, da inserire prima dei risultati)

> **Metrica primaria.** Riportiamo il tasso di veri positivi a tassi di falsi
> positivi fissi (0.1%, 1%, 5%) accanto all'AUC, e trattiamo il primo come
> misura primaria per LiRA. La ragione è l'argomento centrale di Carlini et
> al.: l'AUC è una media su tutte le soglie e pesa allo stesso modo il regime
> ad alto FPR, che non corrisponde ad alcun avversario reale. Un attaccante che
> volesse identificare con sicurezza anche pochi membri opera a FPR molto basso,
> e un attacco che identifichi con certezza l'1% dei membri è una violazione di
> privacy concreta pur lasciando l'AUC praticamente a 0.5.
>
> Le due metriche possono quindi divergere, ed è proprio nella regione in cui
> divergono che si colloca il danno. Riportare solo l'AUC significa mediare via
> la coda in cui l'attacco è pericoloso. Per lo stesso motivo il livello di caso
> per TPR@FPR è TPR = FPR e non 0.5, e lo verifichiamo esplicitamente.
>
> Conserviamo comunque l'AUC per due ragioni: è la metrica con cui la
> letteratura empirica confronta i risultati, e il test di equivalenza che
> sostiene il nostro risultato nullo è formulato su di essa.

---

## 2. Le tre funzioni di punteggio (§5, sottosezione nuova)

> **Tre scorer, un solo segnale.** Tutti gli attacchi che eseguiamo derivano il
> proprio giudizio dalla stessa quantità osservabile, l'errore di ricostruzione
> del campione sotto il modello bersaglio. Differiscono nella trasformazione che
> applicano a quella quantità prima di decidere, e implementiamo le tre che la
> letteratura distingue.
>
> *Raw loss.* L'errore quadratico medio non trasformato. L'assunzione è che un
> campione appartenuto al training venga ricostruito meglio. Il limite è noto e
> dominante: la loss di un campione dipende soprattutto da quanto quel campione
> è intrinsecamente difficile, non da se sia stato addestrato. Un record atipico
> ha loss alta comunque, e questo confonde appartenenza e difficoltà.
>
> *Log-transformed loss.* Il logaritmo della loss. Serve a stabilizzare la
> varianza: la distribuzione delle loss grezze è fortemente asimmetrica, con una
> coda lunga a destra, e LiRA assume invece che le loss dei modelli shadow siano
> approssimativamente gaussiane. Non è un'assunzione cosmetica, perché le formule
> di LiRA stimano media e varianza per ogni campione e su una distribuzione
> asimmetrica quelle stime non descrivono nulla. Sui nostri dati la violazione è
> misurata e grave: il test di Jarque-Bera sulle loss grezze restituisce valori
> di sei ordini di grandezza sopra la soglia, e la trasformazione logaritmica la
> riduce di quattro-cinque ordini.
>
> *Bayes-optimal score.* La trasformazione derivata da Sablayrolles et al., che
> mostrano come, sotto l'ipotesi che l'attaccante modelli correttamente le
> distribuzioni della loss dentro e fuori dal training, una specifica funzione
> della loss estragga tutta l'informazione disponibile. È un limite superiore
> teorico: nessuna ulteriore manipolazione della loss, e nessun accesso ai
> gradienti interni, può fare meglio di così.
>
> Implementarle tutte e tre non è ridondanza. È il modo in cui separiamo
> l'effetto della calibrazione da quello del segnale: se i tre scorer danno lo
> stesso risultato, il limite è nel dato; se divergono, il limite è nella
> trasformazione, e sappiamo quale.

---

## 3. L'asse di forza dell'attacco, formalizzato (§4, dopo il threat model)

Questo è il punto concettuale più forte della tua osservazione. Va scritto come
affermazione esplicita, non lasciato implicito.

> **Cosa rende forte un attacco.** La letteratura ordina spesso gli attacchi di
> appartenenza per livello di accesso dell'avversario, trattando il white-box
> come intrinsecamente più pericoloso del black-box perché legge i gradienti.
> Il nostro apparato è costruito su una premessa diversa, che i risultati
> sostengono: **quando la loss del campione è osservabile, il fattore che
> determina la forza dell'attacco non è il livello di accesso ma la scelta e la
> calibrazione della funzione di punteggio.**
>
> Il fondamento è il risultato di Sablayrolles et al.: sotto l'ipotesi che
> l'avversario modelli correttamente la distribuzione della loss, esiste una
> trasformazione della loss che è Bayes-ottimale, e nessun accesso ulteriore —
> gradienti per strato, attivazioni intermedie, traiettoria di addestramento —
> aggiunge informazione sull'appartenenza oltre quella. L'accesso white-box
> sposta il limite superiore solo nella misura in cui consente di stimare meglio
> quelle distribuzioni, non perché i gradienti contengano un segnale di
> appartenenza separato.
>
> Ne discendono due conseguenze pratiche, e la seconda è una raccomandazione di
> progetto.
>
> La prima riguarda la nostra misura: concentriamo lo sforzo sperimentale sulla
> qualità dei valutatori — tre trasformazioni della loss, calibrazione shadow
> per campione, controlli di simmetria dello scoring — anziché sulla
> moltiplicazione dei livelli di accesso. Un attacco mal calibrato con accesso
> completo è più debole di un attacco ben calibrato con accesso limitato, e i
> nostri dati lo mostrano: LiRA, che ha il privilegio maggiore, non supera la
> soglia sulla loss grezza quando la sua calibrazione degenera.
>
> La seconda riguarda chi progetta difese: **nascondere i gradienti non protegge
> se la loss resta osservabile.** Una difesa che riduce la visibilità
> dell'avversario senza ridurre la separabilità delle distribuzioni di loss
> sposta il costo dell'attacco, non la sua possibilità. È una falsa sicurezza,
> ed è il motivo per cui misuriamo la separabilità direttamente invece di
> dedurla dal livello di accesso concesso.

---

## 4. Threat model, riscritto (§4)

Accolgo la tua indicazione: fuori gli scenari non considerati, e Scenario 2
descritto per quello che è.

> **4.1 L'avversario.** Assumiamo un aggregatore honest-but-curious: segue
> fedelmente il protocollo di aggregazione, non invia mai pesi globali
> malformati e non devia dallo schema di partecipazione, ma registra e analizza
> passivamente ogni artefatto che il protocollo gli consegna. I client sono
> onesti per costruzione. È il modello standard per il cross-silo, dove i
> partecipanti sono organizzazioni identificabili legate da un accordo e il
> rischio realistico non è il sabotaggio ma la curiosità di chi ha accesso
> legittimo.
>
> **4.2 Cosa l'avversario osserva.** Il protocollo consegna all'aggregatore due
> artefatti distinti, e la distinzione determina cosa può concludere. Il
> **modello globale aggregato** è disponibile a chiunque partecipi, e un attacco
> che lo interroga può concludere soltanto che una sessione apparteneva al
> training set della federazione nel suo insieme. L'**aggiornamento del singolo
> client** prima dell'aggregazione è disponibile al solo aggregatore, e un
> attacco che lo interroga attribuisce l'appartenenza a uno specifico operatore.
> La seconda è la minaccia che motiva il lavoro, perché è quella che il
> federated learning promette implicitamente di escludere.
>
> **4.3 Tre punti di privilegio.** Per la superficie per-client, la collocazione
> del rumore determina cosa l'aggregatore vede, e le etichette di livello Purdue
> dell'ML Plane rendono la differenza esplicita. Sotto `dp-fedavg` riceve
> l'aggiornamento grezzo e applica esso stesso clipping e rumore, osservando
> quindi il valore pulito. Sotto `central` riceve aggiornamenti clippati ma non
> rumorizzati. Sotto `local` il valore pulito non lascia mai il livello 1.
> I tre casi ordinano l'avversario per privilegio decrescente, e il primo
> costituisce il limite superiore di ciò che è estraibile da questo canale.
>
> **4.4 Integrità: un consumatore diverso.** ChargeShield-FL include un
> rilevatore di anomalie di integrità che protegge la convergenza da client che
> deviano dal protocollo. Non appartiene al modello di minaccia di questo paper:
> opera su un avversario diverso — client malevolo anziché server curioso — e i
> risultati sulla privacy non ne dipendono in alcun modo. Lo riportiamo perché
> condivide il substrato di osservabilità con il Privacy Auditor, ed è la
> dimostrazione che l'ML Plane serve consumatori con modelli di minaccia
> disgiunti. Gli attacchi Byzantine che eseguiamo servono esclusivamente a
> validare quel componente. Nessun meccanismo di integrità è qui rivendicato
> come mitigazione dell'inferenza di appartenenza.

Scenari 3 e 4 rimossi, come proponi: erano sicurezza di rete convenzionale e
il loro elenco suggeriva completezza dove non serviva.

---

## 5. ML Plane e Auditor nell'introduzione (§1.2, dopo la lacuna Purdue)

> **Cosa introduciamo, e perché due componenti e non uno.** La lacuna descritta
> sopra è di osservabilità prima che di difesa: il flusso attraversa i livelli e
> nessun punto di ispezione lo esamina. Colmarla richiede due cose distinte, ed
> è la ragione per cui il framework ha due parti.
>
> Serve anzitutto un **punto di ispezione** dove il modello Purdue non ne
> prevede. L'ML Plane è quel punto: un substrato event-driven che intercetta tre
> momenti del ciclo di addestramento — quando un nodo produce un aggiornamento
> locale, quando quell'aggiornamento viene privatizzato, quando il round viene
> aggregato — e li espone etichettati col livello Purdue di provenienza. I tre
> momenti non sono una scelta implementativa: corrispondono ai tre punti in cui
> l'artefatto cambia di natura e di livello, e sono esattamente i tre punti di
> privilegio dell'avversario del §\ref{sec:threat-model}.
>
> Serve poi qualcosa che **guardi** attraverso quel punto. Qui la separazione
> conta: un monitor in linea può contabilizzare quanto budget è stato consumato
> e segnalare anomalie di sensibilità, ma non può dire quanta informazione un
> avversario estrarrebbe davvero — per saperlo bisogna instanziarlo. Il Privacy
> Auditor svolge il primo compito, in linea e a costo trascurabile; la suite di
> attacchi svolge il secondo, offline sugli stessi eventi raccolti. Teniamo i
> due nomi distinti per tutto il paper perché confonderli è precisamente
> l'errore che rende inaffidabile buona parte della letteratura empirica: il
> budget dichiarato non è il rischio misurato.

---

## 6. Guida alle sezioni (§1, in chiusura)

> Il resto del lavoro è organizzato così. Il §\ref{sec:related} colloca il
> lavoro rispetto alla letteratura empirica su DP e inferenza di appartenenza in
> federated learning, e rispetto alla definizione di privacy differenziale su cui
> poggia. Il §\ref{sec:threat-model} definisce l'avversario, le due superfici
> che osserva e i tre punti di privilegio, e stabilisce che cosa rende forte un
> attacco. Il §\ref{sec:attacks} descrive la suite che instanzia quell'avversario
> e le tre funzioni di punteggio. Il §\ref{sec:framework} presenta l'ML Plane e i
> suoi consumatori. Il §\ref{sec:setup} riporta dati, modello e protocollo
> sperimentale. Il §\ref{sec:validation} è la parte metodologica: i controlli che
> stabiliscono se la misura è leggibile. Il §\ref{sec:results} riporta i
> risultati sulle due superfici e il costo in utilità, il §\ref{sec:limitations}
> i limiti, il §\ref{sec:discussion} la caratterizzazione del confine e le
> conclusioni.

---

## 7. Figure

**Fig. 1 — il framework**, nel §\ref{sec:framework}. TikZ pronto sotto.
**Fig. 2 — ROC a bassi FPR**, nel §\ref{sec:results-surface-b}: curve ROC in
scala log-log con la diagonale del caso, che è il modo in cui Carlini et al.
mostrano il regime che conta. Serve `roc_curve_dump_dir`, che avete già.
**Fig. 3 — il piano rischio/utilità**, nel §\ref{sec:discussion}: ascissa ε in
scala logaritmica, ordinata doppia con canary AUC e loss di ricostruzione,
un punto per configurazione. È la figura che mostra la finestra operativa e
probabilmente la più citabile del paper.
**Fig. 4 — il controllo di scambio**, nel §\ref{sec:validation}: Δ per seed nei
due bracci, che rende visibile la simmetria in un colpo d'occhio.

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
  font=\footnotesize,
  node distance=4mm,
  box/.style={draw, rounded corners=2pt, minimum height=7mm,
              align=center, inner sep=3pt},
  prod/.style={box, fill=black!5,  text width=24mm},
  cons/.style={box, fill=black!12, text width=27mm},
  bus/.style={draw, fill=black!20, minimum width=52mm,
              minimum height=6mm, align=center},
  ev/.style={->, >=latex, thin},
]
% produttori (livello 1)
\node[prod] (train)  {Local trainer\\\tiny L1};
\node[prod, below=of train] (grad) {Gradient/DP\\manager \tiny L1$\rightarrow$L2};
\node[prod, below=of grad]  (agg)  {FedAvg\\aggregator \tiny L2};

% bus
\node[bus, right=14mm of grad] (plane) {\textbf{ML Plane}};

% consumatori
\node[cons, right=14mm of plane, yshift=11mm]  (aud)  {Privacy Auditor\\\tiny in-line, budget};
\node[cons, right=14mm of plane]               (ids)  {Byzantine detector\\\tiny in-line, integrity};
\node[cons, right=14mm of plane, yshift=-11mm] (atk)  {Attack suite\\\tiny offline, empirical};

% eventi
\draw[ev] (train) -- node[above, sloped, font=\tiny] {local update} (plane);
\draw[ev] (grad)  -- node[above, font=\tiny] {update privatised} (plane);
\draw[ev] (agg)   -- node[below, sloped, font=\tiny] {round aggregated} (plane);
\draw[ev] (plane) -- (aud);
\draw[ev] (plane) -- (ids);
\draw[ev] (plane) -- (atk);

% confine Purdue
\draw[dashed] ($(train.north west)+(-3mm,3mm)$) rectangle ($(agg.south east)+(3mm,-3mm)$);
\node[font=\tiny, above] at ($(train.north)+(0,4mm)$) {Purdue L1--L2};
\end{tikzpicture}
\caption{ChargeShield-FL. Tre produttori emettono tre tipi di evento,
etichettati col livello Purdue di provenienza; tre consumatori indipendenti dal
ciclo di addestramento li ricevono. I due in alto operano in linea, il terzo
offline sugli stessi eventi raccolti.}
\label{fig:framework}
\end{figure}
```

Richiede `\usetikzlibrary{calc,positioning}`.

---

## 8. Pulizia del codice: `run_fedmia()` è Yeom

La disambiguazione sul nome va tolta dal paper, e hai ragione che la via giusta
è correggere il codice invece di spiegare l'incoerenza. **Rinominare, non
cancellare**: la funzione è usata e produce i risultati della superficie
globale.

```bash
cd ~/Documents/ChargeShield-FL
git grep -n "run_fedmia\b" -- '*.py' | grep -v run_fedmia_gradient
```

Se le occorrenze sono contenute, il rename è meccanico:

```bash
python3 - <<'EOF'
import re, pathlib
for p in pathlib.Path('.').rglob('*.py'):
    if '.git' in str(p): continue
    s = p.read_text()
    # \b e negative lookahead: non tocca run_fedmia_gradient
    new = re.sub(r'\brun_fedmia\b(?!_gradient)', 'run_yeom_loss_attack', s)
    if new != s:
        p.write_text(new)
        print(p, s.count('run_fedmia') - s.count('run_fedmia_gradient'), 'occorrenze')
EOF
python3 -m py_compile scripts/run_experiments.py && echo OK
python3 -m pytest tests/ -q 2>&1 | tail -3
```

Stessa logica per `src/plugins/attacks/fedmia.py`, che implementa Yeom: va
rinominato in `yeom_loss.py`, a meno che `yeom.py` non sia già il suo wrapper —
verifica con `head -20 src/plugins/attacks/yeom.py`.

Non farei la copia `_old`: il repository ha già la storia git, e un file
duplicato in albero è una fonte di confusione in più. Se vuoi conservare
tracciabilità, basta il messaggio di commit.

Nel paper, al posto della disambiguazione, una nota nell'artifact: *«Prima dello
Sprint N la funzione che implementa l'attacco di Yeom si chiamava
`run_fedmia()`, per ragioni storiche. I risultati non ne sono affetti.»*
