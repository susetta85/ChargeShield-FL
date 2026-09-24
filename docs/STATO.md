# ChargeShield-FL — Stato del progetto

> **Stato: documento CANONICO.** Aggiornato il 2026-09-24 (revisione: costo sul modello
> rilasciato, analisi per record corretta, E-C, segnalazioni 46-53, controlli del protocollo, celle delle matrici ricomposte). Solo numeri
> presenti in `risultati/` o letti dai JSON grezzi alla data, e lo dice dove.
> Linea scientifica: la guida. Cosa fare: `ESPERIMENTI.md`. Cosa esiste nel codice:
> `SISTEMA.md`. Questo file risponde a una sola domanda: a che punto siamo.

## 1. In una frase

Nel regime naturale non c'è un'esposizione misurabile da ridurre: in media nessun
attacco distingue membri da non membri, con o senza DP, e il test appaiato per record
non trova segnale in nessuna delle 8 celle analizzate (sezione 3.2). L'eccesso per
record senza DP, letto finora come l'unico segnale, compare identico fra i non membri:
è stabilità del ranking, non appartenenza (segnalazione 47). Il costo, misurato ora sul
modello globale rilasciato (segnalazione 46), supera la soglia di 3 volte in ogni cella
client-level: a ε = 16 è 60 volte il no-DP appaiato. Per il client-level nessuna cella
è un punto operativo; i prossimi passi sono E-B, la ricerca di un punto operativo con C
più basso o ε più alti, e il canary su più siti (`ESPERIMENTI.md`).

## 2. Glossario minimo

I termini che non si leggono da soli. La descrizione tecnica completa è in
`SISTEMA.md`, sezioni 4 e 7.

| termine | significato |
|---|---|
| client-level / record-level | due meccanismi DP diversi. Il primo protegge il contributo del client e il suo ε è un parametro di calibrazione; il secondo è DP-SGD e protegge il record, la stessa unità che gli attacchi misurano. Si attivano uno alla volta; il record-level richiede `--no-dp`. |
| A0, A1, A2, A3 | cosa vede l'avversario sull'update del client: grezzo senza DP; grezzo prima di clip e rumore (dp-fedavg); clippato (central); clippato e rumorizzato (local). Ordinati per informazione. Solo per il client-level. |
| loss grezza | errore di ricostruzione senza calibrazione. È la metrica che regge; LiRA calibrato è degenere (3.4). |
| record segnalati, z | sessioni membro in almeno 2 seed con percentile medio ≥ 90 e minimo ≥ 75; z misura quante deviazioni standard il conteggio dista dal livello di caso ottenuto per permutazione dentro il seed. Misura la stabilità del ranking, non l'appartenenza (segnalazione 47). |
| test appaiato | per le sessioni membro in un seed e non membro in un altro, differenza fra il percentile medio da membro e da non membro; z = media / errore standard. È la lettura primaria per record dal 2026-09-24. |
| loss sull'holdout del modello rilasciato / loss locale | la prima è l'MSE del modello globale finale sull'holdout ed è il costo; la seconda è la loss di addestramento locale dell'ultimo round, solo diagnostica (segnalazione 46). |
| regime naturale / canary | dati come sono / memorizzazione indotta con template duplicati, capacità 3.3x, feature temporale quasi univoca, 1000 epoche. Il canary valida lo strumento, non certifica il null naturale. |

## 3. Cosa è stato fatto e regge

### 3.1 Campagna principale, regime naturale, client-level

Fonte: `risultati/Matrice_sintesi.xlsx`, foglio `Utility_privacy_limite`, rigenerato il
2026-09-24 con il costo corretto (segnalazione 46) e con la composizione delle celle di
`check_significance.py` (segnalazioni 44, 45, 53): una run per seed, la più recente,
cartelle `_*`, NVFlare, entity-split e fedmia-gradient escluse. Il riferimento no-DP sono
le 5 run di `nodp-sweep2` (holdout 0.00158); prima della correzione erano 49 run, fra cui
le calibrazioni di overfitting, con holdout 0.00244 e LiRA composto 0.513, e tutti i
rapporti risultavano più bassi. AUC-ROC media sulle run; `n` = run nella cella. Le colonne
di costo sono rapporti sul no-DP: la prima sul modello rilasciato (il costo), la seconda
sulla loss locale (diagnostica).

| cella | n | costo: holdout del modello rilasciato | loss locale (diagnostica) | LiRA composto | Yeom | Shadow | TPR a FPR 1% |
|---|---|---|---|---|---|---|---|
| no-DP (A0) | 5 | 1 | 1 | 0.5004 | 0.5003 | 0.5016 | 0.0105 |
| dp-fedavg ε=16 (A1) | 5 | 59.7 | 3.7 | 0.4992 | 0.5015 | 0.5011 | 0.0097 |
| dp-fedavg ε=8 | 5 | 119.9 | 39.1 | 0.4978 | 0.5001 | 0.5002 | 0.0101 |
| dp-fedavg ε=4 | 5 | 188.4 | 76.2 | 0.4959 | 0.4999 | 0.5002 | 0.0092 |
| dp-fedavg ε=2 | 5 | 230.6 | 169.9 | 0.4957 | 0.5004 | 0.5005 | 0.0096 |
| dp-fedavg ε=1 | 5 | 249.1 | 241.1 | 0.4971 | 0.5002 | 0.5000 | 0.0099 |
| dp-fedavg ε=0.5 | 5 | 257.2 | 252.2 | 0.4980 | 0.4998 | 0.4993 | 0.0095 |
| dp-fedavg ε=0.1 | 5 | 266.8 | 251.9 | 0.4996 | 0.4995 | 0.4994 | 0.0089 |
| central ε=1 (A2) | 5 | 261.5 | 238.1 | 0.5024 | 0.5006 | 0.5001 | 0.0102 |
| central ε=0.5 | 5 | 278.6 | 284.1 | 0.5003 | 0.5002 | 0.4996 | 0.0096 |
| central ε=0.1 | 5 | 274.9 | 287.1 | 0.5020 | 0.5000 | 0.4992 | 0.0105 |
| local ε=1 (A3) | 5 | 249.1 | 241.1 | 0.4983 | 0.5002 | 0.5000 | 0.0108 |
| local ε=0.5 | 5 | 257.2 | 252.2 | 0.4975 | 0.4998 | 0.4993 | 0.0101 |
| local ε=0.1 | 5 | 266.8 | 251.9 | 0.4986 | 0.4995 | 0.4994 | 0.0092 |

Tre cose da tenere insieme leggendo la tabella. Il nullo aggregato vale anche
senza DP. In ogni cella client-level il modello rilasciato costa almeno 60 volte il
riferimento, quindi il nullo sotto DP non distingue "la DP protegge" da "non c'è nulla
da proteggere"; fino al 2026-09-24 la colonna di costo usava la loss locale e faceva
sembrare ε = 16 e 8 celle con utility residua. Nota storica che
conta per la Tabella 2 del paper: dp-fedavg e local hanno coinciso bit a bit fino
al 15 settembre perché eseguivano lo stesso codice; da allora dp-fedavg attacca
l'update grezzo (A1), e quali celle siano post modifica va verificato nel
registro. Il test di equivalenza TOST a margine 0.02 è soddisfatto ovunque, calcolato
però sul LiRA composto e non sulla metrica primaria (segnalazione 50); il Wilcoxon a 5
seed non può essere significativo per costruzione.

### 3.1b Sweep di ε, client-level `dp-fedavg` (A1): E-A

Fonte: i JSON di `experiments/rq1-eps{2,4,8,16}/` e `nodp-sweep2` letti il 2026-09-24,
una run per seed (42, 123, 456, 789, 1234), la più recente: per ε = 2 seed 789 è il
rilancio del 24 settembre, con gli attacchi e la stessa loss del primo tentativo bit per
bit. Costo = loss sull'holdout del modello globale rilasciato al round 10
(segnalazione 46), rapportata a `nodp-sweep2` (media 0.00158), sulla media e appaiata
per seed. La loss locale è la vecchia colonna, tenuta come diagnostica.

| ε | seed | holdout del modello rilasciato | rapporto su no-DP | appaiato, min-max | loss locale | Yeom | Shadow | LiRA composto | TPR a FPR 1% |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 5 | 0.0945 | 60 | 30-157 | 0.00614 | 0.5015 | 0.5011 | 0.4992 | 0.0097 |
| 8 | 5 | 0.190 | 120 | 67-490 | 0.0655 | 0.5001 | 0.5002 | 0.4978 | 0.0101 |
| 4 | 5 | 0.299 | 188 | 131-490 | 0.128 | 0.4999 | 0.5002 | 0.4959 | 0.0092 |
| 2 | 5 | 0.366 | 231 | 153-495 | 0.285 | 0.5004 | 0.5005 | 0.4957 | 0.0096 |

Nessuna cella sta sotto la soglia di 3 volte: è la terza riga della tabella di lettura
di `ESPERIMENTI.md` ("nessun punto operativo utile per il client-level in questo
regime"). Sull'holdout la variabilità fra seed è di circa 3 volte in ogni cella (a ε = 8
0.109-0.341): l'incoerenza di ε = 8, che veniva dalla loss locale (0.014-0.195), non c'è
più e i seed aggiuntivi non servono. TOST soddisfatto in tutte e quattro le celle, sul LiRA
composto (segnalazione 50). I JSON registrano `git_commit`: f145b79 ed ed0e1f9, e
478d471 per il rilancio; il codice di training e attacco è lo stesso.

### 3.1c Controlli del protocollo (segnalazione 48)

Fonte: `logs/ctrl_riproduzione_s42.log` e il JSON di `experiments/_ctrl_common_init/`
(commit `6d31e21`), letti il 2026-09-24, confrontati con `nodp-sweep2`; no-DP, seed 42.
La run ha un'etichetta di cella propria ("no-DP baseline, init comune") e, come
cartella `_*`, resta fuori dalle celle delle matrici.

| | holdout al round 1 | holdout al round 10 | Yeom | LiRA composto | TPR a FPR 1% |
|---|---|---|---|---|---|
| `nodp-sweep2`, seed 42 | 0.0651 | 0.00219 | 0.4988 | 0.5018 | 0.0117 |
| stessi, 5 seed (min-max) | | 0.00070-0.00219 | 0.4975-0.5039 | 0.4961-0.5036 | 0.0086-0.0117 |
| inizializzazione comune, seed 42 | 0.0080 | 0.00108 | 0.4989 | 0.4991 | 0.0099 |

Il codice attuale riproduce `nodp-sweep2` (controllo (a): loss di addestramento uguale
alla sesta cifra nei 10 round). Con l'inizializzazione comune il danno del round 1
sparisce e al round 10 il modello e' migliore, ma dentro la variabilita' fra seed;
gli attacchi restano al caso. Le campagne restano valide, lo scostamento dal protocollo
standard si dichiara. Norme dei delta dal round 3: 0.16-0.29 (caltech, jpl) e
0.51-0.59 (office1), sotto C = 1 (`ESPERIMENTI.md`, punto operativo).

### 3.2 Analisi per record: nessun segnale di appartenenza (E-C)

Fonte: `risultati/worst_case/livello_di_caso.json`, rigenerato il 2026-09-24 con il
controllo sui non membri e il test appaiato (segnalazione 47). 5 seed per cella, 17 561
sessioni membro in almeno 2 seed, 200 permutazioni; test appaiato su 28 129 sessioni
membro in un seed e non membro in un altro.

| cella | segnalati (membri) | attesi | z membri | z non membri | test appaiato | z appaiato |
|---|---|---|---|---|---|---|
| no-DP | 412 | 289.9 | 7.56 | 6.85 | +0.06 ± 0.22 | 0.26 |
| dp-fedavg ε=16 | 326 | 290.4 | 2.12 | 0.22 | +0.07 ± 0.22 | 0.31 |
| dp-fedavg ε=8 | 293 | 291.5 | 0.09 | 3.45 | +0.12 ± 0.22 | 0.56 |
| dp-fedavg ε=4 | 355 | 288.9 | 4.31 | 3.86 | −0.31 ± 0.22 | −1.38 |
| dp-fedavg ε=2 | 337 | 290.3 | 2.88 | 2.71 | −0.30 ± 0.22 | −1.34 |
| dp-fedavg ε=1 | 290 | 289.8 | 0.02 | 0.60 | −0.21 ± 0.22 | −0.94 |
| central ε=1 | 299 | 289.9 | 0.62 | 1.91 | +0.20 ± 0.22 | 0.91 |
| local ε=1 | 291 | 289.8 | 0.07 | 1.53 | −0.29 ± 0.22 | −1.32 |

Il conteggio sui membri, letto fino al 2026-09-24 come "l'unico segnale su dati
naturali", compare quasi uguale fra i non membri: senza DP 406 segnalati contro 412.
Misura quanto un modello ordina i record in modo coerente fra seed, cosa che la DP
distrugge insieme all'utility; non misura l'appartenenza. Il test appaiato non trova un
effetto dell'appartenenza in nessuna cella. La segnalazione 6 (percentile dei due
script, 411 contro 412) riguarda ora solo il conteggio, che non è più la lettura
primaria.

### 3.3 Lo strumento è validato, ma su un sito

Fonte: `CanaryPositiveControl.md`, sezione 6, campagna bilanciata dopo la
correzione del 16 settembre. Office 1, 20 template membro e 20 non membro, 30
duplicati, scambio dei ruoli, baseline a inizializzazione casuale, 5 seed, regime
canary.

| metrica, AUC media sui 3 round | braccio A | braccio B |
|---|---|---|
| loss grezza | 0.745 | 0.715 |
| LiRA calibrato | 0.651 | 0.696 |

Trenta round su trenta sopra 0.5 sulla loss grezza, tutte le somme A+B sopra 1,
Δ medio per seed +0.23 con t(4) = 9.9: il segnale è simmetrico allo scambio dei
ruoli, quindi è appartenenza. Tre artefatti intercettati dai controlli prima di
arrivarci: pool diversi per i due gruppi, scoring asimmetrico fra le classi,
effetto dell'assegnazione sotto DP. Caltech e JPL hanno solo run a seed 42 senza
il protocollo bilanciato; il config per ChargePlace Scotland non è mai stato
eseguito. Il paper riporta ancora i numeri precedenti alla correzione
(segnalazione 35).

### 3.4 LiRA è degenere

Il floor di varianza è colpito nel 97-99% dei record, il controllo di simmetria
dello scoring su central dà 0.21-0.34 contro uno 0.50 riportato, e il punteggio
non è riproducibile a seed fisso mentre la loss grezza lo è. Per questo la loss
grezza è la metrica primaria e lo scorer resta congelato finché lo sweep non è
chiuso (`ESPERIMENTI.md`, regole).

### 3.5 Record-level DP

Implementato il 16 settembre, usato solo in regime canary a Office 1: a σ = 5 la
loss grezza dei canary scende da 0.68 a 0.53 con il modello ancora addestrabile (loss
sull'holdout del modello rilasciato 0.006, simile alla loss locale). Un seed, nessun
braccio swap. L'ε di 7.15 è corretto sotto l'ipotesi di campionamento di Poisson (3
round, n = 1924: segnalazione 4, rettifica); il training usa shuffle, e il limite valido
con lo shuffle, adiacenza per sostituzione e senza amplificazione, è 343. Su dati
naturali non esiste ancora nessuna run valida: la prova del 24 settembre è stata fermata
perché gli attacchi saltavano (segnalazione 49).

### 3.6 Infrastruttura

Simulazione e deployment NVFlare a 5 container convergono sullo stesso nullo a
ε = 1. Venticinque dump NVFlare a ε ≤ 1 attendono rianalisi, rinviata perché nel
regime con utility distrutta; la rianalisi precedente era invalida per il seed
dello split (segnalazione 1). Privacy Auditor: contatore di budget, telemetria
identica fra cella con memorizzazione e senza. ByzantineDetector: mai eseguito
end-to-end. Nessuno dei due entra nel paper come contributo.

### 3.7 Unità protetta e persona

Identificativo utente presente nel 72.6% delle sessioni: JPL 93.5%, Caltech
52.1%, Office 1 34.8% (`risultati/decision_matrix_ACN_membership_DP.xlsx`,
misura del 2026-09-21; il paper li ha invertiti, segnalazione 36). 1 028 utenti,
mediana 14 sessioni, 8.5% multi-sito. Né record-level né client-level limitano il
contributo di una persona.

### 3.8 La scheda scientifica

I campi della sezione 5 della guida sono compilati nella stessa scheda
`decision_matrix_ACN_membership_DP.xlsx`; le decisioni di `ESPERIMENTI.md` sezione 0
sono state congelate il 2026-09-22, prima di leggere E-A. L'ambiguità sul riferimento
di costo (campagna o `nodp-sweep2`) non decide più nulla: con la loss del modello
rilasciato ε = 16 sta a 47 volte il primo e 60 volte il secondo (segnalazione 46). Due
punti del §0 sono cambiati il 2026-09-24 e sono documentati lì come deviazioni: il
costo, per un errore di implementazione (la definizione era giusta), e la metrica
primaria per record, che va sostituita dal test appaiato (segnalazione 47): la seconda
è una decisione del supervisore. Manca il braccio B1 della Fase B, clipping senza
rumore, per cui non esiste un flag.

## 4. Cosa manca

L'elenco ordinato, con comandi e prerequisiti, è `ESPERIMENTI.md`. E-A è chiuso (sezione
3.1b), E-C è eseguito sulle 8 celle esistenti (sezione 3.2). Il braccio no-DP di E-D
(RQ2, 10 run, seconda macchina) e quello di E-E (RQ3, in corso su una terza macchina)
vanno tenuti fuori da `experiments/` finché la segnalazione 45 non è corretta. I controlli
del protocollo sono eseguiti (sezione 3.1c). Restano: E-B record-level su dati naturali
(prova da rifare dopo la segnalazione 49), la griglia del punto operativo client-level
(8 run a un seed, config pronti), il canary su più siti, il braccio DP di E-D ed E-E, la
rianalisi NVFlare. Da decidere col supervisore: la metrica primaria per record (segnalazione 47) e se
le campagne future usano l'inizializzazione comune (segnalazione 48). I bug che toccano i numeri sono in
`Segnalazioni_tecniche_2026-09-22.md`, punti 1, 5, 6, 9, 35, 36, 38, 45, 48, 50, 51. Fuori dal paper, come infrastruttura o lavoro futuro: ML Plane,
Privacy Auditor, PES, ByzantineDetector, FedMIA-gradient, secondo dataset. Per le
frasi da non scrivere senza evidenza: guida, sezione 11.
