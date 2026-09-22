# ChargeShield-FL — Stato del progetto

> **Stato: documento CANONICO.** Aggiornato il 2026-09-22. Solo numeri presenti in
> `risultati/` alla data; dove un numero non è verificabile dai JSON grezzi lo dice.
> Linea scientifica: la guida. Cosa fare: `ESPERIMENTI.md`. Cosa esiste nel codice:
> `SISTEMA.md`. Questo file risponde a una sola domanda: a che punto siamo.

## 1. In una frase

RQ1 non ha ancora una risposta: in media nessun attacco distingue membri da non
membri nemmeno senza DP, l'unico segnale è per singolo record e solo senza DP, e
tutte le celle con DP finora eseguite hanno il modello distrutto. Il prossimo
esperimento, lo sweep di ε in 2-16, è l'unico che può cambiare questa frase.

## 2. Glossario minimo

I termini che non si leggono da soli. La descrizione tecnica completa è in
`SISTEMA.md`, sezioni 4 e 7.

| termine | significato |
|---|---|
| client-level / record-level | due meccanismi DP diversi. Il primo protegge il contributo del client e il suo ε è un parametro di calibrazione; il secondo è DP-SGD e protegge il record, la stessa unità che gli attacchi misurano. Si attivano uno alla volta; il record-level richiede `--no-dp`. |
| A0, A1, A2, A3 | cosa vede l'avversario sull'update del client: grezzo senza DP; grezzo prima di clip e rumore (dp-fedavg); clippato (central); clippato e rumorizzato (local). Ordinati per informazione. Solo per il client-level. |
| loss grezza | errore di ricostruzione senza calibrazione. È la metrica che regge; LiRA calibrato è degenere (3.4). |
| record segnalati, z | sessioni membro in almeno 2 seed con percentile medio ≥ 90 e minimo ≥ 75; z misura quante deviazioni standard il conteggio dista dal livello di caso ottenuto per permutazione dentro il seed. Non misura la gravità. |
| regime naturale / canary | dati come sono / memorizzazione indotta con template duplicati, capacità 3.3x, feature temporale quasi univoca, 1000 epoche. Il canary valida lo strumento, non certifica il null naturale. |

## 3. Cosa è stato fatto e regge

### 3.1 Campagna principale, regime naturale, client-level

Fonte: `risultati/Matrice_sintesi.xlsx`, foglio `Utility_privacy_limite`. AUC-ROC
media sulle run; `n` = run nella cella.

| cella | n | loss finale, rapporto su no-DP | LiRA composto | Yeom | Shadow | TPR a FPR 1% |
|---|---|---|---|---|---|---|
| no-DP (A0) | 49 | 1 | 0.513 | 0.503 | 0.501 | 0.0098 |
| dp-fedavg ε=16 (A1) | 1 | 2.6 | 0.503 | 0.499 | 0.499 | 0 |
| dp-fedavg ε=8 (A1) | 1 | 4.6 | 0.499 | 0.498 | 0.500 | 0 |
| dp-fedavg ε=1 (A1) | 14 | 128 | 0.498 | 0.500 | 0.500 | 0.0104 |
| dp-fedavg ε=0.5 | 10 | 157 | 0.498 | 0.500 | 0.499 | 0.0098 |
| dp-fedavg ε=0.1 | 10 | 157 | 0.499 | 0.500 | 0.499 | 0.0090 |
| central ε=1 (A2) | 14 | 108 | 0.503 | 0.500 | 0.499 | 0.0103 |
| central ε=0.5 | 9 | 176 | 0.502 | 0.499 | 0.499 | 0.0102 |
| central ε=0.1 | 5 | 179 | 0.502 | 0.500 | 0.499 | 0.0105 |
| local ε=1 (A3) | 11 | 140 | 0.498 | 0.500 | 0.500 | 0.0108 |
| local ε=0.5 | 6 | 157 | 0.498 | 0.500 | 0.499 | 0.0102 |
| local ε=0.1 | 6 | 154 | 0.499 | 0.500 | 0.500 | 0.0093 |

Tre cose da tenere insieme leggendo la tabella. Il nullo aggregato vale anche
senza DP. In ogni cella a ε ≤ 1 la loss è oltre 100 volte il riferimento, quindi
quel nullo non distingue "la DP protegge" da "non c'è nulla da proteggere". Le
sole celle con utility residua, ε = 8 e 16, hanno un seed solo. Nota storica che
conta per la Tabella 2 del paper: dp-fedavg e local hanno coinciso bit a bit fino
al 15 settembre perché eseguivano lo stesso codice; da allora dp-fedavg attacca
l'update grezzo (A1), e quali celle siano post modifica va verificato nel
registro. Il test di equivalenza TOST a margine 0.02 è soddisfatto ovunque; il
Wilcoxon a 5 seed non può essere significativo per costruzione.

### 3.2 Vulnerabilità per record: l'unico segnale su dati naturali

Fonte: `risultati/worst_case/livello_di_caso.json`, 5 seed, 17 561 sessioni
membro in almeno 2 seed, 200 permutazioni.

| cella | segnalati | attesi per caso | z |
|---|---|---|---|
| no-DP | 412 | 289.9 | 7.56 |
| dp-fedavg ε=1 | 290 | 289.8 | 0.02 |
| central ε=1 | 299 | 289.9 | 0.62 |
| local ε=1 | 291 | 289.8 | 0.07 |

Senza DP circa 122 sessioni in eccesso sul caso sono riconoscibili in modo
sistematico mentre l'AUC aggregata della stessa cella è 0.51. Sotto DP l'eccesso
scompare, ma con utility distrutta: le due spiegazioni non sono separabili. Il
percentile è calcolato sui `composed_score` di LiRA, che nello stato attuale dello
scorer sono in pratica una funzione monotona della loss: il risultato regge come
classifica per loss e va descritto così. I due script che calcolano il percentile
danno 411 e 412 (segnalazione 6): da unificare prima di citare il numero.

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
loss grezza dei canary scende da 0.68 a 0.53 con il modello ancora addestrabile.
Un seed, nessun braccio swap, e l'ε dichiarato di 7.15 viene da un accountant che
conta un round su tre (segnalazione 4). Su dati naturali non esiste nessuna run.

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
`decision_matrix_ACN_membership_DP.xlsx`. Restano aperti, da congelare prima di
leggere lo sweep: superficie primaria e soglia di costo accettabile
(`ESPERIMENTI.md`, sezione 0). Manca il braccio B1 della Fase B, clipping senza
rumore, per cui non esiste un flag.

## 4. Cosa manca

L'elenco ordinato, con comandi e prerequisiti, è `ESPERIMENTI.md`: E-A sweep di ε,
E-B record-level su dati naturali, E-C analisi per record, canary bilanciato su
Caltech, E-E per RQ3, E-D per RQ2, rianalisi NVFlare al punto operativo. I bug che
toccano i numeri sono in `Segnalazioni_tecniche_2026-09-22.md`, punti 1, 4, 5, 6,
35, 36, 37, 38. Fuori dal paper, come infrastruttura o lavoro futuro: ML Plane,
Privacy Auditor, PES, ByzantineDetector, FedMIA-gradient, secondo dataset. Per le
frasi da non scrivere senza evidenza: guida, sezione 11.
