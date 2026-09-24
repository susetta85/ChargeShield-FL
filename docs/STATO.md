# ChargeShield-FL — Stato del progetto

> **Stato: documento CANONICO.** Aggiornato il 2026-09-24 (chiusura di E-A). Solo numeri
> presenti in `risultati/` o letti dai JSON grezzi alla data, e lo dice dove.
> Linea scientifica: la guida. Cosa fare: `ESPERIMENTI.md`. Cosa esiste nel codice:
> `SISTEMA.md`. Questo file risponde a una sola domanda: a che punto siamo.

## 1. In una frase

RQ1 non ha ancora una risposta; per il punto operativo contano ora due decisioni più
che nuove run.
In media nessun attacco distingue membri da non membri, con o senza DP; l'unico
segnale è per singolo record senza DP. Lo sweep di ε in 2-16 (E-A, 19 run valide su
20) mostra che a ε ≤ 8 il costo supera la soglia di 3 volte, mentre a ε = 16 la loss
sta a 2.3 volte il riferimento no-DP della campagna ma a 3.7 volte quello appaiato per
seed (`nodp-sweep2`): se ε = 16 sia un punto operativo dipende da quale riferimento
vale (sezione 3.8), e poi dall'analisi per record E-C, ferma sulla segnalazione 6.

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

### 3.1b Sweep di ε, client-level `dp-fedavg` (A1): E-A

Fonte: i JSON di `experiments/rq1-eps{2,4,8,16}/` letti il 2026-09-24 (una run per
seed, seed 42, 123, 456, 789, 1234), coerenti con `check_significance.py` (stessi n e
stesse medie per gli attacchi) e con il foglio `Utility_privacy_limite` per ε = 2 e 4.
Per ε = 8 e 16 quel foglio mescola la run pilota a un seed con le cinque di E-A (n = 6,
segnalazione 44): qui sono solo le cinque. Riferimento di costo: la cella no-DP della
campagna (49 run, loss 0.002693) e, appaiato per seed, `nodp-sweep2` (media 0.001676).

| ε | seed validi | loss finale media | rapporto su no-DP campagna | rapporto appaiato per seed, min-max | Yeom | Shadow | LiRA composto | TPR a FPR 1% |
|---|---|---|---|---|---|---|---|---|
| 16 | 5 | 0.00614 | 2.3 | 3.1-4.2 | 0.5015 | 0.5011 | 0.4992 | 0.0097 |
| 8 | 5 | 0.0655 | 24.3 | 8.4-202 | 0.5001 | 0.5002 | 0.4978 | 0.0101 |
| 4 | 5 | 0.128 | 47.5 | 29.5-260 | 0.4999 | 0.5002 | 0.4959 | 0.0092 |
| 2 | 4 (loss su 5) | 0.285 | 105.8 | 120-422 | 0.5000 | 0.5001 | 0.4952 | 0.0094 |

Il TOST a margine 0.02 è soddisfatto in tutte e quattro le celle. La loss finale varia
fra seed di un fattore 1.9 a ε = 16 (0.0039-0.0074), 2 a ε = 2, 4.5 a ε = 4 e 13.8 a
ε = 8 (0.014-0.195), con il seed 456 sempre il più alto a ε ≤ 8: la cella ε = 8 è
incoerente fra seed, caso previsto dalla tabella di lettura di `ESPERIMENTI.md` (seed
aggiuntivi mirati su quella cella sola). A ε = 2 il seed 789 ha
tutti e tre gli attacchi falliti per un errore d'ambiente (segnalazione 42): da
rilanciare. I JSON di E-A registrano `git_commit`; il codice è lo stesso in tutte le
run (f145b79 e poi ed0e1f9, che cambia solo il paper).

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
`decision_matrix_ACN_membership_DP.xlsx`; le decisioni di `ESPERIMENTI.md` sezione 0
sono state congelate il 2026-09-22, prima di leggere E-A. Resta ambiguo un punto che
ora decide RQ1: "loss finale entro 3 volte il riferimento no-DP" non dice quale
riferimento. I valori pilota citati in quella riga (2.6 e 4.6) usano la cella no-DP
della campagna, 49 run di sweep diversi; il protocollo appaiato per seed della guida
(sezione 8) porterebbe a `nodp-sweep2`. Con il primo ε = 16 è a costo accettabile
(2.3), con il secondo no (3.7). È una decisione del supervisore, da prendere prima di
leggere E-C a ε = 16. Manca il braccio B1 della Fase B, clipping senza rumore, per cui
non esiste un flag.

## 4. Cosa manca

L'elenco ordinato, con comandi e prerequisiti, è `ESPERIMENTI.md`. E-A è eseguito
(sezione 3.1b) salvo il rilancio di ε = 2, seed 789. Il braccio no-DP di E-D (RQ2, 10
run) è stato eseguito su una seconda macchina e va copiato in `experiments/` di
questa prima di rigenerare le matrici. Restano: la decisione sul riferimento di costo
(sezione 3.8), E-C analisi per record dopo la segnalazione 6, E-B record-level su dati
naturali, canary bilanciato su Caltech, E-E per RQ3, il braccio DP di E-D, rianalisi
NVFlare al punto operativo. I bug che toccano i numeri sono in
`Segnalazioni_tecniche_2026-09-22.md`, punti 1, 4, 5, 6, 35, 36, 37, 38, 42, 44. Fuori dal paper, come infrastruttura o lavoro futuro: ML Plane,
Privacy Auditor, PES, ByzantineDetector, FedMIA-gradient, secondo dataset. Per le
frasi da non scrivere senza evidenza: guida, sezione 11.
