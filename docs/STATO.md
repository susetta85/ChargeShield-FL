# ChargeShield-FL — Stato del progetto

> **Stato: documento CANONICO.** Aggiornato il 2026-09-29 (revisione: costo sul modello
> rilasciato, analisi per record corretta, E-C, segnalazioni 46-59, controlli del protocollo, celle delle matrici ricomposte, RQ2 senza DP,
> punto operativo a 5 seed, RQ2 con DP, E-B a σ = 1 e 2 con il riferimento GroupNorm, RQ3 senza DP; il 2026-09-29 E-B a σ = 5, RQ3 sul Mac principale con e senza DP, prove su mu e FedAvg con DP su Windows, dai JSON e non ancora nelle matrici). Solo numeri
> presenti in `risultati/` o letti dai JSON grezzi alla data, e lo dice dove.
> Linea scientifica: la guida. Cosa fare: `ESPERIMENTI.md`. Cosa esiste nel codice:
> `SISTEMA.md`. Questo file risponde a una sola domanda: a che punto siamo.

## 1. In una frase

Nel regime naturale non c'è un'esposizione misurabile da ridurre: in media nessun
attacco distingue membri da non membri, con o senza DP, e il test appaiato per record
non trova segnale in nessuna delle celle analizzate (sezioni 3.2 e 3.9). L'eccesso per
record senza DP, letto finora come l'unico segnale, compare identico fra i non membri:
è stabilità del ranking, non appartenenza (segnalazione 47). Il costo, misurato sul
modello globale rilasciato (segnalazione 46), supera la soglia di 3 volte in ogni cella
client-level: a ε = 16 è 60 volte il no-DP appaiato, e il punto operativo scelto con la
griglia, C = 1 con ε = 64, vale 4.5 volte a 5 seed (sezione 3.1b). La DP per record costa
1.9 volte a σ = 1 e 2.1 volte a σ = 2 rispetto al no-DP con la stessa normalizzazione
(GroupNorm), sotto la soglia, con un ε per record di circa 30 e 10, e 2.3 volte a σ = 5 con
ε di circa 3 (sezione 3.5). La partizione IID o per sito non cambia il
successo degli attacchi, con o senza DP (sezione 3.9). Senza DP FedAvg ha la loss
sull'holdout circa 3.7-3.9 volte più bassa di FedProx su due macchine; con DP a C = 1 l'ordine
si inverte in 4 seed su 5, non in modo significativo (sezione 3.10). I prossimi passi sono la
cella di solo clipping, lo screening di C per FedAvg con DP e il canary su più siti
(`ESPERIMENTI.md`).

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
| dp-fedavg ε=64 (punto operativo) | 5 | 4.5 | 1.6 | 0.5011 | 0.5004 | 0.5013 | 0.0105 |
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
senza DP. Nelle celle di E-A (ε ≤ 16 per round) il modello rilasciato costa almeno 60 volte il
riferimento, e al punto operativo ε = 64 in media 4.5 volte, sopra la soglia di 3 (sezione 3.1b):
il nullo sotto DP non distingue "la DP protegge" da "non c'è nulla
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

Punto operativo: griglia C ∈ {0.25, 0.5, 1} × ε ∈ {16, 64, 256}, seed 42 (JSON in
`experiments/_op_*` letti il 2026-09-25, fuori dalle matrici perché prove a un seed). Costo
in volte `nodp-sweep2`:

| C (righe), ε (colonne) | 16 | 64 | 256 |
|---|---|---|---|
| 0.25 | 11.9 | 10.2 | 6.9 |
| 0.5 | 8.8 | 5.3 | 2.6 |
| 1 | 41.6 (E-A) | 2.6 | 2.9 |

A parità di rumore C più grande costa meno: domina la distorsione del clipping. Tre celle
sotto 3 volte a un seed; il punto operativo per la regola era C = 1, ε = 64. **Su 5 seed non
regge**: cella `rq1-eps64` (commit 8d44ae3, nelle matrici), holdout 0.00712, 4.5 volte la media
del no-DP; per seed, rispetto al no-DP dello stesso seed, da 1.9 a 14.7 volte. Il seed 42 della
griglia era il più favorevole. ε = 32, seed 42: 7.3 volte. Attacchi al caso in tutte le celle,
test per record senza segnale anche a ε = 64 (sezione 3.2). È la terza riga della tabella di
lettura di `ESPERIMENTI.md`: nessun punto operativo utile per il client-level in questo regime,
con tre client.

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
| dp-fedavg ε=64 | 473 | 290.3 | 11.61 | 10.35 | +0.15 ± 0.22 | 0.69 |
| record-DP σ=1 | 415 | 289.0 | 7.21 | 9.39 | +0.04 ± 0.22 | 0.18 |
| record-DP σ=2 | 455 | 286.0 | 10.66 | 11.64 | −0.11 ± 0.22 | −0.50 |
| no-DP, GroupNorm | 458 | 289.9 | 10.12 | 8.98 | −0.20 ± 0.22 | −0.91 |

Nella cella record-DP le sessioni sono 17 389 e 27 877 invece di 17 561 e 28 129: il dump
per campione contiene i record che LiRA punteggia, e con record-DP sono circa 60 in meno per
seed (segnalazione 58).

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
naturali: la prova del 24 settembre sul Mac principale è arrivata in fondo con gli
attacchi saltati (segnalazione 49, JSON in `_prova_recorddp` da non usare); il rilancio
`_prova_recorddp_v2` sul secondo Mac esegue tutti gli attacchi. A σ = 1, seed 42, la loss
sull'holdout è 1.29 volte il no-DP della stessa macchina, gli attacchi sono al caso, l'ε
per record di Poisson è 30.2 (Office 1) e il limite con lo shuffle 1212 (JSON in
`experiments_altre_macchine/_prova_recorddp_v2`, riverificato il 2026-09-26). Il confronto col no-DP mescola rumore e cambio di
normalizzazione (GroupNorm).

**Campagna E-B, σ = 1, completa** (`experiments/rq1-recorddp-nm1`, Mac principale, 5 seed
dal 26 al 27 settembre, zero righe `[ERROR]` nel log; JSON letti il 2026-09-27; cella
"record-DP, nm=1.0" nei fogli `Utility_privacy_limite` e `Worst_case_per_record`, confronto
C16). Commit 0dedd34 (seed 123), 1519d16 con `-dirty` dovuto solo a documenti (seed 456,
segnalazione 57), 943c795 (789, 1234, 42): il codice di training e attacco è lo stesso.

| seed | holdout | volte lo stesso seed no-DP | Yeom, ultimo round | Shadow, media | LiRA composto | TPR a FPR 1% | ε Poisson, Office 1 |
|---|---|---|---|---|---|---|---|
| 42 | 0.00240 | 1.10 | 0.498 | 0.497 | 0.502 | 0.0104 | 30.2 |
| 123 | 0.00239 | 1.46 | 0.503 | 0.503 | 0.498 | 0.0100 | 30.4 |
| 456 | 0.00130 | 1.86 | 0.501 | 0.503 | 0.503 | 0.0081 | 31.0 |
| 789 | 0.00233 | 1.48 | 0.503 | 0.504 | 0.497 | 0.0085 | 30.4 |
| 1234 | 0.00247 | 1.35 | 0.501 | 0.504 | 0.503 | 0.0110 | 30.4 |

Loss sull'holdout media 0.00218, 1.4 volte la media di `nodp-sweep2` (1.43 in media
geometrica dei rapporti per seed, tutti sopra 1): sotto la soglia di 3 volte, dove nessuna
cella client-level arriva (la migliore, ε = 64, vale 4.5). Attacchi al caso, LiRA composto
equivalente a 0.5 col TOST (`check_significance.py`), test appaiato per record +0.04 ± 0.22,
z = 0.18 (sezione 3.2). ε per record di Poisson fra 30.2 e 31.0 a Office 1, 4.8 a JPL e 5.0 a
Caltech; con lo shuffle il limite valido è 1212, cioè nessuna garanzia formale utile.

**σ = 2 e riferimento no-DP con GroupNorm, completi il 2026-09-28** (`experiments/rq1-recorddp-nm2`
e `rq1-nodp-groupnorm`, Mac principale, 5 seed ciascuna, zero righe `[ERROR]`, commit puliti
4c0d0ae, fe22124, 42b4561; JSON letti il 2026-09-28; celle "record-DP, nm=2.0" e "no-DP
baseline, norm=group", confronti C17 e C15). Loss sull'holdout del modello rilasciato:

| seed | no-DP, BatchNorm (`nodp-sweep2`) | no-DP, GroupNorm | σ = 1 | σ = 2 | σ = 1 / GroupNorm | σ = 2 / GroupNorm |
|---|---|---|---|---|---|---|
| 42 | 0.00219 | 0.00109 | 0.00240 | 0.00244 | 2.19 | 2.23 |
| 123 | 0.00164 | 0.00153 | 0.00239 | 0.00293 | 1.56 | 1.92 |
| 456 | 0.00070 | 0.00092 | 0.00130 | 0.00137 | 1.41 | 1.49 |
| 789 | 0.00157 | 0.00086 | 0.00233 | 0.00264 | 2.71 | 3.07 |
| 1234 | 0.00184 | 0.00141 | 0.00247 | 0.00297 | 1.75 | 2.10 |
| media (rapporti fra le medie) | 0.00158 | 0.00116 | 0.00218 | 0.00247 | 1.87 | 2.12 |

Il passaggio a GroupNorm, senza DP, non costa: 0.73 volte BatchNorm sulle medie, non
significativo a 5 seed (t = −1.51 sul logaritmo). Il costo della DP per record va quindi letto
sul riferimento GroupNorm ed è più alto dell'1.4 letto su `nodp-sweep2`: 1.87 volte a σ = 1 e
2.10 a σ = 2 in media geometrica dei rapporti per seed, 5 seed su 5 sopra 1 (t = 5.30 e 6.37
sul logaritmo), ancora sotto la soglia di 3 volte con entrambi i riferimenti. Da σ = 1 a
σ = 2 il costo cresce di 1.12 volte, mentre l'ε per record di Poisson scende da circa 30 a
circa 9.7 a Office 1 (1.8 a JPL e Caltech; con lo shuffle 355). Attacchi al caso in tutte e
tre le celle, test per record senza segnale (sezione 3.2). Che il costo quasi non cambi col
rumore faceva pensare che venisse soprattutto dal clipping per esempio (C = 1); la cella con
clipping e σ = 0, in corso dal 2026-09-29, lo verifica. Riserve: gli shadow di LiRA non usano
DP-SGD (segnalazione 51); il confronto con il client-level è fra unità protette diverse a
costo comparabile, non allo stesso ε. Lettura per RQ1: a σ = 1 e 2 la DP per record ha un
costo accettabile e la DP per client no; in nessuna delle due il regime naturale offre un
segnale che la DP possa ridurre.

**σ = 5 e solo clipping, 2026-09-29.** σ = 5 è completo (quarta macchina, 5 seed, commit
64757fd pulito, `experiments_altre_macchine/rq1-recorddp-nm5`, JSON letti il 2026-09-29):
2.32 volte il riferimento GroupNorm in media geometrica dei rapporti per seed (da 1.82 a
3.27; t = 7.74 sul logaritmo) e 1.10 volte σ = 2 (da 1.01 a 1.22; t = 2.44, non
significativo a 4 gdl). L'ε per record di Poisson scende a 3.1 (massimo sui client, Office 1;
0.6 a Caltech e JPL), il limite valido con lo shuffle a 81. Yeom al caso (0.497-0.503). Il
seed 789 supera la soglia di 3 volte già a σ = 2 (3.07) e a σ = 5 (3.27): ha il riferimento
GroupNorm più basso dei cinque. Lettura: da σ = 1 a σ = 5 l'ε per record di Poisson scende
di circa dieci volte e il costo passa da 1.87 a 2.32 volte; la media resta sotto la soglia,
non tutti i seed. Il controllo a 2 round sulla quarta macchina riproduce il Mac principale
(`PROVENIENZA.txt`). La cella di solo clipping (σ = 0, `config/experiment_rq1_recorddp_nm0.yaml`)
è in corso sulla stessa macchina (seed 42 salvato; gli altri rilanciati il 2026-09-29, segnalazione 62); i suoi numeri entrano qui dai JSON quando è importata.

Il seed 42 riproduce la prova `_prova_recorddp_v2` del secondo Mac (commit 2c4a0f0): loss di
addestramento per round e holdout coincidono (0.00239868088 contro 0.00239868097), Yeom alla
settima cifra; LiRA composto no (0.5025 contro 0.5043), come atteso (sezione 3.4). Con
GroupNorm il risultato non dipende dalla macchina (segnalazione 55).

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

### 3.9 RQ2: partizione IID contro per sito (E-D)

Fonte: foglio `RQ2_partizione` di `risultati/Matrice_sintesi.xlsx` e
`risultati/worst_case/livello_di_caso.json`, dai JSON di
`experiments_altre_macchine/rq2-per_site` e `rq2-iid` (secondo Mac, commit 3815da7),
letti il 2026-09-24. Senza DP, 5 seed per braccio appaiati per seed; i bracci differiscono
solo per `partition.strategy`, con le stesse numerosità per client.

| metrica | per sito | IID | IID − per sito, appaiata (± sd) | t, 4 gdl |
|---|---|---|---|---|
| loss sull'holdout del modello rilasciato | 0.00162 | 0.00133 | −0.00029 ± 0.00051 | −1.26 |
| divario holdout − membri | 0.000278 | 0.000274 | −0.000004 ± 0.000079 | −0.12 |
| Yeom, ultimo round | 0.5001 | 0.5011 | +0.0010 ± 0.0041 | 0.57 |
| Shadow, media sui round | 0.5014 | 0.5012 | −0.0001 ± 0.0015 | −0.19 |
| LiRA composto | 0.4982 | 0.4993 | +0.0011 ± 0.0044 | 0.57 |
| TPR a FPR 1% (LiRA composto) | 0.0091 | 0.0089 | −0.0002 ± 0.0020 | −0.21 |

Test appaiato per record: per sito −0.33 ± 0.22 (z = −1.49), IID −0.03 ± 0.22
(z = −0.15), nessun segnale in nessuno dei due. Senza DP, passare dalla partizione per
sito a una IID con le stesse numerosità non cambia il successo degli attacchi valutati,
al caso in entrambi, né il divario fra membri e non membri. Il modello IID ha la loss
sull'holdout più bassa in 4 seed su 5, in media del 18%, non significativa a 5 seed. Nel
regime naturale RQ2 non misura quindi un effetto dell'eterogeneità sulla membership,
perché non c'è segnale da modulare; l'effetto sul costo della DP si misura col braccio
con DP, sotto. Mancano le statistiche delle feature per client nelle due partizioni, che
la guida chiede per mostrare che il fattore è cambiato.

**Con DP.** dp-fedavg, ε = 64 per round, C = 1: il punto operativo scelto con la regola
della griglia prima di guardare RQ2 con DP. Stessi foglio e file, dai JSON di
`experiments_altre_macchine/rq2-per_site-eps64` e `rq2-iid-eps64` (secondo Mac, commit
3d4d318, zero righe `[ERROR]` nel log), letti il 2026-09-26.

| metrica | per sito | IID | IID − per sito, appaiata (± sd) | t, 4 gdl |
|---|---|---|---|---|
| loss sull'holdout del modello rilasciato | 0.00633 | 0.00936 | +0.0030 ± 0.0066 | 1.03 |
| divario holdout − membri | 0.000299 | 0.000311 | +0.000011 ± 0.000050 | 0.51 |
| Yeom, ultimo round | 0.4990 | 0.5014 | +0.0023 ± 0.0048 | 1.08 |
| Shadow, media sui round | 0.5014 | 0.5010 | −0.0003 ± 0.0012 | −0.64 |
| LiRA composto | 0.5015 | 0.4997 | −0.0018 ± 0.0054 | −0.74 |
| TPR a FPR 1% (LiRA composto) | 0.0109 | 0.0082 | −0.0027 ± 0.0015 | −3.88 |

Costo della DP per partizione, come rapporto fra la loss sull'holdout con DP e senza DP
allo stesso seed e sulla stessa macchina: per sito da 2.1 a 8.6, media geometrica 3.9;
IID da 3.2 a 15.0, media geometrica 6.1. Il rapporto è più alto con IID in 4 seed su 5,
di 1.57 volte in media geometrica, ma l'interazione partizione × DP, sul logaritmo del
rapporto, non è significativa a 5 seed (+0.45 ± 0.45, t = 2.26 contro 2.776). Test
appaiato per record: per sito +0.18 ± 0.22 (z = 0.82), IID +0.19 ± 0.22 (z = 0.86),
nessun segnale. Sul rapporto delle medie anche questa macchina supera la soglia di costo
al punto operativo: 3.9 volte per sito e 7.0 volte IID, coerente con la sezione 3.1b.

Lettura. Con e senza DP la partizione non cambia il successo degli attacchi valutati,
che resta al caso, né il divario fra membri e non membri; la DP tende a costare di più
con la partizione IID, che senza DP partiva da una loss più bassa, ma a 5 seed la
differenza non è distinguibile dalla variabilità fra seed. L'unico test del foglio sopra
la soglia è la TPR del LiRA composto con DP (p = 0.018, per sito sopra il caso e IID
sotto in tutti e 5 i seed): è una metrica secondaria con calibrazione degenere (3.4), i
test appaiati del foglio sono 18 e a α = 0.05 senza correzione se ne attende circa uno
per caso, quindi non lo leggiamo come effetto della partizione. RQ2 risponde così sul
costo del modello, non su un beneficio di privacy, per la stessa ragione di RQ1: nel
regime naturale non c'è un segnale da ridurre.

Il braccio per sito ha lo stesso config di `nodp-sweep2` ma gira sull'altra macchina, e
la loss sull'holdout differisce per seed fino al 23% (segnalazione 55, confermata: dipende
dalla macchina, non dal codice): il confronto di RQ2
usa perciò i due bracci della stessa macchina, non `nodp-sweep2`.

### 3.10 RQ3: FedAvg contro FedProx (E-E)

Fonte: JSON di `experiments_altre_macchine/rq3-mu0` e `rq3-mu0.01` (Windows, commit 478d471,
i due bracci in parallelo), letti il 2026-09-27; 5 seed per braccio appaiati per seed, zero
righe `[ERROR]`, round 1 identico nei due bracci per ogni seed. Foglio `RQ3_algoritmo` di
`risultati/Matrice_sintesi.xlsx` (con la curva della loss sull'holdout per round) e confronto
C-RQ3 di `matrice_confronti.xlsx`, dal 2026-09-27.

| metrica | FedProx mu = 0.01 | FedAvg mu = 0 | FedAvg − FedProx, appaiata (± sd) | t, 4 gdl |
|---|---|---|---|---|
| loss sull'holdout del modello rilasciato | 0.00156 | 0.00040 | −0.00117 ± 0.00039 | −6.63 |
| divario holdout − membri | 0.000265 | 0.000213 | −0.000052 ± 0.000072 | −1.64 |
| Yeom, ultimo round | 0.5000 | 0.5011 | +0.0011 ± 0.0025 | 1.01 |
| Shadow, media sui round | 0.5014 | 0.4995 | −0.0019 ± 0.0035 | −1.22 |
| LiRA composto | 0.5015 | 0.5008 | −0.0007 ± 0.0039 | −0.39 |
| TPR a FPR 1% (LiRA composto) | 0.0103 | 0.0105 | +0.0002 ± 0.0014 | 0.34 |

Con FedAvg il modello rilasciato ha la loss sull'holdout 3.9 volte più bassa in media
geometrica (da 3.7 a 4.5 per seed, 5 su 5); attacchi al caso in entrambi, divario membri −
non membri non diverso. Le curve per round: FedAvg si ferma a 0.0004-0.0005 dal round 4-6,
FedProx scende ancora al round 10 in tutti i seed (seed 42: 0.0041 al round 2, 0.0018 al
round 10). Con l'inizializzazione comune (sezione 3.1c, seed 42, Mac principale) FedProx
arriva a 0.0011: metà del divario viene dal round 1 (segnalazione 48), il resto no, a meno
dell'effetto macchina (fino al 23%, segnalazione 55). Nei round 3-10 di FedProx gli update
di Caltech e JPL hanno norma 0.16-0.29 (controllo (a)) e il termine prossimale a fine round
vale il 5-15% della loss locale: non domina la loss, limita lo spostamento per round.
Ipotesi, da verificare con le prove su mu (`ESPERIMENTI.md` E-E): con mu = 0.01 dieci round
non bastano a FedProx. La loss di addestramento locale registrata include il termine
prossimale e non è confrontabile fra i due bracci (segnalazione 59). Tutta la campagna usa
FedProx mu = 0.01 come base: se cambiarla è una decisione del supervisore.

**Mac principale, senza DP, 2026-09-29** (JSON di `experiments/rq3-mu0`, commit 64757fd,
contro `nodp-sweep2`; non ancora nelle matrici). FedProx ha la loss sull'holdout 3.69 volte
quella di FedAvg in media geometrica (da 2.74 a 4.31, 5 seed su 5), coerente con Windows
(3.90). Il round 10 della run a 30 round (seed 42, commit 64757fd) riproduce `nodp-sweep2`
(0.002186): il confronto regge fra i due commit. Attacchi al caso.

**Prove su mu** (seed 42, Mac principale, loss sull'holdout al round 10): mu = 0: 0.000508;
0.001: 0.000536; 0.01: 0.002186; 0.1: 0.003791. Il costo cresce con mu e il salto sta fra
0.001 e 0.01. Con mu = 0.001 gli update hanno norma 0.23-0.52 al round 10 (FedAvg 2.2-6.8) e
la qualità di FedAvg: la dimensione del passo da sola non spiega il costo. A 30 round FedProx
mu = 0.01 scende a 0.00127 al round 20 e a 0.00094 al round 30, ancora in discesa: 1.85 volte
FedAvg a 10 round. FedProx è soprattutto più lento; se raggiunga il livello di FedAvg non è
stabilito.

**Con DP, Mac principale** (ε = 64, C = 1, dp-fedavg; `experiments/rq3-mu0-eps64`, commit
64757fd e 8dceb39, contro `rq1-eps64`, commit 8d44ae3; fra 8d44ae3 e 8dceb39 `src/` e
`run_experiments.py` non cambiano). FedProx ha la loss sull'holdout 0.53 volte quella di
FedAvg in media geometrica (da 0.28 a 1.62; 4 seed su 5 sotto 1; t = −1.93 sul logaritmo,
non significativo a 4 gdl). La DP costa a FedAvg 30.4 volte (da 20 a 42) e a FedProx 4.4
volte (da 1.9 a 14.7). Norme degli update: FedAvg fra 2.5 e 8.5 in tutti i round, quindi
tagliate da C = 1 a ogni round; FedProx fra 0.2 e 0.5 dal round 5, sotto C (tagliate solo al
round 1, quando il termine prossimale non è attivo). Lettura, preliminare: sotto DP l'ordine
fra i due algoritmi dipende dal clipping, e C = 1 è stato scelto sulla griglia di FedProx.
Lo screening di C per FedAvg (C = 2, 4, 8, seed 42) è in corso dal 2026-09-29
(`ESPERIMENTI.md` E-E). Yeom al caso in tutte le celle.

**Con DP, Windows** (FedAvg, `experiments_altre_macchine/rq3-mu0-eps64`, 5 seed, fe22124
con `-dirty` da file non tracciati e codice identico, `PROVENIENZA.txt`). La DP costa a
FedAvg 52 volte (da 34 a 83) sul riferimento Windows senza DP a 478d471 (percorso di
training identico: loss del round 1 uguale a 12 cifre). Con stesso seed e stesso config la
loss con DP su Windows è 1.61 volte quella del Mac (da 1.00 a 2.45), contro 0.94 senza DP:
sotto DP l'effetto macchina è più grande (segnalazione 60), quindi i confronti con DP si
fanno solo sulla stessa macchina. Il braccio FedProx con DP su Windows è in corso.

## 4. Cosa manca

L'elenco ordinato, con comandi e prerequisiti, è `ESPERIMENTI.md`. E-A è chiuso (sezione
3.1b), con la griglia del punto operativo e ε = 64 a 5 seed; E-C è eseguito sulle celle
esistenti, ε = 64 compresa (sezione 3.2). E-D (RQ2) è analizzato con e senza DP (sezione
3.9) e sta in `experiments_altre_macchine/`, come E-E senza DP di Windows (sezione 3.10,
segnalazione 45). I controlli del protocollo sono
eseguiti (sezione 3.1c). E-B a σ = 1, 2 e 5 e il riferimento GroupNorm sono completi (sezione
3.5); E-E sul Mac principale, con e senza DP, e le prove su mu sono complete (sezione 3.10).
Restano: la cella di solo clipping di E-B (in corso), lo screening di C per FedAvg con DP
(in corso) e, se lo indica, un braccio a 5 seed, il braccio FedProx con DP su Windows,
l'aggiornamento delle matrici con le celle del 2026-09-29, il canary su più siti, le
statistiche delle feature per client di E-D, la rianalisi NVFlare. Da decidere col supervisore: la metrica primaria per record (segnalazione 47) e se
le campagne future usano l'inizializzazione comune (segnalazione 48). I bug che toccano i numeri sono in
`Segnalazioni_tecniche_2026-09-22.md`, punti 1, 5, 6, 9, 35, 36, 38, 45, 48, 50, 51, 55. Fuori dal paper, come infrastruttura o lavoro futuro: ML Plane,
Privacy Auditor, PES, ByzantineDetector, FedMIA-gradient, secondo dataset. Per le
frasi da non scrivere senza evidenza: guida, sezione 11.
