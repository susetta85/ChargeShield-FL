# ChargeShield-FL — Risultati per domanda di ricerca (campagna v3)

> Aggiornato al 2026-10-30. Numeri letti dai CSV raccolti sulle tre macchine
> (`risultati/campagna_v3/mac_susy`, `mac_domenico`, `windows`) e dai JSON in `esiti/`.
> Test: Mann-Whitney unilaterale per gli attacchi sul canary persona (Holm sulle metriche
> secondarie, la primaria al round 3 non corretta); permutazione a due code (20 000
> permutazioni) sulla media delle perdite per i confronti di utilità.
> Stato di ogni risultato: **consolidato** (numero di seed definitivo, nessun problema aperto),
> **provvisorio** (seed ancora in corso) o **da rifare** (problema di metodo trovato).

## Come leggere i numeri

- **Somma A+B**: nel canary ogni elemento è membro in un braccio e non membro nell'altro.
  Se l'attacco non vede niente, AUC_A + AUC_B = 1. Sopra 1 l'appartenenza è riconosciuta.
- **AUC appaiato** (canary persona): per ogni utente si confronta la sua perdita quando è
  dentro con quella quando è fuori, così si toglie la difficoltà propria della persona.
- **p**: probabilità di osservare una differenza almeno così grande se in realtà non ci
  fosse nessun effetto. p < 0,05 vuol dire che il caso spiega male la differenza; p = 0,6
  vuol dire che la differenza è compatibile con il caso. Non misura quanto è grande l'effetto:
  per quello c'è il rapporto fra le medie.
- **Rapporto**: perdita media sull'holdout del modello rilasciato in una cella divisa per
  quella della cella di confronto (per esempio FedProx / FedAvg, o DP / no DP).

---

## RQ1 — Rilevabilità dell'appartenenza e controlli positivi

### 1.1 Regime naturale: nessun attacco realistico vede i membri — consolidato

| Dataset | Celle | Seed | Yeom | LiRA | RMIA | Classificatore shadow |
|---|---|---|---|---|---|---|
| ACN-Data (6 feature), no DP μ=0,01 | 1 | 10 | 0,501 [0,496–0,506] | 0,500 | 0,501 | 0,496 |
| ACN-Data, DP locale ε=64 μ=0,01 | 1 | 10 | 0,502 [0,495–0,508] | 0,499 | 0,501 | 0,505 |
| ChargePlace Scotland, tutte le celle | 6 | 5 | 0,492–0,509 | 0,497–0,505 | 0,494–0,505 | 0,486–0,513 |
| ACN 4 feature, tutte le celle | 8 | 5–10 | 0,497–0,503 | 0,498–0,501 | 0,499–0,502 | 0,497–0,504 |
| ACN, curva di capacità (32-16, 64-32, 128-64) | 3 | 5 | 0,502–0,503 | 0,500 | 0,500–0,501 | 0,497–0,508 |

Attacco al soggetto (metrica calibrata, AUC sulle unità, 5 seed): ACN gruppo no DP 0,513
(1/5 seed significativi), DP per client 0,496, session DP σ=2 e 5 0,501, subject DP 0,498;
Scotland entità no DP 0,520 (0/5), DP per client 0,486, subject DP 0,484; Scotland gruppo
fra 0,497 e 0,508. Nessuna differenza fra ACN e Scotland: in nessuno dei due dataset
l'appartenenza si vede in condizioni naturali, con o senza DP.

### 1.1b Avversario B (partecipante o osservatore esterno) — ACN, Scotland e controllo positivo su Office 1 fatti

L'avversario B vede solo i modelli globali, round per round, non gli aggiornamenti dei client.
Implementato il 20 ottobre (`lira.observation_surface: partecipante`, `lira2.py`): le ombre
prendono il posto del client nell'aggregazione e gli aggiornamenti degli altri client sono
emulati con i dati ausiliari del server; il bersaglio è il modello globale. Yeom e lo shadow con
dati ausiliari usano già solo il modello globale, quindi valgono per B in tutte le celle di 1.1.

ACN, regime naturale (`v3_partecipante_acn_*`, 5 seed, round 10; fra parentesi il massimo su
seed e round):

| Cella | Yeom | LiRA | LiRA out / off | RMIA | RMIA out / off |
|---|---|---|---|---|---|
| No DP μ = 0,01 | 0,501 (0,506) | 0,497 (0,512) | 0,501 / 0,501 (0,515) | 0,505 (0,513) | 0,505 / 0,505 (0,515) |
| DP locale ε = 64 | 0,505 (0,511) | 0,497 (0,517) | 0,505 / 0,505 (0,509) | 0,508 (0,516) | 0,506 / 0,507 (0,518) |
| DP centrale ε = 64 | 0,500 (0,506) | 0,502 (0,511) | 0,501 / 0,501 (0,514) | 0,501 (0,512) | 0,500 / 0,500 (0,514) |

Valori uguali a quelli del server sulle stesse celle e seed (differenze ≤ 0,005): previsione V23
confermata, chi vede solo il modello globale non vede più del server.

Scotland, regime naturale (`v3o_partecipante_sc3_*`, Mac di Domenico, 5 seed, round 10; fra
parentesi il massimo su seed e round):

| Cella | Yeom | LiRA | LiRA out / off | RMIA | RMIA out / off |
|---|---|---|---|---|---|
| No DP μ = 0,01 | 0,498 (0,503) | 0,497 (0,518) | 0,498 / 0,498 (0,511) | 0,501 (0,512) | 0,501 / 0,500 (0,514) |
| DP locale ε = 64 μ = 0,01 | 0,501 (0,509) | 0,502 (0,516) | 0,504 / 0,505 (0,517) | 0,501 (0,516) | 0,503 / 0,503 (0,517) |
| DP centrale ε = 64 μ = 0,01 | 0,502 (0,513) | 0,499 (0,514) | 0,496 / 0,497 (0,516) | 0,497 (0,516) | 0,496 / 0,497 (0,517) |

Nessun segnale su Scotland: le medie restano fra 0,496 e 0,505 e il massimo è 0,517, come su ACN.
Anche per B l'appartenenza non si vede in regime naturale, con o senza DP.

Controlli positivi visti da B: il canary persona sul modello rilasciato (1.3) è riconosciuto senza
DP (0,697, 8/10 al round 3) e non con DP locale o centrale. Il canary per sessione sul modello
globale è invece debole. Somme A+B su Office 1 (30 copie, round 3, 5 seed, `v3o_partecipante_canary*`,
Mac di Domenico), media [minimo–massimo] e seed con somma > 1:

| Attacco | No DP | Central DP ε = 64 (A2) | Local DP ε = 64 (A3) |
|---|---|---|---|
| Perdita grezza | 1,010 [0,992–1,050], 3/5 | 0,997 [0,985–1,002], 1/5 | 1,001 [0,992–1,008], 2/5 |
| LiRA offline | 1,031 [0,975–1,087], 4/5 | 1,002 [0,990–1,010], 3/5 | 0,983 [0,953–1,010], 1/5 |
| RMIA offline | 1,044 [0,975–1,176], 3/5 | 0,996 [0,987–1,004], 1/5 | 0,986 [0,945–1,009], 2/5 |
| LiRA out | 1,026 [0,965–1,087], 4/5 | 0,966 [0,912–1,010], 2/5 | 0,982 [0,948–1,008], 2/5 |
| RMIA out | 1,040 [0,976–1,160], 4/5 | 0,980 [0,934–1,020], 1/5 | 0,987 [0,938–1,021], 2/5 |
| LiRA online | 0,958 [0,827–1,130], 1/5 | 1,020 [0,762–1,350], 3/5 | 0,988 [0,912–1,055], 2/5 |
| RMIA online | 1,032 [0,968–1,147], 4/5 | 0,996 [0,985–1,011], 1/5 | 0,979 [0,941–1,022], 1/5 |

Anche senza DP B riconosce poco il canary per sessione: i migliori attacchi offline e "out" danno
somme di 1,03–1,04, appena sopra 1, con variazione fra seed in entrambi i sensi. Sull'aggiornamento
del client (1.2) le stesse somme sono 1,13–1,29. Il segnale del canary per sessione sta quindi
nell'aggiornamento che vede l'aggregatore, non nel modello globale che vede B. Con la DP (centrale o
locale) le somme stanno intorno a 1.

### 1.2 Canary per sessione (Office 1, 30 copie, 5 seed) — consolidato

Superficie: aggiornamento del client osservato dall'aggregatore. Somma A+B media
[minimo–massimo] e seed con somma > 1.

| Attacco | No DP | Central DP ε=64 (aggregatore vede l'update solo clippato, A2) | Local DP ε=64 (A3) |
|---|---|---|---|
| Perdita grezza | 1,127 [1,002–1,308], 5/5 | 1,099 [1,038–1,177], 5/5 | 1,014 [0,992–1,052] |
| LiRA offline | 1,284 [1,018–1,615], 5/5 | 1,280 [1,115–1,485], 5/5 | 0,999 [0,975–1,030] |
| RMIA offline | 1,287 [1,044–1,521], 5/5 | 1,240 [1,080–1,413], 5/5 | 1,012 [1,002–1,024] |
| LiRA online | 0,842 (invertito) | 1,052 | 1,013 |
| RMIA online | 0,774 (invertito) | 0,790 (invertito) | 1,005 |

Seed 42 e 123 dal PC Windows, 456, 789 e 1234 dal Mac di Domenico; central DP dal Mac
principale. **Il solo clipping non protegge**: con la central DP l'aggregatore riceve
l'update clippato (il rumore viene aggiunto dopo, sull'aggregato) e riconosce il canary
come senza DP. Il rumore lato client lo nasconde.

### 1.2b Canary per sessione su ChargePlace Scotland (Shetland Islands, 30 copie, 5 seed) — consolidato

Controllo positivo su Scotland (`coda_v3hk_domenico.csv`, Mac di Domenico): 20 sessioni di
Shetland Islands ripetute 30 volte (Shetland ha 1944 sessioni di training contro circa 25 000 di
Glasgow e Dundee), 3 round da 1000 epoche, hidden [32, 16], latent 8, come il canary di Office 1.
Superficie: aggiornamento del client osservato dall'aggregatore. Somma A+B media e seed con somma > 1.

| Attacco | No DP, round 1 | No DP, round 3 | Central DP ε=64 (A2), round 3 | Local DP ε=64 (A3), round 3 |
|---|---|---|---|---|
| Perdita grezza | 1,408, 5/5 | 1,036, 4/5 | 1,047, 5/5 | 1,012, 4/5 |
| LiRA offline | 1,332, 5/5 | 1,183, 5/5 | 1,145, 5/5 | 0,998, 3/5 |
| LiRA out | 1,330, 5/5 | 1,250, 5/5 | 1,226, 5/5 | 0,973, 1/5 |
| RMIA offline | 1,107, 5/5 | 1,178, 5/5 | 1,160, 5/5 | 1,000, 3/5 |
| RMIA out | 1,283, 5/5 | 1,276, 5/5 | 1,223, 5/5 | 0,994, 3/5 |
| LiRA online | 1,419, 5/5 | 0,823 (invertito) | 1,092 | 1,039 |
| RMIA online | 0,891 (invertito) | 0,861 (invertito) | 0,852 (invertito) | 1,006 |

Stesso quadro di Office 1: senza DP e con la DP centrale l'aggregatore riconosce il canary in
tutti i seed con le varianti offline e "out"; con la DP locale le somme tornano intorno a 1. Con
la DP centrale al round 1 il segnale non c'è ancora (somme 0,98–1,05) e compare dal round 2.
Yeom e shadow sul modello globale restano fra 0,97 e 1,02 (il canary è diluito nella media).
**Ora c'è un controllo positivo anche su Scotland.**

### 1.3 Canary persona (JPL, 40 utenti, 30 copie, spostamento 0,5) — consolidato (no DP 10 seed; local e central con rumore indipendente, 5 seed)

AUC appaiato al round 3, seed significativi. Con la DP valgono solo le run con rumore indipendente
fra i bracci (1.4).

| Regime | Seed | Aggregatore | Modello rilasciato |
|---|---|---|---|
| No DP | 10 | 0,802, 10/10 | 0,697, 8/10 |
| Central DP ε=64, rumore indipendente | 5 | 0,711, 4/5 | 0,547, 0/5 |
| Local DP ε=64, rumore indipendente | 5 | 0,487, 0/5 | 0,564, 1/5 |

Le prime run con la DP (10 seed, Mac principale) avevano il rumore condiviso fra i bracci e non si
usano: central 0,843 (10/10) e 0,574 (2/10), local 0,543 (1/10) e 0,555 (2/10). Il rumore condiviso
gonfia l'AUC appaiato anche per la central sull'aggregatore (0,843 contro 0,711), come spiegato in 1.4.

Al round 1 senza DP l'AUC sull'aggregatore è 0,999 in 10 seed su 10: saturazione (gli AUC
dei singoli bracci sono 0,955–0,988, il p è al minimo possibile per 20 utenti contro 20),
non un difetto. Un seed su cinque significativo sul modello rilasciato con local DP è compatibile
con il caso.

Le conclusioni sono le stesse con le due versioni del rumore: la DP centrale lascia vedere il canary
al server (0,711, 4/5), non sul modello rilasciato; la DP locale non lo lascia vedere.

Spostamento 0, senza DP (10 seed: 42–1234 sul Mac principale, 1618–3141 su Windows): il segnale
è forte al round 1 (0,731, 2/5 e 0,818, 5/5; insieme 0,775, 7/10) e si indebolisce al round 3
(0,536, 0/5 e 0,645, 2/5; insieme 0,59, 2/10). Con rumore indipendente (Mac principale, seed 42–1234, round 3): DP
centrale sull'aggregatore 0,518 (0/5), sul modello rilasciato 0,465 (0/5); DP locale 0,484 (0/5)
e 0,460 (0/5). Senza spostamento il canary non si vede in nessun regime con DP. Ripetizione con rumore indipendente sui seed 1618–3141 (Windows, `coda_v3h`, 5 seed, round 3):
DP centrale sull'aggregatore 0,580 (1/5), sul modello rilasciato 0,505 (1/5); DP locale 0,490 (0/5)
e 0,510 (0/5). Con il rumore condiviso della versione precedente la central dava 0,698 (3/5): la
differenza fra le due versioni è spiegata in 1.4. Insieme (10 seed, rumore indipendente, round 3):
central 0,549 (1/10) sull'aggregatore e 0,485 (1/10) sul modello rilasciato; local 0,487 (0/10) e
0,485 (0/10).

### 1.3b Quante sessioni servono — canary persona con 1, 5, 10 e 30 copie (JPL, round 3, 5 seed) — fatto

Coda v3n (`coda_v3on_domenico.csv`, Mac di Domenico): stesso canary persona di 1.3, spostamento 0,5, con
1, 5 e 10 copie della persona e, per il confronto, 30 copie. Senza DP la configurazione usa GroupNorm,
come le run con DP per sessione: la riga senza DP non è quindi la stessa run di 1.3 (0,802, 10 seed) e
va letta solo dentro questa tabella. AUC appaiato al round 3, seed significativi su 5 (p < 0,05).

| Copie | Senza DP, aggregatore | Senza DP, rilasciato | Session DP σ = 5, aggregatore | Session DP σ = 5, rilasciato |
|---|---|---|---|---|
| 1 | 0,598, 2/5 | 0,580, 1/5 | 0,514, 1/5 | 0,524, 1/5 |
| 5 | 0,630, 3/5 | 0,531, 0/5 | 0,572, 1/5 | 0,531, 0/5 |
| 10 | 0,700, 3/5 | 0,617, 1/5 | 0,642, 2/5 | 0,547, 0/5 |
| 30 | 0,793, 5/5 | 0,666, 3/5 | 0,590, 1/5 (da 3.3b) | 0,544, 0/5 (da 3.3b) |

Senza DP il segnale cresce con il numero di sessioni: già con una sola sessione l'aggregatore arriva a
0,60, ma su 5 seed solo 2 sono significativi; con 30 sessioni la media è 0,79 e tutti e 5 i seed sono
significativi. Il modello rilasciato lo vede meno: 0,67 con 30 sessioni, 3 seed su 5. Con la session DP
a σ = 5 il segnale è più debole a parità di sessioni (0,64 contro 0,70 a 10 copie) e sul modello
rilasciato non è significativo a 5 e 10 copie. La colonna con 30 copie in DP viene da 3.3b, non da questa coda, e lì il valore
(0,59) è più basso di quello a 10 copie: con 5 seed la differenza è dentro il rumore, quindi la curva
con la DP non si può dire né crescente né piatta. Per una risposta precisa su quante sessioni servono
con la DP servono più seed.

### 1.4 Due problemi di metodo trovati

**Rumore condiviso fra i bracci (risolto nel codice il 2026-10-26).** Nel canary persona i
due bracci usavano lo stesso seed; il rumore della DP veniva dal generatore globale di torch
e quindi era identico nei due bracci. La differenza appaiata L_B(u) − L_A(u) cancellava il
rumore. Con la subject DP l'AUC appaiato restava 0,70–0,73 da σ = 5 (ε ≈ 29) a σ = 1000
(ε ≈ 0,08), anche con 1 epoca e con 1 copia, e la somma A+B era esattamente 1,000: i due
modelli erano praticamente identici. Non è una fuga verso un avversario reale (che non vede
mai due addestramenti con lo stesso rumore), ma rende l'AUC appaiato non valido per misurare
la DP. Correzione: chiave `ml.seme_rumore_dp`, seme diverso per braccio
(`canary_persona.seme_rumore_braccio`), test `tests/test_rumore_dp_separato.py`. Le run
senza DP non sono toccate; la central DP sull'aggregatore nemmeno (l'update è letto prima
del rumore). Il rumore condiviso può solo facilitare il riconoscimento, quindi "la local DP
nasconde il canary" resta vero a maggior ragione.
**Confermato con il rumore indipendente (coda `coda_v3g_domenico.csv`, 5 seed).** AUC appaiato
al round 3 sull'aggregatore: subject DP σ = 1000 0,476, σ = 5 0,499, σ = 8 0,522, nessun seed
significativo (prima, con il rumore condiviso, 0,72 / 0,72 / 0,70). Il segnale era interamente un
artefatto del disegno. Da rifare ancora: local/central allo spostamento 0 (code v3h).

**Rumore condiviso e central DP (2026-10-30).** Sui seed 1618–3141 (Windows) l'AUC appaiato della
central DP sull'aggregatore è identico con e senza rumore condiviso al round 1 (0,610, 0,362, 0,453,
0,497, 0,510 per i cinque seed). Dal round 2 le due versioni divergono: al round 3 0,698 (3/5) con il
rumore condiviso contro 0,580 (1/5) con il rumore indipendente. Il modello globale rilasciato porta il
rumore condiviso e lo ritrasmette ai round successivi, quindi l'affermazione precedente che la central
DP sull'aggregatore non è influenzata dal rumore vale solo al round 1. Lo stesso effetto spiega il
canary persona allo spostamento 0,5 (1.3): central sull'aggregatore 0,843 con il rumore condiviso,
0,711 con quello indipendente. Lì le due versioni sono state eseguite su macchine diverse (Mac
principale e Mac di Domenico), quindi anche il round 1 differisce e il confronto per round non è pulito.

**Inversione di RMIA (e a volte LiRA) online sui canary — causa confermata (2026-10-29).** Al
round 3 RMIA online dà somme sotto 1 su Office 1 senza DP in 5 seed su 5 (0,55–0,93) e con la DP
centrale in 4 su 5 (0,59–0,93), e su Scotland (0,82–0,89, 1.2b): i membri risultano meno membri
dei non membri. LiRA online è instabile: sotto 1 in 4 seed su 5 su Office 1 senza DP (0,66–0,85),
sopra 1 nella ripetizione diagnostica del seed 42 (1,45, 1,45, 1,10 ai round 1–3). Le varianti
che non usano ombre addestrate sul candidato funzionano sempre (LiRA e RMIA "out" e offline).

Diagnostica (`coda_v3hk_domenico.csv`, seed 42, due bracci): per i canary `lira2.py` salva ora la
statistica del bersaglio t (log della perdita) e le medie delle ombre IN e OUT
(`lira_per_client.office1.canary`). Mediane al round 1, braccio A / B:

| Gruppo | t del bersaglio | media ombre IN | media ombre OUT |
|---|---|---|---|
| Membri (620 copie) | −13,30 / −11,77 | −12,32 / −12,34 | −10,75 / −10,25 |
| Non membri (40) | −12,25 / −11,46 | −10,94 / −11,24 | −10,33 / −10,51 |

Due fatti. (1) Le ombre IN memorizzano i canary membri quanto o più del bersaglio (nel braccio B
t è sopra la media IN), perché ognuna è addestrata su circa 500 sessioni di cui due terzi sono
copie dei canary. (2) Il bersaglio, addestrato su tutti i dati di Office 1, ricostruisce anche i
non membri molto meglio delle ombre (al round 1 t è 1,0–1,9 sotto la media OUT), perché le ombre del
server hanno pochi dati generici. Rispetto alle ombre IN, quindi, i non membri sembrano "più
membri" dei membri in 4 casi su 6 (due bracci per tre round), ed è questo il confronto che pesa nel
denominatore di RMIA online. Prova diretta della causa: con lo stesso seed ma le ombre addestrate
sull'universo del client (più dati generici, analisi di sensibilità) l'inversione sparisce, RMIA
online 1,27 / 1,16 / 1,41 e LiRA online 1,08 / 1,13 / 1,37 ai round 1–3.

Conclusione: non è un difetto di LiRA o RMIA in generale (nel regime naturale danno 0,50 come gli
altri attacchi), ma del loro uso con le sole risorse del server quando i candidati sono record
ripetuti: le ombre online del server sono modelli generici peggiori del bersaglio e memorizzano i
duplicati più di quanto il bersaglio li distingua. Sui canary si riportano la perdita grezza e le
varianti "out" e offline. L'ipotesi iniziale ("le ombre IN memorizzano più del bersaglio") era solo
metà della spiegazione: conta anche quanto il bersaglio generalizza meglio delle ombre.

### 1.5 Risposta a RQ1

- In condizioni naturali, su due dataset reali, nessun attacco realistico distingue membri e
  non membri, a livello di sessione o di soggetto, con o senza DP.
- Il sistema di audit vede l'appartenenza quando la memorizzazione è indotta: il canary per
  sessione e il canary persona sono riconosciuti senza DP in tutti i seed.
- L'avversario esterno (partecipante o osservatore, vede solo i modelli globali) non vede più
  del server: su ACN tutte le forme di LiRA e RMIA restano fra 0,497 e 0,508 con e senza DP
  (1.1b). Su Scotland, con e senza DP, B non vede più del server (medie fra 0,496 e 0,505). Il
  canary per sessione non si vede da B nemmeno senza DP (somme 1,01–1,04, 1.1b); il canary persona
  sul modello rilasciato è già un controllo positivo per B.
- Il solo clipping non protegge dall'aggregatore: con la central DP il canary è riconosciuto
  come senza DP. Il rumore aggiunto dal client (local DP) lo nasconde. Il rumore sull'aggregato
  protegge solo il modello rilasciato.
- Il controllo positivo c'è anche su ChargePlace Scotland (1.2b): senza DP e con DP centrale
  l'aggregatore riconosce il canary in tutti i seed, con la DP locale no.

---

## RQ2 — Ottimizzazione federata e costo della protezione

Costo: perdita sull'holdout del modello globale rilasciato, 10 round, 10 seed, C = 1, ε = 64
per round (ε totale ≈ 430). "Solo clip": clipping a C senza rumore. "Solo rumore": rumore
tarato su ε = 64 senza clipping.

### 2.1 ACN-Data (6 feature) — consolidato

| Variante | FedAvg μ=0 | FedProx μ=0,01 | Rapporto | p |
|---|---|---|---|---|
| No DP | 5,1·10⁻⁴ | 1,2·10⁻³ | 2,33 | < 0,001 |
| Solo clip | 1,15·10⁻² | 3,2·10⁻³ | 0,27 | < 0,001 |
| Solo rumore | 0,110 | 0,093 | 0,85 | 0,62 |
| DP completa | 0,142 | 0,078 | **0,55** | **0,02** |

Curva in μ con DP completa: μ = 0,001 0,136 (0,96, p = 0,85), μ = 0,01 0,078 (0,55,
p = 0,02), μ = 0,1 0,089 (0,63, p = 0,08). DP / no DP: 66× con μ = 0,01, 276× con μ = 0.
Con altri 10 seed (replica, 2.5) i rapporti a 10 round sono 0,75 e 0,69; con 20 seed 0,65
(p = 0,008) e 0,66 (p = 0,02).
A 30 round: FedAvg 0,182, FedProx μ = 0,01 0,146 (0,80, p = 0,46); DP / no DP 200×.

Griglia di ε (5 seed, μ = 0,01): da ε = 2 a 16 la perdita è 0,37–0,47 (267–343×), a ε = 32
0,258 (189×), a ε = 256 0,028 (21×).

### 2.2 ChargePlace Scotland — consolidato

| Variante | μ=0 | μ=0,01 | μ=0,1 |
|---|---|---|---|
| No DP | 6,6·10⁻⁵ | 3,0·10⁻⁴ (4,5×, p < 0,001) | 1,3·10⁻³ (20×, p < 0,001) |
| Solo clip | 5,4·10⁻³ | 2,4·10⁻³ (0,45, p = 0,006) | 8,9·10⁻³ (1,66, p = 0,01) |
| Solo rumore | 0,106 | 0,196 (1,85, p = 0,05) | 0,164 (1,55, p = 0,23) |
| DP completa | 0,183 | 0,159 (0,87, p = 0,56) | **0,090 (0,49, p = 0,015)** |

μ = 0,001: 0,167 (0,91, p = 0,76). DP / no DP: 536× con μ = 0,01, 68× con μ = 0,1.
A 30 round: μ = 0 0,297, μ = 0,01 0,228 (0,77, p = 0,37), μ = 0,1 0,221 (0,74, p = 0,34).

### 2.3 ACN 4 feature (stesse feature di Scotland) — consolidato

| Variante | μ=0 | μ=0,01 | μ=0,1 |
|---|---|---|---|
| No DP | 2,0·10⁻⁵ | 3,2·10⁻⁴ (16×, p < 0,001) | 1,7·10⁻³ (87×, p < 0,001) |
| Solo clip | 4,7·10⁻³ | 1,55·10⁻³ (0,33, p = 0,03) | 8,5·10⁻³ (1,81, p = 0,10) |
| Solo rumore | 0,162 | 0,172 (1,06, p = 0,86) | 0,156 (0,96, p = 0,91) |
| DP completa | 0,209 | 0,137 (0,65, p = 0,06) | **0,098 (0,47, p = 0,003)** |

μ = 0,001: 0,197 (0,94, p = 0,73). DP / no DP: 422× con μ = 0,01, 57× con μ = 0,1.

### 2.4 Andamento per round (convergenza) — consolidato

Mediana su 10 seed della perdita sull'holdout del modello rilasciato, round per round
(`loss_per_round.csv`). I primi 10 round delle run a 30 round coincidono con le run a 10 round
(stesso seed, stesso codice): il training è deterministico.

| Dataset | Cella | r1 | r3 | r5 | r10 | r30 |
|---|---|---|---|---|---|---|
| ACN | no DP, FedAvg | 1,9·10⁻² | 1,3·10⁻³ | 6,0·10⁻⁴ | 5,1·10⁻⁴ | 4,1·10⁻⁴ |
| ACN | no DP, FedProx 0,01 | 1,9·10⁻² | 2,0·10⁻³ | 1,6·10⁻³ | 1,1·10⁻³ | 6,8·10⁻⁴ |
| ACN | DP, FedAvg | 0,129 | 0,139 | 0,142 | 0,129 | 0,189 |
| ACN | DP, FedProx 0,01 | 0,129 | **0,067** | 0,080 | 0,069 | 0,146 |
| ACN | DP, FedProx 0,1 | 0,129 | 0,061 | 0,094 | 0,085 | — |
| Scotland | no DP, FedAvg | 1,5·10⁻² | 1,5·10⁻⁴ | 6,8·10⁻⁵ | 6,3·10⁻⁵ | 2,0·10⁻⁵ |
| Scotland | no DP, FedProx 0,1 | 1,5·10⁻² | 3,0·10⁻³ | 2,6·10⁻³ | 1,3·10⁻³ | 6,1·10⁻⁴ |
| Scotland | DP, FedAvg | 0,126 | 0,139 | 0,141 | 0,177 | 0,278 |
| Scotland | DP, FedProx 0,01 | 0,126 | 0,082 | 0,060 | 0,189 | 0,214 |
| Scotland | DP, FedProx 0,1 | 0,126 | 0,065 | **0,040** | 0,077 | 0,195 |
| ACN-4f | DP, FedAvg | 0,125 | 0,150 | 0,166 | 0,226 | — |
| ACN-4f | DP, FedProx 0,1 | 0,125 | 0,055 | **0,042** | 0,077 | — |

- **Senza DP** entrambi gli algoritmi convergono, con perdita che scende per tutti i 30 round;
  FedAvg scende più in fretta e più in basso, e il divario resta a 30 round (1,7× su ACN,
  fino a 30× su Scotland con μ = 0,1).
- **Con la DP** nessuno dei due converge. FedAvg non scende mai sotto la perdita del round 1
  e poi sale. FedProx scende per 3–5 round fino a un minimo (0,04–0,07, da 2 a 4 volte sotto
  FedAvg allo stesso round) e poi risale, perché il rumore accumulato prevale. A 30 round le
  due curve sono di nuovo vicine.
- Il confronto a 10 round dipende quindi da dove cade il round 10 rispetto al minimo di
  FedProx (per esempio su Scotland con μ = 0,01 il round 10 è un picco: 0,189). Con la DP
  conviene un numero di round piccolo, che oltretutto riduce l'ε totale; scegliere il round
  migliore guardando l'holdout non è però lecito, va fissato prima.

### 2.4b Perché FedProx aiuta: norme degli aggiornamenti — consolidato

Norma dell'aggiornamento di ciascun client prima del clipping (`delta_norm_per_client` nei JSON),
10 seed, C = 1. "Tenuta": frazione media dell'aggiornamento che sopravvive al clipping,
min(1, C/norma), sui round 2–10.

| Dataset | Variante | Round 1 | Round 2–10, mediana: μ=0 / 0,01 / 0,1 | Client-round tagliati: μ=0 / 0,01 / 0,1 | Tenuta: μ=0 / 0,01 / 0,1 |
|---|---|---|---|---|---|
| ACN (6 feature) | DP completa | 8,4 | 8,9 / 0,72 / 0,34 | 100% / 16% / 2% | 0,16 / 0,97 / 1,00 |
| ACN (6 feature) | Senza DP | 8,4 | 3,3 / 0,19 / — | 100% / 0% / — | 0,35 / 1,00 / — |
| ACN 4 feature | DP completa | 7,1 | 7,2 / 0,64 / 0,30 | 100% / 11% / 0% | 0,19 / 0,98 / 1,00 |
| ACN 4 feature | Senza DP | 7,1 | 1,9 / 0,11 / 0,06 | 75% / 0% / 0% | 0,59 / 1,00 / 1,00 |
| Scotland | DP completa | 7,7 | 8,1 / 0,64 / 0,30 | 100% / 11% / 0% | 0,17 / 0,98 / 1,00 |
| Scotland | Senza DP | 7,7 | 2,3 / 0,09 / 0,06 | 76% / 0% / 0% | 0,54 / 1,00 / 1,00 |

- Al round 1 FedProx non agisce (non c'è ancora un modello globale da cui tenere i client
  vicini): la norma è 7–8 in tutte le celle e il clipping a C = 1 tiene circa un ottavo
  dell'aggiornamento.
- Dal round 2, con FedAvg e la DP la norma resta intorno a 8–9 e viene sempre tagliata: del
  passo di ogni client sopravvive circa un sesto. Con il rumore lo scostamento cresce ancora (il
  modello globale rumoroso spinge i client a correggerlo con passi grandi).
- FedProx con μ = 0,01 porta la norma sotto C (0,6–0,7): il clipping agisce solo nell'11–16%
  dei client-round e lascia passare il 97–98% dell'aggiornamento. μ = 0,1 elimina il clipping.
- Questo spiega perché FedProx aiuta con solo clipping e con la DP completa, e non con solo
  rumore: il rumore ha la stessa ampiezza qualunque sia la norma.
- Le feature contano: con le stesse sessioni di ACN, le 6 feature danno aggiornamenti più grandi
  delle 4 (senza DP, FedAvg: 3,3 contro 1,9 e 100% contro 75% di client-round tagliati) e una
  perdita senza DP 25 volte più alta (5,1·10⁻⁴ contro 2,0·10⁻⁵). Le due feature in più sono
  `kwh_requested` e `minutes_available`, dichiarate dall'utente e assenti in Scotland. Quanto
  ciascuna feature pesa lo misura l'ablazione (2.4c).

### 2.4c Ablazione delle feature: perché FedProx peggiora o migliora — consolidato (10 seed)

ACN togliendo una feature alla volta (5 feature in ingresso, stesso modello, stessi 10 seed,
10 round, ε = 64 per round, C = 1), FedAvg e FedProx μ = 0,01 e 0,1, con e senza DP. 360 run
di sola utilità (`coda_v3i_mac.csv`). Riferimento a 6 feature: le run della campagna v3.
Perdita sull'holdout al round 10. "Extra" = perdita FedProx − perdita FedAvg senza DP. Intervalli
bootstrap al 95% fra parentesi; p da test di permutazione FedProx contro FedAvg con DP.

| Senza | Perdita no DP, FedAvg | FedProx/FedAvg no DP μ=0,01 | Extra μ=0,01 / 0,1 | FedProx/FedAvg con DP μ=0,01 | μ=0,1 | Norma DP μ=0 / 0,01 | Tagliati μ=0,01 |
|---|---|---|---|---|---|---|---|
| — (6 feature) | 5,1·10⁻⁴ | 2,3 (1,9–2,9) | 6,8·10⁻⁴ / 2,2·10⁻³ | 0,55 (0,37–0,84), p = 0,02 | 0,63, p = 0,08 | 8,9 / 0,72 | 16% |
| energia | 2,1·10⁻⁴ | 4,5 (2,8–6,9) | 7,2·10⁻⁴ / 1,8·10⁻³ | 0,68 (0,36–1,23), p = 0,24 | 0,51, p = 0,06 | 8,6 / 0,69 | 14% |
| potenza | 4,0·10⁻⁴ | 1,9 (1,4–2,6) | 3,7·10⁻⁴ / 1,6·10⁻³ | 0,54 (0,26–1,02), p = 0,08 | 0,55, p = 0,07 | 8,3 / 0,72 | 16% |
| kWh richiesti | 1,4·10⁻⁴ | 5,8 (4,6–7,3) | 6,5·10⁻⁴ / 2,1·10⁻³ | 0,71 (0,37–1,25), p = 0,26 | 0,60, p = 0,09 | 7,9 / 0,64 | 10% |
| minuti disponibili | 1,5·10⁻⁴ | 5,2 (3,5–7,7) | 6,4·10⁻⁴ / 2,7·10⁻³ | 0,75 (0,44–1,25), p = 0,31 | 0,58, p = 0,09 | 8,1 / 0,74 | 14% |
| ora | 1,4·10⁻⁴ | 5,7 (3,9–8,0) | 6,8·10⁻⁴ / 2,0·10⁻³ | 0,66 (0,37–1,16), p = 0,18 | 0,55, p = 0,12 | 8,4 / 0,68 | 13% |
| durata | 2,3·10⁻⁴ | 4,1 (2,3–7,7) | 7,3·10⁻⁴ / 2,1·10⁻³ | 0,60 (0,38–0,90), p = 0,02 | 0,53, p = 0,01 | 8,0 / 0,72 | 16% |

**Senza DP FedProx peggiora sempre, e quanto peggiora dipende dalle feature solo attraverso la
difficoltà del compito.** Il peggioramento assoluto è quasi costante: FedProx aggiunge circa
6,5–7,3·10⁻⁴ a μ = 0,01 e 1,6–2,7·10⁻³ a μ = 0,1, qualunque feature si tolga (unica eccezione la
potenza, 3,7·10⁻⁴). Il rapporto cambia perché cambia il denominatore: togliere kWh richiesti,
minuti disponibili o ora rende il compito 3,4–3,8 volte più facile (perdita FedAvg 1,4·10⁻⁴ contro
5,1·10⁻⁴), e lo stesso peggioramento assoluto diventa un rapporto 5–6 invece di 2. Il termine
prossimale impone un "pavimento" di perdita (frena i client prima che arrivino al loro minimo) che
non dipende da cosa il modello deve ricostruire.

**Con la DP FedProx migliora, e il guadagno non dipende in modo misurabile dalle feature.** Le
stime vanno da 0,54 a 0,75 (μ = 0,01) e da 0,51 a 0,63 (μ = 0,1), con intervalli che si
sovrappongono tutti; a μ = 0,1 il rapporto è sotto 1 in 7 varianti su 7. Il meccanismo è lo
stesso di 2.4b e non cambia con le feature: con FedAvg e DP la norma degli aggiornamenti è
7,9–8,9 in tutte le varianti (sempre tagliata), con FedProx μ = 0,01 scende a 0,64–0,74 (tagliata
nel 10–16% dei client-round). La norma senza DP dipende poco dalle feature (2,6–3,3).
μ = 0,1 è migliore di μ = 0,01 in 5 varianti su 7, uguale senza la potenza e peggiore solo a 6
feature (nessuna differenza fra i due μ è significativa).

**Quanto pesano le feature.** Sulla perdita senza DP molto (fattore 3,8 fra 6 feature e senza
kWh richiesti). Sul costo relativo della DP molto, ma solo perché cambia il denominatore (perdita
DP / no DP con FedAvg: 276 a 6 feature, 660–1900 a 5). Sul guadagno di FedProx con DP no, entro
la precisione di 10 seed.

**Perdita con DP a 5 feature più alta che a 6: non è un effetto della procedura, ma non regge
la correzione per confronti multipli.** Con FedAvg e DP la perdita al round 10 è 0,24–0,27 a 5
feature contro 0,14 a 6 (p = 0,02–0,04 per ciascuno dei sei confronti, tutti contro lo stesso
riferimento; con Holm il più piccolo diventa 0,11). Il riferimento a 6 feature rifatto con la
procedura dell'ablazione (`coda_v3j_mac.csv`, 50 run) dà valori identici bit per bit a quelli
della campagna v3 in tutte le 50 run: gli attacchi non toccano la casualità dell'addestramento e
le run sono riproducibili esattamente a parità di seed. Fino al round 5 le curve con 5 e 6 feature
coincidono (0,11–0,19); dopo, a 5 feature la perdita continua a salire (fino a 0,24–0,28), a 6
feature resta intorno a 0,14–0,19. Non ha una spiegazione meccanica e non lo riportiamo come
risultato; le conclusioni sui rapporti FedProx/FedAvg non ne dipendono, perché sono calcolate
dentro ciascuna variante.

### 2.4d Norma di clipping C = 0,25, 0,5, 2 (ACN, 10 seed, solo utilità) — consolidato (coda_v3h, Windows)

Perdita sull'holdout del modello rilasciato al round 10, DP completa (ε = 64 per round), 10 seed
(42, 123, 456, 789, 1234, 1618, 2025, 2026, 2718, 3141). Riferimento C = 1 da 2.1 (10 seed, ACN a 6 feature).

| C | FedAvg μ=0, media (mediana) | FedProx μ=0,01, media (mediana) | Rapporto mediano FedProx/FedAvg | Seed con FedProx migliore |
|---|---|---|---|---|
| 0,25 | 0,120 (0,126) | 0,017 (0,013) | 0,096 | 10/10 |
| 0,5 | 0,098 (0,087) | 0,024 (0,024) | 0,23 | 9/10 |
| 1 (da 2.1) | 0,142 | 0,078 | 0,51 (rapporto delle medie 0,55, p = 0,02) | 8/10 |
| 2 | 0,232 (0,202) | 0,252 (0,217) | 1,00 (media dei rapporti 1,21) | 4/10 |

Il vantaggio di FedProx con la DP cresce quando C diminuisce e scompare a C = 2. Il rumore è tarato
su ε per round e la sua ampiezza è proporzionale a C: con C piccolo il rumore è più piccolo, quindi
il confronto fra valori di C mescola l'effetto del clipping con quello del rumore. A C = 2 il rumore
domina e il termine prossimale non aiuta più (la norma di FedAvg resta sopra C, ma il guadagno di
FedProx sparisce). Questo va riportato come osservazione sulla sensibilità a C, non come una curva
di ottimo: C = 1 resta quello fissato nel resto della campagna.

A C = 2 il modello è in gran parte distrutto dal rumore (perdite 0,03–0,46). In due seed (1618 e
3141) FedAvg e FedProx finiscono sulla stessa perdita (0,2149 e 0,2024) pur con curve diverse nei
round precedenti: il modello cade in una soluzione degenere che non dipende da μ. Non sono run
duplicate, ma la riga C = 2 va letta come "il rumore domina", non come un confronto fra algoritmi.

### 2.5 Orizzonte fissato a 5 round — confermato con una replica dichiarata prima (10 seed nuovi)

Prima serie (esplorativa). Perdita al round 5, FedProx / FedAvg, DP completa, 10 seed, test di permutazione.

| Dataset | μ = 0,01 | μ = 0,1 |
|---|---|---|
| ACN-Data | 0,59 (p = 0,012) | 0,57 (p = 0,006) |
| Scotland | 0,39 (p = 0,004) | 0,29 (p = 0,0007) |
| ACN 4 feature | 0,56 (p = 0,04) | 0,31 (p = 0,0015) |

Tutti e sei i confronti restano significativi con Holm. L'orizzonte era stato scelto dopo aver
visto le curve (2.4), quindi questa prima serie è esplorativa.

**Replica confermativa** (`coda_v3k_domenico.csv`, 90 run, Mac di Domenico). Ipotesi, test e
seed scritti prima di vedere i dati (`LEGGIMI_v3k.txt`): al round 5, con DP locale ε = 64 per
round e C = 1, FedProx ha perdita minore di FedAvg; permutazione a una coda, Holm sui 6
confronti; seed 7, 99, 314, 999, 1999, 2468, 4321, 5555, 8128, 9973, mai usati prima. Stessi
config della prima serie, solo utilità.

| Dataset | μ = 0,01 | μ = 0,1 |
|---|---|---|
| ACN-Data | 0,40 (Holm p = 0,0005; 9/10 seed) | 0,36 (Holm p = 0,0004; 10/10) |
| Scotland | 0,46 (Holm p = 0,003; 9/10) | 0,32 (Holm p = 0,0005; 10/10) |
| ACN 4 feature | 0,36 (Holm p = 0,0005; 10/10) | 0,30 (Holm p = 0,0004; 10/10) |

"Seed" = seed in cui FedProx ha perdita minore di FedAvg a parità di seed. **L'ipotesi è
confermata in tutti e sei i confronti.**

Al round 10 (secondario) la replica è più debole: ACN 0,75 e 0,69, Scotland 1,00 e 0,40, ACN-4f
1,15 e 0,63 (μ = 0,01 e 0,1); solo Scotland μ = 0,1 ha p < 0,05 senza correzione. Mettendo
insieme le due serie (20 seed, permutazione a una coda, senza correzione):

| Dataset | Round 5: μ = 0,01 / 0,1 | Round 10: μ = 0,01 / 0,1 |
|---|---|---|
| ACN-Data | 0,49 / 0,47 (p < 0,001) | 0,65 (p = 0,008) / 0,66 (p = 0,02) |
| Scotland | 0,42 / 0,31 (p < 0,001) | 0,94 (p = 0,39) / 0,45 (p < 0,001) |
| ACN 4 feature | 0,47 / 0,31 (p < 0,001) | 0,85 (p = 0,22) / 0,53 (p = 0,004) |

Senza DP (`coda_v3l_mac.csv`, stessi 10 seed nuovi) FedProx peggiora sempre, in 10 seed su 10
in tutti i confronti (test appaiato sul log del rapporto, p = 0,002–0,008). Rapporto mediano
FedProx/FedAvg al round 10: ACN 2,2 (μ = 0,01) e 5,6 (μ = 0,1); Scotland 10 e 73; ACN-4f 17 e 73.
Un seed (1999) ha su ACN una perdita 30 volte più alta in tutti e tre i bracci senza DP (stesso
split), quindi per questa serie si usano mediane e test appaiati.

Lettura: con la DP il vantaggio di FedProx è robusto nei primi round e si riduce con i round,
perché il rumore accumulato aumenta perdita e variabilità di entrambi; μ = 0,1 lo conserva a
10 round su tutti e tre i dataset, μ = 0,01 solo su ACN. A 30 round sparisce (2.4). Il valore
di 0,55 a 10 round su ACN della prima serie (2.1) era nella parte favorevole della variabilità:
con 20 seed è 0,65.

Senza DP il divario fra FedAvg e FedProx a 30 round vuol dire che FedProx converge più
lentamente: il termine prossimale frena l'allontanamento dei modelli locali dal globale.
FedAvg si stabilizza (ACN: 4·10⁻⁴ dal round 15), FedProx scende ancora (6,8·10⁻⁴ al round 30).
Con più round il divario probabilmente si ridurrebbe, ma per le domande del paper non serve:
senza DP FedAvg è comunque il migliore, e con la DP più round peggiorano.

### 2.6 Risposta a RQ2

- Il costo della DP per client è grande in tutti i dataset: con μ = 0,01, 66× su ACN, 422×
  su ACN-4f, 536× su Scotland. Quasi tutto viene dal rumore: con solo rumore la perdita è
  10–100 volte quella con solo clipping.
- Senza DP FedAvg è sempre migliore (FedProx peggiora da 2 a 87 volte).
- Con la DP completa FedProx riduce il costo. Con 20 seed μ = 0,1 è uguale o migliore di
  μ = 0,01 su tutti e tre i dataset (round 10: ACN 0,66 contro 0,65, Scotland 0,45 contro 0,94,
  ACN-4f 0,53 contro 0,85); μ = 0,01 basta solo su ACN. μ = 0,001 non fa niente.
- A μ = 0,01 il meccanismo è il clipping: FedProx aiuta molto con solo clip (0,27–0,45) e
  niente con solo rumore. A μ = 0,1 non è così: da soli né il clip né il rumore migliorano,
  il guadagno compare solo con i due insieme.
- Le feature non cambiano il quadro (2.4c): senza DP FedProx aggiunge un peggioramento assoluto
  quasi costante (il rapporto è più alto solo quando il compito è più facile); con la DP il
  guadagno (0,51–0,75) e le norme sono simili togliendo qualunque feature.
- A 5 round il vantaggio di FedProx con la DP è confermato da una replica dichiarata prima, con
  10 seed nuovi: rapporti 0,30–0,46 su tutti e tre i dataset, Holm p ≤ 0,003 (2.5). A 10 round,
  con 20 seed, resta per μ = 0,1 ovunque (0,45–0,66) e per μ = 0,01 solo su ACN (0,65).
- Il vantaggio di FedProx non si conserva a 30 round (rapporti 0,74–0,80, p > 0,3): con la DP
  più round accumulano più rumore e la variabilità fra seed cresce.
- Il vantaggio di FedProx con la DP dipende dalla norma di clipping: forte a C = 0,25 e 0,5, nullo a C = 2 (2.4d).

---

## RQ3 — Unità protetta e superfici di osservazione

### 3.1 Costo dei meccanismi su ACN — consolidato (10 seed)

| Meccanismo | Perdita | Rapporto con no DP μ=0,01 (1,2·10⁻³) | ε (accounting) | Yeom / LiRA / RMIA |
|---|---|---|---|---|
| Session DP σ = 0 (solo clip per esempio) | 1,15·10⁻³ | 1,0× | — | 0,501 / 0,500 / 0,501 |
| Session DP σ = 1 | 3,2·10⁻³ | 2,7× | 8,3 / 8,0 / 55,8 | 0,501 / 0,500 / 0,501 |
| Session DP σ = 2 | 3,5·10⁻³ | 2,9× | 3,4 / 3,3 / 19,8 | 0,500 / 0,500 / 0,501 |
| Session DP σ = 5 | 4,6·10⁻³ | 3,9× | 1,2 / 1,2 / 6,3 | 0,500 / 0,500 / 0,501 |
| Subject DP σ = 2 | 1,10·10⁻² | 9,2× | 32,8 / 35,8 / 344,5 | 0,499 / 0,499 / 0,501 |
| Subject DP σ = 5 | 1,31·10⁻² | 11× | 9,9 / 10,8 / 77,3 | 0,499 / 0,499 / 0,500 |
| Central DP ε = 64 | 7,4·10⁻² | 62× | 428 (totale, 10 round) | 0,500 / 0,501 / 0,500 |
| Local DP ε = 64 | 7,8·10⁻² | 66× | 428 (totale, 10 round) | 0,502 / 0,499 / 0,501 |

Nota: con 5 seed la central DP dava 45×; con 10 seed 62×, vicina alla local DP (la
variabilità fra seed è alta: deviazione standard 0,052).

ε per Caltech / JPL / Office 1. Scotland: session DP σ = 1 8×, subject DP σ = 5 72×,
local DP ε = 64 536×.

### 3.2 Session DP e utenti frequenti — consolidato (analitico)

Con la privacy di gruppo, k sessioni dello stesso utente danno (kε, k e^{(k−1)ε} δ). Con
σ = 5 (ε ≈ 1,2 per Caltech e JPL) il δ di gruppo supera 10⁻² già a k = 6 e supera 1 a
k = 14, la mediana delle sessioni per utente in ACN: la garanzia è vuota per l'utente tipico.

### 3.3 Subject DP sul canary persona — consolidato (5 seed, rumore indipendente)

| σ | ε per soggetto | AUC appaiato aggregatore (r3) | Seed sig. | AUC appaiato rilasciato (r3) | Seed sig. |
|---|---|---|---|---|---|
| 5 | ≈ 29 | 0,499 | 0/5 | 0,529 | 0/5 |
| 8 | ≈ 15,6 | 0,522 | 0/5 | 0,562 | 1/5 |
| 1000 | ≈ 0,08 | 0,476 | 0/5 | 0,440 | 0/5 |

Con la DP per soggetto il canary persona non è riconosciuto né dall'aggregatore né dal modello
rilasciato, già a σ = 5 (ε ≈ 29 per soggetto). Senza DP, nella stessa configurazione, l'AUC
appaiato sull'aggregatore è 0,80 in 10 seed su 10 (1.3). In condizioni naturali l'attacco al
soggetto non vede niente con nessun meccanismo (1.1).

### 3.3b Session DP sul canary persona — misurato (5 seed, rumore indipendente)

`coda_v3l_mac.csv`, config `experiment_canary_multisite_record_s5.yaml`: stesso canary persona di
1.3 e 3.3 (JPL, 20 + 20 utenti, 30 copie, spostamento 0,5, 3 round × 1000 epoche), con la DP per
sessione σ = 5, C = 1 (DP-SGD per esempio, GroupNorm come la subject DP). AUC appaiato sulle
sessioni di training, seed significativi su 5.

| Round | Aggregatore | Modello rilasciato |
|---|---|---|
| 1 | 0,740, 4/5 | 0,458, 0/5 |
| 2 | 0,628, 2/5 | 0,505, 0/5 |
| 3 | 0,590, 1/5 | 0,544, 0/5 |

Confronto all'aggregatore, round 1 / round 3: senza DP 0,999 (10/10) / 0,802 (10/10); session DP
σ = 5 0,740 (4/5) / 0,590 (1/5); subject DP σ = 5 0,516 (0/5) / 0,499 (0/5); subject DP σ = 8
0,576 (1/5) / 0,522 (0/5).

Lettura: la session DP riduce il segnale ma non lo toglie; all'aggregatore il canary persona è
ancora riconosciuto al round 1 in 4 seed su 5, mentre con la subject DP allo stesso σ no. È la
verifica sperimentale di 3.2: la garanzia per sessione non copre un utente con molte sessioni (qui
600 sessioni di training della stessa persona). Al round 3 il segnale si indebolisce (1/5) e sul
modello rilasciato non c'è. Con 5 seed il confronto fra session e subject DP è indicativo, non un
test formale. ε per sessione della persona canary (JPL: 41 078 sessioni di training,
q = 32/41 078, 3 × 1000 epoche, 3,85 milioni di passi, σ = 5, RDP): ε ≤ 1,3 con δ = 10⁻⁵, cioè
una garanzia per sessione forte quanto quella di RQ3; per la persona (600 sessioni) la privacy
di gruppo non dà nessuna garanzia.

### 3.4 Risposta a RQ3

- La superficie conta quanto il meccanismo: la central DP protegge il modello rilasciato ma
  non l'aggiornamento visto dal server; la local DP protegge entrambi.
- La session DP costa poco (2,7–3,9×) ma non protegge gli utenti con molte sessioni: lo dice
  l'analisi (3.2) e lo conferma il canary persona, riconosciuto dall'aggregatore al round 1 in
  4 seed su 5 con session DP σ = 5 e in nessuno con subject DP σ = 5 (3.3b).
- La subject DP protegge l'intero profilo a un costo intermedio (9–11× su ACN, 72× su Scotland),
  e nasconde il canary persona già a σ = 5.

---

## Cosa manca

| Cosa | Dove | Stato |
|---|---|---|
| Subject DP sul canary persona con rumore indipendente (σ = 5, 8, 1000) | Mac di Domenico | fatto (3.3) |
| Local e central sul canary persona allo spostamento 0,5 con rumore indipendente | Mac di Domenico | fatto (1.3) |
| Local e central allo spostamento 0 con rumore indipendente | Mac principale (seed 42–1234), Windows (1618–3141) | fatto su 5 seed (1.3) |
| Meccanismi di RQ3 su ACN a 10 seed | Mac principale | fatto (3.1) |
| Conferma dell'inversione di LiRA e RMIA online (dump dei canary e un seed) | Mac di Domenico | fatto (1.4) |
| Controllo positivo su Scotland (canary per sessione a Shetland Islands, 5 seed) | Mac di Domenico | fatto (1.2b) |
| Diagnostica dell'inversione (seed 42, con dump dei canary e con ombre sull'universo del client) | Mac di Domenico | fatto (1.4) |
| Norma di clipping C = 0,25, 0,5, 2 per FedAvg e FedProx (ACN, 10 seed, solo utilità) | Windows | fatto (2.4d) |
| Ablazione delle feature (ACN senza una feature alla volta, 10 seed) | Mac principale | fatto (2.4c) |
| Riferimento a 6 feature con la procedura dell'ablazione | Mac principale | fatto: identico bit per bit alla campagna v3 (2.4c) |
| Curve per round (convergenza) di FedAvg e FedProx | `loss_per_round.csv` | fatto (2.4) |
| RQ2 a un orizzonte di 5 round | dati esistenti | fatto, esplorativo (2.5) |
| Replica confermativa a 5 round (10 seed nuovi, ipotesi dichiarata in `LEGGIMI_v3k.txt`) | Mac di Domenico | fatto: confermata (2.5) |
| Stessa replica senza DP (FedProx peggiora? stessi 10 seed nuovi) | Mac principale | fatto: confermato (2.5) |
| Session DP σ = 5 sul canary persona (verifica empirica di 3.2, 5 seed) | Mac principale | fatto (3.3b) |

Approfondimenti (non necessari per le RQ, decisi il 2026-10-29):

| Cosa | Dove | Stato |
|---|---|---|
| Session DP σ = 5 sul canary persona, altri 5 seed (test formale con 10 seed) | Mac principale | `coda_v3m_mac.csv` |
| RQ2 su Scotland a 32 council (FedAvg/FedProx, con e senza DP, 10 seed, solo utilità) | Mac principale | `coda_v3m_mac.csv` |
| Curva ε: session DP σ = 1, 2, 10 e subject DP σ = 2 sul canary persona (5 seed) | Mac principale | `coda_v3m_mac.csv` |
| Avversario B su Scotland (regime naturale, 15 run) e sul canary per sessione (30 run) | Mac di Domenico | fatto (1.1b) |
| Quante sessioni servono: canary persona con 1, 5, 10, 30 copie, senza DP e session DP σ = 5 | Mac di Domenico | fatto (1.3b, 5 seed: potenza bassa) |
| Canary persona allo spostamento 0,5, central e local con rumore indipendente, seed 1618–3141 (tabella 1.3 a 10 seed) | Windows | `coda_v3p_windows.csv` |
| Clipping a rumore uguale: C = 0,25, 0,5, 2 con ε = 9,83, 23,94, 190,99 (stesso σ di C = 1, ε = 64), FedAvg e FedProx, 10 seed | Windows | `coda_v3p_windows.csv` |
