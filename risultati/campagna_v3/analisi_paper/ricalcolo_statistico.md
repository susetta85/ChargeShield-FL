# Ricalcolo statistico dei confronti di utilità e dei canary (2026-11-07)

Dati: `risultati/campagna_v3/*/riepilogo_run.csv`, `loss_per_round.csv` e `canary_persona.csv` (tutte le macchine). Script: `scripts/analisi_paper/utility_tests.py` (confronti di utilità, replica al round 5, clipping, round 30; risultati in `utility_tests.json`) e `scripts/analisi_paper/canary_summary.py` (AUC appaiata e su una federazione dei canary per soggetto; risultati in `canary_summary.csv`).

## 1. Cosa non tornava

I p-value pubblicati nella Tabella utilità vengono da un test di permutazione **non appaiato** sulla differenza delle medie. Il metodo dichiara invece confronti appaiati per seed, e il disegno è davvero appaiato: a parità di seed FedAvg e FedProx condividono partizione e inizializzazione. Il ricalcolo conferma l'osservazione: i p pubblicati coincidono con quelli non appaiati (per esempio ACN 4 feature, μ = 0,01: 0,060 non appaiato contro 0,027 appaiato).

## 2. Procedura adottata

- **Test:** permutazione esatta a scambio di segno sulle differenze per seed (FedProx − FedAvg), statistica = differenza media, tutte le 2ⁿ assegnazioni (1024 con 10 seed, 1 048 576 con 20).
- **Code:** due code nella Tabella utilità e nei confronti a 30 round; una coda solo dove la direzione era fissata prima delle run (replica a 5 round, studio sulla norma di clipping).
- **Effetto:** rapporto delle medie, più il numero di seed in cui FedProx ha perdita minore. Lo studio sul clipping mantiene il rapporto mediano per seed, dichiarato in didascalia.
- **Famiglie (Holm):**
  - Tabella utilità, 3 client: una famiglia per blocco di configurazione (senza DP 5 confronti, solo clipping 5, solo rumore 5, DP completa 6);
  - Scotland 32: senza DP (2 confronti) e DP completa (2 confronti, 20 seed);
  - replica a 5 round: 6 confronti, una coda (già così nella versione inglese);
  - clipping a rumore uguale: 3 soglie, una coda.
- Con 10 seed il p più piccolo possibile è 2⁻⁹ ≈ 0,002 (due code) prima della correzione: le voci «p < 0,001» non sono più possibili.

## 3. Tabella utilità ricalcolata

| Confronto | Rapporto delle medie | Seed FedProx migliore | p non appaiato (pubblicato) | p appaiato esatto | p Holm |
|---|---|---|---|---|---|
| **DP completa** | | | | | |
| ACN μ = 0,01 | 0,55 | 8/10 | 0,020 | 0,021 | 0,086 |
| ACN μ = 0,1 | 0,63 | 7/10 | 0,080 | 0,088 | 0,18 |
| Scotland μ = 0,01 | 0,87 | 9/10 | 0,56 | 0,17 | 0,18 |
| Scotland μ = 0,1 | 0,49 | 10/10 | 0,014 | 0,002 | **0,012** |
| ACN 4f μ = 0,01 | 0,65 | 8/10 | 0,060 | 0,027 | 0,086 |
| ACN 4f μ = 0,1 | 0,47 | 10/10 | 0,003 | 0,002 | **0,012** |
| **Solo clipping** | | | | | |
| ACN μ = 0,01 | 0,27 | 10/10 | < 0,001 | 0,002 | **0,010** |
| Scotland μ = 0,01 | 0,45 | 8/10 | 0,006 | 0,020 | 0,059 |
| Scotland μ = 0,1 | 1,66 | 1/10 | 0,015 | 0,010 | **0,039** |
| ACN 4f μ = 0,01 | 0,33 | 6/10 | 0,035 | 0,031 | 0,063 |
| ACN 4f μ = 0,1 | 1,81 | 1/10 | 0,10 | 0,094 | 0,094 |
| **Solo rumore** | | | | | |
| ACN μ = 0,01 | 0,85 | 6/10 | 0,62 | 0,36 | 1,00 |
| Scotland μ = 0,01 | 1,85 | 2/10 | 0,050 | 0,053 | 0,26 |
| Scotland μ = 0,1 | 1,55 | 3/10 | 0,23 | 0,32 | 1,00 |
| ACN 4f μ = 0,01 | 1,06 | 5/10 | 0,86 | 0,86 | 1,00 |
| ACN 4f μ = 0,1 | 0,96 | 5/10 | 0,91 | 0,88 | 1,00 |
| **Senza DP** (tutti) | 2,33–86,9 | 0/10 | < 0,001 | 0,002 | **0,010** |
| **Scotland 32, senza DP** | 14,2 e 54,9 | 0/10 | < 0,001 | 0,002 | **0,004** |
| **Scotland 32, DP, μ = 0,01** (20 seed) | 0,57 | 14/20 | 0,036 | 0,048 | **0,048** |
| **Scotland 32, DP, μ = 0,1** (20 seed) | 0,43 | 16/20 | 0,001 | 0,001 | **0,003** |

Cosa cambia nelle conclusioni:

- **DP completa a 10 round, 3 client:** significativi dopo Holm **2 confronti su 6** (Scotland e ACN 4f con μ = 0,1), non 3. Senza correzione sono 4 su 6. Il confronto ACN μ = 0,01 (0,55), citato in più punti, non è più significativo dopo Holm (0,086).
- **Solo clipping:** l'effetto di FedProx è netto su ACN; su Scotland con μ = 0,1 FedProx peggiora (1,66, significativo).
- **ACN a 30 round:** 0,80, p appaiato 0,29 (prima 0,46).
- I rapporti senza DP ricalcolati dalle medie esatte sono 4,51 (non 4,50), 20,1 (non 20,0), 16,3 (non 16,0), 86,9 (non 87,0) e 14,2 e 54,9 a 32 client.

## 4. Studio sulla norma di clipping (rumore uguale)

Erano già confronti appaiati, ma con 20 000 scambi di segno casuali. Con l'enumerazione esatta, una coda e Holm sulle 3 soglie: C = 0,25 p = 0,003; C = 0,5 p = 0,014; C = 2 p = 0,021 (prima 0,003, 0,015, 0,020). Rapporti mediani invariati (0,80, 0,56, 0,35). Il rapporto delle medie sarebbe 0,69, 0,54, 0,55.

## 5. Canary: AUC appaiato e attacco su una sola federazione

L'AUC appaiato del canary persona usa per ogni utente la differenza fra le sue perdite nei due bracci: uno dei due è l'addestramento controfattuale senza quell'utente, che l'aggregatore di una federazione reale non ha. È quindi una **misura di sensibilità del controllo**, non un attacco eseguibile. Il codice salva anche l'AUC di ciascun braccio da solo (`somma_AB`), cioè l'attacco su una sola federazione: l'aggregatore ordina gli utenti candidati per perdita sull'aggiornamento ricevuto. La media dei due bracci, con test a una coda fra i seed su AUC − 0,5:

| Cella (round 3) | AUC appaiato (controllo) | AUC su una federazione, aggregatore | p fra i seed | AUC su una federazione, modello rilasciato | p |
|---|---|---|---|---|---|
| JPL senza DP (10 seed) | 0,802 | 0,671 | 0,001 | 0,598 | 0,001 |
| JPL DP centrale (10) | 0,722 | 0,590 | 0,001 | 0,528 | 0,015 |
| JPL DP locale (10) | 0,504 | 0,514 | 0,22 | 0,516 | 0,22 |
| JPL sessione σ = 1 (5) | 0,595 | 0,588 | 0,062 | 0,554 | 0,031 |
| JPL sessione σ = 2 (5) | 0,551 | 0,545 | 0,125 | 0,535 | 0,19 |
| JPL sessione σ = 5 (10) | 0,614 | 0,580 | 0,002 | 0,535 | 0,023 |
| JPL sessione σ = 10 (5) | 0,617 | 0,572 | 0,031 | 0,543 | 0,125 |
| JPL soggetto σ = 2 (5) | 0,542 | 0,511 | 0,125 | 0,505 | 0,41 |
| JPL soggetto σ = 5 (5) | 0,499 | 0,492 | 0,75 | 0,505 | 0,34 |
| Scotland senza DP (5) | 0,778 | 0,646 | 0,031 | 0,588 | 0,031 |
| Scotland sessione σ = 5 (5) | 0,598 | 0,572 | 0,063 | 0,500 | 0,50 |
| Scotland soggetto σ = 5 (5) | 0,543 | 0,504 | 0,34 | 0,499 | 0,69 |

Al round 1, sull'aggregatore: JPL senza DP 0,973; sessione σ = 5 0,593 (p = 0,002); soggetto σ = 5 0,499; Scotland sessione 0,560 (p = 0,031).

Lettura: l'attacco su una sola federazione dà valori più bassi del controllo appaiato ma lo **stesso ordine**: DP centrale e DP per sessione restano rilevabili dall'aggregatore, DP locale e DP per soggetto no. Un dettaglio nuovo: con la DP centrale il modello rilasciato conserva un segnale debole (0,528, p = 0,015). Il canary per sessione è già un attacco su una sola federazione: la somma dei due bracci è il doppio della media.
