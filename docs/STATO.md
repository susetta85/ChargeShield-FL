# ChargeShield-FL — Stato del progetto

> **Stato: documento CANONICO.** Aggiornato il 2026-09-22. Riporta solo numeri
> presenti in `risultati/` alla data; dove un numero non è verificabile dai JSON
> grezzi lo dice. La linea scientifica è in
> `ChargeShield_FL_spina_dorsale_consolidata.md`; gli esperimenti da fare in
> `ESPERIMENTI.md`.

## 1. La domanda e il contributo, in breve

Tre operatori reali di ricarica EV (Caltech, JPL, Office 1, dataset ACN-Data)
addestrano in federated learning un autoencoder da 570 parametri su 6 feature di
sessione, FedProx con mu = 0.01, 10 round da 50 epoche locali. Un aggregatore
honest-but-curious esegue attacchi di membership inference sui record. RQ1 chiede:
in quali condizioni la differential privacy riduce la capacità di questi attacchi,
e a quale costo sul modello. RQ3 (FedAvg contro FedProx) e RQ2 (partizione IID
contro per sito) seguono.

Il contributo atteso è una misura, non un meccanismo: costo e rischio per cella,
su dati operativi reali multi-operatore, con il rischio letto anche per singolo
record e con uno strumento la cui sensibilità è dimostrata prima di leggere i
risultati.

## 2. Glossario dei termini che non si leggono da soli

| termine | significato |
|---|---|
| client-level | clip L2 più rumore gaussiano una volta per round sull'update del client (`GradientManager`). Protegge il contributo del client, non il record. Non è DP-SGD: con 50 epoche locali l'ε per round è un parametro di calibrazione, non una garanzia verificata. |
| record-level | DP-SGD vero, clipping per esempio a ogni passo, rumore sulla somma (`AutoencoderTrainer._train_step_record_dp`). Protegge il record, la stessa unità che gli attacchi misurano. Il suo budget è `epsilon_record_dp`, non il campo `epsilon`. |
| A0, A1, A2, A3 | cosa vede l'avversario sull'update del client: A0 grezzo senza DP (riferimento); A1 grezzo prima di clip e rumore (placement dp-fedavg, più forte del threat model); A2 clippato prima del rumore (central); A3 clippato e rumorizzato (local). Sono ordinati per informazione. Valgono per il client-level; per il record-level collassano in uno. |
| superficie globale | Yeom e Shadow attaccano il modello aggregato, visibile a ogni partecipante; LiRA attacca l'update del singolo client. Sono due superfici indipendenti, non una gerarchia. |
| loss grezza | errore di ricostruzione del modello bersaglio sul campione, senza calibrazione. È la metrica che oggi regge; LiRA calibrato è degenere (vedi 4.4). |
| z (vulnerabilità per record) | quante deviazioni standard il numero di record segnalati dista dal livello di caso, ottenuto permutando i percentili dentro ogni seed. z ≈ 0 caso; z > 3 eccesso reale. Non misura la gravità. |
| record segnalati | sessioni membro in almeno 2 seed con percentile medio ≥ 90 e minimo ≥ 75. Da leggere sempre accanto agli attesi per caso. |
| AUC max teorica | tetto che (ε, δ)-DP pone su qualunque attacco, sull'ε cumulativo T·ε. A ε_tot = 10 vale 0.99996: il bound è vacuo. |
| regime naturale / canary | naturale: dati come sono; canary: memorizzazione indotta con template duplicati, capacità 3.3x, feature temporale quasi univoca, 1000 epoche. Il canary valida lo strumento e non certifica il null naturale. |

## 3. Cosa è stato fatto e regge

### 3.1 Campagna principale, regime naturale (client-level)

Fonte: `risultati/Matrice_sintesi.xlsx`, foglio `Utility_privacy_limite`. AUC-ROC,
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

Lettura. In media nessun attacco distingue membri da non membri, nemmeno senza DP.
In ogni cella a ε ≤ 1 la loss è oltre 100 volte il riferimento: il modello non
impara, quindi quel nullo non distingue "la DP protegge" da "non c'è nulla da
proteggere". Le sole celle con utility residua, ε = 8 e 16, hanno un seed solo.
Dp-fedavg e local coincidevano bit a bit fino al 15 settembre (stesso codice);
dopo la modifica "Strada B" dp-fedavg attacca l'update grezzo: quali celle siano
post modifica va verificato run per run nel registro. Il test di equivalenza TOST
a margine 0.02 è soddisfatto in tutte le celle; il Wilcoxon a 5 seed non può
essere significativo per costruzione.

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
sistematico, mentre l'AUC aggregata della stessa cella è 0.51. Sotto DP a ε = 1
l'eccesso scompare, ma con utility distrutta: le due spiegazioni non sono
separabili con i dati attuali. È il motivo per cui lo sweep di ε in 2-16 è il
prossimo esperimento. Nota: i due script che calcolano il percentile non usano la
stessa definizione (411 contro 412), da unificare prima di citare il numero.

### 3.3 Lo strumento è validato, ma su un sito

Fonte: `docs/CanaryPositiveControl.md` sezione 6, campagna bilanciata post fix.
Office 1, k = 20 template membro e 20 non membro, 30 duplicati, scambio dei ruoli,
baseline a inizializzazione casuale, 5 seed, regime canary.

| metrica | braccio A | braccio B |
|---|---|---|
| loss grezza, AUC media sui 3 round | 0.745 | 0.715 |
| LiRA calibrato | 0.651 | 0.696 |

Trenta round su trenta sopra 0.5 sulla loss grezza, tutte le somme A+B sopra 1,
Δ medio per seed +0.23 con t(4) = 9.9. Il segnale è simmetrico allo scambio dei
ruoli, quindi è appartenenza e non difficoltà intrinseca dei template. Tre
artefatti sono stati intercettati dai controlli prima di questo risultato: pool
diversi per i due gruppi, scoring asimmetrico fra le classi, effetto
dell'assegnazione sotto DP. Caltech e JPL hanno solo run a seed 42 e senza il
protocollo bilanciato; il config per ChargePlace Scotland esiste e non è mai
stato eseguito.

### 3.4 Record-level DP

Implementato il 2026-09-16, usato solo in regime canary a Office 1: a σ = 5 la
loss grezza dei canary scende da 0.68 a 0.53 con il modello ancora addestrabile
(loss 0.0045). Un seed, nessun braccio swap. L'ε dichiarato di 7.15 viene da un
accountant che conta un round su tre e usa l'n globale: va ricalcolato
(segnalazione 4). Su dati naturali non esiste nessuna run.

### 3.5 Infrastruttura che funziona

Simulazione single-process e deployment NVFlare a 5 container convergono sullo
stesso nullo a ε = 1 per central e local. 28 dump NVFlare a ε ≤ 1 attendono
rianalisi: rinviata, perché stanno tutti nel regime con utility distrutta. Il
Privacy Auditor è strumentato (overhead 0.3-0.7 ms per round) ma la sua telemetria
non distingue una cella con memorizzazione da una senza: è un contatore di
budget, non una misura di rischio. Il ByzantineDetector non è mai stato eseguito
end-to-end.

### 3.6 Unità protetta e persona

Su ACN-Data l'identificativo utente copre il 72.6% delle sessioni (Caltech 52%,
JPL 93%, Office 1 35%); 1 028 utenti contribuiscono in mediana 14 sessioni e
l'8.5% ricarica in più di un sito. Né record-level né client-level limitano il
contributo di una persona.

## 4. Cosa manca, in ordine

1. **E-A**: sweep ε in {2, 4, 8, 16}, 5 seed, con dump per campione. Le uniche
   celle in cui DP attiva e modello funzionante possono coesistere. Circa 2 giorni
   di macchina.
2. **E-B**: record-level su dati naturali, 5 seed, dopo la correzione
   dell'accountant. Prima una run di prova per misurare il tempo.
3. **E-C**: analisi worst-case su tutte le celle nuove, solo calcolo.
4. Canary bilanciato almeno su Caltech.
5. **E-D**, RQ2: partizione IID contro per sito, config pronti.
6. **E-E**, RQ3: mu = 0 contro 0.01 sulla configurazione scelta da E-A.
7. Rianalisi NVFlare limitata a 2-3 celle al punto operativo indicato da E-A.

## 5. Cosa non entra nel paper

ML Plane e Privacy Auditor come contributi (restano infrastruttura), il punteggio
PES, il ByzantineDetector, l'attacco FedMIA-gradient, il secondo dataset.

## 6. Limiti da dichiarare sempre

Il nullo naturale è "nessun segnale rilevato dagli attacchi valutati nelle
condizioni provate", non "nessuna perdita". Il canary valida la sensibilità in un
regime lontano da quello naturale. L'ε client-level è un'etichetta di
configurazione. La qualità del modello come rilevatore di anomalie non è misurata:
ACN-Data non ha etichette di intrusione.
