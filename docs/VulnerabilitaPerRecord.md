# Vulnerabilità per record — analisi worst-case cross-seed

Stato: documento OPERATIVO. Aggiornato al 2026-09-21. **Primo segnale di membership su dati naturali** trovato in
questo progetto.
Codice: `scripts/analyze_worst_case_vulnerability.py` (conteggio),
`scripts/worst_case_livello_di_caso.py` (livello di caso).
Risultati: `risultati/worst_case/`, foglio `Worst_case_per_record` in
`risultati/Matrice_sintesi.xlsx`.

---

## 1. La domanda che l'AUC aggregata non può rispondere

Tutte le metriche usate finora — AUC-ROC, TPR@FPR fisso, MIA Advantage,
matrice di confusione — sono **aggregate**: descrivono quanto bene un attacco
separa i membri dai non-membri *in media, sulla popolazione*.

Un risultato nullo aggregato (AUC ≈ 0.50) dice che **in media** non c'è
separazione. Non dice che nessun record sia esposto. Se 400 sessioni su 17.000
fossero sistematicamente riconoscibili e le altre no, l'AUC resterebbe a 0.50 e
quelle 400 persone sarebbero comunque identificabili.

È la critica di Carlini et al. 2022: *la privacy non è una metrica del caso
medio*. Il risultato di questa pagina è ciò che si vede quando si smette di
guardare la media.

---

## 2. Come si misura

Ogni seed produce un dump per-campione (`--per-sample-dump`) con, per ogni
sessione reale, il punteggio composto dell'attacco LiRA e se quella sessione
era membro in quel seed.

Ogni seed ha uno split train/holdout indipendente, quindi la stessa sessione è
membro in alcuni seed e non-membro in altri. La chiave stabile fra seed è
`session_id` (il campo `sessionID` di ACN-Data): `id(sample)` non servirebbe,
è valido solo dentro un processo.

**Criterio di segnalazione.** Una sessione è marcata come sistematicamente
esposta se, considerando solo i seed in cui era membro:

- è membro in almeno **2** seed (serve evidenza cross-seed);
- il suo percentile medio è **≥ 90** (decile alto del punteggio d'attacco);
- il suo percentile minimo è **≥ 75** (mai sotto il terzo quartile: esclude chi
  finisce in alto una volta per caso e in basso l'altra).

---

## 3. Che cos'è lo z, e perché senza di esso il conteggio non significa nulla

Il conteggio da solo è inutilizzabile. Anche se nessun record fosse davvero
esposto, **qualche sessione finirebbe in alto più volte per puro caso**: con
migliaia di sessioni e una soglia al 90° percentile, un certo numero di
coincidenze è garantito. Dire "411 record vulnerabili" senza un termine di
paragone è dare un numero, non un risultato.

Il **livello di caso** è quante sessioni il criterio segnalerebbe se non
esistesse alcuna corrispondenza fra seed. Si ottiene **permutando i percentili
dentro ogni seed**: si mescolano le etichette `session_id` fra i punteggi di
quel seed, lasciando i punteggi dove sono.

Questa permutazione è quella giusta perché:

- **conserva** la distribuzione dei punteggi di ogni seed, quindi il criterio
  vede esattamente la stessa scala e la stessa forma;
- **distrugge** solo la corrispondenza *fra* seed, che è precisamente ciò che
  il criterio misura.

Ripetendo la permutazione molte volte (qui 200) si ottiene la distribuzione
del conteggio sotto ipotesi nulla: una media e una deviazione standard.

Lo **z** è quante deviazioni standard il conteggio osservato dista da quella
media:

$$z = \frac{\text{osservati} - \text{media dei conteggi permutati}}{\text{dev.\ std.\ dei conteggi permutati}}$$

Lettura:

- **z ≈ 0** → il conteggio è quello che il caso produce. Nessuna vulnerabilità
  per record rilevabile *con questo criterio*.
- **z > 3** → eccesso reale: esiste un sottoinsieme di record sistematicamente
  nel decile alto su seed indipendenti.

Lo z non misura *quanto grave* sia l'esposizione, solo quanto sia improbabile
che il conteggio osservato venga dal caso. La gravità va letta sull'eccesso
assoluto e sulla percentuale.

---

## 4. Il risultato

Cinque seed per cella, 17.561 sessioni presenti come membro in almeno due seed,
200 permutazioni.

| cella | record segnalati | attesi per caso | z | % osservata | % attesa |
|---|---|---|---|---|---|
| **no-DP** | **412** | 289.9 (sd 16.2) | **7.56** | 2.35% | 1.65% |
| dp-fedavg ε=1.0 | 290 | 289.8 (sd 15.9) | 0.02 | 1.65% | 1.65% |
| central ε=1.0 | 299 | 289.9 (sd 14.7) | 0.62 | 1.70% | 1.65% |
| local ε=1.0 | 291 | 289.8 (sd 17.3) | 0.07 | 1.66% | 1.65% |

**Senza DP l'eccesso è reale e netto**: 412 record contro 290 attesi, cioè
circa **122 sessioni in più** del caso, con z = 7.56. Nelle stesse celle l'AUC
aggregata vale 0.5096: il segnale è invisibile alla metrica media e visibile
solo scendendo al record.

Sotto DP, in tutte e tre le posizioni di osservazione, lo z torna a zero.

---

## 5. Il caveat, che va sempre riportato accanto al risultato

**Lo z ≈ 0 sotto DP non prova che la DP protegga i record.**

In tutte e tre le celle DP l'utility del modello è distrutta: la loss finale
sull'holdout è fra 108 e 140 volte quella senza DP (vedi il foglio
`Utility_privacy_limite`). Un modello che non impara non memorizza, e quindi
non espone. Le due spiegazioni

1. la DP ha rimosso l'esposizione;
2. il modello non ha imparato nulla, quindi non c'era esposizione da rimuovere;

**non sono separabili con i dati attuali**, perché non esiste nessuna cella in
cui la DP sia attiva e il modello funzioni ancora.

È questa la ragione per cui lo sweep di ε nella zona 2–16 (`config/
experiment_rq1_eps{2,4,8,16}.yaml`) non è un raffinamento ma la condizione
necessaria perché questo risultato diventi una risposta a RQ1. Serve almeno una
cella con DP attiva e utility residua comparabile al riferimento.

Secondo limite, minore ma da dichiarare: il criterio guarda il **decile alto**.
Un'esposizione concentrata più in alto (top 1%) o distribuita diversamente
richiederebbe soglie diverse. La scelta 90/75 è dichiarata in anticipo e non è
stata ottimizzata sui risultati.

---

## 6. Riprodurre

```bash
# conteggio e top-k (per gruppo)
python3 scripts/analyze_worst_case_vulnerability.py \
    experiments/nodp-sweep2/per_sample_seed*.json \
    --output risultati/worst_case/nodp-sweep2.json

# conteggio CONTRO livello di caso (piu' gruppi insieme)
python3 scripts/worst_case_livello_di_caso.py \
    --gruppo "no-DP=experiments/nodp-sweep2" \
    --gruppo "dp-fedavg_eps1=experiments/dp-sweep4" \
    --gruppo "central_eps1=experiments/central-sweep5" \
    --gruppo "local_eps1=experiments/local-sweep3" \
    --permutazioni 200 --output risultati/worst_case/livello_di_caso.json

# riversare nelle matrici
python3 scripts/genera_matrici_faseA.py
```

I dump per-campione si producono aggiungendo `--per-sample-dump <file>` a
`run_experiments.py`. Senza quelli, per una cella questa analisi non è
possibile: è il motivo per cui i comandi dello sweep di ε lo includono.
