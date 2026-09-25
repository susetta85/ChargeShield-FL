# Regole di lavoro — ChargeShield-FL

Nota per Claude: leggi questo file prima di iniziare a lavorare sul progetto.
Vale per qualunque assistente AI: se non sei Claude Code, il dottorando te lo
incolla come prima istruzione.

## 1. Cosa leggere all'avvio, e cosa no

Tre documenti bastano per capire linea, stato e prossimi passi. Leggerli in
quest'ordine e non leggere altro finché il compito non lo richiede:

1. `docs/ChargeShield_FL_spina_dorsale_consolidata.md`, la guida scientifica
   canonica: obiettivo, RQ1-RQ3, fasi A-F, regole. RQ1 è al centro; l'unità
   protetta, client contro record, è un suo fattore.
2. `docs/STATO.md`, a che punto siamo: risultati verificati, glossario.
3. `docs/ESPERIMENTI.md`, cosa lanciare, in che ordine, con quali prerequisiti.

Su richiesta: `docs/SISTEMA.md` per cosa esiste davvero nel codice,
`docs/Segnalazioni_tecniche_2026-09-22.md` per i bug aperti, gli altri file in
`docs/` come riferimento (ognuno dice in testa a cosa serve), `docs/SprintLog.md`
solo per ricostruire la storia. I documenti superati sono stati eliminati il
2026-09-22 e restano solo nella storia git: non recuperarli per pianificare o
citare numeri.

## 2. Dove vivono i numeri

Solo in `risultati/`: `Matrice_sintesi.xlsx`, `matrice_run_completati.xlsx`,
`matrice_confronti.xlsx`, `decision_matrix_ACN_membership_DP.xlsx`,
`worst_case/*.json`, generati da `scripts/genera_matrici_faseA.py` dai JSON in
`experiments/` e, per le run di altre macchine (oggi E-D), in
`experiments_altre_macchine/`. Queste cartelle e `logs/` non sono versionate: se mancano nel
checkout, si legge dalle matrici e si dichiara che i JSON grezzi non sono stati
riverificati. Un numero che non sta né in `risultati/` né in un JSON letto adesso
non si scrive. Log incollati in chat e commenti nel codice non sono una fonte: in
questo progetto sono stati più volte più ottimisti del codice.

## 3. Protocollo di sessione

**All'inizio.** Prima di proporre o eseguire un'attività, dire quale RQ e quale
esperimento di `ESPERIMENTI.md` serve, e quale conclusione può cambiare (guida,
sezione 11). Se la richiesta contraddice la guida o `ESPERIMENTI.md`, per esempio
modificare lo scorer LiRA mentre lo sweep di ε non è chiuso, cambiare threat model
o unità protetta, aggiungere una RQ: dirlo prima di agire e chiedere conferma
esplicita. Sono decisioni del supervisore, non del dottorando né dell'assistente.

**Durante.**
- Una run alla volta sulla macchina degli esperimenti; niente lanci in parallelo.
- Non toccare il codice di scoring, DP, split o accounting mentre una campagna è
  in corso, e mai senza dire quali celle già prodotte diventano non confrontabili.
- "Verificato" significa eseguito con dati reali. Un `py_compile` o la suite di
  test non-torch non rendono verificato un risultato: scriverlo.
- Un config nuovo per ogni cella, con in testa RQ, fase e cosa cambia rispetto a
  `config/experiment.yaml`; indice in `config/README.md`.
- Non creare nuovi documenti in `docs/`: si aggiornano STATO, ESPERIMENTI,
  SISTEMA o le Segnalazioni. Ogni documento porta in testa una riga di stato.
- Non cancellare né spostare nulla in `experiments/`.
- Regime naturale e canary non si mescolano: il canary valida lo strumento, non
  certifica il null naturale (guida, Fase C).
- Un bug trovato va nelle Segnalazioni con file e riga, anche se corretto subito.

**Alla fine.** Una voce di al massimo dieci righe in `docs/SprintLog.md`, sopra la
più recente: RQ, cosa è stato fatto, dove sono le evidenze, cosa cambia in una
conclusione, prossimo passo. Se le matrici sono cambiate, aggiornare `STATO.md`
dalle matrici, non dai log. Commit con un messaggio che dice la stessa cosa.

## 4. Overleaf è la fonte di verità per il paper

Il paper LaTeX vive sul progetto Overleaf **ChargeShield-FL_DSN2027_paper_skeleton**
(project id `6aa95d88495b72599361110b`). La cartella `docs/paper/latex_dsn2027/`
in questo repository è **solo un mirror** per il versionamento git: non è la
copia autoritativa.

**Regola stabilita dall'utente il 2026-09-16, da rispettare sempre:**

> Prima di modificare qualunque file del paper, allineati SEMPRE a quello che
> c'è su Overleaf. L'utente edita direttamente lì e ti avvisa quando lo fa, ma
> l'allineamento va fatto comunque, non dato per scontato.

In pratica, prima di ogni modifica al paper:

1. Scarica il contenuto corrente da Overleaf, non fidarti del mirror locale.
   L'endpoint autenticato è
   `/project/<project_id>/doc/<doc_id>/download`, raggiungibile via
   `javascript_tool` dentro la sessione del browser già autenticata.
   `/project/<project_id>/entities` elenca i path; i `doc_id` si leggono dagli
   attributi `data-file-id` nel file tree (le cartelle vanno espanse prima).
2. Applica la modifica facendo splice sul contenuto SCARICATO, mai su quello
   locale: così le modifiche dell'utente non vengono sovrascritte.
3. Riscrivi il file con focus su `.cm-content` + `cmd+a` + evento `paste`
   sintetico, poi rileggi dall'endpoint e verifica che il contenuto scritto sia
   identico a quello inteso.
4. Ricompila e controlla che gli errori siano 0.
5. Aggiorna il mirror locale e committa.

Per verificare l'allineamento senza trasferire i file, confronta gli SHA-256:
in browser `crypto.subtle.digest('SHA-256', new TextEncoder().encode(t))`, in
locale `hashlib.sha256(open(p,'rb').read())` — entrambi sui byte UTF-8, quindi
confrontabili. Attenzione: la `length` di JavaScript conta unità UTF-16 e
differisce dai byte UTF-8 quando ci sono accenti (blocco autori, bibliografia),
quindi confronta gli hash, non le lunghezze.

Convenzioni LaTeX: niente Unicode grezzo nel sorgente, `---` e non l'em-dash,
`$\Delta$` e `$\sigma$` e non i simboli. Il blocco autori non è anonimizzato: se
DSN 2027 è double-blind va anonimizzato prima della submission, decisione
dell'utente. Struttura di riferimento: le sette sezioni della sezione 10 della
guida.

## 5. Test

La suite non-torch gira con:
`python3 -m pytest tests/ -q --ignore=tests/test_privacy_auditor_subscriber.py --ignore=tests/test_run_experiments_integration.py --ignore=tests/test_sprint4.py --ignore=tests/test_sprint5.py`
Baseline attesa: 321 passed con `datasets/` scaricato (297 prima del 2026-09-24);
senza, 288 passed e 33 fra errori e fallimenti, tutti `FileNotFoundError` in `test_acn_dataset.py` e
`test_chargeplace_scotland_adapter.py`, che leggono i file reali (segnalazione 39).
Dipendenze minime: pytest, pyyaml, numpy, scipy, pandas, openpyxl, dp-accounting. Sulla macchina
con torch e dati: `make test`.
