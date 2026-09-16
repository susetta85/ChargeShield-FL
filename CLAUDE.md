# Regole di lavoro — ChargeShield-FL

Nota per Claude: leggi questo file prima di iniziare a lavorare sul progetto.

## Overleaf è la fonte di verità per il paper

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

### Convenzioni LaTeX del progetto

- Niente Unicode grezzo nel sorgente: `---` e non l'em-dash, `$\Delta$` e
  `$\sigma$` e non `Δ`/`σ`. Con pdfLaTeX + IEEEtran l'Unicode grezzo rompe o
  rende male.
- Il blocco autori NON è anonimizzato. Se DSN 2027 è double-blind va
  anonimizzato prima della submission: è una decisione dell'utente, non
  toccarlo di iniziativa.

## Altre convenzioni del progetto

- **Sprint-log nel README**: le voci nuove si inseriscono immediatamente SOPRA
  la voce più recente, non in fondo alla tabella.
- **Verificare, non assumere**: i numeri si leggono dai JSON in `experiments/`,
  non dai log incollati in chat né dai commenti nel codice, che in questo
  progetto sono a volte più ottimisti del codice.
- **Suite di test**: nel sandbox torch non è installato. La suite non-torch
  gira con:
  `python3 -m pytest tests/ -q --ignore=tests/test_privacy_auditor_subscriber.py --ignore=tests/test_run_experiments_integration.py --ignore=tests/test_sprint4.py --ignore=tests/test_sprint5.py`
  Baseline attesa: 297 passed.
