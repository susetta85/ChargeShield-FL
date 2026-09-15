# Progetto LaTeX DSN 2027 — Related Work + Framework/Architecture

Creato 2026-09-15, su richiesta esplicita: aggiornare il paper con Related
Work, Framework (con pseudocodice e figure) e bibliografia, in LaTeX,
pronti per Overleaf.

## Perché questi file e non un lavoro diretto dentro Overleaf

Non posso accedere al tuo account Overleaf: ho verificato aprendo
`overleaf.com/project` nel browser di questa sessione e reindirizza al
login — non sei autenticata qui, e per policy di sicurezza non posso
autenticarmi o creare un account per tuo conto (nemmeno se me lo chiedi
esplicitamente). Quello che posso fare, e ho fatto, è produrre sorgente
LaTeX completo e pronto: tu apri un progetto Overleaf (o nuovo o
esistente), carichi questi file, e compili — senza dover scrivere nulla
da zero.

## Cosa c'è in questa cartella

- `main.tex` — documento skeleton (IEEEtran, formato conference) che
  include tutto il resto. Contiene SOLO Related Work e Architecture per
  ora (Introduction/Threat Model/Results/Discussion/etc. restano nei
  due `.docx` esistenti in `docs/paper/` finché non mi chiedi di
  portare anche quelle sezioni qui).
- `sections/related_work.tex` — §2, 5 sottosezioni (MIA, DP federato,
  auditing empirico della DP, robustezza Byzantine, rapporto col vostro
  QRS 2026).
- `sections/framework.tex` — §3, 6 sottosezioni (ML Plane, Privacy
  Auditor, ByzantineDetector, attack suite offline, 3 collocazioni DP,
  adattamenti LiRA), con 4 pseudocodici inline (`algorithm`/`algorithmic`)
  e i riferimenti alle due figure.
- `figures/ml_plane_architecture.tex` — figura TikZ, i tre consumatori
  dell'ML Plane.
- `figures/dp_placements.tex` — figura TikZ, le tre collocazioni DP come
  punti di privilegio lungo la pipeline.
- `references.bib` — 26 voci: le 19 numerate già esistenti nello
  skeleton `.docx` (convertite verbatim in BibTeX, nessuna modifica di
  sostanza) più 4 nuove, verificate via web search il 2026-09-15:
  Sablayrolles et al. 2019 (ICML, lo scorer già implementato in
  `run_lira()`), Zhu et al. 2025 (FedMIA — vedi nota di disambiguazione
  sotto), Salem et al. 2019 (ML-Leaks), Bonawitz et al. 2017 (Secure
  Aggregation).

## Verifica di compilazione fatta qui (non su Overleaf)

`IEEEtran.cls` e i pacchetti `algorithm`/`algorithmic` NON sono
installabili in questo sandbox (nessun accesso root, `tlmgr` non
inizializzabile). Non ho potuto quindi compilare `main.tex` cosi' com'è.
Ho invece verificato la sostanza — sintassi LaTeX, bilanciamento di
parentesi graffe, ambienti, TikZ, bibliografia — con una compilazione di
prova usando `article` al posto di `IEEEtran` e un finto ambiente
`algorithm`/`algorithmic` (solo per il test, non nei file consegnati):
0 errori, 26 voci bibliografiche parsate correttamente da `bibtex`. Un
vero bug l'ho trovato e corretto in questo modo: un `\\` dentro
`\emph{...}` in un nodo TikZ con `align=center` manda in errore TikZ
("Undefined control sequence") — corretto in
`ml_plane_architecture.tex`. Overleaf, con TeX Live completo, compilerà
`main.tex` così com'è; se emerge un errore che questo test non poteva
vedere (specifico di `IEEEtran`/`algorithm`), fammelo sapere e lo
correggo.

## Nota di disambiguazione FedMIA (importante per la revisione double-blind)

Durante questo lavoro hai segnalato 5 paper esterni; uno di questi,
"FedMIA: An Effective Membership Inference Attack Exploiting 'All for
One' Principle in Federated Learning" (Zhu et al., CVPR 2025,
arXiv:2402.06289), è un lavoro reale con lo stesso nome di due artefatti
storici nel nostro codice (la classe `fedmia.py`, mai attiva, e
`run_fedmia_gradient()`). Non implementano il metodo di Zhu et al. — la
coincidenza di nome è casuale e precede la nostra conoscenza della loro
pubblicazione. L'ho citato e disambiguato esplicitamente in
`related_work.tex` §2.1 per evitare che un revisore legga i nostri
risultati come un confronto (mai fatto) con quel metodo.

## Label usate come forward-reference verso sezioni non ancora in questo progetto

`related_work.tex` e `framework.tex` referenziano via `\label`/`\ref`
alcune sezioni che oggi esistono solo nei `.docx`
(`sec:introduction`, `sec:threat-model`, `sec:results`,
`sec:results-surface-b`, `sec:results-utility`, `sec:validation`,
`sec:limitations`, `sec:discussion`). Finché quelle sezioni non sono
anche in questo progetto LaTeX, Overleaf compilerà con warning "Citation
... undefined" per questi `\ref` — non è un errore, sparisce da solo
quando/se porti anche quelle sezioni qui con gli stessi nomi di label.

## Consistenza verificata rispetto al codice/documentazione reale (2026-09-15)

Ogni claim tecnico in `framework.tex` è stato confrontato con lo stato
reale del codice/README prima di essere scritto, non preso dalla sola
bozza di riscrittura IT che mi hai dato:

- `run_yeom()`/`run_shadow()` (rinominati Sprint 10zz+107) — nomi
  aggiornati usati ovunque, non i vecchi `run_fedmia`/`run_fedmia_shadow`.
- Il bug di tagging canary lato non-membro (Sprint 10zz+108, corretto
  oggi) è esplicitamente citato come caveat sulla validazione
  dell'Auditor via canary, con la stessa cautela "pre-fix, indicativo
  non definitivo" già in `docs/TestRoadmap_DSN2027.md`.
- L'overhead dell'Auditor è descritto come timer singolo sempre attivo
  (non un design A/B) — coerente con l'implementazione reale in
  `privacy_auditor_subscriber.py`, non con la formulazione `[DA
  ESEGUIRE: overhead con e senza Auditor]` della bozza IT, che è
  superata dall'implementazione.
- I numeri della simulazione cosine-similarity (media ≈ -0.0024, 100%
  sotto soglia 0.3, controllo senza rumore ≈ 0.9999) sono da una mia
  simulazione NumPy indipendente di verifica, leggermente diversi dalla
  stima originale in README (Sprint 10zz+93, "~0.01") per via della
  casualità della simulazione — entrambe confermano la stessa
  conclusione qualitativa.
- `n_shadow=16` verificato in `config/experiment_chargeplace_scotland.yaml`
  (non assunto dalla bozza).

## Prossimi passi possibili (non ancora fatti)

1. Portare qui anche Introduction/Threat Model/Case Study/Results/
   Validation/Discussion/Limitations (oggi solo nei `.docx`), cosicché
   `main.tex` sia il paper completo in un unico progetto Overleaf.
2. Scrivere l'Abstract (oggi placeholder in `main.tex`) — esiste già
   una bozza rivista nella tua riscrittura IT, da tradurre e adattare.
3. Se vuoi, posso anche generare una versione `.docx` di queste due
   sezioni da affiancare ai due file esistenti in `docs/paper/`, per chi
   preferisce continuare a lavorare in Word invece che in Overleaf.
