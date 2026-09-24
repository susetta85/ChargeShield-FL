#!/usr/bin/env python3
"""
Fase A della guida scientifica (ChargeShield_FL_spina_dorsale_consolidata.md,
sezione 7, righe 148-154): genera i TRE output previsti.

    risultati/matrice_run_completati.xlsx  -> registro run
    risultati/matrice_confronti.xlsx       -> matrice di confrontabilita'
    risultati/Matrice_sintesi.xlsx         -> lacune effettive + minimo rerun

REGOLA NON NEGOZIABILE della guida (riga 150):
    "Nessun valore mancante va riempito per analogia."
Quindi: commit e versione/hash degli split NON sono registrati nei JSON di
questo progetto e restano VUOTI, con la dicitura "non registrato". Non si
deducono dalla data, dal nome della cartella o da altre run.

Gli stati di validita' ammessi sono i cinque della guida:
    verificata | completata da verificare | invalidata | incompleta | pianificata

Uso:
    python3 scripts/genera_matrici_faseA.py
"""
from __future__ import annotations
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USCITA = os.path.join(RADICE, "risultati")
FONT = "Arial"

# Commit del fix che rende la loss grezza dei canary indipendente dal filtro
# LiRA (Sprint 10zz+119). I JSON canary precedenti hanno canary_raw_mse_auc_roc
# calcolata su un pool filtrato dalla calibrazione shadow: la metrica c'e' ma
# non e' confrontabile con i baseline, che girano senza shadow.
STACCO_FIX_RAW = datetime(2026, 9, 16, 21, 0)

VERDE = PatternFill("solid", fgColor="E2EFDA")
GIALLO = PatternFill("solid", fgColor="FFF2CC")
ROSSO = PatternFill("solid", fgColor="FCE4E4")
GRIGIO = PatternFill("solid", fgColor="EDEDED")
BORDO = Border(*[Side(style="thin", color="BFBFBF")] * 4)


# ─────────────────────────────────────────────────────────────────────────────
# Lettura delle evidenze
# ─────────────────────────────────────────────────────────────────────────────
def mappa_log() -> dict[str, list[str]]:
    """Associa il nome di uno sweep ai file di log che lo nominano."""
    m = defaultdict(list)
    for p in glob.glob(os.path.join(RADICE, "logs", "*.log")):
        nome = os.path.basename(p)
        chiave = re.sub(r"\.log$", "", nome)
        m[chiave].append(os.path.join("logs", nome))
    return m


def trova_log(sweep: str, log_disponibili: dict[str, list[str]]) -> str:
    """Log plausibile per uno sweep. Solo corrispondenze di nome, mai inferenze."""
    s = sweep.lstrip("_")
    esatti = [v for k, vs in log_disponibili.items() if k == s for v in vs]
    if esatti:
        return "; ".join(sorted(esatti))
    parziali = [v for k, vs in log_disponibili.items()
                if s and (s in k or k in s) for v in vs]
    return "; ".join(sorted(set(parziali))) if parziali else ""


def parametri_canary_dai_log() -> dict[str, dict]:
    """k, duplicati e swap letti dai LOG, non dedotti.

    Risolve le run canary anteriori allo Sprint 10zz+113, che non salvano il
    blocco 'canary' nel JSON. La riga di log
        "[CANARY] Iniettati K template x D duplicati ... + N gemelli non-membro"
    contiene entrambe le cardinalita' in chiaro, e "[SWAP ATTIVO]" dice se il
    braccio e' quello scambiato. E' lettura di un'evidenza, non un'analogia:
    il log e' l'output della run stessa.
    """
    pat = re.compile(
        r"Iniettati (\d+) template . (\d+) duplicati.*?\+ (\d+) gemelli non-membro"
    )
    fuori = {}
    for p in glob.glob(os.path.join(RADICE, "logs", "*.log")):
        txt = open(p, errors="replace").read()
        m = pat.search(txt)
        if not m:
            continue
        k, dup, nnm = int(m.group(1)), int(m.group(2)), int(m.group(3))
        fuori[os.path.basename(p)] = {
            "k": k, "dup": dup, "nnm": nnm,
            "swap": "[SWAP ATTIVO]" in txt,
            "bilanciato": k == nnm,
        }
    return fuori


def canary_dal_log(sweep: str, par: dict[str, dict]) -> dict | None:
    """Parametri canary per uno sweep, se un log con quel nome esiste."""
    s = sweep.lstrip("_")
    for nome, v in par.items():
        if re.sub(r"\.log$", "", nome) == s:
            return v
    return None


def superficie(cfg: dict) -> str:
    """Punto di osservazione dell'avversario (sezione 5 della guida)."""
    if cfg.get("no_dp"):
        return "A0 — update non privatizzato (nessun clipping, nessun rumore)"
    return {
        "dp-fedavg": "A1 — update grezzo g (limite superiore, oltre lo Scenario 1)",
        "central": "A2 — update clippato g_bar (Scenario 1)",
        "local": "A3 — update clippato e rumorizzato g_tilde (Scenario 1)",
    }.get(cfg.get("dp_mode"), "non determinabile dal config")


def regime(cfg: dict, sweep: str) -> str:
    can = cfg.get("canary") or {}
    if can.get("enabled") or "canary" in sweep.lower():
        return "canary (indotto)"
    if cfg.get("record_dp"):
        return "naturale — record-DP (diagnostico)"
    return "naturale"


def rq_di(cfg: dict, sweep: str, reg: str) -> str:
    """Mappa sulle RQ della guida, sezione 4."""
    if reg.startswith("canary"):
        return "CTRL — validazione dello strumento, non una RQ"
    if cfg.get("no_dp") is not None or cfg.get("dp_mode"):
        return "RQ1"
    return "non classificabile"


def stato_validita(cfg: dict, sweep: str, reg: str, ts: datetime,
                   mia: dict, dal_log: dict | None = None) -> tuple[str, str]:
    """Stato + motivo. Solo criteri oggettivi e verificabili nel file stesso.

    Il criterio "post-fix" NON usa il timestamp, che sarebbe una congettura:
    usa la presenza del campo canary_raw_n_member_distinct, introdotto dal fix
    stesso (Sprint 10zz+119). Se il campo c'e', la loss grezza e' stata
    raccolta prima del filtro di calibrazione LiRA; se non c'e', no.
    """
    can = cfg.get("canary") or {}
    e_canary = reg.startswith("canary")
    raw_pulita = mia.get("canary_raw_n_member_distinct") is not None

    if e_canary and not can:
        if dal_log:
            k, nnm, sw = dal_log["k"], dal_log["nnm"], dal_log["swap"]
            if sw and not dal_log["bilanciato"]:
                return ("invalidata",
                        f"RISOLTO DAL LOG (2026-09-21): braccio di scambio con gruppi "
                        f"sbilanciati, k_membro={k} contro k_nonmembro={nnm}. Con "
                        f"cardinalita' diverse i due bracci pescano insiemi di membri "
                        f"DISGIUNTI, quindi non e' un controllo negativo (guida sez. 6; "
                        f"docs/CanaryPositiveControl.md 4.1). Il config non era nel JSON, "
                        f"ma la riga '[CANARY] Iniettati ...' del log lo dichiara")
            return ("completata da verificare",
                    f"il blocco 'canary' non e' nel JSON, ma il log risolve il disegno: "
                    f"k_membro={k}, duplicati={dal_log['dup']}, k_nonmembro={nnm}, "
                    f"swap={'si' if sw else 'no'}, "
                    f"{'bilanciato' if dal_log['bilanciato'] else 'SBILANCIATO'}. "
                    f"Il braccio base resta un positive control leggibile; la coppia con "
                    f"lo swap no, se sbilanciata")
        return ("incompleta",
                "il blocco 'canary' non e' registrato nel config del JSON (i run "
                "anteriori allo Sprint 10zz+113 non lo salvavano) e NON esiste un log "
                "con il nome di questo sweep: k, duplicati e swap_assignment non sono "
                "ricavabili da nessuna evidenza. Dedurli dal nome della cartella "
                "sarebbe l'analogia che la guida vieta")

    if can.get("enabled"):
        nt, nnm = can.get("n_templates"), can.get("n_nonmember_templates")
        if can.get("swap_assignment") and nt is not None and nt != nnm:
            return ("invalidata",
                    f"braccio di scambio con gruppi sbilanciati ({nt} vs {nnm}): i due "
                    "bracci pescano insiemi di membri DISGIUNTI, quindi non e' un "
                    "controllo negativo (guida sez. 6; docs/CanaryPositiveControl.md 4.1)")
        if not raw_pulita:
            return ("completata da verificare",
                    "canary anteriore al fix Sprint 10zz+119 (manca il campo "
                    "canary_raw_n_member_distinct): canary_raw_mse_auc_roc e' calcolata "
                    "su un pool filtrato da insufficient_calibration di LiRA, quindi non "
                    "e' appaiata con la baseline a init casuale, che gira senza shadow")
        if re.fullmatch(r"_canary_balanced(_swap)?_s\d+", sweep):
            return ("verificata",
                    "campagna bilanciata post-fix: numeri ricontrollati dai JSON il "
                    "2026-09-21 (30/30 round sopra 0.5, min 0.6125, t(4)=9.90 appaiato "
                    "sui 5 seed, 400 coppie distinte in tutte le celle)")
        return ("completata da verificare",
                "cella di ablation post-fix: il dato e' pulito ma non e' stato "
                "sottoposto alla stessa verifica numerica della campagna principale")

    if not mia:
        return ("incompleta", "nessuna metrica MIA nel JSON")
    return ("completata da verificare",
            "nessun difetto noto, ma commit e hash degli split non sono registrati: "
            "la tracciabilita' completa richiesta dalla Fase A non e' ricostruibile")


def leggi_run() -> list[dict]:
    log_disp = mappa_log()
    par_canary = parametri_canary_dai_log()
    righe = []
    for f in sorted(glob.glob(os.path.join(RADICE, "experiments", "*", "experiment_*.json"))):
        sweep = os.path.basename(os.path.dirname(f))
        base = os.path.basename(f)
        try:
            d = json.load(open(f))
        except Exception as e:
            righe.append({"id": f"{sweep}/{base}", "errore": type(e).__name__})
            continue
        cfg = d.get("config") or {}
        summ = d.get("summary") or {}
        pr = d.get("per_round") or {}
        ultimo = max(pr, key=int) if pr else None
        mia = (pr.get(ultimo) or {}).get("mia", {}) if ultimo else {}
        try:
            ts = datetime.strptime(d.get("timestamp", ""), "%Y%m%d_%H%M%S")
        except Exception:
            ts = None
        reg = regime(cfg, sweep)
        stato, motivo = stato_validita(cfg, sweep, reg, ts, mia,
                                       canary_dal_log(sweep, par_canary))
        m = re.search(r"seed(\d+)", sweep)
        seed_nome = m.group(1) if m else None
        discorde = seed_nome is not None and str(cfg.get("seed")) != seed_nome
        can = cfg.get("canary") or {}
        righe.append({
            "id": f"{sweep}/{base}",
            "sweep": sweep,
            "rq": rq_di(cfg, sweep, reg),
            "commit": "",
            "configurazione": (
                f"dp_mode={cfg.get('dp_mode')}; no_dp={cfg.get('no_dp')}; "
                f"eps={cfg.get('epsilon')}; delta={cfg.get('delta')}; "
                f"round={cfg.get('fl_rounds')}; epoche={cfg.get('epochs')}; "
                f"batch={cfg.get('batch_size')}; feature={len(cfg.get('feature_names') or []) or 'n.d.'}; "
                f"hidden={cfg.get('hidden_dims')}; latent={cfg.get('latent_dim')}"
                + (f"; canary k={can.get('n_templates')}/{can.get('n_nonmember_templates')}, "
                   f"dup={can.get('n_duplicates')}, swap={can.get('swap_assignment')}"
                   if can.get("enabled") else "")
                # Fix 2026-09-22 (segnalazione 5/37): una run record-DP ha
                # no_dp=True (serve a disattivare il client-level) e senza
                # questo campo la stringa "configurazione" la mostrava
                # indistinguibile da una vera baseline no-DP, anche se la
                # colonna "regime" già la marcava "naturale — record-DP
                # (diagnostico)" (funzione regime() sopra). Reso esplicito
                # qui per chi legge solo questa colonna.
                + (f"; record_dp: enabled={ (cfg.get('record_dp') or {}).get('enabled') }, "
                   f"nm={ (cfg.get('record_dp') or {}).get('noise_multiplier') }, "
                   f"eps_record_dp={cfg.get('epsilon_record_dp')}"
                   if cfg.get("record_dp") else "")
            ),
            "hash_split": "",
            "seed": cfg.get("seed"),
            "seed_nome": seed_nome,
            "seed_discorde": discorde,
            "algoritmo": f"FedProx, mu={cfg.get('proximal_mu')}"
                         if cfg.get("proximal_mu") else "non registrato",
            "regime": reg,
            "superficie": superficie(cfg),
            "dp_accounting": (
                f"eps_naive={cfg.get('epsilon_cumulative_naive')}; "
                f"eps_avanzata={cfg.get('epsilon_cumulative_advanced')}; "
                f"eps_migliore={cfg.get('epsilon_cumulative_best_known')}; "
                "nessun accountant RDP, nessun enforcement"
            ),
            "checkpoint": f"{len(pr)} round valutati",
            "metriche": (
                f"LiRA medio={summ.get('mean_lira_auc_roc')}; "
                f"Yeom medio={summ.get('mean_auc_roc')}; "
                f"Shadow medio={summ.get('mean_shadow_auc_roc')}; "
                f"LiRA composto={mia.get('composed_lira_auc_roc')}"
                + (f"; canary raw={mia.get('canary_raw_mse_auc_roc')}"
                   if mia.get("canary_raw_mse_auc_roc") is not None else "")
            ),
            "log": trova_log(sweep, log_disp),
            "artefatti": f"experiments/{sweep}/ (JSON + xlsx + history/)",
            "stato": stato,
            "motivo": motivo,
            "ts": ts,
        })
    return righe


# ─────────────────────────────────────────────────────────────────────────────
# Scrittura
# ─────────────────────────────────────────────────────────────────────────────
def stile(ws, larghezze, altezza=100):
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            cel = ws.cell(r, c)
            cel.font = Font(name=FONT, size=9, bold=(r == 1))
            cel.alignment = Alignment(wrap_text=True, vertical="top")
            cel.border = BORDO
            if r == 1:
                cel.fill = GRIGIO
    for col, w in zip("ABCDEFGHIJKLMNO", larghezze):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = altezza


def colora_stato(ws, col):
    for r in range(2, ws.max_row + 1):
        v = str(ws.cell(r, col).value or "")
        if v.startswith("verificata"):
            ws.cell(r, col).fill = VERDE
        elif v.startswith("invalidata") or v.startswith("incompleta"):
            ws.cell(r, col).fill = ROSSO
        elif v:
            ws.cell(r, col).fill = GIALLO


def scrivi_registro(righe):
    p = os.path.join(USCITA, "matrice_run_completati.xlsx")
    wb = openpyxl.load_workbook(p)
    ws = wb["Matrice_run"]
    # Colonne "commit" e "versione/hash degli split" RIMOSSE il 2026-09-21 su
    # decisione dell'utente: erano vuote su tutte le 235 righe perche' non sono
    # mai state registrate. Da oggi run_experiments.py salva config.git_commit,
    # quindi le run FUTURE saranno tracciabili e la colonna potra' tornare.
    for c, v in enumerate(["ID run", "RQ", "configurazione", "seed", "algoritmo/mu",
                           "regime naturale o canary", "superficie dell'attacco",
                           "DP/accounting", "checkpoint", "metriche", "log",
                           "artefatti", "stato di validita'",
                           "fase (spina dorsale)", "motivo dello stato"], start=1):
        ws.cell(1, c).value = v
    for c in range(16, 18):
        ws.cell(1, c).value = None
    for i, r in enumerate(righe, start=2):
        note_seed = (f"{r['seed']} (cartella dice {r['seed_nome']}: DISCORDE, "
                     "usare il nome cartella)") if r.get("seed_discorde") else r.get("seed")
        for c, v in enumerate([
            r["id"], r["rq"], r["configurazione"],
            note_seed, r["algoritmo"], r["regime"],
            r["superficie"], r["dp_accounting"], r["checkpoint"], r["metriche"],
            r["log"] or "nessun log associabile per nome", r["artefatti"],
            r["stato"], "Fase A — inventario", r["motivo"],
        ], start=1):
            ws.cell(i, c).value = v
    stile(ws, [34, 12, 52, 20, 18, 18, 30, 34, 14, 40, 30, 30, 22, 16, 50], 90)
    colora_stato(ws, 13)

    lg = wb["Legenda_stati"]
    for i, (s, d) in enumerate([
        ("verificata", "Numeri ricontrollati dai JSON da una seconda persona/passata, e nessun difetto noto sulla pipeline che li ha prodotti."),
        ("completata da verificare", "La run e' terminata e le metriche esistono, ma la tracciabilita' richiesta dalla Fase A non e' completa: commit e hash degli split non sono registrati in nessun JSON di questo progetto."),
        ("invalidata", "Esiste un motivo documentato per cui la run non risponde alla domanda per cui e' stata eseguita. Conservata, non cancellata (guida riga 152)."),
        ("incompleta", "La run manca di metriche o si e' interrotta."),
        ("pianificata", "Non ancora eseguita. Nessuna run in questo registro ha questo stato: le lacune stanno in Matrice_sintesi."),
        ("", ""),
        ("NOTA su 'commit' e 'hash split'", "Colonne RIMOSSE il 2026-09-21. La guida (riga 150) le elenca, ma erano vuote su tutte le 235 righe: non sono mai state salvate nei JSON, e dedurle dalla data sarebbe l'analogia che la guida vieta. Due colonne vuote non informano. Da oggi run_experiments.py registra config.git_commit (con marcatore -dirty se l'albero aveva modifiche non committate), quindi le run future saranno tracciabili e la colonna potra' tornare con dati veri."),
    ], start=2):
        lg.cell(i, 1).value = s
        lg.cell(i, 2).value = d
    stile(lg, [30, 110], 60)
    wb.save(p)
    return p, ws.max_row - 1


def scrivi_confronti(righe):
    p = os.path.join(USCITA, "matrice_confronti.xlsx")
    wb = openpyxl.load_workbook(p)
    ws = wb["Matrice_confronti"]
    ws.cell(1, 12).value = "fase (spina dorsale)"

    nat = [r for r in righe if r["regime"] == "naturale" and r["rq"] == "RQ1"]
    celle = defaultdict(list)
    for r in nat:
        chiave = ("no-DP" if (r["configurazione"].find("no_dp=True") >= 0)
                  else re.search(r"dp_mode=([\w-]+).*?eps=([\d.]+)", r["configurazione"]).group(0)
                  if re.search(r"dp_mode=([\w-]+).*?eps=([\d.]+)", r["configurazione"]) else "?")
        celle[chiave].append(r)

    conf = []
    nodp = celle.get("no-DP", [])
    n = 0
    for chiave, rs in sorted(celle.items(), key=str):
        if chiave == "no-DP":
            continue
        n += 1
        seed_dp = {str(r["seed_nome"] or r["seed"]) for r in rs}
        seed_nodp = {str(r["seed_nome"] or r["seed"]) for r in nodp}
        appaiati = sorted(seed_dp & seed_nodp)
        conf.append([
            f"C{n:02d}", "RQ1", "no-DP (regime naturale)", chiave,
            "si, se appaiati per seed" if appaiati else "no: nessun seed in comune",
            "dati, split, architettura, algoritmo, round, epoche, batch: identici",
            ("nessuna differenza non controllata individuata oltre al fattore DP"
             if appaiati else "manca il riferimento appaiato"),
            f"{len(rs)} run nella cella DP, {len(nodp)} nella cella no-DP; "
            f"seed appaiabili: {appaiati or 'nessuno'}",
            "da verificare: alcune celle contengono run di sweep diversi, e "
            "check_significance.py deduplica scegliendo il file piu' recente",
            "confronto disponibile" if appaiati else "confronto non costruibile",
            "Il contrasto B0/B2 della guida (sez. 8) e' questo. Manca invece il "
            "braccio B1 'clipping senza rumore', mai eseguito.",
            "Fase A — confrontabilita' (il confronto si esegue in Fase B)",
        ])

    conf.append([
        "C-RQ2", "RQ2", "partizione IID", "partizione non-IID (per sito)",
        "no", "—",
        "il braccio IID non esiste: i client sono i 3 siti reali raggruppati per "
        "site_id, quindi la partizione e' non-IID per natura, non per costruzione",
        "nessuna configurazione registra split.strategy: default 'random' in tutte "
        "le 235 run; l'alternativa 'entity_aware' non e' mai stata attivata",
        "—", "confronto non costruibile",
        "Serve costruire il riferimento IID rimescolando le sessioni fra i siti a "
        "parita' di numerosita' per client (guida Fase E).",
        "Fase A — lacuna rilevata (si esegue in Fase E)",
    ])
    conf.append([
        "C-RQ3", "RQ3", "FedAvg (mu=0)", "FedProx (mu=0.01)",
        "no", "—",
        "il braccio mu=0 non esiste",
        "proximal_mu = 0.01 su 235 run su 235, verificato con "
        "scripts/build_run_registry.py il 2026-09-21",
        "—", "confronto non costruibile",
        "Da eseguire da zero riusando split, candidati, checkpoint e condizioni DP "
        "della Fase B (guida Fase D).",
        "Fase A — lacuna rilevata (si esegue in Fase D)",
    ])

    for i, riga in enumerate(conf, start=2):
        for c, v in enumerate(riga, start=1):
            ws.cell(i, c).value = v
    stile(ws, [12, 10, 26, 30, 22, 34, 42, 42, 34, 24, 46, 30], 110)
    colora_stato(ws, 10)
    for r in range(2, ws.max_row + 1):
        if "non costruibile" in str(ws.cell(r, 10).value):
            ws.cell(r, 10).fill = ROSSO

    lg = wb["Legenda_stati"]
    for i, (s, d) in enumerate([
        ("confronto disponibile", "Esistono run appaiabili per seed su entrambi i bracci. Non significa che il confronto sia gia' stato fatto ne' che sia valido: significa che i dati ci sono."),
        ("confronto non costruibile", "Manca del tutto uno dei due bracci. Nessuna analisi sui dati esistenti puo' produrlo."),
        ("", ""),
        ("NOTA", "La guida (riga 152) chiede di segnalare le configurazioni duplicate: le celle central eps=0.5, central eps=1.0 e local eps=1.0 contengono piu' run per lo stesso seed, provenienti da sweep diversi."),
    ], start=2):
        lg.cell(i, 1).value = s
        lg.cell(i, 2).value = d
    stile(lg, [30, 110], 60)
    wb.save(p)
    return p, len(conf)


def scrivi_sintesi(righe):
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    ws = wb["Matrice_sintesi"]
    ws.cell(1, 10).value = "fase (spina dorsale)"

    n_tot = len(righe)
    n_nat = sum(1 for r in righe if r["regime"] == "naturale")
    n_can = sum(1 for r in righe if r["regime"].startswith("canary"))
    n_inv = sum(1 for r in righe if r["stato"] == "invalidata")
    n_ver = sum(1 for r in righe if r["stato"] == "verificata")
    n_incompl = sum(1 for r in righe if r["stato"] == "incompleta")

    S = [
        [f"{n_nat} run in regime naturale", "RQ1", "completata da verificare",
         "C01..C11", "si, appaiando per seed",
         "Nessuna lacuna sul fattore DP. Manca il braccio B1 'clipping senza "
         "rumore' (sigma=0), che la guida chiede in Fase B per separare "
         "l'effetto del clipping da quello del rumore.",
         "1 cella B1: clipping attivo, sigma=0, stessi 5 seed della cella di "
         "riferimento. Riusa i config esistenti cambiando un solo parametro.",
         "alta", "Da decidere in Fase B: quale superficie e' primaria. "
         "Raccomandazione motivata: A1/dp-fedavg, perche' un nullo li' limita "
         "anche A2 e A3.",
         "Fase A -> B"],
        [f"{n_tot} run totali", "RQ2", "lacuna strutturale", "C-RQ2", "no",
         "Non esiste alcun riferimento IID. I client sono i 3 siti reali, quindi "
         "l'eterogeneita' non e' un fattore manipolato ma una proprieta' dei dati. "
         "Nessuna run puo' colmarla a posteriori.",
         "Costruire la partizione IID rimescolando le sessioni fra i siti a parita' "
         "di numerosita' per client, poi IID/non-IID x senza-DP/con-DP: 4 celle x 5 "
         "seed = 20 run.",
         "media", "La guida (Fase E) avverte che una partizione IID artificiale e' "
         "un riferimento sperimentale, non un deployment reale: va dichiarato.",
         "Fase A -> E"],
        [f"{n_tot} run totali", "RQ3", "lacuna strutturale", "C-RQ3", "no",
         "proximal_mu = 0.01 su 235 run su 235. Nessuna esecuzione con mu=0, quindi "
         "il confronto FedAvg/FedProx non e' recuperabile dai dati esistenti.",
         "Ripetere la configurazione di Fase B con mu=0, appaiata per split e seed: "
         "1 algoritmo x 2 livelli di protezione x 5 seed = 10 run.",
         "alta", "Attenzione: proximal_mu vive in cfg['ml'], non in "
         "cfg['experiment']. Impostarlo nel posto sbagliato verrebbe ignorato in "
         "silenzio.",
         "Fase A -> D"],
        ["—", "RQ4", "esclusa per scelta", "—", "—",
         "Threat model passivo (Scenario 1, aggregatore honest-but-curious). "
         "ByzantineDetector e' implementato e testato a unita' ma mai esercitato "
         "end-to-end contro un client Byzantine attivo.",
         "Nessun rerun. La guida (sez. 4) chiede di registrare l'esclusione e non "
         "rivendicare una risposta.",
         "nessuna", "Da dichiarare come RQ rinviata in Fase F, con ragione e "
         "conseguenza sul claim: non deve sparire dalla narrativa.",
         "Fase A -> F"],
        [f"{n_can} run canary ({n_ver} verificate, {n_incompl} incompleta)", "CTRL",
         "verificata per la campagna bilanciata post-fix", "—", "non applicabile",
         f"Il controllo positivo e' superato e il negativo bilanciato regge "
         f"(30/30 round sopra 0.5, min 0.6125, t(4)=9.90 appaiato sui 5 seed). "
         f"Ma {n_incompl} run canary hanno il blocco 'canary' NON registrato nel "
         f"config: k, duplicati e swap_assignment non sono leggibili dal file, "
         f"quindi non si puo' stabilire dal JSON se quei bracci di scambio fossero "
         f"bilanciati. Nessuna e' marcata 'invalidata' proprio per questo: "
         f"invalidare richiederebbe un'analogia, che la guida vieta.",
         "Nessun rerun. Serve pero' una verifica documentale: associare a ciascuna "
         "delle run incomplete il YAML di config corrispondente e registrare k e "
         "swap_assignment nel registro. E' lettura di file, non nuovo calcolo.",
         "bassa", "La guida (sez. 6) e' esplicita: 'simmetrico = membership' non "
         "e' un teorema generale, e il canary non certifica il null naturale. Tenere "
         "il risultato separato dal regime naturale (guida Fase C).",
         "Fase A (strumento) — non e' una RQ"],
    ]
    for i, riga in enumerate(S, start=2):
        for c, v in enumerate(riga, start=1):
            ws.cell(i, c).value = v
    stile(ws, [30, 10, 26, 14, 20, 56, 52, 12, 52, 22], 160)
    for r in range(2, ws.max_row + 1):
        v = str(ws.cell(r, 3).value or "")
        ws.cell(r, 3).fill = (VERDE if v.startswith("verificata")
                              else ROSSO if "lacuna" in v else GIALLO)

    rg = wb["Regole_intersezione"]
    for i, (k, d) in enumerate([
        ("Una run puo' servire piu' RQ", "Una run in regime naturale serve RQ1; la stessa run servirebbe RQ3 solo se esistesse la gemella con mu=0. Non esiste, quindi non la si conta per RQ3."),
        ("Regime naturale e canary non si mescolano", "La guida (Fase C) vieta di usare il canary per certificare il null naturale. Nel registro sono due popolazioni separate e non vanno aggregate."),
        ("Deduplica", "check_significance.py deduplica per (dp_mode, epsilon, seed) scegliendo il file piu' recente. E' una scelta implicita: va dichiarata nel paper o sostituita con un criterio esplicito."),
        ("Priorita'", "alta = blocca una RQ dichiarata; media = blocca una RQ ma con costo maggiore; nessuna = non serve rerun."),
        ("Cosa NON e' in queste matrici", "Qualunque decisione sulle fasi B-F. La Fase A si ferma a: cosa esiste, cosa e' confrontabile, cosa manca. Le raccomandazioni sono marcate come tali e rinviate alla fase competente."),
    ], start=2):
        rg.cell(i, 1).value = k
        rg.cell(i, 2).value = d
    stile(rg, [34, 110], 60)
    wb.save(p)
    return p, len(S)



def scrivi_costo_per_sito():
    """Costo per sito: NON e' nei JSON, solo nei log.

    Errore trovato e corretto il 2026-09-21: la decision matrix affermava che
    il dato per client fosse "nella stessa struttura" dei JSON. Non e' vero.
    per_round[*].fl contiene SOLO mean_loss globale; nessuna chiave nomina un
    client o un sito. La loss per client esiste unicamente nella riga di log
    "[<sito>-NN] Round R — loss=..., n=...". Questa tabella la estrae da li'.
    """
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Costo_per_sito1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Costo_per_sito" in wb.sheetnames:
        del wb["Costo_per_sito"]
    ws = wb.create_sheet("Costo_per_sito")
    ws.append(["log", "condizione DP", "sito", "round finale",
               "loss finale", "n sessioni del client"])
    pat = re.compile(r"\[(\w+)-\d+\] Round (\d+) — loss=([\d.eE+-]+), n=(\d+)")
    righe = []
    for f in sorted(glob.glob(os.path.join(RADICE, "logs", "*.log"))):
        txt = open(f, errors="replace").read()
        if "[NO-DP BASELINE]" in txt:
            cond = "no-DP"
        else:
            m = re.search(r"GradientManager — .=([\d.]+)", txt)
            cond = f"DP (eps={m.group(1)})" if m else "non determinabile dal log"
        ultimo = {}
        for m in pat.finditer(txt):
            sito, rnd, loss, n = m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(4))
            if sito not in ultimo or rnd > ultimo[sito][0]:
                ultimo[sito] = (rnd, loss, n)
        for sito, (rnd, loss, n) in sorted(ultimo.items()):
            righe.append([os.path.basename(f), cond, sito, rnd, loss, n])
    for r in righe:
        ws.append(r)
    stile(ws, [42, 26, 12, 14, 18, 22], 18)
    wb.save(p)
    return len(righe)



def scrivi_utility_privacy():
    """Griglia utility x privacy x limite teorico (2026-09-21).

    Tre cose in un foglio solo, perche' vanno lette insieme:
      - il COSTO (loss finale sull'holdout, rapporto rispetto al no-DP)
      - la PRIVACY misurata (Yeom, Shadow, LiRA, LiRA composto, TPR@1%FPR)
      - il LIMITE TEORICO che la garanzia (eps,delta) permetterebbe
    Il limite usa l'epsilon CUMULATIVO su T round (composizione base,
    eps_tot = T*eps), non l'epsilon per round: e' quello che copre il
    transcript che l'avversario osserva davvero.
    """
    import math
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Utility_privacy_limite1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Utility_privacy_limite" in wb.sheetnames:
        del wb["Utility_privacy_limite"]
    ws = wb.create_sheet("Utility_privacy_limite")
    ws.append(["cella", "n run", "loss finale", "x rispetto a no-DP",
               "Yeom", "Shadow", "LiRA", "LiRA composto", "TPR@1%FPR",
               "eps per round", "eps_tot (T=10, base)",
               "Adv max teorica", "AUC max teorica", "lettura"])
    celle = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RADICE, "experiments", "*", "experiment_*.json"))):
        sw = os.path.basename(os.path.dirname(f))
        j = json.load(open(f)); c = j.get("config") or {}
        if (c.get("canary") or {}).get("enabled") or "canary" in sw.lower():
            continue
        if c.get("record_dp"):
            continue
        pr = j.get("per_round") or {}; su = j.get("summary") or {}
        if not pr:
            continue
        ult = max(pr, key=int)
        loss = [pr[r].get("fl", {}).get("mean_loss") for r in sorted(pr, key=int)]
        loss = [x for x in loss if x is not None]
        k = ("no-DP", None) if c.get("no_dp") else (c.get("dp_mode"), c.get("epsilon"))
        celle[k].append((loss[-1] if loss else None, su.get("mean_auc_roc"),
                         su.get("mean_shadow_auc_roc"), su.get("mean_lira_auc_roc"),
                         pr[ult].get("mia", {}).get("composed_lira_auc_roc"),
                         pr[ult].get("mia", {}).get("composed_tpr_at_fpr_0.01")))

    def med(v, i):
        x = [t[i] for t in v if t[i] is not None]
        return sum(x) / len(x) if x else None

    base = med(celle.get(("no-DP", None), []), 0)
    for k in sorted(celle, key=lambda t: (str(t[0]), -(t[1] or 0))):
        v = celle[k]
        lf = med(v, 0)
        eps = k[1]
        if eps:
            et = 10 * eps
            adv = (math.exp(et) - 1) / (math.exp(et) + 1)
            teo = [eps, et, round(adv, 6), round(0.5 + adv / 2, 6)]
            lettura = ("bound VACUO: a questo eps cumulativo la garanzia permette "
                       "un attacco quasi perfetto, quindi 'siamo sotto il limite' "
                       "non e' informativo")
        else:
            teo = [None, None, None, None]
            lettura = "riferimento senza DP: nessuna garanzia da confrontare"
        rap = round(lf / base, 1) if (lf and base) else None
        if rap and rap > 50:
            lettura = ("UTILITY DISTRUTTA (loss oltre 50x il riferimento): un nullo "
                       "di privacy qui non distingue 'DP protegge' da 'il modello "
                       "non impara, quindi non memorizza'. " + lettura)
        ws.append([f"{k[0]}" + (f" eps={eps}" if eps else ""), len(v),
                   round(lf, 6) if lf else None, rap,
                   round(med(v, 1) or 0, 4), round(med(v, 2) or 0, 4),
                   round(med(v, 3) or 0, 4), round(med(v, 4) or 0, 4),
                   round(med(v, 5) or 0, 4), *teo, lettura])
    stile(ws, [22, 7, 13, 16, 9, 9, 9, 13, 11, 13, 18, 15, 16, 70], 80)
    for r in range(2, ws.max_row + 1):
        rap = ws.cell(r, 4).value
        if rap and rap > 50:
            ws.cell(r, 4).fill = ROSSO
        elif rap:
            ws.cell(r, 4).fill = GIALLO
    wb.save(p)
    return ws.max_row - 1


def scrivi_worst_case():
    """Vulnerabilita' per record: osservato contro livello di caso."""
    fjson = os.path.join(USCITA, "worst_case", "livello_di_caso.json")
    if not os.path.exists(fjson):
        return 0
    d = json.load(open(fjson))
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Worst_case_per_record1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Worst_case_per_record" in wb.sheetnames:
        del wb["Worst_case_per_record"]
    ws = wb.create_sheet("Worst_case_per_record")
    ws.append(["gruppo", "n seed", "sessioni multi-seed", "record segnalati",
               "attesi per caso", "sd", "z", "% osservata", "% attesa", "lettura"])
    for et, r in d["gruppi"].items():
        z = r.get("z")
        if z is not None and z > 3:
            let = ("ECCESSO REALE: esiste un sottoinsieme di record sistematicamente "
                   "nel decile alto su seed indipendenti. Invisibile nell'AUC "
                   "aggregata, che in questa cella e' ~0.51.")
        else:
            let = ("indistinguibile dal caso. ATTENZIONE: in questa cella l'utility "
                   "e' distrutta (loss oltre 100x), quindi l'assenza di eccesso NON "
                   "prova che la DP protegga: un modello che non impara non espone.")
        ws.append([et, r["n_seed"], r["sessioni_multi_seed"], r["osservati"],
                   r["attesi_per_caso"], r["sd_nulla"], z,
                   r["percentuale_osservata"], r["percentuale_attesa"], let])
    stile(ws, [22, 8, 20, 16, 16, 8, 9, 13, 12, 80], 76)
    for r in range(2, ws.max_row + 1):
        z = ws.cell(r, 7).value
        ws.cell(r, 7).fill = ROSSO if (z and z > 3) else VERDE
    par = d.get("parametri", {})
    ws.append([])
    ws.append([f"criterio: membro in >= {par.get('min_seed')} seed, percentile medio >= "
               f"{par.get('soglia')}, percentile minimo >= {par.get('pavimento')}; "
               f"livello di caso da {par.get('permutazioni')} permutazioni dei percentili "
               f"DENTRO ogni seed (conserva la distribuzione marginale, distrugge solo "
               f"la corrispondenza fra seed)"])
    wb.save(p)
    return ws.max_row - 1



def scrivi_glossario():
    """Glossario delle metriche che compaiono nelle matrici.

    Esiste perche' due numeri di queste tabelle NON si leggono da soli:
    lo z della vulnerabilita' per record, e il limite teorico della DP.
    Chi apre il foglio senza il contesto rischia di leggerli al contrario.
    """
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Glossario_metriche1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Glossario_metriche" in wb.sheetnames:
        del wb["Glossario_metriche"]
    ws = wb.create_sheet("Glossario_metriche")
    ws.append(["termine", "definizione", "come si legge", "dove compare",
               "documento di riferimento"])
    voci = [
        ("z (vulnerabilita' per record)",
         "Quante deviazioni standard il numero di record segnalati dista dalla media "
         "dei conteggi ottenuti permutando i percentili DENTRO ogni seed. La "
         "permutazione conserva la distribuzione dei punteggi di ogni seed e "
         "distrugge solo la corrispondenza FRA seed, che e' cio' che il criterio "
         "misura. Formula: z = (osservati - media permutata) / dev.std. permutata.",
         "z ~ 0 -> il conteggio e' quello che il caso produce, nessuna vulnerabilita' "
         "rilevabile con questo criterio. z > 3 -> eccesso reale: esiste un "
         "sottoinsieme di record sistematicamente nel decile alto su seed "
         "indipendenti. Lo z dice quanto e' improbabile il caso, NON quanto grave "
         "sia l'esposizione: per quella si guardano l'eccesso assoluto e la %.",
         "foglio Worst_case_per_record", "docs/VulnerabilitaPerRecord.md sez. 3"),
        ("record segnalati",
         "Sessioni reali che sono membro in almeno 2 seed, con percentile medio del "
         "punteggio d'attacco >= 90 e percentile minimo >= 75. Le soglie sono "
         "dichiarate in anticipo e non ottimizzate sui risultati.",
         "Da confrontare SEMPRE con 'attesi per caso'. Il conteggio da solo non e' "
         "un risultato: con migliaia di sessioni e soglia al 90 percentile un certo "
         "numero di coincidenze e' garantito.",
         "foglio Worst_case_per_record", "docs/VulnerabilitaPerRecord.md sez. 2"),
        ("AUC max teorica",
         "Tetto che la garanzia (eps,delta)-DP pone sull'AUC di QUALUNQUE attacco: "
         "0.5 + Adv/2 con Adv = (e^eps - 1 + 2delta)/(e^eps + 1) (Humphries et al.). "
         "Calcolata sull'eps CUMULATIVO T*eps, non su quello per round, perche' deve "
         "coprire il transcript che l'avversario osserva davvero.",
         "A eps_tot = 10 vale 0.99996: il bound e' VACUO, non esclude quasi nulla. "
         "Dire 'siamo sotto il limite teorico' e' vero e privo di contenuto. Va usato "
         "al contrario: la distanza fra permesso e misurato quantifica quanto la "
         "garanzia formale sia lasca rispetto al comportamento reale.",
         "foglio Utility_privacy_limite", "docs/LimiteTeoricoDP.md"),
        ("x rispetto a no-DP",
         "Rapporto fra la loss finale sull'holdout naturale della cella e quella "
         "della cella senza DP. E' la misura di costo.",
         "Oltre 50x significa utility distrutta. In quelle celle un nullo di privacy "
         "NON distingue 'la DP protegge' da 'il modello non impara, quindi non "
         "memorizza e non espone'. Le due spiegazioni non sono separabili.",
         "foglio Utility_privacy_limite", "docs/LimiteTeoricoDP.md sez. 3"),
        ("A0 / A1 / A2 / A3",
         "Punti di osservazione dell'avversario sulla stessa pipeline di "
         "privatizzazione. A0 = nessuna DP. A1 = update grezzo g (dp-fedavg). "
         "A2 = update clippato (central). A3 = update clippato e rumorizzato (local).",
         "Sono ORDINATI per informazione: A2 e A3 si ottengono da A1 con "
         "trasformazioni che distruggono informazione. Chi osserva A1 puo' calcolarsi "
         "A2 e A3; il contrario e' impossibile. Quindi un nullo ad A1 limita anche "
         "gli altri due: non c'era segnale da sopprimere.",
         "registro run, colonna superficie dell'attacco",
         "docs/paper/latex_dsn2027/sections/threat_model.tex"),
        ("stato di validita'",
         "I cinque stati previsti dalla guida: verificata, completata da verificare, "
         "invalidata, incompleta, pianificata.",
         "'completata da verificare' e' lo stato di default e NON significa "
         "sospetta: significa che la tracciabilita' completa richiesta dalla Fase A "
         "non e' ricostruibile, perche' commit e hash degli split non erano "
         "registrati prima del 2026-09-21.",
         "registro run", "ChargeShield_FL_spina_dorsale_consolidata.md riga 150"),
    ]
    for v in voci:
        ws.append(list(v))
    stile(ws, [26, 76, 76, 30, 34], 150)
    wb.save(p)
    return len(voci)


def main():
    os.makedirs(USCITA, exist_ok=True)
    righe = leggi_run()
    p1, n1 = scrivi_registro(righe)
    p2, n2 = scrivi_confronti(righe)
    p3, n3 = scrivi_sintesi(righe)
    n4 = scrivi_costo_per_sito()
    n5 = scrivi_utility_privacy()
    n6 = scrivi_worst_case()
    n7 = scrivi_glossario()
    print(f"registro run   : {n1} righe -> {p1}")
    print(f"confrontabilita: {n2} righe -> {p2}")
    print(f"sintesi        : {n3} righe -> {p3}")
    print(f"costo per sito : {n4} righe (foglio in Matrice_sintesi)")
    print(f"utility/privacy: {n5} righe (foglio in Matrice_sintesi)")
    print(f"worst-case     : {n6} righe (foglio in Matrice_sintesi)")
    print(f"glossario      : {n7} voci (foglio in Matrice_sintesi)")
    stati = defaultdict(int)
    for r in righe:
        stati[r["stato"]] += 1
    print("\nstati di validita':")
    for k, v in sorted(stati.items()):
        print(f"   {k:28s} {v}")


if __name__ == "__main__":
    main()
