#!/usr/bin/env python3
"""
Coda v3 (2026-10-20): attacchi realistici su tutte le celle FL delle campagne v2 e v2c.

Decisione della dottoranda, 20 ottobre: nessun attacco e nessuna ipotesi che il server
non possa realizzare. Il server possiede i propri dati ausiliari (meta' dell'holdout per
gli attacchi di ogni sito) e le sessioni che interroga, mai il resto dei dati dei client.
Gli attacchi della v3 sono quindi:
  - Yeom (soglia sulla loss del modello globale), invariato;
  - shadow federato con dati ausiliari del server (shadow.fonte_membri: esterna), con la
    soglia globale e il classificatore di tipo Shokri (shadow.classificatore: logistica);
  - LiRA e RMIA, online e offline, con ombre solo su dati del server
    (lira.fonte_ombre: server; 32 ombre online e 32 offline).
I risultati v2 e v2c con LiRA e RMIA sull'universo del client, lo shadow sui membri del
client e l'attacco alla persona restano solo come analisi di sensibilita': non si rifanno
e non si estendono (code v2d, v2e blocco A, v2f, v2clf e v2c residua annullate).

Ogni run ripete la cella e il seed della v2 o v2c con il config *_v3.yaml (stessi training,
DP, split e accounting; cambiano solo gli attacchi), sulla macchina che l'ha eseguita:
  - Mac di Susy: celle ACN (75 run); avversario partecipante, che vede solo i modelli globali
    (lira.observation_surface: partecipante, 15 run: senza DP, local e central a epsilon 64,
    priorita' 3); il blocco a 30 round della v2e (utilita', 40 run, priorita' 1, subito dopo la
    prova: dal 20 ottobre e' la verifica dei 30 cicli della RQ2 dell'introduzione, senza LiRA ne' shadow);
    le celle ACN hanno priorita' 2 (3 per DP per soggetto sigma 2 e per record sigma 0 e 5);
  - Mac di Domenico: Scozia ridotta (60), ACN a quattro feature (40), curva di capacita' (15);
  - PC Windows: canary persona con DP per client local e central a epsilon 64 (20 run),
    controllo positivo con la loss del modello (nessun dato ausiliario).
Aggiunte del 2026-10-20 sera (RQ dell'introduzione):
  - Mac di Susy: ablazione solo clipping / solo rumore su ACN a sei feature, con attacchi
    realistici (20 run, priorita' 1); i 30 round passano a --common-init e DP locale;
  - Mac di Domenico: 30 round in Scozia (40 run, 10 seed, priorita' 1);
  - PC Windows: canary per sessione con shadow, LiRA e RMIA realistici, bracci A e B, senza DP e
    con DP locale a epsilon 64, 3 seed (12 run, priorita' 1; prova v3_canaryA_nodp_s42). Le run
    v2 con LiRA non sono mai partite (Segnalazioni 108). Il canary persona passa a priorita' 2.
  - Attacco realistico al soggetto (scripts/mia_soggetto.py): persona ACN livello entita' sul Mac di
    Susy (24 run), livello gruppo sul PC Windows (24), punto di ricarica in Scozia sul Mac di
    Domenico (35).

Ogni run con attacchi scrive anche il dump per campione (--per-sample-dump), con i punteggi composti
di LiRA e RMIA realistici, per l'analisi per record (E-C) e il test appaiato.

Ore: stime, da correggere con la prima run cronometrata di ogni macchina (riga "prova",
da cui dipendono le altre). Ore per run = T (training della cella) + 0,05 T (federazione
shadow su circa il 3,5% delle sessioni) + 0,16 L / w, con L le ore di LiRA della v2 con un
worker e w i worker: le ombre v3 si allenano su circa il 9,8% (online) e il 2,8% (offline)
delle sessioni del sito invece dell'80%.

Uso: python3 scripts/genera_coda_v3.py --out <cartella>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
from genera_coda_v2 import SEEDS3, SEEDS5, Coda  # noqa: E402

TEST = ("{PY} -m pytest tests/test_lira_server.py tests/test_lira2.py tests/test_rmia.py "
        "tests/test_rmia_offline.py tests/test_shadow2.py tests/test_shadow_esterna.py "
        "tests/test_shadow_clf.py tests/test_record_dp_poisson.py tests/test_client_dp_accounting.py "
        "tests/test_subject_dp.py tests/test_profili_abitudini.py tests/test_canary_persona.py tests/test_mia_soggetto.py -q")
QUOTA_LIRA = 0.16
SEEDS_NUOVI = [2025, 2026, 3141, 2718, 1618]


def ore_v3(t: float, l_1w: float, w: int) -> float:
    return round(max(t * 1.05 + QUOTA_LIRA * l_1w / w, 0.05), 2)


def riga(coda: Coda, id_: str, prio: int, ore: float, gb: int, cfg: str, flags: str, seed: int,
         w: int, dopo: str, suffisso: str = "_v3") -> None:
    coda.cmd(id_, prio, ore, gb,
             (f"{{PY}} scripts/run_experiments.py --config config/{cfg}{suffisso}.yaml {flags} --seed {seed} "
              f"--common-init --threads 1 --n-shadow 32 --lira-workers {w} "
              f"--per-sample-dump {{ESITI}}/{id_}/per_sample.json "
              f"--sweep-dir {{ESITI}}/{id_}").replace("  ", " "),
             dopo=dopo, thread=w + 1)


# celle ACN (Mac di Susy): nome, config di base, opzioni, semi, ore di training T, ore LiRA v2 con 1 worker L
CELLE_ACN = [
    ("nodp_mu0.01", "experiment", "--no-dp", SEEDS5, 0.25, 8.75),
    ("nodp_mu0", "experiment_rq3_mu0", "--no-dp", SEEDS5, 0.25, 8.75),
    ("eps64_mu0.01", "experiment_rq1_eps64", "--dp-mode local", SEEDS5, 0.25, 8.75),
    ("eps64_mu0", "experiment_rq3_mu0_eps64", "--dp-mode local", SEEDS5, 0.25, 8.75),
    ("central_eps64", "experiment_rq1_eps64", "--dp-mode central", SEEDS5, 0.25, 8.75),
    ("subject_s5", "experiment_acn_subjectdp_s5", "--no-dp", SEEDS5, 0.15, 4.35),
    ("subject_s2", "experiment_acn_subjectdp", "--no-dp", SEEDS3, 0.15, 4.35),
    ("record_nm1", "experiment_rq1_recorddp_nm1", "--no-dp", SEEDS3, 0.75, 26.25),
    ("record_nm2", "experiment_rq1_recorddp_nm2", "--no-dp", SEEDS3, 0.75, 26.25),
    ("record_nm0", "experiment_rq1_recorddp_nm0", "--no-dp", SEEDS3, 0.75, 26.25),
    ("record_nm5", "experiment_rq1_recorddp_nm5", "--no-dp", SEEDS3, 0.75, 26.25),
] + [(f"eps{e}", f"experiment_rq1_eps{e}", "--dp-mode local", SEEDS5, 0.25, 8.75)
     for e in (2, 4, 8, 16, 32, 256)]

# Scozia ridotta a tre consigli (Mac di Domenico)
CELLE_SC3 = [
    ("nodp_mu0.01", "experiment_sc3", "--no-dp", 0.3, 5.7),
    ("nodp_mu0", "experiment_sc3_mu0", "--no-dp", 0.3, 5.7),
    ("eps64_mu0.01", "experiment_sc3_eps64", "--dp-mode local", 0.3, 5.7),
    ("eps64_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local", 0.3, 5.7),
    ("clip_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation clip-only", 0.3, 5.7),
    ("clip_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation clip-only", 0.3, 5.7),
    ("noise_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation noise-only", 0.3, 5.7),
    ("noise_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation noise-only", 0.3, 5.7),
    ("eps64_mu0.001", "experiment_sc3_mu0.001_eps64", "--dp-mode local", 0.3, 5.7),
    ("eps64_mu0.1", "experiment_sc3_mu0.1_eps64", "--dp-mode local", 0.3, 5.7),
    ("record_s1_mu0.01", "experiment_sc3_recorddp", "--no-dp", 2.0, 70.0),
    ("subject_s5_mu0.01", "experiment_sc3_subjectdp", "--no-dp", 0.3, 5.7),
]

# ACN a quattro feature (Mac di Domenico)
CELLE_4F = [
    ("nodp_mu0.01", "experiment_acn4f", "--no-dp"),
    ("nodp_mu0", "experiment_acn4f_mu0", "--no-dp"),
    ("eps64_mu0.01", "experiment_acn4f_eps64", "--dp-mode local"),
    ("eps64_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local"),
    ("clip_mu0.01", "experiment_acn4f_eps64", "--dp-mode local --dp-ablation clip-only"),
    ("clip_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local --dp-ablation clip-only"),
    ("noise_mu0.01", "experiment_acn4f_eps64", "--dp-mode local --dp-ablation noise-only"),
    ("noise_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local --dp-ablation noise-only"),
]

# curva di capacita' (Mac di Domenico); cap16_8 = acn_nodp_mu0.01, sul Mac di Susy
CELLE_CAP = [("cap32_16", "experiment_cap32_16", 0.33, 11.7), ("cap64_32", "experiment_cap64_32", 0.5, 17.5),
             ("cap128_64", "experiment_cap128_64", 0.83, 29.2)]

# avversario partecipante (solo modelli globali), ACN, Mac di Susy: sottoinsieme di celle
# deciso il 20 ottobre (senza DP, DP per client local e central a epsilon 64); lo shadow si
# salta perche' vede gia' solo il modello globale e gira nella run v3 della stessa cella
CELLE_PARTECIPANTE = [
    ("nodp_mu0.01", "experiment", "--no-dp"),
    ("eps64_mu0.01", "experiment_rq1_eps64", "--dp-mode local"),
    ("central_eps64", "experiment_rq1_eps64", "--dp-mode central"),
]

# blocco a 30 round della v2e (utilita' sola, verifica del budget di RQ3)
BRACCI_30R = [
    ("mu0_nodp", "experiment_rq3_mu0", "--no-dp"),
    ("mu0.01_nodp", "experiment", "--no-dp"),
    ("mu0_eps64", "experiment_rq3_mu0_eps64", "--dp-mode local"),
    ("mu0.01_eps64", "experiment_rq1_eps64", "--dp-mode local"),
]
# 30 round in Scozia (2026-10-20, RQ2 dell'introduzione: durata del training nei due dataset)
BRACCI_30R_SC3 = [
    ("mu0_nodp", "experiment_sc3_mu0", "--no-dp"),
    ("mu0.01_nodp", "experiment_sc3", "--no-dp"),
    ("mu0_eps64", "experiment_sc3_mu0_eps64", "--dp-mode local"),
    ("mu0.01_eps64", "experiment_sc3_eps64", "--dp-mode local"),
]
# ablazione solo clipping / solo rumore su ACN a sei feature (2026-10-20, RQ2), con attacchi realistici
CELLE_ACN_ABL = [
    ("clip_mu0.01", "experiment_rq1_eps64", "--dp-mode local --dp-ablation clip-only"),
    ("clip_mu0", "experiment_rq3_mu0_eps64", "--dp-mode local --dp-ablation clip-only"),
    ("noise_mu0.01", "experiment_rq1_eps64", "--dp-mode local --dp-ablation noise-only"),
    ("noise_mu0", "experiment_rq3_mu0_eps64", "--dp-mode local --dp-ablation noise-only"),
]


# attacco realistico al soggetto (2026-10-20 sera, scripts/mia_soggetto.py): persona su ACN (Mac di
# Susy), punto di ricarica in Scozia (Mac di Domenico); livello entita' e gruppo di 14 sessioni.
# nome, config della cella, opzioni, semi, ore stimate per run (bersaglio + riferimento del server)
SOGGETTO_ACN = [
    ("nodp", "experiment", "--no-dp", SEEDS5, 0.8),
    ("client_eps64", "experiment_rq1_eps64", "--dp-mode local", SEEDS5, 0.8),
    ("subject_s5", "experiment_acn_subjectdp_s5", "--no-dp", SEEDS5, 1.5),
    ("subject_s2", "experiment_acn_subjectdp", "--no-dp", SEEDS3, 1.5),
    ("record_s2", "experiment_rq1_recorddp_nm2", "--no-dp", SEEDS3, 3.0),
    ("record_s5", "experiment_rq1_recorddp_nm5", "--no-dp", SEEDS3, 3.0),
]
SOGGETTO_SC3 = [   # livelli in cui la cella gira
    ("nodp", "experiment_sc3", "--no-dp", ("entita", "gruppo"), 0.4),
    ("client_eps64", "experiment_sc3_eps64", "--dp-mode local", ("entita", "gruppo"), 0.4),
    ("subject_s5", "experiment_sc3_subjectdp", "--no-dp", ("entita", "gruppo"), 0.6),
    ("record_s1", "experiment_sc3_recorddp", "--no-dp", ("gruppo",), 1.5),
]


def riga_soggetto(coda: Coda, id_: str, prio: int, ore: float, cfg: str, flags: str, livello: str,
                  seed: int, dopo: str) -> None:
    chiave = "node_id" if cfg.startswith("experiment_sc3") else "user_id"
    coda.cmd(id_, prio, ore, 4,
             (f"{{PY}} scripts/mia_soggetto.py --config config/{cfg}.yaml {flags} --livello {livello} "
              f"--chiave {chiave} "
              f"--etichetta {id_.split('_', 3)[3].rsplit('_s', 1)[0]} --seed {seed} --threads 1 "
              f"--out {{ESITI}}/mia_soggetto").replace("  ", " "), dopo=dopo)


def riga_30r(coda: Coda, id_: str, cfg: str, flags: str, seed: int, ore: float) -> None:
    """30 round, solo utilita' e Yeom. Dal 2026-10-20 con --common-init e DP locale come le celle a 10
    round: i round 1-10 devono coincidere con la cella a 10 round dello stesso seed e della stessa
    macchina, e i cinque seed nuovi aggiungono repliche anche al round 10."""
    coda.cmd(id_, 1, ore, 3,
             (f"{{PY}} scripts/run_experiments.py --config config/{cfg}.yaml {flags} --rounds 30 "
              f"--seed {seed} --common-init --threads 1 --skip-attacks lira,shadow "
              f"--sweep-dir {{ESITI}}/{id_}").replace("  ", " "), dopo="test_codice")


def costruisci(w_mac: int, w_dom: int, w_win: int = 6) -> list[Coda]:
    mac = Coda("coda_v3_mac.csv", w_mac)
    mac.cmd("test_codice", 0, 0.2, 2, TEST)
    prova = "v3_acn_nodp_mu0.01_s42"
    riga(mac, prova, 0, ore_v3(0.25, 8.75, w_mac), 4, "experiment", "--no-dp", 42, w_mac, "test_codice")
    for seed in SEEDS5:
        for nome, cfg, flags, semi, t, l in CELLE_ACN:
            id_ = f"v3_acn_{nome}_s{seed}"
            if seed in semi and id_ != prova:
                prio = 3 if nome in ("subject_s2", "record_nm0", "record_nm5") else 2
                riga(mac, id_, prio, ore_v3(t, l, w_mac), 4, cfg, flags, seed, w_mac, prova)
    for nome, cfg, flags, semi, ore in SOGGETTO_ACN:   # livello entita' sul Mac di Susy
        for seed in semi:
            riga_soggetto(mac, f"v3_sogg_entita_{nome}_s{seed}", 3, ore, cfg, flags, "entita", seed,
                          "test_codice")
    for seed in SEEDS5:
        for nome, cfg, flags in CELLE_PARTECIPANTE:
            riga(mac, f"v3_partecipante_acn_{nome}_s{seed}", 3, ore_v3(0.25, 8.75 * 1.1, w_mac), 4, cfg,
                 f"{flags} --skip-attacks shadow", seed, w_mac, prova, suffisso="_v3_partecipante")
    for seed in list(SEEDS5) + SEEDS_NUOVI:
        for nome, cfg, flags in BRACCI_30R:
            riga_30r(mac, f"v3_30r_{nome}_s{seed}", cfg, flags, seed, 1.0)
    for seed in SEEDS5:
        for nome, cfg, flags in CELLE_ACN_ABL:
            riga(mac, f"v3_acn_{nome}_s{seed}", 1, ore_v3(0.25, 8.75, w_mac), 4, cfg, flags, seed, w_mac, prova)

    dom = Coda("coda_v3_domenico.csv", w_dom)
    dom.cmd("test_codice", 0, 0.2, 2, TEST)
    prova_d = "v3_sc3_nodp_mu0.01_s42"
    riga(dom, prova_d, 0, ore_v3(0.3, 5.7, w_dom), 5, "experiment_sc3", "--no-dp", 42, w_dom, "test_codice")
    for seed in SEEDS5:
        for nome, cfg, flags, t, l in CELLE_SC3:
            id_ = f"v3_sc3_{nome}_s{seed}"
            if id_ != prova_d:
                riga(dom, id_, 1, ore_v3(t, l, w_dom), 5, cfg, flags, seed, w_dom, prova_d)
    for seed in SEEDS5:
        for nome, cfg, flags in CELLE_4F:
            riga(dom, f"v3_acn4f_{nome}_s{seed}", 1, ore_v3(0.25, 8.75, w_dom), 3, cfg, flags, seed, w_dom,
                 prova_d)
    for seed in SEEDS5:
        for nome, cfg, t, l in CELLE_CAP:
            riga(dom, f"v3_{nome}_s{seed}", 2, ore_v3(t, l, w_dom), 5, cfg, "--no-dp", seed, w_dom, prova_d)
    for nome, cfg, flags, livelli, ore in SOGGETTO_SC3:
        for livello in livelli:
            for seed in SEEDS5:
                riga_soggetto(dom, f"v3_sogg_{livello}_sc3{nome}_s{seed}", 2, ore, cfg, flags, livello, seed,
                              "test_codice")
    for seed in list(SEEDS5) + SEEDS_NUOVI:
        for nome, cfg, flags in BRACCI_30R_SC3:
            riga_30r(dom, f"v3_sc3_30r_{nome}_s{seed}", cfg, flags, seed, 1.0)

    win = Coda("coda_v3_windows.csv", 1)
    win.cmd("test_codice", 0, 0.2, 2, TEST)
    # canary per sessione con attacchi realistici (2026-10-20): controllo positivo di shadow, LiRA e
    # RMIA con i soli dati del server; sostituisce le run v2 con LiRA, mai partite (Segnalazioni 108)
    prova_w = "v3_canaryA_nodp_s42"
    for seed in SEEDS3:
        for braccio, cfg in (("A", "experiment_canary_multisite"), ("B", "experiment_canary_multisite_swap")):
            for nome, flags in (("nodp", "--no-dp"), ("A3eps64", "--dp-mode local --epsilon 64")):
                id_ = f"v3_canary{braccio}_{nome}_s{seed}"
                riga(win, id_, 0 if id_ == prova_w else 1, ore_v3(1.8, 55, w_win), 4, cfg, flags, seed, w_win,
                     "test_codice" if id_ == prova_w else prova_w)
    for nome, cfg, flags, semi, ore in SOGGETTO_ACN:   # livello gruppo sul PC Windows (carico delle macchine)
        for seed in semi:
            riga_soggetto(win, f"v3_sogg_gruppo_{nome}_s{seed}", 3, ore, cfg, flags, "gruppo", seed,
                          "test_codice")
    cp = "{PY} scripts/canary_persona.py --out {ESITI}/canary_persona --combina-se-pronto"
    for modo in ("local", "central"):
        for seed in SEEDS5:
            for b in ("A", "B"):
                win.cmd(f"v3_cp_indotto_c30_d0.5_{modo}_eps64_{b}_s{seed}", 2, 1.8, 3,
                        f"{cp} --config config/experiment_canary_multisite.yaml "
                        f"--etichetta indotto_{modo}_eps64 --dp-mode {modo} --epsilon 64 "
                        f"--copie 30 --spostamento 0.5 --braccio {b} --seed {seed}", dopo="test_codice")
    return [mac, dom, win]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers-mac", type=int, default=10)
    ap.add_argument("--workers-domenico", type=int, default=14)
    ap.add_argument("--workers-windows", type=int, default=6)
    ap.add_argument("--out", default=str(QUI))
    a = ap.parse_args()
    code = costruisci(a.workers_mac, a.workers_domenico, a.workers_windows)
    ids = [r["id"] for c in code for r in c.righe if r["id"] != "test_codice"]
    assert len(ids) == len(set(ids)), "id duplicati"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in code:
        c.scrivi(out)
        ore = sum(float(r["ore"]) for r in c.righe)
        print(f"{c.nome}: {len(c.righe) - 1} run, circa {ore:.0f} ore una alla volta")
    print(f"totale run: {len(ids)}")


if __name__ == "__main__":
    main()
