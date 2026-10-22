#!/usr/bin/env python3
"""
Coda v2d (2026-10-19): attacco shadow versione 2 (federato, stessa DP del
bersaglio, soglia globale; src/plugins/attacks/shadow2.py) su tutte le celle FL
delle campagne v2 e v2c, piu' i meccanismi di DP per client mancanti negli
script per persona. Da lanciare SOLO a campagne v2c e Windows concluse.

Decisione della dottoranda, 19 ottobre: tutti gli attacchi e tutti i modelli
sono federati e ogni attacco gira in tutte le configurazioni DP, nessuna DP
inclusa (docs/Segnalazioni_tecniche_2026-09-22.md, voci 92-94).

Ogni run FL ripete la cella della v2 (stesso config, stesse opzioni, stesso
seed, --common-init, --threads 1) con --skip-attacks lira: LiRA e RMIA sono gia'
nelle run v2 e v2c. Yeom resta e fa da controllo: il training del bersaglio
precede gli attacchi e lo shadow ripristina i generatori casuali, quindi sulla
stessa macchina Yeom deve coincidere con la run v2 o v2c della stessa cella e
dello stesso seed. Per questo ogni cella torna sulla macchina che l'ha eseguita.

  - Mac di Susy: celle ACN (75 run) e mia_persona con DP central (5 run).
  - Mac di Domenico: Scozia (60), ACN a quattro feature (40), curva di
    capacita' (15; cap16_8 non si ripete: e' identica a acn_nodp_mu0.01).
  - PC Windows: canary_persona con DP per client local e central, epsilon 64,
    disegno "indotto" con spostamento 0.5 della v2 (20 run).

Non incluse: le run canary per sessione della v2 su Windows (1000 epoche,
circa 55 ore l'una con LiRA): per i canary lo shadow v2 non calcola metriche
dedicate (la metrica dei canary e' la loss grezza).

Ore: stime del solo training della cella (dalle code v2, colonne senza LiRA)
moltiplicate per 1,5 per la federazione shadow, allenata sulla meta' dei
membri. Da correggere con la prima run cronometrata di ogni macchina, come
nella v2. Una run alla volta per macchina (esegui_coda.py --max-run 1).

Uso: python3 scripts/genera_coda_v2d.py --out <cartella>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
from genera_coda_v2 import SEEDS3, SEEDS5, Coda  # noqa: E402

TEST = ("{PY} -m pytest tests/test_shadow2.py tests/test_lira2.py tests/test_record_dp_poisson.py "
        "tests/test_client_dp_accounting.py tests/test_subject_dp.py tests/test_profili_abitudini.py "
        "tests/test_canary_persona.py -q")
FATTORE_SHADOW = 1.5


def riga_fl(coda: Coda, id_: str, prio: int, ore_training: float, gb: int, cfg: str, flags: str,
            seed: int) -> None:
    coda.cmd(id_, prio, round(max(ore_training * FATTORE_SHADOW, 0.05), 2), gb,
             f"{{PY}} scripts/run_experiments.py --config config/{cfg}.yaml {flags} --seed {seed} "
             f"--common-init --threads 1 --skip-attacks lira --sweep-dir {{ESITI}}/{id_}",
             dopo="test_codice")


# celle ACN della v2 (Mac di Susy): nome, config, opzioni, semi, ore di training
CELLE_ACN = [
    ("nodp_mu0.01", "experiment", "--no-dp", SEEDS5, 0.25),
    ("nodp_mu0", "experiment_rq3_mu0", "--no-dp", SEEDS5, 0.25),
    ("eps64_mu0.01", "experiment_rq1_eps64", "--dp-mode local", SEEDS5, 0.25),
    ("eps64_mu0", "experiment_rq3_mu0_eps64", "--dp-mode local", SEEDS5, 0.25),
    ("subject_s5", "experiment_acn_subjectdp_s5", "--no-dp", SEEDS5, 0.15),
    ("subject_s2", "experiment_acn_subjectdp", "--no-dp", SEEDS3, 0.15),
    ("record_nm1", "experiment_rq1_recorddp_nm1", "--no-dp", SEEDS3, 0.75),
    ("record_nm2", "experiment_rq1_recorddp_nm2", "--no-dp", SEEDS3, 0.75),
    ("record_nm0", "experiment_rq1_recorddp_nm0", "--no-dp", SEEDS3, 0.75),
    ("record_nm5", "experiment_rq1_recorddp_nm5", "--no-dp", SEEDS3, 0.75),
    ("central_eps64", "experiment_rq1_eps64", "--dp-mode central", SEEDS5, 0.25),
] + [(f"eps{e}", f"experiment_rq1_eps{e}", "--dp-mode local", SEEDS5, 0.25) for e in (2, 4, 8, 16, 32, 256)]

# Scozia ridotta a tre consigli (Mac di Domenico)
CELLE_SC3 = [
    ("nodp_mu0.01", "experiment_sc3", "--no-dp", 0.3),
    ("nodp_mu0", "experiment_sc3_mu0", "--no-dp", 0.3),
    ("eps64_mu0.01", "experiment_sc3_eps64", "--dp-mode local", 0.3),
    ("eps64_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local", 0.3),
    ("clip_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation clip-only", 0.3),
    ("clip_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation clip-only", 0.3),
    ("noise_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation noise-only", 0.3),
    ("noise_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation noise-only", 0.3),
    ("eps64_mu0.001", "experiment_sc3_mu0.001_eps64", "--dp-mode local", 0.3),
    ("eps64_mu0.1", "experiment_sc3_mu0.1_eps64", "--dp-mode local", 0.3),
    ("record_s1_mu0.01", "experiment_sc3_recorddp", "--no-dp", 2.0),
    ("subject_s5_mu0.01", "experiment_sc3_subjectdp", "--no-dp", 0.3),
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

# curva di capacita' (Mac di Domenico); cap16_8 = acn_nodp_mu0.01, gia' sul Mac di Susy
CELLE_CAP = [("cap32_16", "experiment_cap32_16", 0.33), ("cap64_32", "experiment_cap64_32", 0.5),
             ("cap128_64", "experiment_cap128_64", 0.83)]


def costruisci() -> list[Coda]:
    mac = Coda("coda_v2d_mac.csv", 1)
    mac.cmd("test_codice", 0, 0.2, 2, TEST)
    for seed in SEEDS5:
        for nome, cfg, flags, semi, ore in CELLE_ACN:
            if seed in semi:
                riga_fl(mac, f"v2d_acn_{nome}_s{seed}", 1, ore, 4, cfg, flags, seed)
    for seed in SEEDS5:
        mac.cmd(f"v2d_persona_client_central_eps64_s{seed}", 2, 0.8, 3,
                f"{{PY}} scripts/mia_persona.py --seed {seed} --meccanismo client_central_eps64 "
                f"--out {{ESITI}}/mia_persona", dopo="test_codice")

    dom = Coda("coda_v2d_domenico.csv", 1)
    dom.cmd("test_codice", 0, 0.2, 2, TEST)
    for seed in SEEDS5:
        for nome, cfg, flags, ore in CELLE_SC3:
            riga_fl(dom, f"v2d_sc3_{nome}_s{seed}", 1, ore, 5, cfg, flags, seed)
    for seed in SEEDS5:
        for nome, cfg, flags in CELLE_4F:
            riga_fl(dom, f"v2d_acn4f_{nome}_s{seed}", 1, 0.25, 3, cfg, flags, seed)
    for seed in SEEDS5:
        for nome, cfg, ore in CELLE_CAP:
            riga_fl(dom, f"v2d_{nome}_s{seed}", 2, ore, 5, cfg, "--no-dp", seed)

    win = Coda("coda_v2d_windows.csv", 1)
    win.cmd("test_codice", 0, 0.2, 2, TEST)
    CP = "{PY} scripts/canary_persona.py --out {ESITI}/canary_persona --combina-se-pronto"
    for modo in ("local", "central"):
        for seed in SEEDS5:
            for b in ("A", "B"):
                win.cmd(f"v2d_cp_indotto_c30_d0.5_{modo}_eps64_{b}_s{seed}", 1, 1.8, 3,
                        f"{CP} --config config/experiment_canary_multisite.yaml "
                        f"--etichetta indotto_{modo}_eps64 --dp-mode {modo} --epsilon 64 "
                        f"--copie 30 --spostamento 0.5 --braccio {b} --seed {seed}", dopo="test_codice")
    return [mac, dom, win]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(QUI))
    a = ap.parse_args()
    code = costruisci()
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
