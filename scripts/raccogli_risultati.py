#!/usr/bin/env python3
"""
Raccoglie i risultati della coda in risultati/ (2026-10-13; colonne della
campagna v3 il 2026-10-20: fonte e classe di LiRA/RMIA e dello shadow, LiRA e RMIA
offline, classificatore dello shadow; letta la chiave rmia_off_auc_roc, che la
versione precedente cercava con il nome rmia_offline_auc_roc).

  - riepilogo_run.csv: una riga per run di run_experiments (id, seed, cella),
    con la loss sull'holdout del modello rilasciato all'ultimo round (il
    costo), Yeom, LiRA e RMIA (medie sui round e composti), canary;
  - loss_per_round.csv: loss sull'holdout del modello rilasciato a ogni round
    (per i confronti "a parita' di round" e "a parita' di qualita'");
  - mia_persona.csv: le AUC della MIA a livello di persona.
Il costo e le frazioni recuperate si calcolano dopo, su questi file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

QUI = Path(__file__).resolve().parent
ESITI = QUI / "esiti"
OUT = QUI / "risultati"


def ultimo_round(pr: dict) -> dict:
    return pr[max(pr, key=int)] if pr else {}


def shadow(m: dict, chiave: str):
    """Campo dello shadow solo dalla versione 2 (federato, 2026-10-19): i campi
    shadow_* delle run precedenti vengono dalla versione 1 e non si usano."""
    return m.get(chiave) if m.get("shadow_version") == 2 else None


def media_round(pr: dict, chiave: str):
    v = [((x or {}).get("mia") or {}).get(chiave) for x in pr.values()]
    v = [float(a) for a in v if a is not None]
    return round(sum(v) / len(v), 6) if v else None


def fonte_lira(m: dict):
    """Fonte delle ombre di LiRA e RMIA (2026-10-20): "server" = solo dati che il
    server possiede (realistico); "universo" = ombre su tutte le sessioni del client
    (campagne v2 e v2c: solo analisi di sensibilita')."""
    if m.get("lira_fonte_ombre"):
        return m["lira_fonte_ombre"]
    if m.get("lira_auc_roc") is None:
        return None
    # LiRA v1 (prima del 16 ottobre): non valida (Segnalazioni R, voci 65-73)
    return "universo" if m.get("lira_version") == 2 else "v1"


def fonte_shadow(m: dict):
    """Fonte dei membri dello shadow v2: "esterna" = dati ausiliari del server
    (realistico); "bersaglio" o "bersaglio_ridotto" = membri del client (sensibilita')."""
    if m.get("shadow_version") != 2:
        return None
    return ((m.get("shadow_fl") or {}).get("fonte_membri")) or "bersaglio"


def classe(fonte):
    if fonte is None:
        return None
    if fonte == "v1":
        return "non_valido"
    return "realistico" if fonte in ("server", "esterna") else "sensibilita"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    righe, per_round = [], []
    for d in sorted(p for p in ESITI.iterdir() if p.is_dir() and p.name != "mia_persona"):
        js = sorted(d.glob("experiment_*.json"))
        if not js:
            continue
        j = json.loads(js[-1].read_text())
        c, s, pr = j.get("config") or {}, j.get("summary") or {}, j.get("per_round") or {}
        mu = ultimo_round(pr).get("mia") or {}
        hold = mu.get("non_member_score_mean")
        righe.append({
            "id": d.name, "seed": c.get("seed"), "epsilon": c.get("epsilon"),
            "no_dp": c.get("no_dp"), "dp_mode": c.get("dp_mode"), "dp_ablation": c.get("dp_ablation"),
            "proximal_mu": (c.get("proximal_mu") if "proximal_mu" in c else None),
            "rounds": len(pr),
            "loss_holdout_rilasciato": (-float(hold) if hold is not None else None),
            "yeom_auc_ultimo": mu.get("auc_roc"),
            "lira_auc_medio": s.get("mean_lira_auc_roc"),
            "rmia_auc_medio": s.get("mean_rmia_auc_roc"),
            "rmia_auc_ultimo": mu.get("rmia_auc_roc"),
            "rmia_offline_auc_ultimo": mu.get("rmia_off_auc_roc", mu.get("rmia_offline_auc_roc")),
            "lira_fonte_ombre": fonte_lira(mu),
            "lira_superficie": mu.get("lira_observation_surface"),
            "lira_rmia_classe": classe(fonte_lira(mu)),
            "lira_out_auc_medio": media_round(pr, "lira_out_auc_roc"),
            "lira_out_auc_ultimo": mu.get("lira_out_auc_roc"),
            "rmia_out_auc_ultimo": mu.get("rmia_out_auc_roc"),
            "lira_offline_auc_medio": media_round(pr, "lira_off_auc_roc"),
            "lira_offline_auc_ultimo": mu.get("lira_off_auc_roc"),
            "composed_lira_out_auc": mu.get("composed_lira_out_auc_roc"),
            "composed_rmia_out_auc": mu.get("composed_rmia_out_loglr_auc_roc"),
            "composed_lira_offline_auc": mu.get("composed_lira_off_auc_roc"),
            "composed_rmia_offline_auc": mu.get("composed_rmia_off_loglr_auc_roc"),
            "rmia_tpr_1pct_ultimo": mu.get("rmia_tpr_at_fpr_0.01"),
            "composed_lira_auc": s.get("composed_lira_auc_roc") or mu.get("composed_lira_auc_roc"),
            "composed_rmia_auc": s.get("composed_rmia_auc_roc") or mu.get("composed_rmia_auc_roc") or mu.get("composed_rmia_loglr_auc_roc"),
            "canary_raw_auc_ultimo": mu.get("canary_raw_mse_auc_roc"),
            "canary_rmia_auc_ultimo": mu.get("canary_rmia_auc_roc"),
            "canary_lira_auc_ultimo": mu.get("canary_auc_roc"),
            "canary_lira_out_auc_ultimo": mu.get("canary_lira_out_auc_roc"),
            "canary_lira_offline_auc_ultimo": mu.get("canary_lira_off_auc_roc"),
            "canary_rmia_out_auc_ultimo": mu.get("canary_rmia_out_auc_roc"),
            "canary_rmia_offline_auc_ultimo": mu.get("canary_rmia_off_auc_roc"),
            "shadow_canary_auc_ultimo": shadow(mu, "shadow_canary_auc_roc"),
            "shadow_canary_tpr_soglia_fpr_ultimo": shadow(mu, "shadow_canary_tpr_cal_fpr"),
            "shadow_canary_fpr_soglia_fpr_ultimo": shadow(mu, "shadow_canary_fpr_cal_fpr"),
            "shadow_clf_canary_auc_ultimo": shadow(mu, "shadow_clf_canary_auc_roc"),
            "shadow_clf_canary_tpr_ultimo": shadow(mu, "shadow_clf_canary_tpr"),
            "shadow_clf_canary_fpr_ultimo": shadow(mu, "shadow_clf_canary_fpr"),
            "shadow_auc_ultimo": shadow(mu, "shadow_auc_roc"),
            "shadow_tpr_soglia_fpr_ultimo": shadow(mu, "shadow_tpr_cal_fpr"),
            "shadow_fpr_soglia_fpr_ultimo": shadow(mu, "shadow_fpr_cal_fpr"),
            "shadow_vantaggio_soglia_fpr_ultimo": shadow(mu, "shadow_advantage_cal_fpr"),
            "shadow_vantaggio_soglia_vantaggio_ultimo": shadow(mu, "shadow_advantage_cal_adv"),
            "shadow_vantaggio_ottimo_ultimo": shadow(mu, "shadow_oracle_advantage"),
            "shadow_fonte_membri": fonte_shadow(mu),
            "shadow_classe": classe(fonte_shadow(mu)),
            "shadow_clf_auc_ultimo": shadow(mu, "shadow_clf_auc_roc"),
            "shadow_clf_tpr_ultimo": shadow(mu, "shadow_clf_tpr"),
            "shadow_clf_fpr_ultimo": shadow(mu, "shadow_clf_fpr"),
            "shadow_clf_vantaggio_ultimo": shadow(mu, "shadow_clf_advantage"),
            "shadow_clf_vantaggio_ottimo_ultimo": shadow(mu, "shadow_clf_oracle_advantage"),
            "epsilon_record_dp": c.get("epsilon_record_dp"),
            "epsilon_subject_dp": c.get("epsilon_subject_dp"),
            "json": js[-1].name,
        })
        for r, v in sorted(pr.items(), key=lambda kv: int(kv[0])):
            m = (v or {}).get("mia") or {}
            h = m.get("non_member_score_mean")
            per_round.append({"id": d.name, "round": int(r),
                              "loss_holdout_rilasciato": (-float(h) if h is not None else None),
                              "loss_locale": ((v or {}).get("fl") or {}).get("mean_loss"),
                              "rmia_auc": m.get("rmia_auc_roc"), "lira_auc": m.get("lira_auc_roc"),
                              "lira_out_auc": m.get("lira_out_auc_roc"),
                              "rmia_out_auc": m.get("rmia_out_auc_roc"),
                              "lira_offline_auc": m.get("lira_off_auc_roc"),
                              "rmia_offline_auc": m.get("rmia_off_auc_roc", m.get("rmia_offline_auc_roc")),
                              "lira_fonte_ombre": fonte_lira(m),
                              "shadow_clf_auc": shadow(m, "shadow_clf_auc_roc"),
                              "yeom_auc": m.get("auc_roc"),
                              "shadow_auc": shadow(m, "shadow_auc_roc"),
                              "shadow_tpr_soglia_fpr": shadow(m, "shadow_tpr_cal_fpr"),
                              "shadow_fpr_soglia_fpr": shadow(m, "shadow_fpr_cal_fpr"),
                              "shadow_vantaggio_soglia_vantaggio": shadow(m, "shadow_advantage_cal_adv"),
                              "shadow_vantaggio_ottimo": shadow(m, "shadow_oracle_advantage")})
    for nome, dati in (("riepilogo_run.csv", righe), ("loss_per_round.csv", per_round)):
        if dati:
            with open(OUT / nome, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(dati[0]))
                w.writeheader()
                w.writerows(dati)
    persona = []
    for f in sorted((ESITI / "mia_persona").glob("mia_persona_*.json")) if (ESITI / "mia_persona").exists() else []:
        j = json.loads(f.read_text())
        base = {k: j.get(k) for k in ("seed", "meccanismo", "costo_loss_non_membri", "membri", "non_membri")}
        for k, v in (j.get("auc") or {}).items():
            persona.append({**base, "attacco": k, "auc": v["auc"], "ci95_basso": v["ci95"][0],
                            "ci95_alto": v["ci95"][1]})
    if persona:
        with open(OUT / "mia_persona.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(persona[0]))
            w.writeheader()
            w.writerows(persona)
    # attacco realistico al soggetto (scripts/mia_soggetto.py, 2026-10-20)
    sogg = []
    cart_s = ESITI / "mia_soggetto"
    for f in sorted(cart_s.glob("mia_soggetto_*.json")) if cart_s.exists() else []:
        j = json.loads(f.read_text())
        base = {k: j.get(k) for k in ("livello", "chiave", "etichetta", "seed", "config", "costo_loss_non_membri",
                                      "unita_membri", "unita_non_membri", "unita_server", "metrica_primaria")}
        for k, v in (j.get("auc") or {}).items():
            sogg.append({**base, "attacco": k, "primaria": k == j.get("metrica_primaria"),
                         "limite_superiore": k.startswith("tutte_"), "auc": v["auc"],
                         "ci95_basso": v["ci95"][0], "ci95_alto": v["ci95"][1], "p": v.get("p"),
                         "p_holm": v.get("p_holm"), "json": f.name})
    if sogg:
        with open(OUT / "mia_soggetto.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sogg[0]))
            w.writeheader()
            w.writerows(sogg)
    # canary persona: i file appaiati (_AB_) dei due bracci
    cp = []
    cartella = ESITI / "canary_persona"
    for f in sorted(cartella.glob("canary_persona_*_AB_*.json")) if cartella.exists() else []:
        j = json.loads(f.read_text())
        for r, v in sorted(j["per_round"].items(), key=lambda kv: int(kv[0])):
            riga = {"file": f.name, "seed": j["seed"], "regime": j["config"], "copie": j["copie"],
                    "spostamento": j.get("spostamento"), "round": int(r),
                    "dp_client_modo": j.get("dp_client_modo"), "dp_client_epsilon": j.get("dp_client_epsilon")}
            for sup in ("globale", "client"):
                for parte in ("training", "fresche"):
                    k = f"{sup}_{parte}"
                    riga[f"{k}_somma_AB"] = v[f"{k}_somma_AB"]
                    riga[f"{k}_appaiata_auc"] = v[f"{k}_appaiata"]["auc"]
                    riga[f"{k}_appaiata_ci_basso"] = v[f"{k}_appaiata"]["ci95"][0]
                    riga[f"{k}_appaiata_ci_alto"] = v[f"{k}_appaiata"]["ci95"][1]
                    a = v[f"{k}_appaiata"]
                    riga[f"{k}_appaiata_p"] = a.get("p", a.get("p_permutazione"))
                    riga[f"{k}_appaiata_p_holm"] = a.get("p_holm")
            cp.append(riga)
    if cp:
        with open(OUT / "canary_persona.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cp[0]))
            w.writeheader()
            w.writerows(cp)
    print(f"{len(righe)} run, {len(per_round)} righe per round, {len(persona)} righe MIA persona, "
          f"{len(cp)} righe canary persona, {len(sogg)} righe MIA al soggetto -> {OUT}")


if __name__ == "__main__":
    main()
