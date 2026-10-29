"""
Tabelle e verifica delle previsioni della campagna v3 (2026-10-23).

Legge i CSV prodotti da `raccogli_risultati.py` su ciascuna macchina e scrive, nella
cartella di uscita:
  - tabella_celle_v3.csv / .md: una riga per macchina e cella, media e deviazione
    standard sui seed di utilita' (loss sull'holdout del modello rilasciato) e attacchi
    realistici (Yeom, LiRA e RMIA composti nelle tre forme, shadow e classificatore);
  - soggetto_v3.csv / .md: attacco realistico al soggetto, per dataset, livello e cella;
  - previsioni_v3.md: esito delle previsioni V18-V23 e V25-V28 scritte prima del lancio
    (ESPERIMENTI.md, "Campagna v3"); V24 (canary per sessione, PC Windows) resta aperta.

Uso:
    python scripts/tabelle_v3.py --macchina "Mac di Susy" ~/Downloads/ChargeShield_offline/risultati \
        --macchina "Mac di Domenico" ~/Downloads/risultati --out risultati_v3

Ogni confronto fra run (V18, V19, V20, V23, V25, V26) e' fatto solo fra run della stessa
macchina e dello stesso seed: fra macchine diverse il training non e' identico bit per bit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATTACCHI = {
    "yeom_auc_ultimo": "Yeom",
    "composed_lira_auc": "LiRA on",
    "composed_lira_out_auc": "LiRA off (OUT)",
    "composed_lira_offline_auc": "LiRA off (aus.)",
    "composed_rmia_auc": "RMIA on",
    "composed_rmia_out_auc": "RMIA off (OUT)",
    "composed_rmia_offline_auc": "RMIA off (aus.)",
    "shadow_auc_ultimo": "Shadow",
    "shadow_clf_auc_ultimo": "Shadow clf",
}
REALISTICI = [c for c in ATTACCHI if c.startswith("composed_")]


def carica(macchine: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run, sog, lpr = [], [], []
    for nome, cart in macchine:
        cart = Path(cart).expanduser()
        r = pd.read_csv(cart / "riepilogo_run.csv")
        r["macchina"] = nome
        run.append(r)
        if (cart / "mia_soggetto.csv").exists():
            s = pd.read_csv(cart / "mia_soggetto.csv")
            s["macchina"] = nome
            sog.append(s)
        lp = pd.read_csv(cart / "loss_per_round.csv")
        lp["macchina"] = nome
        lpr.append(lp)
    run_df = pd.concat(run, ignore_index=True)
    run_df["cella"] = run_df["id"].str.replace(r"_s\d+$", "", regex=True)
    run_df["seed"] = run_df["id"].str.extract(r"_s(\d+)$")[0].astype(int)
    lp_df = pd.concat(lpr, ignore_index=True)
    lp_df["cella"] = lp_df["id"].str.replace(r"_s\d+$", "", regex=True)
    lp_df["seed"] = lp_df["id"].str.extract(r"_s(\d+)$")[0].astype(int)
    return run_df, (pd.concat(sog, ignore_index=True) if sog else pd.DataFrame()), lp_df


def tabella_celle(v3: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for (mac, cella), g in v3.groupby(["macchina", "cella"], sort=True):
        r = {"macchina": mac, "cella": cella, "n_seed": len(g),
             "loss_media": g["loss_holdout_rilasciato"].mean(), "loss_sd": g["loss_holdout_rilasciato"].std()}
        for c in ATTACCHI:
            x = g[c].dropna()
            r[f"{c}_media"] = x.mean() if len(x) else np.nan
            r[f"{c}_sd"] = x.std() if len(x) > 1 else np.nan
        x = g[REALISTICI].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        r["auc_realistica_max"] = float(x.max()) if x.size else np.nan
        r["auc_realistica_min"] = float(x.min()) if x.size else np.nan
        righe.append(r)
    return pd.DataFrame(righe)


def md_celle(t: pd.DataFrame) -> str:
    f = lambda m, s: "-" if pd.isna(m) else (f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}")  # noqa: E731
    out = []
    for mac, g in t.groupby("macchina"):
        out.append(f"\n**{mac}**\n")
        intest = ["cella", "seed", "loss holdout"] + list(ATTACCHI.values()) + ["min-max realistiche"]
        out.append("| " + " | ".join(intest) + " |")
        out.append("|" + "---|" * len(intest))
        for _, r in g.iterrows():
            celle = [r["cella"].removeprefix("v3_"), str(r["n_seed"]),
                     f"{r['loss_media']:.4f}" + ("" if pd.isna(r["loss_sd"]) else f" ± {r['loss_sd']:.4f}")]
            celle += [f(r[f"{c}_media"], r[f"{c}_sd"]) for c in ATTACCHI]
            celle.append("-" if pd.isna(r["auc_realistica_min"]) else
                         f"{r['auc_realistica_min']:.3f}-{r['auc_realistica_max']:.3f}")
            out.append("| " + " | ".join(celle) + " |")
    return "\n".join(out)


def appaia(df: pd.DataFrame, a: str, b: str, cols: list[str]) -> pd.DataFrame:
    """Run della cella a e della cella b con lo stesso seed e la stessa macchina."""
    x = df[df["cella"] == a][["macchina", "seed"] + cols]
    y = df[df["cella"] == b][["macchina", "seed"] + cols]
    return x.merge(y, on=["macchina", "seed"], suffixes=("_a", "_b"))


def previsioni(run: pd.DataFrame, lp: pd.DataFrame) -> str:
    v3 = run[run["id"].str.startswith("v3_")]
    L: list[str] = []

    # V18: riproducibilita' con v2 / v2c della stessa cella, seed e macchina
    L.append("### V18 - riproducibilita' (stessa macchina, stesso seed)")
    mappa = []
    for c in sorted(v3["cella"].unique()):
        base = c.removeprefix("v3_")
        for pre in ("v2_", "v2c_"):
            if (run["cella"] == pre + base).any():
                mappa.append((c, pre + base))
    tot = ug_loss = ug_yeom = 0
    diff_max = 0.0
    for c3, c2 in mappa:
        p = appaia(run, c3, c2, ["loss_holdout_rilasciato", "yeom_auc_ultimo"])
        for _, r in p.iterrows():
            tot += 1
            d_l = abs(r["loss_holdout_rilasciato_a"] - r["loss_holdout_rilasciato_b"])
            d_y = abs(r["yeom_auc_ultimo_a"] - r["yeom_auc_ultimo_b"])
            ug_loss += d_l <= 1e-9 * max(1.0, abs(r["loss_holdout_rilasciato_b"]))
            ug_yeom += d_y <= 1e-9
            diff_max = max(diff_max, d_l / max(abs(r["loss_holdout_rilasciato_b"]), 1e-12))
    L.append(f"- Coppie v3/v2 confrontate: {tot} ({len(mappa)} celle). Loss sull'holdout identica: {ug_loss}/{tot}; "
             f"AUC di Yeom identica: {ug_yeom}/{tot}; massima differenza relativa della loss {diff_max:.2e}.")
    L.append(f"- Esito: {'confermata' if tot and ug_loss == tot and ug_yeom == tot else 'NON confermata, vedi sopra'}.")

    # V19: ACN senza DP, AUC realistiche in [0,45; 0,56] e non sopra l'universo + 0,01 in >= 4/5 seed
    L.append("\n### V19 - ACN senza DP, attacchi realistici")
    for mu in ("mu0", "mu0.01"):
        c3 = f"v3_acn_nodp_{mu}"
        g = v3[v3["cella"] == c3]
        if g.empty:
            continue
        x = g[REALISTICI].to_numpy(dtype=float)
        dentro = int(((x >= 0.45) & (x <= 0.56)).sum())
        L.append(f"- {c3}: {dentro}/{x.size} valori (6 forme x {len(g)} seed) in [0,45; 0,56]; "
                 f"intervallo {np.nanmin(x):.3f}-{np.nanmax(x):.3f}.")
        c2 = f"v2_acn_nodp_{mu}"
        for f3, f2 in (("composed_lira_auc", "composed_lira_auc"), ("composed_rmia_auc", "composed_rmia_auc")):
            p = appaia(run, c3, c2, [f3])
            if p.empty:
                continue
            ok = int((p[f"{f3}_a"] <= p[f"{f2}_b"] + 0.01).sum())
            L.append(f"  - {ATTACCHI[f3]} realistico <= universo + 0,01: {ok}/{len(p)} seed "
                     f"({'si' if ok >= 4 else 'no'}, soglia 4).")

    # V20: con DP l'AUC realistica non supera quella senza DP dello stesso seed di oltre 0,01
    L.append("\n### V20 - con DP l'AUC realistica non supera quella senza DP (+0,01)")
    coppie = []
    for c in sorted(v3["cella"].unique()):
        b = c.removeprefix("v3_")
        if "30r" in b or "partecipante" in b or "nodp" in b or "clip" in b or "noise" in b or b.startswith("cap"):
            continue
        fam = b.split("_")[0]  # acn, acn4f, sc3
        mu = "mu0" if b.endswith("_mu0") else "mu0.01"
        coppie.append((c, f"v3_{fam}_nodp_{mu}"))
    tot = viol = 0
    peggiori = []
    for c, rif in coppie:
        p = appaia(v3, c, rif, REALISTICI)
        for f in REALISTICI:
            d = p[f"{f}_a"] - p[f"{f}_b"]
            tot += int(d.notna().sum())
            viol += int((d > 0.01).sum())
            if d.notna().any():
                peggiori.append((float(d.max()), c, ATTACCHI[f]))
    peggiori.sort(reverse=True)
    L.append(f"- Confronti (cella con DP, forma, seed): {tot}; sopra +0,01: {viol} ({100 * viol / max(tot, 1):.0f}%).")
    L.append("- Differenze massime: " + "; ".join(f"{c.removeprefix('v3_')} {a} +{d:.3f}" for d, c, a in peggiori[:5]) + ".")
    L.append("- Lettura: senza DP l'AUC e' gia' circa 0,50, quindi le differenze con DP sono rumore di seed; una "
             "quota di superamenti di 0,01 e' attesa anche senza alcun effetto della DP.")

    # V21: shadow, FPR realizzato alla soglia dell'1% fuori da [0,3%; 5%] in >= 3/5 seed; vantaggio entro 0,02
    L.append("\n### V21 - shadow con dati ausiliari, celle senza DP")
    for c in sorted(c for c in v3["cella"].unique() if "nodp" in c and "30r" not in c and "partecipante" not in c):
        g = v3[v3["cella"] == c].dropna(subset=["shadow_fpr_soglia_fpr_ultimo"])
        if g.empty:
            continue
        fpr = g["shadow_fpr_soglia_fpr_ultimo"]
        fuori = int(((fpr < 0.003) | (fpr > 0.05)).sum())
        vant = g["shadow_vantaggio_soglia_fpr_ultimo"].abs().max()
        L.append(f"- {c.removeprefix('v3_')}: FPR realizzato {fpr.min() * 100:.2f}%-{fpr.max() * 100:.2f}%, fuori "
                 f"intervallo {fuori}/{len(g)} ({'si' if fuori >= 3 else 'no'}); |vantaggio| max {vant:.3f} "
                 f"({'si' if vant <= 0.02 else 'no'}).")

    # V22: classificatore di tipo Shokri entro 0,01 dalla soglia nelle celle senza DP
    L.append("\n### V22 - classificatore di tipo Shokri contro soglia sulla loss, celle senza DP")
    for c in sorted(c for c in v3["cella"].unique() if "nodp" in c and "30r" not in c and "partecipante" not in c):
        g = v3[v3["cella"] == c].dropna(subset=["shadow_clf_auc_ultimo", "shadow_auc_ultimo"])
        if g.empty:
            continue
        d = g["shadow_clf_auc_ultimo"] - g["shadow_auc_ultimo"]
        L.append(f"- {c.removeprefix('v3_')}: |clf - soglia| <= 0,01 in {int((d.abs() <= 0.01).sum())}/{len(g)}; "
                 f"clf sopra di oltre 0,01 in {int((d > 0.01).sum())}/{len(g)} (segnale se >= 3).")

    # V23: partecipante
    L.append("\n### V23 - avversario partecipante (solo modelli globali)")
    for cp, cs in (("v3_partecipante_acn_nodp_mu0.01", "v3_acn_nodp_mu0.01"),
                   ("v3_partecipante_acn_eps64_mu0.01", "v3_acn_eps64_mu0.01"),
                   ("v3_partecipante_acn_central_eps64", "v3_acn_central_eps64")):
        g = v3[v3["cella"] == cp]
        if g.empty:
            continue
        x = g[["composed_lira_auc", "composed_rmia_auc"]].to_numpy(dtype=float)
        p = appaia(v3, cp, cs, ["composed_lira_auc", "composed_rmia_auc"])
        ok = int(((p["composed_lira_auc_a"] <= p["composed_lira_auc_b"] + 0.01)
                  & (p["composed_rmia_auc_a"] <= p["composed_rmia_auc_b"] + 0.01)).sum())
        L.append(f"- {cp.removeprefix('v3_')}: LiRA e RMIA online {np.nanmin(x):.3f}-{np.nanmax(x):.3f} "
                 f"(in [0,45; 0,56]: {'si' if np.nanmin(x) >= 0.45 and np.nanmax(x) <= 0.56 else 'no'}); "
                 f"non sopra il server + 0,01 in {ok}/{len(p)} seed ({'si' if ok >= 4 else 'no'}).")

    # V25: ablazione ACN a sei feature
    L.append("\n### V25 - ablazione solo taglio / solo rumore, ACN a sei feature")
    p = appaia(v3, "v3_acn_clip_mu0", "v3_acn_clip_mu0.01", ["loss_holdout_rilasciato"])
    if not p.empty:
        ok = int((p["loss_holdout_rilasciato_a"] > p["loss_holdout_rilasciato_b"]).sum())
        rap = (p["loss_holdout_rilasciato_a"] / p["loss_holdout_rilasciato_b"])
        L.append(f"- Solo taglio, FedAvg con loss piu' alta di FedProx: {ok}/{len(p)} seed "
                 f"({'si' if ok >= 4 else 'no'}); rapporto FedAvg/FedProx {rap.min():.2f}-{rap.max():.2f}.")
    p = appaia(v3, "v3_acn_noise_mu0.01", "v3_acn_eps64_mu0.01", ["loss_holdout_rilasciato"])
    if not p.empty:
        rap = p["loss_holdout_rilasciato_a"] / p["loss_holdout_rilasciato_b"]
        ok = int(((rap >= 1 / 1.5) & (rap <= 1.5)).sum())
        L.append(f"- Solo rumore contro DP completa (FedProx): rapporto {rap.min():.2f}-{rap.max():.2f}, entro un "
                 f"fattore 1,5 in {ok}/{len(p)} seed ({'si' if ok == len(p) else 'no'}).")

    # V26: 30 round, round 1-10 uguali alla cella a 10 round
    L.append("\n### V26 - 30 round: i round 1-10 coincidono con la cella a 10 round")
    for c30, c10 in (("v3_30r_mu0_nodp", "v3_acn_nodp_mu0"), ("v3_30r_mu0.01_nodp", "v3_acn_nodp_mu0.01"),
                     ("v3_30r_mu0_eps64", "v3_acn_eps64_mu0"), ("v3_30r_mu0.01_eps64", "v3_acn_eps64_mu0.01"),
                     ("v3_sc3_30r_mu0_nodp", "v3_sc3_nodp_mu0"), ("v3_sc3_30r_mu0.01_nodp", "v3_sc3_nodp_mu0.01"),
                     ("v3_sc3_30r_mu0_eps64", "v3_sc3_eps64_mu0"),
                     ("v3_sc3_30r_mu0.01_eps64", "v3_sc3_eps64_mu0.01")):
        a = lp[(lp["cella"] == c30) & (lp["round"] <= 10)][["macchina", "seed", "round", "loss_holdout_rilasciato"]]
        b = lp[lp["cella"] == c10][["macchina", "seed", "round", "loss_holdout_rilasciato"]]
        m = a.merge(b, on=["macchina", "seed", "round"], suffixes=("_30", "_10"))
        if m.empty:
            L.append(f"- {c30}: nessuna coppia con {c10}.")
            continue
        rel = (m["loss_holdout_rilasciato_30"] - m["loss_holdout_rilasciato_10"]).abs() / m["loss_holdout_rilasciato_10"].abs().clip(lower=1e-12)
        sem = m.groupby("seed").apply(lambda g: bool((((g["loss_holdout_rilasciato_30"] - g["loss_holdout_rilasciato_10"]).abs())
                                                      <= 1e-9 * g["loss_holdout_rilasciato_10"].abs().clip(lower=1)).all()),
                                      include_groups=False)
        L.append(f"- {c30} contro {c10}: seed con round 1-10 identici {int(sem.sum())}/{len(sem)}; "
                 f"massima differenza relativa {rel.max():.2e}.")
    L.append("- Differenza FedAvg - FedProx al round 30 (loss sull'holdout, media sui seed, senza previsione):")
    for pre, nome in (("v3_30r", "ACN"), ("v3_sc3_30r", "Scozia")):
        for dp in ("nodp", "eps64"):
            p = appaia(v3, f"{pre}_mu0_{dp}", f"{pre}_mu0.01_{dp}", ["loss_holdout_rilasciato"])
            if p.empty:
                continue
            rap = p["loss_holdout_rilasciato_a"] / p["loss_holdout_rilasciato_b"]
            L.append(f"  - {nome} {dp}: FedAvg/FedProx {rap.mean():.2f} (min {rap.min():.2f}, max {rap.max():.2f}, "
                     f"FedAvg peggiore in {int((rap > 1).sum())}/{len(p)} seed).")
    L.append("\n### V24 - canary per sessione (PC Windows): aperta, run in corso.")
    return "\n".join(L)


def previsioni_soggetto(s: pd.DataFrame) -> str:
    """V27 e V28 (attacco realistico al soggetto)."""
    if s.empty:
        return ""
    s = s.copy()
    s["dataset"] = np.where(s["chiave"] == "node_id", "Scozia", "ACN")
    pr = s[s["primaria"].astype(str) == "True"]
    L = ["\n### V27 - entita', senza DP: AUC primaria in [0,45; 0,58]; DP per soggetto non sopra +0,02"]
    for ds in ("ACN", "Scozia"):
        n = pr[(pr["dataset"] == ds) & (pr["livello"] == "entita") & (pr["etichetta"].str.endswith("nodp"))]
        if n.empty:
            continue
        dentro = int(((n["auc"] >= 0.45) & (n["auc"] <= 0.58)).sum())
        L.append(f"- {ds}: {dentro}/{len(n)} seed in [0,45; 0,58] (AUC {n['auc'].min():.3f}-{n['auc'].max():.3f}).")
        for et in sorted(pr[(pr["dataset"] == ds) & (pr["livello"] == "entita")
                            & pr["etichetta"].str.contains("subject")]["etichetta"].unique()):
            d = pr[(pr["dataset"] == ds) & (pr["livello"] == "entita") & (pr["etichetta"] == et)]
            m = d.merge(n, on=["macchina", "seed"], suffixes=("_dp", "_no"))
            ok = int((m["auc_dp"] <= m["auc_no"] + 0.02).sum())
            L.append(f"  - {et}: non sopra senza DP + 0,02 in {ok}/{len(m)} seed.")
    L.append("\n### V28 - gruppo, DP per record: segnale se AUC primaria > 0,55 in almeno 2 seed su 3 con sigma 2")
    g = pr[(pr["livello"] == "gruppo") & pr["etichetta"].str.contains("record")]
    if g.empty:
        L.append("- Nessuna run di gruppo con DP per record raccolta.")
    for (ds, et), d in g.groupby(["dataset", "etichetta"]):
        L.append(f"- {ds} {et}: AUC primaria {d['auc'].min():.3f}-{d['auc'].max():.3f}, sopra 0,55 in "
                 f"{int((d['auc'] > 0.55).sum())}/{len(d)} seed.")
    return "\n".join(L)


CANARY = ["canary_raw_auc_ultimo", "canary_lira_auc_ultimo", "canary_lira_out_auc_ultimo",
          "canary_lira_offline_auc_ultimo", "canary_rmia_auc_ultimo", "canary_rmia_out_auc_ultimo",
          "canary_rmia_offline_auc_ultimo", "shadow_canary_auc_ultimo", "shadow_clf_canary_auc_ultimo"]


def canary(run: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Canary per sessione v3 (bracci A e B a ruoli scambiati) e verifica di V24 sui seed disponibili."""
    c = run[run["id"].str.match(r"v3_canary[AB]_")].copy()
    if c.empty:
        return c, "\n### V24 - canary per sessione: nessuna run raccolta.\n"
    c["braccio"] = c["id"].str.extract(r"v3_canary([AB])_")[0]
    c["dp"] = np.where(c["id"].str.contains("_nodp_"), "nodp", "local_eps64")
    tab = c[["macchina", "seed", "dp", "braccio", "loss_holdout_rilasciato"] + CANARY].sort_values(
        ["dp", "seed", "braccio"])
    L = ["\n### V24 - canary per sessione senza DP: loss grezza sopra 0,5 in ogni braccio, A + B sopra 1 in ogni seed"]
    n = c[c["dp"] == "nodp"]
    for (mac, seed), g in n.groupby(["macchina", "seed"]):
        v = dict(zip(g["braccio"], g["canary_raw_auc_ultimo"]))
        tot = sum(v.values())
        L.append(f"- seed {seed} ({mac}): A {v.get('A', float('nan')):.3f}, B {v.get('B', float('nan')):.3f}, "
                 f"somma {tot:.3f} ({'si' if len(v) == 2 and min(v.values()) > 0.5 and tot > 1 else 'no o incompleto'}).")
    semi = sorted(n["seed"].unique())
    L.append(f"- Seed raccolti: {', '.join(map(str, semi))}. La previsione e' su 3 seed (42, 123, 456); "
             "789 e 1234 sono un'estensione decisa il 23 ottobre.")
    return tab, "\n".join(L)


def soggetto(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["dataset"] = np.where(s["chiave"] == "node_id", "Scozia", "ACN")
    righe = []
    for (ds, liv, et), g in s.groupby(["dataset", "livello", "etichetta"]):
        pr = g[g["primaria"].astype(str) == "True"]
        ls = g[(g["limite_superiore"].astype(str) == "True") & g["attacco"].str.endswith("calibrato")]
        righe.append({
            "dataset": ds, "livello": liv, "cella": et.removeprefix("sc3"), "n_seed": pr["seed"].nunique(),
            "metrica_primaria": pr["attacco"].iloc[0] if len(pr) else None,
            "auc_primaria_media": pr["auc"].mean(), "auc_primaria_min": pr["auc"].min(),
            "auc_primaria_max": pr["auc"].max(), "seed_p_sotto_0_05": int((pr["p"] < 0.05).sum()),
            "auc_limite_sup_calibrato_media": ls["auc"].mean() if len(ls) else np.nan,
            "costo_loss_non_membri": g["costo_loss_non_membri"].mean(),
        })
    return pd.DataFrame(righe).sort_values(["dataset", "livello", "cella"])


def md_soggetto(t: pd.DataFrame) -> str:
    out = ["| dataset | livello | cella | seed | metrica primaria | AUC media (min-max) | seed con p < 0,05 | "
           "limite sup. calibrato |", "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        ls = "-" if pd.isna(r["auc_limite_sup_calibrato_media"]) else f"{r['auc_limite_sup_calibrato_media']:.3f}"
        out.append(f"| {r['dataset']} | {r['livello']} | {r['cella']} | {r['n_seed']} | {r['metrica_primaria']} | "
                   f"{r['auc_primaria_media']:.3f} ({r['auc_primaria_min']:.3f}-{r['auc_primaria_max']:.3f}) | "
                   f"{r['seed_p_sotto_0_05']}/{r['n_seed']} | {ls} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--macchina", nargs=2, action="append", metavar=("NOME", "CARTELLA_RISULTATI"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run, sog, lp = carica(a.macchina)
    v3 = run[run["id"].str.startswith("v3_")]
    t = tabella_celle(v3)
    t.to_csv(out / "tabella_celle_v3.csv", index=False)
    (out / "tabella_celle_v3.md").write_text(
        "# Campagna v3, celle per macchina (media ± deviazione standard sui seed)\n\n"
        "AUC composte sui round per LiRA e RMIA; Yeom, shadow e classificatore all'ultimo round. "
        "Le run a 30 round non hanno attacchi (solo utilita' e Yeom).\n" + md_celle(t) + "\n")
    if not sog.empty:
        ts = soggetto(sog)
        ts.to_csv(out / "soggetto_v3.csv", index=False)
        (out / "soggetto_v3.md").write_text("# Attacco realistico al soggetto, campagna v3\n\n" + md_soggetto(ts) + "\n")
    tc, v24 = canary(run)
    if not tc.empty:
        tc.to_csv(out / "canary_sessione_v3.csv", index=False)
    (out / "previsioni_v3.md").write_text("# Campagna v3: verifica delle previsioni scritte prima del lancio\n\n"
                                          + previsioni(run, lp).replace(
                                              "\n### V24 - canary per sessione (PC Windows): aperta, run in corso.", "")
                                          + "\n" + v24 + "\n" + previsioni_soggetto(sog) + "\n")
    print(f"{len(v3)} run v3, {t['cella'].nunique()} celle -> {out}")


if __name__ == "__main__":
    main()
