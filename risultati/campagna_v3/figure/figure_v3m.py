"""Figure dei risultati aggiornate al 2026-11-01 (code v3p e v3m).

Uso: python figure_v3m.py <cartella_campagna_v3> <cartella_uscita>
Produce, in PDF e PNG:
  fig_canary       controlli positivi; canary persona con DP a 10 seed (RISULTATI.md 1.2, 1.3)
  fig_clipping     rapporto FedProx/FedAvg per norma di clipping C, tabelle A e B (RISULTATI.md 2.4d)
  fig_sessione_dp  canary persona con DP per sessione e per soggetto (RISULTATI.md 3.3, 3.3b)
  fig_scozia32     Scotland con 3 e con 32 client, rapporto FedProx/FedAvg con la DP (RISULTATI.md 2.2, 2.2b)
I valori delle tabelle già consolidate sono scritti qui; quelli per seed sono letti dai CSV.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

CAMP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "hatch.color": "white", "hatch.linewidth": 1.2})
BLU, ARANCIO, VERDE = "#2a78d6", "#ee6a35", "#1fa97a"
INK, MUTED, GRID = "#222222", "#555555", "#e6e6e6"


def stile(a):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)


def virgole(a, cifre=2):
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.{cifre}f}".replace(".", ",")))


def salva(fig, nome):
    fig.savefig(OUT / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nome}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def virgola(x, cifre=2):
    return f"{x:.{cifre}f}".replace(".", ",")


run = pd.concat([pd.read_csv(CAMP / m / "riepilogo_run.csv") for m in ["mac_susy", "mac_domenico", "windows"]])
run = run.drop_duplicates("id")
run["cella"] = run["id"].str.replace(r"_s\d+$", "", regex=True)


def per_seed(cella):
    return run[run.cella == cella].set_index("seed")["loss_holdout_rilasciato"]


# ------------------------------------------------------------------ fig_canary
REG = [("Senza DP", BLU, ""), ("DP centrale", ARANCIO, "///"), ("DP locale", VERDE, "..")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"wspace": 0.28})
att = ["Perdita grezza", "LiRA offline", "RMIA offline"]
val = {"Senza DP": [1.127, 1.284, 1.287], "DP centrale": [1.099, 1.280, 1.240], "DP locale": [1.014, 0.999, 1.012]}
w = 0.26
for k, (nome, col, h) in enumerate(REG):
    xs = [i + (k - 1) * w for i in range(len(att))]
    a1.bar(xs, [v - 1 for v in val[nome]], w * 0.92, bottom=1, color=col, hatch=h, edgecolor="white",
           linewidth=0, label=nome)
a1.axhline(1, color="#444444", lw=1)
a1.text(2.45, 1.0, "caso", va="bottom", ha="left", fontsize=9, color=MUTED)
a1.set_xticks(range(len(att)), att)
a1.set_ylim(0.95, 1.33)
a1.set_ylabel(r"AUC$_A$+AUC$_B$ (media, 5 seed)")
a1.set_title("Canary per sessione — aggregatore")
grp = ["Aggregatore", "Modello rilasciato"]
pv = {"Senza DP": ([0.802, 0.697], ["10/10", "8/10"]),
      "DP centrale": ([0.722, 0.527], ["9/10", "0/10"]),
      "DP locale": ([0.504, 0.533], ["0/10", "1/10"])}
for k, (nome, col, h) in enumerate(REG):
    xs = [i + (k - 1) * w for i in range(len(grp))]
    v, sig = pv[nome]
    a2.bar(xs, [x - 0.5 for x in v], w * 0.92, bottom=0.5, color=col, hatch=h, edgecolor="white", linewidth=0)
    for x, y, s in zip(xs, v, sig):
        a2.text(x, max(y, 0.5) + 0.006, s, ha="center", va="bottom", fontsize=8.5, color=MUTED)
a2.axhline(0.5, color="#444444", lw=1)
a2.set_xticks(range(len(grp)), grp)
a2.set_ylim(0.45, 0.9)
a2.set_ylabel("AUC appaiato (round 3)")
a2.set_title("Canary persona (10 seed; DP con rumore indipendente)")
for a in (a1, a2):
    stile(a)
    virgole(a)
fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
salva(fig, "fig_canary")

# ------------------------------------------------------------------ fig_clipping
C = [0.25, 0.5, 1, 2]
celle_a = ["v3h_acn_C0.25", "v3h_acn_C0.5", "v3_acn_eps64", "v3h_acn_C2"]
celle_b = ["v3p_acn_rC1_C0.25", "v3p_acn_rC1_C0.5", "v3_acn_eps64", "v3p_acn_rC1_C2"]


def rapporti(celle):
    med, sig, avg, prox = [], [], [], []
    for c in celle:
        a, b = per_seed(f"{c}_mu0"), per_seed(f"{c}_mu0.01")
        comuni = a.index.intersection(b.index)
        r = b[comuni] / a[comuni]
        med.append(float(r.median()))
        sig.append(f"{int((r < 1).sum())}/{len(comuni)}")
        avg.append(float(a.mean()))
        prox.append(float(b.mean()))
    return med, sig, avg, prox


ma, sa, _, _ = rapporti(celle_a)
mb, sb, avg_b, prox_b = rapporti(celle_b)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"wspace": 0.3})
a1.plot(C, ma, color=BLU, lw=2, marker="o", ms=7, label="A: rumore proporzionale a C")
a1.plot(C, mb, color=ARANCIO, lw=2, ls="--", marker="s", ms=7, label="B: rumore uguale per tutti i C")
for x, y, s in zip(C, ma, sa):
    a1.annotate(s, (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=MUTED)
for x, y, s in zip(C, mb, sb):
    if x != 1:
        a1.annotate(s, (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=MUTED)
a1.axhline(1, color="#444444", lw=1)
a1.text(0.23, 1.06, "nessun vantaggio", fontsize=8.5, color=MUTED)
a1.set_xscale("log", base=2)
a1.set_yscale("log")
a1.set_xticks(C, ["0,25", "0,5", "1", "2"])
a1.set_yticks([0.1, 0.2, 0.5, 1], ["0,1", "0,2", "0,5", "1"])
a1.minorticks_off()
a1.set_ylim(0.07, 1.6)
a1.set_xlabel("norma di clipping C")
a1.set_ylabel("FedProx / FedAvg (mediana per seed)")
a1.set_title("Vantaggio di FedProx (μ = 0,01)")
a1.legend(frameon=False, fontsize=8.5, loc="lower right")
x = np.arange(len(C))
a2.bar(x - 0.2, avg_b, 0.38, color=BLU, label="FedAvg (μ = 0)")
a2.bar(x + 0.2, prox_b, 0.38, color=ARANCIO, hatch="///", edgecolor="white", linewidth=0, label="FedProx (μ = 0,01)")
a2.set_xticks(x, ["0,25\nε = 9,83", "0,5\nε = 23,94", "1\nε = 64", "2\nε = 190,99"])
a2.set_xlabel("norma di clipping C (tabella B, stesso σ)")
a2.set_ylabel("perdita sull'holdout (media, 10 seed)")
a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: virgola(v, 3)))
a2.set_title("Rumore uguale: perdite al round 10")
a2.legend(frameon=False, fontsize=8.5)
for a in (a1, a2):
    stile(a)
salva(fig, "fig_clipping")

# ------------------------------------------------------------------ fig_sessione_dp
cat = ["Nessuna\nDP", "Sessione\nσ = 1\nε ≈ 10,8", "Sessione\nσ = 2\nε ≈ 3,7", "Sessione\nσ = 5\nε ≈ 1,3",
       "Sessione\nσ = 10\nε ≈ 0,6", "Soggetto\nσ = 2\nε ≈ 112", "Soggetto\nσ = 5\nε ≈ 29"]
r1 = [0.999, 0.645, 0.694, 0.664, 0.697, 0.596, 0.516]
r3 = [0.802, 0.595, 0.551, 0.614, 0.617, 0.542, 0.499]
s1 = ["10/10", "2/5", "3/5", "4/10", "4/5", "2/5", "0/5"]
s3 = ["10/10", "1/5", "1/5", "4/10", "2/5", "1/5", "0/5"]
fig, a = plt.subplots(figsize=(10.2, 3.9))
x = np.arange(len(cat))
a.bar(x - 0.2, [v - 0.5 for v in r1], 0.38, bottom=0.5, color=BLU, label="round 1")
a.bar(x + 0.2, [v - 0.5 for v in r3], 0.38, bottom=0.5, color=ARANCIO, hatch="///", edgecolor="white",
      linewidth=0, label="round 3")
for xi, v, s in zip(x - 0.2, r1, s1):
    a.text(xi, max(v, 0.5) + 0.008, s, ha="center", va="bottom", fontsize=8, color=MUTED)
for xi, v, s in zip(x + 0.2, r3, s3):
    a.text(xi, max(v, 0.5) + 0.008, s, ha="center", va="bottom", fontsize=8, color=MUTED)
a.axhline(0.5, color="#444444", lw=1)
for sep in (0.5, 4.5):
    a.axvline(sep, color="#bbbbbb", lw=0.8, ls=":")
a.text(2.5, 1.06, "DP per sessione (ε per sessione)", ha="center", fontsize=9, color=INK)
a.text(5.5, 1.06, "DP per soggetto (ε per soggetto)", ha="center", fontsize=9, color=INK)
a.set_xticks(x, cat, fontsize=8.5)
a.set_ylim(0.45, 1.1)
a.set_ylabel("AUC appaiato all'aggregatore")
a.legend(frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.03))
stile(a)
virgole(a, 1)
salva(fig, "fig_sessione_dp")

# ------------------------------------------------------------------ fig_scozia32
fig, a = plt.subplots(figsize=(6.4, 3.6))
gruppi = [("Scotland, 3 client", "v3_sc3_eps64", [0.87, 0.49], ["p = 0,56", "p = 0,015"]),
          ("Scotland, 32 client", "v3m_sc32_eps64", [1.04, 0.69], ["p = 0,89", "p = 0,21"])]
cols = [(BLU, "", "μ = 0,01"), (ARANCIO, "///", "μ = 0,1")]
rng = np.random.default_rng(0)
for g, (nome, pref, rapp, pp) in enumerate(gruppi):
    base = per_seed(f"{pref}_mu0")
    for k, (mu, (col, h, lab)) in enumerate(zip(["0.01", "0.1"], cols)):
        xc = g + (k - 0.5) * 0.36
        a.bar(xc, rapp[k] - 1, 0.33, bottom=1, color=col, hatch=h, edgecolor="white", linewidth=0,
              label=lab if g == 0 else None)
        b = per_seed(f"{pref}_mu{mu}")
        comuni = base.index.intersection(b.index)
        r = (b[comuni] / base[comuni]).values
        a.scatter(xc + rng.uniform(-0.07, 0.07, len(r)), r, s=14, color="white", edgecolor=INK, linewidth=0.8,
                  zorder=3, label="rapporto per seed" if (g == 0 and k == 0) else None)
        a.text(xc, max(rapp[k], 1) * 1.04 if rapp[k] >= 1 else 1.04, f"{virgola(rapp[k])}\n{pp[k]}",
               ha="center", va="bottom", fontsize=8, color=MUTED)
a.axhline(1, color="#444444", lw=1)
a.set_yscale("log")
a.set_yticks([0.2, 0.5, 1, 2, 5], ["0,2", "0,5", "1", "2", "5"])
a.minorticks_off()
a.set_xticks([0, 1], [g[0] for g in gruppi])
a.set_ylabel("FedProx / FedAvg con la DP\n(rapporto delle medie, round 10)")
a.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.32))
stile(a)
salva(fig, "fig_scozia32")
print("ok")
