"""fig_canary: controlli positivi (2026-10-30).

Sinistra: canary per sessione su Office 1, somma AUC_A + AUC_B al round 3 (5 seed), RISULTATI.md 1.2.
Destra: canary persona al round 3; con la DP solo le run con rumore indipendente fra i bracci
(5 seed), RISULTATI.md 1.3. Le prime run con rumore condiviso (0,843 / 0,543) non si usano.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "hatch.color": "white", "hatch.linewidth": 1.2})

BLU, ARANCIO, VERDE = "#2a78d6", "#ee6a35", "#1fa97a"
REG = [("Senza DP", BLU, ""), ("DP centrale", ARANCIO, "///"), ("DP locale", VERDE, "..")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"wspace": 0.28})

# --- canary per sessione (Office 1, 5 seed, round 3)
att = ["Perdita grezza", "LiRA offline", "RMIA offline"]
val = {"Senza DP": [1.127, 1.284, 1.287], "DP centrale": [1.099, 1.280, 1.240], "DP locale": [1.014, 0.999, 1.012]}
w = 0.26
for k, (nome, col, h) in enumerate(REG):
    xs = [i + (k - 1) * w for i in range(len(att))]
    hs = [v - 1 for v in val[nome]]
    a1.bar(xs, hs, w * 0.92, bottom=1, color=col, hatch=h, edgecolor="white", linewidth=0, label=nome)
a1.axhline(1, color="#444444", lw=1)
a1.text(2.45, 1.0, "caso", va="bottom", ha="left", fontsize=9, color="#444444")
a1.set_xticks(range(len(att)), att)
a1.set_ylim(0.95, 1.33)
a1.set_ylabel(r"AUC$_A$+AUC$_B$ (media, 5 seed)")
a1.set_title("Canary per sessione — aggregatore")

# --- canary persona (round 3)
grp = ["Aggregatore", "Modello rilasciato"]
pv = {"Senza DP": ([0.802, 0.697], ["10/10", "8/10"]),
      "DP centrale": ([0.711, 0.547], ["4/5", "0/5"]),
      "DP locale": ([0.487, 0.564], ["0/5", "1/5"])}
for k, (nome, col, h) in enumerate(REG):
    xs = [i + (k - 1) * w for i in range(len(grp))]
    v, sig = pv[nome]
    a2.bar(xs, [x - 0.5 for x in v], w * 0.92, bottom=0.5, color=col, hatch=h, edgecolor="white", linewidth=0)
    for x, y, s in zip(xs, v, sig):
        a2.text(x, max(y, 0.5) + 0.006, s, ha="center", va="bottom", fontsize=8.5, color="#444444")
a2.axhline(0.5, color="#444444", lw=1)
a2.set_xticks(range(len(grp)), grp)
a2.set_ylim(0.45, 0.9)
a2.set_ylabel("AUC appaiato (round 3)")
a2.set_title("Canary persona (DP: rumore indipendente, 5 seed)")

for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#e6e6e6", lw=0.8)
    a.set_axisbelow(True)
fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print("ok")
