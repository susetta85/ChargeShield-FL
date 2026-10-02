"""Clipping su tutti i JSON con DP per client che hanno le norme dei delta (solo lettura).

Per ogni run e round: quota trattenuta f = sum_k w_k min(1, C/||Delta_k||) con w_k = n_k/N
(n_train_per_client), quota cumulata S = sum_t f_t, loss sull'holdout del modello rilasciato
(-per_round[r].mia.non_member_score_mean). Scrive risultati/clipping/per_round.csv e per_run.csv
e stampa la tabella per impostazione. Non tocca esperimenti ne' codice di training.
Uso: python3 scripts/analisi_clipping_json.py  (dalla radice del repository). 2026-10-02.
"""
import json, glob, math, statistics as st, csv, os
OUT = "risultati/clipping"
os.makedirs(OUT, exist_ok=True)
EA = "experiments_altre_macchine"
DELTA = 1e-5; D_PARAMS = 570
# (etichetta, cartella DP, cartella no-DP, macchina, algoritmo)
SETS = [
 ("FedAvg C1 e64 (Mac)",      "experiments/rq3-mu0-eps64",           "experiments/rq3-mu0",       "Mac princ.", "FedAvg"),
 ("FedProx C1 e64 (Mac)",     "experiments/rq1-eps64",               "experiments/nodp-sweep2",   "Mac princ.", "FedProx"),
 ("FedAvg C1 e64 (Win)",      f"{EA}/rq3-mu0-eps64",                 f"{EA}/rq3-mu0",             "Windows",    "FedAvg"),
 ("FedProx C1 e64 (Win)",     f"{EA}/rq3-mu0.01-eps64",              f"{EA}/rq3-mu0.01",          "Windows",    "FedProx"),
 ("FedProx C1 e64 per_site (Mac2)", f"{EA}/rq2-per_site-eps64",      f"{EA}/rq2-per_site",        "secondo Mac","FedProx"),
 ("FedProx C1 e64 iid (Mac2)",      f"{EA}/rq2-iid-eps64",           f"{EA}/rq2-iid",             "secondo Mac","FedProx"),
 ("FedProx C0.25 e16",  "experiments/_op_C0.25_eps16",  "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C0.25 e64",  "experiments/_op_C0.25_eps64",  "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C0.25 e256", "experiments/_op_C0.25_eps256", "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C0.5 e16",   "experiments/_op_C0.5_eps16",   "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C0.5 e64",   "experiments/_op_C0.5_eps64",   "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C0.5 e256",  "experiments/_op_C0.5_eps256",  "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C1 e32",     "experiments/_op_eps32",        "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C1 e64 (op)","experiments/_op_eps64",        "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedProx C1 e256",    "experiments/_op_eps256",       "experiments/nodp-sweep2", "Mac princ.", "FedProx"),
 ("FedAvg C2 e64",      "experiments/_rq3_mu0_eps64_C2_s42", "experiments/rq3-mu0",  "Mac princ.", "FedAvg"),
 ("FedAvg C4 e64",      "experiments/_rq3_mu0_eps64_C4_s42", "experiments/rq3-mu0",  "Mac princ.", "FedAvg"),
 ("FedAvg C8 e64",      "experiments/_rq3_mu0_eps64_C8_s42", "experiments/rq3-mu0",  "Mac princ.", "FedAvg"),
]
def load(d):
    out = {}
    for f in glob.glob(d + "/experiment_*.json"):
        j = json.load(open(f)); out[j["config"]["seed"]] = j
    return out
def hold(j, r): return -j["per_round"][str(r)]["mia"]["non_member_score_mean"]
def rounds(j): return sorted(int(k) for k in j["per_round"])
def info(j):
    c = j["config"]; C = c["max_grad_norm"]; eps = c["epsilon"]
    w = c["n_train_per_client"]; N = sum(w.values())
    sig = C * math.sqrt(2 * math.log(1.25 / DELTA)) / eps
    sig_agg = sig * math.sqrt(sum((v / N) ** 2 for v in w.values()))
    return C, eps, w, N, sig, sig_agg
def per_round(j):
    C, eps, w, N, sig, sig_agg = info(j)
    rows = []; S = 0.0
    for r in rounds(j):
        nm = j["per_round"][str(r)]["fl"]["delta_norm_per_client"]
        tot = sum(w[k] for k in nm)
        f = sum(w[k] * min(1, C / v) for k, v in nm.items()) / tot          # quota trattenuta, pesata sui dati
        applied = sum(w[k] * min(v, C) for k, v in nm.items()) / tot        # norma applicata media (limite sup.)
        frac_clipped = sum(w[k] for k, v in nm.items() if v > C) / tot       # quota (dati) tagliata, binaria
        S += f
        rows.append(dict(r=r, f=f, S=S, applied=applied, frac_clipped=frac_clipped,
                         noise_norm=sig_agg * math.sqrt(D_PARAMS), loss=hold(j, r),
                         med_norm=st.median(nm.values())))
    return rows
def interp_logloss(rows, S0):
    # interpolazione lineare di log(loss) in S; None se S0 fuori dall'intervallo
    pts = [(0.0, None)] + [(x["S"], math.log(x["loss"])) for x in rows]
    for (s1, l1), (s2, l2) in zip(pts[1:], pts[2:]):
        if s1 <= S0 <= s2:
            return math.exp(l1 + (l2 - l1) * (S0 - s1) / (s2 - s1))
    return None
out_rows = []; summary = []
for lab, ddp, dnd, mach, alg in SETS:
    DP = load(ddp); ND = load(dnd)
    if not DP: print("VUOTO", ddp); continue
    for s, j in sorted(DP.items()):
        rows = per_round(j); C, eps, w, N, sig, sig_agg = info(j)
        R = rows[-1]["r"]
        nd = hold(ND[s], R) if s in ND and str(R) in ND[s]["per_round"] else None
        for x in rows:
            out_rows.append(dict(setting=lab, machine=mach, alg=alg, C=C, eps=eps, seed=s, **x))
        summary.append(dict(setting=lab, alg=alg, C=C, eps=eps, seed=s, sigma=sig, noise_norm=sig_agg*math.sqrt(D_PARAMS),
            f1=rows[0]["f"], f2=rows[1]["f"], f3=rows[2]["f"], f10=rows[-1]["f"], S10=rows[-1]["S"],
            loss2=rows[1]["loss"], loss10=rows[-1]["loss"], nodp10=nd, ratio10=(rows[-1]["loss"]/nd if nd else None),
            L_at_065=interp_logloss(rows, 0.65), L_at_150=interp_logloss(rows, 1.5)))
with open(f"{OUT}/per_round.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); wr.writeheader(); wr.writerows(out_rows)
with open(f"{OUT}/per_run.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); wr.writeheader(); wr.writerows(summary)
def g(x): 
    x=[v for v in x if v is not None]; return math.exp(st.mean(math.log(v) for v in x)) if x else float('nan')
print(f"{'setting':32s} n  noise|  f1    f2    f3   f10 |  S10 | loss2   loss10  | L@S=.65 L@S=1.5 | ratio10")
labs = []
for x in summary:
    if x["setting"] not in labs: labs.append(x["setting"])
for lab in labs:
    xs = [x for x in summary if x["setting"] == lab]; m = lambda k: st.mean(v[k] for v in xs)
    print(f"{lab:32s} {len(xs)} {xs[0]['noise_norm']:5.2f}| {m('f1'):.3f} {m('f2'):.3f} {m('f3'):.3f} {m('f10'):.3f} | {m('S10'):4.2f} | {g([v['loss2'] for v in xs]):.4f} {g([v['loss10'] for v in xs]):.5f} | {g([v['L_at_065'] for v in xs]):.4f}  {g([v['L_at_150'] for v in xs]):.4f} | {g([v['ratio10'] for v in xs]):.2f}")
