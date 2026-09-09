#!/usr/bin/env python3
"""
Privacy Exposure Score (PES) — calcolo retroattivo su JSON già esistenti.

Implementa PES v1 (docs/PrivacyExposureScore_v1.md, definizione originale
2026-07-22) e un primo PES v1.1 concreto (task #41, Sprint 10zz+13,
2026-09-02) — la sequenza che lo stesso documento indicava come necessaria
("implementare TPR@low-FPR per primo, poi PES v1.1 diventa ben definito",
sezione "Concrete refinement", 2026-08-27) è ora soddisfatta: TPR@low-FPR
è stato aggiunto in Sprint 10pp (2026-08-28).

Riusa discover_groups() da check_significance.py (stesso raggruppamento
per config letto dal JSON, non dal nome cartella — stesso fix del
2026-08-27/2026-08-28) invece di duplicarne la logica.

PES v1 è calcolabile su QUALUNQUE esperimento già completato (usa solo
mean_lira_auc_roc + config.epsilon/delta, sempre salvati). PES v1.1
richiede tpr_at_fpr_0.01, salvato SOLO nei run successivi a Sprint 10pp
(2026-08-28) — riportato "N/A (tpr_at_fpr assente)" altrove, mai stimato
o approssimato da AUC.

Uso:
    python3 scripts/compute_pes.py [--fpr-target 0.01] [--include-diagnostic]
"""
import argparse
import glob
import json
import math
import os

from check_significance import discover_groups


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pes_v1(auc: float | None, epsilon: float | None, no_dp: bool) -> dict:
    """
    PES v1 (docs/PrivacyExposureScore_v1.md):
        L(AUC)      = clip(2 * max(0, AUC - 0.5), 0, 1)
        strength(ε) = 1/(1+ε) per una config DP, 0 per la baseline no-DP
        PES_v1      = L(AUC) * strength(ε)
    """
    if auc is None:
        return {"L": None, "strength": None, "pes_v1": None}
    L = clip(2 * max(0.0, auc - 0.5), 0.0, 1.0)
    if no_dp or epsilon is None:
        strength = 0.0
    else:
        strength = 1.0 / (1.0 + epsilon)
    return {"L": round(L, 6), "strength": round(strength, 6), "pes_v1": round(L * strength, 6)}


def humphries_bound(epsilon: float, delta: float) -> float:
    """
    Humphries et al. 2020, "Differentially Private Learning Does Not Bound
    Membership Inference": Adv ≤ (e^ε − 1 + 2δ) / (e^ε + 1). Bound più
    stretto di quello di Yeom (e^ε−1) perché tiene conto di δ — corretto
    per il meccanismo Gaussiano (ε,δ)-DP usato in questo progetto (vedi
    docs/PrivacyExposureScore_v1.md, "Tighter bound", 2026-08-27).
    """
    e_eps = math.exp(epsilon)
    return (e_eps - 1 + 2 * delta) / (e_eps + 1)


def pes_v1_1(
    tpr_at_target_fpr: float | None,
    fpr_target: float,
    epsilon: float | None,
    delta: float,
    no_dp: bool,
) -> dict:
    """
    PES v1.1 (task #41, Sprint 10zz+13, 2026-09-02) — prima implementazione
    concreta della "raffinatezza" proposta ma non implementata in
    docs/PrivacyExposureScore_v1.md ("Concrete refinement", 2026-08-27).

    SCELTA METODOLOGICA NUOVA, non ancora nel documento originale — da
    dichiarare esplicitamente come tale nel paper, non presentarla come
    "la" definizione consolidata:

        L_v1.1  = clip(TPR@FPR=fpr_target − fpr_target, 0, 1)
                  # "vantaggio" dell'attaccante a UNA soglia fissa e bassa
                  # (l'operating point che Carlini et al. 2022 argomentano
                  # essere quello realistico), invece della media su tutte
                  # le soglie (AUC) usata da L(AUC) in v1. A differenza di
                  # max(TPR-FPR) (Youden J, vedi _mia_advantage() in
                  # scripts/run_experiments.py) che sceglie la soglia
                  # OTTIMALE per l'attaccante, qui la soglia è FISSATA a
                  # fpr_target — più conservativo, più direttamente
                  # confrontabile con l'operating point che TPR@low-FPR
                  # già riporta come "headline" nei Sprint-log.
        PES_v1.1 = L_v1.1 / humphries_bound(ε, δ)
                  # frazione del vantaggio massimo formalmente permesso da
                  # (ε,δ)-DP che l'attaccante ha realmente ottenuto — la
                  # normalizzazione "contro il soffitto teorico" che
                  # PES v1's strength(ε) = 1/(1+ε) esplicitamente NON è
                  # (vedi "What is NOT grounded" nel documento originale).

    Returns:
        {"L_v1.1": ..., "humphries_bound": ..., "pes_v1_1": ...} — valori
        None quando l'input necessario manca (no-DP: nessun soffitto da
        normalizzare contro, N/A non 0 — diverso da v1's strength=0, che
        E' definito anche per no-DP).
    """
    if tpr_at_target_fpr is None:
        return {"L_v1_1": None, "humphries_bound": None, "pes_v1_1": None}
    L = clip(tpr_at_target_fpr - fpr_target, 0.0, 1.0)
    if no_dp or epsilon is None:
        return {"L_v1_1": round(L, 6), "humphries_bound": None, "pes_v1_1": None}
    bound = humphries_bound(epsilon, delta)
    pes = (L / bound) if bound > 0 else None
    return {
        "L_v1_1": round(L, 6),
        "humphries_bound": round(bound, 6),
        "pes_v1_1": round(pes, 6) if pes is not None else None,
    }


def _last_round_mia(d: dict) -> dict:
    """Il round finale di per_round contiene anche i campi *composed_*/
    tpr_at_fpr_* (accumulati multi-round) — vedi run_lira() in
    scripts/run_experiments.py. Fallback {} se per_round è vuoto/assente
    (es. JSON di tipo diverso, controllo centralizzato)."""
    pr = d.get("per_round", {})
    if not pr:
        return {}
    last_key = max(pr.keys(), key=lambda k: int(k))
    return pr[last_key].get("mia", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpr-target", type=float, default=0.01,
                     help="FPR fisso per PES v1.1 (default 0.01, deve essere uno di "
                          "_TPR_AT_FPR_TARGETS in run_experiments.py: 0.001/0.01/0.05)")
    ap.add_argument("--include-diagnostic", action="store_true",
                     help="includi anche le cartelle diagnostiche/calibrazione (prefisso _)")
    ap.add_argument("--exclude-methodology-variants", action="store_true",
                     help="escludi entity-split-sweep* (default: incluso — vedi nota sotto)")
    args = ap.parse_args()

    # Fix (2026-09-04, Sprint 10zz+44 — trovato da un run reale dell'utente:
    # entity-split-sweep1 spariva silenziosamente dall'output). discover_groups()
    # esclude di default le "methodology variant" (entity-split-sweep1) per
    # non conflarle in una media aggregata con nodp-sweep1 sotto lo stesso
    # gruppo statistico (fix task #67/#70, vedi check_significance.py) — ma
    # QUESTO script non calcola nessuna media/CI cross-file: ogni riga
    # stampata è un singolo esperimento, etichettato col nome della sua
    # cartella (non con la label di gruppo), quindi il rischio di conflazione
    # non si applica qui. Escludere di default avrebbe fatto perdere in
    # silenzio 5 righe PES legittime (i 5 seed di entity-split-sweep1) senza
    # alcun beneficio — per questo script il default è invertito rispetto a
    # check_significance.py: incluse a meno di --exclude-methodology-variants.
    groups = discover_groups(
        include_diagnostic=args.include_diagnostic,
        include_methodology_variants=not args.exclude_methodology_variants,
    )

    print(f"{'gruppo':<32} {'n':>3} {'mean_lira_auc':>14} {'eps':>6} "
          f"{'PES_v1':>8} {'TPR@FPR='+str(args.fpr_target):>14} {'PES_v1.1':>10}")
    print("-" * 100)

    for label, files in sorted(groups.items()):
        for f in files:
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            cfg = d.get("config", {})
            summ = d.get("summary", {})
            epsilon = cfg.get("epsilon")
            delta = cfg.get("delta", 1e-5)
            no_dp = cfg.get("no_dp", epsilon is None)
            auc = summ.get("mean_lira_auc_roc")

            v1 = pes_v1(auc, epsilon, no_dp)

            last_mia = _last_round_mia(d)
            # Fix (Sprint 10zz+41, 2026-09-04, task #66): prima di questo fix,
            # run_lira() scriveva il TPR@low-FPR del composto multi-round SENZA
            # prefisso in composed_output, che poi sovrascriveva silenziosamente
            # (via .update() in src/plugins/attacks/lira.py) il tpr_at_fpr_* del
            # SOLO ultimo round — quindi la chiave bare letta qui, in ogni JSON
            # storico prodotto dopo Sprint 10pp (2026-08-28), era già di fatto il
            # valore composto (da qui il commento pre-esistente su
            # _last_round_mia che lo descriveva correttamente come "accumulati
            # multi-round"). Il fix in run_lira() ora scrive "composed_tpr_at_fpr_*"
            # (chiave dedicata, coerente con composed_lira_advantage/_confusion) e
            # lascia il tpr_at_fpr_* bare come il vero valore del solo ultimo
            # round. Per NON invalidare silenziosamente la tabella PES v1.1 già
            # pubblicata in docs/PrivacyExposureScore_v1.md (calcolata sul valore
            # composto, non sul round isolato — scelta comunque più sensata per
            # un "exposure score": l'evidenza cumulativa che un attaccante reale
            # avrebbe a fine training), si legge qui esplicitamente la chiave
            # composta, con fallback al bare per compatibilità con eventuali
            # JSON futuri senza LiRA composto attivo.
            tpr_key = f"composed_tpr_at_fpr_{args.fpr_target}"
            tpr_val = last_mia.get(tpr_key, last_mia.get(f"tpr_at_fpr_{args.fpr_target}"))
            v11 = pes_v1_1(tpr_val, args.fpr_target, epsilon, delta, no_dp)

            # cfg["epsilon"] resta popolato nel JSON anche per run no-DP
            # (valore di default/leftover mai applicato) — mostrare "—" per
            # non far credere che un ε sia stato davvero usato.
            eps_str = "—" if no_dp else (f"{epsilon:.2f}" if epsilon is not None else "—")
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            tpr_str = f"{tpr_val:.4f}" if tpr_val is not None else "N/A"
            pes1_str = f"{v1['pes_v1']:.4f}" if v1['pes_v1'] is not None else "N/A"
            pes11_str = f"{v11['pes_v1_1']:.4f}" if v11['pes_v1_1'] is not None else "N/A"

            print(f"{os.path.basename(os.path.dirname(f)) or '.':<32} "
                  f"{'':>3} {auc_str:>14} {eps_str:>6} {pes1_str:>8} {tpr_str:>14} {pes11_str:>10} "
                  f"  ({os.path.basename(f)})")


if __name__ == "__main__":
    main()
