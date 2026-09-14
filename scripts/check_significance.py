#!/usr/bin/env python3
"""Bootstrap CI sulla media di mean_lira_auc_roc tra seed, per gruppo sweep.

Risponde alla domanda: un AUC medio ~0.5 osservato su N=5 seed e' davvero
statisticamente indistinguibile dal caso (0.5), o e' solo un punto stimato
senza intervallo di confidenza intorno? Nessuna dipendenza da scipy (non
installato in questo sandbox e non in requirements.txt) - bootstrap con solo
`random`/`statistics` di libreria standard.

RISCRITTO 2026-08-27: la versione precedente aveva un GROUPS dict con
un'associazione FISSA "nome-cartella -> epsilon" (es. "dp-sweep2" = eps=0.1)
basata sull'ordine in cui gli sweep erano stati lanciati manualmente in
quel momento. Le sweep-dir (dp-sweep1/2/3, central-sweep1/2, ecc.) sono
numerate automaticamente dal Makefile in base a quante cartelle con quel
prefisso esistono gia' al momento del lancio (stesso meccanismo del bug
reale trovato e corretto in README Sprint 10r) - NON portano l'epsilon nel
nome. Con la nuova campagna post-fix (scripts/run_multiseed_consolidation.sh,
riscritto 2026-08-26) l'ordine di lancio e' diverso da quello che aveva
originato questa mappa (es. dp-sweep2 sara' ora eps=0.5, non eps=0.1) - usare
la vecchia mappa avrebbe silenziosamente etichettato i risultati con
l'epsilon sbagliato. Fix: ogni JSON porta gia' dp_mode/epsilon nel proprio
config - li si legge direttamente da li', raggruppando dinamicamente,
invece di fidarsi del nome della cartella.
"""
import json
import glob
import os
import random
import statistics
import sys
from collections import defaultdict
from math import comb
from typing import Any

random.seed(0)  # riproducibilita' del bootstrap stesso (non del training)

EXPERIMENTS_GLOB = "experiments/*/experiment_*.json"

# Wilcoxon (roadmap #5, Sprint 10pp 2026-08-28) — richiesto esplicitamente da
# un revisore in aggiunta al bootstrap CI gia' presente. scipy non e'
# installato in questo sandbox ne' in requirements.txt (stessa ragione per
# cui il bootstrap sopra e' stdlib-only) - va verificato sulla macchina reale
# dell'utente. Se assente, fallback su un test dei segni (sign test) exact-
# binomiale, meno potente ma pure-Python, gia' anticipato come alternativa
# accettabile in docs/TestRoadmap_DSN2027.md #5.
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _sign_test(diffs: list[float]) -> float | None:
    """Test dei segni esatto (binomiale), due code, contro l'ipotesi nulla
    che i valori sopra/sotto lo zero siano equiprobabili (p=0.5 ciascuno).

    Fallback quando scipy non e' disponibile. Meno potente del test dei
    ranghi con segno di Wilcoxon (ignora l'ampiezza delle differenze, solo
    il segno) - con n=5 seed, anche nel caso piu' estremo (5/5 differenze
    con lo stesso segno) il p-value minimo raggiungibile e' 2*(1/32)=0.0625,
    quindi il test non puo' MAI risultare significativo ad alpha=0.05 con
    questa numerosita' campionaria, qualunque sia il dato osservato - un
    limite noto dei test non parametrici a numerosita' molto piccola, non un
    difetto di questa implementazione. Va riportato onestamente, non nascosto.

    Returns:
        p-value a due code, o None se tutte le differenze sono zero
        (nessuna informazione sul segno).
    """
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return None
    k = sum(1 for d in nonzero if d > 0)

    def _binom_cdf_le(k: int, n: int) -> float:
        return sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)

    p_le = _binom_cdf_le(k, n)
    p_ge = _binom_cdf_le(n - k, n)
    return min(1.0, 2 * min(p_le, p_ge))


def significance_test(values: list[float], popmean: float = 0.5) -> tuple[float | None, str]:
    """
    Testa se `values` (tipicamente gli AUC medi dei 5 seed di un gruppo) sono
    significativamente diversi da `popmean` (0.5 = livello del caso).

    Usa scipy.stats.wilcoxon (test dei ranghi con segno) se disponibile —
    il test propriamente detto "Wilcoxon" richiesto dal revisore — altrimenti
    il sign test sopra come fallback dichiarato, MAI silenzioso: il metodo
    usato e' sempre riportato nel valore di ritorno, cosi' chi legge l'output
    sa esattamente quale test ha prodotto quel p-value.

    Returns:
        (p_value o None, nome del metodo usato: "wilcoxon" | "sign_test" | "n/a")
    """
    diffs = [v - popmean for v in values]
    if not diffs:
        return None, "n/a"
    if _SCIPY_AVAILABLE:
        try:
            _, p = _scipy_wilcoxon(diffs)
            return float(p), "wilcoxon"
        except ValueError:
            # scipy solleva ValueError se tutte le differenze sono zero
            # (nessuna informazione sul segno per il test dei ranghi).
            return None, "wilcoxon"
    return _sign_test(diffs), "sign_test"



# Fix (2026-09-04, Sprint 10zz+42, task #67/#70 — trovato durante un audit del
# codice, non da un run fallito): entity-split-sweep1 (task #10/#38, split
# train/holdout ENTITY-AWARE invece che random) ha epsilon/dp_mode/no_dp
# IDENTICI a nodp-sweep1 nel proprio config JSON (nessun campo distingue la
# metodologia di split — verificato: il config non salva quale split è stato
# usato) — quindi, prima di questo fix, `discover_groups()` la raggruppava
# silenziosamente insieme a nodp-sweep1 sotto "no-DP baseline" (n=10 invece di
# n=5), esattamente la stessa classe di conflazione silenziosa già trovata e
# corretta una volta in questo stesso file (mapping cartella→epsilon,
# 2026-08-27) e nel Makefile (Sprint 10r) — solo non ancora chiusa per QUESTO
# caso specifico. `docs/DSN2027_Positioning.md` dichiara esplicitamente che
# entity-split-sweep1 è "un esperimento di robustezza, non un sostituto... non
# direttamente comparabile alla campagna 5-seed×8-config" — pooling coi due
# gruppi non è quindi solo un accidente tecnico, è metodologicamente
# scorretto. A differenza del fix del 2026-08-27 (leggere dal config invece
# che dal nome cartella), qui il nome cartella è l'UNICO segnale disponibile
# (il config non registra la metodologia di split) — esclusione esplicita e
# intenzionale, non un ritorno al bug originale.
_METHODOLOGY_VARIANT_PREFIXES = ("entity-split",)

# Fix (2026-09-14, Sprint 10zz+65 — trovato da una review indipendente,
# eseguendo davvero check_significance.py e confrontando l'output con
# docs/MetricsReference_DSN2027.md invece di fidarsi del testo): le cartelle
# `nvflare-*` (nvflare-central-seed<N>/, nvflare-local-seed<N>/,
# nvflare-stepb-*) sono un deployment DIVERSO (container reali multi-sito,
# non la simulazione single-process) — per esplicita scelta di progetto
# (docs/TestRoadmap_DSN2027.md: "questi run NVFLARE sono una validazione
# supplementare... non un sostituto") vanno riportati SEPARATAMENTE dalla
# campagna principale, non mediati insieme ad essa. Prima di questo fix
# `discover_groups()` non lo sapeva: dedupava per (dp_mode, epsilon, no_dp,
# seed) SENZA distinguere la provenienza, quindi un file NVFLARE con lo
# stesso (dp_mode, epsilon, seed) di un file single-process — e un nome
# lessicograficamente più recente — vinceva silenziosamente la dedup e
# sostituiva il dato single-process nella tabella della campagna. Impatto
# reale, verificato: (a) `nvflare-stepb-dpfedavg-eps1/` (un tentativo di
# validazione dp-fedavg su NVFLARE mai completato, README Sprint 10zz+55 —
# "seed 123 in corso, altri 3 seed + central ancora da fare") contamina la
# cella dp-fedavg/eps=1.0 (mean 0.5000→0.4998, p=0.8125→0.3125 — entrambi
# non significativi, la conclusione qualitativa non cambia, ma il numero
# esatto sì); (b) le 5 `nvflare-local-seed*/` (task #95, completate
# 2026-09-13) hanno silenziosamente sostituito TUTTI e 5 i punti dati
# single-process nella cella local/eps=1.0. Escluse di default, stesso
# pattern di include_diagnostic/include_methodology_variants sopra.
# include_nvflare=True le reintegra per chi vuole analizzare la campagna
# NVFLARE stessa (separatamente, non mescolata).
_NVFLARE_PREFIX = "nvflare-"


def discover_groups(
    pattern: str = EXPERIMENTS_GLOB,
    include_diagnostic: bool = False,
    include_methodology_variants: bool = False,
    include_nvflare: bool = False,
) -> dict[str, list[str]]:
    """Raggruppa i file per (dp_mode, epsilon) letti dal config di ognuno,
    non dal nome della cartella che li contiene.

    Fix (2026-08-28, Sprint 10pp — trovato eseguendo davvero lo script, non
    solo py_compile, contro l'`experiments/` reale durante l'aggiunta del
    test di Wilcoxon): il glob di default scansiona OGNI file
    experiment_*.json in QUALUNQUE sottocartella di experiments/, incluse le
    cartelle diagnostiche/di calibrazione (`_calibration_*`, `_diag_*`,
    `_verify_*`, `_archive_*`) — non solo le sweep-dir della campagna vera
    (`dp-sweep1`, `nodp-sweep1`, ecc.). Prima dell'aggiunta dei 5 file di
    calibrazione LiRA (Sprint 10ee/10jj/10kk/10mm/10nn) questo restava quasi
    innocuo (poche cartelle diagnostiche, quasi tutte con `no_dp=True` come
    la maggioranza dei nodp-sweep reali) — ma ora "no-DP baseline" raggruppa
    silenziosamente 20 file eterogenei: i 5 seed reali di `nodp-sweep1` (3
    siti, 10 round) insieme a 5+ run di calibrazione a 1 solo sito/3 round/
    architetture diverse, mai pensati per essere comparabili o mediati
    insieme — esattamente lo stesso tipo di conflazione silenziosa già
    trovato e corretto una volta in questo script (mapping cartella→epsilon,
    2026-08-27) e nel Makefile (Sprint 10r). **Fix**: le cartelle
    diagnostiche/di calibrazione seguono TUTTE la stessa convenzione di
    naming già in uso in questo progetto (prefisso `_`, mai usato dalle
    sweep-dir reali generate da `scripts/run_multiseed_consolidation.sh`) —
    escluse di default. `include_diagnostic=True` le reintegra esplicitamente
    per chi vuole comunque ispezionarle (es. per rileggere i risultati di
    calibrazione stessi, non per la significatività della campagna).

    Fix (2026-09-04, Sprint 10zz+42): stesso principio applicato a
    `entity-split-sweep1` (split train/holdout entity-aware, non random) — il
    suo config JSON è indistinguibile da `nodp-sweep1` (stesso
    dp_mode/epsilon/no_dp, nessun campo registra la metodologia di split), ma
    non è metodologicamente comparabile (vedi `_METHODOLOGY_VARIANT_PREFIXES`
    sopra). Esclusa di default per prefisso cartella — l'unico segnale
    disponibile, a differenza del fix del 2026-08-27 sopra (che leggeva dal
    config proprio per NON fidarsi del nome cartella): qui il config non ha
    l'informazione necessaria, quindi il nome cartella è usato di proposito,
    non per pigrizia. `include_methodology_variants=True` la reintegra.

    Fix (2026-09-08, Sprint 10zz+46 — trovato al completamento del task #52,
    non da un run fallito): il task #52 ha rilanciato OGNI config della
    campagna con gli STESSI 5 seed (42/123/456/789/1234) già usati nei run
    precedenti, per aggiungere i campi Advantage/Confusion/TPR@low-FPR
    corretto/Sablayrolles/log-LiRA mai salvati la prima volta — non per
    ottenere nuova potenza statistica. Verificato sui dati reali: per
    dp-fedavg/local/no-DP, il rerun produce AUC IDENTICO bit-per-bit al run
    originale per ogni singolo seed (piena determinismo, stesso seed →
    stesso risultato) — 5 osservazioni indipendenti contate due volte
    ciascuna, non 10 osservazioni indipendenti. Per `central` la storia è
    diversa e più seria: central-sweep1/2 (2026-08-29) precedono il fix di
    sensibilità DP pesata in `privatize_aggregate` (task #26, poi verificato
    di nuovo come task #36) — central-sweep3/4 (2026-09-02/03, POST-fix)
    danno valori AUC diversi per lo stesso seed (es. seed=456 ε=1.0:
    0.5009 pre-fix vs 0.4992 post-fix) — quindi central-sweep1/2 non sono
    solo duplicati ridondanti, sono dati PRE-FIX che non andrebbero mediati
    insieme a quelli corretti. central-sweep6/7 (task #52) sono di nuovo
    identici bit-per-bit a central-sweep3/4 (nessun ulteriore fix nel
    frattempo). Prima di questo fix, `discover_groups()` restituiva n=10
    (dp-fedavg/local/no-DP) o n=15 (central, mischiando pre-fix e post-fix)
    invece dei veri n=5 seed indipendenti — gonfiando artificialmente il
    campione e, per central, contaminando la media con dati noti come
    scorretti. **Fix**: deduplica per (dp_mode, epsilon, no_dp, seed),
    tenendo SOLO il file più recente per ogni seed (i nomi file
    `experiment_YYYYMMDD_HHMMSS.json` ordinano cronologicamente in modo
    lessicografico) — elimina sia i duplicati ridondanti sia i dati
    pre-fix superati, sempre a favore della versione più recente/completa.
    Chiude anche task #36 (mai chiuso esplicitamente: la sensibilità
    corretta è già in central-sweep3/4/6/7, ora correttamente l'unica
    versione usata nelle statistiche aggregate).
    """
    groups: dict[str, list[str]] = defaultdict(list)
    # (label, seed) -> file più recente visto finora per quella combinazione.
    latest_by_seed: dict[tuple[str, Any], str] = {}
    for f in sorted(glob.glob(pattern)):
        sweep_dir_name = os.path.basename(os.path.dirname(f))
        if not include_diagnostic and sweep_dir_name.startswith("_"):
            continue
        if not include_methodology_variants and sweep_dir_name.startswith(
            _METHODOLOGY_VARIANT_PREFIXES
        ):
            continue
        if not include_nvflare and sweep_dir_name.startswith(_NVFLARE_PREFIX):
            continue
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        cfg = d.get("config", {})
        no_dp = cfg.get("no_dp", cfg.get("epsilon") is None)
        dp_mode = cfg.get("dp_mode", "dp-fedavg")
        eps = cfg.get("epsilon")
        seed = cfg.get("seed")
        # Normalizza il tipo di `seed` per la chiave di dedup: un rerun i cui
        # config.yaml/JSON salvano lo stesso seed logico con tipi diversi
        # (es. 42 int vs "42" str, capita se qualcuno quota il valore in YAML
        # o se una versione futura dello script cambia serializzazione) non
        # deve essere trattato come un seed DIVERSO — altrimenti la dedup del
        # 2026-09-08 (task #52/#73) sopra non si applicherebbe silenziosamente
        # a quel run, reintroducendo la stessa conflazione/n-inflation che
        # quel fix doveva chiudere. `None` (seed assente/non impostato) resta
        # un valore a sé, non normalizzato a stringa.
        seed_key = str(seed) if seed is not None else None
        if no_dp or eps is None:
            label = "no-DP baseline"
        else:
            label = f"{dp_mode}, eps={eps}"
        key = (label, seed_key)
        # `sorted(glob.glob(...))` non garantisce ordine cronologico tra
        # sweep-dir diverse (solo alfabetico per path) — confronto esplicito
        # sul basename (timestamp) invece di assumere che l'ultimo visto sia
        # il più recente.
        prev = latest_by_seed.get(key)
        if prev is None or os.path.basename(f) > os.path.basename(prev):
            latest_by_seed[key] = f

    for (label, _seed), f in latest_by_seed.items():
        groups[label].append(f)
    return groups


# Ri-analisi TPR@low-FPR (2026-09-11, su richiesta esplicita dell'utente dopo
# aver verificato che il paper cita Carlini et al. 2022 — che argomenta
# esplicitamente che le metriche average-case come AUC-ROC nascondono la
# capacità di un attacco a bassi FPR — mentre la statistica primaria
# effettivamente testata (questo script) era solo mean_lira_auc_roc).
#
# Metrica usata: composed_tpr_at_fpr_{0.001,0.01,0.05} — il valore LiRA
# "composto" (evidenza sommata su tutti i round, non la media dei singoli
# round), stessa metrica "headline" già usata per composed_lira_auc_roc nel
# paper (§7/§8). Posizione nel JSON: per_round[<ultimo round>]["mia"]
# (composed_output viene mergiato lì da src/plugins/attacks/lira.py — vedi
# il commento sul fix Sprint 10zz+41 in run_experiments.py per la ragione
# del prefisso "composed_").
#
# IMPORTANTE — livello di caso per TPR@FPR fisso: a differenza di AUC-ROC
# (caso = 0.5 sempre), sotto un attacco non informativo la ROC è la
# diagonale, quindi TPR@FPR=t == t al caso — es. TPR@1%FPR=0.01, non 0.5.
# bootstrap_ci()/significance_test() sotto sono già generiche sul parametro
# `popmean` per questo motivo esatto: qui viene passato il target FPR stesso,
# non 0.5.
TPR_FPR_TARGETS = ("0.001", "0.01", "0.05")


def extract_composed_tpr(d: dict, fpr_target: str) -> float | None:
    """Legge composed_tpr_at_fpr_<fpr_target> dall'ultimo round del file.

    Ritorna None se il file non ha ancora questo campo (run precedenti al
    Sprint 10pp/2026-08-28, o run diagnostici a 1 solo round senza LiRA
    composto) — silenziosamente escluso dal gruppo, stessa convenzione già
    usata per mean_lira_auc_roc in main().
    """
    per_round = d.get("per_round", {})
    if not per_round:
        return None
    last_round = max(per_round.keys(), key=int)
    mia = per_round[last_round].get("mia", {})
    return mia.get(f"composed_tpr_at_fpr_{fpr_target}")


def bootstrap_ci(values, n_resamples=10000, alpha=0.05):
    n = len(values)
    if n < 2:
        return None
    means = []
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return means[lo_idx], means[hi_idx]


def main():
    groups = discover_groups()
    method_label = "wilcoxon" if _SCIPY_AVAILABLE else "sign_test (scipy assente)"
    print(
        f"{'gruppo (letto da config, non dal nome cartella)':<42} {'n':>3} {'mean':>8} "
        f"{'std':>8} {'95% CI':>20} {'contiene 0.5?':>14} {'p (' + method_label + ')':>26}"
    )
    print("-" * 140)
    if not _SCIPY_AVAILABLE:
        print(
            "NOTA: scipy non disponibile — uso il sign test come fallback (meno potente, "
            "vedi docstring significance_test()). Verificare disponibilità scipy sulla "
            "macchina reale per il test dei ranghi con segno di Wilcoxon vero e proprio."
        )
        print("-" * 140)
    for label, files in sorted(groups.items()):
        aucs = []
        for f in files:
            d = json.load(open(f))
            s = d.get("summary", {})
            v = s.get("mean_lira_auc_roc")
            if v is not None:
                aucs.append(v)
        if not aucs:
            print(f"{label:<42} nessun dato")
            continue
        mean = statistics.mean(aucs)
        std = statistics.stdev(aucs) if len(aucs) > 1 else float("nan")
        ci = bootstrap_ci(aucs)
        if ci is None:
            ci_str = "n<2, N/A"
            contains = "N/A"
        else:
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            contains = "SI" if ci[0] <= 0.5 <= ci[1] else "NO"
        p_value, method = significance_test(aucs)
        if p_value is None:
            p_str = "N/A (tutti uguali a 0.5)" if len(aucs) > 1 else "N/A (n<2)"
        else:
            p_str = f"{p_value:.4f} [{method}]"
        print(
            f"{label:<42} {len(aucs):>3} {mean:>8.4f} {std:>8.4f} {ci_str:>20} "
            f"{contains:>14} {p_str:>26}"
        )
    if not _SCIPY_AVAILABLE:
        print(
            "\nNOTA sul sign test: con n=5 seed per gruppo, il p-value minimo raggiungibile "
            "e' 2*(1/32)=0.0625 (5/5 differenze con lo stesso segno) — il test non puo' MAI "
            "risultare significativo ad alpha=0.05 con questa numerosita', qualunque sia il "
            "dato osservato. Non e' un difetto dell'implementazione: e' un limite noto dei "
            "test non parametrici a campioni molto piccoli, e va riportato come tale nel "
            "paper se questi p-value vengono citati (il bootstrap CI sopra resta il test "
            "primario per questa numerosita' campionaria)."
        )

    # ── Ri-analisi TPR@low-FPR (2026-09-11) ─────────────────────────────────
    for fpr_target in TPR_FPR_TARGETS:
        print()
        print("=" * 140)
        print(
            f"TPR @ FPR={fpr_target} (composed_lira, evidenza sommata su tutti i round) "
            f"— livello di caso = {fpr_target} (ROC diagonale, NON 0.5)"
        )
        print("=" * 140)
        print(
            f"{'gruppo (letto da config, non dal nome cartella)':<42} {'n':>3} {'mean':>8} "
            f"{'std':>8} {'95% CI':>20} {'contiene ' + fpr_target + '?':>17} "
            f"{'p (' + method_label + ')':>26}"
        )
        print("-" * 140)
        for label, files in sorted(groups.items()):
            tprs = []
            for f in files:
                d = json.load(open(f))
                v = extract_composed_tpr(d, fpr_target)
                if v is not None:
                    tprs.append(v)
            if not tprs:
                print(f"{label:<42} nessun dato (campo assente in questi file)")
                continue
            mean = statistics.mean(tprs)
            std = statistics.stdev(tprs) if len(tprs) > 1 else float("nan")
            ci = bootstrap_ci(tprs)
            popmean = float(fpr_target)
            if ci is None:
                ci_str = "n<2, N/A"
                contains = "N/A"
            else:
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                contains = "SI" if ci[0] <= popmean <= ci[1] else "NO"
            p_value, method = significance_test(tprs, popmean=popmean)
            if p_value is None:
                p_str = f"N/A (tutti uguali a {fpr_target})" if len(tprs) > 1 else "N/A (n<2)"
            else:
                p_str = f"{p_value:.4f} [{method}]"
            print(
                f"{label:<42} {len(tprs):>3} {mean:>8.4f} {std:>8.4f} {ci_str:>20} "
                f"{contains:>17} {p_str:>26}"
            )
    print()
    print(
        "NOTA metodologica: un gruppo con TPR@FPR=t significativamente > t (CI sopra la "
        "diagonale, p<0.05) indicherebbe un attacco capace di isolare membri con confidenza "
        "a quel FPR specifico anche quando l'AUC-ROC medio resta ~0.5 — esattamente il tipo "
        "di segnale che Carlini et al. 2022 argomenta essere nascosto da una metrica "
        "average-case come l'AUC. Un CI che include t (o lo attraversa) e p non significativo "
        "sono coerenti con la stessa conclusione null-leakage già riportata per l'AUC."
    )


if __name__ == "__main__":
    main()
