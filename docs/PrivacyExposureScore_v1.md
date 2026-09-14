# Privacy Exposure Score (PES) — v1

Status: **draft v1, in use for internal ranking only — not yet peer-reviewed or paper-ready.**
Owner task: #63 (v1, this document) / #64 (v2/full, blocked on Gradient Inversion).

> **⚠ Superseded (2026-08-26) — the worked-example numbers below are invalidated, the formula is
> not.** Every LiRA AUC value in this document (0.7430, 0.8118, and the PES_v1 scores 0.243/0.567
> computed from them) predates a six-fix investigation into the LiRA implementation (README
> Sprint 10x through 10cc) that found and corrected a chain of real bugs, the last of which was
> structural (the shadow-model sampling universe could never include non-members, making real
> IN-calibration for non-members architecturally impossible). With the fix chain applied, a
> four-way convergent re-verification shows **no LiRA-detectable leakage at any tested ε** (README
> Sprint 10dd) — the "first nonzero PES_v1" and "most damning PES_v1" readings below were computed
> from AUC numbers that are now known to be substantially an artifact, not real leakage. **The PES
> formula itself (`L(AUC)`/`strength(ε)`/`U_cost` below) is unaffected and remains valid** — it is
> a generic function of whatever AUC/ε pair you feed it. **Aggiornamento 2026-09-09**: la campagna
> a 5-seed × 10-config (task #52) è ora completa e questo documento è stato aggiornato con i
> numeri corretti (vedi tabella più sotto) — le cifre invalidate qui sopra restano solo come
> storico di cosa è stato superato, non vanno più cercate altrove nel documento.

> **Addendum (2026-08-31, Fase 8) — the ε fed into `strength(ε)` is a nominal noise-calibration
> parameter, not a proven formal (ε,δ)-DP guarantee for the training procedure actually used.**
> Two caveats, detailed in `docs/DSN2027_Positioning.md` limitation #9: (a) `GradientManager`
> implements weight perturbation (noise on the post-training weight vector once per round), not
> DP-SGD — the Gaussian mechanism's formal guarantee holds exactly only for `epochs=1`, not the
> `epochs=50` used in the main campaign; (b) under `dp_mode="central"`, noise calibration used to
> assume a uniform (unweighted) mean across participants (`σ/n_participants`) — corrected to use the
> true weighted-mean sensitivity, `max_grad_norm × max_i(n_i/N)`, which is always ≥ the old value
> given the real sites' uneven sizes (Office 1 ≪ Caltech/JPL). Neither caveat changes the PES_v1
> *formula* (still a generic function of whatever AUC/ε pair is supplied) nor the paper's central
> empirical finding (LiRA AUC ≈ 0.5 regardless of DP configuration) — but it does mean the ε value
> plugged into `strength(ε)` for already-published Central DP configurations should be described in
> the paper as a nominal calibration target, not an exact, independently-verified formal guarantee.

## Portability beyond this project (added 2026-07-24)

PES v1's inputs are deliberately generic, not ChargeShield-FL-specific: any attack AUC-ROC
(∈ [0.5, 1.0], from *any* membership-inference attack, not just LiRA) and any nominal DP budget ε
(from *any* DP mechanism, not just this project's three placements). Nothing in the formula below
references EV charging data, ACN-Data, or this project's FL pipeline. A researcher benchmarking a
different FL/DP setup — a different dataset, a different attack, a different aggregation scheme —
could compute PES on their own AUC/ε pairs with no adaptation. That portability is what makes it a
candidate community metric rather than a one-off number for this paper's results table, alongside
the benchmark/framework positioning in `docs/DSN2027_Positioning.md`. What is *not* yet portable:
the "v1" scope excludes gradient-reconstruction and attribute-recovery terms (see below) because
this project doesn't yet measure them — a full v2 metric, once Gradient Inversion lands, would
need those terms validated on more than one dataset before claiming general applicability.

## Why a new metric

"Attack AUC-ROC" alone tells you whether an attack beats chance, but it does not tell you
whether a *specific DP configuration's own privacy claim* is being honoured. A configuration
running ε=0.1 is nominally claiming very strong privacy. If LiRA still succeeds at ε=0.1, that
is a much more damning result than the same AUC at ε=5.0 — and a single AUC column in the sweep
table does not surface that difference. PES is designed to make that gap the headline number.

**PES v1 combines only the components the project already produces for real**: LiRA AUC-ROC,
the nominal DP budget ε, and the FL utility cost of the defence. It deliberately excludes the
two terms from the original proposal (gradient-reconstruction quality, recovered-attribute
count) because those require a working Gradient Inversion module, which does not exist in the
project yet (see Task #64, and Task #57/central-DP work in progress). Publishing a "full" metric
with a term that is currently undefined would misrepresent what the project can measure today.

## Definition

For a given experiment configuration (attack = LiRA, DP mode, ε):

```
L(AUC)        = clip( 2 * max(0, AUC_LiRA - 0.5), 0, 1 )        # residual leakage, 0 = chance, 1 = perfect attack
strength(ε)   = 1 / (1 + ε)      for a DP configuration
                0                for the no-DP baseline (no privacy claim to break)
U_cost        = (mean_loss_DP - mean_loss_noDP) / mean_loss_noDP   # relative utility degradation

PES_v1        = L(AUC) * strength(ε)
```

`PES_v1` is high only when **both** conditions hold at once: the attack still beats chance
(`L > 0`) *and* the configuration is nominally claiming strong privacy (`ε` small, `strength`
close to 1). A configuration that blocks the attack (`L = 0`) scores 0 regardless of ε. A
configuration with weak nominal privacy (large ε) that still leaks scores low too — because a
large-ε defence never promised much, so leakage there isn't surprising. High PES is reserved for
the specific failure mode this paper is built around: *"the epsilon guarantee is misleading."*

`U_cost` is reported alongside PES, not folded into it multiplicatively, for a specific reason:
in DP-FedAvg, noise is large enough at low ε to plausibly destroy attackable signal *and* model
utility at the same time. If `U_cost` is very large exactly where `L` collapses to near-zero,
that is a confound — the attack may be failing because there is nothing left to attack, not
because the defence is meaningfully protecting membership information. Folding U_cost into a
single multiplied scalar would hide that confound; reporting it as a companion axis keeps it
visible; see caveat below.

## Computed on real experiment data (2026-07-22, single seed=42, DP-FedAvg only)

> **Nota sul vintage dei dati (aggiunta 2026-07-25, file archiviati lo stesso giorno).** Questi run (originariamente `experiments/experiment_20260722_*.json`) usano `n_shadow=8`, lo stesso valore instabile che ha portato all'archiviazione di `nodp-sweep1`/`dp-sweep1`/`dp-sweep2`. Sono stati spostati (non cancellati) in `experiments/_archive_invalid_n_shadow8/loose_experiment_jsons_20260722/` per coerenza con quel trattamento — non sono quindi più presenti in `experiments/` al livello superiore, ma restano consultabili lì per riferimento storico. A differenza della sezione Central DP più sotto, che usa correttamente n_shadow=16, questa tabella non segnalava la differenza. In pratica l'impatto sul PES_v1 qui riportato è basso: tutti i valori sono 0 perché LiRA resta sotto 0.5 in ogni configurazione, quindi anche con n_shadow=16 il verdetto qualitativo ("DP-FedAvg sopprime LiRA nel range testato") difficilmente cambierebbe — ma questi numeri non vanno citati come dato finale nel paper senza prima essere rieseguiti a n_shadow=16, coerentemente con il resto del consolidamento multi-seed in corso (vedi README Sprint 10n / task #78).

| Config | ε | LiRA mean AUC | L(AUC) | strength(ε) | U_cost (loss increase vs no-DP) | PES_v1 |
|---|---|---|---|---|---|---|
| no-DP | — | 0.5845 | 0.169 | 0 (n/a) | 0 (baseline) | 0.000 |
| DP-FedAvg | 5.0 | 0.4904 | 0.000 | 0.167 | ×13.0 | 0.000 |
| DP-FedAvg | 2.0 | 0.4892 | 0.000 | 0.333 | ×61.5 | 0.000 |
| DP-FedAvg | 1.0 | 0.4857 | 0.000 | 0.500 | ×122.1 | 0.000 |
| DP-FedAvg | 0.5 | 0.4946 | 0.000 | 0.667 | ×136.7 | 0.000 |
| DP-FedAvg | 0.1 | 0.4987 | 0.000 | 0.909 | ×139.2 | 0.000 |

(`mean_loss` = mean reconstruction loss over the 10-round run, from `per_round[r]["fl"]["mean_loss"]`
in `experiments/experiment_20260722_*.json`; `U_cost` shown as a multiplier of the no-DP mean loss
for readability, since the raw fractional increase is 1200%–13900%.)

### Reading this table honestly

Every DP-FedAvg configuration in this sweep scores `PES_v1 = 0`, because LiRA's mean AUC never
clears 0.5 once any DP-FedAvg noise is added (this matches the "DP-FedAvg suppresses LiRA across
the tested range" slide). That is **not** evidence that the metric has nothing to show — it is
the correct, expected output of a metric designed to flag *false* privacy claims, applied to a
case where the claim happens to hold up. The important open question, already flagged as the
project's next step (Task #66, `make experiment-central-dp EPS=1.0/0.1 N_SHADOW=16`, not yet run
as of 2026-07-24 — corrected here after independent review: `experiments/dp-sweep3/` is **not**
a central-DP run, it is two additional dp-fedavg ε=0.1 seeds, 123/456, confirming the round-8
anomaly was single-seed variance, see Task #57), is whether **Central DP** — which clips
client-side but noises only the aggregate once, leaving the raw per-client update LiRA actually
attacks untouched — produces nonzero, possibly large, `PES_v1` at low ε. That would be the first
genuinely newsworthy PES number, and the reason Central DP is the priority experiment right now
rather than an afterthought. Note for whoever computes PES on that run: `experiment-central-dp`
does not pass `--sweep-dir`, so its JSON lands directly in `experiments/`, not inside
`dp-sweep3/` or any subdirectory — check there (or in `experiments/central-sweep1/` if the
`-sweep` multi-seed variant is used instead).

Also note the `U_cost` column: at ε≤0.5 the mean reconstruction loss is already ~135–139× the
no-DP baseline. That level of degradation is close to a non-functional model. Any future claim
of the form "PES is near zero at ε=0.1, so DP-FedAvg is safe" needs this caveat attached — the
model may simply be too noisy to memorize anything, which is a Pyrrhic privacy win, not a useful
one. This caveat is exactly the kind of thing a single-scalar metric can hide if `U_cost` isn't
kept visible.

## Central DP, ε=1.0 — the first nonzero PES_v1 (real result, 2026-07-24) — ⚠ superseded, see banner at top

`experiments/experiment_20260724_111109.json` (`dp_mode=central`, ε=1.0, 10 rounds, seed=42,
n_shadow=16) is the first completed run of the experiment this document flagged above as "the
priority experiment right now." It is exactly the failure mode PES was built to catch:

| Config | ε | LiRA mean AUC | L(AUC) | strength(ε) | U_cost (loss increase vs no-DP) | PES_v1 |
|---|---|---|---|---|---|---|
| Central DP | 1.0 | **0.7430** | **0.486** | 0.500 | ×38.2 | **0.243** |
| Central DP | 0.1 | **0.8118** | **0.624** | 0.909 | ×102.0 | **0.567** |

(`mean_lira_auc_roc` from `summary`; `L(AUC) = clip(2*max(0, 0.7430224-0.5), 0, 1) = 0.4860`;
`strength(1.0) = 1/(1+1) = 0.5`; `PES_v1 = 0.4860 * 0.5 = 0.2430`. `U_cost`: mean reconstruction
loss over the 10 rounds is 0.10324 vs 0.002706 for the matching no-DP seed=42 baseline
(`experiments/experiment_20260722_185408.json`) — a ×38.2 increase, notably smaller than the
×122–139 seen at the corresponding ε under DP-FedAvg, so this is not the same "noised into
uselessness" confound flagged above.)

This is the first PES_v1 value in the project greater than zero. Read plainly: at ε=1.0, Central
DP's nominal privacy claim (`strength=0.5`, a "moderate" budget) is not honoured — LiRA recovers
membership at 0.743 AUC, well above chance, because Central DP clips client-side but only noises
the aggregate once, leaving the raw per-client update (what LiRA actually attacks) untouched. This
is architecturally expected — see the "Three DP placements" note in the project's methodology —
but this is the first *measured*, multi-round confirmation of it with a real PES number attached,
not just a qualitative "expected little/no suppression" prediction. `privacy_risk` in the same
JSON is independently flagged `"HIGH"` by the existing (non-PES) risk heuristic, corroborating
the PES reading.

## Central DP, ε=0.1 — the most damning PES_v1 to date (real result, 2026-07-24, 14:49) — ⚠ superseded, see banner at top

`experiments/experiment_20260724_144952.json` (`dp_mode=central`, ε=0.1, 10 rounds, seed=42,
n_shadow=16) completed the same day. Contrary to the naive expectation that a *tighter* nominal
budget (ε=0.1 vs ε=1.0) should mean *more* protection, LiRA's mean AUC actually **increased** to
0.8118 (max 0.9356, min 0.7014) — every single round beat the ε=1.0 run's per-round AUC. `L(AUC) =
clip(2*max(0, 0.8117862-0.5), 0, 1) = 0.6236`; `strength(0.1) = 1/(1.1) = 0.9091`; `PES_v1 =
0.6236 * 0.9091 = 0.5669` — more than double the ε=1.0 value, and the single largest PES_v1 number
this project has produced.

Read plainly, this is the sharpest version yet of the paper's central thesis: at ε=0.1 — a budget
that reads, to anyone skimming a methods section, as "very strong differential privacy" — LiRA
recovers membership at 0.81 AUC, i.e. the DP guarantee is not just "somewhat" misleading, it is
*most* misleading exactly where the nominal claim is strongest. This is not a contradiction of the
architectural explanation given for ε=1.0 above (Central DP still only noises the aggregate, never
the raw per-client update LiRA attacks) — if anything it strengthens it: shrinking ε increases the
noise added to the *aggregate*, which should further blur the global model's fine-grained fit, yet
LiRA's signal comes from the per-client raw update, which the added aggregate noise never touches.
`U_cost` is also severe here (×102.0 mean-loss increase vs no-DP) — worth flagging per the caveat
above: at this level of degradation, the argument "the model works fine but is also risky" is
harder to make than at ε=1.0 (×38.2), and a skeptical reviewer could ask whether the model is
close to non-functional. This caveat does not erase the PES finding, but it belongs in the same
sentence as the 0.567 number whenever this result is cited.

**Both central-DP legs of Task #66 are now complete** (single seed=42 each). Before either number
goes into the paper as a headline claim, the project's own established standard (5 seeds for every
DP-FedAvg point in the existing sweep) should be applied here too — a single seed is not yet
sufficient evidence against the possibility that this is seed-42-specific variance, especially
given the round-8 anomaly precedent already found and documented for DP-FedAvg sweeps. Multi-seed
Central DP repeats are the natural next step before this becomes a paper table.

## PES v1.1 — implementata (task #41, Sprint 10zz+13, 2026-09-02)

La sequenza indicata sopra ("implementare TPR@low-FPR per primo, poi PES v1.1 diventa ben
definito") è ora soddisfatta — TPR@low-FPR è nel codice dal Sprint 10pp (2026-08-28). Implementata
in `scripts/compute_pes.py` (nuovo script, retroattivo su qualunque JSON già esistente, riusa
`discover_groups()` da `check_significance.py`):

```
L_v1.1   = clip( TPR@FPR=0.01 − 0.01, 0, 1 )            # vantaggio a UNA soglia fissa e bassa,
                                                          # non mediato su tutte le soglie come L(AUC)
PES_v1.1 = L_v1.1 / humphries_bound(ε, δ)               # frazione del vantaggio massimo
                                                          # formalmente permesso da (ε,δ)-DP
humphries_bound(ε, δ) = (e^ε − 1 + 2δ) / (e^ε + 1)      # Humphries et al. 2020
```

**Scelta metodologica dichiarata**: `L_v1.1` usa la soglia FISSA `FPR=0.01` (coerente con l'operating
point "headline" già citato nei Sprint-log per TPR@low-FPR), non la soglia ottimale
`max(TPR-FPR)` (Youden J — implementata separatamente come `_mia_advantage()` in
`scripts/run_experiments.py`, task #41). Non ancora nella letteratura come "la" definizione di
PES v1.1 — una proposta nuova di questo progetto, da dichiarare come tale nel paper. Per no-DP,
`PES_v1.1 = None` (non 0): non esiste un soffitto ε contro cui normalizzare, diverso da
`strength(ε)=0` di v1, che è invece un valore definito ("nessuna promessa di privacy da violare").

**AGGIORNAMENTO 2026-09-09 (post task #52/#73) — tabella sotto sostituita con i numeri
DEFINITIVI**, ricalcolati con `python3 scripts/compute_pes.py` dopo il completamento della
campagna a 10 configurazioni (task #52) e il fix del bug di pseudo-replicazione in
`discover_groups()` (task #73, di cui `compute_pes.py` beneficia automaticamente essendo
importato da `check_significance.py`). La tabella precedente (2026-09-02) usava solo 8
configurazioni (mancavano central/local ε=0.5) e il gruppo no-DP era ancora conflazionato con
`entity-split-sweep1` (n=10 invece di 5) — entrambi i problemi sono ora risolti: **ogni gruppo
sotto ha esattamente n=5 file** (verificato, un file per seed).

| Sweep | ε | n file | PES_v1 (range) | PES_v1.1 (range, dove disponibile) |
|---|---|---|---|---|
| dp-sweep4 (dp-fedavg) | 1.0 | 5 | 0.0000–0.0003 | 0.0000–0.0053 |
| dp-sweep5 (dp-fedavg) | 0.5 | 5 | 0.0000–0.0012 | 0.0000–0.0066 |
| dp-sweep6 (dp-fedavg) | 0.1 | 5 | 0.0000–0.0024 | 0.0000–0.0009 |
| central-sweep6 | 1.0 | 5 | 0.0000–0.0018 | 0.0000–0.0029 |
| central-sweep5 | 0.5 | 5 | 0.0000–0.0010 | 0.0000–0.0023 |
| central-sweep7 | 0.1 | 5 | 0.0000–0.0036 | 0.0000–0.0309 |
| local-sweep4 | 1.0 | 5 | 0.0000–0.0003 | 0.0000–0.0053 |
| local-sweep3 | 0.5 | 5 | 0.0000–0.0012 | 0.0000–0.0066 |
| local-sweep5 | 0.1 | 5 | 0.0000–0.0024 | 0.0000–0.0009 |
| nodp-sweep2 | — (no-DP) | 5 | 0.0000 (per costruzione) | N/A (nessun ε da normalizzare) |

(`dp-fedavg`/`local` mostrano numeri identici allo stesso ε — atteso e già documentato, README
nota 2026-08-06: le due modalità coincidono in questa simulazione single-process.)

**Lettura onesta**: sia PES_v1 sia PES_v1.1 restano vicinissimi a zero in OGNI configurazione DP
ora testata — tutte e 9 le combinazioni central/dp-fedavg/local × ε∈{1.0,0.5,0.1}, incluse le 2
(central/local ε=0.5) assenti dalla tabella precedente — coerente, con due formulazioni
indipendenti, con il risultato principale già pubblicato (AUC LiRA composito 0.4995–0.5005
ovunque, task #52, confermato con Wilcoxon reale). Nessuna configurazione mostra il "PES alto" che
il metric fu progettato per segnalare (ε nominale piccolo ma leakage reale) — perché,
semplicemente, non c'è leakage reale da segnalare in questa architettura/dataset, con nessuna
delle due normalizzazioni. Questo NON invalida il disegno della metrica (progettata correttamente
per catturare quel caso, se si fosse presentato) — conferma solo, con un secondo strumento più
teoricamente fondato (v1.1 normalizza contro il bound di Humphries et al. 2020, non contro un peso
euristico come v1), la stessa conclusione null-leakage già stabilita.

**Nota sui valori PES_v1 diversi da zero (2026-08-28/29/30, es. 0.0029, 0.0024)**: non sono errori
— corrispondono a run in cui `mean_lira_auc_roc` è marginalmente sopra 0.5 per varianza campionaria
(es. 0.5016, 0.5013), dando un `L(AUC)` piccolo ma non nullo. Con `strength(ε)` anch'esso piccolo
(ε=0.1 → strength=0.909, ε=1.0 → strength=0.5), il prodotto resta comunque ≤0.003 in ogni caso —
un ordine di grandezza sotto qualunque soglia che si potrebbe ragionevolmente chiamare "PES alto"
(i valori realmente alti già documentati sopra, 0.243/0.567, erano pre-fix LiRA e superati).

**Nota di rigore (2026-09-04, task #66, Sprint 10zz+41) — chiarimento su cosa misurava davvero
`tpr_at_fpr_0.01` nella tabella sopra**: un audit del codice ha trovato che `run_lira()` scriveva
il TPR@low-FPR del composto multi-round SENZA prefisso dedicato in `composed_output`, che poi
sovrascriveva silenziosamente (via merge in `src/plugins/attacks/lira.py`) il `tpr_at_fpr_0.01` del
SOLO ultimo round — un dato diverso (evidenza cumulativa su tutti i round, non del round isolato).
Bug live dal Sprint 10pp (2026-08-28), quindi presente in ogni JSON usato per la tabella sopra: la
tabella ha **sempre** riportato il TPR composto, non quello "dell'ultimo round" come la dicitura
letterale suggeriva. **I numeri della tabella restano validi e non vanno ricalcolati** — la
cumulativa multi-round è, se anything, la scelta più sensata per un "exposure score" (l'evidenza
che un attaccante reale avrebbe a fine training, non solo nell'ultimo round preso isolatamente).
Fix applicato: `run_lira()` ora scrive `composed_tpr_at_fpr_*` (chiave dedicata, come gli altri
campi composti), e `scripts/compute_pes.py` legge esplicitamente quella chiave (con fallback al
valore bare per compatibilità) — stesso identico calcolo di prima, ora esplicito invece che
accidentale. Vedi `tests/test_composed_tpr_prefix.py` per il test di regressione.

## Naming

Working name is "Privacy Exposure Score (PES)"; "Operational Leakage Score (OLS)" and "Critical
Infrastructure Privacy Risk (CIPR)" remain open alternatives — naming has no effect on the
formula above and can be decided later (e.g. based on which framing the introduction ends up
using — see `docs/DSN2027_Positioning.md`).

## Theoretical grounding (informal) — added 2026-08-27, in response to external review

A reviewer correctly flagged that PES_v1 is an heuristic combination (`L(AUC) * strength(ε)`)
with no proof that it approximates an optimal or even well-defined theoretical quantity. This
section states plainly what IS and is NOT grounded, rather than asserting a derivation that
doesn't exist.

**What is grounded.** Yeom, Fredrikson, and Jha (2018) — already one of this project's three
benchmark attacks — formally define membership advantage for a fixed attacker/threshold as
`Adv = |Pr[attacker says "member" | is member] − Pr[attacker says "member" | is non-member]|`
and prove that for an ε-differentially-private training mechanism, `Adv ≤ e^ε − 1`. This is the
standard, widely-cited theoretical link between the DP parameter ε and how well *any* membership
attacker can possibly do against an ε-DP mechanism — a real ceiling, not a heuristic.

**Tighter bound, more applicable to this project (added 2026-08-27):** Humphries et al. (2020,
"Differentially Private Learning Does Not Bound Membership Inference") prove a tighter bound that
accounts for δ, not just ε: `Adv ≤ (e^ε − 1 + 2δ) / (e^ε + 1)`. This is the more appropriate
citation for ChargeShield-FL specifically, since the Gaussian Mechanism used throughout this
project gives (ε, δ)-DP, not pure ε-DP (δ = 1e-5 in every experiment, see `config/experiment.yaml`)
— the Yeom bound above ignores δ entirely, while the Humphries bound is defined for exactly the
mechanism class this project actually uses. Both bounds should be cited; Humphries et al. is the
one PES v1.1 (below) should normalize against, not Yeom's.

`L(AUC) = clip(2 * max(0, AUC_LiRA − 0.5), 0, 1)` is designed to play the *same conceptual role*
as `Adv` above: both are 0 at chance-level attacker performance and increase monotonically as the
attacker does better than chance, both live on roughly the same [0, 1] scale. **This document does
not claim `L(AUC)` is mathematically identical to `Adv`** — AUC integrates attacker performance
over every possible decision threshold, while `Adv` (as Yeom define it) is evaluated at one fixed
threshold; establishing the precise relationship between the two (they coincide exactly only under
specific symmetry assumptions on the score distributions) is a real gap, not asserted here as
closed.

**What is NOT grounded, stated honestly.** `strength(ε) = 1/(1+ε)` is a *designed*, not *derived*,
weighting — chosen because it is 1 at ε→0 (strongest nominal privacy) and shrinks toward 0 as ε
grows (weakest nominal privacy, so a large-ε defence "never promised much"), which is the right
qualitative shape, but it is not derived from the Yeom bound or any other formal DP result. The
Yeom bound `e^ε − 1` has a different shape entirely (it is a ceiling that grows without bound as
ε increases, not a weighting factor that decays), so `PES_v1 = L(AUC) * strength(ε)` should not be
read as "the empirical advantage normalized against its theoretical ceiling" — it isn't that,
today.

**Concrete refinement this suggests (PES v1.1 — implemented since Sprint 10zz+13, 2026-09-02;
this "Theoretical grounding" section predates that and is kept as the original proposal
rationale, see the "PES v1.1" section above for the final, computed numbers):** a more tightly grounded
successor metric would normalize the *observed* advantage against the *Humphries (ε,δ) ceiling*
directly — `PES_v1.1 = L(AUC) / ((e^ε − 1 + 2δ) / (e^ε + 1))`, read as "what fraction of the
maximum advantage this (ε,δ)-DP mechanism formally permits did the attacker actually achieve."
This is a specific, falsifiable proposal, not a vague "needs more theory" placeholder — but it
requires `L(AUC)`'s single-threshold behavior to be pinned down first (see "What is NOT grounded"
above), which is exactly what implementing **TPR@fixed-FPR** (`docs/TestRoadmap_DSN2027.md` #4)
would give: TPR at a fixed operating point IS a single-threshold quantity with a direct, provable
correspondence to both bounds above, unlike AUC. Sequencing: implement TPR@low-FPR first, then PES
v1.1 becomes a well-posed, citable refinement rather than another heuristic guess.

**Rényi DP / tighter composition bounds**: the Yeom bound above is stated for pure ε-DP; this
project's Gaussian Mechanism composition is more naturally analyzed under Rényi DP (RDP) or
zero-concentrated DP, which give tighter, ε-and-δ-aware advantage bounds under Gaussian noise
specifically. Substituting a Gaussian-mechanism-specific bound for the generic Yeom ε-DP bound
above is a natural next refinement for v1.1 but is not done here — flagged as a specific,
scoped future-work item (not a vague "more theory needed"), consistent with `docs/ReadingList_DSN2027.md`'s existing citation practice.

**Why this gap is open, and how big it is (verified 2026-09-14).** Our composition accounting
across FL rounds (`GradientManager`, `src/ml/gradient_manager.py`) is naive (ε_total =
ε_per_round × rounds) by design choice, not oversight: the module implements a closed-form
single-round Gaussian-mechanism calibration and was scoped without an external DP-accounting
library (Opacus / TensorFlow Privacy / `dp-accounting` — zero occurrences anywhere in this
codebase, confirmed by grep), because the paper's central empirical measurement (does LiRA detect
membership under a DP-nominal configuration?) does not itself depend on how tight the *reported*
composition bound is — a looser ε_total only makes the DP configuration nominally weaker, which
is conservative for a paper reporting no detected leakage. Jayaraman & Evans (USENIX Security '19)
— verified directly against the paper text and the primary author's own blog summary — quantify
how loose naive composition is in practice: RDP matches naive composition's utility at roughly a
**50× tighter ε budget** (53% accuracy loss at ε=10 under RDP vs. ε=500 for the same loss under
naive composition, two-layer neural-network benchmark). This is the same 50× figure now cited in
the DSN 2027 paper (§2, §9) as the concrete grounding for why our reported multi-round ε should be
read as a conservative upper bound, not a tight guarantee. Remediation options, cheapest first:
(A) closed-form advanced composition (Dwork & Roth, 2014) — no new dependency, retroactively
computable from already-logged `epsilon`/`fl_rounds`; (B) closed-form RDP accounting specific to
the Gaussian mechanism — tighter than (A), still no external library; (C) a full external
DP-accounting library (Opacus / TensorFlow Privacy / `dp-accounting`) with true per-sample DP-SGD
— most rigorous, most engineering effort, left as future work given the 25/11 abstract deadline.
None of the three is implemented today.

## v2 / full metric — blocked, tracked as Task #64

The originally proposed formula was:

```
metric = FedMIA_AUC * Gradient_reconstruction_quality * Amount_of_recovered_sensitive_attributes * DP_level * FL_accuracy
```

Two of these five terms have no implementation to compute from today:

- **Gradient reconstruction quality** — requires a working Gradient Inversion attack module
  (none exists yet in `src/plugins/attacks/`; DLG/iDLG or similar would need to be implemented
  against the autoencoder's gradient updates).
- **Amount of recovered sensitive attributes** — requires defining which of the 6 ACN-Data
  features count as "sensitive" (likely `hour_of_day`, `duration_hours` as proxies for
  home/work location and routine) and a reconstruction-to-ground-truth distance threshold for
  "recovered."

Task #64 stays open and will be picked up once Gradient Inversion work starts. Until then, PES
v1 above — LiRA AUC × nominal DP strength, with utility cost reported alongside — is the honest,
fully-computable version, and is what should go in the DSN 2027 submission if the paper ships
before Gradient Inversion is implemented.
