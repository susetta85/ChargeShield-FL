# Privacy Exposure Score (PES) — v1

Status: **draft v1, in use for internal ranking only — not yet peer-reviewed or paper-ready.**
Owner task: #63 (v1, this document) / #64 (v2/full, blocked on Gradient Inversion).

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

## Central DP, ε=1.0 — the first nonzero PES_v1 (real result, 2026-07-24)

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

## Central DP, ε=0.1 — the most damning PES_v1 to date (real result, 2026-07-24, 14:49)

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

## Naming

Working name is "Privacy Exposure Score (PES)"; "Operational Leakage Score (OLS)" and "Critical
Infrastructure Privacy Risk (CIPR)" remain open alternatives — naming has no effect on the
formula above and can be decided later (e.g. based on which framing the introduction ends up
using — see `docs/DSN2027_Positioning.md`).

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
