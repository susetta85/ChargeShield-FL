# Privacy Exposure Score (PES) — v1

Status: **draft v1, in use for internal ranking only — not yet peer-reviewed or paper-ready.**
Owner task: #63 (v1, this document) / #64 (v2/full, blocked on Gradient Inversion).

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
