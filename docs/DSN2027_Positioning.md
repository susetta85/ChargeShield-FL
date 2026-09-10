# DSN 2027 Positioning — Framework, not FedMIA Paper

Status: draft v1, ready to fold into the actual paper draft. Owner task: #62. **Updated 2026-07-24
(independent review, same day):** corrected two overclaims about `src/plugins/attacks/` being
"already" pluggable when it wasn't yet. **Updated again 2026-07-24 (later the same day, user
requested it be made real rather than just described honestly):** the gap is now closed —
`BaseAttack` (`src/core/base_attack.py`) and a real registry (`src/plugins/attacks/ATTACK_REGISTRY`)
now exist, and `scripts/run_experiments.py`'s `main()` dispatches through the registry instead of
calling `run_fedmia`/`run_fedmia_shadow`/`run_lira` by name. The corrected "not yet implemented"
notes below are kept as a dated record of what was found and why the fix was scoped the way it
was — see "Implemented 2026-07-24" note after each for the current status.

> **Updated 2026-07-31 (user-requested correction, most important framing note in this document):**
> everything below still holds, but the emphasis was in the wrong place. **FedMIA/Yeom/Shadow/LiRA
> are not the contribution — they are the case study.** The framework's actual novel components are
> the **ML Plane** (`src/ml/ml_plane.py`, `MLPlaneListener`) and the **Privacy Auditor**
> (`src/auditor/privacy_auditor.py`), both domain-agnostic pieces of infrastructure that exist
> independently of which attack is plugged in. See the new "The actual contribution: ML Plane +
> Privacy Auditor" section below, inserted before the older "Reframed contribution list," which now
> describes the attack suite's role as case-study validation rather than as a contribution in its
> own right.

## Positioning statement (added 2026-08-27, use as the paper's framing sentence)

**ChargeShield-FL fills a real gap: it provides a practical bridge between theoretical DP
guarantees (ε, δ) and their actual empirical effectiveness against real attacks, applied to a
critical, emerging domain — electric vehicle charging infrastructure and the smart grid.** Its
distinctiveness is integrating a decoupled, event-driven observability layer (`MLPlane`), a
real-time auditor, and a composite risk metric (PES_v1) — validated **both in simulation and on a
genuine containerized deployment** (NVFLARE/Containerlab), not simulation-only like most
comparable work. Use this sentence (or a close paraphrase) as the paper's positioning statement —
it is the clearest one-sentence answer to "why does this paper exist" produced so far.

## The actual contribution: ML Plane + Privacy Auditor

This is the framing that should lead the paper's contributions section. Everything under
"Reframed contribution list" below is still accurate, but item 2 there (the pluggable attack
interface) and the whole Yeom/Shadow/LiRA suite are **the case study used to validate this
architecture**, not the architecture itself.

1. **The ML Plane** (`src/ml/ml_plane.py`) is a domain-agnostic, event-driven observability
   substrate for FL training. `AbstractMLModel` (`src/ml/base_ml.py`) gives every training/DP/
   aggregation component (`AutoencoderTrainer`, `GradientManager`, `FedAvgAggregator`) an
   `emit_event()`/`subscribe()` pair; `MLPlane.wire(*components)` subscribes itself to all of them
   as a single hub, and `FLArtifactCollector` is the first real consumer, assembling each round's
   raw and privatized updates without the training loop itself knowing anything is listening. None
   of this is EV-specific or attack-specific — it would wire up identically for any FL training
   loop (image classifiers, LLM fine-tuning, tabular models) that emits the same three event kinds
   (local update produced, update privatized, round aggregated). This is the part of the codebase
   that should be pitched as reusable open-source infrastructure, not the autoencoder or the
   ACN-Data adapter.
2. **The Privacy Auditor** (`src/auditor/privacy_auditor.py`, `PrivacyAuditor.audit()`) is a
   generic, config-driven (`config/auditor.yaml`) membership-inference-risk auditor: given any
   node's model update as a flat dict of numeric weights, it computes a gradient-sensitivity proxy,
   tracks cumulative differential-privacy budget consumption per node across rounds, and flags
   threats (`GRADIENT_EXPLOSION`, `PRIVACY_BUDGET_NEAR_EXHAUSTION`/`_EXHAUSTED`) — all without any
   dependency on EV charging semantics, the autoencoder architecture, or a specific attack. It is
   the framework's real-time, always-on privacy telemetry layer, complementary to (and independent
   of) whichever benchmark attack (Yeom/Shadow/LiRA today) is run offline against the same round
   data for a full post-hoc leakage measurement.
   >  **Accuracy note on "activated by the ML Plane" — resolved 2026-08-31 (Fase 8), originally
   >  flagged 2026-07-31.** The gap described below is now closed: `PrivacyAuditor` is wrapped by a
   >  new `PrivacyAuditorSubscriber` (`src/auditor/privacy_auditor_subscriber.py`) that genuinely
   >  implements `MLPlaneListener` and is registered via `MLPlane.subscribe()`, exactly like
   >  `FLArtifactCollector`. It reacts to the `"aggregation"` event (the ML Plane's "round complete"
   >  signal) and, at that point, reads the round's updates from the same `FLArtifactCollector`
   >  instance already assembling raw/privatized artifacts, then applies the identical
   >  peer-relative normalization formula `run_ids()`/`ChargeShieldAggregator` always used — same
   >  inputs, same outputs, no published number changes; only the activation mechanism moved from an
   >  imperative call to a genuine event subscription. This is live in both the real NVFLARE
   >  Aggregator (subscribed to its `MLPlane` during real training) and the simulation's `run_ids()`
   >  (which replays the saved `fl_results` dict through a fresh `MLPlane`/`FLArtifactCollector`,
   >  preserving its existing post-hoc design — the same choice already made for `run_lira()` — while
   >  making the Auditor's own activation genuinely event-driven either way). The original note is
   >  kept below, struck through in spirit but left readable, since it accurately described the state
   >  of the code for over a month and the fix it anticipated is exactly what was implemented:
   >  ~~today this is true at the *data-flow* level, not yet at the *code* level — `PrivacyAuditor`
   >  does not subclass `MLPlaneListener` and is not registered via `MLPlane.subscribe()`~~.
3. **Empirical goal these two components exist to demonstrate:** that differential privacy applied
   to the FL *communication channel* (DP-FedAvg noises each client's clipped update before it
   leaves the client; Central DP noises only the server-side aggregate) does not eliminate the
   *privacy leakage* the Privacy Auditor and the LiRA case-study attack can both detect from the
   raw per-client updates a semi-honest aggregator can observe before that noising happens (for
   DP-FedAvg, after clipping but this is what the client actually sends; for Central DP, the raw
   per-client contribution before the single aggregate noise draw). In other words: **protecting
   the channel is not the same as protecting the computation** — that gap remains the architectural
   point the ML Plane and Privacy Auditor exist to make measurable. **Superseded (2026-08-26):**
   the Central DP numbers originally cited here (LiRA AUC 0.743 at ε=1.0, 0.812 at ε=0.1) were
   later found, after a six-fix LiRA investigation (README Sprint 10x–10dd), to be substantially an
   artifact of the attack implementation, not a real measurement of this phenomenon — the corrected
   pipeline finds no LiRA-detectable leakage at any tested ε, in simulation or on the real
   Containerlab/NVFLARE deployment. The architectural claim above (channel-level DP ≠
   computation-level privacy) is a hypothesis this framework is built to test, not something these
   particular numbers demonstrate; see "Current validation status" below and README Sprint 10dd for
   what the corrected measurement actually shows.
3b. **Prior empirical DP-vs-MIA work exists — cited explicitly, not ignored (added 2026-09-03,
   in response to a direct question: "isn't this already studied empirically? what's our
   contribution then?").** Jayaraman & Evans (2019, USENIX Security), Rahman et al. (2018,
   Transactions on Data Privacy), and Nasr et al. (2019, IEEE S&P) all empirically measure whether
   DP mechanisms reduce MIA leakage — see `docs/LiteratureReview.md` §5 for the full entries. This
   is genuinely prior art and must not be presented as if it were novel. The honest answer to "what
   is our contribution then": none of the three combine real cross-silo FL clients (not a
   synthetically partitioned single dataset), a production FL framework validated with a genuine
   multi-container deployment (not simulation-only), and a reusable, domain-agnostic
   observability/audit architecture (the ML Plane + Privacy Auditor) that persists independently of
   which specific attack or DP configuration is being measured. The DP-vs-MIA *measurement itself*
   is not the novel part of this paper — items 1–2 above are.
4. **Why FedMIA/Yeom/Shadow/LiRA are the case study, not the contribution:** they are one
   concrete, swappable instantiation of "an attack that reads what the ML Plane observes and the
   Privacy Auditor already flags as risky." Registering a second attack (Gradient Inversion, next
   on the roadmap) validates the same ML Plane + Privacy Auditor architecture against a different
   adversarial capability, without either component changing. If Yeom/Shadow/LiRA were removed
   entirely and replaced with three different attacks tomorrow, the ML Plane and Privacy Auditor
   would not need to change — that is the test for "is X the contribution or the case study," and
   the attack suite fails it (by design) while the ML Plane and Privacy Auditor pass it.
5. **Reusability beyond EV charging — what is generic vs. what is domain-specific today.** For the
   "open-source framework usable for other scenarios" claim to be defensible rather than asserted,
   it needs to name which parts:
   - **Generic today, no EV/charging assumptions anywhere in the code:** `MLPlane`/`AbstractMLModel`/
     `MLPlaneListener` (`src/ml/base_ml.py`, `ml_plane.py`); `PrivacyAuditor`/`AbstractPrivacyAuditor`
     (`src/auditor/`, `src/core/base_auditor.py` — operates on a flat numeric `model_update` dict,
     nothing domain-specific); `BaseAttack`/`ATTACK_REGISTRY` (`src/core/base_attack.py`,
     `src/plugins/attacks/`); the three DP-mode implementations in `GradientManager`
     (`src/ml/gradient_manager.py` — dp-fedavg/central/local are generic mechanisms, not EV-specific
     math); `ByzantineDetector`'s Krum/CUSUM/Cosine-Similarity core (`src/ids/charging_ids.py` — Byzantine-
     robust aggregation checks that apply to any FL client population, despite the EV-flavoured
     class name).
   - **Domain-specific today — what an adopter targeting a different scenario would replace:**
     `ACNDataset` (`src/adapters/acn_dataset.py`, EV-session-schema-specific), the `Autoencoder`
     architecture and its 6 input features (`src/core/autoencoder.py`, tuned to EV session numeric
     fields), the `sites`/cluster naming in `config/experiment.yaml`, and the OCPP/MQTT protocol
     adapters (`src/adapters/ocpp16_adapter.py` etc. — themselves confirmed dead/unwired scaffolding
     as of Sprint 10d, not a real dependency of the current pipeline).
   - The class name `ByzantineDetector` is itself slightly misleading for the "generic framework" claim —
     its Krum/CUSUM/Cosine-Similarity logic has no EV-specific code path, only an EV-flavoured name;
     worth a rename (e.g. `ByzantineDetector`) if/when this becomes an actual public framework
     release, flagged here as a low-priority naming cleanup, not a functional gap.

## Why this reframe

The project's original framing was "we show FedMIA still works despite DP" — a single-attack,
single-result paper. That story has a shelf life: once the LiRA-vs-DP result is published, the
paper's contribution is spent, and every future addition (Gradient Inversion, Property
Inference, Secure Aggregation) reads as a new, disconnected paper rather than growth of the same
line of work.

Most of the reframe costs nothing in engineering terms, because most of the architecture already
supports it: the IDS/defence stack (`src/ids/`, `src/auditor/`) is already decoupled from any
specific attack, and the three DP placements (DP-FedAvg / Central / Local) are already
attack-agnostic knobs — switching between them is a config flag, not a code change. What changes
for those two pieces is the narrative layer — abstract, introduction, contributions — not the code.

**Correction (2026-07-24, caught by independent review):** this section originally also claimed
`src/plugins/attacks/` "already treats FedMIA as a swappable module." That overstates the current
code: the directory holds a single file, `fedmia.py`, confirmed unused by the live pipeline
(`run_ids()`/`ChargeShieldAggregator` never instantiate it), with no shared base class and an empty
`__init__.py` — there is no registration mechanism to swap a different attack in. `docs/Architecture.md`
§4.4 already says this honestly: *"PluginRegistry and filesystem-based plugin discovery are not yet
implemented... a design goal for a future sprint."* The paper must describe the attack interface as
**designed for pluggability, not yet plug-and-play** — either make that the honest wording, or
implement a minimal `BaseAttack` interface (Yeom/Shadow/LiRA as classes against a common contract,
plus a small registry) before claiming it in a submission. This is a real, scoped, non-trivial
piece of work (it touches `run_experiments.py`, `run_lira()`, `run_ids()` — code that already
produced published Central DP numbers), not a doc-only fix, so it should be a deliberate decision,
not something silently done in passing.

**Implemented 2026-07-24 (later the same day):** done, as a thin wrapper layer rather than a
rewrite, specifically to avoid the regression risk flagged above. `src/core/base_attack.py` defines
`BaseAttack`; `src/plugins/attacks/{yeom,shadow,lira}.py` each implement it by calling the existing,
unmodified `run_fedmia()`/`run_fedmia_shadow()`/`run_lira()` — zero change to their internal logic,
so none of the empirically-validated fixes documented in those functions' docstrings (LiRA alone
has 5 rounds of them) are at risk. `src/plugins/attacks/ATTACK_REGISTRY` maps name → class. A new
`run_registered_attacks()` in `run_experiments.py` iterates the registry (same execution order,
same per-attack error handling, same per-round merge logic as the pre-refactor direct calls) and is
now the single call site used by both `run_experiments.py::main()` and
`scripts/run_nvflare_mia.py::main()` — previously two near-duplicate copies of the same dispatch
logic that could silently diverge. Adding a new attack now genuinely means "add a file + one
registry entry," not touching either `main()`. Verified: `py_compile` on all changed/new files; the
new `BaseAttack`/registry layer is fully unit-tested without torch (`tests/test_attack_registry.py`,
8 tests — subclass contract, abstract-method enforcement, registry membership); the dispatcher
function itself and both `main()` changes could only be `py_compile`-checked, not executed, in this
sandbox (torch unavailable) — same limitation as the rest of this project's FL/attack code. The
existing torch-dependent integration tests (`tests/test_run_experiments_integration.py`) still call
`run_fedmia()`/`run_lira()` directly and are unaffected, since those functions are unchanged.

## New Abstract (replaces the current README abstract for paper purposes)

> **Updated 2026-07-31 to lead with the ML Plane + Privacy Auditor architecture rather than the
> attack suite** — see "The actual contribution" section above for the full rationale.

> Federated Learning (FL) is increasingly proposed as the privacy-preserving path for training
> shared models over critical infrastructure — smart grids, EV charging networks, industrial
> control systems — where centralising raw operational data is both a regulatory liability and a
> security risk. Whether FL actually delivers on that promise in realistic, production-style
> deployments is an empirical question that current research answers piecemeal: one paper per
> attack, one dataset per paper, rarely a production FL framework, rarely an industrial dataset. We
> present **ChargeShield-FL**, built around two domain-agnostic components: the **ML Plane**, an
> event-driven observability substrate that any FL training loop can wire into without coupling
> the training code to whatever monitors it, and the **Privacy Auditor**, a config-driven,
> real-time membership-inference-risk auditor that consumes what the ML Plane observes to flag
> per-node, per-round privacy risk independent of any specific attack. Using real EV charging
> session data (ACN-Data, 3 sites: Caltech, JPL, Office 1) as our case study — not synthetic data
> standing in for a real deployment — we instantiate a pluggable benchmark attack suite
> (Yeom/Shadow/LiRA today; gradient inversion and property inference as planned extensions) against
> this architecture to answer the practically relevant question: **does differential privacy
> applied to the FL communication channel actually stop the privacy leakage the Privacy Auditor is
> built to detect, or only make the channel itself look private while the underlying computation
> still leaks?** A companion NVFLARE job/app implementing the same pipeline now runs on a genuine
> multi-container Containerlab/Docker federation, not just a scaffold — see "Current validation
> status" below. Our results, using the LiRA case-study attack under a corrected and independently
> re-verified implementation (see `README.md` Sprint 10x–10dd for the six-fix investigation this
> required), converge across four independent tests — a local simulation, a full DP-mode sweep,
> and the real multi-container deployment — on a single finding: **no LiRA-detectable membership
> signal survives in this architecture, under any tested DP configuration, including no DP at
> all** (composed AUC-ROC 0.4992–0.5018 throughout). This is itself the practically relevant
> answer, not a null result to be explained away: it means the ML Plane + Privacy Auditor
> architecture, instrumented against a strong, well-established MIA (LiRA, Carlini et al. 2022),
> did not find the exposure gap the initial hypothesis expected — either because this particular
> model/dataset/FL configuration does not memorise enough for LiRA to detect at this scale, or
> because the architecture's real-world exposure is genuinely lower than the literature's
> synthetic-benchmark results would suggest. Both are reportable, useful findings for anyone
> deciding whether to deploy FL with DP over EV charging infrastructure; distinguishing between
> them (via a larger memorisation regime, more rounds, or a stronger/different attack) is the
> immediate next step, not a reason to withhold the result.

### Current validation status — do not overclaim this in the paper

Caught by independent review (2026-07-24) before this draft went further: the sentence above
originally claimed results came from "a genuine multi-site NVFLARE federation with DP, Byzantine
aggregation, and IDS all active simultaneously." That is not what happened and must not reach a
submission. Concretely:

- All reported numbers (`experiments/experiment_*.json`, `dp-sweep*/`, `nodp-sweep1/`, and the
  Central DP results — see below) come from the single-process Python simulation in
  `scripts/run_experiments.py` — real ACN-Data, real DP/attack/IDS code, but one process, not a
  deployed multi-container NVFLARE federation.
- **Updated 2026-07-24, later the same day**: `nvflare/jobs/chargeshield_poc/` was executed for the
  first time via `nvflare simulator` (`make nvflare-sim-smoke`/`make nvflare-sim`, local
  processes/threads, no Docker/Containerlab) — it is no longer accurate to say it "has never been
  run." Two real bugs were found and fixed on that first run (a path-resolution bug and an
  initialization-state bug — see `docs/NVFlareIntegration.md`'s "First real run" section for the
  full account).
- DP and Byzantine/IDS are **not** measured together in one privacy-leakage run today:
  `run_experiments.py` explicitly skips FedMIA/Shadow/LiRA whenever `byzantine_attack.enabled` is
  true (Byzantine sweeps validate Krum detection only, in `experiments/ids_validation/`, and are
  a separate, non-privacy measurement — see the Makefile's `experiment-byzantine-sweep` comment
  block). So "DP + Byzantine + IDS active simultaneously while measuring MIA" is not a result this
  project has produced.

**Superseded (2026-08-03 through 2026-08-26) — the Containerlab/Docker multi-container deployment
has now actually been attempted and completed, closing the gap the paragraph above used to
describe as "genuinely has not been attempted."** `containerlab/topology.clab.yml` was rewritten
against the real 5-node topology (Sprint 10q), then deployed for real across four rounds of
genuine bugs found and fixed on the user's own machine (Sprint 10s bind-path resolution, 10t
read-only jobs mount, 10u CPU thread-thrashing, 10w missing hold-out split — see
`docs/NVFlareIntegration.md` for the full account of each). A complete 10-round job
(`e19cfa15-efde-4eea-ac85-438289fe3a4f`) ran end-to-end across the real `server`/`caltech`/`jpl`/
`office1`/`fl-admin` containers on 2026-08-26, and `scripts/run_nvflare_mia.py` — which reuses the
exact same, now-fixed `run_lira()`/`run_ids()`/`run_fedmia()` functions the simulation uses — was
run against its output for the first time in this project's history. Result: composed LiRA
AUC-ROC 0.4992, converging with the local simulation's no-DP and Central DP results (README
Sprint 10dd). **The honest version of the claim, updated**: real dataset, real DP mechanism code,
real attack code, validated in both a single-process simulation and a genuine multi-container
Containerlab/Docker/NVFLARE deployment, with both now converging on the same finding — no
LiRA-detectable membership leakage in any tested configuration. Multi-seed statistical
confirmation of this null result is now complete (5-seed × 10-config bootstrap campaign, task #52,
completed 2026-09-08 — see `docs/MetricsReference_DSN2027.md` §8 for the final table): all 10
configurations show n=5, CI containing 0.5, and Wilcoxon p-values between 0.3125 and 1.0000. The
finding rests on real data, real code, and now-confirmed multi-seed statistics, not a single-seed
or simulation-only basis.

### Worst-case vs. average-case privacy evaluation (added 2026-09-03, user-raised methodological point)

**The concern, stated precisely**: Carlini et al. (2022, the paper LiRA is built on) explicitly
argue that AUC-ROC is an inadequate MIA metric — it averages over every possible decision
threshold, weighting regimes where the attacker makes enormous numbers of false positives exactly
as heavily as the low-FPR regime where a privacy attack is actually meaningful. "Privacy is not an
average-case guarantee": DP itself is a worst-case guarantee (holds for every possible pair of
neighboring datasets), so validating it with an average-case empirical metric is a real
methodological mismatch, not a nuance. **This project's headline number, reported everywhere in
every Sprint-log entry and every results table, has been `mean_lira_auc_roc` — exactly the metric
Carlini et al. argue against relying on.** This is a real, previously-flagged gap (see
`docs/ReadingList_DSN2027.md`'s Carlini 2022 entry, point (1), 2026-08-14) that has not yet been
corrected in how results are *reported*, even though the code-level fix has existed since Sprint
10pp (2026-08-28): `_tpr_at_fixed_fpr()` computes TPR at FPR∈{0.001, 0.01, 0.05} — exactly the
low-FPR regime Carlini et al. say is the only one that matters for privacy — alongside AUC-ROC, for
every round of every attack.

**Verified with real data (2026-09-03), not assumed**: does the low-FPR-regime metric, already
computed for most of the main campaign, tell a different story than the average-case AUC? Checked
directly against the saved JSON files for `dp-sweep3`, `local-sweep1`, `local-sweep2`,
`entity-split-sweep1`, `central-sweep1`, `central-sweep2` (300 real LiRA rounds total, all
post-Sprint-10pp so `tpr_at_fpr_*` is populated): **mean TPR@1%FPR = 0.0098 (chance ≈ 0.01, max
observed 0.0204 across all 300 rounds), mean TPR@0.1%FPR = 0.0009 (chance ≈ 0.001, max observed
0.0034).** No worst-case-sensitive signal hiding behind the average-case AUC null result — the two
lenses agree. This is reassuring for the central claim, but it is a genuinely new check, not
something the project could previously cite as already done.

**What this means for the paper — and what does NOT require re-running experiments**:
1. **Reporting convention must change, not the experiments.** `tpr_at_fpr_0.01` (or MIA Advantage,
   task #41) should be the headline metric in every results table and in the abstract, with
   AUC-ROC reported as a secondary, literature-comparison number — the reverse of the convention
   used throughout this project's Sprint log (`"LiRA AUC-ROC medio: X — Privacy risk: LOW"`). The
   underlying data for this already exists for 5 of the 6 major campaign families (dp-sweep3,
   local-sweep1/2, entity-split-sweep1, central-sweep1/2) — a documentation/analysis pass, not new
   experiments.
2. **Small, scoped backfill still needed, not "almost everything"**: `dp-sweep1`, `dp-sweep2`, and
   `nodp-sweep1` predate Sprint 10pp and have no `tpr_at_fpr_*` fields — these three (not the whole
   campaign) would need a re-run for complete low-FPR-regime coverage of every published
   configuration.
3. **MIA Advantage / Confusion Matrix (tasks #41/#49) are brand new (2026-09-02/03)** and, like
   `tpr_at_fpr_*` before them, are not retroactively computable from historical JSON files (they
   need the full ROC curve, never saved historically). Only `central-sweep3`/`central-sweep4`
   (2026-09-03) currently have them. Whether the paper needs these for the *entire* published
   campaign or only for the corrected Central DP re-sweep is a scoping decision, not an
   automatic "re-run everything."
4. **Still genuinely missing, and the strongest direct answer to Carlini's point**: a true
   *per-sample* worst-case check — is there any single real (non-canary) record that is reliably,
   individually distinguishable across seeds/rounds, even though the population-level TPR@low-FPR
   is at chance? `tpr_at_fpr_*` is already a population-level low-FPR metric, a real improvement
   over AUC, but it still aggregates over the whole eval pool rather than flagging individual
   vulnerable records. This has not been attempted and is not a re-run — it is a new post-hoc
   analysis question, and the canary experiments (point 8 above) are this project's closest
   existing proxy: a *deliberately constructed* worst-case record, found to be reliably detectable
   at office1's dilution level (~11% duplicated) but not at caltech/jpl's (~0.5%) — itself a
   worst-case-oriented finding, not an average-case one, worth foregrounding in the paper
   alongside the TPR@low-FPR numbers above. **Resolved 2026-09-03 (task #50)**: implemented —
   `--per-sample-dump` + `scripts/analyze_worst_case_vulnerability.py`, cross-seed percentile
   consistency check on real `session_id`s. See `docs/MetricsReference_DSN2027.md` §10c. Not yet
   run on real data (needs 2+ future seeds of the same config).

**Follow-up round (2026-09-03, same session, three further points raised by the user):**

5. **The averaging fallacy also applies ACROSS attacks, not just within one.** A separate Carlini
   quote: comparing attack A vs. attack B by a single aggregated accuracy can rank a surgically
   precise attack (nails a tiny 0.1% subgroup, fails elsewhere) as equal to a uniformly mediocre
   one. This project never literally uses "balanced accuracy", but it does use AUC-ROC as the de
   facto cross-attack comparison metric (Yeom vs. Shadow vs. LiRA) — same blind spot, different
   name. **Resolved (task #53)**: TPR@low-FPR/Advantage/Confusion, previously wired only into
   LiRA, now also computed for Yeom (`run_fedmia()`), Shadow (`run_fedmia_shadow()`), and the
   canary raw-loss diagnostic. See `docs/MetricsReference_DSN2027.md` §4/§5/§10d. Not retroactive.
6. **Theoretical grounding of why LiRA's log-likelihood-ratio score is the right statistic**: the
   user walked through Carlini et al.'s own derivation — the Neyman-Pearson lemma (optimal test at
   fixed FPR is a likelihood-ratio threshold) → the exact test is intractable (would need the full
   distribution over trainable models) → reduced to a tractable 1D statistic (the *loss* of the
   target model on the query point) → approximated as Gaussian (μ_in/σ_in vs. μ_out/σ_out,
   estimated from shadow models). This is precisely the formula already implemented in `run_lira()`
   (§3) — previously undocumented as such. **Resolved**: full three-step derivation, mapped
   line-by-line to the code, now in `docs/MetricsReference_DSN2027.md` §3 ("Fondamento teorico").
   Useful for the paper's theoretical background section — this is the DSN reviewer-facing
   justification for why LiRA (not Yeom/Shadow) is the primary attack, beyond "it performed best
   empirically."
7. **Recommendation for future research: always report full ROC curves in log-log scale, not just
   point metrics.** The user's own suggestion, directly downstream of point 1 above: a single
   TPR@0.01 number is still just one point on the curve; a full log-log ROC plot (both axes
   logarithmic, per Carlini et al.'s own Figure 3 convention) shows the entire low-FPR behavior at
   once, not a discrete sample of it. **Resolved (task #54)**: implemented for this project's own
   use, not just recommended to others — `run_fedmia()`/`run_fedmia_shadow()`/`run_lira()` can now
   dump full fpr/tpr arrays (`--roc-curve-dump-dir`), and `scripts/plot_roc_log_scale.py` renders
   them log-log with a chance-level diagonal reference. See `docs/MetricsReference_DSN2027.md`
   §10e. **For the paper itself**: recommend citing this convention explicitly in the Methodology
   section, and including at least one log-log ROC figure (composed LiRA, main population vs.
   canary) as supplementary evidence alongside the TPR@low-FPR table — the community-facing
   recommendation the user asked for is this practice itself, demonstrated rather than just stated.
8. **Parametric vs. non-parametric modeling — a design choice worth naming explicitly, not a
   neutral implementation detail.** The user asked directly: is this project's LiRA parametric or
   non-parametric, and does it matter for the paper? Answer, verified against the actual paper text
   (arXiv 2112.03570, fetched directly, not from memory): **parametric** — the Gaussian fit
   (μ_in/σ_in, μ_out/σ_out already in §3 above) is exactly Carlini et al.'s own §IV-C choice
   ("Estimating the likelihood-ratio with parametric modeling"), justified there by two concrete,
   citable reasons — ~400× fewer shadow models than a non-parametric alternative the paper compares
   against, and extensibility to multivariate fits for multi-query attacks. Worth a sentence in the
   Methodology section for exactly this reason: it preempts "why Gaussians and not an empirical
   estimate?" **A genuine, previously undocumented deviation surfaces alongside this**: Carlini et
   al. only fit Gaussians after a *logit-scaling* transform of model confidence, specifically
   because raw confidence/loss is empirically NOT Gaussian-shaped while the logit-transformed value
   is (their Fig. 4/8). ChargeShield-FL fits the Gaussian directly to raw reconstruction MSE, with
   no equivalent transform — there is no natural logit for an unsupervised autoencoder's
   reconstruction error, so this isn't an oversight, but its effect has never been checked. **Open
   question, not a claim**: the project has repeatedly observed σ_in/σ_out hitting a numerical floor
   in 96–99% of cases across several sweeps, so far always attributed to sample-size/pooling causes
   (see `docs/TestRoadmap_DSN2027.md`). Whether raw MSE is poorly-approximated by a Gaussian (a
   distributional-shape problem, independent of sample size) is an untested alternative explanation
   — the data needed to check it (raw per-shadow MSE distributions, not just the composed
   post-fit score already dumped by task #50) isn't currently saved anywhere. Flagged as a stated
   limitation for the paper and a candidate for future work, not investigated further here. Full
   detail and the exact Carlini citations: `docs/MetricsReference_DSN2027.md` §3.
9. **Sablayrolles et al. 2019 as a free ablation baseline (task #58, Sprint 10zz+33, 2026-09-03).**
   The user quoted Carlini et al. 2022's own surprised aside (§V-C, Table I): despite being
   published in 2019, Sablayrolles et al.'s attack — a *non-parametric* per-example threshold,
   `τ_{x,y}=(μ_in+μ_out)/2`, no Gaussian fit at all — beats most other pre-LiRA attacks at low FPR,
   and is explicitly called out as "the most direct influence for LiRA" by Carlini's own authors.
   Implemented as a second scorer computed inside the same `run_lira()` round loop, reusing the
   exact same per-example μ_in/μ_out already computed there (our LiRA is the *online* variant, so
   both distributions are always available per sample) — zero additional shadow models, zero
   additional training. This turns point 8 above from a theoretical claim into a directly
   measurable one: comparing `sablayrolles_auc_roc`/`sablayrolles_tpr_at_fpr_0.001` against LiRA's
   own on the same round is literally the ablation Carlini reports in their Table II ("+
   Per-example thresholds" → "+ Gaussian Likelihood"), run here on real EV charging data instead of
   CIFAR-10. **Update 2026-09-03/04**: now run on real data, twice (seed=42 and, independently,
   seed=123) — `sablayrolles_auc_roc` lands at 0.5011/0.5006 in the two runs, statistically
   indistinguishable from LiRA's own 0.5043/0.4993 on the same rounds. See
   `docs/MetricsReference_DSN2027.md` §10f/§3 for the full implementation, sign-convention, and
   both real-data results.
10. **Bug-hunting round + log-MSE ablation + independent replication (tasks #59–#62, #66,
    2026-09-03/04).** Three real bugs found and fixed in the attack-scoring pipeline, each a
    variant of the same failure shape — a new metric written under a key that collided with an
    existing one during a later dict merge, silently overwriting data rather than erroring:
    (a) task #59, Yeom's/Shadow's own TPR@low-FPR silently overwritten by LiRA's in
    `run_registered_attacks()`'s merge (never actually triggered on published data — no historical
    JSON had those fields yet); (b) task #60, diagnostic dumps (`raw_loss_dump_path` etc.) failing
    with `FileNotFoundError` because the sweep-dir was created only at the very end of the
    pipeline, costing the user ~47 minutes of real compute before being caught; (c) task #66, the
    "LiRA composto" (multi-round) TPR@low-FPR silently overwriting the last round's own TPR@low-FPR
    in `results[_final_round]` — live since 2026-08-28, affecting every LiRA run since, though the
    already-published PES v1.1 table turned out to have been reading that same composed value all
    along (see `docs/PrivacyExposureScore_v1.md`), so no republished numbers were needed, only an
    explicit key rename. Separately, task #61 ran a targeted ablation testing whether the
    frequently-observed σ_in/σ_out floor-hit-rate (point 8 above) is caused by raw MSE's
    non-Gaussian shape: fitting the same LiRA Gaussian on log(MSE) instead cuts the floor-hit-rate
    from ~99.8% to <0.1% (~2000×, confirmed causally) — but AUC/TPR/Advantage stay statistically
    unchanged, meaning the floor was a real technical artifact that nonetheless was NOT masking any
    actual membership signal. Task #62 replicated this exact finding with an independent seed
    (123, not just 42): floor-hit-rate 99.85%→0.10%, AUC/TPR/Advantage again unchanged across raw,
    Sablayrolles, and log-MSE scorers — "Scenario B" (no detectable leakage) now rests on evidence
    replicated across an independent seed, not a single run. Full detail:
    `docs/MetricsReference_DSN2027.md` §3/§8, README Sprint 10zz+34/37/38/39/40/41.

## Reframed contribution list

> **Note (2026-07-31):** read this list alongside "The actual contribution: ML Plane + Privacy
> Auditor" above, not instead of it. Items 1, 3, and 4 below are genuine supporting contributions.
> Item 2 (the pluggable attack interface, and by extension Yeom/Shadow/LiRA themselves) is the
> **case-study validation layer** for the ML Plane + Privacy Auditor architecture, not a
> stand-alone contribution — kept in this list because "pluggable" is itself a real, useful
> property of the harness, but the paper should not present "we made the attacks pluggable" as
> equal in weight to "we built a reusable FL privacy-observability architecture and used it to
> rigorously test whether DP-on-the-channel stops leakage." **Updated 2026-08-26**: the answer that
> test now returns, on the corrected LiRA implementation, is "no leakage was detectable in the
> first place, at any tested DP configuration" (README Sprint 10dd) — a different conclusion than
> the "DP-on-the-channel doesn't stop leakage" phrasing above assumed, but the harness itself (the
> actual contribution this list is about) is what produced that answer credibly, and that is the
> point to keep making in the paper regardless of which way the empirical finding landed.

Where the old framing had one contribution ("we show LiRA beats DP"), the benchmark framing
supports a list that keeps growing:

1. **A reproducible measurement harness** for FL privacy leakage — a real multi-site industrial
   dataset (ACN-Data, 3 real sites) and real DP/IDS/aggregation code, validated in a
   single-process simulation (not synthetic data standing in for any of the three). **Updated
   2026-08-26**: the matching NVFLARE job/app is no longer a scaffold — a genuine multi-container
   Containerlab/Docker/NVFLARE federation was deployed and a full 10-round job completed and
   analyzed for the first time (README Sprint 10s–10dd), converging with the simulation's result.
   This is now a current claim of this paper, not planned work — see "Current validation status"
   below for the full account. (Corrected 2026-07-24: this bullet originally claimed "real NVFLARE
   federation... not a simulated stand-in" prematurely, before that was true — flagged then as an
   overclaim, and is genuinely true as of 2026-08-26.)
2. **A pluggable attack interface** (`src/core/base_attack.py`'s `BaseAttack`, registered in
   `src/plugins/attacks/ATTACK_REGISTRY`) — **implemented for real 2026-07-24**, closing the gap
   this bullet used to describe as aspirational. Yeom/Shadow/LiRA are now registered classes;
   `run_experiments.py::main()` dispatches through the registry rather than calling each attack by
   name, and the same dispatcher is shared with `scripts/run_nvflare_mia.py`. Adding a new attack —
   Gradient Inversion is next (Task #64/roadmap) — means adding one file and one registry entry,
   not touching either `main()`. The three existing wrappers are intentionally thin: each calls the
   original, unmodified `run_fedmia()`/`run_fedmia_shadow()`/`run_lira()`, so none of their
   empirically-validated fixes (LiRA alone has 5 documented rounds of them) were touched by this
   refactor.
3. **An empirical audit of DP's real-world guarantee** across three placements (DP-FedAvg,
   Central, Local) and a realistic ε range, on real session data rather than a synthetic or
   IID-shuffled proxy for it. (Clarified 2026-09-01, anticipating a likely reviewer question:
   no-DP is not a fourth placement — it carries no noise-injection mechanism to audit. It is the
   baseline every one of the three placements is measured against, used throughout as the
   reference point for both leakage (mean_auc_roc) and utility cost (`U_cost` in
   `docs/PrivacyExposureScore_v1.md`, `strength(ε) = 0` for the no-DP case) — essential to the
   comparison, but not itself one of the three audited mechanisms.)
4. **A composite risk metric (PES, `docs/PrivacyExposureScore_v1.md`)** that scores the gap
   between a DP configuration's nominal privacy claim and its empirically measured protection —
   designed to flag exactly the "epsilon is misleading" failure mode this line of work cares
   about, and built to grow as more attack modules are added.
5. **A public benchmark, not a one-off result** — every new attack or defence added to the
   harness produces a new row in an existing comparison table, rather than requiring a new paper
   framing from scratch.

## What does NOT need to change

- The DP/IDS modules never assumed FedMIA was the only attack that would ever run against them —
  no changes needed there for the reframe.
- The existing experiment pipeline, Makefile targets, and result JSON schema stay exactly as
  they are — the benchmark framing describes what the project already does, it does not require
  redoing it.
- This reframe does not commit the project to actually building Gradient Inversion or Property
  Inference before DSN 2027 — the "roadmap" framing works even if LiRA remains the only attack
  in the submitted paper, as long as the paper is honest that the others are planned extensions
  of the same harness, not vague future work with no interface to attach to.

**Correction (2026-07-24):** this section previously also claimed `src/plugins/attacks/fedmia.py`
"is already isolated behind a plugin boundary" and needed no changes. That was not accurate at the
time — see the correction under "Why this reframe" above. **Update, later the same day:** the
pluggable-attack contribution (item 2 above) is now genuinely real — a `BaseAttack` class and
registry exist and are the actual dispatch path in `run_experiments.py`. `fedmia.py` itself remains
untouched and still unused by the live pipeline (it's a different thing — an IDS-facing shadow
plugin, not one of the three experiment-level attacks); it was never part of what needed fixing here.

## Suggested introduction restructuring

Current README introduction ("Why ChargeShield-FL?") leads with the EV-specific privacy problem.
For the paper, keep that as *motivation* but move the "no benchmark exists" argument earlier and
sharpen it: the DSN reviewer needs to see, in the first page, that this is filling an
infrastructure-benchmark gap (few FL-privacy papers use a production framework like NVFLARE,
fewer use real industrial data, essentially none combine DP + Byzantine-robust aggregation + IDS
in one measured pipeline) rather than adding one more MIA-on-a-toy-dataset paper to an already
crowded space. Section 6 of `docs/LiteratureReview.md` (once populated, Task #65) should supply
the citations that make that gap claim defensible rather than asserted.

## Limitations and scope boundaries (added 2026-08-27, in response to external review)

A fourth round of external review raised four methodological concerns for a top-tier submission:
limited model/task diversity, small client count, PES_v1's lack of theoretical grounding, and no
adaptive-attacker evaluation. Decision (2026-08-27, user-confirmed): given the 2026-11-25 abstract
deadline, points 1–2 below are stated as explicit scope boundaries in the paper (not new
experiments), and points 3–4 get a cost-zero analytical/discussion treatment now (not new code or
experiments) — full detail lives in the documents cross-referenced below, not duplicated here.

1. **Single model architecture (570-parameter autoencoder), no ResNet/Transformer/DNN
   comparison.** Valid concern for a general theory of "how MIA behaves across architectures," but
   that is not this paper's claim. The paper's contribution (see "The actual contribution" above)
   is the ML Plane + Privacy Auditor architecture and an empirical measurement on one real
   industrial case study, not a cross-architecture MIA characterization. **How to write this
   (refined 2026-08-27):** in the Threat Model & Limitations section, frame the focus as end-to-end
   privacy auditing on real cyber-physical/edge systems, where local computational capacity is
   genuinely constrained (a well-established property of OT/ICS edge deployments — see the Purdue
   Model discussion already in `docs/Architecture.md`/`docs/ThreatModel.md`, not a claim specific
   to this exact model's measured hardware budget, which was never benchmarked against a real EVSE
   controller). Explicitly defer LLM/Vision-scale model study to a pure information-security
   context, distinct from this paper's cyber-physical-systems framing — do not imply a hardware
   measurement that was never taken; frame it as "consistent with the constraints typical of this
   deployment class," not as a proven bound.

   **Added 2026-09-02 (user question: "usiamo un modello supervisionato o no? è importante
   questo nel progetto?") — reframe "single model" as "single model, and it is unsupervised,"
   which is a distinguishing feature, not just a narrower limitation.** The target model
   (`src/core/autoencoder.py`) is trained **without labels** — a reconstruction-error anomaly
   detector (encoder 6→16→8→4, decoder 4→8→16→6, MSE loss), not a classifier. Yeom et al. (2018),
   Shadow (Shokri et al. 2017), and LiRA (Carlini et al. 2022) were all originally formulated and
   evaluated against **supervised classifiers** (cross-entropy loss as the membership signal); this
   paper's case study substitutes reconstruction MSE for cross-entropy loss as the same kind of
   per-sample signal, without any change to the attacks' underlying logic (a lower loss = more
   likely a member, regardless of what produces that loss). This is worth stating explicitly in the
   paper as a **secondary contribution of the case study**, not merely a caveat: most published MIA
   evaluations target classification, and evidence (or absence, per our null result) of membership
   leakage in an unsupervised, reconstruction-based FL model is a less-studied setting in the
   literature this paper can point to directly (see `docs/LiteratureReview.md`'s note on iDLG for a
   related "our autoencoder has no class labels" observation already on record for gradient-inversion
   attacks specifically). **Still a real, unaddressed limitation**: whether the null result
   generalizes to a supervised FL classifier on the same or different data remains untested and
   should be named as such alongside the existing single-architecture caveat above — the two are
   related but distinct claims (breadth of architectures tested vs. breadth of learning paradigms
   tested).

2. **3 real sites / 5 Containerlab nodes, not 50–100 heterogeneous clients.** Building a
   50–100-client testbed would require synthetic client partitioning — the opposite of this
   project's actual differentiator (real, organizationally-distinct clients rather than an
   artificially sliced single dataset, already a rarity in FL-privacy literature per the
   benchmark-gap argument above). **How to write this (refined 2026-08-27):** in the Experimental
   Setup section, name the paradigm explicitly as **Enterprise Cross-Silo Federated Learning**
   (Kairouz et al. 2021's standard FL taxonomy: few, large, reliable, organizationally-distinct
   participants — e.g. hospital consortia or regional grid operators — as opposed to cross-device
   FL's many small unreliable clients). Argue the natural heterogeneity of 3 genuinely distinct
   real organizations has substantially higher ecological validity than synthetically fragmenting
   one generic dataset into N pieces, which is what most published cross-silo FL-privacy papers
   actually do. State plainly: PES_v1's behavior under large-N, high-heterogeneity cross-device FL
   is untested and out of scope here.

3. **PES_v1 is a heuristic score, no formal theoretical grounding.** Addressed with an informal
   analytical grounding, not a new proof — see `docs/PrivacyExposureScore_v1.md`'s new "Theoretical
   grounding (informal)" section (added 2026-08-27, extended same day with a tighter, more
   applicable bound): `L(AUC)` is designed to play the same conceptual role as the
   membership-advantage quantity formally bounded by Yeom et al. (2018) at `e^ε − 1` for ε-DP
   mechanisms, and — more precisely, since this project's Gaussian Mechanism gives (ε,δ)-DP, not
   pure ε-DP — by **Humphries et al. (2020)** at `(e^ε − 1 + 2δ)/(e^ε + 1)`. Both are citations
   this project can defend (Yeom is already one of the three benchmark attacks; Humphries gives the
   δ-aware version this project's actual mechanism needs). This motivates the `ε`-dependence of
   PES_v1 without claiming an exact derived bound; the document is explicit about where the analogy
   is precise and where `strength(ε) = 1/(1+ε)` remains a designed-not-derived heuristic weighting,
   and proposes a concrete "PES v1.1" refinement (normalizing observed advantage against the
   Humphries bound directly, once TPR@low-FPR — `docs/TestRoadmap_DSN2027.md` #4 — gives a metric
   with a cleaner formal correspondence than AUC) rather than a vague "needs more theory" gap.
   **How to write this in the paper (2026-08-27):** in the Privacy Auditor section, spend roughly
   half a column formalizing PES_v1 against the Humphries bound above, and explain explicitly *why*
   `U_cost` is reported as a companion axis rather than folded multiplicatively into PES_v1 — to
   avoid what is worth naming directly as the "perfect privacy at zero utility" fallacy: a
   configuration that destroys model utility can trivially drive `L(AUC)` to 0 without providing
   any privacy protection worth the name (`docs/PrivacyExposureScore_v1.md` already makes this
   argument under "Reading this table honestly" — cite it directly, don't re-derive it).

5. **Single dataset (ACN-Data only).** A fifth review point (2026-08-27) asked for a second EV
   dataset (ChargePlace Scotland, already downloaded at `datasets/alt/chargeplace_scotland/`, task
   #89/roadmap #6) to show PES_v1/the architecture aren't overfit to ACN-Data's specific
   collection process. **Decision (2026-08-27, user-confirmed): deferred, stated as an explicit
   limitation, not scheduled before 2026-11-25** — integration is a multi-week adapter effort, not
   a re-run. State in the paper: validated on one real EV dataset; replication on a second,
   independently-collected dataset is future work (see `docs/TestRoadmap_DSN2027.md` #6 for the
   reasoning already on record).

   **Update (2026-09-09, task #37):** the adapter (`src/adapters/chargeplace_scotland_adapter.py`)
   has since been written, tested, and wired into `scripts/run_experiments.py` — faster than the
   "multi-week" estimate above assumed. This does **not** change the paper-facing claim yet: no
   full campaign has been run on ChargePlace Scotland (only a timed smoke test is planned before
   any scope decision), so there are no paper-citable ChargePlace Scotland numbers as of this
   writing. Continue to state replication on a second dataset as future work until a campaign
   actually completes.
6. **Fixed-threshold gradient clipping only, no adaptive clipping comparison.** Same review point:
   `GradientManager` uses a fixed `max_grad_norm` in every DP mode tested; no adaptive/percentile-based
   clipping (e.g. Andrew et al. 2021) is implemented or compared. **Decision (2026-08-27,
   user-confirmed): deferred, stated as an explicit limitation.** State in the paper: the DP
   comparison is across *placement* (dp-fedavg/central/local — already a real, implemented,
   3-way comparison, see `docs/CaseStudies.md` §2.4.3), not across *clipping strategy*; adaptive
   clipping is future work. Note for whoever writes this: **do not describe Local-vs-Central DP as
   a missing comparison** — that one already exists and is exactly what the three DP-mode sweep
   produces; only the clipping-strategy axis is genuinely missing.
7. **No adaptive/auditor-aware attacker.** Addressed with a qualitative threat-model discussion,
   not an implementation — see `docs/ThreatModel.md`'s new "Adaptive / custom-tailored attacker"
   section (added 2026-08-27). Key distinction made explicit there: the current null result ("Yeom/
   Shadow/LiRA detect nothing") only rules out those three specific, literature-standard attacks —
   it does not rule out a bespoke attack tailored to this exact autoencoder/FedProx/ACN-Data setup,
   and the paper should say so rather than imply the absence of leakage is attack-agnostic. Also
   clarified: the Privacy Auditor is an observability/audit tool today, not an active blocking
   defense, so "evading the Auditor" has no operational consequence yet — a distinct, separately
   flagged future-work item from the MIA-attacker question.

   **Correction (2026-08-27, twice-refined) to a proposed Discussion paragraph**: a suggested
   framing argued that an adaptive attacker manipulating gradients to fool the Auditor "would be
   caught directly by the system's stability controls (`GRADIENT_EXPLOSION`, weight anomalies)."
   First correction proposed an "active attacker" middle ground (an attacker deviating from
   protocol to elicit more signal, plausibly caught by CUSUM/Krum/Cosine) — **user-caught as still
   wrong**: this project's Scenario 1 (the only one its results are about) defines clients as
   honest by definition, so *any* deviating/active attacker is actually Scenario 2 (malicious FL
   client, `docs/ThreatModel.md` §3.3, already explicitly "Future work") — there is no legitimate
   in-between "somewhat active but still Scenario 1" attacker to invoke.

   **Final, correct framing — state this as a formal dual-defense architecture with two orthogonal
   domains** (see `docs/ThreatModel.md` Scenario 5, S5b, for the full text):
   - **Integrity (`ByzantineDetector`: Krum, Cosine Distance, `GRADIENT_EXPLOSION`)** — Scenario 2,
     client-side, intercepts poisoned/backdoored updates before aggregation to protect
     convergence. Acts only on update scale/direction; **provides no privacy guarantee** against
     an honest-but-curious server, by design.
   - **Privacy (`PrivacyAuditor` + Yeom/Shadow/LiRA)** — Scenario 1, server-side passive observer.
     The only actual mitigation is Differential Privacy (clipping + Gaussian noise); the Auditor
     and PES_v1 *measure* risk, they do not *mitigate* it.
   Do not present `ByzantineDetector` as helping with privacy/MIA in any capacity, active or passive —
   it is a formally separate domain. S5a (bespoke attack) and S5b (Auditor evasion) both remain
   open, out-of-scope limitations; no existing module in this architecture should be cited as
   mitigating either.

8. **Positive control for the LiRA sanity-check: not found via natural-data escalation, but
   obtained via canary insertion (updated 2026-08-31; originally "no positive-control found",
   2026-08-28).** The main empirical finding (Sprint 10dd, statistically confirmed by the
   5-seed×10-config campaign, completed task #52, 2026-09-08) is a null result — LiRA AUC ≈ 0.50
   in every DP configuration tested (0.4995–0.5005 across all 10 configurations), in both simulation and the
   real deployment. A reasonable reviewer question is whether this reflects genuine absence of
   leakage or an attack/harness that simply cannot detect membership under any circumstance
   (Scenario B, `docs/CaseStudies.md` §2.4.3's own framing).

   **Phase 1 — natural-data escalation (2026-08-28): five independent factors varied on a no-DP
   baseline**, where memorization, if achievable at all, should be easiest to observe: local
   training epochs (50→1000), model capacity (~3.3× the 570-parameter baseline), feature
   informativeness (adding a near-unique per-session timestamp), the joint combination of capacity
   and features, and the training population (Caltech instead of Office 1). **In every one of the
   five configurations, composite LiRA AUC remained in the 0.48–0.54 range**, well below the ≥0.60
   threshold set beforehand as evidence of detectable memorization — see the full results table in
   `docs/TestRoadmap_DSN2027.md` #2. This phase alone was inconclusive about harness sensitivity:
   it rules out easy natural overfitting, but not an insensitive attack.

   **Phase 2 — canary insertion (2026-08-31, user-requested after reasonably pointing out that
   phase 1 alone cannot establish attack effectiveness without a known-vulnerable configuration):**
   5 real Office 1 sessions duplicated 30× each into that client's training set (a standard
   DP/MIA "canary insertion" technique — Carlini, *The Secret Sharer*, 2019; Jagielski et al.,
   *Auditing Differentially Private Machine Learning*, 2020), with 20 held-out sibling sessions
   from the same site as a clean member/non-member comparison group. **Result: the target model's
   raw reconstruction loss (no shadow calibration) shows an unambiguous, stable positive-control
   signal — AUC 0.70 / 0.90 / 0.85 across the three rounds, always in the same direction, always
   above the pre-declared 0.60 threshold** (canary members reconstruct 4–7× more accurately than
   canary non-members in every round). **This is the positive control**: the model and a
   loss-threshold detector both clearly pick up a deliberately engineered vulnerability. The
   *calibrated* LiRA score on the same canaries, by contrast, stays unstable and near chance
   (0.49 / 0.63 / 0.43) — the most likely explanation is that LiRA's shadow models are trained on
   the same shared population that now contains 150 duplicate canary copies, so their own random
   IN/OUT subsampling draws multiple near-identical copies regardless of which specific copy is
   nominally held out for a given shadow, diluting the differential signal LiRA's calibration
   relies on. This is a **methodological artifact specific to literal near-duplicate records in a
   population shared between target and shadow training** — natural EV sessions in this project's
   real data are never literal duplicates of one another, so this does not extend to, or
   undermine, the natural-data null result above.

   **Decision (2026-08-31, user-confirmed): treat this as the sanity-check's final, positive
   conclusion** rather than pursue further canary escalation (e.g., investigating the shadow-pool
   contamination mechanism directly was considered and not pursued, since the raw-loss result
   already settles the harness-sensitivity question).

   **How to write this in the paper**: this is a substantially stronger, more precise validation
   story than "no positive control found" — state both phases. Suggested phrasing for the
   Validation Methodology / Limitations section: *"To rule out an insensitive attack harness as an
   explanation for our null finding, we conducted a two-phase sanity check. First, we varied five
   natural-data factors (training epochs, model capacity, feature informativeness, their
   combination, and training population) on a no-DP baseline; composite LiRA AUC remained within a
   0.48–0.54 noise band in every configuration, inconclusive about harness sensitivity on its own.
   Second, we inserted synthetic canaries — five real sessions duplicated 30× into one client's
   training set, following standard DP/MIA canary-insertion methodology — and found that the
   target model's raw reconstruction loss cleanly and consistently distinguishes canary members
   from held-out canary non-members (AUC 0.70–0.90 across three rounds), confirming the harness
   and underlying model are sensitive to a genuine, deliberately engineered privacy violation. The
   calibrated LiRA score on the same canaries remained near chance, which we attribute to LiRA's
   shadow models being trained on the same population containing the duplicated canaries — a
   documented limitation of shadow-based calibration in the presence of literal near-duplicate
   records, not evidence against harness sensitivity in general, and not applicable to the natural,
   non-duplicated EV session data our main results are based on."* This framing turns what was a
   declared gap into a citable methodological contribution — consistent with this paper's overall
   positioning (see "Positioning statement" above) as a rigorous empirical measurement, not an
   overclaimed one.

   **Resolved 2026-09-02 (task #35 closed) — jpl replication completed and reconciled with caltech,
   confirming "dilution, scale-dependent," not "caltech-specific issue."** Caveat added 2026-09-01
   is superseded: a third site (`config/experiment_canary_positive_control_jpl.yaml`, same 150
   duplicated records, same method, run with the Sprint 10zz+16 shadow-group-sampling fix already
   active) was run to completion. Result: **canary AUC 0.5789/0.2632/0.6842 across the three rounds
   (composed 0.536842), raw-loss AUC 0.4737/0.4632/0.4947** — no stable positive-control signal,
   matching caltech's failure to reproduce (canary AUC 0.42–0.50, raw-loss AUC 0.35–0.44) far more
   than office1's clean, stable success (canary AUC 0.6375/0.6625/0.5875 post-fix, composed 0.6875;
   raw-loss AUC 0.66–0.90). The duplicated fraction at jpl is 150/27,053 ≈ 0.55% — essentially the
   same order of magnitude as caltech's ~0.5%, both roughly 20× more diluted than office1's ~11%.
   **Two independent large-site replications (caltech, ~31k sessions; jpl, ~27k sessions after
   injection) both fail to reproduce the canary signal in the same direction, while the one
   small-site run (office1, ~1.3k–1.7k sessions) succeeds cleanly** — this is now a real
   scale-dependence pattern, not a single anomalous site. **One additional honest observation from
   jpl**: the calibrated LiRA-on-canary score stays noisy and unstable across rounds (even dipping
   below chance in round 2, 0.2632) rather than converging near 0.5 smoothly — consistent with the
   Sprint 10zz+16 fix correctly reflecting an absent underlying signal (small `n_nonmember=19`
   amplifies round-to-round noise around a true near-chance value) rather than the fix failing or a
   contamination artifact re-appearing; it does not manufacture a spurious signal the way the
   pre-fix contaminated sampling did.

   **Corrected phrasing for the manuscript** (replaces the single-site-generalizing sentence in the
   suggested paragraph above): *"...when the duplicated fraction of the training population is
   large enough [~11% at office1]; two independent replications on sites roughly 20× more diluted
   (~0.5%, caltech and jpl) did not reproduce the signal in either case, consistent with a
   dilution-driven, scale-dependent effect rather than a site-specific anomaly — see README Sprint
   10zz+4 and Sprint 10zz+22b for the full account."* Documented in README (new Sprint entry);
   `docs/TestRoadmap_DSN2027.md` should also be checked and updated with this reconciled three-site
   table before the paper draft references it.

9. **The nominal ε does not correspond to a formal end-to-end (ε,δ)-DP guarantee — stated
   explicitly, not left implicit (2026-08-31, Fase 8).** Two distinct, honest caveats, both already
   present as comments in `src/ml/gradient_manager.py` before this pass and now surfaced here for
   the paper: (a) `GradientManager` implements **weight perturbation** (noise added to the
   post-training weight vector, once per round) — not **DP-SGD** (per-sample gradient clipping and
   noise during training). The formal (ε,δ)-DP guarantee of the Gaussian mechanism used to derive σ
   holds exactly only for `epochs=1`; with `epochs>1` per round (50 in the main campaign), the
   sensitivity of the final weight vector to a single training record is not formally bounded by
   `max_grad_norm` the way DP-SGD's per-step sensitivity is. (b) Under `dp_mode="central"`, the
   noise added to the aggregate must be calibrated to the sensitivity of a **weighted** mean
   (FedAvg weights by `n_samples`, and the three real sites are unevenly sized — Office 1 is an
   order of magnitude smaller than Caltech/JPL). Until this fix, `privatize_aggregate()` calibrated
   noise using `σ/n_participants` (correct only for a uniform, unweighted mean); the corrected
   sensitivity is `max_grad_norm × max_i(n_i/N)`, which is provably ≥ the old value whenever
   participants are unequally sized — meaning the noise added to already-published Central DP runs
   was slightly less than the stated ε formally requires. **Neither caveat threatens the paper's
   central empirical finding**: LiRA/Yeom/Shadow measure leakage from the AUC-ROC of the resulting
   model directly, independent of whether the noise calibration exactly matches a textbook DP proof
   — and more noise (the corrected formula's effect) can only push measured AUC closer to 0.5, never
   create a leakage signal that wasn't there. The correct framing for the paper, consistent with
   point 3's "measure, don't assume" reframe: report ε as *"a nominal Gaussian-mechanism noise
   parameter calibrated per round, not a proven formal (ε,δ)-DP guarantee for the full multi-epoch,
   weighted-aggregation training procedure actually used"* — never as a formal guarantee on its own.

10. **Non-member independence in the train/holdout split — now testable, not previously
    addressed (2026-08-31, Fase 8).** The default split (`random.shuffle` over all sessions from
    all sites/years pooled together, then 80/20) can place two sessions from the *same* EVSE
    station in train and holdout respectively — a reasonable reviewer objection is that the
    "non-members" are then not fully independent of the training distribution, since they may share
    station-level behavioral patterns with members. A new opt-in split strategy,
    `entity_aware_split()` (`scripts/run_experiments.py`, `config/experiment_robustness_entity_split.yaml`),
    groups sessions by station (`node_id`) before splitting, so no station appears on both sides —
    closing the objection by construction. This is a **robustness experiment, not a replacement**:
    it changes which specific sessions are members/non-members, so it is not directly comparable to
    the 5-seed×10-config campaign (which used the default random split) — the default is unchanged,
    and every already-published result remains valid and reproducible as-is.

    **Run to completion 2026-08-31** (`experiment_20260831_151156.json`, no-DP, 3 sites, 10 rounds,
    seed=42, n_shadow=16, 115 station-entities, train=53276/holdout=13437 — the intended 80/20 split
    held at the entity level, confirming `entity_aware_split()` groups correctly): **LiRA composite
    AUC-ROC = 0.4983** (gap=0.034890, n=26860, TPR@1%FPR=0.0083), mean per-round LiRA AUC 0.4998,
    Yeom mean AUC 0.4989 — chance level, consistent with every other configuration in the main
    5-seed×10-config campaign (0.4995–0.5005). The null-leakage finding holds under a split that
    guarantees non-members share no EVSE station with any training session, closing the
    non-independence objection empirically, not just architecturally.

    **5-seed replication completed 2026-09-01** (`make experiment-entity-split-sweep`,
    `experiments/entity-split-sweep1/`, seeds={42,123,456,789,1234}, all 5 succeeded): **LiRA mean
    AUC-ROC = 0.5015 ± 0.0017** (per-seed: 0.4998/0.5009/0.5001/0.5045/0.5025) — tightly clustered
    around chance, confirming the single-seed point (0.4983) was not an artifact. Yeom shows more
    seed-to-seed variance (mean 0.5088 ± 0.0191, range 0.4869–0.5396) but LiRA — the primary,
    stronger attack — is the one that matters for the central claim, and it is the tightest result
    of the two. As predicted, this is a confirmatory result, not a new finding: the null-leakage
    conclusion holds under entity-aware splitting with the same statistical rigor as the main
    5-seed×10-config campaign.

11. **Capacity vs. FL-as-regularizer — a reviewer-style question, now answered with a real
    control experiment (2026-09-01).** A reasonable reviewer objection to the main null result: is
    the absence of leakage because the model's capacity is too limited to memorize at all, or
    because the federated architecture itself acts as a natural regularizer that suppresses
    memorization that a centralized model of the same capacity *would* exhibit? The five natural-data
    escalation experiments (point 8, Phase 1) rule out "capacity alone" (varying epochs, parameter
    count, and feature informativeness never moved AUC out of the 0.48–0.54 band), but every one of
    those runs was still a standard federated run — none isolated the FEDERATION variable itself.
    New `run_centralized_control()` (`scripts/run_experiments.py`, `--centralized-control`) trains a
    single model on the same pooled data, same architecture, same *total* epoch budget as the
    federated run (rounds × local_epochs — no advantage from extra training), but centralized: no
    per-cluster partitioning, no FedAvg, no aggregation step at all. Scored with the same Yeom-style
    loss-based AUC on the same train/holdout split, for a direct, matched-budget comparison.

    **Full run completed 2026-09-01** (10 rounds × 50 epochs = 500 total epochs, no-DP, n_shadow=8,
    `_centralized_control_full`): **centralized-control AUC-ROC = 0.5003** (members=53370,
    non-members=13343) — against the federated LiRA composite AUC-ROC = 0.5030 (n=25759) and Yeom
    mean AUC = 0.4981 from the *same* run. All three numbers sit inside the same 0.48–0.54 chance
    band already established for the federated-only escalation experiments. **This directly answers
    the reviewer question**: a centralized model — with zero FL regularization, zero decentralization
    effect, trained on strictly more information (the full pooled dataset, not a per-client shard) —
    shows the *same* absence of memorization as the federated model at matched epoch budget. This is
    evidence for Scenario B (`docs/CaseStudies.md` §2.4.3: the architecture/capacity genuinely does
    not memorize at this training budget) rather than "FL suppresses a leakage signal that would
    otherwise be present" — because removing FL entirely does not surface any signal.

    **Multi-seed campaign COMPLETED (2026-09-02)** — the caveat above is resolved. Five seeds
    (42, 123, 456, 789, 1234), same full budget (10 rounds × 50 epochs), `experiments/_centralized_control_sweep1/`:

    | seed | centralized-control AUC | federated LiRA AUC (same run) | Yeom AUC (same run) |
    |---|---|---|---|
    | 42 | 0.500343 | 0.500847 | 0.498056 |
    | 123 | 0.499164 | 0.501103 | 0.497769 |
    | 456 | 0.502592 | 0.499178 | 0.501232 |
    | 789 | 0.495145 | 0.500630 | 0.503756 |
    | 1234 | 0.498330 | 0.501342 | 0.500496 |

    **Bootstrap statistics (centralized-control AUC, 10000 resamples)**: mean = 0.499115,
    std = 0.002738, 95% CI = **[0.496988, 0.501221]** — contains 0.5. Same conclusion as the
    single-seed result, now with the identical statistical rigor (5-seed bootstrap CI) already
    applied to the main campaign and the entity-split replication (point 10): a centralized model
    shows no detectable memorization at matched epoch budget, in every seed, not just one. This is
    now a fully citable result for the reviewer question above — **caveat resolved, no longer a
    single-seed limitation.**
