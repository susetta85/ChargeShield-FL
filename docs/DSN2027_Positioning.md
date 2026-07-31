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
   >  **Accuracy note on "activated by the ML Plane" (2026-07-31):** today this is true at the
   >  *data-flow* level, not yet at the *code* level. `run_ids()` (simulation) and
   >  `ChargeShieldAggregator`'s equivalent logic (NVFLARE) call `PrivacyAuditor.audit()`
   >  imperatively, once per node per round, immediately after aggregation — reading the raw
   >  updates the ML Plane's `FLArtifactCollector` already assembled that round. `PrivacyAuditor`
   >  itself does **not** subclass `MLPlaneListener` and is not registered via
   >  `MLPlane.subscribe()`, unlike `FLArtifactCollector`. So the honest claim is: *the Privacy
   >  Auditor's input is entirely sourced from what the ML Plane observes*, not yet *the Privacy
   >  Auditor is a live ML Plane subscriber*. Making it a literal subscriber (so `audit()` fires
   >  from an event callback instead of a loop in `run_ids()`) is a small, well-scoped refactor —
   >  not done in this pass, flagged here so the paper's architecture diagram doesn't overstate the
   >  current wiring.
3. **Empirical goal these two components exist to demonstrate:** that differential privacy applied
   to the FL *communication channel* (DP-FedAvg noises each client's clipped update before it
   leaves the client; Central DP noises only the server-side aggregate) does not eliminate the
   *privacy leakage* the Privacy Auditor and the LiRA case-study attack can both detect from the
   raw per-client updates a semi-honest aggregator can observe before that noising happens (for
   DP-FedAvg, after clipping but this is what the client actually sends; for Central DP, the raw
   per-client contribution before the single aggregate noise draw). In other words: **protecting
   the channel is not the same as protecting the computation.** The current Central DP result
   (LiRA AUC 0.743 at ε=1.0, 0.812 at ε=0.1 — see `docs/PrivacyExposureScore_v1.md`) is exactly this
   phenomenon measured, not a LiRA-specific curiosity: it is the framework (ML Plane feeding both
   the Privacy Auditor and the case-study attacks a consistent view of the raw update) doing its
   job of exposing a leakage channel that a purely ε-based compliance check would miss entirely.
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
     math); `ChargingIDS`'s Krum/CUSUM/Cosine-Similarity core (`src/ids/charging_ids.py` — Byzantine-
     robust aggregation checks that apply to any FL client population, despite the EV-flavoured
     class name).
   - **Domain-specific today — what an adopter targeting a different scenario would replace:**
     `ACNDataset` (`src/adapters/acn_dataset.py`, EV-session-schema-specific), the `Autoencoder`
     architecture and its 6 input features (`src/core/autoencoder.py`, tuned to EV session numeric
     fields), the `sites`/cluster naming in `config/experiment.yaml`, and the OCPP/MQTT protocol
     adapters (`src/adapters/ocpp16_adapter.py` etc. — themselves confirmed dead/unwired scaffolding
     as of Sprint 10d, not a real dependency of the current pipeline).
   - The class name `ChargingIDS` is itself slightly misleading for the "generic framework" claim —
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
> still leaks?** (A companion NVFLARE job/app implementing the same pipeline on a genuine
> multi-container federation exists as a scaffold and is planned validation work, not a claim of
> this paper — see "Current validation status" below.) Our first results, using the LiRA case-study
> attack, show [result — to be finalised once Central DP + multi-seed results land]: differential
> privacy's nominal guarantee (ε) does not reliably predict the empirical protection an attacker
> experiences, and the gap between the two is largest under exactly the DP placement (central,
> aggregate-only noising) that many production FL deployments favour for utility reasons — evidence
> that channel-level DP and computation-level privacy are not the same guarantee, which is the
> architectural point the ML Plane and Privacy Auditor exist to make measurable in the first place.

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
  full account); confirmation of a fully successful multi-round re-run is still pending as of this
  writing. What has **not** changed: this NVFLARE execution has not yet produced any
  privacy-measurement number used anywhere in this project — it validates the infrastructure
  (transport, DP wiring, IDS/audit export), not a new source of LiRA/Yeom/Shadow AUC data. The
  Containerlab/Docker multi-container deployment (the stronger claim the original overclaimed
  sentence was really describing) genuinely has not been attempted — `containerlab/topology.clab.yml`
  remains stale/unrewritten (see that document's own note) and is deliberately deferred until after
  the simulator-level job is fully validated.
- DP and Byzantine/IDS are **not** measured together in one privacy-leakage run today:
  `run_experiments.py` explicitly skips FedMIA/Shadow/LiRA whenever `byzantine_attack.enabled` is
  true (Byzantine sweeps validate Krum detection only, in `experiments/ids_validation/`, and are
  a separate, non-privacy measurement — see the Makefile's `experiment-byzantine-sweep` comment
  block). So "DP + Byzantine + IDS active simultaneously while measuring MIA" is not a result this
  project has produced.

The honest version of the claim: real dataset, real DP mechanism code, real attack code, all in a
validated single-process simulation, now including a real Central DP result (see
`docs/PrivacyExposureScore_v1.md`); a matching NVFLARE job for the same pipeline has now run once
(simulator mode only) and found/fixed real bugs, but has not yet produced a privacy-measurement
result of its own, and the full multi-container Containerlab deployment remains future/ongoing work
rather than part of the DSN 2027 results. The paper should say exactly that, not the stronger claim
above.

(The bracketed sentence is intentionally left open — it should be filled in once the Central DP
+ multi-seed results are in, so the abstract states a real number rather than an anticipated
one.)

## Reframed contribution list

> **Note (2026-07-31):** read this list alongside "The actual contribution: ML Plane + Privacy
> Auditor" above, not instead of it. Items 1, 3, and 4 below are genuine supporting contributions.
> Item 2 (the pluggable attack interface, and by extension Yeom/Shadow/LiRA themselves) is the
> **case-study validation layer** for the ML Plane + Privacy Auditor architecture, not a
> stand-alone contribution — kept in this list because "pluggable" is itself a real, useful
> property of the harness, but the paper should not present "we made the attacks pluggable" as
> equal in weight to "we built a reusable FL privacy-observability architecture and used it to show
> DP-on-the-channel doesn't stop leakage."

Where the old framing had one contribution ("we show LiRA beats DP"), the benchmark framing
supports a list that keeps growing:

1. **A reproducible measurement harness** for FL privacy leakage — a real multi-site industrial
   dataset (ACN-Data, 3 real sites) and real DP/IDS/aggregation code, validated today in a
   single-process simulation (not synthetic data standing in for any of the three). A matching
   NVFLARE job/app implementing the same pipeline on a genuine multi-container federation is
   designed and scaffolded as the next validation step — see "Current validation status" below;
   it is planned work, not a current claim of this paper. (Corrected 2026-07-24: this bullet
   originally claimed "real NVFLARE federation... not a simulated stand-in," directly
   contradicting the validation-status section below — same overclaim caught by the same
   independent review, missed in this second location on the first pass.)
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
   IID-shuffled proxy for it.
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
