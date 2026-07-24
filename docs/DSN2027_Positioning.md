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

> Federated Learning (FL) is increasingly proposed as the privacy-preserving path for training
> shared models over critical infrastructure — smart grids, EV charging networks, industrial
> control systems — where centralising raw operational data is both a regulatory liability and a
> security risk. Whether FL actually delivers on that promise in realistic, production-style
> deployments is an empirical question that current research answers piecemeal: one paper per
> attack, one dataset per paper, rarely a production FL framework, rarely an industrial dataset.
> We present **ChargeShield-FL**, an open, reproducible benchmark for measuring how much privacy
> leakage survives standard FL mitigations in a realistic critical-infrastructure setting, using
> real EV charging session data (ACN-Data, 3 sites: Caltech, JPL, Office 1) in a multi-site FL
> simulation with the real differential-privacy, Byzantine-robust-aggregation, and
> intrusion-detection code paths this project implements — not synthetic data standing in for any
> of the three. (A companion NVFLARE job/app implementing the same pipeline on a genuine
> multi-container federation exists as a scaffold and is planned validation work, not a claim of
> this paper — see "Current validation status" below.) Privacy attacks (membership inference
> today; gradient inversion and property inference as planned extensions) are implemented as
> pluggable modules against a common measurement harness, so the benchmark's value compounds with
> each new attack or defence added, rather than resetting with every new paper. Our first results,
> using a likelihood-ratio membership inference attack (LiRA), show [result — to be finalised once
> Central DP + multi-seed results land]: differential privacy's nominal guarantee (ε) does not
> reliably predict the empirical protection an attacker experiences, and the gap between the two
> is largest under exactly the DP placement (central, aggregate-only noising) that many production
> FL deployments favour for utility reasons.

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
