# NVFLARE / Containerlab Integration — Status and Plan

**Started:** 2026-07-22
**Status:** Skeleton only — job scaffold + client Executor written, **not executed, not tested**.
**Why:** the environment used to write this code cannot install `torch` (proxy blocks `download.pytorch.org`) or, by extension, verify `nvflare` behaviour (nvflare depends on torch). Every NVFLARE API call below was written from documented/standard NVFLARE 2.x patterns and careful reading of the existing `src/ml/`/`src/auditor/`/`src/ids/` code, but **none of it has run**. Treat this as a first draft to debug on a machine with the real dependencies installed, not as working code.

This document exists because the prior state of the repo's Containerlab/NVFLARE scaffolding was audited (2026-07-21, see `docs/CaseStudies.md` §2.4.3's "the privacy pipeline does not run on the containerised network" limitation) and found to be unused: `src/flare/flare_connector.py` is an explicit Sprint-3 placeholder that never imports `nvflare` and simulates gradients with `random.gauss()`; `nvflare/project.yml` only provisions PKI/network participants, no job/app existed; the `docker/` Dockerfiles are orphaned (unreferenced, and their `CMD`s have no `if __name__ == "__main__"` guard, so they'd crash on start). This document and the files under `nvflare/jobs/chargeshield_poc/` are the first concrete step toward closing that gap — not a completed integration.

## What exists now

```
nvflare/jobs/chargeshield_poc/
  meta.json                          — job metadata, deploy_map (single "app" to all sites)
  app/config/config_fed_server.json  — NVFLARE built-in ScatterAndGather workflow
  app/config/config_fed_client.json  — points at the custom Executor below
  app/custom/chargeshield_executor.py — wraps AutoencoderTrainer.train_local()
```

**Server side** (`config_fed_server.json`) uses NVFLARE's *built-in* components — `ScatterAndGather` workflow, `InTimeAccumulateWeightedAggregator`, `FullModelShareableGenerator`, `PTFileModelPersistor` pointed at `core.autoencoder.Autoencoder` — deliberately, **not** a custom controller wrapping `FedAvgAggregator`/`PrivacyAuditor`/`ChargingIDS` yet. The goal of this first slice is only to validate that a model round-trips client → server → client through NVFLARE's transport layer with 4 real client processes, before adding any of ChargeShield-FL's own logic server-side. This mirrors the "simplest possible round-trip first" recommendation from the infrastructure audit.

**Client side** (`chargeshield_executor.py`) reuses the *real, already-tested* `AutoencoderTrainer` class from `src/ml/autoencoder_trainer.py` — the same class `scripts/run_experiments.py::run_fl_rounds()` uses in single-process simulation. It does **not** reimplement training logic. What it does: on `START_RUN`, instantiate `AutoencoderTrainer` and load this client's slice of sessions (same contiguous per-cluster split as the simulation, for now — see limitations below); on each `train` task, convert NVFLARE's incoming `Shareable`/`DXO` into the weight list `AutoencoderTrainer.set_weights()`/`apply_global_model()` expects, call `train_local()`, and convert the resulting `GradientUpdate` back into an outgoing `DXO`.

## What is explicitly NOT done yet

- **No DP.** `GradientManager.clip_only()`/`privatize()`/`privatize_aggregate()` (today's central/local DP work, see README Engineering Fixes 2026-07-22) are not called anywhere in the Executor. Phase 1 is transport-only.
- **No IDS/audit.** `PrivacyAuditor.audit()` and `ChargingIDS.analyze_round()` are not wired in. The server uses NVFLARE's built-in aggregator, which has no hook for this.
- **No LiRA/Shadow/Yeom attacks.** Those still only run against the single-process simulation's `fl_results` dict, in `scripts/run_experiments.py`. Extracting per-client raw updates from inside NVFLARE's server-side aggregation (needed for LiRA and Shadow) is unsolved — NVFLARE's normal flow doesn't expose this by default; a custom aggregator/controller would need to capture updates before calling `InTimeAccumulateWeightedAggregator`'s equivalent logic.
- **Not on Containerlab.** This targets NVFLARE's own simulator/POC mode (`nvflare simulator`, in-process or local multi-process), not the `topology.clab.yml` containerised network. Docker/Containerlab wiring is a separate, later step (see Open Items below).
- **Per-client dataset access is fake.** Every client currently loads the *same* shared JSON file and takes a slice by index — not representative of a real deployment where each station already holds only its own data. Fine for validating transport; not fine for any claim about "real per-site data isolation."

## Points marked `VERIFY:` in the code — check these first

Search `chargeshield_executor.py` for `VERIFY:`. The three riskiest assumptions:
1. **DXO data format**: assumed `dxo.data` is `dict[str, np.ndarray]` keyed by the exact `state_dict()` key names `AutoencoderTrainer.get_weight_keys()` returns. If `FullModelShareableGenerator` + `PTFileModelPersistor` produce a different structure (e.g., wrapped in another dict, or numpy vs. tensor), the conversion in `execute()` breaks immediately and loudly (KeyError) — easy to spot, not a silent-corruption risk.
2. **Aggregation weighting**: the outgoing DXO puts `n_samples` in `meta`, but the exact meta key `InTimeAccumulateWeightedAggregator` reads for per-client weighting in NVFLARE 2.7.2 needs confirming against `nvflare.apis.fl_constant.MetaKey` — if wrong, aggregation would silently fall back to unweighted averaging (wrong, but not a crash — the more dangerous failure mode of the three).
3. **Round number**: the Executor counts rounds locally (`self._round_num += 1` on every `execute()` call) instead of reading the authoritative round from `fl_ctx`. Almost certainly wrong in any scenario with retries or non-trivial task dispatch; low risk for a straight-through single-task-per-round POC, but flagged for correctness.

## Suggested next steps (in order)

1. Install `nvflare==2.7.2` + `torch` in a real environment (not this sandbox). Run `nvflare simulator` against `nvflare/jobs/chargeshield_poc/` with `-n 4` (or fewer, for a first smoke test — even `-n 1` validates the transport contract before scaling to 4).
2. Fix whatever the three `VERIFY:` points above turn out to need — expect at least one to need a real fix, not just confirmation.
3. Once the round-trip works and the global model loss decreases across rounds (same sanity check `scripts/run_experiments.py` already does in simulation), add DP: call `GradientManager.privatize()`/`clip_only()` inside `execute()` before returning the outgoing DXO, matching the `dp_mode` semantics already implemented in the simulation path.
4. Write a custom server-side Controller/Aggregator that wraps `FedAvgAggregator` (already tested, already used in simulation) instead of `InTimeAccumulateWeightedAggregator`, and that also runs `PrivacyAuditor.audit()`/`ChargingIDS.analyze_round()` per round the same way `scripts/run_experiments.py::run_ids()` does today.
5. Solve raw-update extraction for LiRA/Shadow (the hardest unsolved piece) — likely requires a custom aggregator that stashes each client's raw DXO before combining them, since that's exactly the "semi-honest aggregator" threat model LiRA already assumes in the simulation (see `run_lira()` docstring).
6. Only after 1-5 work against the NVFLARE simulator: wire this same job into `topology.clab.yml`/Docker — fix the orphaned `docker/charging-node` build, point the topology at whatever image actually gets built, reconcile the two PKI trees (`certs/` vs. the NVFLARE-provisioned workspace).

Steps 1-2 are a few hours of real debugging once someone has `nvflare` installed. Steps 3-6 are the "multi-week" part of the original estimate — this document doesn't shrink that estimate, it just gives it a concrete starting point.
