# Literature Review — First Pass (not PRISMA-systematic)

Status: **draft, first pass, ongoing.** Owner task: #65 — kept permanently open, update
incrementally as more categories/papers are added in future sessions.

## Scope and honesty caveats

This is a first solid pass across the 10 requested categories, built from targeted web searches
(one search per category, run 2026-07-24) rather than a systematic PRISMA-style review (no
exhaustive database query across ACM DL/IEEE Xplore/Scopus, no dual-reviewer screening, no
inclusion/exclusion protocol). Entries below are based on abstracts and search-result snippets,
**not full-text reads** — venue/dataset/defense details are as accurate as the snippet allowed;
where a detail could not be confirmed from the snippet it is marked `unconfirmed`. Treat this as
a solid starting bibliography and gap-map for the related-work section, to be tightened (full-text
verification, additional per-category papers, actual PRISMA protocol if a systematic review is
later required) in follow-up passes.

Per-paper fields, as requested: venue, year, datasets, attack, defenses, limitations, realistic
deployment considered?, NVIDIA FLARE / production FL framework used?, industrial dataset used?,
extension opportunity for ChargeShield-FL.

---

## 1. Membership Inference Attacks in Federated Learning

| Paper | Venue/Year | Datasets | Attack | Defenses discussed | Limitations | Realistic deployment? | Production FW (FLARE etc.)? | Industrial dataset? | ChargeShield-FL extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| "Membership Inference Attacks and Defenses in Federated Learning: A Survey" | ACM Computing Surveys, Dec 2024 (arXiv:2412.06157) | Survey — spans multiple benchmark datasets used by surveyed papers, `unconfirmed` per-paper | Survey of MIA taxonomy (output-based, gradient-based, shadow-model) | Surveys DP, regularization, knowledge distillation defenses | Explicitly notes most prior FL-MIA work uses toy/benchmark datasets, not production frameworks | Not itself an empirical study | No | No | Directly citable to justify our "few papers use production framework + industrial data" gap claim |
| "Perfectly Accurate Membership Inference by a Dishonest Central Server in FL" | arXiv:2203.16463, 2022 | `unconfirmed`, likely CV benchmarks | Dishonest-server MIA exploiting per-client updates before aggregation | Discusses secure aggregation as mitigation | Threat model closest to our own (honest-but-curious/dishonest server attacking raw pre-aggregation updates) but on benchmark data | No | No | No | Closest prior threat-model precedent for our LiRA-on-raw-update design — worth citing to establish the threat model isn't novel, only the deployment context is |
| "The Federation Strikes Back: A Survey of FL Privacy Attacks, Defenses, Applications, and Policy Landscape" | arXiv:2405.03636, 2024 | Survey | Broad attack survey incl. MIA | Broad defense survey + policy/regulatory angle | Breadth over depth; policy section is useful for our "regulatory pressure" motivation paragraph | Discusses applications generally | `unconfirmed` | `unconfirmed` | Good source for the regulatory-motivation citations already used in our README (GDPR/AFIR/NEVI) |
| CS-MIA (confidence-series MIA) | ScienceDirect, `unconfirmed` year | `unconfirmed` | Uses prediction-confidence *time series* across FL rounds rather than a single snapshot | `unconfirmed` | Needs full-text check — potentially relevant alternative to our per-round AUC tracking design | `unconfirmed` | No | No | Compare against our per-round LiRA tracking methodology — may suggest an additional per-round feature |

**Note (Carlini et al. 2022, LiRA)** — the paper this project's primary attack is built on — did not
surface as a top hit for this particular query (search engines favor recent 2024–2026 work); it
is already correctly cited in the project's own README/code and does not need re-discovery here.

---

## 2. Gradient Inversion Attacks

| Paper | Venue/Year | Datasets | Attack | Defenses discussed | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | ChargeShield-FL extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| DLG — "Deep Leakage from Gradients" (Zhu et al.) | NeurIPS 2019 (landmark, found via secondary citation, not top search hit directly) | MNIST/CIFAR-style image benchmarks | Optimization-based gradient matching to reconstruct input+label pixel-for-pixel | Gradient pruning/compression discussed as mitigation | Only works well on small batches, image-domain assumptions | No | No | No | Foundational reference; our autoencoder's small feature vector (6 features) is a very different attack surface than DLG's image gradients — worth an explicit note in the paper that GI against tabular/low-dim EV features is an open methodological question |
| iDLG (Zhao et al.) | arXiv, 2020 | Image benchmarks | Analytic (not iterative) ground-truth label extraction from last-layer gradients | — | Same image-domain limitation as DLG | No | No | No | Same as above; label-extraction trick may not transfer directly since our autoencoder has no class labels — relevant instead to any future supervised module |
| "A Survey on Gradient Inversion: Attacks, Defenses and Future Directions" | IJCAI 2022 | Survey | Taxonomy: optimization-based (OP-GIA), generation-based (GEN-GIA), analytics-based (ANA-GIA) | Surveys DP, secure aggregation, gradient compression as defenses | Survey, not empirical | Discusses gaps in realistic evaluation | `unconfirmed` | `unconfirmed` | Good taxonomy reference for structuring our own future GI module design (Task #64 depends on this) |
| "Exploring the Vulnerabilities of FL: A Deep Dive into GIA" | arXiv:2503.11514, 2026 | `unconfirmed` | Systematic empirical comparison of GIA methods and their limiting factors | `unconfirmed` | Notes prior GIA literature lacks extensive experiments on limiting factors — directly supports our "no rigorous benchmark" gap claim | `unconfirmed` | `unconfirmed` | `unconfirmed` | Most directly relevant recent paper to read in full before starting our own GI module (Task #64) |

---

## 3. Property Inference Attacks

| Paper | Venue/Year | Datasets | Attack | Defenses | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| "Poisoning-Assisted Property Inference Attack Against FL" | IEEE (journal), 2022 | `unconfirmed` | Combines poisoning + property inference to amplify signal | `unconfirmed` | Assumes active/malicious client capability, stronger than our honest-but-curious model | No | No | No | Could motivate an active-attacker variant of our threat model as future work |
| PPA — "Preference Profiling Attack Against FL" | arXiv:2202.04856, 2022 | `unconfirmed` | Infers user preference/property from FL updates | `unconfirmed` | `unconfirmed` | `unconfirmed` | No | No | Directly analogous to inferring EV owner routine/preference (charging time patterns) — good motivating citation for why `hour_of_day`/`duration_hours` are "sensitive" features in our PES v2 definition (Task #64) |
| "Survey on FL Threats: concepts, taxonomy on attacks and defences" | arXiv:2201.08135, 2022 | Survey | Broad taxonomy incl. property inference | Broad | General survey | `unconfirmed` | `unconfirmed` | `unconfirmed` | General taxonomy citation for related-work section |

---

## 4. Label Leakage Attacks

| Paper | Venue/Year | Datasets | Attack | Defenses | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| LLG — "User-Level Label Leakage from Gradients in FL" | PoPETs 2022 | `unconfirmed`, classification benchmarks | Extracts ground-truth labels from last-layer gradient direction/magnitude | Gradient compression, encrypting final-layer gradients | Assumes classification with mini-batch SGD; **does not apply directly to our unsupervised autoencoder** (no labels) | No | No | No | Not directly applicable to ChargeShield-FL's unsupervised setting — worth an explicit scoping note in related work explaining why label leakage is out of scope for an autoencoder-based FL pipeline |
| "Breaking Secure Aggregation: Label Leakage from Aggregated Gradients in FL" | arXiv:2406.15731, 2024 | `unconfirmed` | Label leakage surviving secure aggregation | Discusses secure aggregation as (insufficient) defense | Shows secure aggregation alone is not sufficient — relevant precedent for our own "DP alone is not sufficient" thesis | `unconfirmed` | `unconfirmed` | `unconfirmed` | Strong rhetorical parallel to cite: "leakage surviving a nominal defense" is the same shape of result as ours, for a different attack/defense pair |

---

## 5. Differential Privacy in Federated Learning

| Paper | Venue/Year | Datasets | Attack/mechanism | Defenses | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| McMahan et al., DP-FedAvg / DP-FedSGD | ICLR 2018 (landmark, confirmed via snippet reference, not top hit directly) | Language modeling benchmarks (unconfirmed exact dataset) | N/A (defense paper) | Proposes DP-FedAvg (Algorithm 1): client-side clip, single server-side aggregate noise draw — NOT per-client clip+noise before aggregation | Full-text read (2026-08-14) confirmed the original formulation is exactly our **`central`** mode, not our `dp-fedavg` mode (which noises per-client before aggregation — a stricter, non-standard variant not literally in this paper) — foundational citation for `central`, already implicitly used in our architecture | No | No | No | This is literally the mechanism our `central` mode implements — must be cited as the source of that placement; `dp-fedavg`/`local` need a different citation basis (see `docs/ReadingList_DSN2027.md`'s 2026-08-14 entry) |
| "A Systematic Survey for Differential Privacy Techniques in FL" | SCIRP, `unconfirmed` year (recent) | Survey | Survey of DP mechanisms in FL | Surveys central vs local DP placements | Survey, not empirical | `unconfirmed` | `unconfirmed` | `unconfirmed` | Good source for structuring our own "three DP placements tested" table (already in the slide deck, slide 18) |
| "Convergence Analysis for Differentially Private Federated Averaging in Heterogeneous Settings" | MDPI, 2025 | Theoretical | N/A | Theoretical convergence bounds under DP-FedAvg with non-IID data | Theoretical only, no empirical attack evaluation | No | No | No | Relevant to explain *why* our non-IID 3-site setup (Caltech/JPL/Office1, very different session counts) affects DP-FedAvg convergence/utility, independent of the privacy question |

---

## 6. Secure Aggregation

| Paper | Venue/Year | Datasets | Mechanism | Defenses(=itself) | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving ML" | ACM CCS 2017 (landmark) | N/A (systems paper) | Pairwise masking + Shamir secret sharing + symmetric encryption | Is the defense itself | Tolerates up to 1/3 dropout; communication-efficient for high-dim updates. Not itself evaluated against MIA/GIA in the original paper | Systems-realistic (designed for real mobile-scale deployment) | No (predates NVFLARE) | No | ChargeShield-FL does **not** currently implement secure aggregation — worth flagging explicitly as an *absent* defense in our threat model discussion: our honest-but-curious server sees raw pre-aggregation updates precisely because secure aggregation is not in the pipeline; adding it would be a natural way to test whether it blocks LiRA where DP alone does not |
| "A Survey on Secure Aggregation for Privacy-Preserving FL" | Springer, 2024 | Survey | Survey | Survey | Survey | `unconfirmed` | `unconfirmed` | `unconfirmed` | General citation |
| "Breaking Secure Aggregation: Label Leakage..." (also in §4) | arXiv:2406.15731, 2024 | `unconfirmed` | Shows label leakage survives SecAgg | — | Cross-cutting with label leakage category | `unconfirmed` | `unconfirmed` | `unconfirmed` | Same paper as §4 — supports the "SecAgg is not a silver bullet either" framing if we ever add it |

---

## 7. Federated Learning for EV Charging Infrastructure

| Paper | Venue/Year | Datasets | Focus | Defenses | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| "Federated Learning for Early Prediction of EV Charging Demand" | 2025 (ResearchGate preprint) | `unconfirmed`, likely a public charging dataset | Demand prediction, not privacy attack | None (utility paper) | No privacy/attack evaluation at all — pure utility application | `unconfirmed` | `unconfirmed` | Possibly, `unconfirmed` | Confirms our "EV+FL utility papers exist, EV+FL+privacy-attack papers are rare" gap claim |
| "Anomaly Detection in EV Charging Stations Using Federated Learning" | arXiv:2509.18126, 2025 | `unconfirmed` | FL for anomaly/cyberattack detection at charging stations | None specific to MIA/GIA | Security-focused (intrusion/anomaly), not privacy-leakage-focused — different threat model than ours | `unconfirmed` | `unconfirmed` | `unconfirmed` | Complementary rather than overlapping — worth citing to show our IDS layer (CUSUM/Krum/Cosine) sits in the same problem space as this line of work, but ours additionally measures MIA leakage, which this paper does not |
| "Fuse and Federate: Enhancing EV Charging Station Security with Multimodal Fusion and FL" | arXiv:2506.06730, 2025 | `unconfirmed` | Multimodal security fusion + FL | `unconfirmed` | Security fusion, not membership-inference leakage | `unconfirmed` | `unconfirmed` | `unconfirmed` | Same gap: no privacy-attack evaluation in the EV+FL literature found so far |
| "Privacy-preserving collaborative battery fault warning...via heterogeneous data from charging stations" | Nature Communications, 2025 | Real multi-site charging data (`unconfirmed` exact sites) | Federated battery fault detection | Privacy-preserving framing, mechanism `unconfirmed` from snippet | Needs full-text check for whether it evaluates any MIA/GIA attack against its own privacy claims (many "privacy-preserving" papers do not attack-test their own claim) | Possibly, needs check | `unconfirmed` | Likely yes (real charging data) | **Most important gap check in this category**: if this Nature Comms paper claims "privacy-preserving" without an adversarial evaluation, it is exactly the kind of unverified privacy claim our benchmark framing argues against — worth reading in full |

**Category-level finding**: across this search, no EV-charging FL paper was found that combines
(a) a real multi-operator dataset, (b) a production-grade FL framework, and (c) an adversarial
MIA/GIA evaluation of its own privacy claim, in one study. This is the strongest single piece of
evidence so far for the "no benchmark exists" claim in `docs/DSN2027_Positioning.md` — but it
rests on one search pass and should be stress-tested with more targeted queries before the paper
asserts it as a confirmed gap.

---

## 8. Federated Learning for Smart Grid

| Paper | Venue/Year | Datasets | Focus | Defenses | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| "Federated Learning for Smart Grid: A Survey on Applications and Potential Vulnerabilities" | ACM TCPS (arXiv:2409.10764), 2024/2025 | Survey across generation/transmission/consumption stages | Surveys FL applications + vulnerabilities in SG | Surveys SSL/TLS, generic FL defenses | Explicitly notes the gap between FL research and *practical* SG deployment — directly citable | Discusses this gap explicitly | `unconfirmed` | `unconfirmed` | This survey's own stated research-practice gap is a strong secondary citation alongside our EV-specific gap claim |
| "Federated intelligence for smart grids: security and privacy strategies" | J. Electrical Systems and Info. Tech., 2025 | Survey | Security/privacy strategy review | Broad | Survey | `unconfirmed` | `unconfirmed` | `unconfirmed` | General citation |
| FUSE (FL + split learning for electricity theft detection) | `unconfirmed` venue, referenced within the above survey | Smart meter data | Electricity theft / anomaly detection | Split learning as a privacy mechanism | Utility-focused, not adversarially evaluated for MIA per the snippet | `unconfirmed` | `unconfirmed` | Likely (smart meter data) | Comparable "adjacent critical infrastructure, FL, real-ish data, no adversarial privacy evaluation" pattern — reinforces the cross-domain gap claim |

---

## 9. Federated Intrusion Detection Systems

| Paper | Venue/Year | Datasets | Focus | Defenses | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| "Survey on FL for IDS: Concept, Architectures, Aggregation Strategies..." | ACM Computing Surveys, 2024 | Survey | FL-IDS taxonomy | Surveys aggregation strategies as implicit defenses | Broad survey | `unconfirmed` | `unconfirmed` | `unconfirmed` | Directly relevant to our own CUSUM/Krum/Cosine IDS layer — good structural reference |
| "A comprehensive survey of Federated IDS: Techniques, challenges and solutions" | ScienceDirect, 2024 | Survey | FL-IDS | Surveys DP, HE, SMC, gradient-leakage protection as IDS privacy techniques | Notes privacy-preserving FL-IDS must implement *at least one* of these techniques — a useful minimum-bar citation | `unconfirmed` | `unconfirmed` | `unconfirmed` | Supports our framing that IDS (Krum/CUSUM/Cosine) and privacy (DP) are usually evaluated separately in this literature — our combined pipeline (IDS + DP + MIA measured together) is comparatively rare |
| "A survey of privacy-preserving FL for IDS" | Artificial Intelligence Review (Springer), 2026 | Survey | Privacy-preserving FL-IDS | Broad | Recent, likely the most up-to-date survey in this category — worth a full read | `unconfirmed` | `unconfirmed` | `unconfirmed` | Priority read for the final related-work pass given recency |

**Category-level note**: this category's own surveys explicitly observe that intrusion-detection
research and privacy-attack research on FL are usually siloed from each other — which is close to
a direct citation for our claim that a pipeline testing IDS (Krum/CUSUM/Cosine), DP, and MIA
*together* is unusual.

---

## 10. Privacy Benchmarks for Federated Learning

| Paper | Venue/Year | Datasets | Focus | Defenses evaluated | Limitations | Realistic deployment? | Production FW? | Industrial dataset? | Extension opportunity |
|---|---|---|---|---|---|---|---|---|---|
| FEDLAD — "Federated Evaluation of Deep Leakage Attacks and Defenses" | arXiv:2411.03019, 2024/2025 | `unconfirmed`, likely CV benchmarks | Unified benchmark for gradient-inversion (Deep Leakage) attacks + defenses | Multiple GIA defenses compared | **Closest existing "benchmark" precedent to our own positioning** — but appears CV/benchmark-dataset-only, no industrial data or production FW per the snippet | `unconfirmed` | `unconfirmed`, likely no | Likely no | Must be read in full and directly contrasted in our introduction: "FEDLAD benchmarks GIA defenses on benchmark data; we benchmark MIA (and later GIA) on a real multi-site industrial deployment" is exactly the differentiation our paper needs to state precisely |
| "On the Efficiency of Privacy Attacks in Federated Learning" (EPAFL) | arXiv:2404.09430, 2024 | `unconfirmed` | Framework for improving efficiency (early-stopping) of privacy attacks, not a leakage benchmark per se | — | Different goal (attack efficiency, not benchmark/measurement) | `unconfirmed` | `unconfirmed` | `unconfirmed` | Tangential; possible citation for attack efficiency if our LiRA re-training cost becomes a discussion point (it is already a known runtime bottleneck in this project) |
| "A Framework for Evaluating Client Privacy Leakages in FL" | Springer (ESORICS?), 2020 | `unconfirmed` | Formal + experimental framework comparing client leakage attacks | — | Earlier/foundational; predates production FL frameworks like NVFLARE | No | No | No | Foundational "evaluation framework" precedent, useful in related work as an early ancestor of the benchmark idea |
| FedInverse — "Evaluating Privacy Leakage in Federated Learning" | OpenReview, `unconfirmed` year (recent) | `unconfirmed` | GIA-focused evaluation framework | `unconfirmed` | `unconfirmed` | `unconfirmed` | `unconfirmed` | Needs full read — potentially another direct "benchmark" competitor/precedent to differentiate against, similar to FEDLAD |

---

## Synthesis: does the "no benchmark exists" gap claim hold up?

Based on this first pass: **partially confirmed, not yet fully verified.** The closest existing
work is FEDLAD (§10) — a genuine benchmark, but for gradient-inversion attacks/defenses on
benchmark (not industrial) data, with no confirmed use of a production FL framework. No paper
found in this pass combines all of: (1) a production-grade FL framework (NVFLARE or equivalent),
(2) a real industrial/critical-infrastructure dataset, (3) more than one attack category measured
against more than one defense placement, in a single reusable harness. That combination is the
actual gap `docs/DSN2027_Positioning.md` claims. This pass supports that claim more than it
contradicts it, but it is one search pass per category — a wider systematic search (larger
per-category paper counts, backward/forward citation chasing from FEDLAD and the MIA/GIA surveys
above, and a direct search for "NVFLARE industrial privacy attack") is the natural next step
before the paper states the gap as settled fact.

## Targeted follow-up: NVIDIA FLARE + privacy attacks (done, 2026-07-24)

Ran the priority query flagged above (`"NVIDIA FLARE" OR NVFLARE membership inference attack
privacy evaluation`) immediately rather than deferring it, since it is the single most
load-bearing search for the positioning doc's gap claim. Result: **no paper found that runs a
membership-inference (or other privacy) attack against an NVIDIA FLARE deployment.** The only
NVFLARE-related hits were NVIDIA's own documentation, which describes FLARE's built-in DP filters
and confidential-computing features in general protective terms, without an adversarial
evaluation of their own claims. The MIA papers that did surface (a neuroimaging regression-model
MIA, and a data-free MIA on "FL in hardware assurance") are unrelated domains and do not use
NVFLARE. This is the strongest single data point so far for the claim in
`docs/DSN2027_Positioning.md` that production FL frameworks are rarely stress-tested against the
privacy attacks they claim to defend against — worth stating in the paper roughly as: "to our
knowledge, no prior work adversarially evaluates NVIDIA FLARE's own DP guarantees against a
membership-inference attack," with the explicit caveat that this is based on a single search
pass, not an exhaustive check.

## Next steps for this task (still open, #65)

1. Full-text read of: FEDLAD, the 2026 GIA survey (arXiv:2503.11514), the Nature Comms battery
   paper, and the 2026 AI Review FL-IDS survey — the four flagged above as highest-priority.
2. ~~A dedicated search for "NVIDIA FLARE" + "privacy attack"...~~ **Done, 2026-07-24** — see the
   "Targeted follow-up" section above (no prior work found attacking NVFLARE's own DP guarantees
   with a membership-inference attack). Worth re-running periodically as new 2026 papers appear,
   since this remains the single most load-bearing claim in the positioning doc, but it is no
   longer an open item as this list previously (incorrectly) implied.
3. Backward citation chase from the four MIA/GIA surveys to catch older landmark papers this
   search-engine-driven pass under-weighted (e.g. Shokri et al. 2017, Carlini et al. 2022 LiRA,
   Zhu et al. 2019 DLG — all already known/cited in this project but not re-surfaced here since
   recent-year search queries favor 2024–2026 results).
