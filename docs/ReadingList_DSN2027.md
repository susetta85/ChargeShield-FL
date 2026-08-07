# Reading List for DSN 2027 Paper

Status: **living document, update as papers are read.** Derived from `docs/LiteratureReview.md`
(first-pass, 10-category search, 2026-07-24) — this file re-organizes that material by *reading
priority* rather than by topic category, to answer a narrower question: "what should I actually
sit down and read, in what order, before/while writing the paper?"

Mark items `[x]` as you finish a full-text read. Snippet-only entries in `docs/LiteratureReview.md`
are `unconfirmed` on several fields precisely because they haven't been read yet — reading an item
here is what upgrades it from "found by search" to "verified."

**Update (2026-08-07):** all 9 Tier 0/Tier 1 papers below were fetched and their bibliographic
metadata + abstract confirmed for real (correct title/authors/venue/year, abstract matches how this
project cites them) — see the `[abstract-confirmed 2026-08-07]` tags below. This is **not** the
same as the full-text read the checkbox `[x]` requires: in particular, LiRA's exact per-round
warm-started shadow/likelihood-ratio construction (the single most citation-sensitive detail, since
it's implemented not just cited) still needs a real methodology-section read before the paper cites
it as "matching," and the two Tier 1 gap-claim papers (FEDLAD, the battery paper) were only
confirmed at the abstract level, not deep enough to fully certify the "no industrial dataset" /
"no adversarial privacy test" claims yet — abstracts are *consistent* with those claims, not proof
of them. Checkboxes stay unchecked until that full-text pass happens.

---

## Tier 0 — Foundational papers this project's code directly implements or descends from

Read (or re-read) these in full regardless of paper deadline pressure — they are not
background reading, they are the specification our code follows. Getting a detail wrong here
would be citing our own method incorrectly.

- [ ] **Carlini, Chien, Nasr, Song, Terzis, Tramèr (2022)** — "Membership Inference Attacks From
  First Principles," IEEE S&P 2022. [arXiv:2112.03570](https://arxiv.org/abs/2112.03570) —
  **LiRA**, the project's primary attack (★). Confirm our per-round warm-started shadow retraining
  and likelihood-ratio scoring match the paper's actual construction, not just its name — this is
  the single most citation-sensitive paper in the bibliography since it's implemented, not just
  cited. `[abstract-confirmed 2026-08-07]` Real paper, correctly attributed (Carlini/Chien/Nasr/
  Song/Terzis/Tramèr, submitted 2021-12-07, v2 2022-04-12, cs.CR). Abstract confirms the core
  framing this project relies on: LiRA is explicitly a *likelihood-ratio* attack designed to be
  evaluated at low false-positive rates rather than average-case accuracy — still need the actual
  methodology section (shadow-model training procedure, per-example variance estimation) to confirm
  our per-round warm-start variant is a faithful adaptation, not just a same-named attack.
- [ ] **Shokri, Stronati, Song, Shmatikov (2017)** — "Membership Inference Attacks Against Machine
  Learning Models," IEEE S&P 2017. [arXiv:1610.05820](https://arxiv.org/abs/1610.05820) — origin
  of MIA as a field; needed for the related-work opening paragraph and to correctly position Yeom
  (our weakest attack) and LiRA relative to the attack's evolution. `[abstract-confirmed 2026-08-07]`
  Real paper (Shokri/Stronati/Song/Shmatikov, IEEE S&P 2017, arXiv v1 2016-10-18/v2 2017-03-31).
  Abstract confirms the "shadow model" idea originates here (train an attack model to recognize
  differences in target-model behavior on member vs. non-member inputs) — the direct ancestor of
  this project's own Shadow attack, not just of LiRA.
- [ ] **McMahan, Ramage, Talwar, Zhang (2018)** — "Learning Differentially Private Recurrent
  Language Models," ICLR 2018. [arXiv:1710.06963](https://arxiv.org/abs/1710.06963) — this is the
  literal mechanism our `dp-fedavg` mode implements (per-client clip+noise before aggregation).
  Must be cited as the source of that specific DP placement, not a generic "DP-SGD" citation.
  `[abstract-confirmed 2026-08-07]` Real paper (McMahan/Ramage/Talwar/Zhang, ICLR 2018, arXiv
  submitted 2017-10-18, camera-ready v3 2018-02-24). Abstract confirms it adds *user-level* DP to
  Federated Averaging itself — matches this project's per-client (not per-example) clip+noise
  placement. **Important nuance to resolve on full read**: after the 2026-08-06 finding that
  `dp-fedavg`≡`local` numerically in this project's simulation, worth checking whether McMahan et
  al.'s own protocol assumes the *server* or the *user's device* executes the per-user clip+noise
  step — that would tell us which of our two mode names this paper is actually the correct citation
  for.
- [ ] **Zhu, Liu, Han (2019)** — "Deep Leakage from Gradients," NeurIPS 2019.
  [arXiv:1906.08935](https://arxiv.org/pdf/1906.08935) — foundational Gradient Inversion attack
  (DLG). Read before Task #64 starts: our autoencoder's 6-feature tabular gradients are a very
  different attack surface than DLG's image gradients, and the paper needs to say precisely how
  and why, not just gesture at "GI exists." `[abstract-confirmed 2026-08-07]` Real paper (Zhu/Liu/
  Han, NeurIPS 2019, arXiv v1 2019-06-21/v2 2019-12-19). Abstract confirms DLG's headline result is
  *pixel-wise accurate* image recovery and *token-wise matching* text recovery from raw gradients —
  a much stronger reconstruction claim than membership inference. Confirms the gap ChargeShield-FL's
  own Sprint 14 needs to address explicitly: our target (6-feature tabular EV session records) has
  no pixel/token structure for DLG's optimization-based matching loss to exploit directly — the
  future Gradient Inversion module cannot just port DLG's loss function unchanged.
- [ ] **Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, et al. (2017)** — "Practical Secure
  Aggregation for Privacy-Preserving Machine Learning," ACM CCS 2017.
  [ACM DOI](https://dl.acm.org/doi/10.1145/3133956.3133982) ·
  [IACR ePrint mirror](https://eprint.iacr.org/2017/281.pdf) — the defense ChargeShield-FL does
  **not** implement. Needed to correctly frame, in the threat model section, *why* our
  honest-but-curious server can see raw pre-aggregation updates at all (SecAgg is simply absent
  from the pipeline, not defeated by it) — an important precision to get right so the paper isn't
  read as claiming to break SecAgg. `[abstract-confirmed 2026-08-07]` Real paper (Bonawitz/Ivanov/
  Kreuter/Marcedone/McMahan/Patel/Ramage/Segal/Seth, ACM CCS 2017, pp. 1175-1191). Abstract confirms
  SecAgg's actual guarantee: the server learns only the *sum* of user updates, never any individual
  contribution. Confirms the framing this project needs — ChargeShield-FL's threat model has no
  such protocol running at all, so the server sees each client's update individually by design; the
  paper should say this plainly rather than imply SecAgg was "bypassed."

## Tier 1 — Load-bearing for the paper's central "no benchmark exists" gap claim

These four are the ones `docs/LiteratureReview.md` already flagged as highest-priority full
reads. The gap claim in `docs/DSN2027_Positioning.md` currently rests on search-snippet evidence
only — these are the papers whose full text would either confirm or force a revision of that
claim before submission.

- [ ] **FEDLAD** — "Federated Evaluation of Deep Leakage Attacks and Defenses,"
  [arXiv:2411.03019](https://arxiv.org/abs/2411.03019), 2024/2025. Closest existing *benchmark*
  precedent to our own framing. Needs a precise, explicit contrast sentence in the introduction:
  FEDLAD benchmarks GIA defenses on benchmark-image data; we benchmark MIA (and later GIA) on a
  real multi-site industrial deployment. Read in full to confirm it really has no industrial
  dataset / production framework (currently `unconfirmed` from the snippet).
  `[abstract-confirmed 2026-08-07, gap claim STILL open]` Real paper, real venue framing ("FEDLAD
  Framework... a comprehensive benchmark for evaluating Deep Leakage attacks and defenses within a
  realistic Federated context"). Abstract confirms it is squarely a **Gradient Inversion (Deep
  Leakage) benchmark**, not an MIA benchmark — good, sharpens the contrast sentence (FEDLAD = GIA
  benchmark; ChargeShield-FL = MIA benchmark now, GIA later, on real industrial data either way).
  The abstract does **not** state what datasets FEDLAD actually uses — "no industrial dataset"
  remains unconfirmed pending an actual methodology-section read; do not cite this as settled yet.
- [ ] **"Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion
  Attacks"** — [arXiv:2503.11514](https://arxiv.org/abs/2503.11514), 2026. Most directly relevant
  paper to read *before* starting the Gradient Inversion module (Task #64) — it's a systematic
  empirical comparison of GIA methods and their limiting factors, exactly the design space Task
  #64 will need to navigate. `[abstract-confirmed 2026-08-07]` Real paper. Abstract confirms a
  3-way GIA taxonomy — optimization-based (OP-GIA), generation-based (GEN-GIA), analytics-based
  (ANA-GIA) — with a clear empirical verdict: **OP-GIA is the most practical setting despite
  unsatisfactory performance; GEN-GIA has too many dependencies; ANA-GIA is easily detectable.**
  Directly actionable for Sprint 14 scoping: start the Gradient Inversion module with an
  optimization-based approach (DLG-style) rather than a generative or analytics-based one, per this
  paper's own recommendation — worth a full read once Sprint 14 actually starts, not urgent now.
- [ ] **"Privacy-preserving collaborative battery fault warning for massive electric vehicles by
  heterogeneous data from charging stations"** — Nature Communications, published 2025-12-19.
  [nature.com/articles/s41467-025-67703-7](https://www.nature.com/articles/s41467-025-67703-7) ·
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/41419740/). `[abstract-confirmed 2026-08-07, gap claim
  STRENGTHENED]` Exact title/venue/DOI now confirmed (was only approximately known before). Uses
  **personalized federated learning** across EV charging-station data owners for battery fault
  detection, explicitly framed as privacy-preserving because it "integrates the information of all
  data owners without data sharing." The abstract summary available makes **no mention whatsoever**
  of any adversarial privacy evaluation (no MIA, no DP, no leakage test) — consistent with, though
  not full-text-proof of, the predicted gap: a real EV+FL industrial paper that asserts privacy
  preservation on architectural grounds ("no raw data shared") without testing that claim the way
  ChargeShield-FL does. Strong candidate contrastive citation; a full read would need to confirm
  the "no data sharing" claim isn't accompanied by an attack evaluation buried later in the paper.
- [ ] **"A survey of privacy-preserving federated learning for intrusion detection systems"** —
  Bunko, Johnstone, Yang, Scott. *Artificial Intelligence Review* (Springer), 2026.
  [link.springer.com/article/10.1007/s10462-026-11519-4](https://link.springer.com/article/10.1007/s10462-026-11519-4).
  `[abstract-confirmed 2026-08-07]` Exact authors/DOI now confirmed. Abstract states explicitly:
  **"there was no systematic review of privacy-preserving federated learning for IDS prior to this
  survey"** — confirms this is genuinely the first FL-IDS-privacy survey, a clean, quotable
  structural citation for positioning ChargeShield-FL's combined IDS+DP+MIA pipeline (the survey
  itself frames prior work as siloed: FL-broadly, or IDS-without-privacy, or PPFL-without-IDS —
  exactly the three-way combination this project's benchmark covers in one harness).

## Tier 2 — Closest prior threat-model precedent

- [ ] **"Perfectly Accurate Membership Inference by a Dishonest Central Server in Federated
  Learning"** — [arXiv:2203.16463](https://arxiv.org/abs/2203.16463), 2022. Closest prior work to
  our own threat model (a server attacking per-client updates before aggregation). Establishes
  that the *threat model* isn't novel — only the deployment context (real industrial data,
  production-adjacent pipeline, DP-vs-Central-vs-Local comparison) is. Important for precisely
  scoping the paper's actual contribution claim.

## Tier 3 — Surveys to cite/skim for related-work structure (not necessarily full-text reads)

Useful for citations and structuring the related-work section; lower priority for a cover-to-cover
read unless time allows.

- "Membership Inference Attacks and Defenses in Federated Learning: A Survey," ACM Computing
  Surveys, Dec 2024 — [arXiv:2412.06157](https://arxiv.org/abs/2412.06157)
- "The Federation Strikes Back: A Survey of FL Privacy Attacks, Defenses, Applications, and Policy
  Landscape" — [arXiv:2405.03636](https://arxiv.org/abs/2405.03636), 2024
- "A Survey on Gradient Inversion: Attacks, Defenses and Future Directions," IJCAI 2022
- "Survey on Federated Learning Threats: concepts, taxonomy on attacks and defences" —
  [arXiv:2201.08135](https://arxiv.org/abs/2201.08135), 2022
- "Federated Learning for Smart Grid: A Survey on Applications and Potential Vulnerabilities" —
  [arXiv:2409.10764](https://arxiv.org/abs/2409.10764), 2024/2025
- "Survey on Federated Learning for Intrusion Detection Systems: Concept, Architectures,
  Aggregation Strategies..." — ACM Computing Surveys, 2024
- "A comprehensive survey of Federated Intrusion Detection Systems: Techniques, challenges and
  solutions" — ScienceDirect, 2024

## Tier 4 — EV+FL secondary evidence (supports the gap claim, low individual weight)

Together these establish the pattern "EV/charging-infrastructure FL papers exist, but none test
their own privacy claims adversarially" — worth a paragraph citing all three rather than deep
individual reads.

- "Federated Learning for Early Prediction of EV Charging Demand," 2025 (ResearchGate preprint) —
  utility-only, no privacy evaluation
- "Anomaly Detection in EV Charging Stations Using Federated Learning" —
  [arXiv:2509.18126](https://arxiv.org/abs/2509.18126), 2025 — security-focused (intrusion), not
  privacy-leakage-focused
- "Fuse and Federate: Enhancing EV Charging Station Security with Multimodal Fusion and FL" —
  [arXiv:2506.06730](https://arxiv.org/abs/2506.06730), 2025 — same gap pattern

## Tier 5 — Adjacent attack types relevant to the PES v2 / Task #64 definition

Read when actually scoping the v2 composite metric (property inference / recovered-attribute
term) — not urgent for the v1 paper if it ships before Gradient Inversion is implemented.

- "Preference Profiling Attack Against Federated Learning" (PPA) —
  [arXiv:2202.04856](https://arxiv.org/abs/2202.04856), 2022 — directly analogous to inferring an
  EV owner's routine from charging-time patterns; motivates treating `hour_of_day`/
  `duration_hours` as "sensitive" features in the PES v2 definition
- "User-Level Label Leakage from Gradients in Federated Learning" (LLG), PoPETs 2022 — **scoping
  note, not directly applicable**: assumes classification with labels, ChargeShield-FL's
  autoencoder is unsupervised. Worth one sentence in related work explaining why label leakage is
  explicitly out of scope, not silently ignored.
- "Breaking Secure Aggregation: Label Leakage from Aggregated Gradients in Federated Learning" —
  [arXiv:2406.15731](https://arxiv.org/abs/2406.15731), 2024 — strong rhetorical parallel:
  "leakage surviving a nominal defense" is the same shape of result as ours, for a different
  attack/defense pair. Good companion citation when framing the "DP alone is not sufficient"
  thesis.

---

## Suggested reading order given limited time before the abstract deadline (2026-11-25)

1. Tier 0 (5 papers) — non-negotiable, these are what the code implements.
2. Tier 1's four papers — directly determines whether the "no benchmark exists" claim can be
   stated as confirmed or needs softening in the submission.
3. Tier 2 (1 paper) — quick, sharpens the contribution-scoping paragraph.
4. Tier 3 — cite from abstracts/snippets already in `docs/LiteratureReview.md`; only read in full
   if time remains.
5. Tier 4 — cite as a group, one paragraph, no individual deep reads needed.
6. Tier 5 — defer until Task #64 (Gradient Inversion / PES v2) actually starts.

## Still open (tracked under Task #65 in the project's task list)

- Backward citation chase from the Tier 3 surveys, to catch older landmark papers a
  recent-year-biased search under-weighted.
- Re-run the "NVIDIA FLARE + privacy attack" search periodically (see
  `docs/LiteratureReview.md`'s "Targeted follow-up" section) — no prior work found as of
  2026-07-24, but this is the single most load-bearing empirical claim in the positioning doc and
  is worth re-checking as 2026 papers continue to appear.
