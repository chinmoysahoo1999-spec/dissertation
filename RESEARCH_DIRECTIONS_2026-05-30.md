# Research Directions — improving beyond SOTA

_Written: 2026-05-30. For: Chinmoy Sahoo (CS2412), ISI Kolkata. Supervisor: Prof. Ujjwal Bhattacharya._
_Scope: answers to four questions — (1) a novel + simple feature to beat SOTA, (2) alternative directions that keep the title, (3) target venues, (4) reading list. Grounded in a web scan of 2024–2026 internal-representation hallucination-detection literature; every claim links to a source in §6._

---

## 0. Read this first — the GPT-2 comparison is noise, not a result

Your current cross-method table (`Code/project_smoke_gpt2/comparison_table.md`) is on **GPT-2 (124M)**. Every AUROC in it sits between **0.32 and 0.71, clustered around 0.50** — i.e. coin-flip territory. "Variant L beats SOTA on Wikipedia (0.510)" and "Variant C wins TriviaQA (0.711)" are **within the noise band of a 200-sample test split**, not real wins. Your own `HANDOVER.md` already says this: _"gpt2 results are diagnostic, not scientific."_

Two consequences:

1. **Do not pick a new feature based on which one wins on GPT-2.** The ranking will reshuffle completely on a real backbone. Any feature-selection decision made on these numbers is built on sand.
2. **The single highest-value action is still to run `all_variants.ipynb` + `baselines_sota.ipynb` on Qwen2.5-0.5B → Qwen2.5-3B → Llama-2-7B.** Signal in internal-representation detectors only emerges at ≥1–3B scale (HaloScope/INSIDE report AUROC ~0.75–0.80 at 7B; you measured 0.673 at Qwen-3B in the mid-eval). Beating HalluShift is only meaningful once your own numbers leave the 0.5 band.

Everything below assumes you will validate on a real backbone first. The features and directions are chosen to be worth implementing **once you have a model that actually shows signal**.

---

## 1. Question 1 — a novel, simple feature to beat SOTA

### 1a. Honest framing on "novel and not published yet"

This sub-field is moving at roughly one preprint per week (see §6 — many of the papers I found are from the last 60–90 days). A guaranteed never-seen single scalar derived from hidden states is, realistically, no longer available: effective rank, HSIC decoupling, FFT of layer signals, layer-wise geometry, intrinsic dimension — all already have a 2025 preprint. **So the defensible novelty for your dissertation is not "one magic feature nobody has tried" — it is the *combination* you already own:** a single supervised probe that fuses geometric + spectral + information-theoretic internal signals, ablated systematically across 4 models × 7 datasets, kept strictly disjoint from HalluShift's optimal-transport / token-probability features.

Within that framing, here are four features that are **simple (one forward pass, 1–2 scalars each), distinct from your existing F1–F10 and from HalluShift, and under-explored as a supervised probe feature.** Ranked by my estimate of novelty × payoff × ease.

### 1b. Recommended feature #1 — Layer-trajectory **curvature** (2nd-order drift) ★ top pick

Your `D_mean` is a **first-order** signal: the average size of the step between adjacent layers. Curvature is the **second-order** signal it's missing: how much the trajectory *bends* as the representation climbs the layers.

Let `z_0, z_1, …, z_L` be the last-token hidden states at each layer (you already extract these). Define step (velocity) vectors:

```
Δ_l = z_{l+1} − z_l           for l = 0 … L−1
```

Curvature at layer l = the turning angle between consecutive steps:

```
c_l = 1 − cos(Δ_{l−1}, Δ_l)   for l = 1 … L−1
```

Features to store: `κ_mean = mean_l c_l` and (optionally) `κ_max = max_l c_l`.

- **Why it should help:** a confident, "knows-the-answer" forward pass moves smoothly and directionally up the residual stream; a hallucinating pass tends to wobble / re-route in mid-layers. `D_mean` cannot see this (a large smooth step and a large bent step have the same magnitude); curvature can.
- **Why it's distinct:** `D_mean` (1st-order magnitude), `V_last` (variance), `F4 ICR` (residual-update contribution), `F10` (intra-layer dispersion) are all different quantities. No Wasserstein, no token-prob → HalluShift-clean.
- **Cost:** ~5 lines of NumPy on tensors you already have. One forward pass. Two scalars.
- **Novelty (honest):** curvature of the layer trajectory is used as a *layer-selection criterion* in one very recent paper (Automatic Layer Selection, 2026), but I did **not** find it used as a per-sample **feature feeding the final supervised detector**. That gap is real and defensible.

### 1c. Recommended feature #2 — Single-pass **cross-layer effective rank** ★ strong pick

Stack the per-layer last-token states into a matrix `Z ∈ R^{(L+1)×d}`, mean-center columns, take singular values `σ_1 ≥ … ≥ σ_r`. Normalize `p_i = σ_i / Σ_j σ_j`. Then:

```
erank(Z) = exp( − Σ_i p_i · ln p_i )      # Roy & Vetterli effective rank
```

Cheaper variant (no SVD entropy, just two norms) — participation ratio on eigenvalues `λ_i = σ_i²`:

```
PR(Z) = (Σ_i λ_i)² / Σ_i λ_i²
```

Store one scalar (`erank` or `PR`).

- **Why it should help:** effective rank measures how many dimensions the layer trajectory genuinely occupies. A collapsed (over-committed) trajectory and an over-diffuse one both correlate with error. Wang et al. (Oct 2025) show this spectral signal detects hallucination and **generalizes robustly** — the generalization angle is exactly your weak spot.
- **Why it's distinct:** `F3 EigenScore-Lite` and INSIDE compute the log-det of a covariance built from **K sampled responses** (multiple generations). This is the **single-pass, cross-layer** version — no sampling, no extra forward passes. Different object, much cheaper.
- **Cost:** one `torch.linalg.svdvals` on a small `(L+1)×d` matrix per sample.
- **Novelty (honest):** the *signal* is validated by Wang et al. 2510.08389, but they use multiple outputs **and** layers; the **single-pass cross-layer-only** framing as a supervised probe feature is a fair adaptation, not a brand-new claim. Cite them, position yours as the cheap single-pass variant.

### 1d. Backup feature #3 — Prompt↔response **representation decoupling** (HSIC)

From HIDE (2506.17748): hallucination = statistical *decoupling* between the model's internal representation of the **input context** and of its **generated output**. Take input-token states `{h_i^in}` and output-token states `{h_j^out}` at one mid-layer, compute the Hilbert–Schmidt Independence Criterion with an RBF kernel → one scalar (low dependence ⇒ hallucination).

- **Reported payoff:** HIDE reports ~29% relative AUROC improvement over the best single-pass baseline, ~3% over multi-pass SOTA at ~half the compute.
- **Honest novelty:** published as a **standalone training-free score**; folding it in as *one feature* inside your supervised probe is integration novelty, not a new feature. Still cheap (one pass, one scalar) and a strong signal — good as feature #3 in the stack.

### 1e. Backup feature #4 — **Intrinsic dimension** of the layer point-cloud (TwoNN)

Treat `{z_0 … z_L}` as a point cloud and estimate its intrinsic dimension with TwoNN (Facco et al. 2017): per point take `μ = r₂/r₁` (2nd- vs 1st-nearest-neighbour distance), fit ID from the slope of the empirical CDF of `log μ`. One scalar.

- **Why distinct:** related-but-different to effective rank (ID is a local manifold estimate; erank is a global spectral one). Geometry-of-truth work touches ID conceptually but TwoNN-on-the-layer-cloud as a supervised feature appears unclaimed.
- **Honest novelty:** medium; keep as an ablation row rather than the headline.

### 1f. Suggested experiment (drop-in, no GPU-regeneration needed)

You already persist per-layer states in `<tag>_dataset_with_features.json`, so all four are **offline ablations** — no re-running the LLM. Add four variant rows to the `VARIANTS` dict in `all_variants.ipynb`:

| New variant | Feature added on top of E (MIND+) | Tests |
|---|---|---|
| M | `+ κ_mean, κ_max` (curvature) | does 2nd-order drift beat 1st-order `D_mean`? |
| N | `+ erank` (single-pass) | does spectral occupancy add over `V_last`? |
| O | `+ HSIC decoupling` | does prompt↔response coupling transfer? |
| P | `E + κ + erank + HSIC` (best trio) | does the fused geometric+spectral+coupling stack win? |

**Acceptance bar (on Qwen-3B, not GPT-2):** variant P AUROC ≥ E + 0.02 on Wikipedia held-out, **and** P's *average* across the 7 downstream sets ≥ HalluShift's average. If curvature alone (M) already clears E, that's your headline single-feature story.

---

## 2. Question 2 — alternative directions that DON'T change your title

Your title — _"Hallucination Detection in LLMs Using Internal Representations"_ — comfortably covers all of the following. Each turns a current weakness into a contribution.

### 2a. ★ Cross-domain / OOD generalization study (strongest; uses results you already have)

This is the single best alternative, because it **explains your own table** and addresses the field's biggest open problem. Two 2025 papers I verified say it directly:

- _"Representation-based Broad Hallucination Detectors Fail to Generalize Out of Distribution"_ (2509.19372): **"Out-of-distribution generalization is currently out of reach, with all of the analyzed methods performing close to random."**
- _"LLMs Know More Than They Show"_ (Orgad et al., ICLR 2025, 2410.02707): error detectors **"fail to generalize across datasets … truthfulness encoding is not universal but multifaceted."**

That is *exactly* what your table shows — a detector that wins where it was trained-ish and loses elsewhere. Reframe the dissertation's second half around the question **"when do internal-representation detectors transfer across datasets/models, and what makes a feature domain-invariant?"** You already have the apparatus to answer it: 4 models × 7 datasets × ~16 feature variants. Concretely: train on dataset A, test on B for all pairs → a 7×7 transfer matrix; show which features (geometric vs token-prob) survive the shift; propose a fix (cross-benchmark training, per-feature standardization, or a domain-invariant subset). This is a publishable contribution in its own right and needs **zero** new model runs beyond what you're already doing.

### 2b. Token-/span-level detection instead of sequence-level

Orgad et al. show **truthfulness is concentrated in specific tokens**, and using that "significantly enhances error detection." You currently pool to one vector per sample. A version that scores **per generated token** (and flags the hallucinated span) is a more useful product and a cleaner novelty than another scalar. Title-compatible, and your features (curvature, entropy) compute per-token for free.

### 2c. Error-type prediction from internal states

Same paper: internal representations can **predict the *type* of error** the model will make. A small head that classifies {factual / reasoning / refusal-worthy / fine} from the same hidden states is a novel, title-compatible add-on that supervisors like because it's diagnostic, not just a number.

### 2d. Calibration & selective prediction (not just AUROC)

Report ECE, Brier, and risk–coverage curves, and frame detection as **selective answering** ("abstain below threshold"). Almost no internal-representation paper does proper calibration; it's low-effort given your trained probes and adds a whole results sub-chapter.

### 2e. "Where does the signal live?" — automatic layer selection

Several 2026 papers (Automatic Layer Selection; the XGBoost-on-mid-layers result reporting ~91% AUC) show the signal concentrates in **specific mid-layers**, not the last layer you currently default to. A systematic per-layer sweep + a principled layer-selection criterion is simple, title-compatible, and often buys more AUROC than any new feature.

**My recommendation:** make **2a (OOD/transfer)** the spine of the second half and fold **2d (calibration)** in as supporting results — together they convert "we lose on most datasets" from a problem into the actual research question, with no extra GPU cost.

---

## 3. Question 3 — target venues (conferences + journals)

Tiered for a master's dissertation that becomes one paper. Note: it is **2026-05-30 today**, so dates matter.

### Conferences (NLP/ML)

| Venue | Fit | Timing (verify on site) | Notes |
|---|---|---|---|
| **EMNLP 2026** (Budapest, Oct 24–29) | ★ best fit | via **ARR** — the May 2026 ARR cycle deadline (May 25) just passed; aim for the next ARR cycle then *commit* to EMNLP | The natural home — most papers in §6 are EMNLP/ACL. |
| **AACL-IJCNLP 2026** | ★ strong, realistic | check site | Asia-Pacific ACL chapter; very reasonable bar for a solid empirical dissertation paper; good regional fit. |
| **ACL 2026 / NAACL 2026** | strong (reach) | via ARR | Same ARR pipeline; higher bar than AACL. |
| **AAAI-27** | broad AI | AAAI-26 deadline (Aug 2025) is long past; AAAI-27 abstracts ~Aug 2026 | Good if you want a non-NLP-specialist audience. |
| **IJCAI 2026** | broad AI | ~Jan 2026 deadline (likely passed) | IJCAI-25 had an internal-representation hallucination paper, so it's in-scope. |
| **NeurIPS / ICLR (+ their workshops)** | reach (main) / realistic (workshop) | rolling | A reliability/UQ **workshop** is a very achievable, citable target and a good first submission. |

**How ARR works (important):** ACL/EMNLP/NAACL/AACL now run through ARR (ACL Rolling Review). You submit once to an ARR cycle, get reviews, then *commit* to a venue. So "missing EMNLP's date" isn't fatal — submit to the next ARR cycle and commit to whichever venue is open.

### Journals (no fixed deadline — good for a dissertation timeline)

| Venue | Fit | Notes |
|---|---|---|
| **TMLR** (Transactions on ML Research) | ★ excellent | Rolling, fast, judges *correctness + interest* not novelty-hype — ideal for a thorough empirical study like the OOD-transfer angle. No page-limit pressure. |
| **TACL** (Transactions of the ACL) | ★ excellent | Rolling; presented at ACL/EMNLP. Prestigious, NLP-native. |
| **Computational Linguistics (CL)** | strong | MIT Press journal; rigorous. |
| **IEEE TASLP / IEEE TNNLS** | strong | Good if you want an IEEE journal line on the CV. |
| **Neurocomputing / Knowledge-Based Systems (Elsevier)** | safe | Fast, broad; reasonable fallback. |
| **Springer (e.g. *Automatic Documentation and Mathematical Linguistics*, *Cognitive Computation*)** | safe | One of the papers I found is in exactly this Springer line, so the topic is welcome there. |

**Suggested play:** target **TMLR or TACL** for the full study (rolling deadlines fit a dissertation), with an **AACL / reliability-workshop** version as the faster, lower-risk first publication.

---

## 4. The "minimum publishable" recipe

To beat SOTA *and* have a paper, the field's own guidelines (2509.19372 proposes evaluation guidelines) point to this:

1. **Validate on a real backbone** (Qwen-3B then Llama-2-7B) — kill the GPT-2 noise problem.
2. **Add curvature (M) + single-pass effective rank (N)** as two new disjoint features; report the fused stack (P).
3. **Report the 7×7 cross-dataset transfer matrix** (the OOD story) — this is your differentiator vs HalluShift, who report in-domain numbers.
4. **Report calibration (ECE/Brier) + risk–coverage**, not just AUROC.
5. Keep the HalluShift firewall (no Wasserstein, no mtp/Mps/Mg, no automated selection) — already locked in `STATUS.md §4`.

That is a complete, novel, title-faithful contribution that doesn't depend on winning every column — it wins the *generalization* column, which nobody currently does.

---

## 5. Caveats on this research

- **Recency vs verification:** I verified the four papers I lean on most (effective rank 2510.08389, HIDE 2506.17748, OOD-failure 2509.19372, Orgad et al. 2410.02707) by reading their abstracts directly. Other papers in §6 come from search snippets; confirm details before citing in the thesis. Some arXiv IDs are from the last few weeks (April–May 2026) — treat those as fast-moving preprints, not settled results.
- **"Novel" is a moving target.** Re-run the §6 searches the week before you submit; in this area something adjacent to your feature may appear. Your insurance is the *fusion + systematic ablation + OOD study*, which a single new feature paper cannot scoop.
- **None of the four features touch the HalluShift forbidden zone** — confirmed against `STATUS.md §4`.

---

## 6. Reading list (grouped; links verified live 2026-05-30)

**Verified by direct read (lean on these):**
- Effective Rank-based Uncertainty — Wang, Wei, Yue, Sun, arXiv 2510.08389 (Oct 2025): https://arxiv.org/abs/2510.08389
- HIDE: Hallucination detection via Decoupled Representations (HSIC) — arXiv 2506.17748 (Jun 2025): https://arxiv.org/abs/2506.17748
- Representation-based Detectors Fail to Generalize OOD — arXiv 2509.19372 (Sep 2025): https://arxiv.org/abs/2509.19372
- LLMs Know More Than They Show — Orgad et al., ICLR 2025, arXiv 2410.02707: https://arxiv.org/abs/2410.02707

**Geometry / trajectory of hidden states (feature ideas):**
- The Geometry of Truth: Layer-wise Semantic Dynamics — arXiv 2510.04933: https://arxiv.org/abs/2510.04933
- ICR Probe: Tracking Hidden State Dynamics — ACL 2025, arXiv 2507.16488: https://arxiv.org/abs/2507.16488
- Cross-Layer Attention Probing (CLAP) — arXiv 2509.09700: https://arxiv.org/pdf/2509.09700
- HSAD: FFT of hidden-layer temporal signals — arXiv 2509.13154: https://arxiv.org/pdf/2509.13154
- Hallucination Detection with the Internal Layers of LLMs — arXiv 2509.14254: https://arxiv.org/pdf/2509.14254

**Uncertainty / probing baselines (SOTA context):**
- Semantic Entropy Probes — Kossen et al., arXiv 2406.15927: https://arxiv.org/abs/2406.15927
- Pre-trained UQ Heads — Shelmanov et al., EMNLP 2025: https://aclanthology.org/2025.emnlp-main.1809/
- UQ for Hallucination Detection (survey) — arXiv 2510.12040: https://arxiv.org/pdf/2510.12040
- REDEEP (RAG hallucination) — ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/7daf60e805e596c3bd1e843e72ea5560-Paper-Conference.pdf

**SAE / interpretability angle (optional, more ambitious):**
- HalluSAE — arXiv 2604.16430: https://arxiv.org/html/2604.16430
- SAFE (SAE framework) — arXiv 2503.03032: https://arxiv.org/pdf/2503.03032

**Generalization / cross-domain (for the §2a direction):**
- SpikeScore: Cross-Domain Hallucination Detection — arXiv 2601.19245: https://arxiv.org/pdf/2601.19245
- Detecting hallucinations via semantic entropy — Farquhar et al., Nature 2024: https://pubmed.ncbi.nlm.nih.gov/38898292/

**Survey / list to mine for more:**
- EdinburghNLP awesome-hallucination-detection (curated list): https://github.com/EdinburghNLP/awesome-hallucination-detection
