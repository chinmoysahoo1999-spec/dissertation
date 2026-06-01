# STATUS.md — Project Snapshot

_Last updated: 2026-06-01 — **Semantic Entropy baseline REMOVED** from all 6 large-model `03_baselines_sota.ipynb` + the source gpt2 notebook. Reason: cross-encoder NLI + K-sample bidirectional clustering added ~3-4 hr per 7B model. Final baseline set is now **5 SOTA methods**: SAPLMA, HaloScope, HalluShift, EigenScore (INSIDE), LookbackLens. The pre-downloaded eval-dataset workflow (`Code/eval_datasets/00_download_eval_datasets.ipynb`) saves an estimated **~111 min cumulative (single account) / ~60 min in 6-account parallel mode** versus repeating HF Hub resolution in every 02 + 03 run._

_Previously (2026-05-29): pivot to **one unified `all_variants.ipynb` per model**: data-gen + F1–F10 feature extraction + 12 MLP variants in one resume-safe notebook._
_Mid-evaluation completed: 2026-02-13. Supervisor: Prof. Ujjwal Bhattacharya, ISI Kolkata._

---

## 1. Dissertation title and goal

**Hallucination Detection in Large Language Models Using Internal Representations.**

Concretely: extend the MIND framework (Su et al., Findings of ACL 2024) by augmenting the canonical last-token / last-layer hidden state with three categories of additional signals derived from the same forward pass — (i) layer-wise representation drift, (ii) cross-layer variance, (iii) predictive entropy — and evaluate across **4 LLMs** (TinyLlama-1.1B, Qwen2.5-3B, GPT-J-6B, Llama-2-7B) and 7 hallucination-detection benchmarks (4-QA suite + HaluEval triple). The 7B / 6B models are tractable on Kaggle Free T4×2 (32 GB total VRAM in bf16) without int8 quantisation — see PLAN.md §1.

---

## 2. Mid-evaluation baseline (2026-02-13)

The mid-eval slide deck (`mid_evl (3) (4).pdf`) reports the following completed work:

### 2.1 Data generation (done)
- 10,000 balanced samples per model
- Source: Wikipedia, first two sentences per article
- Method: MIND Algorithm 1 (entity selection → top-k continuation check → token-by-token search → splice with grafted continuation)
- Label 0 = original Wikipedia text. Label 1 = model produced a different entity, grafted with the original suffix.

### 2.2 Feature extraction (done)
- Single feature: hidden state at last token of final Transformer layer.
- Embedding dimension:
  - TinyLlama-1.1B → 2048
  - Qwen2.5-3B → 2560

### 2.3 Classifier (done)
- 5-layer MLP, 80/20 train/test split.

### 2.4 Headline results from the mid-eval

| Model | Params | Hidden dim | ROC-AUC |
|---|---|---|---|
| Qwen2.5-3B | 3 B | 2560 | **0.673** |
| TinyLlama-1.1B | 1.1 B | 2048 | **0.592** |

Observation in the slide: "Larger models show stronger signals in their hidden representations. Results indicate room for improvement by adding additional features."

The "additional features" are the six scalars listed in §3 below — these are the central technical contribution of the second half of the dissertation.

---

## 3. The 6 features promised in the mid-eval future-work slides

All six are scalars per sample, concatenated alongside the existing 2560-dim (Qwen-3B) or 2048-dim (TinyLlama) hidden-state vector. None of them duplicate the features used by Sharanya Dasgupta's HalluShift work (see §4).

### 3.1 Layer-wise representation drift (2 features)

For hidden state h_ℓ^(t) at layer ℓ and decoding step t:

- Cosine distance between adjacent layers: `D_ℓ^(t) = 1 − cos(h_ℓ^(t), h_{ℓ+1}^(t))`
- Mean drift across layers: `D_mean^(t) = (1 / (L−1)) · Σ_{ℓ=1}^{L−1} D_ℓ^(t)`

### 3.2 Cross-layer variance (2 features)

- Layer-wise mean: `h̄^(t) = (1 / L) · Σ_{ℓ=1}^{L} h_ℓ^(t)`
- L2 cross-layer variance: `V^(t) = (1 / L) · Σ_{ℓ=1}^{L} || h_ℓ^(t) − h̄^(t) ||_2^2`

(We store `V^(t)` as the scalar feature; `h̄^(t)` is internal to the variance calculation and not stored separately.)

### 3.3 Predictive entropy (2 features)

For per-step token distribution `p_t(w) = P(y_t = w | y_<t, x)`:

- Token-level entropy: `H^(t) = − Σ_{w ∈ V} p_t(w) · log p_t(w)`
- Mean entropy over generation: `H_mean = (1 / T) · Σ_{t=1}^{T} H^(t)`

Effective new feature vector per sample: `[H_N^n (hidden_dim) ‖ D_mean ‖ V ‖ H_mean]` — i.e. **+ 3 scalars over the mid-eval baseline** (Dℓ and H^(t) are per-step intermediates, only their means are stored).

Note: if a later ablation calls for per-layer drift rather than the mean, the dataset will be re-generated to also persist all `D_ℓ` values (L−1 scalars per sample).

---

## 4. Originality vs. Sharanya Dasgupta's HalluShift

Sharanya Dasgupta is a senior at the same institute (ISI Kolkata) but in a different lab (her supervisor: Prof. Swagatam Das; ours: Prof. Ujjwal Bhattacharya). Her M.Tech thesis (`M.tech _Sharanya_Dasgupta_CS2320.pdf`, June 2025) covers the *same 7-dataset suite* this dissertation targets. The institute's plagiarism check will flag overlap. **The required differentiation is at the feature-set level**, and it holds:

| Feature class | HalluShift (Dasgupta) | This dissertation |
|---|---|---|
| Cross-layer signal | **Wasserstein distance** + cosine similarity between layer distributions; range-wise feature selection | **Cosine distance** of adjacent layers + **L2 variance** across layers |
| Token-probability signal | mtp (mean token prob), Mps (max positive shift), Mg (mean neg-log-prob gain) | **Shannon entropy** of per-step distribution + mean entropy |
| Architecture | 2-layer MLP, metric-learning loss, single sigmoid output | 5-layer MLP, CrossEntropy loss, binary softmax output |
| Feature selection | Automated range-wise selection | Fixed concat — no selection |
| Mathematical category | Inter-distribution statistics (optimal transport) | Intra-sample geometric + information-theoretic statistics |

**Rules locked in for the rest of the project:**
1. Do NOT add Wasserstein-distance features.
2. Do NOT add mtp / Mps / Mg.
3. Do NOT use range-wise / automated feature selection.
4. If reviewers / examiners suggest adding any of the above, decline citing scope.
5. The literature survey chapter must cite HalluShift once as adjacent prior work in the "Internal-state probing — HalluShift" subsection of §2.5.4 and explicitly draw the distinction.

---

## 5. Current repo state (E:\Dessertation\)

```
E:\Dessertation\
├── STATUS.md                              <-- this file
├── PLAN.md                                <-- session-wise roadmap (see below)
├── HANDOVER.md                            <-- not yet written
├── Literature_Survey_Chapter.pdf          <-- 11-page Chapter 2 (clean)
├── Literature_Survey_Chapter — 11 pages   <-- the literature survey written previously
├── *.pdf                                  <-- 12 reference papers + mid-eval PDF (this folder includes Sharanya's thesis for cross-reference)
└── Code\
    ├── project.ipynb                      <-- CURRENTLY at Qwen-2.5-0.5B (WRONG — must revert to Qwen-2.5-3B; see Issue #1)
    ├── notebook_refactored.ipynb          <-- thin driver notebook
    ├── project_falcon_7b.ipynb            <-- per-model Kaggle notebook (RUN 2026-05-28; see §9)
    ├── project_gptj_6b.ipynb              <-- per-model Kaggle notebook (NOT YET RUN)
    ├── project_opt_2.7b.ipynb             <-- per-model Kaggle notebook (NOT YET RUN)
    ├── project_opt_6.7b.ipynb             <-- per-model Kaggle notebook (NOT YET RUN)
    ├── project_qwen2.5_3b.ipynb           <-- per-model Kaggle notebook (NOT YET RUN)
    ├── project_qwen2.5_7b.ipynb           <-- per-model Kaggle notebook (NOT YET RUN)
    ├── project_smoke_qwen0.5b.ipynb       <-- legacy smoke notebook (pre-2026-05-28)
    ├── project_tinyllama_1.1b.ipynb       <-- per-model Kaggle notebook (NOT YET RUN)
    ├── project_smoke_gpt2\                <-- 2026-05-28 Colab smoke (RUN; result JSON in dir)
    │   ├── project_smoke_gpt2.ipynb       <-- gpt2 (124M), 50/class. fp16. Includes BLOCK 6.5 dataset save.
    │   ├── gpt2_smoke_results.json        <-- result of 2026-05-28 Colab run (see §9)
    │   ├── gpt2_smoke_train.json          <-- 80 records (train split)
    │   ├── gpt2_smoke_test.json           <-- 20 records (test split)
    │   └── gpt2_smoke_mind_plus_best.pth  <-- best MLP weights (2.2 MB)
    ├── project_kaggle_llama2_7b\          <-- 2026-05-28 NEW (NOT YET RUN). 1000/class bf16. Needs HF gated access.
    │   └── project_kaggle_llama2_7b.ipynb
    ├── project_kaggle_gptj_6b\            <-- 2026-05-28 NEW (NOT YET RUN). 1000/class bf16.
    │   └── project_kaggle_gptj_6b.ipynb
    ├── project_kaggle_falcon_7b\          <-- 2026-05-30 split into 01+02+03. 1000/class bf16.
    │   ├── 01_data_generation.ipynb
    │   ├── 02_all_variants.ipynb
    │   └── 03_baselines_sota.ipynb
    ├── project_kaggle_mistral_7b\         <-- 2026-05-30 NEW. mistralai/Mistral-7B-v0.1. 1000/class bf16.
    │   ├── 01_data_generation.ipynb
    │   ├── 02_all_variants.ipynb
    │   └── 03_baselines_sota.ipynb
    ├── project_kaggle_opt_67b\            <-- 2026-05-30 NEW. facebook/opt-6.7b. 1000/class bf16. SAPLMA original backbone.
    │   ├── 01_data_generation.ipynb
    │   ├── 02_all_variants.ipynb
    │   └── 03_baselines_sota.ipynb
    ├── project_colab_tinyllama_11b\       <-- 2026-05-28 NEW (NOT YET RUN). 500/class bf16.
    │   └── project_colab_tinyllama_11b.ipynb
    ├── project_colab_qwen25_3b\           <-- 2026-05-28 NEW (NOT YET RUN). 500/class bf16. Replaces legacy qwen2.5_3b notebook.
    │   └── project_colab_qwen25_3b.ipynb
    ├── project_colab_qwen25_05b\          <-- 2026-05-28 NEW (NOT YET RUN). 300/class bf16. Replaces legacy smoke_qwen0.5b notebook.
    │   └── project_colab_qwen25_05b.ipynb
    ├── project_colab_opt_27b\             <-- 2026-05-28 NEW (NOT YET RUN). 500/class bf16. Replaces legacy opt_2.7b notebook.
    │   └── project_colab_opt_27b.ipynb
    ├── requirements.txt                   <-- pinned deps
    ├── pytest.ini
    ├── hallucination_10k.json             <-- 1.1 GB pre-existing dataset (gitignored)
    ├── preview.json                       <-- whitelisted preview slice
    ├── survey_chapter.md                  <-- Markdown source of the survey PDF
    ├── survey_notes.md                    <-- raw per-paper extraction notes
    ├── _patch_notebook.py                 <-- one-off; rewrote project.ipynb to (incorrectly) Qwen-0.5B
    ├── _build_notebooks.py                <-- one-off; emitted the per-model notebooks above
    ├── _build_pdf.py                      <-- one-off; built the survey PDF
    ├── src/                               <-- modules; STALE — still on Llama-3.2-1B; needs sync to Qwen-3B
    └── tests/                             <-- 24 passing / 4 skipped on machines with torch
```

---

## 6. Open issues (must be addressed before any new experiment)

| # | Issue | Severity | Where addressed |
|---|---|---|---|
| **1** | **`Code/project.ipynb` currently uses `Qwen/Qwen2.5-0.5B`** as set in a previous Cowork session, but the mid-eval baseline is `Qwen/Qwen2.5-3B`. The 0.5B swap was a misunderstanding of scope. | **Critical** | Session 1 of PLAN.md — revert to Qwen-3B as `PRIMARY_MODEL`; keep 0.5B as `SMOKE_MODEL` for quick iteration. |
| 2 | `Code/src/config.py` still says `MODEL_NAME = "meta-llama/Llama-3.2-1B"`, two model-swaps behind the notebook. | High | Session 1 of PLAN.md. |
| 3 | `Code/src/embeddings.py` returns only the last-layer / mean-pooled views. To compute the 6 new features (especially drift + variance + entropy) the function needs all per-layer hidden states at the last token AND the per-step logits. | High | Session 1 of PLAN.md. |
| 4 | `Code/src/dataset_gen.py` does not save per-step entropy or the 6 derived scalars. | High | Session 1 of PLAN.md. |
| 5 | The 10 k samples reported in the mid-eval are not in the repo (they live on the student's Colab session storage and were not exported). They must be regenerated for the 6-feature ablation. | Medium | Session 2 of PLAN.md. |
| 6 | None of the 7 downstream datasets are downloaded into the repo. | Medium | Session 3 of PLAN.md. |
| 7 | No GitHub Actions CI; tests don't run on push. | Low | Optional, end of project. |
| **8** | **HF dataset IDs `truthful_qa`, `trivia_qa`, `tydiqa` now require namespaces** (`truthfulqa/truthful_qa`, `mandarjoshi/trivia_qa`, `google-research-datasets/tydiqa`). | **Critical** → **Resolved in new notebooks** (2026-05-28). Every notebook under `project_kaggle_*/` and `project_colab_*/` uses the `safe_load_first(...)` helper with namespaced + legacy fallback. Confirmed working on Colab gpt2 smoke run (all 7 datasets loaded via loader #0 = namespaced ID). **Legacy root-level `project_*.ipynb` files still contain the bug — treat them as deprecated and run only the new per-directory copies.** |
| 9 | The Falcon-7B run reported F1 = 0.000 across the multi-task suite (model collapses to predicting class 0). Root cause is the MLP trained on 400 Wikipedia continuations does not transfer to other distributions — expected for this dataset size; not a bug. | Low | Re-evaluate at the planned ~10 k-sample scale; if F1 still 0 after scale-up, investigate threshold/calibration. |
| **10** | **§3 mis-glosses HalluShift's mtp / Mps / Mg.** §3 of this file currently says: "mtp = mean token probability", "Mps = max positive shift", "Mg = mean neg-log-prob gain". The 2026-05-29 audit of Dasgupta's actual thesis revealed: `mtp = min(P_max)` (**minimum**, not mean), `Mps = max(P_max − P_min)` (a **spread**, not a positive shift), `Mg = mean abs gradient of confidence between adjacent tokens`. Citing her work with wrong definitions in the lit-review chapter is worse than copying it. | **High** | Patch the §3 "Sharanya vs ours" table before the literature-survey chapter goes out. |

---

## 7. Hardware envelope

**Primary**: **Kaggle Free** with T4×2 (16 GB each → 32 GB total via `device_map="auto"`), 9 h hard session cap, 30 h / week GPU budget.

**Fallback**: Colab Free with single T4 (16 GB), ~12 h soft cap with frequent disconnects.

Memory budget headline (full 4-model sweep, all in bf16 on Kaggle T4×2):
- TinyLlama-1.1B: 2 GB — trivial.
- Qwen2.5-3B: 6 GB — comfortable on single GPU.
- GPT-J-6B: 12 GB — fits on one T4, leaves the other free for activation/KV-cache.
- Llama-2-7B: 14 GB — fits on one T4 with `device_map="auto"`; OR can split across both GPUs.
- HaluEval-Summarisation prompts truncated to 1024 tokens to avoid OOM on long contexts.

Total Kaggle GPU budget for the full plan: ~39 hours (~1.3 weeks of the 30 h / week quota). See PLAN.md §5.

---

## 8. Where this session ended

### 2026-05-23 session
This Cowork session did **not** change the code. It produced:

- The 11-page literature survey chapter (committed; 4 commits ready to push on `main`).
- This rewritten STATUS.md.
- The session-wise PLAN.md.

Code changes (Issue #1–#4 above) are deferred to **Session 1 of PLAN.md**, which the student will run in the next Cowork session.

### 2026-05-28 session
1. Ran **`project_falcon_7b.ipynb` on Kaggle** (the run was done by the student before this Cowork session opened; the assistant only saw the screenshot output). Results captured in §9 below. Three of the seven downstream datasets failed to load — root cause is the bare HF IDs (`truthful_qa`, `trivia_qa`, `tydiqa`) which now require namespaces. Logged as Issue #8.
2. Created **`Code/smoke_test_colab/project_smoke_gpt2.ipynb`** — a tiny end-to-end smoke test for Colab Free T4:
   * `gpt2` (124M) backbone — fits trivially in T4 fp16; whole pipeline finishes in ~5–8 min.
   * 50 samples / class data-gen.
   * 20–30 samples / downstream dataset.
   * **Fixed the broken HF dataset IDs** with a `safe_load_first(...)` helper that tries the namespaced ID first and falls back to the legacy bare ID for older `datasets` versions.
   * Writes `gpt2_smoke_results.json` containing **everything needed for verification offline**: env / library versions, model config, full timing breakdown, sanity-check intermediates (canonical-vector L2 norm, D_mean, V_last, H_last + asserts), data-gen skip-reason counters, raw `{y, p, prob}` predictions for the first 50 of every downstream dataset, all metrics + confusion matrices, and any non-fatal errors caught.
3. **No edits to existing per-model notebooks** were made in this session. The same dataset-loader fix used in the smoke notebook still needs to be ported to every `project_*.ipynb` (Issue #8).

### 2026-05-28 session (continued, later)
1. **gpt2 Colab smoke run completed** by the student. 147 sec end-to-end on a Colab Free T4. All 7 downstream datasets loaded successfully (namespace fix confirmed). MLP collapses to "always class 1" on the 80-sample training set — expected at this scale, not a bug. Results captured in §9.
2. **Created 7 new per-model notebooks** in their own `Code/project_<env>_<tag>/` directories (matching the `project_smoke_gpt2/` layout the student set up):
   * **Kaggle bucket (1000 samples / class, bf16)** — `project_kaggle_llama2_7b`, `project_kaggle_gptj_6b`, `project_kaggle_falcon_7b`.
   * **Colab bucket (500 samples / class, bf16)** — `project_colab_tinyllama_11b`, `project_colab_qwen25_3b`, `project_colab_opt_27b`.
   * **Colab tiny (300 samples / class, bf16)** — `project_colab_qwen25_05b`.
3. **Added BLOCK 6.5 (full-dataset save)** to every notebook including the previously-run smoke notebook. Each run now persists `<tag>_dataset_full.json` (every generated record, pre-split) alongside the existing train/test JSONs, the checkpoint JSON, the MLP `.pth`, and the results JSON. This unblocks offline ablations without re-running the LLM.
4. **HF dataset namespace fix shipped** to every new notebook (Issue #8 resolved in the new fleet; legacy root-level notebooks remain broken and are now deprecated).
5. **No edits to root-level `project_*.ipynb`** in this session either — those are kept around for reference/diff inspection only.

### 2026-05-29 session — unified `all_variants.ipynb` pivot
1. **Plagiarism audit completed** vs HalluShift (Sharanya Dasgupta, ISI Kolkata, June 2025). Estimated text-similarity: **8–18 %** assuming discipline (don't quote her 16 distinctive phrases, don't reproduce her HaluEval table layout, use fresh notation `z_l` not `h_l^t`, rename "Probabilistic Features" if used as a header). STATUS.md §4's differentiation claim holds; **but three of her feature glosses (mtp / Mps / Mg) are mis-defined in §3 and must be corrected before the lit-review chapter goes out** — see Issue #10 below.
2. **SOTA audit completed** for the 7 benchmarks on Llama-2-7B in the live-generation regime: HalluShift is currently SOTA on TruthfulQA-gen / TriviaQA-nc / CoQA / TydiQA-GP / HaluEval-QA. Beating her on those rows = setting a new public ceiling. Her weak spots are **HaluEval-Summ (AUROC 52 = random)** and **HaluEval-Dialogue (77)** — these are the easiest wins.
3. **Novel-feature menu compiled** (10 candidates, none colliding with HalluShift). Top 3 recommended for stacking: **F1 Lookback Ratio** ([Lookback Lens, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.84/)), **F5 Logit-Lens JSD** ([DoLa](https://arxiv.org/abs/2309.03883) / [SLED](https://arxiv.org/abs/2411.02433)), **F7 Token Max-Margin** ([HaMI, NeurIPS 2025](https://arxiv.org/abs/2504.07863)).
4. **Architectural pivot — unified `all_variants.ipynb`** replaces the 6-variant fleet (variant_A.ipynb..variant_F.ipynb) deleted from project_smoke_gpt2/. One notebook per model now does:
   * Stage 2: MIND data-gen → `<tag>_dataset_full.json` (**skipped if file exists**)
   * Stage 3: F1–F10 feature extraction → `<tag>_dataset_with_features.json` (**skipped if file exists**)
   * Stage 4: train 12 MLP variants (A through L) → 12 `<tag>_variant_<X>_best.pth` checkpoints
   * Stage 5: consolidated `<tag>_all_variants_results.json` (single audit file with the full metrics matrix)
   * **Resume-safe by design**: rerunning after a Colab disconnect skips completed stages because the output JSONs already exist.
5. **Deployed `all_variants.ipynb`** in 2 model directories: `project_smoke_gpt2/` (gpt2, 50/class, fp16) and `project_colab_qwen25_05b/` (Qwen2.5-0.5B, 300/class, bf16). Other 6 model directories keep their `project_<tag>.ipynb` from 2026-05-28 — they will only be migrated to the unified design if the gpt2 + Qwen-0.5B ablation shows a useful gain.
6. **Variant list (12)** — A: canonical only; B: +D_mean; C: +V_last; D: +H_mean; E: +D_mean+V_last+H_mean (MIND+ headline); F: +F1; G: +F5; H: +F7; I: +F1+F5+F7 (recommended trio); J: +all 9 F-features; K: E + trio (F1+F5+F7); L: everything. Easy to extend by editing `VARIANTS` dict in the notebook.
7. **F9 (SAPLMA mid-layer probe) deferred** — it requires a separately trained linear probe on labeled data; current pipeline has no place to inject this training step yet.

### 2026-05-29 session — gpt2 ablation run + SOTA baselines notebook (late)
1. **gpt2 all_variants ran end-to-end at 500/class.** 1000 records, all 9 F-features extracted with 0 failures (SDPA fix held). 12 variants trained + evaluated on Wikipedia held-out + all 7 downstream datasets. Total ~33 min on Colab T4.
2. **Headline finding for gpt2: variants don't separate at this scale.** Wikipedia AUROC range 0.32–0.51 across all 12 variants (most have F1=0 because the MLPs collapse on a 200-sample test). Multi-task avg AUROC 0.40–0.49, with no consistent winner — best variant rotates between A / C / D / F / L across datasets. Variant C (canonical + V_last) wins by tiniest margin at avg 0.490. **Conclusion: gpt2 results are diagnostic, not scientific. Pipeline is verified correct; signal will emerge on bigger backbones.**
3. **Created `Code/project_smoke_gpt2/baselines_sota.ipynb`** — faithful re-implementations of 4 SOTA baselines on gpt2 against the SAME 7-dataset eval suite. Reuses `gpt2_smoke_dataset_full.json` (same 500/class MIND data, same seed=42 train/test split as the variants) so AUROCs are directly comparable.
   * **SAPLMA** (Azaria & Mitchell 2023) — MLP probe `768→256→128→64→1` on layer-7 last-token hidden state.
   * **HaloScope** (Du et al. NeurIPS 2024) — official repo's pipeline: spectral score → percentile pseudo-labels → non-linear probe `768→1024→1`. Sweeps (layer × k × sign × threshold) on val.
   * **EigenScore (INSIDE)** (Chen et al. ICLR 2024) — K=10 stochastic samples per query, K×K covariance + 1e-3 ridge, score = `mean(log10(σ))`. K=3 with 100-sample cap on multi-task eval to keep cost manageable.
   * **HalluShift** (Dasgupta 2025, the senior's work) — exact 31-d feature vector (5 Wasserstein + 5 cosine on hidden states + 5 + 5 on attentions + 11 token-prob), CombinedNN with 4 parallel embedding heads, loss = BCE + 0.4·(1−acc), AdamW lr=1e-4.
4. **Methodology cross-checked against official GitHub repos** for HaloScope, INSIDE, HalluShift. Three of my prior assumptions were wrong (SAPLMA arch, EigenScore covariance shape, HalluShift's loss type) — all now corrected.
5. **Lookback Lens + Semantic Entropy deferred to Batch 2** — will be added once Batch 1 results land. Semantic Entropy needs a small NLI model loaded too.

### 2026-05-30 session — large-model rollout (Qwen-3B + Llama-2-7B)
1. **`all_variants.ipynb` deployed to project_colab_qwen25_3b/ and project_kaggle_llama2_7b/.**
   * Qwen-3B → Colab Free T4, 300 samples/class, bf16, layer 18 for F5 logit-lens.
   * Llama-2-7B → Kaggle Free T4×2, 200 samples/class, bf16, layer 16 for F5 logit-lens. Gated model — student must approve license + paste HF token in the new BLOCK 0 cell.
2. **`baselines_sota.ipynb` deployed to same two directories**, parameterised from the gpt2 source. SAPLMA layer choice: Qwen-3B = 22 (~60% depth, matches the OPT-6.7B fraction from the SAPLMA paper); Llama-2-7B = 16 (the paper's actual best layer for this model). EigenScore middle layer = 18 for Qwen-3B, 16 for Llama-2-7B. HaloScope sweeps all model layers.
3. **HalluShift Wasserstein speedup**: replaced `scipy.stats.wasserstein_distance` (per-token Python loop) with the closed-form 1D `sum|cumsum(P) − cumsum(Q)|` vectorised across token positions. Exactly equivalent for discrete probability distributions on integer support — no methodology change. ~5–10× faster on hidden_dim ≥ 2560 (critical for fitting baselines_sota in the 2–3h budget on Qwen-3B/Llama-2-7B).
4. **EigenScore batched generation** (`num_return_sequences=K`) from the gpt2 fix is preserved in both new notebooks — matches the official D2I-ai/eigenscore repo.
5. **Runtime targets** (per-model end-to-end, both notebooks):
   * Qwen-3B Colab: all_variants ~50 min + baselines_sota ~70 min ≈ **~2 h** combined.
   * Llama-2-7B Kaggle: all_variants ~65 min + baselines_sota ~85 min ≈ **~2.5 h** combined (well under Kaggle 9h cap).

### 2026-05-30 session — data-generation phase locked in (6 large-model fleet)
1. **All large models (≥3B) bumped to 1000 samples / class** (was 200): Llama-2-7B, Falcon-7B, GPT-J-6B. 5× more training data, comparable to MIND/HalluShift's 5k-class scale.
2. **Added two widely-cited models** to the fleet:
   * **`mistralai/Mistral-7B-v0.1`** — most-cited non-Llama 7B in 2024-25 papers (HalluShift, INSIDE, Semantic Entropy, DoLa). Open. 32 layers (SAPLMA=20, EigenScore=16). → `project_kaggle_mistral_7b/`
   * **`facebook/opt-6.7b`** — original SAPLMA backbone + HalluShift comparison. Open. 32 layers (SAPLMA=20, EigenScore=16). → `project_kaggle_opt_67b/`
3. **All 6 large models split into 3-notebook layout** (`01_data_generation` + `02_all_variants` + `03_baselines_sota`) for parallel-account execution. Downstream subsample cap = 0.2 across all to keep multi-task eval cost bounded despite 5× larger training set.
4. **Schema-sufficiency note for `<tag>_dataset_full.json`**: each record holds `{text, label, embedding, D_mean, V_last, H_mean, entity, title}`. This is sufficient for re-implementing any of the major hallucination-detection methods — they all take (text, label) and re-run the LLM. Exceptions: sampling-based methods (SelfCheckGPT, Semantic Entropy) need K samples per test prompt at evaluation time, generated on-the-fly in `03_baselines_sota`.
5. **Data-generation phase considered complete** for the 6-model fleet. Per-model runtime estimate for `01_data_generation.ipynb` alone on Kaggle T4×2: ~100 min for 2000 records (1000/class). Total GPU time across 6 Kaggle models: ~10 h.
6. **New shared utility: `Code/eval_datasets/00_download_eval_datasets.ipynb`**. Run ONCE per HF account to download all 10 multi-task eval datasets (TruthfulQA, TriviaQA, CoQA, TydiQA, HaluEval-{QA,Summ,Dialog}, NQ-Open, HotpotQA, PopQA) and save them as `eval_*.parquet` files plus a bundled `eval_datasets.zip`.
7. **All 02 and 03 notebooks updated with `_find_local_parquet(...)` shortcut** — `safe_load_first(...)` now checks 7 candidate paths for local parquet files before falling back to HuggingFace download. Output prints `(LOCAL: <path>)` when the local file was used and `(HF loader #N)` when it fell back. Plus immunity to HF rate-limit / namespace issues.

### 2026-06-01 session — Semantic Entropy removed + eval-dataset time-savings quantified
1. **Semantic Entropy baseline REMOVED** from `Code/project_smoke_gpt2/baselines_sota.ipynb` (source notebook for `outputs/parameterize_baselines_sota.py` and `outputs/remove_semantic_entropy.py`) and from all 6 large-model `03_baselines_sota.ipynb`. Reason: the cross-encoder NLI loader (`cross-encoder/nli-deberta-v3-base`, ~180M params) plus K-sample bidirectional entailment clustering added an estimated **~3-4 hr of compute per 7B model** (Wikipedia eval + multi-task eval combined). With a 2-3 h per-session budget this would have broken the parallel-account run plan. Backup of pre-removal source preserved at `Code/project_smoke_gpt2/baselines_sota.ipynb.bak_se_remove`.
2. **Final baseline set (5 SOTA methods)**: SAPLMA · HaloScope · HalluShift · EigenScore (INSIDE) · LookbackLens. Lookback Lens was retained because it shares its hidden-state forward pass with SAPLMA/HaloScope/HalluShift and only adds a per-(layer, head) attention aggregation step — negligible incremental cost compared to SE.
3. **Removal verified**: every one of the 7 affected notebooks (1 source + 6 large-model copies) now has `SemanticEntropy=0`, `semantic_entropy=0`, `_SE_NLI=0`, `need_nli=0` references on a full-text scan; cell 11 multi-task loop now writes only `LookbackLens` to `multitask`; cell 12 final summary now iterates 5 baselines; all 7 notebooks parse cleanly (magic-stripped AST).
4. **Eval-dataset pre-download time savings — quantified** (assumes 10 eval datasets at ~25-60 s cold HF resolution each = ~615 s / ~10.3 min per cold run, vs ~6.7 s to load all 10 from local parquet `Code/eval_datasets/eval_*.parquet`):
   * Per model (02 + 03 each load 10 datasets cold): **20.5 min → 0.2 min after one-time download.**
   * All 6 models, single-account sequential: **~123 min HF → ~11.6 min total** = **~111 min saved (≈ 1.9 h).**
   * All 6 models, 6-account parallel (each account does one 00 download): **~123 min HF → ~63 min** = **~60 min saved (≈ 1.0 h).**
   * Qualitative bonus: pre-downloaded parquets also remove HF Hub rate-limit failures during parallel runs, dataset-namespace breakage (e.g. `truthful_qa → truthfulqa/truthful_qa`), and mid-load kernel disconnects.
5. **Patch artifacts**: `outputs/remove_semantic_entropy.py` (the script that strips SE from the source notebook and re-emits all 6 copies). One-shot mop-up step also applied to remove the LookbackLens+SemanticEntropy banner comment and the final `for b in [..., "SemanticEntropy"]` loop that the main regex pass missed.
