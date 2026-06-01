# PLAN.md — Session-wise Roadmap (Kaggle-optimised)

_Written: 2026-05-23. Based on the Feb 2026 mid-evaluation baseline._
_Updated: 2026-05-23 (afternoon) — 7B / 6B models reinstated; targeting better headline AUROC._
_Updated: 2026-05-28 — Falcon-7B added to the per-model fleet (first Kaggle run done); Colab-T4 smoke test introduced in `Code/smoke_test_colab/`; HF dataset namespace fix logged as STATUS Issue #8._
_Author: Chinmoy Sahoo, CS2412. Supervisor: Prof. Ujjwal Bhattacharya, ISI Kolkata._

This document is the **single source of truth for what to do next**. It is divided into self-contained "sessions" sized to fit one **Kaggle free-tier session (9-hour hard cap, 30 h / week of T4×2 GPU)**. Each session declares its inputs, outputs, hardware requirements, acceptance criteria, and risks.

Read `STATUS.md` first to remember where the project stands. This file then says exactly what to do in the next window of compute.

---

## 0. Locked decisions (as of 2026-05-23)

1. **Primary compute environment**: **Kaggle Free** with T4×2 (= 32 GB total VRAM with model parallelism via `device_map="auto"`), 9-hour session cap, 30 h / week budget. Colab Free is the fallback (single T4, 16 GB, frequent disconnects).
2. **Models (4-model sweep, in priority order)**:
   - **Primary**: `Qwen/Qwen2.5-3B` (mid-eval baseline; AUROC 0.673).
   - **Secondary**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (mid-eval baseline; AUROC 0.592).
   - **Scale-up 1**: `meta-llama/Llama-2-7b-hf` — *requires HuggingFace gated-access approval; apply on Day 1.*
   - **Scale-up 2**: `EleutherAI/gpt-j-6b` (different architecture from Qwen / Llama — cross-arch story).
   - Smoke test (Kaggle): `Qwen/Qwen2.5-0.5B` (fast iteration only; not reported in final results).
   - Smoke test (Colab T4, NEW 2026-05-28): `gpt2` (124M) via `Code/smoke_test_colab/project_smoke_gpt2.ipynb` — 5–8 min end-to-end; only for pipeline-correctness verification.
   - **Opportunistic** (notebooks emitted but not in the headline 4-model story): `tiiuae/falcon-7b` (first Kaggle run 2026-05-28, partial results — see STATUS §9), `Qwen/Qwen2.5-7B`, `facebook/opt-2.7b`, `facebook/opt-6.7b`. Whether these appear in the thesis is a stretch decision; main story remains the original 4.
3. **Why 7B / 6B back in scope**: Kaggle's T4×2 = 32 GB fits both in **bf16** (Llama-2-7B uses ~14 GB, GPT-J-6B ~12 GB on one of the two GPUs). No int8 quantisation needed — clean bf16 numbers, directly comparable to the Qwen-3B baseline. Larger models give a stronger AUROC story (HaloScope reports ~0.78 on TruthfulQA with Llama-2-7B vs. your current 0.673).
4. **Dataset suite**: all 7 (TruthfulQA, TriviaQA, CoQA, TydiQA-GP English, HaluEval-QA, HaluEval-Summarisation, HaluEval-Dialogue).
5. **Originality discipline**: strict separation from Sharanya Dasgupta's HalluShift — no Wasserstein, no mtp / Mps / Mg, no automated feature selection. See STATUS.md §4 for the full table.
6. **Feature set is fixed at six (3 categories × 2 stats each)** as defined in STATUS.md §3. No additions without supervisor approval.

---

## 1. Hardware-feasibility table (Kaggle T4×2 bf16)

| Model | Quant | Model VRAM | Kaggle T4×2 (32 GB) | Time / 1k samples (Kaggle T4×2 bf16) |
|---|---|---|---|---|
| Qwen2.5-0.5B | bf16 | 1.0 GB | ✓ trivial | ~3 min |
| TinyLlama-1.1B | bf16 | 2.2 GB | ✓ comfortable | ~6 min |
| **Qwen2.5-3B** | **bf16** | **6.0 GB** | **✓ comfortable** | **~9 min** |
| **GPT-J-6B** | **bf16** | **12.0 GB** | **✓ fits one GPU** | **~18 min** |
| **Llama-2-7B** | **bf16** | **14.0 GB** | **✓ fits one GPU** | **~20 min** |
| Llama-2-13B (stretch) | bf16 | 26.0 GB | ✓ across both T4s | ~40 min |

Activation memory grows linearly with sequence length × hidden_dim × num_layers. For the **HaluEval-Summarisation** dataset (prompts up to 2 k tokens), truncate to 1024 tokens before feeding the LLM.

**Single-call decoding** (one `model.generate(max_new_tokens=N, output_scores=True)` per article instead of N separate calls) is essential for the 7B model — it cuts data-gen time by ~20×.

---

## 2. Session-by-session plan

### Session 1 — Code sync + 6-feature extraction infrastructure (≈ 3 h, CPU OK)

**Goal**: bring `src/` and `project.ipynb` in line with the mid-eval reality (Qwen-3B as primary) and add the plumbing for the 6 new features. No GPU needed.

Sub-tasks:
1. Edit `src/config.py`:
   - `PRIMARY_MODEL = "Qwen/Qwen2.5-3B"`
   - `SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"`
   - `SECONDARY_MODELS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "meta-llama/Llama-2-7b-hf", "EleutherAI/gpt-j-6b"]`
   - `MODEL_TAG = MODEL_NAME.split("/")[-1].replace(".", "").lower()`
   - `FEATURE_FLAGS = {"canonical": True, "cos_drift": True, "cross_var": True, "pred_entropy": True}` — ablation toggles.
2. Edit `src/embeddings.py` — return `(canonical_mind, D_mean, V_last)`. Internal per-layer hidden states are reduced to scalars before persisting (do NOT save the raw all-layer tensor — too big).
3. Edit `src/dataset_gen.py` — during `generate()`:
   - Capture `output.scores` (per-step logits).
   - Compute per-step Shannon entropy `H^(t) = − Σ p_t log p_t`; average to `H_mean`.
   - Persist record schema: `{label, text, entity, embedding (hidden_dim), D_mean, V_last, H_mean, title}`.
4. Edit `notebook_refactored.ipynb` Cell 4 to print the 4 new features in the sanity check.
5. Update `Code/_patch_notebook.py` so a future re-run produces a Qwen-3B notebook (not 0.5B).
6. Update `tests/test_config.py` invariant to expect Qwen-3B.
7. Update `tests/test_classifier.py` to construct embeddings of new dim (hidden_dim + 3).
8. **Apply for Llama-2-7B access on HuggingFace** (`https://huggingface.co/meta-llama/Llama-2-7b-hf`). Approval takes <1 hour. Do this in Session 1 so it's ready by Session 7.

**Acceptance criterion**: `pytest tests/ -v` returns 24 pass / 3 skip. `notebook_refactored.ipynb` runs Cells 1–3 (no generation) on CPU; Cell 4 prints `embedding shape = 2563` for Qwen-3B (2560 + 3 scalars).

---

### Session 2 — Data generation for Qwen-3B + TinyLlama (≈ 3 h on Kaggle T4×2)

**Goal**: produce `qwen25-3b_train.json` / `_test.json` and `tinyllama-11b_train.json` / `_test.json`, each ~10 k samples, with the 6-feature schema.

Sub-tasks:
1. Upload the repo as a Kaggle Dataset or `!git clone` it.
2. Run `project.ipynb` Cells 0–7 with `MODEL_NAME = "Qwen/Qwen2.5-3B"`. Checkpoint every 500 samples to `/kaggle/working/`.
3. Repeat with TinyLlama.
4. Save the four JSONs to a Kaggle Dataset (versioned) so subsequent sessions can attach them as inputs.
5. Verify the 6 features: `D_mean` ∈ [0, 2]; `V_last` > 0 finite; `H_mean` ∈ [0, log|V|].

**Time budget**: Qwen-3B 10k = ~1.5 h. TinyLlama 10k = ~1 h. Total ~2.5 h, well under the 9-h cap.

**Acceptance criterion**: 4 JSONs, ~250 MB (Qwen-3B) and ~200 MB (TinyLlama), balanced classes ±10%, no NaN.

---

### Session 3 — Classifier ablation on Qwen-3B + TinyLlama (≈ 2 h, CPU fine)

**Goal**: prove the 6 features improve AUROC over the mid-eval baseline.

Train 5 MLP variants on each model:
- **A**: canonical embedding only (replicate mid-eval).
- **B**: canonical + `D_mean`.
- **C**: canonical + `V_last`.
- **D**: canonical + `H_mean`.
- **E**: canonical + all 3 (headline).

Compute Acc / Prec / Rec / F1 / AUROC + Brier + ECE for each. Persist `qwen25-3b_mind_plus_best.pth` and `tinyllama_mind_plus_best.pth`.

**Acceptance criterion**:
- Variant A on Qwen-3B reproduces AUROC 0.673 ± 0.02.
- Variant E AUROC > Variant A AUROC by ≥ 0.02 on Qwen-3B.
- If gain < 0.02, debug feature scaling (StandardScaler fitted on train).

---

### Session 4 — Download + cache the 7 downstream datasets (≈ 1 h, network-bound, CPU)

TruthfulQA (817), TriviaQA-nocontext (9 960), CoQA dev (7 983), TydiQA-GP English (3 696), HaluEval-{QA, Summ, Dialog} (10 000 each). Save as parquet under `Code/datasets/cache/`. Single uniform loader `Code/datasets/loaders.py: load(name) -> list[dict]`.

**Acceptance criterion**: 7 parquet files, total ~100 MB. Unit test asserts expected sample counts.

---

### Session 5 — Multi-task eval on Qwen-3B, 4-QA suite (≈ 5 h on Kaggle T4×2)

For each of TruthfulQA, TydiQA-GP, CoQA, TriviaQA: generate Qwen-3B answer (greedy, max_new_tokens=64), extract features at the last generated token, BLEURT-score against gold, threshold to binary label, score via the trained MLP. Write rows to `Code/multi_task/results_qwen25_3b.json`.

**Acceptance criterion**: 4 rows in the results file, each with 5 metrics + threshold + BLEURT histogram.

---

### Session 6 — Multi-task eval on Qwen-3B, HaluEval triple (≈ 3 h on Kaggle T4×2)

HaluEval gives pre-labelled `right_answer` (label 0) and `hallucinated_answer` (label 1) per sample — no BLEURT scoring needed. Faster than Session 5. Truncate HaluEval-Summ prompts to 1024 tokens to avoid OOM.

**Acceptance criterion**: 3 additional rows in `results_qwen25_3b.json`; total = 7.

---

### Session 7 — Data generation for Llama-2-7B (≈ 4 h on Kaggle T4×2)

**Goal**: produce `llama-2-7b-hf_train.json` / `_test.json` with the 6-feature schema, 10 k samples.

Sub-tasks:
1. Verify HuggingFace gated-access for `meta-llama/Llama-2-7b-hf` is approved (applied in Session 1).
2. Load with `device_map="auto"`, `torch_dtype=torch.bfloat16` — on Kaggle T4×2, the model auto-splits across both GPUs.
3. Run `project.ipynb` Cells 0–7. Hidden dim = 4096. Time ~4 h for 10k samples.
4. Save JSONs to Kaggle Dataset.

**Acceptance criterion**: 2 JSONs, ~400 MB each (4096-d embeddings × 10k), no NaN, balanced classes.

**Risk**: HuggingFace access still pending. Mitigation: do Session 9 (GPT-J-6B) first instead; GPT-J is non-gated.

---

### Session 8 — Classifier ablation on Llama-2-7B (≈ 1 h, CPU fine)

Same 5 variants as Session 3 (A: canonical only … E: all 6 features). Persist `llama-2-7b_mind_plus_best.pth`.

**Acceptance criterion**: Variant E AUROC ≥ 0.70 (HaloScope reports 0.78 on TruthfulQA-style data with the same model; expect similar range here on Wikipedia-continuation). Variant E > Variant A by ≥ 0.02.

---

### Session 9 — Data generation for GPT-J-6B (≈ 4 h on Kaggle T4×2)

**Goal**: produce `gpt-j-6b_train.json` / `_test.json` with the 6-feature schema, 10 k samples. Different architecture (`GPTJForCausalLM`) — gives the dissertation a cross-architecture story.

Sub-tasks:
1. Load `EleutherAI/gpt-j-6b` with `device_map="auto"`, bf16. Hidden dim = 4096, 28 layers.
2. Run data generation pipeline.

**Acceptance criterion**: 2 JSONs, ~400 MB each.

**Risk**: GPT-J's tokeniser is BPE-based with different splits than Llama / Qwen → entity-substitution may have a higher rejection rate. Mitigation: lower top-k from 4 to 3 in `find_first_and_next_token` if reject rate > 50 %.

---

### Session 10 — Classifier ablation on GPT-J-6B (≈ 1 h, CPU fine)

Same 5 variants. Persist `gpt-j-6b_mind_plus_best.pth`.

**Acceptance criterion**: Variant E > Variant A by ≥ 0.02 AUROC.

---

### Session 11 — Multi-task eval on Llama-2-7B, 4-QA suite (≈ 6 h on Kaggle T4×2)

Same protocol as Session 5, but Llama-2-7B is ~2× slower in bf16 than Qwen-3B. May need to split:
- **Session 11a**: TruthfulQA + TydiQA-GP (~2 h).
- **Session 11b**: CoQA + TriviaQA (~4 h).

Write to `results_llama-2-7b.json`.

---

### Session 12 — Multi-task eval on Llama-2-7B, HaluEval triple (≈ 5 h on Kaggle T4×2)

HaluEval-{QA, Summ, Dialog} on Llama-2-7B. Same protocol as Session 6.

---

### Session 13 — Multi-task eval on GPT-J-6B, all 7 datasets (≈ 9 h, split across 2 Kaggle sessions)

- **Session 13a**: 4-QA suite (~5 h).
- **Session 13b**: HaluEval triple (~4 h).

Writes to `results_gpt-j-6b.json`.

---

### Session 14 — Write-up: methods + results chapters (≈ 10 h, no GPU)

**Goal**: turn the 4-model × 7-dataset × 5-feature-variant results into thesis-ready prose.

Sub-tasks:
1. **Chapter 3 (Methods)**, ~15 pages: problem setup; MIND pseudo-labelling; canonical feature; 3 new feature categories; 5-layer MLP; multi-task evaluation protocol; Kaggle T4×2 implementation notes.
2. **Chapter 4 (Results)**, ~20 pages: Wikipedia classifier results (4 models × 5 variants); multi-task transfer heatmap (4 models × 7 datasets); cross-model comparison; cross-architecture analysis (Llama / Qwen vs. GPT-J); calibration table; cost comparison; literature comparison (HaloScope on 4-QA, HalluShift on all 7).

**Acceptance criterion**: two markdown / docx files in `E:\Dessertation\`. Combined word count ≥ 8000.

---

## 3. Stretch sessions (only if time allows)

### Session 15 (stretch) — Llama-2-13B in bf16 across both T4 GPUs (~10 h)

26 GB bf16 across 2× 16 GB = 13 GB / GPU. Tight, but works. Worth a single run on TruthfulQA + HaluEval-QA only (smallest datasets) to add a "scaling story" data point.

### Session 16 (stretch) — HELM benchmark sanity-check on Qwen-3B (~3 h)

Reproduce MIND paper Table 2 within ±0.03 AUROC. Confirms the implementation is faithful.

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HuggingFace gated access for Llama-2-7B delayed | Med | High | Apply in Session 1; if not approved by Session 7, swap order — do GPT-J-6B first |
| Kaggle 30 h / week GPU budget exhausted | Med (full plan uses ~38 h) | Med | Plan one model-pair per week; CPU sessions (1, 3, 4, 8, 10, 14) don't consume GPU budget |
| Kaggle 9-h session limit mid-data-gen for 7B model | Low (gen takes ~4 h) | Low | Checkpoint every 500 samples; resume |
| HaluEval-Summ OOMs on single-T4 with 7B model | Low (using T4×2) | Med | Pre-truncate prompts to 1024 tokens |
| Feature-scale mismatch hurts MLP training | High | Med | StandardScaler fitted on train; persist with model |
| GPT-J tokeniser causes high entity-substitution rejection | Med | Low | Lower top-k from 4 to 3 |
| Plagiarism flag vs. HalluShift | Low (given strict separation) | Critical | Rules locked in STATUS.md §4; supervisor review |

---

## 5. Total compute estimate

| Phase | Sessions | Kaggle GPU hours | CPU-only hours |
|---|---|---|---|
| Code sync + 6-feature plumbing | 1 | 0 | ~3 |
| Qwen-3B + TinyLlama gen + classifier | 2, 3 | ~3 | ~2 |
| Datasets download | 4 | 0 | ~1 |
| Multi-task eval Qwen-3B | 5, 6 | ~8 | — |
| Llama-2-7B gen + classifier | 7, 8 | ~4 | ~1 |
| GPT-J-6B gen + classifier | 9, 10 | ~4 | ~1 |
| Multi-task eval Llama-2-7B | 11, 12 | ~11 | — |
| Multi-task eval GPT-J-6B | 13 | ~9 | — |
| Write-up | 14 | 0 | ~10 |
| **Total** | **14** | **~39 GPU hours** | **~18 CPU hours** |

That's ~1.3 weeks of Kaggle GPU budget. Realistic schedule: **2 calendar weeks** with one session/day.

If you have **Kaggle Pro** (`$5 / month`), the weekly budget rises to 60 h — entire plan in one week.

---

## 6. What to do *right now*

1. **Push the existing commits to GitHub** from Windows: `cd E:\Dessertation && git push origin main`.
2. **Apply for HuggingFace gated access** for `meta-llama/Llama-2-7b-hf` (do this NOW, even before starting Session 1 — approval takes < 1 hour but you don't want it blocking Session 7).
3. **Open a Kaggle account and create a new notebook** with T4×2 accelerator.
4. In your next Cowork session, ask: *"do Session 1 of PLAN.md"* — CPU-only code work; assistant will execute the 7 sub-tasks.

---

## 7. Decision log

- **2026-05-23 (morning)** — Primary model set to Qwen2.5-3B (not 0.5B as in the prior Cowork session). Feature set locked to the 6 in STATUS.md §3.
- **2026-05-23 (early afternoon)** — Compute environment changed from Colab Free to **Kaggle Free** (T4×2 = 32 GB total; 9 h session, 30 h / week).
- **2026-05-23 (late afternoon)** — **Llama-2-7B + GPT-J-6B reinstated**. Rationale: better headline AUROC (HaloScope reports 0.78 on TruthfulQA at 7B vs. our current 0.673 at 3B); Kaggle T4×2 fits both in bf16 without int8 overhead; matches the mid-eval future-work slide commitments.
- **2026-05-23** — Llama-2-7B HuggingFace gated access must be applied for in Session 1 (Day 1).
- **2026-05-28** — First Kaggle run completed (`project_falcon_7b.ipynb`, ~400 Wiki samples). Three of seven downstream datasets failed at load time due to the HuggingFace bare-ID → namespaced-ID transition (`truthful_qa` → `truthfulqa/truthful_qa`, `trivia_qa` → `mandarjoshi/trivia_qa`, `tydiqa` → `google-research-datasets/tydiqa`). Logged as STATUS Issue #8. **All `project_*.ipynb` BLOCK 11 cells must be patched with the same `safe_load_first(...)` pattern now in `smoke_test_colab/project_smoke_gpt2.ipynb`** before further Kaggle runs.
- **2026-05-28** — New Colab-T4 smoke test added at `Code/smoke_test_colab/project_smoke_gpt2.ipynb`. Backbone: `gpt2` (124M). Purpose: pipeline-correctness verification before committing GPU budget to 7B-class models. Result-dump JSON is intentionally verbose (env / library versions / sanity intermediates / raw `{y, p, prob}` slices / timings) so an offline reviewer can judge a run from the single file.
- **2026-05-28 (later)** — gpt2 Colab smoke completed (147 sec; all 7 datasets loaded via namespaced HF IDs; MLP-collapse-to-class-1 is expected at 80 train samples and is NOT a bug). Smoke notebook moved into `Code/project_smoke_gpt2/` alongside its result files.
- **2026-05-28 (later)** — **Per-model notebook fleet built.** 7 new notebooks live under their own `Code/project_<env>_<tag>/` directories:
   * Kaggle (1000 samples / class, bf16): `kaggle_llama2_7b`, `kaggle_gptj_6b`, `kaggle_falcon_7b`.
   * Colab (500 samples / class, bf16): `colab_tinyllama_11b`, `colab_qwen25_3b`, `colab_opt_27b`.
   * Colab tiny (300 samples / class, bf16): `colab_qwen25_05b`.
   All inherit the `safe_load_first(...)` namespace fix (Issue #8 resolved in the new fleet) and the new BLOCK 6.5 (`<tag>_dataset_full.json` save). Root-level legacy `project_*.ipynb` notebooks are now deprecated.
- **2026-05-28** — Llama-2-7B on Kaggle requires `huggingface_hub.login()` at the top of the notebook. The student must add a `login()` cell before BLOCK 0 and paste a Hugging Face token (gated-model access must be approved on the model page first).
- **2026-05-29** — **Architectural pivot: unified `all_variants.ipynb` per model**. Replaces the variant_A..variant_F.ipynb fleet with a single resume-safe notebook that does data-gen, F1–F10 feature extraction, and training of 12 MLP variants in one pass. Outputs a single consolidated `<tag>_all_variants_results.json` plus per-variant `.pth` checkpoints. Deployed for gpt2 (smoke) and Qwen2.5-0.5B (Colab) only — other models will be migrated only if the small-model ablation shows a useful gain.
- **2026-05-29** — **Feature set extension**. The 4 MIND+ features (canonical, D_mean, V_last, H_mean) are now joined by 9 features from recent literature: F1 Lookback Ratio (Lookback Lens, EMNLP 2024), F2 Attention-Sink Score, F3 EigenScore-Lite (INSIDE, ICLR 2024), F4 ICR Score (ACL 2025), F5 Logit-Lens JSD (DoLa / SLED), F6 Attention-Head Entropy, F7 Token Max-Margin (HaMI, NeurIPS 2025), F8 Token Rank, F10 Intra-Layer Dispersion (D²HScore). F9 (SAPLMA-style mid-layer probe) deferred — needs separate supervised training.
- **2026-05-29** — **HalluShift comparison locked**. Estimated text-similarity 8–18% with discipline (don't quote her 16 distinctive phrases; don't replicate her HaluEval table layout; use fresh notation). Her weak spots — HaluEval-Summarisation AUROC 52 (≈ random) and HaluEval-Dialogue 77 — are the easiest wins. On TruthfulQA / TriviaQA / CoQA / TydiQA / HaluEval-QA she is current public SOTA on Llama-2-7B in the live-generation regime, so beating her there = setting a new public ceiling.
- **2026-05-29 (late)** — gpt2 ablation completed at 500/class. Results were noise-level (Wikipedia AUROC 0.32–0.51 across all 12 variants; multi-task AUROC 0.40–0.49). Confirmed expectation: gpt2 is too small for the methodology to produce a meaningful detector. Pipeline is verified end-to-end; scale-up to Qwen-3B+ is the next signal-test.
- **2026-05-29 (late)** — **SOTA baselines code shipped for gpt2** at `Code/project_smoke_gpt2/baselines_sota.ipynb`. Four baselines (SAPLMA, HaloScope, EigenScore/INSIDE, HalluShift) reimplemented faithfully from their official GitHub repos and run on the SAME 7-dataset eval suite as `all_variants.ipynb`. Reuses `gpt2_smoke_dataset_full.json` so seed=42 train/test split is identical → cross-method AUROC comparison is directly valid. Two more (Lookback Lens, Semantic Entropy) queued as Batch 2.
- **2026-05-30** — **Large-model rollout: Qwen-3B + Llama-2-7B**. `all_variants.ipynb` and `baselines_sota.ipynb` both deployed to `Code/project_colab_qwen25_3b/` (Colab Free T4, 300 samples/class) and `Code/project_kaggle_llama2_7b/` (Kaggle Free T4×2, 200 samples/class). SAPLMA layer choices: 22 for Qwen-3B, 16 for Llama-2-7B (paper's verified best). Llama-2-7B is gated — student must approve license + paste HF token in the new BLOCK 0 login cell. HalluShift Wasserstein loop replaced with vectorised closed-form 1D Wasserstein (`sum|cumsum(P) − cumsum(Q)|`) — same math, ~5–10× faster.
- **2026-05-30 (later)** — **3-notebook split** rolled out to all large models: `01_data_generation` + `02_all_variants` + `03_baselines_sota` per model directory. Enables running 02 and 03 IN PARALLEL on two separate Colab/Kaggle accounts after 01 finishes. Saves ~30–60 min wall clock per model.
- **2026-05-30 (later)** — **Falcon-7B + GPT-J-6B integrated** with the same 3-notebook split. Both Kaggle T4×2. Falcon: layer 20 SAPLMA, 16 EigenScore. GPT-J: layer 18 SAPLMA, 14 EigenScore.
- **2026-05-30 (final)** — **Data-generation phase locked in**. All 6 large models (≥3B) bumped to **1000 samples/class** (was 200). 5× more training data, comparable scale to MIND/HalluShift papers. DOWNSTREAM_SCALE_CAP=0.2 across all Kaggle models so multi-task eval cost stays bounded.
- **2026-05-30 (final)** — **Added 2 widely-cited models**: `mistralai/Mistral-7B-v0.1` (most-cited non-Llama 7B in 2024-25 papers) and `facebook/opt-6.7b` (SAPLMA original backbone + HalluShift cross-arch comparison). Both Kaggle, open, 1000/class. Brings the large-model fleet to **6 models** (Qwen-3B Colab + Llama-2-7B/Falcon-7B/GPT-J-6B/Mistral-7B/OPT-6.7B Kaggle).
- **2026-05-30** — Confirmed: **`<tag>_dataset_full.json` schema (text + label + canonical + MIND scalars) is sufficient for extending to any future baseline** t