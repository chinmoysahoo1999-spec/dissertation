# PLAN.md — Session-wise Roadmap (Kaggle-optimised)

_Written: 2026-05-23. Based on the Feb 2026 mid-evaluation baseline._
_Updated: 2026-05-23 — Llama-2-7B and GPT-J-6B work deferred (see §3 "Deferred future scale-up")._
_Author: Chinmoy Sahoo, CS2412. Supervisor: Prof. Ujjwal Bhattacharya, ISI Kolkata._

This document is the **single source of truth for what to do next**. It is divided into self-contained "sessions" sized to fit one **Kaggle free-tier session (9-hour hard cap, 30 h / week of T4×2 GPU)**. Each session declares its inputs, outputs, hardware requirements, acceptance criteria, and risks.

Read `STATUS.md` first to remember where the project stands. This file then says exactly what to do in the next window of compute.

---

## 0. Locked decisions (as of 2026-05-23)

1. **Primary compute environment**: **Kaggle Free** with T4×2 (= 32 GB total VRAM with model parallelism via `device_map="auto"`), 9-hour session cap, 30 h / week budget. Colab Free is the fallback (single T4, 16 GB, frequent disconnects).
2. **Primary model**: `Qwen/Qwen2.5-3B` (matches mid-eval baseline). Secondary `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Smoke-test model `Qwen/Qwen2.5-0.5B` for fast iteration.
3. **Scale-up models DEFERRED**: Llama-2-7B and GPT-J-6B are paused — see §3 below. The remaining sessions focus exclusively on Qwen-3B + TinyLlama.
4. **Dataset suite**: all 7 (TruthfulQA, TriviaQA, CoQA, TydiQA-GP English, HaluEval-QA, HaluEval-Summarisation, HaluEval-Dialogue).
5. **Originality discipline**: strict separation from Sharanya Dasgupta's HalluShift — no Wasserstein, no mtp / Mps / Mg, no automated feature selection. See STATUS.md §4 for the full table.
6. **Feature set is fixed at six (3 categories × 2 stats each)** as defined in STATUS.md §3. No additions without supervisor approval.

---

## 1. Hardware-feasibility table (Kaggle-first)

The single biggest difference between Colab Free and Kaggle Free is that **Kaggle gives you T4×2 = 32 GB total VRAM** via model parallelism, which makes bf16 inference of 7B-class models painless and 13B-class possible. We do *not* exploit that in the current scope (per §3), but it's documented here so re-opening the scale-up work later is a small change.

| Model | Quant | Model VRAM | Kaggle T4×2 (32 GB) | Colab Free T4 (16 GB) | Time / 1k samples (Kaggle T4×2) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | bf16 | 1.0 GB | ✓ trivial | ✓ trivial | ~3 min |
| TinyLlama-1.1B | bf16 | 2.2 GB | ✓ comfortable | ✓ comfortable | ~6 min |
| **Qwen2.5-3B** | **bf16** | **6.0 GB** | **✓ comfortable** | **✓ comfortable** | **~9 min** |
| GPT-J-6B (deferred) | bf16 | 12.0 GB | ✓ fits on one GPU | ⚠️ borderline | ~20 min |
| Llama-2-7B (deferred) | bf16 | 14.0 GB | ✓ fits on one GPU | ⚠️ tight, OOM on long contexts | ~20 min |
| Llama-2-13B (deferred) | bf16 | 26.0 GB | ✓ across both T4s | ✗ does not fit | ~40 min |

Activation memory still grows linearly with sequence length × hidden_dim × num_layers. For the **HaluEval-Summarisation** dataset (prompts up to 2 k tokens), truncate to 1024 tokens before feeding the LLM.

---

## 2. Session-by-session plan (Sessions 1–6 + write-up)

### Session 1 — Code sync + 6-feature extraction infrastructure (≈ 3 h, CPU OK)

**Goal**: bring `src/` and `project.ipynb` in line with the mid-eval reality (Qwen-3B) and add the plumbing for the 6 new features. No GPU needed — this is pure code work, can be done locally on Windows.

Sub-tasks:
1. Edit `src/config.py`:
   - `PRIMARY_MODEL = "Qwen/Qwen2.5-3B"`
   - `SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"`
   - `SECONDARY_MODELS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]` (deferred models removed from this list)
   - `MODEL_TAG = MODEL_NAME.split("/")[-1].replace(".", "").lower()`
   - `FEATURE_FLAGS = {"canonical": True, "cos_drift": True, "cross_var": True, "pred_entropy": True}` — easy ablation toggles.
2. Edit `src/embeddings.py` — return:
   - `canonical_mind`: last-token, last-layer (unchanged from MIND).
   - `D_mean`: mean cosine distance between adjacent layers at the last token.
   - `V_last`: L2 variance of last-token activations across layers.
   - (Internally, compute these from the per-layer hidden states; do NOT persist the full per-layer tensor — it would be ~750 KB / sample for Qwen-3B and 320 MB for 10 k samples, way too much.)
3. Edit `src/dataset_gen.py` — during `generate()`:
   - Capture `output.scores` (per-step logits).
   - Compute per-step Shannon entropy `H^(t) = − Σ p_t log p_t`.
   - Average over T to get `H_mean` and persist that scalar.
   - Persist record schema: `{label, text, entity, embedding (hidden_dim), D_mean, V_last, H_mean, title}`.
4. Edit `notebook_refactored.ipynb` Cell 4 to print the 4 new features in the sanity check.
5. Update `Code/_patch_notebook.py` so a future re-run produces a Qwen-3B notebook (not 0.5B).
6. Update `tests/test_config.py` invariant to expect Qwen-3B.
7. Update `tests/test_classifier.py` to construct embeddings of new dim (hidden_dim + 3).

**Acceptance criterion**: `pytest tests/ -v` returns 24 pass / 3 skip. `notebook_refactored.ipynb` runs Cells 1–3 (no generation) on CPU; Cell 4 prints `embedding shape = 2563` for Qwen-3B (2560 + 3 scalars).

**Risk**: schema change breaks the old test JSONs. Mitigation: dataset files are regenerated in Session 2, so old JSONs become obsolete by design.

---

### Session 2 — Regenerate the two mid-eval datasets with 6 features (≈ 4 h on Kaggle T4×2)

**Goal**: produce `qwen25-3b_train.json` / `_test.json` and `tinyllama-11b_train.json` / `_test.json`, each ~10 k samples, with the 6-feature schema.

Sub-tasks:
1. Upload the repo as a Kaggle Dataset (or clone via Kaggle Notebook's `!git clone`).
2. Run `project.ipynb` Cells 0–7 with `MODEL_NAME = "Qwen/Qwen2.5-3B"`. Stop after Cell 7 (split + save). Checkpoint every 500 samples to `/kaggle/working/`.
3. After the session, **download** the JSON files and **commit them to Google Drive or the repo** (gitignored — too big for GitHub) so the next session can resume from a clean state.
4. Repeat with `MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`.
5. Verify the 6 features by inspecting 5 random samples per dataset:
   - `D_mean` ∈ [0, 2] (cosine-distance range).
   - `V_last` > 0 and finite.
   - `H_mean` ∈ [0, log |V|] where |V| ≈ 152 k for Qwen tokeniser.

**Acceptance criterion**: four JSON files exist, sizes ~250 MB each (Qwen-3B) and ~200 MB (TinyLlama). Class balance ±10 % of 50/50. No NaN / Inf values.

**Kaggle-specific risks**:
- Session 9-hour hard cap → at ~9 min / 1k Qwen-3B samples (T4×2 bf16), 10k samples = ~1.5 h. Plenty of headroom.
- TinyLlama is even faster.
- Both fit comfortably in a single Kaggle session.

**Risk**: Wikipedia streaming rate-limits. Mitigation: HuggingFace `datasets.load_dataset(..., streaming=True)` retries automatically; bake in a 30-second backoff on HTTP 429.

**Risk**: Kaggle "Save Version" output is capped at 20 GB. JSONs are well under that; no issue.

---

### Session 3 — Train + ablate the classifier on Qwen-3B and TinyLlama (≈ 2 h, CPU is fine)

**Goal**: prove that the 6 features improve over the mid-eval AUROC of 0.673 for Qwen-3B and 0.592 for TinyLlama.

Sub-tasks:
1. Load `qwen25-3b_train.json` / `_test.json`.
2. Train 5 classifier variants (all 5-layer MLP, CE loss, identical hyper-params):
   - **A**: canonical embedding only (replicates mid-eval; expect AUROC ≈ 0.67 for Qwen-3B, 0.59 for TinyLlama).
   - **B**: canonical + `D_mean` only.
   - **C**: canonical + `V_last` only.
   - **D**: canonical + `H_mean` only.
   - **E**: canonical + all 3 (the headline configuration).
3. Compute Acc / Prec / Rec / F1 / AUROC + Brier + ECE for each.
4. Save the headline model `qwen25-3b_mind_plus_best.pth` for downstream eval.
5. Repeat steps 1–4 for TinyLlama (faster, ~30 min).

**Acceptance criterion**:
- Variant A reproduces mid-eval AUROC 0.673 ± 0.02 for Qwen-3B (sanity check).
- Variant E AUROC > Variant A AUROC by at least 0.02 on Qwen-3B. If the gain is smaller than that, debug feature scaling (likely culprit: `V_last` has a much larger magnitude than the embedding components — add per-feature normalisation).

**Risk**: features need to be scaled. Mitigation: fit a `StandardScaler` on training features, apply to test features, persist the scaler with the model.

**Stretch**: also try logistic regression and gradient boosting (xgboost) on the 3 scalar features alone — a strong baseline to report.

---

### Session 4 — Download + cache the 7 downstream datasets (≈ 1 h, mostly network, CPU)

**Goal**: get all 7 datasets into Google Drive (or Kaggle Dataset) so subsequent eval sessions are not bottlenecked on downloads.

Sub-tasks:
1. TruthfulQA → `datasets.load_dataset("truthful_qa", "generation")` → 817 samples.
2. TriviaQA → `datasets.load_dataset("trivia_qa", "rc.nocontext", split="validation")` → 9,960 dedup samples.
3. CoQA → `datasets.load_dataset("stanfordnlp/coqa", split="validation")` → 7,983 samples (flatten conversations to single-turn).
4. TydiQA-GP → `datasets.load_dataset("tydiqa", "secondary_task", split="validation")` then filter to English → 3,696 samples.
5. HaluEval-QA → `datasets.load_dataset("pminervini/HaluEval", "qa")` → 10,000.
6. HaluEval-Summ → `datasets.load_dataset("pminervini/HaluEval", "summarization")` → 10,000.
7. HaluEval-Dialog → `datasets.load_dataset("pminervini/HaluEval", "dialogue")` → 10,000.
8. Save each as a single `parquet` file. Write a `Code/datasets/loaders.py` with a uniform interface: `load("truthfulqa") → list[dict]`.

**Acceptance criterion**: 7 parquet files exist, sizes ~100 KB (TruthfulQA) to ~40 MB (HaluEval-Summ). Unit test loads each and asserts the expected sample count.

**Risk**: HaluEval is on `pminervini`'s HF mirror — original repo `JeffreyHsu/HaluEval` may have moved. Fall back to downloading the JSON files from `pkulcwmzx/HaluEval` on GitHub.

---

### Session 5 — Multi-task evaluation on Qwen-2.5-3B, datasets 1–4 (≈ 5 h on Kaggle T4×2)

**Goal**: evaluate the trained Qwen-3B + 6-feature classifier on the 4-QA suite.

Sub-tasks (in size-ascending order so you can stop part-way without losing the small datasets):

1. **TruthfulQA (817 samples, ~10 min)**:
   - Generate Qwen-3B answer with greedy decoding, `max_new_tokens=64`.
   - Extract `[canonical, D_mean, V_last, H_mean]` from the *last token of the generated answer*.
   - Score against gold: use BLEURT-base. Threshold τ on the score (validated on a 100-sample holdout) → binary label.
   - Feed features into `qwen25-3b_mind_plus_best.pth`. Compute Acc / Prec / Rec / F1 / AUROC.
2. **TydiQA-GP (3,696 samples, ~40 min)**: same protocol, BLEURT scoring against `gold_answer`.
3. **CoQA (7,983 samples, ~1.3 h)**: flatten conversations to single-turn `(passage, question, gold_answer)` triples.
4. **TriviaQA (9,960 samples, ~1.7 h)**: same protocol.

Write per-dataset rows into `Code/multi_task/results_qwen25_3b.json`. **Total session time ≈ 4 h**, comfortably inside Kaggle's 9 h cap.

**Acceptance criterion**: 4 rows in the results file, each containing all 5 metrics + the threshold used + the BLEURT-score histogram.

---

### Session 6 — Multi-task evaluation on Qwen-2.5-3B, HaluEval triple (≈ 3 h on Kaggle T4×2)

**Goal**: evaluate the same trained model on the 3 HaluEval splits. **Faster than Session 5** because the labels are *pre-existing in the dataset* — no LLM-judge scoring required.

Per HaluEval sample, the schema is `{question/knowledge/dialogue, right_answer, hallucinated_answer}`. For each sample we compute features twice:
- Once using `right_answer` → expected label 0.
- Once using `hallucinated_answer` → expected label 1.

That's 20,000 forward passes per HaluEval task. With Qwen-3B in bf16 on Kaggle T4×2, ~20 min per task.

Sub-tasks:
1. HaluEval-QA (~20 min). Per sample, feed `Context: {knowledge}\n Q: {question} A: {answer}` and extract features from the last token.
2. HaluEval-Summ (~40 min — long contexts; **truncate input to 1024 tokens** before tokenising).
3. HaluEval-Dialog (~20 min).

**Acceptance criterion**: 3 additional rows in `results_qwen25_3b.json`; total now 7 rows.

**Risk**: HaluEval-Summ contexts are too long. Mitigation: `tokenizer(..., max_length=1024, truncation=True)`. Document this in the methods chapter.

---

### Session 7 — Write-up: methods chapter + results chapter (≈ 10 h, no GPU)

**Goal**: turn the experimental results into thesis-ready prose.

Sub-tasks:
1. **Chapter 3 (Methods)**: ~15 pages. Subsections:
   - 3.1 Problem setup.
   - 3.2 MIND-style pseudo-labelling (cite Su et al.).
   - 3.3 The canonical MIND feature: H_N^n.
   - 3.4 The three new feature categories (drift, variance, entropy) — full equations, motivation, computational cost.
   - 3.5 Classifier architecture (5-layer MLP, CE loss, hyper-params).
   - 3.6 Multi-task evaluation protocol (prompt formats, BLEURT thresholding, HaluEval gold-label use, metric set).
   - 3.7 Implementation: Kaggle T4×2 specifics, single-call decoding, checkpointing.
2. **Chapter 4 (Results)**: ~15–20 pages. Subsections:
   - 4.1 Wikipedia-continuation classifier results (2 models × 5 ablation variants) — Table + bar chart.
   - 4.2 Multi-task transfer (2 models × 7 datasets) — heatmap.
   - 4.3 Cross-dataset analysis.
   - 4.4 Calibration (Brier, ECE).
   - 4.5 Computational cost.
   - 4.6 Comparison to literature (HaloScope on the 4-QA suite, HalluShift on all 7).

**Acceptance criterion**: two new markdown / docx files in `E:\Dessertation\` ready for inclusion in the dissertation. Word count combined ≥ 6000.

---

## 3. Deferred future scale-up (NOT in current scope)

The following sessions were planned in the 2026-05-23 review but are **deferred** pending dissertation deadline visibility. They are kept here so re-opening the work later is a small lift, not a redesign.

### Session 8 (deferred) — Scale-up: Llama-2-7B in bf16 on Kaggle T4×2 (~6 h gen + eval)

- Kaggle T4×2 (32 GB total) fits Llama-2-7B in bf16 across two GPUs with `device_map="auto"`. No int8 needed.
- Repeat Sessions 2 + 3 on Llama-2-7B.
- HuggingFace access required (`meta-llama/Llama-2-7b-hf`).

### Session 9 (deferred) — Scale-up: GPT-J-6B in bf16 on Kaggle T4×2 (~6 h gen + eval)

- Different architecture (`GPTJForCausalLM`) — useful for the cross-architecture story when re-opened.
- Hidden_dim = 4096, layers = 28.

### Session 10 (deferred) — Multi-task evaluation on Llama-2-7B and GPT-J-6B

- Repeat Sessions 5 + 6 on each scale-up model.
- Total ~16 h, will need to be split across 2–3 Kaggle sessions.

### Session 11 (deferred) — HELM benchmark sanity check + Llama-2-13B stretch

- Llama-2-13B in bf16 across both Kaggle T4 GPUs (~13 GB per GPU). Tight but feasible.
- HELM benchmark sanity check on Qwen-3B (reproduce MIND paper Table 2 within ±0.03 AUROC).

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaggle 9-hour session timeout mid-data-gen | Low (jobs are <4 h) | Low | Checkpoint every 500 samples; resume from checkpoint |
| Kaggle weekly 30 h GPU budget exhausted | Med (if iterating heavily) | Med | Plan one session per day; CPU work (Sessions 1, 3, 4, 7) does not consume GPU budget |
| HaluEval-Summ OOMs on T4 (single-GPU) | Low (T4×2 has headroom) | Med | Truncate prompts to 1024 tokens; document limitation |
| Feature-scale mismatch (`V_last` >> embedding components) hurts MLP training | High | Med | StandardScaler fitted on train; persist with model |
| Kaggle dataset streaming hits 429 from HuggingFace | Low | Low | Backoff + retry |
| Plagiarism check flags overlap with HalluShift | Low (given strict separation) | Critical | Rules locked in STATUS.md §4; supervisor review before submission |

---

## 5. Total compute estimate (current scope only — 7B/6B deferred)

| Phase | Sessions | Total Kaggle GPU hours | CPU-only hours |
|---|---|---|---|
| Code sync + 6-feature plumbing | 1 | 0 | ~3 h |
| Qwen-3B + TinyLlama data gen | 2 | ~3 h | — |
| Classifier ablation | 3 | 0 (CPU fine) | ~2 h |
| Datasets download | 4 | 0 | ~1 h |
| Multi-task eval Qwen-3B | 5, 6 | ~7 h | — |
| Write-up | 7 | 0 | ~10 h |
| **Total in current scope** | **1–7** | **~10 GPU hours** | **~16 CPU hours** |

That's well inside Kaggle's 30 h / week GPU budget — comfortable for one calendar week of work.

For the deferred sessions (8–11), the marginal compute is another ~30 GPU hours, which would fit a second Kaggle week.

---

## 6. What to do *right now*

1. **Push the existing commits to GitHub** (see Task #9 from the 2026-05-23 session log).
2. **In your next session**, open Kaggle, start a new notebook with the T4×2 accelerator, clone the repo, and **execute Session 1** (CPU work — bring `src/` and `project.ipynb` in line with this plan).
3. Once Session 1's tests pass, **execute Session 2** (4 h Kaggle GPU; you should finish well within the 9-h cap).

---

## 7. Decision log

- **2026-05-23 (morning)** — Primary model set to Qwen2.5-3B (not 0.5B as in the prior Cowork session). Feature set locked to the 6 in STATUS.md §3.
- **2026-05-23 (afternoon)** — Compute environment changed from Colab Free to **Kaggle Free** (T4×2 = 32 GB total; 9 h session, 30 h / week).
- **2026-05-23 (afternoon)** — Llama-2-7B + GPT-J-6B scale-up work **deferred** (see §3). Current scope is Qwen-3B + TinyLlama with the full 7-dataset multi-task evaluation.
- **2026-05-23 (afternoon)** — Note: on Kaggle T4×2, the deferred 7B / 6B models would actually be feasible in **bf16** (no int8 needed, unlike on Colab Free). When the deferred sessions are re-opened, this avoids quantisation overhead and keeps the bf16 numbers comparable to Qwen-3B's results.
