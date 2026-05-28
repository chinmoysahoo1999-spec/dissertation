# HANDOVER.md — Next-session briefing

_Written: 2026-05-28 (end of session). Author: Cowork assistant. For: the next Cowork session._

This document is the first thing the next session should read. It is intentionally short and direct. The longer state lives in `STATUS.md` (snapshot) and `PLAN.md` (roadmap) — this file just gives the new session enough context to pick up the thread without re-reading both.

---

## a. Goal (the big picture)

**Dissertation title:** *Hallucination Detection in Large Language Models Using Internal Representations.*

**Concrete plan:** extend the MIND framework (Su et al., Findings of ACL 2024) by augmenting the canonical last-token / last-layer hidden state with three categories of additional signals derived from the same forward pass:

1. **Layer-wise representation drift** — `D_mean` = mean cosine distance between adjacent layers at the last token.
2. **Cross-layer variance** — `V_last` = L2 variance of last-token activations across layers.
3. **Predictive entropy** — `H_mean` = mean per-step Shannon entropy of the token distribution during generation.

Evaluate across **4 LLMs** (TinyLlama-1.1B, Qwen2.5-3B, GPT-J-6B, Llama-2-7B) and **7 hallucination-detection benchmarks** (TruthfulQA, TriviaQA, CoQA, TydiQA-GP, HaluEval-{QA, Summ, Dialog}). The 7B/6B models are tractable on Kaggle Free T4×2 (32 GB total VRAM in bf16) without int8 quantisation.

**Originality discipline (locked):** strict separation from Sharanya Dasgupta's HalluShift work — no Wasserstein, no mtp/Mps/Mg, no automated feature selection. See STATUS.md §4.

---

## b. Current status

### What's already done

* **Literature survey chapter (11 pages)** — committed; PDF + Markdown source at `Literature_Survey_Chapter.pdf` and `Code/survey_chapter.md`.
* **Pipeline implementation** — the full 14-block notebook pipeline (MIND data-gen → feature extraction → MLP train → Wikipedia held-out → 7-dataset multi-task eval → JSON dump) is working end-to-end. Confirmed by the gpt2 Colab smoke run on 2026-05-28 (147 sec, all datasets loaded, all metrics produced).
* **Per-model notebook fleet (8 notebooks)** — each lives in its own `Code/project_<env>_<tag>/` directory; see "Important context" below.
* **HF dataset namespace bug fixed** in the new fleet (`truthful_qa → truthfulqa/truthful_qa`; `trivia_qa → mandarjoshi/trivia_qa`; `tydiqa → google-research-datasets/tydiqa`). Each notebook uses `safe_load_first(...)` to try the namespaced ID first and fall back to the legacy bare ID.
* **Full-dataset save (BLOCK 6.5)** added to every notebook — each run persists `<tag>_dataset_full.json` (every generated record, pre-split) alongside the existing train/test JSONs, so ablations can be run offline without re-generating data on a GPU.

### What's in progress / partially done

* **Falcon-7B Kaggle run (2026-05-28)** — the legacy `Code/project_falcon_7b.ipynb` was run with 200 H + 200 ¬H samples; 3 of 7 downstream datasets failed (namespace bug, now fixed in the new fleet). Multi-task F1 = 0.0 across the board (expected at this sample size — MLP collapses to constant predictor). Re-run the **new** `Code/project_kaggle_falcon_7b/` notebook at 1000/class to get a meaningful number.
* **gpt2 Colab smoke run (2026-05-28)** — completed cleanly, but predictions collapsed to "always class 1" with 80 training samples on a 124M backbone. Smoke purpose was pipeline verification, not a meaningful AUROC. **Don't waste time interpreting gpt2 metrics — they are diagnostic, not scientific.**

### What still needs to happen

In priority order (matches PLAN.md sessions 1–14):

1. **Apply for Llama-2-7B HuggingFace gated access** at https://huggingface.co/meta-llama/Llama-2-7b-hf (approval typically < 1 hour). Required before `project_kaggle_llama2_7b/` can run.
2. **Run the 4 Colab notebooks** in this order to verify they work end-to-end on Colab Free T4:
   * `project_colab_qwen25_05b/` (300/class, ~10 min) — quickest sanity check.
   * `project_colab_tinyllama_11b/` (500/class, ~15 min).
   * `project_colab_qwen25_3b/` (500/class, ~20 min) — this matches the mid-eval baseline target (AUROC 0.673 expected).
   * `project_colab_opt_27b/` (500/class, ~15 min).
3. **Run the 3 Kaggle notebooks** (1000/class each, in priority order):
   * `project_kaggle_falcon_7b/` first (re-do of today's run, no gating).
   * `project_kaggle_gptj_6b/` second (no gating; cross-architecture story).
   * `project_kaggle_llama2_7b/` third (after gated access lands).
4. After each run, drop the `<tag>_results.json` (and ideally `<tag>_dataset_full.json` if it'll fit in Git) into the matching `Code/project_<tag>/` directory. The assistant can then examine the JSON for sanity — same audit format as the 2026-05-28 gpt2 smoke result.
5. Once all 4 mid-eval models have results, write the methods + results chapters (PLAN.md §Session 14).

---

## c. Important context

### File locations (Windows paths)

| What | Where |
|---|---|
| Repo root | `E:\Dessertation\` |
| GitHub remote | `https://github.com/chinmoysahoo1999-spec/dissertation.git`, branch `main` |
| Plan + Status + Handover | `E:\Dessertation\PLAN.md`, `E:\Dessertation\STATUS.md`, `E:\Dessertation\HANDOVER.md` (this file) |
| Literature survey PDF | `E:\Dessertation\Literature_Survey_Chapter.pdf` |
| Per-model run directories | `E:\Dessertation\Code\project_<env>_<tag>\` — each holds the `.ipynb` and the run's output files |
| Legacy auto-generated notebooks (DEPRECATED — still have HF namespace bug) | `E:\Dessertation\Code\project_*.ipynb` at the root |
| Notebook builder (re-runs are idempotent) | the assistant has this builder in its scratch outputs; ask the assistant to re-run it if the fleet needs to be regenerated |
| Pinned deps | `E:\Dessertation\Code\requirements.txt` |
| Tests | `E:\Dessertation\Code\tests\` (24 pass / 4 skip without torch) |

### Per-notebook output files

Each notebook drops these into its working directory when run:

| File | Purpose |
|---|---|
| `<tag>_dataset_full.json` | every generated record (pre-split) — for offline ablations |
| `<tag>_train.json` / `<tag>_test.json` | the 80 / 20 split used for the MLP |
| `<tag>_checkpoint.json` | incremental save every 200 samples — resume after disconnect |
| `<tag>_mind_plus_best.pth` | best MLP weights + scaler tensors |
| `<tag>_results.json` | env / config / model info / metrics / timings — single-file audit |

### The 8-notebook fleet

| Notebook | Env | Samples / class | dtype | Status |
|---|---|---|---|---|
| `project_smoke_gpt2` | Colab T4 | 50 | fp16 | **RUN** 2026-05-28 — pipeline verified |
| `project_colab_qwen25_05b` | Colab T4 | 300 | bf16 | not yet run |
| `project_colab_tinyllama_11b` | Colab T4 | 500 | bf16 | not yet run |
| `project_colab_qwen25_3b` | Colab T4 | 500 | bf16 | not yet run — **mid-eval primary model, target AUROC 0.673** |
| `project_colab_opt_27b` | Colab T4 | 500 | bf16 | not yet run |
| `project_kaggle_falcon_7b` | Kaggle T4×2 | 1000 | bf16 | not yet run — replaces the broken legacy run |
| `project_kaggle_gptj_6b` | Kaggle T4×2 | 1000 | bf16 | not yet run |
| `project_kaggle_llama2_7b` | Kaggle T4×2 | 1000 | bf16 | not yet run — **needs HF gated access + `login()` cell at top** |

### Hardware envelope

* **Primary:** Kaggle Free, T4×2 (= 32 GB VRAM via `device_map="auto"`), 9-hour session cap, 30 h / week budget.
* **Fallback:** Colab Free, single T4 (16 GB), ~12 h soft cap, frequent disconnects. Checkpoint cell already handles disconnects via `<tag>_checkpoint.json` every 200 samples.

---

## d. Decisions already made (do NOT revisit unless supervisor asks)

1. **Feature set is fixed at six (3 categories × 2 stats each).** See STATUS.md §3. No additions.
2. **No Wasserstein-distance features, no mtp/Mps/Mg, no automated feature selection.** Originality discipline vs HalluShift (STATUS.md §4).
3. **MLP architecture locked**: 5-layer (512 → 256 → 128 → 64 → 2), CrossEntropy loss, Adam(lr=5e-4, wd=1e-5), 10 epochs, batch size 32, 80/20 train/test split.
4. **Sentence-Transformer MiniLM** is the gold-vs-generation scorer for open-ended QA (not BLEURT — too heavy).
5. **HaluEval prompts truncated to 1024 tokens** to avoid OOM on long contexts.
6. **bf16 model loading** (not int8/4bit) for the 4-model sweep. Kaggle T4×2 fits everything in bf16 without quantisation.
7. **Per-model notebook fleet structure**: each model gets its own `Code/project_<env>_<tag>/` directory. Result JSONs land in the same directory. The assistant audits each `<tag>_results.json` after the run.
8. **Dataset namespace fix** (2026-05-28): every new notebook uses `safe_load_first(...)` with namespaced + legacy fallback. Don't revert to bare IDs.
9. **BLOCK 6.5 dataset save** (2026-05-28): every notebook now saves the full pre-split dataset to JSON. Don't remove this — downstream ablations depend on it.
10. **Legacy root-level `project_*.ipynb` notebooks are deprecated.** Run the new per-directory fleet instead. The legacy files are kept in the repo for diff reference only.

---

## e. How to start the next session

A good first message to the assistant in the next session:

> "Read HANDOVER.md, then STATUS.md §6 (open issues) and §8 (last session). What should we run next?"

If a run has just completed, instead say:

> "I ran `project_<env>_<tag>/`. The result JSON is at `Code/project_<env>_<tag>/<tag>_results.json`. Audit it."

The assistant has a calibrated audit format from the 2026-05-28 Colab gpt2 smoke run — same format works for every model.
