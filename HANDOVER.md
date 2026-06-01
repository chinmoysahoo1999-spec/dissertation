# HANDOVER.md — Next-session briefing

_Written: 2026-05-28; updated 2026-06-01. Author: Cowork assistant. For: the next Cowork session._

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
* **(2026-05-29) unified `all_variants.ipynb` ran on gpt2 (500/class).** Pipeline verified; SDPA attention bug fixed; all 9 F-features extracted with 0 failures. But results are noise-level: Wikipedia AUROC 0.32–0.51, multi-task avg AUROC 0.40–0.49, no consistent variant winner. gpt2 too small to produce meaningful detector. Re-cap: pipeline is correct, signal needs bigger backbone.
* **(2026-05-29) `baselines_sota.ipynb` shipped for gpt2** — 4 SOTA baselines (SAPLMA, HaloScope, EigenScore/INSIDE, HalluShift) faithfully ported from their official GitHub repos. Reuses `gpt2_smoke_dataset_full.json` so AUROCs are directly comparable to the 12 variants. NOT YET RUN — that's the next student action.
* **(2026-05-29) `project_colab_qwen25_05b/all_variants.ipynb` NOT yet run** — should be the headline ablation since the 0.5B backbone is the smallest one expected to show real signal.

### What still needs to happen

In priority order:

1. **Run `project_smoke_gpt2/baselines_sota.ipynb`** on Colab — this is the cross-method comparison. Will produce `gpt2_smoke_baselines_results.json`. Compare against `gpt2_smoke_all_variants_results.json` to see if any of our 12 variants beats HalluShift / HaloScope / SAPLMA / EigenScore on the same backbone. **Expected runtime: ~60–90 min** (the dataset/feature cache reuses Stage 2 from all_variants so no data-gen; HaloScope sweep + EigenScore K=10 wiki eval dominate the cost).
1b. **(NEW 2026-05-30) `project_colab_qwen25_3b/` is the headline large-model run on Colab.** Both `all_variants.ipynb` (300/class, ~50 min) and `baselines_sota.ipynb` (~70 min) deployed. Combined ~2h on Colab T4 — fits a single session. Paste both result JSONs back when done; assistant will repeat the comparison table at Qwen-3B scale (the gpt2 results were too noisy — Qwen-3B should give real signal).
1c. **(NEW 2026-05-30) `project_kaggle_llama2_7b/` is the gated large-model run on Kaggle T4×2.** Both notebooks deployed (200/class, combined ~2.5h). Requires HF gated-access approval at https://huggingface.co/meta-llama/Llama-2-7b-hf and pasting an HF token in BLOCK 0 of each notebook.

1d. **(NEW 2026-05-30 final) Data-generation phase LOCKED IN for 6-model fleet.** Sample size bumped to 1000/class across all large models. Two new models added: Mistral-7B-v0.1 and OPT-6.7B. All split into 3-notebook layout. **Next student action: run `01_data_generation.ipynb` for each of the 6 large models, one at a time on Kaggle (or in parallel across Kaggle accounts).** Per-model runtime ~100 min. Total GPU time ~10h across all 6 models, well within Kaggle 30h/week budget. After 01 produces `<tag>_dataset_full.json`, the analysis notebooks 02 and 03 can run in parallel on two more accounts.

| Model | Where | 01 runtime | Gated? |
|---|---|---|---|
| Qwen-3B | Colab T4 | ~50 min (already ran with old config) | no |
| Llama-2-7B | Kaggle T4×2 | ~100 min | YES (license + token) |
| Falcon-7B | Kaggle T4×2 | ~100 min | no |
| GPT-J-6B | Kaggle T4×2 | ~100 min | no |
| Mistral-7B-v0.1 | Kaggle T4×2 | ~100 min | no |
| OPT-6.7B | Kaggle T4×2 | ~100 min | no |

**Schema-sufficiency confirmed.** `<tag>_dataset_full.json` contains `{text, label, embedding, D_mean, V_last, H_mean, entity, title}` per record. This is sufficient input for ANY future baseline that takes (text, label) and re-runs the LLM to extract its own features. The 4 baselines in `03_baselines_sota.ipynb` (SAPLMA, HaloScope, EigenScore, HalluShift) all work this way; any future addition (Lookback Lens, Semantic Entropy, SelfCheckGPT, etc.) will work identically. **No need to re-run data-gen when adding new baselines.**

### Shared eval-dataset downloader (2026-05-30 final)

`Code/eval_datasets/00_download_eval_datasets.ipynb` — runs ONCE on any HF-authenticated session. Produces:
- `eval_truthfulqa.parquet`, `eval_triviaqa.parquet`, `eval_coqa.parquet`, `eval_tydiqa.parquet`
- `eval_halueval_qa.parquet`, `eval_halueval_summ.parquet`, `eval_halueval_dialog.parquet`
- `eval_datasets.zip` (all 7 bundled)

**Two workflows for distributing the parquet files to the per-model analysis sessions:**

| Where | How to upload | Files appear at |
|---|---|---|
| Kaggle | New Dataset → upload `eval_datasets.zip` (slug `dissertation-eval-datasets`). Each notebook: Add Input → pick the dataset. | `/kaggle/input/dissertation-eval-datasets/eval_*.parquet` |
| Colab | Upload zip to session, then `!unzip eval_datasets.zip` | Current working dir |

The `_find_local_parquet(label)` function inside every `02` and `03` notebook checks 7 candidate paths automatically. If a parquet is found, prints `(LOCAL: <path>)` and skips the HF download. If not found, prints `(HF loader #N)` and falls back. **No code change required per model** — the same notebook works in both modes.

### 5 SOTA baselines + 10 datasets locked in (Semantic Entropy removed 2026-06-01) (2026-05-30 final)

`03_baselines_sota.ipynb` evaluates **5 baselines** on **10 datasets** + Wikipedia held-out:

| # | Baseline | Paradigm | Paper / Repo |
|---|---|---|---|
| 1 | SAPLMA | supervised, single-layer probe | Azaria & Mitchell 2023 |
| 2 | HaloScope | unsupervised, spectral + non-linear probe | Du et al. NeurIPS 2024 — [github.com/deeplearning-wisc/haloscope](https://github.com/deeplearning-wisc/haloscope) |
| 3 | EigenScore (INSIDE) | sampling-based, geometric (K×K cov log-det) | Chen et al. ICLR 2024 — [github.com/D2I-ai/eigenscore](https://github.com/D2I-ai/eigenscore) |
| 4 | HalluShift | supervised, Wasserstein + token-prob | Dasgupta IJCNN 2025 — [github.com/sharanya-dasgupta001/hallushift](https://github.com/sharanya-dasgupta001/hallushift) |
| 5 | **Lookback Lens** | supervised, **attention-based** | Chuang et al. EMNLP 2024 — [github.com/voidism/Lookback-Lens](https://github.com/voidism/Lookback-Lens) |
| ~~6~~ | ~~**Semantic Entropy**~~ — REMOVED 2026-06-01 (NLI loader + K-sample clustering added ~3-4 hr per 7B model; broke 2-3 h per-session budget) | sampling-based, NLI clustering | Farquhar et al. Nature 2024 |

Datasets: TruthfulQA, TriviaQA, CoQA, TydiQA-GP, HaluEval-{QA, Summ, Dialog}, NQ-Open, HotpotQA distractor, PopQA. Plus Wikipedia held-out (in-distribution).

**Why this exact set of 5 baselines (after Semantic Entropy removal on 2026-06-01):**

* SAPLMA — establishes a *floor*: simplest possible supervised probe. Anything fancier must beat raw hidden + MLP.
* HaloScope — current best **unsupervised** detector. Critical for the "what if no labels?" thread.
* EigenScore — current best **sampling-based** detector with geometric scoring. Different paradigm from hidden-state methods.
* HalluShift — the *senior's* work. The dissertation must beat this on Llama-2-7B to claim contribution. Required.
* Lookback Lens — adds the **attention paradigm**. Cheap, complements hidden-state and sampling baselines.
* ~~Semantic Entropy — Nature 2024, most-cited 2024 hallucination paper.~~ **REMOVED 2026-06-01** for runtime reasons (cross-encoder NLI inference between every K-sample pair); SE is acknowledged in the dissertation as published prior art but not run as a baseline. Eval-time budget on 7B models reallocated to LookbackLens and the multi-task eval extension.

### Auto-download behavior (per notebook)

| Notebook | Downloads |
|---|---|
| `01_data_generation.ipynb` | `<tag>_dataset_full.json` |
| `02_all_variants.ipynb` | `<tag>_all_variants_results.json` + 12 × `<tag>_variant_<A..L>_best.pth` |
| `03_baselines_sota.ipynb` | `<tag>_baselines_results.json` + 4 × `.pth` (SAPLMA, HaloScope, HalluShift) + 1 × `.pkl` (Lookback Lens) |

Excluded from auto-download (regeneratable, large): `<tag>_dataset_with_features.json` (~100 MB+), `<tag>_baseline_feature_cache.json` (~150 MB+ to ~1 GB).
2. **Run `project_colab_qwen25_05b/all_variants.ipynb`** on Colab T4 (~50–60 min at 500/class). This is the **headline ablation** since 0.5B is the smallest backbone expected to show real signal. The acceptance criterion is **E ≥ A + 0.02** (the original MIND+ story holds) AND ideally **K > E** (adding F1+F5+F7 on top of E gives more lift than E alone). If yes, the feature stack is real — scale up to bigger models. If no, rethink the feature stack before burning GPU budget.
3. **Apply for Llama-2-7B HuggingFace gated access** at https://huggingface.co/meta-llama/Llama-2-7b-hf (approval typically < 1 hour). Required before `project_kaggle_llama2_7b/` can run.
4. **(Conditional on step 2 passing)** Migrate the other 6 model notebooks to the unified `all_variants.ipynb` design. The builder script is `outputs/build_all_variants.py` (in the assistant's scratch directory) — extending it is a 2-line edit to the `MODELS` list.
5. **Otherwise (step 2 fails)** Reopen the feature design before committing GPU budget. The assistant has a ranked menu of 10 candidate features; the F-stack can be re-ordered or replaced.
6. **Patch STATUS.md §3** to fix the three mis-glossed HalluShift feature definitions (mtp / Mps / Mg — see STATUS.md Issue #10). Required before the lit-review chapter goes out, regardless of step 2 outcome.
7. **Run the 4 Colab notebooks** (existing per-model fleet, not unified) once the ablation question is answered:
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
| Per-model run directories | `E:\Dessertation\Code\project_<env>_<tag
---

## f. 2026-06-01 session delta — Semantic Entropy removed + eval-dataset time savings

**What changed since the last HANDOVER:**

1. **Semantic Entropy baseline REMOVED from all 6 large-model `03_baselines_sota.ipynb` + the source `Code/project_smoke_gpt2/baselines_sota.ipynb`.** Backup of pre-removal source: `Code/project_smoke_gpt2/baselines_sota.ipynb.bak_se_remove`. Removal script: `outputs/remove_semantic_entropy.py` (re-emits all 6 copies after patching the source).

2. **The final baseline set is now 5:** SAPLMA · HaloScope · HalluShift · EigenScore (INSIDE) · LookbackLens.

3. **Eval-dataset pre-download (`Code/eval_datasets/00_download_eval_datasets.ipynb`) time savings — quantified:**

   | Scenario | Without pre-download | With pre-download | Saved |
   |---|---|---|---|
   | Per model (02 + 03 each, 10 datasets) | 20.5 min HF resolve | 0.2 min parquet load | **~20 min / model** |
   | All 6 models, single account | ~123 min | ~12 min (one 00 download + 12 parquet loads) | **~111 min ≈ 1.9 h** |
   | All 6 models, 6 parallel accounts | ~123 min | ~63 min (6 × 00 downloads + 12 parquet loads) | **~60 min ≈ 1.0 h** |

   Plus qualitative wins: no HF rate-limit failures under parallel runs, no dataset-namespace breakage (e.g. `truthful_qa → truthfulqa/truthful_qa`), no mid-load kernel disconnects.

4. **Why SE was removed (for the dissertation defense, if asked):** Semantic Entropy requires loading an extra ~180M-parameter cross-encoder NLI model (`cross-encoder/nli-deberta-v3-base`) and running bidirectional NLI inference between every pair of K stochastically-sampled responses for every eval row. On a 7B-class generator this is on the order of 3-4 hours extra compute per model — incompatible with the 2-3 h per-session Colab/Kaggle budget. SE is still acknowledged as prior art in the literature survey chapter; only the eval-time baseline run is dropped.

**What to do next (priority unchanged):**

1. Run `Code/eval_datasets/00_download_eval_datasets.ipynb` ONCE per Colab/Kaggle account to produce `eval_*.parquet` files (10 datasets).
2. Run `01_data_generation.ipynb` for each of the 6 large models (1000/class, ~100 min each).
3. Run `02_all_variants.ipynb` and `03_baselines_sota.ipynb` for each model — both pick up the local parquets automatically via `_find_local_parquet(...)`. Combined target ~2-2.5 h per model.
4. Paste each model's `<tag>_baselines_results.json` and `<tag>_all_variants_results.json` back to the assistant for the cross-method comparison table at full scale.
