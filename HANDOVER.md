# HANDOVER.md — Next-session briefing

_Written 2026-05-28; **major refresh 2026-06-04** (16-variant + 6-baseline fleet, feature-dimension tables, NaN-robustness, opt unified). Author: Cowork assistant. For: the next session._

Read this first. Longer detail lives in `STATUS.md` (snapshot) and `PLAN.md` (roadmap).

---

## a. Goal

**Dissertation:** _Hallucination Detection in Large Language Models Using Internal Representations._ Extend the MIND framework (Su et al., Findings of ACL 2024) — the canonical last-token / last-layer hidden state — with additional internal-representation signals from the same forward pass. Evaluate a **16-variant ablation (A–P)** against **6 SOTA baselines** across several LLMs and **10 hallucination-detection datasets**.

Originality discipline (locked): strict separation from the senior's HalluShift work — HalluShift is included only as a baseline.

---

## b. Current status (2026-06-04)

### Experiment shape (LOCKED)
Each model has two runnable notebooks. Both evaluate the **same 10 datasets** at a fixed `QUICK_EVAL_N = 350` **deterministic first-N rows / dataset** (so `02` and `03` score the *same* rows → directly comparable; HaluEval scores 2 rows/sample).

**`02_all_variants_<model>.ipynb` — 16 variants (A–P)**
- **A–L (original 12):** canonical embedding (MIND backbone) + MIND+ scalars (`D_mean`, `V_last`, `H_mean`) + F1–F10 ablations.
- **M/N/O/P (4 new)** built on new features **F11–F16**: F11 URP (8-d unembedding-reasoning projection), F12 LTC (layer-trajectory curvature), F13 CTS (confidence-trajectory slope+variance), F14 EAR (effective attention rank), F15 PEA (prompt-echo alignment), F16 HID (head-importance divergence).
  - M = embedding+URP+LTC · N = embedding+CTS+EAR+PEA · O = embedding+URP+LTC+CTS · P = embedding+MIND+(D,V,H)+all-new.

**`03_baselines_sota_<model>.ipynb` — 6 baselines**
SAPLMA · HaloScope · HalluShift · EigenScore (the 4 SOTA) **+ MIND** (Su 2024 — supervised probe on the last-layer hidden; the method this work extends) **+ Perplexity** (unsupervised mean-token-NLL, rank-normalised).

Both notebooks write a **feature-dimension + justification table** to the output JSON: `feature_spec` / `feature_spec_table_md` (02), `baseline_feature_spec` / `baseline_feature_spec_table_md` (03).

### Per-model state
| Model | folder | records | 02 / 03 | run on |
|---|---|---|---|---|
| OPT-6.7B | `project_kaggle_opt_67b` | 2000 | 16 / 6 ✓ (also has redundant `02_..._new_...`) | Kaggle T4×2 |
| GPT-J-6B | `project_kaggle_gptj_6b` | 2000 | 16 / 6 ✓ (ran; `_nan_fill` fix applied) | Kaggle T4×2 (borderline single) |
| Mistral-7B | `project_kaggle_mistral_7b` | 2000 | 16 / 6 ✓ | Kaggle T4×2 |
| Falcon-7B | `project_kaggle_falcon_7b` | 2000 | 16 / 6 ✓ (extraction done) | Kaggle T4×2 |
| Llama-2-7B | `project_kaggle_llama2_7b` | **400 ⚠** | 16 / 6 ✓ but **BLOCKED (gated)** | Kaggle T4×2 |
| Qwen2.5-3B | `project_colab_qwen25_3b` | 2000 | 16 / 6 ✓ | Colab T4 |
| Qwen2.5-0.5B | `project_colab_qwen25_05b` | 600 | 16 ✓ (no 03 — never had one) | Colab T4 |
| gpt2 (smoke) | `project_smoke_gpt2` | 1000 | 16 / 6 ✓ (diagnostic only) | any |
| TinyLlama-1.1B | `project_colab_tinyllama_11b` | none | not built | Colab T4 |

### Robustness fixes (applied to all `02` + `03`, 2026-06-04)
- **`_nan_fill` (02):** all-NaN feature columns → 0.0 (no crash; numpy warning suppressed). _GPT-J's `F5_logit_lens_jsd` comes out all-NaN — this neutralises it so the run completes; that one feature then contributes nothing._
- **NaN-safe metrics (02 + 03):** NaN probabilities → 0.5 before `brier`/`AUROC`.
- **HalluShift 31-d features (03):** `nan_to_num` before scaling so one all-NaN sub-feature can't poison the scaler.

### Results so far
- **OPT-6.7B** (earlier, 500-cap split notebooks): variant **N best** (avg AUROC ≈ 0.551), ahead of SAPLMA (0.514) and HalluShift (0.486) on average. Comparison table at `Code/project_kaggle_opt_67b/opt_67b_variants_vs_baselines.{tex,html}`.
- GPT-J `02` ran end-to-end after the `_nan_fill` fix; Falcon feature-extraction completed; Llama-2 blocked on gated access.

---

## c. Important context

### Files / repo
- Repo root `E:\Dessertation`; GitHub `chinmoysahoo1999-spec/dissertation`, branch `main`.
- Per-model dirs `Code/project_<env>_<tag>/` with `01_data_generation`, `02_all_variants`, `03_baselines_sota`. **gpt2** uses `all_variants.ipynb` / `baselines_sota.ipynb`; **qwen25_05b** has `all_variants.ipynb` only (no 03).
- **Patcher scripts** (assistant scratch — reuse if re-applying): `patch_02_add_new_variants.py`, `patch_03_add_baselines.py`, `fix_headers.py`, `fix_correctness.py`, `do_opt.py`.
- **Git recovery point: commit `af1c901`** holds the full patched fleet. A later "cleanup" commit (`7704b22`) deleted the 02/03 notebooks; they were restored (`f758ffe`). If notebooks go missing again: `git checkout af1c901 -- <paths>`.
- **VS Code gotcha:** its Git integration keeps recreating `.git/index.lock`, which blocks every git command. Close VS Code before git ops; if needed `Remove-Item -Force .git\index.lock`.

### Hardware / running (Kaggle)
- **7B models (falcon, mistral, llama2, opt) REQUIRE Kaggle "GPU T4 × 2"** — a single T4 OOMs (7B in bf16 ≈ 14 GB fills one 14.56 GB T4, no room for activations/generation). **Do NOT select P100** — its CUDA sm_60 is incompatible with the installed PyTorch (needs sm_70+).
- GPT-J-6B is borderline on a single T4 (≈12 GB). Small models (Qwen2.5-3B/0.5B, TinyLlama) run fine on a single T4 / Colab.
- **Block-0 setup cell** (prepend to 02/03 on Kaggle) copies `<tag>_dataset_full.json` + `eval_*.parquet` from `/kaggle/input` into the working dir, so the notebooks' loaders find them (no HF download for eval data).
- **Llama-2 is gated:** requires HF access approved at `huggingface.co/meta-llama/Llama-2-7b-hf` **and** an `HF_TOKEN` set via Kaggle **Add-ons → Secrets** (never hardcode — repo is public on GitHub). The token pasted into chat earlier must be **revoked**.

### Data status
gptj / mistral / opt / qwen25_3b / falcon = **2000** records (1000/class) · gpt2 = 1000 · qwen25_05b = 600 · **llama2 = 400 (200/class — undersized; regenerate at 1000/class)** · tinyllama = none.

---

## d. Decisions already made
- 16 variants (A–P) + 6 baselines (added **MIND** + **Perplexity** as the cheap, paradigm-complementary additions); 10-dataset eval; `QUICK_EVAL_N = 350` deterministic first-N so variants and baselines score identical rows.
- Feature-dimension + justification tables emitted in both output JSONs.
- **opt_67b unified** into a single 16-variant `02_all_variants_opt_67b.ipynb` (its old M–P-only `02_all_variants_new_opt_67b.ipynb` is now redundant; safe to delete).
- Fleet-wide NaN robustness.
- HF token via Kaggle Secrets, never committed.

---

## e. What to do next (priority order)
1. **Commit the working-tree changes** (16-var/6-base/headers/correctness across all models) — currently uncommitted. Close VS Code → `Remove-Item -Force .git\index.lock` → `git add -u Code/` → commit → `git push origin main` (use a HF/GitHub PAT for auth).
2. **Llama-2:** approve HF gated access, set `HF_TOKEN` Kaggle secret, **regenerate data at 1000/class** (currently 400), then run 02+03 on **T4×2**.
3. **Run the fleet on T4×2** — one model per Kaggle session (02 and 03 can be separate sessions). Paste each `<tag>_all_variants_results.json` + `<tag>_baselines_results.json` back to the assistant.
4. **Build per-model variant-vs-baseline comparison tables** (like the OPT `.tex`/`.html`), then the cross-model summary.
5. (optional) Build TinyLlama 02+03 once its data exists; delete the redundant OPT `..._new_...` notebook.
6. While running, watch `n_eval_failures_per_dataset` — heavy OOM on a 7B during eval generation means the session isn't actually on T4×2.
