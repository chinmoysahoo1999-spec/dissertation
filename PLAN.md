# PLAN.md — Project Plan

_Written: 2026-05-23. Author: Chinmoy Sahoo. Supervisor: Dr. Swagatam Das (ISI Kolkata)._

This document is the running plan for the remainder of the dissertation. It pairs with **STATUS.md** (a snapshot of what has been done) and is the **single source of truth for what to do next**. Each section ends with a concrete acceptance criterion so the work is testable, not aspirational.

---

## 1. The thesis question, in one sentence

**Does the MIND framework, applied at the sub-1B-parameter scale with a single hidden-state feature, transfer from Wikipedia-continuation pseudo-labels to multi-task hallucination detection (QA, summarisation, dialogue) competitively with the HalluShift state-of-the-art that uses 7–8B-parameter models and multi-layer probabilistic features?**

We answer this with three experiments (§5–§7) and one ablation (§8). The literature framing for the question is given in `Literature_Survey_Chapter.pdf` §2.9.

---

## 2. Snapshot — what is done

(See STATUS.md §3 for the detailed log; recap here.)

- [x] Loaded `Qwen/Qwen2.5-0.5B` in `project.ipynb` with bf16 (4-bit toggle still available).
- [x] Fixed seven concrete deviations from MIND in `project.ipynb`, most importantly:
  - using the canonical MIND embedding `H = hidden_states[-1][0][-1]` (last token, last layer);
  - replacing the per-token `model.generate(max_new_tokens=1)` loop with a single call per article;
  - using stricter, non-over-aggressive new-entity filtering;
  - removing 4-bit quantisation for the 0.5B model.
- [x] Renamed all artefact files from `llama32_*` to `{model_tag}_*` so future model swaps don't drift.
- [x] Wrote the 11-page literature survey chapter (`Literature_Survey_Chapter.pdf` + Markdown source).
- [x] Identified that `hallucination_detection using unsupervised method.pdf` IS the MIND paper (STATUS was wrong about its absence) and that the Sharanya Dasgupta thesis covers the exact 7-dataset suite this work targets.

---

## 3. Immediate next session — sync the refactor (≈ 0.5–1 day)

Bring `src/` up to parity with the patched `project.ipynb`. This is mechanical: the notebook is now the source of truth and `src/` is behind.

- [ ] `src/config.py` → set `MODEL_NAME = "Qwen/Qwen2.5-0.5B"`. Replace any `llama32`-derived constants with `MODEL_TAG = MODEL_NAME.split("/")[-1].replace(".", "").lower()`.
- [ ] `src/config.py` → add `USE_4BIT = False`, `DTYPE = torch.bfloat16`.
- [ ] `src/model_loader.py` → drop 4-bit quantisation by default; honour `USE_4BIT`.
- [ ] `src/embeddings.py` → return 4-tuple `(canonical_mind, hds, mean1, mean2)` instead of 3-tuple. Canonical is the new MIND embedding `hidden_states[-1][0][-1]`.
- [ ] `src/dataset_gen.py` → switch to single `model.generate(max_new_tokens=entity_len + WINDOWS, output_scores=True)` per article. Update saved schema to store `embedding = canonical_mind` and `ablation = {hds, mean1, mean2}`.
- [ ] `src/dataset_gen.py` → use `f"{config.MODEL_TAG}_train.json"` / `_test.json` / `_checkpoint.json` / `_mind_best.pth`.
- [ ] `tests/test_config.py` → update the model-name assertion; rerun.
- [ ] `tests/test_classifier.py` → no change expected (still consumes `embedding`).
- [ ] `tests/test_smoke.py` → no change expected.

**Acceptance criterion (§3).** `pytest tests/ -v` shows 27 passed / 0 skipped on a machine with torch installed; `notebook_refactored.ipynb` (which drives `src/`) and `project.ipynb` produce a sample with identical `embedding` shape and `label`.

---

## 4. Generate and validate the Wikipedia pseudo-training data (≈ 1 day)

The notebook will produce per-class 1,000 samples. We additionally need to:

- [ ] Bump `TARGET_SAMPLES` to MIND's 4,096-per-class sweet spot (their Table 5).
- [ ] Run end-to-end on Colab T4 or Kaggle P100 — expected ≈ 2 h for 8,000 samples on Qwen-2.5-0.5B.
- [ ] Confirm `qwen25-05b_train.json` and `qwen25-05b_test.json` are written and have approximately balanced labels (each class within ±10 % of half).
- [ ] **Embedding-source ablation (deferred from STATUS §3.3 item):** also persist a second view, `embedding_at_entity = hidden_states[-1][0][last_prompt_token_index]` — the MIND-true "moment before the entity is emitted" embedding. Compare classifier AUROC of the two views (`embedding` vs. `embedding_at_entity`) on the same train/test split.

**Acceptance criterion (§4).** Two JSON files of ≥ 7,000 samples each (train+test ≥ 8,000 combined), both with `embedding` and `embedding_at_entity` fields and a class balance within ±10 %.

---

## 5. Multi-task evaluation extension (≈ 3–5 days, the headline contribution)

This is the big extension the supervisor and the project description ask for. We trained the MIND probe on Wikipedia; now we transfer it to 7 downstream datasets.

### 5.1 Datasets and prompt formats (verbatim from §2.7 of the survey chapter)

| Dataset | Size used | Prompt template |
|---|---|---|
| TruthfulQA | 817 | `Answer the question concisely. Q: {q} A:` |
| TriviaQA (rc.nocontext, dedup valid) | 9 960 | `Answer the question concisely. Q: {q} A:` |
| CoQA (dev, single-turn slice) | 7 983 | `Answer the question concisely based on the context: Context: {context} Q: {q} A:` |
| TydiQA-GP (English, GoldP) | 3 696 | `Answer the question concisely based on the context: Context: {context} Q: {q} A:` |
| HaluEval-QA | 10 000 | `Answer the question concisely based on the context: Context: {context} Q: {q} A:` |
| HaluEval-Summarisation | 10 000 | `{article} Please summarise the above article concisely. A:` |
| HaluEval-Dialogue | 10 000 | `You are an assistant... Knowledge: {kb} Conversations: {turns} [Assistant]:` |

### 5.2 Per-dataset evaluation protocol

For each (model, dataset) pair:

1. **Generate** the LLM's answer/summary/response with greedy decoding, `max_new_tokens = 64` (Dasgupta thesis convention).
2. **Label the generation** as hallucinated or not. Three options to compare:
   - (a) **BLEURT vs. gold** — threshold τ on BLEURT score against the gold reference; we use the threshold that maximises validation Youden's J. Matches HaloScope and HalluShift.
   - (b) **HaluEval's pre-labelled split** for HaluEval — the dataset ships with both `right_answer` and `hallucinated_answer`; treat the hallucinated one as label 1 and the right one as label 0. No threshold needed.
   - (c) **TruthfulQA-judge** for TruthfulQA — use the official `truthfulqa-judge` GPT-3-like classifier on the model output (HuggingFace `allenai/truthfulqa-judge` or equivalent).
3. **Extract** the MIND embedding `H = hidden_states[-1][0][last_token]` at *generation time*, for the *final token of the generated answer*. This is the trained probe's input.
4. **Score** with the trained MLP classifier from §3. Output: per-sample probability of hallucination.
5. **Compare** the binary predictions against the labels from step 2. Report Accuracy, Precision, Recall, F1, AUC-ROC.
6. **Baseline comparison:** SelfCheckGPT-NLI (cheap subset, N=5 samples), and (if compute permits) HaloScope-style SVD threshold using the same hidden states.

### 5.3 Outputs

- `Code/multi_task/evaluate_dataset.py` — driver script taking `--dataset {truthfulqa|triviaqa|coqa|tydiqa|halueval_qa|halueval_summ|halueval_dialog}`.
- `Code/multi_task/results.json` — flat-file results with one row per (dataset, method) tuple.
- `Code/multi_task/results.md` — pretty table for inclusion in the thesis results chapter.

**Acceptance criterion (§5).** `results.md` contains a table with rows = {TruthfulQA, TriviaQA, CoQA, TydiQA-GP, HaluEval-QA, HaluEval-Summ, HaluEval-Dialog} and columns = {Acc, Prec, Rec, F1, AUROC} for at least two methods (MIND-on-Qwen, SelfCheckGPT-NLI). HalluShift numbers may be referenced from the Dasgupta thesis without re-running.

---

## 6. HELM benchmark evaluation (optional, ≈ 1 day, if time permits)

MIND's own benchmark (HELM) ships with 3,582 human-annotated sentences from 6 LLMs, including LLaMA-2 variants. Since Qwen-2.5-0.5B is not in HELM's set, we cannot evaluate our trained probe directly on HELM, but we can:

- [ ] Train a *parallel* MIND probe on LLaMA-2-Base-7B's HELM training data (released).
- [ ] Verify the probe reaches the AUC ranges in MIND's Table 2.
- [ ] If yes, this serves as a sanity check that our re-implementation is faithful.

**Acceptance criterion (§6).** Either: reproduce a MIND result on LLaMA-2-Base-7B HELM within ± 0.03 AUROC of the paper, *or* document the deviation and move on.

---

## 7. Layer-selection and feature ablation (≈ 1 day)

We saved 4 hidden-state views during data generation. Use them.

- [ ] Train 4 separate MLPs, one per view: canonical (last-token last-layer), `hds` (all-layers last-token average), `mean1` (first-layer mean), `mean2` (last-layer mean). Report AUROC on the test split.
- [ ] Plot AUROC vs. layer index for the canonical view (intermediate-layer probes, in the style of HaloScope's Fig 6 — train one probe per layer in a Qwen-2.5-0.5B with 24 layers; cheap).
- [ ] Document which layer / view wins on this small model — fills the open problem in survey §2.8 ("layer selection").

**Acceptance criterion (§7).** A table + line plot of AUROC across {4 views} ∪ {per-layer probe sweep}. The thesis results chapter cites this directly.

---

## 8. Calibration ablation (≈ 0.5 day)

Following Kadavath *et al.* 2022, hallucination detectors should be evaluated as calibrated probabilities, not only as binary classifiers.

- [ ] Compute Brier score and Expected Calibration Error (ECE) for the MIND classifier on each of the 7 datasets.
- [ ] Fit Platt scaling on the validation split and report whether it improves Brier/ECE.

**Acceptance criterion (§8).** Brier + ECE columns added to `Code/multi_task/results.md`.

---

## 9. Write-up — dissertation chapters

The thesis will likely follow this structure. Cross-references to existing material:

| Chapter | Source | Status |
|---|---|---|
| Ch 1 — Introduction | new | to write |
| Ch 2 — Related Work | `Literature_Survey_Chapter.pdf` (this session) | done |
| Ch 3 — Methodology (MIND framework + Qwen choice + our adaptations) | `project.ipynb` + STATUS §3 + PLAN §3 | partial — needs prose |
| Ch 4 — Experiments and Results | `Code/multi_task/results.md` (PLAN §5/§6/§7/§8) | not started |
| Ch 5 — Ablations | PLAN §4 + §7 + §8 | not started |
| Ch 6 — Discussion (gap to HalluShift; future work) | `Literature_Survey_Chapter.pdf` §2.9 | partial |
| Ch 7 — Conclusion | new | to write |
| Appendix — code listings + hyperparameters | `Code/src/` + `project.ipynb` | already in repo |

**Acceptance criterion (§9).** All 7 chapters drafted; total length 70–90 pages. Ch 2 (already a chapter) feeds in directly.

---

## 10. Risks and contingencies

| Risk | Mitigation |
|---|---|
| HuggingFace `datasets` rate-limiting on Wikipedia streaming during §4 | Cache the 8,000 selected articles locally on first run; treat as a static input thereafter |
| Qwen-2.5-0.5B's NER spaCy mismatch (BPE subword splits inside entities make `find_first_and_next_token` return `[]` more often than for LLaMA tokenisers) | Drop a sample if token search fails (already the behaviour). Monitor the drop rate; if > 30 % consider switching to Qwen-2.5-1.5B |
| GPU memory exhaustion when running HaluEval-Summarisation on a free Colab T4 (long contexts) | Chunk article to 2 048 tokens before generation; the article appears in the prompt only, not the prediction |
| BLEURT model download fails inside Colab/Kaggle | Use `unbabel-comet` or `bert-score` as fallback metric; rerun pretty table |
| Dataset distribution shift (Wikipedia-trained probe doesn't transfer well to dialogue) | Frame as a *finding* rather than a failure: the dissertation can report transferability gap as a contribution |

---

## 11. Quick-reference commands

Run the entire pipeline:

```bash
cd Code
pip install -r requirements.txt
jupyter execute project.ipynb               # data gen + train + eval on Wikipedia
python multi_task/evaluate_dataset.py \
    --dataset truthfulqa \
    --model Qwen/Qwen2.5-0.5B \
    --probe qwen25-05b_mind_best.pth         # repeat for the other 6 datasets
```

Rebuild this session's artefacts:

```bash
cd Code
python _patch_notebook.py    # rewrites project.ipynb
python _build_pdf.py         # rebuilds Literature_Survey_Chapter.pdf
```

---

## 12. Decision log

- **2026-05-23 (this session) — Model choice locked to Qwen 2.5-0.5B (base).** Rationale: smaller than anything in the MIND family literature so far (smallest was OPT-6.7B), fits free Colab T4 in bf16 without quant overhead, and MIND probes raw next-token prediction so the base variant is the correct match (not Instruct).
- **2026-05-23 — Embedding standard fixed: canonical = last-token, last-layer.** Rationale: matches MIND §3.2.2 exactly; the previous notebook stored mean-pooled embeddings which corresponds to a different row of MIND's ablation table (0.6986 acc) than the chosen one (0.7123 acc).
- **2026-05-23 — Single `model.generate()` call per article.** Rationale: ~20-30× fewer kernel launches; same logic since we read intermediate scores from `output.scores`.
- **2026-05-23 — Survey chapter shipped at 11 pages.** Rationale: target was 10–15; landed inside the window with room to grow if needed.
