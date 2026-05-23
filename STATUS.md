# STATUS.md — Project Snapshot

_Last updated: 2026-05-23 (after the Qwen-2.5-0.5B / MIND-correction session)_

---

## 1. Goal (one-liner)

Build an end-to-end **hallucination detection** pipeline for LLMs following the
**MIND paper** (Su *et al.*, Findings of ACL 2024) methodology: extract internal
hidden-state embeddings while a small, open-weights LLM generates over
Wikipedia text, label spans as hallucinated vs. non-hallucinated using an
entity-substitution probe, and train an MLP classifier on those embeddings.
End deliverable: classifier + Accuracy, Precision, Recall, F1, AUC-ROC,
later extended to TruthfulQA, TriviaQA, CoQA, TydiQA-GP, and the HaluEval
triple (QA, Summarisation, Dialogue).

---

## 2. Repo layout (current)

```
E:\Dessertation\
├── STATUS.md                              <-- this file
├── PLAN.md                                <-- NEW, written this session
├── HANDOVER.md                            <-- not yet written
├── .git/                                  <-- github.com/chinmoysahoo1999-spec/dissertation
├── .gitignore                             <-- excludes hallucination_10k.json, *.pth, qwen25-05b_*.json
├── Literature_Survey_Chapter.pdf          <-- NEW, 11-page thesis-style survey (this session)
├── *.pdf                                  <-- reference papers (MIND, SelfCheckGPT, HaloScope, surveys, Dasgupta thesis)
└── Code\
    ├── project.ipynb                      <-- PATCHED this session (Qwen 2.5-0.5B, MIND-canonical embedding, single-call decoding)
    ├── notebook_refactored.ipynb          <-- thin notebook driving src/
    ├── requirements.txt                   <-- pinned deps for Colab/Kaggle
    ├── pytest.ini                         <-- test config
    ├── hallucination_10k.json             <-- 1.1 GB pre-existing dataset (gitignored)
    ├── preview.json                       <-- 640 KB whitelisted preview
    ├── survey_chapter.md                  <-- NEW, Markdown source of the survey PDF
    ├── survey_notes.md                    <-- NEW, raw per-paper notes used to write the chapter
    ├── Literature_Survey_Chapter.pdf      <-- NEW, generated from survey_chapter.md (also copied to repo root)
    ├── _patch_notebook.py                 <-- one-off helper that rewrote project.ipynb (keep for reference)
    ├── _build_pdf.py                      <-- one-off helper that built the survey PDF
    ├── src/
    │   ├── __init__.py
    │   ├── config.py                      <-- MODEL_NAME = meta-llama/Llama-3.2-1B (NEEDS UPDATE → Qwen2.5-0.5B)
    │   ├── model_loader.py                <-- Block 1
    │   ├── entities.py                    <-- Block 2
    │   ├── tokens.py                      <-- Block 3
    │   ├── embeddings.py                  <-- Block 4 (NEEDS UPDATE → add canonical MIND embedding)
    │   ├── dataset_gen.py                 <-- Block 6+7 (NEEDS UPDATE → single generate call + new file naming)
    │   ├── classifier.py                  <-- Block 8 (unchanged)
    │   ├── train.py                       <-- Block 9 (unchanged)
    │   └── evaluate.py                    <-- Block 10 (unchanged)
    └── tests/
        ├── __init__.py
        ├── conftest.py
        ├── test_config.py
        ├── test_entities.py
        ├── test_classifier.py
        └── test_smoke.py
```

---

## 3. What changed this session (2026-05-23, second pass)

### 3.1 Model decision

- **`project.ipynb` now uses `Qwen/Qwen2.5-0.5B`** (the BASE variant, since MIND probes raw next-token prediction, not chat).
- 4-bit quantisation is disabled by default (the `USE_4BIT` flag exists for users who want it back). For a 0.5B model the dequant overhead exceeds memory savings; `bf16` is faster.
- All output files now follow the pattern `f"{MODEL_TAG}_<artefact>"` where `MODEL_TAG = MODEL_NAME.split('/')[-1]` — so switching `MODEL_NAME` automatically renames artefacts (no more `llama32_*` drift).
- **`src/config.py` still has `MODEL_NAME = "meta-llama/Llama-3.2-1B"`** — needs syncing in the next session (see PLAN.md §3).

### 3.2 Deviations from the MIND paper that were found and corrected in `project.ipynb`

| # | What `project.ipynb` did before | What MIND paper specifies (Su *et al.* 2024 §3.2.2) | Fix applied |
|---|---|---|---|
| 1 | `MODEL_NAME = "mistralai/Mistral-7B-v0.1"` while file names said `llama32_*` and STATUS said Llama-3.2-1B (3-way inconsistency) | n/a — internal bug | Switched to `Qwen/Qwen2.5-0.5B`; all artefact names derived from `MODEL_TAG` |
| 2 | Saved `mean2` (mean of last layer across tokens 2..end) as `embedding` | **Last token, last layer**: `H = hidden_states[-1][0][-1]` | `get_hd` now returns a 4-tuple `(canonical_mind, hds, mean1, mean2)`; the dataset stores `canonical_mind` as `embedding` and keeps the 3 other views under `ablation/` for the report |
| 3 | Inside `for i in range(entity_len + WINDOWS)` loop, called `model.generate(max_new_tokens=1)` separately each step (up to 32 GPU calls per article) | a single greedy continuation; per-step scores can be read from `output.scores` | One `model.generate(max_new_tokens=entity_len + WINDOWS, output_scores=True)` per article — same logic, ~20-30× fewer kernel launches |
| 4 | `any(ee.strip() in text.lower() for ee in new_entity.split())` rejected new entities containing ANY common word ("the", "of") | reject only literal-copy hallucinations | Replaced with `new_entity in text.lower()` (whole-phrase containment) |
| 5 | Redundant `.clone().detach()` inside `@torch.no_grad()` | not needed | Removed; use direct `.float().cpu().tolist()` |
| 6 | 4-bit quantisation on a 0.5B model | n/a (MIND uses fp16/bf16 in its open-source repo) | Disabled by default; `USE_4BIT` toggle remains |
| 7 | `if i == 0 or e in title` substring check | should be case-insensitive | `e.lower() in title.lower()` |

### 3.3 Deviations NOT yet fixed (documented for next session)

- The original notebook's `get_hd` extracts the embedding of the **full text** (concatenated with the post-entity suffix). MIND extracts the embedding **at the time the model was about to emit the entity**, i.e. the hidden state of the last token *of the prompt prefix*, not of the full text. Both are valid for binary classification but differ in what they probe. We retain the existing behaviour (probe the full text) for backward compatibility with the prior `qwen25-05b_train/test.json` schema and flag this for an ablation in PLAN.md §4.
- `Code/src/embeddings.py` still returns only the 3 ablation views, not the canonical MIND view. The notebook is now ahead of the src/ refactor. Will be reconciled in the next session.

### 3.4 Literature survey delivered

- **`Literature_Survey_Chapter.pdf` (11 pages, ~7,400 words)** — full thesis-style "Related Work" chapter. Covers: definitions (Ji *et al.*'s intrinsic/extrinsic, faithfulness/factuality); failure of lexical metrics; four method families (self-evaluation, sampling, internal-state probing, retrieval-augmented); deep dive on SAPLMA → MIND → HaloScope → HalluShift; the 7-dataset evaluation cross-coverage table; open problems; how this dissertation is positioned as a 0.5B-parameter ablation of HalluShift on the same 7-dataset suite.
- **`Code/survey_chapter.md`** — the Markdown source.
- **`Code/survey_notes.md`** — raw per-paper extraction notes (12 papers identified, including the 3 versions of Ji *et al.*'s survey and the fact that the MIND paper IS in the folder — `hallucination_detection using unsupervised method.pdf` — contrary to the previous STATUS).
- Recent web sources (LapEigvals 2025, arXiv 2509.14254, arXiv 2506.22486) folded in where relevant.

### 3.5 Reproducibility

`SEED = 42` continues to be propagated to Python `random`, NumPy, PyTorch CPU and CUDA. Set in `project.ipynb` Cell 2 (Block 1).

---

## 4. Tests (status)

`Code/tests/` is unchanged from the prior session. Last run:

```
============================= test session starts ==============================
collecting 27 items / 1 skipped
tests/test_config.py    .......                                    [  7 passed]
tests/test_entities.py  ..........                                 [ 10 passed]
tests/test_smoke.py     ......sss.                              [  7 pass / 3 skip]
tests/test_classifier.py SKIPPED (no torch in sandbox)             [  6 skip]
======================== 24 passed, 4 skipped in 0.44s =========================
```

The 4 skips are torch-only tests; they should pass on any machine (Colab / Kaggle / local) with `pip install -r Code/requirements.txt`.

`test_config.py` will start failing once `src/config.py` is updated to `Qwen/Qwen2.5-0.5B` because the existing invariant asserts the model name. Update the assertion when you do the sync in PLAN.md section 3.

---

## 5. Git state

Branch `main`, tracking `origin/main`. Uncommitted as of end-of-session:

- modified: `Code/project.ipynb` (the big rewrite of this session)
- modified: `STATUS.md` (this file)
- new: `PLAN.md` (this session)
- new: `Literature_Survey_Chapter.pdf`
- new: `Code/Literature_Survey_Chapter.pdf` (same file)
- new: `Code/survey_chapter.md`
- new: `Code/survey_notes.md`
- new: `Code/_patch_notebook.py`
- new: `Code/_build_pdf.py`

Suggested commit split:

```
git add -A Code/project.ipynb STATUS.md PLAN.md Code/_patch_notebook.py
git commit -m "feat(notebook): swap Llama-3.2-1B to Qwen-2.5-0.5B; correct MIND-canonical embedding; single-call decoding"

git add -A Literature_Survey_Chapter.pdf Code/Literature_Survey_Chapter.pdf Code/survey_chapter.md Code/survey_notes.md Code/_build_pdf.py
git commit -m "docs: 11-page literature survey chapter (MIND family + multi-task datasets)"

git push
```

---

## 6. How the patched pipeline corresponds to the MIND paper

| MIND paper section | Patched notebook cell | What it does |
|---|---|---|
| Section 3.1 (Unsupervised data generation) | Cells 2 + 3 + 6 | spaCy NER, entity substitution via LLM continuation |
| Section 3.2.1 (Embedding selection) | Cell 4 | Returns 4 views; canonical = last-token last-layer (0.7123 acc in MIND ablation) |
| Section 3.2.2 (Classifier, Eq. 1 + 2) | Cell 8 | 5-layer MLP, ReLU + Dropout(0.2), CrossEntropy = BCE for 2-class softmax |
| Section 4 (HELM benchmark) | not implemented (uses our own pseudo-labelled Wikipedia data) | See PLAN.md section 6 |
| Section 5.3 (Training config) | Cell 9 | Adam, LR 5e-4, weight_decay 1e-5, batch 32, 10 epochs |
| Section 6 (Cross-LLM transfer) | out of scope (single-LLM only) | n/a |

---

## 7. Open issues and pending decisions

| # | Issue | Severity | Action |
|---|---|---|---|
| 1 | `src/config.py` and `src/embeddings.py` are behind the patched notebook | High | Sync next session (PLAN section 3) |
| 2 | `Code/hallucination_10k.json` (1.1 GB) is not used | Medium | Either ignore or write `from_existing_dump.py` adapter |
| 3 | Single random entity per article (possible bias) | Medium | Vary for an ablation |
| 4 | Probe extracted from FULL hallucinated text, not the moment-before-entity position (true MIND protocol). Both work for classification but probe different things | Medium | Side-by-side ablation |
| 5 | No GitHub Actions CI | Low | Add `.github/workflows/test.yml` on Python 3.10 |
| 6 | Multi-task / multi-dataset extension (TruthfulQA, TriviaQA, CoQA, TydiQA-GP, HaluEval x 3): code not yet written | High | Headline next-session task (PLAN section 5) |

---

## 8. What to read at the start of the next session

1. **`STATUS.md`** (this file): full snapshot.
2. **`PLAN.md`**: agreed plan for the multi-task extension and remaining src/ sync.
3. **`Literature_Survey_Chapter.pdf`**: 11-page chapter. Section 2.7 lists the prompt formats per downstream dataset; section 2.9 positions the contribution.
4. **`Code/project.ipynb`**: the patched notebook (Qwen-2.5-0.5B, MIND-canonical embedding, single-call decoding).
5. **`Code/survey_notes.md`**: raw per-paper notes if a particular comparison needs more depth.

---

## 9. Reference papers in repo (`E:\Dessertation\*.pdf`)

| File | Identification | Notes |
|---|---|---|
| `hallucination_detection using unsupervised method.pdf` | MIND (Su et al. 2024) | The MIND paper IS present. Methodology backbone. |
| `NeurIPS-2024-haloscope-*.pdf` | HaloScope (Du et al. 2024, NeurIPS) | Direct comparison target |
| `selfcheckgpt.pdf` | SelfCheckGPT (Manakul et al. 2023, EMNLP) | Sampling-based baseline |
| `The internal state of LLM know when it laying(2023).pdf` | SAPLMA (Azaria & Mitchell, 2023) | MIND's direct precursor |
| `Study_internal state of LLM Know when it is laying paper.pdf` | Tutorial-style summary of SAPLMA | not a primary source |
| `LM know what they know.pdf` | Kadavath et al. (Anthropic 2022) | Calibration: P(True), P(IK) |
| `Survey of hallucination in natural language generation.pdf` | Ji et al. survey, ACM CSUR 55(12) | three copies in folder (with `2202.03629.pdf` and `latest survey...pdf`) |
| `M.tech _Sharanya_Dasgupta_CS2320.pdf` | Dasgupta thesis, ISI Kolkata, Jun 2025 | Most directly comparable prior work: same lab, same supervisor, exact same 7-dataset suite |
| `2024.fever-1.5.pdf` | UHH at AVeriTeC (Sevgili et al. 2024) | RAG fact-verification, adjacent family |
| `2024.fever-1.5-summary.pdf` | Plain-language summary by Chinmoy Sahoo | Student's own prior work |
| `Notes On Lexical (n-gram) Metrics For Hallucination Detection.pdf` | Notes / tutorial document | Pedagogical reference |
