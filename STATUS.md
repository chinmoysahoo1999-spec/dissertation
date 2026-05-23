# STATUS.md — Project Snapshot

_Last updated: 2026-05-23 (initial creation, first session)_

---

## 1. Goal (one-liner)

Build an end-to-end **hallucination detection** pipeline for LLMs based on the **MIND paper** methodology: extract internal hidden-state embeddings while a quantized LLM generates over Wikipedia text, label spans as hallucinated vs. non-hallucinated using an entity-substitution probe, and train an MLP classifier on those embeddings.

This is a dissertation / M.Tech style research project. The end deliverable is a classifier + evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix) demonstrating the approach works.

---

## 2. Repo layout (as of today)

```
E:\Dessertation\
├── *.pdf                              <-- reference papers (MIND, SelfCheckGPT, HaloScope, surveys, etc.)
├── Code\
│   ├── project.ipynb                  <-- THE codebase. 11 cells. ~53 KB.
│   ├── hallucination_10k.json         <-- 1.1 GB pre-generated dataset (likely from a prior run)
│   ├── preview.json                   <-- 640 KB preview (first 4 samples of the 10k)
│   ├── check.py                       <-- 12-line helper: slices preview.json from hallucination_10k.json
│   └── nn.py                          <-- UNRELATED (Iris perceptron / coursework leftover) -- candidate for removal
└── STATUS.md / HANDOVER.md / PLAN.md  <-- docs (this file is new; the other two not yet created)
```

Git: **not initialised yet** (`git status` returns "not a git repository"). This is important — see Section 6.

---

## 3. What the notebook currently does — block by block

| Cell | Block | Purpose | Status |
|---|---|---|---|
| 0 | Install | `pip install` bitsandbytes, transformers, accelerate, datasets, spacy, nltk + spaCy model | Works (Colab/Kaggle setup) |
| 1 | Model load | Loads `mistralai/Mistral-7B-v0.1` with 4-bit NF4 quantization via `BitsAndBytesConfig`. Sets `TOPK_FIRST_TOKEN=4`, `WINDOWS=16`, `TARGET_SAMPLES=1000`. Loads spaCy `en_core_web_sm`. | Works on GPU. Comment in code mentions Llama-3.2-1B but actual `MODEL_NAME` is Mistral-7B. Inconsistency to resolve. |
| 2 | Entity extraction | `delete_substrings`, `find_boundaries`, `get_entities` — uses spaCy NER, expands to word boundaries, returns `(entity, char_idx)` tuples. | Works |
| 3 | Token-level generation | `find_first_and_next_token` — inserts a `@` marker after the entity, tokenizes, finds the first token of the entity and the token right after it. Returns `[first_token, next_token, entity_len, last_id]`. | Works |
| 4 | Embedding extraction | `get_hd(text)` — returns 3 embeddings: avg of all layers' last token, mean of first layer (from pos 2), mean of last layer (from pos 2). MIND paper uses **mean of last layer**. | Works |
| 5 | Sanity test | Runs the full pipeline on a single Einstein paragraph. Includes an explicit `attention_mask` fix for the HF warning. Outputs hallucinated text and embedding dims. | Works |
| 6 | Dataset generation | Streams `wikimedia/wikipedia 20231101.en`, picks random valid entity per article, runs the generation+entity-mismatch test, builds `dataset_hall` (label=1) and `dataset_non_hall` (label=0) lists with `mean2` embeddings. Checkpoints every 500 samples to `llama32_checkpoint.json`. Target: 1000 + 1000 samples. | Logic works. Runtime: **1–2 hours on a free GPU** for the full 2000. |
| 7 | Train/test split | 80/20 shuffle, writes `llama32_train.json` and `llama32_test.json`, frees the LLM from VRAM. | Works |
| 8 | Classifier definition | `MINDClassifier`: 5-layer MLP (input → 512 → 256 → 128 → 64 → 2), ReLU + dropout(0.2) after first layer. `MINDDataset` wrapper. | Works (matches MIND paper Eq. 1) |
| 9 | Training | 10 epochs, Adam, `lr=5e-4`, `weight_decay=1e-5`, BCE loss, batch 32. Saves best by test accuracy to `llama32_mind_best.pth`. | Works |
| 10 | Evaluation | Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix, per-class report, 20 random qualitative examples. | Works. Prints a "Llama-3.2-1B" line in the final summary — stale string, doesn't match current `MODEL_NAME`. |

---

## 4. What works (verified by reading)

1. End-to-end pipeline is implemented and self-consistent in a single notebook.
2. 4-bit quantization is wired correctly for free-tier Colab/Kaggle GPUs.
3. The `attention_mask` fix is present in Block 5 (no HF warnings in test path).
4. Embedding extraction follows the MIND paper convention (last-layer mean, `start_at=2`).
5. The classifier architecture and training loop match the paper's Equation 1 / 2.
6. Checkpointing every 500 samples means partial runs are recoverable.

## 5. What is broken / weak / needs attention

| # | Issue | Severity | Notes |
|---|---|---|---|
| 1 | **Model name inconsistency** — code uses Mistral-7B but file names (`llama32_*.json`, `llama32_mind_best.pth`) and Block 10's printout still say Llama-3.2-1B. | Low | Pure cosmetic / reproducibility risk. Pick one and rename. |
| 2 | **`nn.py` looks unrelated** (Iris dataset perceptron). | Low | Either delete or move to an `archive/` folder. |
| 3 | **`hallucination_10k.json` (1.1 GB) is not used by the notebook.** Notebook regenerates from Wikipedia. | Medium | Decide: is this an older 10k dump we should be using instead of re-generating 2k each run? Could shorten experiments massively. |
| 4 | Dataset generation loop **uses one random entity per article** — small entity pool, may bias the dataset. | Medium | MIND paper iterates more aggressively. Worth tuning later. |
| 5 | **Only `mean2` embedding (last-layer mean) is saved to disk** — the other two extracted embeddings (`hds`, `mean1`) are discarded. | Medium | If we want ablations across layers, we should save all three. |
| 6 | No deterministic seeding for `random.choice` / shuffle / model — different runs produce different datasets. | Medium | Cheap fix: set `random.seed`, `np.random.seed`, `torch.manual_seed`. |
| 7 | Single notebook = hard to diff / review / unit-test. | Medium | This is the user's open question for the next step (see Section 7). |
| 8 | No CLAUDE.md, no PLAN.md, no HANDOVER.md, no `requirements.txt`, no `.gitignore`. | Medium | Will need to create as we go. |
| 9 | The 1.1 GB JSON should never be committed to git. | High | Needs `.gitignore` BEFORE git init / first commit. |

## 6. Git status — important

**The Code directory is NOT under git yet.** Project instruction #2 says:
> "make sure to have a reference from git, when too much error code is generated and debugging becomes nightmare, proper fallback logic should be planned ahead and prompted to me before any changes, like if the previous version is not updated in the git we must do that first."

Action required before any code changes are made: initialise git, add a sane `.gitignore` (excluding the 1.1 GB JSON, `.pth` weights, `__pycache__`, `.ipynb_checkpoints`), and make an initial commit pinning the current working state of `project.ipynb`. This gives us a safe rollback point.

## 7. Open decisions (currently with the user)

1. **Split the codebase into logical Python modules vs. keep one big `.ipynb`?** User is on Colab/Kaggle free GPU and uploads a single file currently. Tradeoffs discussed in chat — awaiting decision.
2. **Which model is canonical?** Mistral-7B (current code) or Llama-3.2-1B (file names)?
3. **Use the existing `hallucination_10k.json`** or keep regenerating 2k samples each session?
4. **Initialise git now**, or wait until splitting decision is made?

## 8. Key external references (PDFs already in repo)

- MIND paper concepts → see notebook code (the architecture and notation follow it directly)
- `selfcheckgpt.pdf` — alternative hallucination detection baseline
- `NeurIPS-2024-haloscope-*.pdf` — unsupervised baseline (HaloScope)
- `LM know what they know.pdf`, `The internal state of LLM...` — motivation for using internal states
- `latest survey on hallucination...pdf`, `Survey of hallucination in natural language generation.pdf` — survey/background

## 9. Files Claude should read at the start of every new session

1. `STATUS.md` (this file) — current snapshot.
2. `HANDOVER.md` — what happened in the previous session and the user's last instruction. _(not yet created)_
3. `PLAN.md` — agreed plan being followed. _(not yet created)_
4. `Code/project.ipynb` — the actual code, only if changes are anticipated.
