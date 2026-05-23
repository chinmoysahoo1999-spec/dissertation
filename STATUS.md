# STATUS.md — Project Snapshot

_Last updated: 2026-05-23 (post mid-eval review session)._
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
    ├── requirements.txt                   <-- pinned deps
    ├── pytest.ini
    ├── hallucination_10k.json             <-- 1.1 GB pre-existing dataset (gitignored)
    ├── preview.json                       <-- whitelisted preview slice
    ├── survey_chapter.md                  <-- Markdown source of the survey PDF
    ├── survey_notes.md                    <-- raw per-paper extraction notes
    ├── _patch_notebook.py                 <-- one-off; rewrote project.ipynb to (incorrectly) Qwen-0.5B
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

This Cowork session (2026-05-23) did **not** change the code. It produced:

- The 11-page literature survey chapter (committed; 4 commits ready to push on `main`).
- This rewritten STATUS.md.
- The session-wise PLAN.md.

Code changes (Issue #1–#4 above) are deferred to **Session 1 of PLAN.md**, which the student will run in the next Cowork session.

---

## 9. Results log

Each time `project.ipynb` finishes running on Kaggle, it now writes two summary files (Block 11 of the notebook):

- `results_summary.json` — machine-readable; paste back to the assistant for verification.
- `results_summary.md` — human-readable; can be downloaded from Kaggle's "Output" tab.

Both files appear in `/kaggle/working/`. The notebook also prints a copy-paste-ready JSON block at the very end of the run so you can copy directly from the notebook output.

### Run history

| # | Date (UTC) | Host | Model | Samples | AUC-ROC | F1 | Acc | Notes |
|---|---|---|---|---|---|---|---|---|
| — | _pending_ | — | — | — | — | — | — | first Kaggle run to be appended here |

After each run, copy the row from `results_summary.md` and paste it under "Run history". Keep the most recent run at th "Run history". Keep the most recent run at the bottom.

### Latest results JSON (paste-area)

```
[paste the most recent results_summary.json contents here]
```

---

## 10. Reference papers in repo

(unchanged from prior STATUS; the relevant additions for this dissertation's specific contribution are the mid-eval PDF, MIND paper, HalluShift thesis, and SelfCheckGPT)

| File | Role |
|---|---|
| `mid_evl (3) (4).pdf` | The student's own mid-evaluation slides (Feb 2026). Single most important file for understanding what's done and what remains. |
| `hallucination_detection using unsupervised method.pdf` | MIND (Su et al., Findings of ACL 2024). The methodology backbone. |
| `M.tech _Sharanya_Dasgupta_CS2320.pdf` | HalluShift (Dasgupta, ISI 2025). Adjacent prior work; originality-differentiation built against. |
| `NeurIPS-2024-haloscope-*.pdf` | HaloScope (Du et al., NeurIPS 2024). Sister method, same 4-QA suite. |
| `selfcheckgpt.pdf` | SelfCheckGPT (Manakul et al., EMNLP 2023). Sampling-based baseline. |
| `The internal state of LLM know when it laying(2023).pdf` | SAPLMA (Azaria & Mitchell, 2023). MIND's direct precursor. |
| `LM know what they know.pdf` | Kadavath et al., 2022. Calibration / self-knowledge. |
| `Survey of hallucination in natural language generation.pdf` | Ji et al. survey (3 copies in repo — same paper). |
| `2024.fever-1.5.pdf`, `2024.fever-1.5-summary.pdf` | Student's own prior FEVER work. |
| `Notes On Lexical (n-gram) Metrics For Hallucination Detection.pdf` | Pedagogical reference. |
