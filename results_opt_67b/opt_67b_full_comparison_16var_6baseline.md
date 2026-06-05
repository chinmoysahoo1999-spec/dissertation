# OPT-6.7B — full comparison: 16 feature configurations (A–P) + 6 baselines

_Multi-task hallucination-detection AUROC (higher is better; 0.5 = chance), QUICK_EVAL_N = 350 deterministic first-N rows/dataset._

**Sources.** A–L: `kaggle_opt_67b_all_variants_results.json` (older 7-dataset run, small N — NQ/HotpotQA/PopQA not evaluated, shown as `--`). M–P and SAPLMA/HaloScope/HalluShift/EigenScore: the project's `opt_67b_comparison.md` (10-dataset run). **MIND and Perplexity: `kaggle_opt_67b_extra_baselines_results.json` (new 10-dataset run, verified).**

> **Caveat:** A–L come from an earlier, smaller run on 7 datasets and are **not directly comparable** to M–P and the baselines (10 datasets). Use **Avg(7)** for an apples-to-apples comparison across every row; **Avg(10)** is only defined for the 10-dataset rows. **Bold** = best in column.

| Cfg / Baseline | TruthQA | TrivQA | CoQA | TydiQA | HE-QA | HE-Sm | HE-Dl | NQ | HotpotQA | PopQA | Avg(7) | Avg(10) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A  canonical only | 0.328 | 0.535 | 0.571 | 0.503 | 0.054 | 0.423 | 0.345 | -- | -- | -- | 0.394 | -- |
| B  +drift | 0.338 | 0.567 | 0.564 | 0.523 | 0.039 | 0.316 | 0.313 | -- | -- | -- | 0.380 | -- |
| C  +variance | 0.308 | 0.561 | 0.550 | 0.537 | 0.025 | 0.321 | 0.310 | -- | -- | -- | 0.373 | -- |
| D  +entropy | 0.303 | 0.539 | 0.556 | 0.542 | 0.126 | 0.342 | 0.336 | -- | -- | -- | 0.392 | -- |
| E  +internal-state trio | 0.276 | 0.485 | 0.502 | 0.492 | 0.228 | 0.380 | 0.342 | -- | -- | -- | 0.386 | -- |
| F  +lookback | 0.286 | 0.546 | 0.558 | 0.506 | 0.032 | 0.361 | 0.316 | -- | -- | -- | 0.372 | -- |
| G  +logit-lens | 0.313 | 0.527 | 0.542 | 0.514 | 0.026 | 0.336 | 0.309 | -- | -- | -- | 0.367 | -- |
| H  +max-margin | 0.308 | 0.528 | 0.559 | 0.516 | 0.037 | 0.335 | 0.313 | -- | -- | -- | 0.371 | -- |
| I  +lookback/lens/margin | 0.303 | 0.519 | 0.551 | 0.504 | 0.055 | 0.334 | 0.349 | -- | -- | -- | 0.374 | -- |
| J  +all 9 literature | 0.308 | 0.626 | 0.562 | **0.567** | 0.246 | 0.395 | 0.353 | -- | -- | -- | 0.437 | -- |
| K  +trio+lit-trio | 0.328 | 0.553 | 0.551 | 0.538 | 0.049 | 0.286 | 0.302 | -- | -- | -- | 0.373 | -- |
| L  +trio+all literature | 0.293 | 0.528 | 0.554 | 0.498 | 0.020 | 0.352 | 0.318 | -- | -- | -- | 0.366 | -- |
| M  +URP+curvature | 0.449 | 0.585 | 0.550 | 0.434 | 0.448 | 0.415 | 0.437 | 0.521 | 0.446 | 0.530 | 0.474 | 0.482 |
| N  +conf-traj/rank/echo | 0.587 | 0.525 | 0.537 | 0.493 | **0.878** | 0.503 | **0.554** | 0.518 | 0.439 | 0.477 | **0.582** | **0.551** |
| O  +URP/curv/conf | 0.515 | 0.567 | 0.523 | 0.528 | 0.764 | 0.383 | 0.482 | 0.468 | 0.414 | 0.488 | 0.537 | 0.513 |
| P  +trio+all new | 0.388 | 0.572 | 0.536 | 0.479 | 0.521 | 0.439 | 0.430 | 0.407 | 0.428 | 0.518 | 0.481 | 0.472 |
| _— baselines —_ | | | | | | | | | | | | |
| SAPLMA | 0.488 | 0.478 | 0.489 | 0.502 | 0.787 | 0.534 | 0.382 | 0.553 | 0.507 | 0.418 | 0.523 | 0.514 |
| HaloScope | 0.665 | 0.500 | 0.479 | 0.481 | 0.272 | 0.136 | 0.377 | 0.509 | 0.541 | 0.210 | 0.416 | 0.417 |
| HalluShift | 0.646 | 0.522 | 0.524 | 0.522 | 0.465 | **0.547** | 0.464 | 0.509 | 0.515 | 0.143 | 0.527 | 0.486 |
| EigenScore | 0.504 | 0.519 | 0.520 | 0.407 | 0.500 | 0.501 | 0.499 | **0.588** | 0.444 | 0.071 | 0.493 | 0.455 |
| MIND | 0.536 | 0.456 | 0.454 | 0.500 | 0.553 | 0.356 | 0.429 | 0.539 | 0.545 | **0.687** | 0.469 | 0.506 |
| Perplexity | **0.754** | **0.638** | **0.590** | 0.550 | 0.431 | 0.397 | 0.424 | 0.500 | **0.548** | 0.328 | 0.540 | 0.516 |

**Ranking by Avg(10) (10-dataset rows only, top 8):** N 0.551 · Perplexity 0.516 · SAPLMA 0.514 · O 0.513 · MIND 0.506 · HalluShift 0.486 · M 0.482 · P 0.472

**Headline (OPT-6.7B, Avg-10).** Strongest overall: **N (0.551)**. Among the six baselines, Perplexity and MIND are the strongest of the newly added pair; the proposed new-feature configuration **N (confidence-trajectory/attention-rank/prompt-echo)** leads the feature configurations. All transfer AUROCs sit in the ~0.4–0.55 band and one dataset (HaluEval-QA) heavily influences the means — report per-dataset, not just the average.

**Configuration legend.** trio = the three internal-state scalars (drift `D_mean`, cross-layer variance `V_last`, predictive entropy `H_mean`); literature = F1–F10 (F9 deferred); new = F11–F16 (URP, layer-trajectory curvature, confidence-trajectory, attention-rank, prompt-echo, head-importance).

**Baselines.** SAPLMA (mid-layer supervised probe), HaloScope (unsupervised spectral), HalluShift (31-d supervised multi-feature), EigenScore/INSIDE (sampling-based score), MIND (final-layer supervised probe — the method this work extends), Perplexity (mean token NLL, unsupervised).
