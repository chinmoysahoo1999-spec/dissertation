# OPT-6.7B — variants vs. baselines (multi-task AUROC)

_Source: `kaggle_opt_67b_all_variants_results.json` (A–L), `kaggle_opt_67b_all_variants_NEW_results.json` (M–P), baselines from the Kaggle 03 run. AUROC ↑._

> **Caveat:** variants **A–L are from an earlier run on 7 datasets at small N (40–400/dataset)** — NQ-Open, HotpotQA and PopQA were not evaluated for them (shown as —). Variants **M–P and the baselines use the newer 10-dataset / ~500-per-dataset eval**, so A–L are **not directly comparable** to the rest. For a fully consistent 16-variant table, re-run the unified `02_all_variants_opt_67b.ipynb` (now 16-var, 350/dataset). Use the **Avg(7)** column for an apples-to-apples comparison across all rows.

Baselines here are **4 of 6** (SAPLMA, HaloScope, HalluShift, EigenScore); **MIND + Perplexity pending** (run `03b_extra_baselines_opt_67b.ipynb`).

## Table 1 — clean comparison: 4 new variants (M–P) vs. 4 baselines, all 10 datasets

| Method | TruthQA | TrivQA | CoQA | TydiQA | HE-QA | HE-Sum | HE-Dial | NQ | HotpotQA | PopQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M | 0.449 | 0.585 | 0.550 | 0.434 | 0.448 | 0.415 | 0.437 | 0.521 | 0.446 | **0.530** | 0.482 |
| N | 0.587 | 0.525 | 0.537 | 0.493 | **0.878** | 0.503 | **0.554** | 0.518 | 0.439 | 0.477 | 0.551 |
| O | 0.515 | 0.567 | 0.523 | 0.528 | 0.764 | 0.383 | 0.482 | 0.468 | 0.414 | 0.488 | 0.513 |
| P | 0.388 | 0.572 | 0.536 | 0.479 | 0.521 | 0.439 | 0.430 | 0.407 | 0.428 | 0.518 | 0.472 |
| SAPLMA | 0.488 | 0.478 | 0.489 | 0.502 | 0.787 | 0.534 | 0.382 | 0.553 | 0.507 | 0.418 | 0.514 |
| HaloScope | **0.665** | 0.500 | 0.479 | 0.481 | 0.272 | 0.136 | 0.377 | 0.509 | **0.541** | 0.210 | 0.417 |
| HalluShift | 0.646 | 0.522 | 0.524 | 0.522 | 0.465 | **0.547** | 0.464 | 0.509 | 0.515 | 0.143 | 0.486 |
| EigenScore | 0.504 | 0.519 | 0.520 | 0.407 | 0.500 | 0.501 | 0.499 | **0.588** | 0.444 | 0.071 | 0.455 |

_In the comparable set, variant **N (avg 0.551)** is the strongest method overall — ahead of SAPLMA (0.514), HalluShift (0.486), EigenScore (0.455) and HaloScope (0.417) — and wins HE-QA and HE-Dial outright. M wins PopQA; baselines win TruthQA/HotpotQA (HaloScope), HE-Sum (HalluShift), NQ (EigenScore)._

## Table 2 — full 16 variants (A–P) + 4 baselines  (A–L: 7-dataset older run)

| Method | TruthQA | TrivQA | CoQA | TydiQA | HE-QA | HE-Sum | HE-Dial | NQ | HotpotQA | PopQA | Avg(7) | Avg(10) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.328 | 0.535 | **0.571** | 0.503 | 0.054 | 0.423 | 0.345 | — | — | — | 0.394 | 0.394 |
| B | 0.338 | 0.567 | 0.564 | 0.523 | 0.039 | 0.316 | 0.313 | — | — | — | 0.380 | 0.380 |
| C | 0.308 | 0.561 | 0.550 | 0.537 | 0.025 | 0.321 | 0.310 | — | — | — | 0.373 | 0.373 |
| D | 0.303 | 0.539 | 0.556 | 0.542 | 0.126 | 0.342 | 0.336 | — | — | — | 0.392 | 0.392 |
| E | 0.276 | 0.485 | 0.502 | 0.492 | 0.228 | 0.380 | 0.342 | — | — | — | 0.386 | 0.386 |
| F | 0.286 | 0.546 | 0.558 | 0.506 | 0.032 | 0.361 | 0.316 | — | — | — | 0.372 | 0.372 |
| G | 0.313 | 0.527 | 0.542 | 0.514 | 0.026 | 0.336 | 0.309 | — | — | — | 0.367 | 0.367 |
| H | 0.308 | 0.528 | 0.559 | 0.516 | 0.037 | 0.335 | 0.313 | — | — | — | 0.371 | 0.371 |
| I | 0.303 | 0.519 | 0.551 | 0.504 | 0.055 | 0.334 | 0.349 | — | — | — | 0.374 | 0.374 |
| J | 0.308 | **0.626** | 0.562 | **0.567** | 0.246 | 0.395 | 0.353 | — | — | — | 0.437 | 0.437 |
| K | 0.328 | 0.553 | 0.551 | 0.538 | 0.049 | 0.286 | 0.302 | — | — | — | 0.373 | 0.373 |
| L | 0.293 | 0.528 | 0.554 | 0.498 | 0.020 | 0.352 | 0.318 | — | — | — | 0.366 | 0.366 |
| M | 0.449 | 0.585 | 0.550 | 0.434 | 0.448 | 0.415 | 0.437 | 0.521 | 0.446 | **0.530** | 0.474 | 0.482 |
| N | 0.587 | 0.525 | 0.537 | 0.493 | **0.878** | 0.503 | **0.554** | 0.518 | 0.439 | 0.477 | 0.582 | 0.551 |
| O | 0.515 | 0.567 | 0.523 | 0.528 | 0.764 | 0.383 | 0.482 | 0.468 | 0.414 | 0.488 | 0.537 | 0.513 |
| P | 0.388 | 0.572 | 0.536 | 0.479 | 0.521 | 0.439 | 0.430 | 0.407 | 0.428 | 0.518 | 0.481 | 0.472 |
| SAPLMA | 0.488 | 0.478 | 0.489 | 0.502 | 0.787 | 0.534 | 0.382 | 0.553 | 0.507 | 0.418 | 0.523 | 0.514 |
| HaloScope | **0.665** | 0.500 | 0.479 | 0.481 | 0.272 | 0.136 | 0.377 | 0.509 | **0.541** | 0.210 | 0.416 | 0.417 |
| HalluShift | 0.646 | 0.522 | 0.524 | 0.522 | 0.465 | **0.547** | 0.464 | 0.509 | 0.515 | 0.143 | 0.527 | 0.486 |
| EigenScore | 0.504 | 0.519 | 0.520 | 0.407 | 0.500 | 0.501 | 0.499 | **0.588** | 0.444 | 0.071 | 0.493 | 0.455 |

**Bold** = best in that dataset column. Avg(7) = mean over the 7 datasets all rows share (use this to compare A–L against everything else); Avg(10) = mean over all 10 (— for A–L).

_Note: on the comparable Avg(10), variant **N (0.551)** leads. On Avg(7) (all 16 + 4 comparable), N is again top (0.582), with SAPLMA (0.523) and HalluShift (0.527) the strongest baselines; the old A–L variants cluster at 0.37–0.44 — but remember they ran at much smaller N, so treat as indicative only. Re-run the unified `02` for a clean 16-variant comparison._
