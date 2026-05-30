# GPT-2 smoke run — AUROC comparison (Baselines vs Variants)

Each row is one detection method. Each column is a dataset (Wikipedia held-out + 7 downstream).
Cells are AUROC. **Bold = column winner.** `n.a.` = column has no finite AUROC for that row.

Backbone: GPT-2 (124M). Training set: 800 records (500/class, MIND-style Wikipedia continuations). 
Test split: seed=42, identical across baselines and variants → AUROCs are directly comparable.

Methodology notes:
- SAPLMA: MLP probe on layer-7 last-token hidden state. Supervised.
- HaloScope: spectral score → percentile pseudo-labels → non-linear probe. Unsupervised pseudo-labels.
- HalluShift: 31-d Wasserstein+cosine+token-prob features → CombinedNN. Supervised.
- EigenScore: K=10 sampled responses → K×K cov log-det. Unsupervised. (downstream subsampled to n=100 per dataset, K=3.)
- Variants A–L: 5-layer MLP on canonical embedding + scalar-feature subsets. Supervised.

| Method | **Wiki** | TruthQA | TrivQA | CoQA | TydiQA | HE-QA | HE-Sum | HE-Dial | **Avg-MT** |
|---|---|---|---|---|---|---|---|---|---|
| **[B] SAPLMA** | 0.321 | 0.531 | 0.424 | 0.536 | 0.518 | 0.498 | 0.475 | **0.576** | 0.508 |
| **[B] HaloScope** | 0.437 | 0.495 | 0.545 | 0.515 | 0.419 | 0.089 | 0.422 | 0.459 | 0.421 |
| **[B] HalluShift** | 0.443 | 0.535 | 0.404 | **0.583** | **0.619** | 0.451 | **0.561** | 0.515 | **0.524** |
| **[B] EigenScore** | 0.457 | 0.505 | 0.629 | 0.426 | 0.455 | **0.499** | 0.500 | 0.500 | 0.502 |
| [V] A (canonical only) | 0.501 | 0.515 | 0.627 | 0.557 | 0.563 | 0.140 | 0.476 | 0.364 | 0.463 |
| [V] B (+D_mean) | 0.405 | 0.544 | 0.676 | 0.413 | 0.441 | 0.073 | 0.459 | 0.368 | 0.425 |
| [V] C (+V_last) | 0.394 | 0.524 | **0.711** | 0.484 | 0.470 | 0.270 | 0.542 | 0.429 | 0.490 |
| [V] D (+H_mean) | 0.405 | **0.547** | 0.679 | 0.481 | 0.507 | 0.187 | 0.525 | 0.400 | 0.475 |
| [V] E (+D_mean+V_last+H_mean (MIND+)) | 0.467 | 0.533 | 0.638 | 0.533 | 0.565 | 0.133 | 0.487 | 0.372 | 0.466 |
| [V] F (+F1 (Lookback)) | 0.322 | 0.494 | 0.637 | 0.463 | 0.391 | 0.273 | 0.539 | 0.532 | 0.476 |
| [V] G (+F5 (Logit-Lens JSD)) | 0.393 | 0.500 | 0.504 | 0.437 | 0.423 | 0.063 | 0.485 | 0.395 | 0.401 |
| [V] H (+F7 (Max-Margin)) | 0.467 | 0.519 | 0.705 | 0.493 | 0.469 | 0.052 | 0.475 | 0.379 | 0.442 |
| [V] I (+F1+F5+F7 (trio)) | 0.502 | 0.537 | 0.605 | 0.533 | 0.576 | 0.084 | 0.467 | 0.358 | 0.451 |
| [V] J (+all 9 F-feats) | 0.475 | 0.537 | 0.620 | 0.538 | 0.567 | 0.089 | 0.492 | 0.364 | 0.458 |
| [V] K (+MIND+ +F1+F5+F7) | 0.498 | 0.516 | 0.624 | 0.550 | 0.550 | 0.071 | 0.469 | 0.355 | 0.448 |
| [V] L (+everything) | **0.510** | 0.540 | 0.569 | 0.535 | 0.591 | 0.041 | 0.437 | 0.341 | 0.436 |

## Best per column

| Column | Winning method | AUROC |
|---|---|---|
| Wikipedia | [V] L (+everything) | 0.5097 |
| TruthQA | [V] D (+H_mean) | 0.5468 |
| TrivQA | [V] C (+V_last) | 0.7113 |
| CoQA | **[B] HalluShift** | 0.5834 |
| TydiQA | **[B] HalluShift** | 0.6195 |
| HE-QA | **[B] EigenScore** | 0.4994 |
| HE-Sum | **[B] HalluShift** | 0.5605 |
| HE-Dial | **[B] SAPLMA** | 0.5757 |
| Avg-MT | **[B] HalluShift** | 0.5239 |