# GPT-J-6B — variants vs. baselines (multi-task AUROC)

_16 variants (A–P) + 6 baselines, all on the same 10 datasets at QUICK_EVAL_N=350 first-N. AUROC ↑. Variant rows transcribed from the run console (the 238 KB output JSON exceeded the upload and was lost); baselines from the complete `kaggle_gptj_6b_baselines_results.json`._

**Bold** = best in that dataset column.

| Method | TruthQA | TrivQA | CoQA | TydiQA | HE-QA | HE-Sum | HE-Dial | NQ | HotpotQA | PopQA | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.503 | 0.476 | 0.580 | 0.502 | 0.124 | 0.398 | 0.330 | 0.562 | 0.569 | 0.541 | 0.459 |
| B | 0.492 | 0.459 | 0.566 | 0.492 | 0.127 | 0.337 | 0.339 | 0.551 | 0.558 | 0.553 | 0.447 |
| C | 0.478 | 0.453 | 0.564 | 0.497 | 0.109 | 0.381 | 0.349 | 0.535 | 0.554 | 0.534 | 0.445 |
| D | 0.519 | 0.410 | 0.541 | 0.478 | 0.260 | 0.389 | 0.389 | 0.513 | 0.524 | **0.692** | 0.472 |
| E | 0.526 | 0.503 | 0.570 | 0.487 | 0.272 | 0.371 | 0.330 | 0.403 | 0.557 | 0.620 | 0.464 |
| F | 0.498 | 0.483 | 0.591 | 0.505 | 0.068 | 0.342 | 0.321 | 0.544 | 0.550 | 0.548 | 0.445 |
| G | 0.472 | 0.482 | 0.573 | 0.502 | 0.079 | 0.383 | 0.344 | 0.535 | 0.546 | 0.545 | 0.446 |
| H | 0.522 | 0.478 | 0.525 | 0.536 | 0.434 | 0.374 | 0.396 | 0.558 | 0.517 | 0.417 | 0.476 |
| I | 0.504 | 0.511 | 0.563 | 0.490 | 0.126 | 0.348 | 0.329 | 0.455 | 0.563 | 0.569 | 0.446 |
| J | 0.494 | 0.540 | 0.454 | 0.501 | 0.844 | 0.597 | **0.668** | 0.513 | 0.446 | 0.426 | 0.548 |
| K | 0.484 | 0.518 | 0.546 | 0.520 | 0.635 | 0.391 | 0.464 | 0.539 | 0.603 | 0.553 | 0.525 |
| L | 0.555 | 0.534 | 0.444 | 0.465 | 0.408 | 0.462 | 0.522 | 0.484 | 0.453 | 0.355 | 0.468 |
| M | 0.494 | 0.507 | 0.576 | 0.507 | 0.065 | 0.315 | 0.311 | 0.572 | 0.567 | 0.596 | 0.451 |
| N | 0.481 | 0.497 | 0.421 | 0.493 | **0.944** | 0.564 | 0.665 | 0.516 | 0.427 | 0.451 | 0.546 |
| O | 0.504 | 0.481 | 0.440 | 0.461 | 0.351 | 0.527 | 0.406 | 0.604 | 0.449 | 0.483 | 0.471 |
| P | 0.465 | 0.471 | 0.475 | 0.475 | 0.620 | 0.580 | 0.585 | 0.522 | 0.466 | 0.424 | 0.508 |
| *— baselines —* |  |  |  |  |  |  |  |  |  |  |  |
| SAPLMA | 0.470 | 0.486 | 0.515 | 0.534 | **0.985** | **0.611** | 0.649 | 0.471 | 0.555 | 0.341 | **0.562** |
| HaloScope | 0.523 | 0.472 | 0.512 | 0.545 | 0.455 | 0.531 | 0.535 | **0.627** | 0.534 | 0.352 | 0.509 |
| HalluShift | 0.466 | 0.512 | 0.558 | 0.547 | 0.446 | 0.474 | 0.469 | 0.612 | 0.578 | 0.430 | 0.509 |
| EigenScore | 0.523 | **0.641** | 0.400 | 0.402 | 0.501 | 0.500 | 0.500 | 0.395 | **0.656** | 0.347 | 0.487 |
| MIND | 0.522 | 0.462 | 0.542 | 0.458 | 0.520 | 0.384 | 0.505 | 0.532 | 0.493 | 0.453 | 0.487 |
| Perplexity | **0.570** | 0.580 | **0.609** | **0.600** | 0.434 | 0.397 | 0.426 | 0.589 | 0.597 | 0.429 | 0.523 |

**Ranking by average AUROC (top 8):** SAPLMA 0.562 · J 0.548 · N 0.546 · K 0.525 · Perplexity 0.523 · HalluShift 0.509 · HaloScope 0.509 · P 0.508.

**Headline (GPT-J-6B):** the strongest method overall is the **SAPLMA baseline (0.562)** — unlike OPT-6.7B, where proposed variant N led. The best proposed variants here are **J (0.548, the full F1–F10 stack)** and the best *new-feature* variant **N (0.546, CTS+EAR+PEA)** — essentially tied just behind SAPLMA. Caveat: the **HaluEval-QA** column dominates the averages (SAPLMA 0.985, N 0.944, J 0.844, K 0.635, P 0.620), so the ranking largely reflects that one dataset; on the non-HaluEval datasets the methods are closer and mostly near 0.5. So on GPT-J the new features are **competitive but not ahead** of the strongest baseline — a model-dependent result worth reporting honestly alongside the OPT result.

_Variant feature stacks (per the run): A=canonical only; B/C/D=+D_mean/V_last/H_mean; E=+MIND⁺ trio; F/G/H=+F1/F5/F7; I=+F1,F5,F7; J=+all F1–F10; K=+MIND⁺&F1,F5,F7; L=+all MIND⁺/F1–F10; M=URP+LTC; N=CTS+EAR+PEA; O=URP+LTC+CTS; P=MIND⁺+all F11–F16._
