# Hallucination Detection in Large Language Models Using Internal Representations

Code and results accompanying the dissertation and the paper *"What Do Internal-State
Hallucination Detectors Actually Detect? A Confound-Aware Re-Analysis of
Entity-Substitution Features."*

## Overview
Detectors are trained on label-free, entity-substitution Wikipedia data and read a
model's internal state in a **single forward pass**: the canonical last-token /
last-layer hidden vector plus cheap scalar features (layer-wise drift, cross-layer
variance, predictive entropy, and a battery of attention / hidden-state-geometry /
output-distribution signals). Sixteen feature configurations are compared against six
baselines under one fixed classifier and one fixed evaluation protocol, in-distribution
and on up to ten transfer benchmarks.

## Repository contents
- **`Code/`** — the full pipeline, one folder per model (`project_<model>/`):
  - `01_data_generation_*.ipynb` — build the balanced, pseudo-labelled dataset.
  - `02_all_variants_*.ipynb` — extract features, train the 16 configurations, evaluate.
  - `03_baselines_sota_*.ipynb` — the six baselines in the same harness.
- **`results_gptj_6b/`, `results_opt_67b/`** — per-model result files.
- **`thesis_report/`** — LaTeX source of the dissertation.

## Models and benchmarks
- **Models:** GPT-J-6B, OPT-6.7B, Llama-2-7B, Mistral-7B, Qwen2.5-3B
  (with smaller reference models used for smoke tests).
- **Benchmarks:** TruthfulQA, TriviaQA, CoQA, TyDi QA, Natural Questions, HotpotQA,
  PopQA, and the three HaluEval subsets (QA / dialogue / summarisation).

## Reproducing
Run the notebooks in order (`01` -> `02` -> `03`) per model on a single 16 GB GPU
(e.g. a free Kaggle/Colab T4). Large intermediate artefacts (datasets, feature caches,
checkpoints) are git-ignored; the per-configuration result JSONs are tracked.
