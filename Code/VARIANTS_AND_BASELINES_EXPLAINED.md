# Variants A–P and the 6 Baselines — what each is actually made of

_Reference for `02_all_variants_*.ipynb` (16 variants) and `03_baselines_sota_*.ipynb` (6 baselines)._
_Verified against the `VARIANTS` dict and the saved result JSONs. Input dims shown for **Falcon-7B (hidden = 4544)** — they match your run's `input_dim` column exactly, which proves all 16 trained._

## 1. The feature menu (building blocks)

**Canonical feature** — present in EVERY variant. The last-token, last-layer hidden-state vector. Its length = the model's hidden size (Falcon-7B 4544, GPT-J 4096, OPT 4096, Qwen-3B 2560…). This is the MIND backbone.

**Internal-state scalars (this work's core — 3):**
- `D_mean` — layer-wise drift: mean cosine distance between adjacent layers.
- `V_last` — cross-layer variance: L2 spread of the layer vectors around their mean.
- `H_mean` — predictive entropy: mean Shannon entropy of the per-step output distribution.

**Literature scalars (9; from prior papers):**
- `F1` lookback_ratio — attention on context vs newly generated tokens (Lookback Lens).
- `F2` attention_sink — attention mass collapsing onto the first/"sink" token.
- `F3` eigenscore_lite — eigenvalue spread of hidden states (INSIDE-style).
- `F4` icr_score — internal-consistency ratio.
- `F5` logit_lens_jsd — JS divergence between early- and late-layer vocabulary projections (DoLa / logit lens).
- `F6` head_entropy — entropy of attention-head distributions.
- `F7` max_margin — top-1 minus top-2 probability margin per token (HaMI-style).
- `F8` token_rank — rank of the chosen token in the output distribution.
- `F10` intra_dispersion — within-layer dispersion of token embeddings (D2HScore).
- (`F9`, a SAPLMA-style mid-layer probe, is **deferred** — not used.)

**New / exploratory scalars (the F11–F16 family):**
- `F11` URP — unembedding-reasoning projection, an **8-dim** vector (`F11_urp_0…7`).
- `F12` LTC — layer-trajectory curvature (`F12_ltc_mean`, `F12_ltc_max` → 2 values).
- `F13` CTS — confidence-trajectory (`F13_cts_slope`, `F13_cts_variance` → 2 values).
- `F14` EAR — effective attention rank (1 value).
- `F15` PEA — prompt-echo alignment (1 value).
- `F16` HID — head-importance divergence (1 value).

> Input dimension of any variant = **hidden_size + (number of scalars)**.

## 2. The 16 variants (A–P)

| ID | Plain name | Scalars added on top of the canonical vector | #scalars | Falcon dim |
|----|------------|-----------------------------------------------|:-------:|:----------:|
| A | Canonical only | (none) | 0 | 4544 |
| B | + drift | `D_mean` | 1 | 4545 |
| C | + variance | `V_last` | 1 | 4545 |
| D | + entropy | `H_mean` | 1 | 4545 |
| **E** | **+ internal-state trio (core)** | `D_mean, V_last, H_mean` | 3 | 4547 |
| F | + lookback | `F1` | 1 | 4545 |
| G | + logit-lens | `F5` | 1 | 4545 |
| H | + max-margin | `F7` | 1 | 4545 |
| I | + lookback/lens/margin | `F1, F5, F7` | 3 | 4547 |
| J | + all 9 literature | `F1,F2,F3,F4,F5,F6,F7,F8,F10` | 9 | 4553 |
| K | + trio + lit-trio | `D_mean,V_last,H_mean, F1,F5,F7` | 6 | 4550 |
| L | + trio + all literature | trio + `F1…F10` (= all 12 classic scalars) | 12 | 4556 |
| M | + URP + curvature | `F11`(8) + `F12_ltc_mean, F12_ltc_max` | 10 | 4554 |
| N | + confidence/rank/echo | `F13_cts_slope, F13_cts_variance, F14, F15` | 4 | 4548 |
| O | + URP/curvature/confidence | `F11`(8) + `F12`(2) + `F13`(2) | 12 | 4556 |
| P | + trio + all new | trio + `F11`(8) + `F12`(2) + `F13`(2) + `F14 + F15 + F16` | 18 | 4562 |

Notes: **A** is the pure MIND backbone (the floor). **E** is the headline proposal (canonical + the 3 internal-state scalars). **J** is the full classic-literature stack. **L** = everything classic (trio + F1–F10). **M–P** explore the newer F11–F16 features; **P** is the most ambitious (trio + every new feature) but deliberately leaves out F1–F10.

## 3. The 6 baselines (`03_baselines_sota`)

| Baseline | Type | What it uses | Paper |
|----------|------|--------------|-------|
| SAPLMA | supervised probe | MLP on **one mid layer's** last-token hidden (e.g. layer 17 of 28) | Azaria & Mitchell 2023 |
| HaloScope | unsupervised spectral | SVD of activations → spectral score → percentile pseudo-labels → probe on the best layer | Du et al. 2024 |
| HalluShift | supervised, 31-d | 5 Wasserstein + 5 cosine distances between adjacent-layer hidden distributions, the same on attention, plus token-probability features; small 2-layer net | Dasgupta 2025 |
| EigenScore (INSIDE) | unsupervised score | K sampled generations → K×K covariance of a mid-layer last-token hidden → mean(log eigenvalues); rank-normalised | Chen et al. 2024 |
| MIND | supervised probe | MLP on the **last layer's** last-token hidden (the method this work extends) | Su et al. 2024 |
| Perplexity | unsupervised score | mean token negative log-likelihood of the text; rank-normalised | classic |

Useful contrasts: variant **A** (canonical) and the **MIND** baseline are both last-layer probes (so they should score similarly); **SAPLMA** is the same idea but on a *middle* layer; **EigenScore** and **Perplexity** are threshold-free scores (no training); **HalluShift** is the related senior work, included only as a baseline.

## 4. The "12 vs 16" fix (2026-06)

The training loop always iterated the whole `VARIANTS` dict (16 entries), so **16 variants were always trained** — only the printed/comment labels were stale ("Training 12 MLP variants…", "12 VARIANTS x 7 DATASETS"). Those are now corrected to `len(VARIANTS)` / 16 and 10 datasets across every notebook. In the baselines notebooks the multi-task stage already ran all 6, but the **Wikipedia held-out stage scored only 4** — MIND and Perplexity have now been added there too, so all 6 are evaluated in both regimes. Backups of the originals are saved as `*.ipynb.bak_count_2026-06`.
