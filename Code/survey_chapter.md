# Chapter 2 — Related Work: Hallucination Detection in Large Language Models

*A thesis-style literature survey for the dissertation "MIND-style Internal-State Hallucination Detection in a 0.5B-parameter Open-Weights LLM, Extended to Multi-Task Question Answering, Summarisation, and Dialogue."*

---

## 2.1 Introduction and definitions

Large language models (LLMs) routinely produce text that is fluent, syntactically well-formed, and rhetorically convincing, yet factually wrong. The phenomenon is widely termed *hallucination* and has become the dominant non-safety failure mode of generative systems deployed at scale.

The most cited taxonomy is due to Ji *et al.* (2023; updated 2024), who distinguish two facets of the problem. **Intrinsic hallucination** is output that contradicts the source material the model was conditioned on — for example, a summariser claiming the article said the opposite of what it did. **Extrinsic hallucination** is output that cannot be verified from the source — additional fabricated material that may or may not be consistent with world knowledge. A second, orthogonal axis separates **faithfulness** (consistency with the prompt or document) from **factuality** (consistency with the real world). These distinctions matter because evaluation benchmarks, and therefore the appropriate detection methods, target different combinations: SelfCheckGPT's WikiBio benchmark targets extrinsic factuality, the FEVER family targets evidence-grounded faithfulness, and HaluEval splits cleanly into faithfulness (Summarisation, Dialogue) and factuality (open-domain QA).

A second definitional issue is granularity. Hallucination detection has been formulated as *passage-level* (one label per generation), *sentence-level* (Manakul *et al.* 2023 — SelfCheckGPT, Su *et al.* 2024 — MIND), *atomic-fact-level* (Min *et al.* 2023 — FActScore), and *token-level* (the MIND classifier when applied step-by-step at inference). Finer granularity is more useful in practice but harder to label. The pseudo-labelling procedure used by MIND, which we describe in detail in §2.5, is one of the few that yields per-token supervision automatically.

Throughout this chapter we follow Ji *et al.*'s vocabulary, and we adopt the operational definition used by Su *et al.* (2024): a hallucination is a *coherent but factually incorrect* segment of LLM-generated text, where "incorrect" is judged against an external reference (Wikipedia, gold question/answer pairs, the source document, or a downstream knowledge base).

## 2.2 Why lexical metrics are insufficient

The earliest attempts to score generation quality, including ROUGE, BLEU, METEOR, and CIDEr, measure n-gram overlap between the generation and a reference. They are computationally trivial and remain ubiquitous as headline metrics in summarisation and translation benchmarks. The internal n-gram-metrics notes included in this work's bibliography give a useful pedagogical example: the sentence *Paris is the capital of Germany* has very high n-gram overlap with *Paris is the capital of France* and would receive an almost-perfect ROUGE score, despite being factually wrong. PARENT (Dhingra *et al.* 2019) partially mitigates this by also comparing the output against the source table, but the underlying weakness — surface form is a poor proxy for meaning, let alone truth — persists. Subsequent literature has converged on three families of *semantic* detectors that do not rely on n-gram overlap, and these are the subject of the remainder of this chapter.

## 2.3 Family I — Self-evaluation and calibration

The earliest attempt to make an LLM aware of its own uncertainty is the calibration line begun by Guo *et al.* (2017) for classification models and brought to LLMs by Kadavath *et al.* (2022) at Anthropic. Their paper "Language Models (Mostly) Know What They Know" introduces two complementary probes:

- **P(True)** — after the model has produced an answer, it is *prompted* to evaluate its own answer's correctness, and the probability assigned to the *True* token in the next-token distribution is read off as the calibration signal. Kadavath *et al.* show that P(True) becomes more discriminative when the model can see multiple of its own sampled answers, suggesting that comparison between candidate answers carries useful self-knowledge.
- **P(IK)** — a separately *trained* value head sits on top of the LLM and predicts, from the question alone (before any answer is produced), the probability that the model will sample a correct answer. The head is trained from many sampled rollouts on labelled training data and generalises only partially across tasks: a head trained on TriviaQA is competent on Lambada and arithmetic but breaks under stronger distribution shift.

Kadavath *et al.* show, across BIG-Bench, MMLU, TruthfulQA, LogiQA, QuALITY, TriviaQA, Lambada, Codex HumanEval, GSM8k, and arithmetic, that large Transformer LMs are well-calibrated on multiple-choice formats and somewhat calibrated on open-ended generation. They also show that P(True) and P(IK) move in the expected direction when relevant source material is provided in the prompt and when correct hints are added, supporting the interpretation that the calibration signal genuinely reflects internal evidence rather than surface fluency.

For our purposes, three observations matter. First, P(True)-style self-evaluation is the conceptual root of the *Prompt* variant of SelfCheckGPT (§2.4), which uses a separate LLM as judge rather than the same LLM. Second, P(IK) is itself a probing classifier — a small head on top of the LLM — and is therefore a direct ancestor of the SAPLMA/MIND/HaloScope line (§2.5). Third, P(IK) is reported in AUC and Brier-score terms rather than as a binary hallucination classifier, reminding us that any detection method should be evaluated as a *calibrated probabilistic estimator* and not only as a binary classifier.

## 2.4 Family II — Sampling-based and consistency methods

A second strategy abandons internals entirely and treats the LLM as a black box. If a generation R is sampled deterministically (e.g. greedy or temperature 0), and N alternative samples {S¹, …, Sᴺ} are drawn at a higher temperature, the claim is that *factual* content remains consistent across samples whereas *hallucinated* content varies. Disagreement between R and the sample set is then an inconsistency signal.

The canonical paper is **SelfCheckGPT** (Manakul *et al.* 2023, EMNLP). The authors define five interchangeable scoring functions, each implementing the consistency check at a different level of abstraction:

1. **BERTScore** — for each sentence in R, take the maximum BERTScore similarity against the N samples; aggregate to a sentence score by taking 1 − max.
2. **MQAG (question-answering)** — generate multiple-choice questions from R, answer them against R and against each sample, and measure agreement.
3. **n-gram** — fit a small n-gram LM on the samples and score the negative log-probability of R under that LM; either average or take the max per sentence.
4. **NLI** — pass R and each sample through a DeBERTa-v3-large fine-tuned on MNLI; the contradiction score is the inconsistency signal.
5. **Prompt (judge)** — for each sentence of R, ask another LLM (or the same LLM) the question *"Is the sentence supported by the context? Yes/No"* with the samples as context.

On the WikiBio GPT-3 benchmark (238 GPT-3-generated bios of obscure individuals, 1908 sentences hand-annotated as Major Inaccurate / Minor Inaccurate / Accurate), SelfCheck-Prompt achieves sentence-level Non-Factual AUC-PR of 93.4%, dominating the other variants (NLI 92.5%, Unigram-max 85.6%, QA 84.3%, BERTScore 82.0%), and passage-level Pearson correlation with human judgement of 78.3%.

SelfCheckGPT is the natural baseline for any new hallucination detector for two reasons. It is the highest-cited published method (over five thousand citations as of mid-2025) and it requires no labels, no internals, and no retrieval — only the ability to sample multiple completions. Its weakness is computational cost: with N = 20, end-to-end detection latency is roughly 18× the generation latency itself. The MIND paper measures this and reports SelfCheckGPT-NLI at 15.3 seconds per response versus MIND's 0.05 seconds — a 306× difference. For real-time applications, SelfCheckGPT is impractical.

Related sampling-based methods, also evaluated by Du *et al.* (2024) in the HaloScope baselines, include **Semantic Entropy** (Kuhn *et al.* 2023), which clusters semantically equivalent samples and computes entropy over clusters; **EigenScore** (Chen *et al.* 2024), which uses the spectral properties of the sample covariance; and **LN-Entropy** (Length-Normalised Entropy), a simple baseline that normalises raw entropy by sequence length to account for the bias toward longer answers.

A 2025 development worth highlighting is **LapEigvals** (Binkowski *et al.* 2025, "Hallucination Detection in LLMs Using Spectral Features of Attention Maps"). Rather than sample multiple generations, this approach models the *attention map* of a single generation as a graph Laplacian and extracts its spectral features. The intuition is that factual retrieval produces stable eigen-structures whereas hallucination produces diffuse, chaotic patterns. LapEigvals reports AUROC 88.9% on TriviaQA, competitive with the best internal-state methods, while remaining single-pass.

## 2.5 Family III — Internal-state probing (the line this dissertation extends)

The third family asks: if hidden states encode meaning, do they also encode *truth*? Can we read off a hallucination signal directly from the activations as the model produces tokens, without sampling, without retrieval, and without prompting the model to reflect on itself?

The trajectory of this idea passes through four major papers and culminates, for our purposes, in the MIND framework (Su *et al.* 2024), which is the methodological backbone of this dissertation.

### 2.5.1 SAPLMA (Azaria & Mitchell, 2023) — internal states "know when the LLM is lying"

Azaria and Mitchell (Findings of EMNLP 2023) introduced **SAPLMA** — Statement Accuracy Prediction based on Language Model Activations. The setup is supervised. The authors hand-crafted a 6,084-sentence True/False corpus across six disjoint topics (Cities, Inventions, Chemical Elements, Animals, Companies, Scientific Facts). False sentences are produced by swapping a factual attribute — *"The atomic number of Hydrogen is 34"*, *"Beijing is the capital of Brazil"* — a procedure that is the structural ancestor of MIND's entity-substitution pseudo-labelling.

The classifier is a three-hidden-layer feedforward network with sigmoid output that consumes the hidden activations of an LLM as it *reads* the statement. Training follows a leave-one-topic-out protocol: train on five topics, test on the held-out sixth, so that the probe must learn a topic-agnostic *truthfulness* feature rather than topic-correlated patterns. SAPLMA is evaluated on OPT-6.7B (Zhang *et al.* 2022) and LLaMA-2-7B; the best probing layer is the 20th for OPT and roughly middle-of-the-network for LLaMA-2-7B. Average per-topic accuracy ranges from 60–80% on OPT and 70–90% on LLaMA-2-7B. Critically, simpler baselines — few-shot prompting, prefixing the sentence with *"It is true that"* and reading the token probability — give at most ~59%, confirming that the hidden states encode information *not* visible in the surface logits.

Azaria & Mitchell's contribution is to establish empirically that *the model knows when it is lying* — at least in the linear-probe sense — and that the relevant signal is in the hidden state rather than in the surface logits. Two limitations motivate the next generation of methods: the labels are hand-crafted, which limits scale, and the probe is trained on *reading-mode* statements rather than *generation-mode* outputs, so it may not transfer to the LLM's own free generation.

### 2.5.2 MIND (Su *et al.*, 2024) — unsupervised real-time detection from internal states

**MIND** (Su, Wang, Ai, Hu, Wu, Zhou & Liu, Findings of ACL 2024) is the central reference of this dissertation. Its contribution is to remove the need for hand-crafted labels, removing SAPLMA's main bottleneck, and to demonstrate that the resulting probe is fast enough for real-time deployment.

**Unsupervised label generation.** The authors take Wikipedia articles (specifically WikiText-103). For each article they identify named entities using spaCy and select one entity *that occurs after the first sentence*. They truncate the article at that entity's position and ask the LLM to continue. If the LLM's first generated token matches the entity in the article (specifically, if it is among the top-K predictions, with K = 4 in their setup), the continuation is labelled *non-hallucinated* (y = 0). Otherwise the LLM produced a different entity — by definition factually incorrect with respect to the article — and the continuation is labelled *hallucinated* (y = 1). The substitution is then completed: the model is allowed to generate until a *post-entity anchor token* (whatever followed the gold entity, e.g. a comma) reappears, at which point we splice the rest of the original article back on. The result is a pair of texts that differ only in the entity span, one factual and one hallucinated, suitable for supervised training of a binary classifier.

This is essentially SAPLMA's swap-the-attribute idea, but the swap is performed by the *LLM itself*, conditioned on a real Wikipedia prefix. No human labels are needed.

**Hidden-state feature.** MIND extracts the contextualised embedding of *the last token at the last Transformer layer*, denoted Hₙᴺ in their notation, where n is the position of the last input token and N is the index of the final Transformer layer. The authors ablate this choice in §3.2.1 of their paper. Among the rows of their Table 1 — *All Layers, All Tokens*; *First & Last Layer, All Tokens*; *All Layers, Last Token*; *Last Layer, All Tokens*; *Last Layer, Last Token*; *Last Layer + All-Layers Last Token* — the last-token, last-layer view gives 0.7123 accuracy, the highest among single-feature configurations, and 0.7191 in combination with the all-layers last-token view. The classifier is a four-layer MLP (linear → ReLU → linear → ReLU → linear → ReLU → linear → sigmoid) with hidden sizes 256, 128, 64, 2; the loss is Binary Cross-Entropy (their Eq. 2); the optimiser is Adam with learning rate 5e-4, weight decay 1e-5, batch size 32, ten epochs. We adopt the same hyper-parameters in our implementation, save that we expand the input layer to 512 units, consistent with the choice made by Sharanya Dasgupta (§2.5.4) when generalising MIND to higher-dimensional hidden states.

**Benchmark.** The authors release **HELM** — Hallucination detection Evaluation for multiple LLMs — a 3,582-sentence corpus from 1,224 passages generated by six LLMs (Falcon-40B, GPT-J-6B, LLaMA-2-Base-7B, LLaMA-2-Chat-7B, LLaMA-2-Chat-13B, OPT-6.7B). Each passage is annotated for hallucination at both sentence and passage level, and the hidden states / attention maps recorded during inference are released alongside the text. HELM is currently the only public benchmark that provides aligned internal states from multiple LLMs.

**Results.** On HELM, MIND outperforms PP (Perplexity), PE (Predictive Entropy), all four SelfCheckGPT variants, SAPLMA, EUBHD, and GPT-4-as-judge at both sentence and passage levels. Representative passage-level AUC on LLaMA-2-Chat-7B is 0.8547. Critically, the authors show in Table 4 that the *customised per-LLM* training data is essential: using LLaMA-2-13B-Chat data to detect hallucinations in OPT-6.7B is little better than chance, so each target LLM needs its own pseudo-training corpus generated by the same LLM. This finding shapes the present dissertation's protocol: we generate per-model training data for the chosen 0.5B target model rather than reusing data from a larger or different LLM.

### 2.5.3 HaloScope (Du, Xiao & Li, NeurIPS 2024) — pseudo-labels from a latent subspace

**HaloScope** (Du, Xiao, Li, NeurIPS 2024) develops an alternative unsupervised labelling procedure that does not depend on Wikipedia entity substitution. The starting point is the observation that, in a corpus of LLM generations, the activations of factual and hallucinated samples occupy *different sub-spaces* of the activation manifold even before any labelling.

Concretely, HaloScope collects activations F ∈ ℝᴺˣᵈ from unlabelled LLM generations, centres them, and performs an SVD F = UΣV⊤. Each sample's projection onto the top-k singular vectors yields a score ζᵢ = (1/k) Σⱼ σⱼ ⟨fᵢ, vⱼ⟩². The empirical claim is that factual samples cluster near the mean (small ζ) while hallucinated samples are pushed into the principal direction of variation (large ζ). A simple threshold on ζ gives pseudo-labels, which are then used to train a small two-layer MLP truthfulness classifier with a sigmoid-loss surrogate.

HaloScope is the first work to evaluate on the four-dataset suite — TruthfulQA (817 QA pairs, generation track), TriviaQA (rc.nocontext, 9,960 dedup validation), CoQA (7,983 dev), and TydiQA-GP English (3,696) — that this dissertation also targets. On LLaMA-2-Chat-7B the reported AUROC is 78.64 / 77.40 / 76.42 / 94.04 across the four datasets, beating the next best (CCS*) by +10.69% on TruthfulQA in particular. The authors also evaluate on OPT-6.7B and 13B and find that the *layer* of activation matters: middle layers (8–14 on LLaMA-2-7B) are stronger than the final layer for OPT, while LLaMA prefers block outputs. This contrast with MIND's last-layer choice is one of the open problems we identify in §2.8.

### 2.5.4 HalluShift (Dasgupta, 2025) — multi-layer fusion with probabilistic features

The most directly relevant prior thesis is **HalluShift**, the detection contribution of Sharanya Dasgupta's M.Tech dissertation *"From Vigilance to Veracity: Hallucination Detection, Mitigation, and Safety Enhancement in Large Language Models"* (Indian Statistical Institute, Kolkata, June 2025, supervised by Dr. Swagatam Das). HalluShift is conceptually a *generalisation* of MIND in three directions.

First, instead of selecting a single hidden-state view, HalluShift extracts features from *every* Transformer layer of the target LLM. For each layer it computes Wasserstein distance and cosine similarity between successive hidden-state and attention distributions, capturing the *distribution shift* between layers. Range-wise feature selection chooses informative subsets automatically, removing the manual layer-choice problem that MIND, SAPLMA, and HaloScope all face by different ad-hoc rules.

Second, HalluShift fuses these distribution-shift features with three *token-probability* features: mean token probability (mtp), maximum positive shift (Mps), and mean negative log-probability gain (Mg). The fusion is performed in a 2-layer MLP with metric-learning loss; the projected representation is mapped to a single output node in [0, 1].

Third, the labelling protocol differs. HalluShift uses BLEURT-derived similarity between the LLM's free-generation answer and the gold reference as the source of soft labels, following HaloScope's convention; greedy decoding with a 64-token output cap is used.

The empirical evaluation is the strongest argument for studying HalluShift. The system is reported on TruthfulQA, TriviaQA, CoQA, TydiQA-GP, and the HaluEval triple (QA, Summarisation, Dialogue), on five base models (LLaMA-2-7B, LLaMA-3.1-8B, OPT-6.7B, Vicuna-7B, Qwen2.5-7B). Headline AUROCs on (TruthfulQA / TriviaQA / CoQA / TydiQA-GP) are 89.93 / 89.03 / 87.60 / 87.61 for LLaMA-2-7B and 92.97 / 99.23 / 90.38 / 87.70 for LLaMA-3.1-8B. On HaluEval, HalluShift dominates HalluDetect on Dialogue (Accuracy 0.88 vs. 0.66) and matches it on QA and Summarisation. This positions HalluShift as the current internal-state state-of-the-art for the same dataset suite that the present dissertation targets.

The relationship to our work is straightforward. MIND is a *strict subset* of HalluShift's representational toolkit: HalluShift includes MIND's hidden-state feature (last token at the last layer is one of the many layer signals it processes) and adds multi-layer fusion and token-probability features. The present dissertation deliberately *restricts* itself to MIND's representation in order to obtain a small-model ablation. Our research question becomes: *how much performance is sacrificed when one drops HalluShift's multi-layer fusion and probabilistic features in exchange for MIND's single-feature simplicity and the smaller compute footprint of a 0.5B-parameter target LLM?*

### 2.5.5 Other 2025 developments

Two further 2025 papers complete the internal-state line for our purposes. **"Hallucination Detection with the Internal Layers of LLMs"** (Sky-Mountain Lab, arXiv:2509.14254, Sept 2025) probes every Transformer layer of LLaMA-3-8B, Qwen-2.5-7B, and Mistral-7B against the same TruthfulQA / TriviaQA / HaluEval benchmarks and reports that the *optimal layer is model-dependent*, echoing the finding HalluShift bakes into its automated layer-selection. Their best AUROCs on TruthfulQA exceed 88% with single-layer probes — comparable to HalluShift's multi-layer fusion. This is methodologically important because it shows that the layer-selection problem is empirical, not merely a matter of intuition. **"Hallucination Detection with Small Language Models"** (arXiv:2506.22486, Jun 2025) takes a complementary angle: instead of probing the *generating* LLM, it uses several auxiliary small LMs to score the generation at sentence level. F1 improves by 10% over single-judge baselines. This work is interesting because it suggests that the *detector* itself does not need to be large, only well-trained — a finding that is broadly consistent with the MLP-on-small-LLM-features setup we adopt here.

## 2.6 Family IV — Retrieval-augmented fact verification

A fourth, fundamentally different family treats hallucination detection as a *fact verification* problem against an external knowledge base. The pipeline is canonical: parse the generation into atomic claims (or treat the whole generation as one claim), retrieve evidence from a corpus, and apply a textual entailment or stance classifier to decide *Supported*, *Refuted*, *Not Enough Evidence*, or *Conflicting*.

The reference dataset for this line is **FEVER** (Thorne *et al.* 2018) and its successors. The most recent shared task instantiation, **AVeriTeC** (Schlichtkrull *et al.* 2023), uses 3,068 train / 500 dev / 2,215 test real-world claims drawn from the open Web and forces a four-way verdict. The UHH submission (Sevgili *et al.* 2024, *"UHH at AVeriTeC: RAG for Fact-Checking with Real-World Claims"*, FEVER Workshop) is representative: BM25 retrieves a candidate pool of 10,000 evidences per claim; a Question-Retrieve or Retrieve-Question two-stage pipeline reranks to a small evidence shortlist; and a Mixtral-class LLM produces the verdict. Their Veracity@25 of 0.45 places them 6th of 23 systems.

For our purposes RAG-based fact verification is *adjacent* rather than competitive. MIND-style detectors need no retrieval, no external knowledge base, and no document parsing — they trade those off for the requirement that the *generating* LLM's internals are accessible. Conversely, RAG verification works on closed-API models but is bottlenecked by retrieval quality and is fundamentally slower. The two families are complementary in production deployments and should both be reported in a complete evaluation.

It is worth noting that the candidate's own prior work — the AVeriTeC summary (Sahoo, 2025) included with the project files — sits in this RAG family. The shift to internal-state probing in the present dissertation is therefore a *change of family*, not an incremental improvement on a single line.

## 2.7 Datasets and evaluation protocols

Six datasets are central to the present dissertation. Their characteristics and prompt formats — as used by the candidate in the multi-task extension — are summarised below.

**TruthfulQA** (Lin, Hilton & Evans, 2022). 817 questions hand-designed to elicit common human misconceptions. The generation track requires a free-text answer scored against a set of true and false reference answers using GPT-judge or BLEURT. Prompt format used here:

> *Answer the question concisely. Q: What happens to you if you eat watermelon seeds? A:*

**TriviaQA** (Joshi *et al.* 2017). 9,960 dedup validation questions from quiz-bowl style competitions. The `rc.nocontext` configuration provides only the question, no supporting document. Prompt format:

> *Answer the question concisely. Q: In Greek mythology, who were Arges, Brontes and Steropes? A:*

**CoQA** (Reddy *et al.* 2019). 7,983 dev questions over passages of conversational context (children's stories, news, exams). Each question depends on the prior turn; we evaluate single-turn slices. Prompt format:

> *Answer the question concisely based on the context: Context: Once upon a time, in a barn near a farm house ... Q: Where did Cotton's mother put her to clean the paint off? A:*

**TydiQA-GP (English)** (Clark *et al.* 2020). 3,696 dev questions; we use the English GoldP subset for parity with HaloScope and HalluShift. Prompt format:

> *Answer the question concisely based on the context: Context: X-Men, also known as X-Men: The Animated Series ... Q: When was the X-Men cartoon television series first aired? A:*

**HaluEval-QA / HaluEval-Summarisation / HaluEval-Dialogue** (Li *et al.* 2023, EMNLP). 30,000 synthetic examples (10,000 per task), each containing a model-generated answer/summary/response that has been programmatically modified to introduce a hallucinated span. The task is binary detection at the sample level. Prompt formats used here:

> HaluEval-QA: *Answer the question concisely based on the context: Context: The nine mile byway starts south of Morehead, Kentucky ... Q: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail? A:*
>
> HaluEval-Summarisation: *Residents of central Sanaa, the Yemeni capital, have learned the hard way that key strategic bombing targets are located ... Please summarise the above article concisely. A:*
>
> HaluEval-Dialogue: *You are an assistant that answers questions concisely and accurately. Use the knowledge and conversation to respond naturally to the most recent message. Knowledge: Iron Man is starring Robert Downey Jr. Robert Downey Jr. starred in Zodiac. Zodiac is starring Jake Gyllenhaal. Conversations: [Human]: Do you like Iron Man? [Assistant]: Sure do! Robert Downey Jr. is a favourite. [Human]: Yes I like him too — did you know he also was in Zodiac, a crime fiction film? [Assistant]:*

Across the literature, three evaluation metrics dominate: **AUROC** (the most discriminative, used by HaloScope, HalluShift, MIND), **Accuracy / F1** (used by HaluEval and HalluDetect), and **PR-AUC** (used by SelfCheckGPT for the heavily class-imbalanced WikiBio benchmark). For the present dissertation we report Accuracy, Precision, Recall, F1, and AUROC, following the present-author's existing pipeline.

The dataset coverage matrix below summarises which prior work evaluates on which dataset; **only HaloScope and HalluShift evaluate on the four-QA suite this dissertation targets, and only HalluShift evaluates on the HaluEval triple.**

| Dataset | MIND | SCG | SAPLMA | HaloScope | Kadavath | HalluShift |
|---|---|---|---|---|---|---|
| TruthfulQA | — | — | — | ✔ | ✔ | ✔ |
| TriviaQA | — | — | — | ✔ | ✔ | ✔ |
| CoQA | — | — | — | ✔ | — | ✔ |
| TydiQA-GP | — | — | — | ✔ | — | ✔ |
| HaluEval-QA | — | — | — | — | — | ✔ |
| HaluEval-Summ | — | — | — | — | — | ✔ |
| HaluEval-Dialog | — | — | — | — | — | ✔ |
| HELM (MIND) | ✔ | — | — | — | — | — |
| WikiBio-GPT3 | — | ✔ | — | — | — | — |

## 2.8 Open problems

The literature converges on several open questions, each of which directly affects the present dissertation.

**Cross-model generalisation.** MIND's Table 4 demonstrates that a classifier trained on LLaMA-2-13B-Chat data transfers poorly to OPT-6.7B; the same finding is replicated by HaloScope and by the 2025 Sky-Mountain probing paper. Per-LLM training data appears non-negotiable. The present dissertation therefore generates its pseudo-training corpus on its target Qwen-2.5-0.5B model and does *not* reuse data from a larger LLM.

**Layer selection.** SAPLMA selects layer 20 on OPT-6.7B by held-out validation; MIND uses the last layer by ablation; HaloScope finds middle layers (8–14 on LLaMA-2-7B) best for the LLaMA family but feedforward-output for OPT; HalluShift automates the layer choice via range-wise feature selection. A 0.5B model has only 24 layers, fewer degrees of freedom than the 32-layer LLaMA-2-7B that most reported results assume. Our implementation defaults to MIND's last-layer choice but exposes alternative layer views (`mean1` for first-layer, `hds` for all-layer last-token average) so that a small ablation on Qwen-2.5-0.5B's layer profile is feasible.

**Span-level vs. sentence-level granularity.** MIND, SelfCheckGPT, and HaloScope all operate at sentence or passage level. Min *et al.* 2023 (FActScore) and Chen *et al.* 2023 (FELM) push toward atomic-fact granularity, finding that 30–60% of sentences contain *both* factual and hallucinated atoms. The MIND classifier, applied per-token at inference, produces a sequence of hallucination probabilities and is therefore one of the few methods that *natively* supports span-level detection without modification — a property the present dissertation will exploit for its qualitative analysis.

**Calibration as well as classification.** Kadavath *et al.* 2022 evaluate calibration (Brier, ECE) rather than only AUROC. Hallucination detectors are downstream decision-support tools — a calibrated probability is more useful than a binary verdict — and yet AUROC and Accuracy dominate the literature. We follow the convention but note this as a gap.

**Computational cost.** SelfCheckGPT requires N (≈20) full generations per query, putting it at 18–300× the cost of the original generation. MIND's overhead is ~3% of generation time. The cost gap is the practical motivation for the entire internal-state probing line, and is a strong argument for adopting MIND in latency-sensitive production deployments.

**Black-box vs. white-box.** Internal-state methods presume access to hidden states, ruling out closed-API systems. The market reality is that mission-critical applications are increasingly deployed on open-weights models (LLaMA, Qwen, Mistral, Phi, Gemma) precisely because hidden states *are* needed — for hallucination detection, watermarking, alignment audits, and interpretability. The present dissertation, targeting Qwen-2.5-0.5B, is squarely in the open-weights regime.

**Sample efficiency.** MIND trains on 5,000 pseudo-labelled samples; the authors' Table 5 shows that 4,096 is the empirical sweet spot. HaloScope and HalluShift use 7,000–10,000 per dataset. For a 0.5B model, generating 4,000 samples takes roughly two hours on a free Colab T4, so the data-generation phase is feasible at modest cost. We adopt 1,000 pseudo-samples per class (2,000 total) as a deliberate small-data ablation, then scale to MIND's 5,000-sample regime in a follow-up run.

## 2.9 Positioning the present work

The dissertation occupies a specific, deliberately narrow slot in this landscape. The contribution is *not* a new method — MIND's framework is adopted essentially unchanged — but a *re-implementation, ablation, and extension* with three concrete deliverables.

First, we reproduce MIND on **Qwen-2.5-0.5B**, the smallest model in the contemporary instruction-tuned-open-weights frontier (mid-2026). This contributes a new data point to the model-size sweep for MIND-family detectors: all prior reports use 6.7B–13B parameter models and 4-bit quantisation has been applied only to enable larger-model inference, never to a sub-1B base model. The smallest model previously reported in the MIND-family line is OPT-6.7B in HaloScope. We test whether the MIND signal — last-token, last-layer activations distinguishing hallucinated from non-hallucinated text — persists at one-thirteenth that scale.

Second, we evaluate the trained classifier on the **seven-dataset suite** used by HalluShift (TruthfulQA, TriviaQA, CoQA, TydiQA-GP, HaluEval-{QA, Summarisation, Dialogue}). This means transferring a probe trained on Wikipedia-continuation pseudo-labels to QA, summarisation, and dialogue tasks. The transferability of MIND-style probes across task domains has not been systematically evaluated in the literature; this is therefore an empirical contribution in its own right. We adopt the prompt formats specified in the project requirements (reproduced verbatim in §2.7) for consistency with the candidate's planned ablation matrix.

Third, we provide a **side-by-side small-model ablation of HalluShift**. By holding all evaluation conditions constant — same datasets, same prompt formats, same metric set — and varying only (i) the model size (0.5B vs. 7B–8B), (ii) the feature set (single hidden-state view vs. multi-layer fusion plus probabilistic features), we obtain a controlled comparison of *what HalluShift's additions buy*. If MIND-on-Qwen-2.5-0.5B closes most of the AUROC gap to HalluShift-on-LLaMA-2-7B, the multi-layer feature engineering is empirically less critical than the literature has assumed; if not, the gap quantifies the value of HalluShift's contribution.

In summary, the present work positions itself as a *minimum-viable* member of the internal-state probing family: smallest target model, simplest hidden-state feature, simplest MLP classifier, broadest dataset coverage. It is intended as a baseline against which more sophisticated methods can be measured, and as a demonstration that hallucination detection from internal states remains viable at the sub-1B parameter scale where the operational cost of any add-on (a sampling loop, an SVD step, a multi-layer feature stack) is most strongly felt.

---

## References

- Azaria, A., & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying*. Findings of EMNLP 2023. arXiv:2304.13734.
- Binkowski, J., et al. (2025). *Hallucination Detection in LLMs Using Spectral Features of Attention Maps*. arXiv:2502.17598.
- Chen, S., Xiong, M., et al. (2024). *EigenScore: Inside-Out Hallucination Detection*. (referenced via HaloScope baselines).
- Clark, J. H., Choi, E., Collins, M., Garrette, D., Kwiatkowski, T., Nikolaev, V., & Palomaki, J. (2020). *TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages*. TACL 8, 454–470.
- Dasgupta, S. (2025). *From Vigilance to Veracity: Hallucination Detection, Mitigation, and Safety Enhancement in Large Language Models*. M.Tech dissertation, Indian Statistical Institute, Kolkata. Supervised by Dr. Swagatam Das.
- Dhingra, B., Faruqui, M., Parikh, A., Chang, M.-W., Das, D., & Cohen, W. W. (2019). *Handling Divergent Reference Texts when Evaluating Table-to-Text Generation* (PARENT). ACL.
- Du, X., Xiao, C., & Li, Y. (2024). *HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection*. NeurIPS 2024.
- Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Chen, D., Dai, W., Chan, H. S., Madotto, A., & Fung, P. (2023/2024). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys 55(12). arXiv:2202.03629.
- Joshi, M., Choi, E., Weld, D., & Zettlemoyer, L. (2017). *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*. ACL.
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). *Language Models (Mostly) Know What They Know*. arXiv:2207.05221.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*. ICLR.
- Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*. EMNLP.
- Lin, S., Hilton, J., & Evans, O. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. ACL.
- Manakul, P., Liusie, A., & Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*. EMNLP. arXiv:2303.08896.
- Min, S., Krishna, K., Lyu, X., et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. EMNLP.
- Reddy, S., Chen, D., & Manning, C. D. (2019). *CoQA: A Conversational Question Answering Challenge*. TACL.
- Sahoo, C. (2025). *Plain-language Summary of "UHH at AVeriTeC: RAG for Fact-Checking with Real-World Claims"*. Working document.
- Sevgili, Ö., Nikishina, I., Yimam, S. M., Semmann, M., & Biemann, C. (2024). *UHH at AVeriTeC: RAG for Fact-Checking with Real-World Claims*. Proceedings of the Seventh FEVER Workshop, ACL 2024, 55–63.
- Sky-Mountain Lab (2025). *Hallucination Detection with the Internal Layers of LLMs*. arXiv:2509.14254.
- Su, W., Wang, C., Ai, Q., Hu, Y., Wu, Z., Zhou, Y., & Liu, Y. (2024). *Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models*. Findings of ACL 2024, 14379–14391.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL-HLT.
- Anonymous (2025). *Hallucination Detection with Small Language Models*. arXiv:2506.22486.
