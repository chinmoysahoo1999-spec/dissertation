"""
Build the enhanced project.ipynb template + 5 model-specific copies.

Each copy is a self-contained notebook that:
  1. Loads its target LLM (bf16, device_map=auto)
  2. Generates 1000 hallucinated + 1000 non-hallucinated samples from Wikipedia
     using the MIND entity-substitution protocol; for each sample it stores
     the canonical MIND embedding (last-token last-layer) + 3 new features:
       D_mean   = mean cosine distance between adjacent layers (drift)
       V_last   = L2 variance of last-token activations across layers
       H_mean   = mean Shannon entropy of per-step token distribution
  3. Trains a 5-layer MLP on the concatenated feature vector
     [canonical (D) || D_mean || V_last || H_mean]   (length = D + 3)
     with StandardScaler fitted on the 3 scalar tail.
  4. Reports Wikipedia held-out metrics.
  5. Downloads & evaluates on the 7 downstream datasets:
       TruthfulQA, TriviaQA, CoQA, TydiQA-GP English,
       HaluEval-{QA, Summarization, Dialogue}
  6. Writes ONE combined results file:
       {MODEL_TAG}_results.json    HalluShift-style per-dataset rows

Run:  python _build_notebooks.py
Produces:
       project_tinyllama_1.1b.ipynb         (Colab Free)
       project_qwen2.5_3b.ipynb             (Colab Free)
       project_opt_2.7b.ipynb               (Colab Free)
       project_gptj_6b.ipynb                (Colab Free, tight)
       project_qwen2.5_7b.ipynb             (Kaggle Free T4x2)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Seven model targets — all non-gated, all used in MIND / HaloScope / HalluShift / SAPLMA literature.
#
# Colab Free (single T4, 16 GB):
#   tinyllama_1.1b   — mid-eval baseline (Llama family)
#   qwen2.5_3b       — mid-eval baseline (Qwen family)
#   opt_2.7b         — small OPT, used in MIND/HaloScope/HalluShift/SAPLMA
#   gptj_6b          — used in MIND (12 GB bf16, tight on 16 GB T4)
#
# Kaggle Free (T4 x 2, 32 GB total via device_map='auto'):
#   qwen2.5_7b       — used in HalluShift, headline scale-up
#   opt_6.7b         — used in MIND/HaloScope/HalluShift/SAPLMA (4-paper baseline)
#   falcon_7b        — Falcon family from MIND paper (MIND uses Falcon-40B; 7B is the free sibling)
MODELS = [
    ("tinyllama_1.1b",   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "colab",  1000),
    ("qwen2.5_3b",       "Qwen/Qwen2.5-3B",                     "colab",  1000),
    ("opt_2.7b",         "facebook/opt-2.7b",                   "colab",  1000),
    ("gptj_6b",          "EleutherAI/gpt-j-6b",                 "colab",   600),  # heavier; smaller N
    ("qwen2.5_7b",       "Qwen/Qwen2.5-7B-Instruct",            "kaggle", 1000),
    ("opt_6.7b",         "facebook/opt-6.7b",                   "kaggle", 1000),
    ("falcon_7b",        "tiiuae/falcon-7b",                    "kaggle", 1000),
]


def cell_code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def cell_md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


# -----------------------------------------------------------------------------
# Cell sources — common template (MODEL_NAME and TARGET_SAMPLES are placeholders)
# -----------------------------------------------------------------------------

CELL_INTRO = """# `__NOTEBOOK_NAME__.ipynb` — MIND-style hallucination detection with drift + variance + entropy

**Model:** `__MODEL_NAME__`
**Target compute:** __TARGET_ENV__
**Samples / class:** __TARGET_SAMPLES__ (Wikipedia continuation, MIND Algorithm 1)

Pipeline:
1. Load LLM (bf16, model-parallel across all available GPUs).
2. Generate hallucinated + non-hallucinated Wikipedia continuations.
3. Per sample, extract:
   - **canonical**: last-token last-layer hidden state (D-dimensional)
   - **D_mean**: mean cosine distance between adjacent layers
   - **V_last**: L2 variance of last-token activations across layers
   - **H_mean**: mean Shannon entropy of per-step token distribution
4. Train a 5-layer MLP on `[canonical || D_mean || V_last || H_mean]` (StandardScaler on the 3 scalars).
5. Evaluate on Wikipedia held-out + 7 downstream datasets (TruthfulQA, TriviaQA, CoQA, TydiQA-GP, HaluEval-{QA, Summ, Dialog}).
6. Write `__MODEL_TAG___results.json` (HalluShift-style) for downstream comparison.
"""

CELL_0_INSTALL = '''# =============================================================================
# BLOCK 0: PIP INSTALLS
# =============================================================================
!pip install -q -U "transformers>=4.45" "tokenizers>=0.19" "accelerate>=0.30"
!pip install -q -U datasets spacy nltk scikit-learn tqdm sentence-transformers
!pip install -q -U bitsandbytes   # optional, only used if USE_4BIT
!python -m spacy download en_core_web_sm
'''

CELL_1_LOAD = '''# =============================================================================
# BLOCK 1: CONFIG + IMPORTS + MODEL LOADING
# =============================================================================
import os, gc, json, random, math, time, datetime, platform
import numpy as np
import torch
import torch.nn.functional as F
import spacy, nltk
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import StandardScaler

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

print("=" * 80)
print("BLOCK 1: MODEL LOADING")
print("=" * 80)

# -----------------------------------------------------------------------------
# Configuration — these are the only things to edit per-model
# -----------------------------------------------------------------------------
MODEL_NAME      = "__MODEL_NAME__"
TARGET_SAMPLES  = __TARGET_SAMPLES__       # per class
TOPK_FIRST_TOKEN = 4
WINDOWS         = 16
SEED            = 42
USE_4BIT        = False
DTYPE           = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MODEL_TAG = MODEL_NAME.split("/")[-1].replace(".", "").replace("-", "").lower()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# Tokeniser + model
# -----------------------------------------------------------------------------
print(f"\\nLoading {MODEL_NAME} (4bit={USE_4BIT}, dtype={DTYPE}) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

load_kwargs = dict(trust_remote_code=False, torch_dtype=DTYPE,
                   device_map="auto" if torch.cuda.is_available() else None)
if USE_4BIT and torch.cuda.is_available():
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE, bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()
for p in model.parameters():
    p.requires_grad = False

nlp = spacy.load("en_core_web_sm")

# Banner
if torch.cuda.is_available():
    print(f"\\u2713 GPU: {torch.cuda.get_device_name(0)}")
    print(f"\\u2713 Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
print(f"\\u2713 Hidden dim: {model.config.hidden_size}")
print(f"\\u2713 Layers (N): {model.config.num_hidden_layers}")
print(f"\\u2713 Model tag : {MODEL_TAG}")
'''

CELL_2_ENT = '''# =============================================================================
# BLOCK 2: ENTITY EXTRACTION (spaCy NER + post-processing)
# =============================================================================
def delete_substrings(lst):
    substrings = []
    lst = list(set(lst))
    for s in lst:
        if any(s in o for o in lst if o != s):
            substrings.append(s)
    for s in substrings:
        lst.remove(s)
    return lst


def find_boundaries(text, words):
    boundaries = []
    for word in words:
        ntext = text
        while True:
            start = ntext.find(word)
            if start == -1: break
            end = start + len(word) - 1
            while start > 0 and ntext[start-1] != " ": start -= 1
            while end < len(ntext)-1 and ntext[end+1] != " ": end += 1
            boundaries.append("".join(ntext[i] for i in range(start, end+1)))
            ntext = ntext[end+1:]
    return boundaries


def get_entities(text):
    ents = list({str(e) for e in nlp(text).ents})
    ents = find_boundaries(text, ents)
    ents = delete_substrings(ents)
    out = []
    for i in range(len(text)):
        for e in ents:
            if text[i:].startswith(e):
                out.append((e, i))
    return out


print("\\u2713 entity extractor defined")
'''

CELL_3_TOK = '''# =============================================================================
# BLOCK 3: TOKEN-BOUNDARY FINDER (MIND Algorithm 1, step 2)
# =============================================================================
def find_first_and_next_token(text, e, idx, input_id, prompt=""):
    new_text = f"{text[:idx].strip()} {text[idx:].replace(e, e+' @', 1).strip()}"
    new_input_id = tokenizer(prompt + new_text.strip(), return_tensors="pt")["input_ids"].tolist()[0]
    for i in range(len(input_id[0])):
        if input_id[0][i] != new_input_id[i]:
            return []
    first_token = new_input_id[len(input_id[0])]
    at_cands = tokenizer("@", add_special_tokens=False)["input_ids"]
    at_cands += tokenizer(" @", add_special_tokens=False)["input_ids"]
    at_pos = None
    for at_tok in at_cands:
        try:
            at_pos = new_input_id.index(at_tok, len(input_id[0]))
            break
        except ValueError: continue
    if at_pos is None or at_pos >= len(new_input_id) - 1:
        return []
    next_token = new_input_id[at_pos + 1]
    entity_len = at_pos - len(input_id[0])
    last_id = new_input_id[at_pos + 1:]
    return [first_token, next_token, entity_len, last_id]


print("\\u2713 token finder defined")
'''

CELL_4_FEATURES = '''# =============================================================================
# BLOCK 4: FEATURE EXTRACTION  (canonical MIND + D_mean + V_last)
# =============================================================================
# canonical_mind = H_N^n          (MIND paper Sec 3.2.2 — last token, last layer)
# D_mean         = mean over layers of cosine distance between adjacent layers
#                  at the last-token position  (layer-wise representation drift)
# V_last         = L2 cross-layer variance of the last-token activations
# H_mean         = mean per-step Shannon entropy of token distribution
#                  (computed during model.generate, not here)
# =============================================================================
@torch.no_grad()
def extract_features(text: str):
    enc = tokenizer(text.strip(), return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                # tuple length (N+1); skip embedding layer
    last_token_per_layer = torch.stack(
        [h[0, -1, :].float() for h in hs[1:]], dim=0
    )                                     # [N, D]

    # canonical (MIND default)
    canonical = last_token_per_layer[-1].cpu().tolist()

    # D_mean — cosine distance between consecutive layers (drift)
    a = last_token_per_layer[:-1]
    b = last_token_per_layer[1:]
    cos = F.cosine_similarity(a, b, dim=1)                # [N-1]
    D_mean = float((1.0 - cos).mean().item())

    # V_last — L2 variance across layers at the last token
    mean_h = last_token_per_layer.mean(dim=0)             # [D]
    V_last = float(((last_token_per_layer - mean_h) ** 2).sum(dim=1).mean().item())

    return canonical, D_mean, V_last


print("\\u2713 extract_features defined (canonical, D_mean, V_last)")
'''

CELL_5_SANITY = '''# =============================================================================
# BLOCK 5: SANITY TEST
# =============================================================================
sample_text = ("Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the "
               "German Empire, on 14 March 1879. His parents were Hermann Einstein, "
               "a salesman and engineer, and Pauline Koch.")

ents = get_entities(sample_text)
title = "Albert Einstein"
valid = [(e, i) for e, i in ents if i != 0 and e.lower() not in title.lower()]
print(f"\\u2713 entity occurrences: {len(ents)}")
print(f"\\u2713 valid entities    : {len(valid)}")
if valid:
    canon, D_mean, V_last = extract_features(sample_text)
    print(f"\\u2713 canonical dim     : {len(canon)}  (== model hidden_size)")
    print(f"\\u2713 D_mean            : {D_mean:.4f}  (cosine drift in [0, 2])")
    print(f"\\u2713 V_last            : {V_last:.4f}  (L2 variance, > 0)")
'''

CELL_6_DATAGEN = '''# =============================================================================
# BLOCK 6: DATA GENERATION  (MIND Algorithm 1, with H_mean computed during gen)
# =============================================================================
def per_step_entropy(scores):
    """scores: tuple of [1, vocab] logits per step. Returns mean Shannon entropy."""
    if not scores:
        return 0.0
    Hs = []
    for s in scores:
        p = torch.softmax(s[0].float(), dim=-1)
        # add tiny epsilon to avoid log(0)
        H = -(p * (p.clamp_min(1e-12)).log()).sum().item()
        Hs.append(H)
    return float(sum(Hs) / len(Hs))


def generate_sample(text, entity, idx, title):
    # Guard against OPT/GPT-J/Llama2 max_position_embeddings overflow.
    # If the prefix + windowed continuation + suffix would exceed the model's
    # positional window, skip the article — otherwise CUDA gather will assert.
    MAX_POS = getattr(model.config, "max_position_embeddings", 4096)
    HEADROOM = WINDOWS + 64
    enc = tokenizer(text[:idx].strip(), return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]; attn = enc["attention_mask"]
    if input_ids.shape[1] + HEADROOM >= MAX_POS:
        return None
    input_id_list = input_ids.tolist()
    toks = find_first_and_next_token(text, entity, idx, input_id_list)
    if not toks:
        return None
    first_, next_, entity_len, last_id = toks
    # Also guard the suffix length
    if input_ids.shape[1] + entity_len + len(last_id) + HEADROOM >= MAX_POS:
        return None

    gen = model.generate(input_ids, attention_mask=attn,
                         max_new_tokens=entity_len + WINDOWS,
                         return_dict_in_generate=True, output_scores=True,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
    # if model already knows the gold entity, skip
    if first_ in torch.topk(gen.scores[0], k=TOPK_FIRST_TOKEN).indices[0].tolist():
        return None
    found_step = None
    for step, sc in enumerate(gen.scores):
        if next_ in torch.topk(sc, k=TOPK_FIRST_TOKEN).indices[0].tolist():
            found_step = step; break
    if found_step is None:
        return None

    H_mean_hall = per_step_entropy(gen.scores[: found_step + 1])
    new_seq = gen.sequences[0, : input_ids.shape[1] + found_step].tolist()
    new_entity_ids = new_seq[input_ids.shape[1]:]
    hallucinated_ids = input_id_list[0] + new_entity_ids + last_id
    hallucinated_text = tokenizer.decode(hallucinated_ids, skip_special_tokens=True)
    new_entity = tokenizer.decode(new_entity_ids, skip_special_tokens=True).strip().lower()
    if not new_entity or entity.lower() in new_entity or new_entity in text.lower():
        return None

    # For the non-hallucinated reference we need its entropy too — re-generate just to read scores
    # Cheap proxy: forward pass and read p(x_{t+1} | x_{<=t}) at each entity-suffix position.
    # We approximate H_mean for the original by computing entropy of the next-token distribution
    # at the position immediately before the gold entity (single token sample).
    with torch.no_grad():
        out_orig = model(input_ids)
        last_logits = out_orig.logits[0, -1, :].float()
        p_orig = torch.softmax(last_logits, dim=-1)
        H_mean_orig = float(-(p_orig * p_orig.clamp_min(1e-12).log()).sum().item())

    canon_h, D_h, V_h = extract_features(hallucinated_text)
    canon_o, D_o, V_o = extract_features(text)

    return {
        "text_hall": hallucinated_text, "entity_hall": new_entity,
        "canon_h": canon_h, "D_h": D_h, "V_h": V_h, "H_h": H_mean_hall,
        "text_orig": text, "entity_orig": entity,
        "canon_o": canon_o, "D_o": D_o, "V_o": V_o, "H_o": H_mean_orig,
        "title": title,
    }


print(f"\\nLoading Wikipedia (streaming) ...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

dataset_hall, dataset_non_hall = [], []
processed = 0
print(f"Generating up to {TARGET_SAMPLES*2} samples ({TARGET_SAMPLES}/class)\\n")
pbar = tqdm(total=TARGET_SAMPLES * 2, desc="samples")
for row in wiki:
    if len(dataset_hall) >= TARGET_SAMPLES and len(dataset_non_hall) >= TARGET_SAMPLES:
        break
    processed += 1
    try:
        sents = sent_tokenize(row["text"])
        if len(sents) < 2: continue
        text = " ".join(sents[:2])
        title = row.get("title", "")
        ents = get_entities(text)
        if not ents: continue
        seen, ents_filt = set(), []
        for e, i in ents:
            if i == 0 or e.lower() in title.lower(): continue
            if i not in seen:
                seen.add(i); ents_filt.append((e, i))
        if not ents_filt: continue
        entity, char_idx = random.choice(ents_filt)
        result = generate_sample(text, entity, char_idx, title)
        if result is None: continue

        if len(dataset_hall) < TARGET_SAMPLES:
            dataset_hall.append({
                "label": 1, "text": result["text_hall"], "entity": result["entity_hall"],
                "embedding": result["canon_h"], "D_mean": result["D_h"],
                "V_last": result["V_h"], "H_mean": result["H_h"], "title": result["title"],
            })
            pbar.update(1)
        if len(dataset_non_hall) < TARGET_SAMPLES:
            dataset_non_hall.append({
                "label": 0, "text": result["text_orig"], "entity": result["entity_orig"],
                "embedding": result["canon_o"], "D_mean": result["D_o"],
                "V_last": result["V_o"], "H_mean": result["H_o"], "title": result["title"],
            })
            pbar.update(1)
        pbar.set_postfix(H1=len(dataset_hall), H0=len(dataset_non_hall), proc=processed)
        if (len(dataset_hall) + len(dataset_non_hall)) % 200 == 0:
            with open(f"{MODEL_TAG}_checkpoint.json", "w") as f:
                json.dump(dataset_hall + dataset_non_hall, f)
    except Exception as e:
        if processed % 1000 == 0:
            print(f"\\n[warn at {processed}]: {e}")
        continue
    if processed % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

pbar.close()
print(f"\\n\\u2713 done. H=1: {len(dataset_hall)}  H=0: {len(dataset_non_hall)}")
'''

CELL_7_SPLIT = '''# =============================================================================
# BLOCK 7: TRAIN / TEST SPLIT + SAVE
# =============================================================================
dataset = dataset_hall + dataset_non_hall
random.shuffle(dataset)
split = int(0.8 * len(dataset))
train_data, test_data = dataset[:split], dataset[split:]
TRAIN_PATH = f"{MODEL_TAG}_train.json"
TEST_PATH  = f"{MODEL_TAG}_test.json"
with open(TRAIN_PATH, "w") as f: json.dump(train_data, f)
with open(TEST_PATH,  "w") as f: json.dump(test_data,  f)
print(f"\\u2713 train: {len(train_data)}   test: {len(test_data)}")
'''

CELL_8_MLP = '''# =============================================================================
# BLOCK 8: MLP CLASSIFIER  (input = [canonical || z(D_mean) || z(V_last) || z(H_mean)])
# =============================================================================
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score, confusion_matrix, classification_report,
                              brier_score_loss)


class MINDPlusClassifier(nn.Module):
    """5-layer MLP. Input includes canonical embedding + 3 scalar features."""
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.layers(x)


def build_feature_matrix(records, scaler=None, fit=False):
    """Concatenate [canonical || z(D_mean) || z(V_last) || z(H_mean)]."""
    canon = np.array([r["embedding"] for r in records], dtype=np.float32)
    scalars = np.array([[r["D_mean"], r["V_last"], r["H_mean"]] for r in records],
                       dtype=np.float32)
    if fit:
        scaler = StandardScaler().fit(scalars)
    scalars_z = scaler.transform(scalars).astype(np.float32)
    X = np.concatenate([canon, scalars_z], axis=1)
    y = np.array([r["label"] for r in records], dtype=np.int64)
    return X, y, scaler


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


print("\\u2713 MLP + scaler helpers defined")
'''

CELL_9_TRAIN = '''# =============================================================================
# BLOCK 9: TRAIN MLP ON COMBINED FEATURES
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tr, y_tr, scaler = build_feature_matrix(train_data, fit=True)
X_te, y_te, _      = build_feature_matrix(test_data,  scaler=scaler, fit=False)
input_dim = X_tr.shape[1]
print(f"\\u2713 feature dim: {input_dim}  (= {input_dim-3} embedding + 3 scalars)")

mlp = MINDPlusClassifier(input_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(mlp.parameters(), lr=5e-4, weight_decay=1e-5)
train_loader = DataLoader(FeatureDataset(X_tr, y_tr), batch_size=32, shuffle=True)
test_loader  = DataLoader(FeatureDataset(X_te, y_te), batch_size=32, shuffle=False)

EPOCHS = 10; best_acc = 0.0
for ep in range(EPOCHS):
    mlp.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); l = loss_fn(mlp(x), y); l.backward(); opt.step()
    mlp.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (mlp(x).argmax(1) == y).sum().item(); total += y.size(0)
    acc_ep = correct / total
    star = " \\u2605" if acc_ep > best_acc else ""
    if acc_ep > best_acc:
        best_acc = acc_ep
        torch.save({"model_state": mlp.state_dict(), "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_, "input_dim": input_dim},
                   f"{MODEL_TAG}_mind_plus_best.pth")
    print(f"epoch {ep+1:2d}/{EPOCHS}  test_acc={acc_ep:.4f}{star}")
print(f"\\u2713 best test acc: {best_acc:.4f}")
'''

CELL_10_WIKI_EVAL = '''# =============================================================================
# BLOCK 10: WIKIPEDIA HELD-OUT EVALUATION (Acc/Prec/Rec/F1/AUROC/Brier)
# =============================================================================
mlp.load_state_dict(torch.load(f"{MODEL_TAG}_mind_plus_best.pth")["model_state"])
mlp.eval()
all_p, all_y, all_pr = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = mlp(x)
        prob = torch.softmax(logits, dim=1)[:, 1]
        all_p.extend(logits.argmax(1).cpu().numpy())
        all_y.extend(y.numpy())
        all_pr.extend(prob.cpu().numpy())
all_p = np.array(all_p); all_y = np.array(all_y); all_pr = np.array(all_pr)
acc  = accuracy_score(all_y, all_p)
prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_p, average="binary")
auc  = roc_auc_score(all_y, all_pr)
brier = brier_score_loss(all_y, all_pr)
cm   = confusion_matrix(all_y, all_p)
print(f"\\nWikipedia held-out test ({len(all_y)} samples):")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1       : {f1:.4f}")
print(f"  AUROC    : {auc:.4f}")
print(f"  Brier    : {brier:.4f}")
wiki_metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec),
                "f1": float(f1), "auc_roc": float(auc), "brier": float(brier),
                "n": int(len(all_y)),
                "cm": {"tn": int(cm[0,0]), "fp": int(cm[0,1]),
                       "fn": int(cm[1,0]), "tp": int(cm[1,1])}}
'''

CELL_11_DOWNLOAD_DATASETS = '''# =============================================================================
# BLOCK 11: DOWNLOAD 7 MULTI-TASK EVAL DATASETS (cached after first run)
# =============================================================================
print("Loading the 7 datasets ...")
def safe_load(loader_fn, label, subsample=None):
    try:
        ds = loader_fn()
        if subsample and len(ds) > subsample:
            ds = ds.shuffle(seed=SEED).select(range(subsample))
        print(f"  \\u2713 {label}: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"  \\u2717 {label}: FAILED — {e}")
        return None

# Subsample sizes — tuned to fit free-tier Colab/Kaggle session limits
SUBSAMPLES = {"truthfulqa": None,        # 817 — small, use all
              "triviaqa":   500,         # ~10k native, subsample
              "coqa":       500,         # ~8k native, subsample
              "tydiqa":     500,         # ~3.7k native, subsample
              "halueval_qa":     1000,   # 10k native, subsample (× 2 for both labels)
              "halueval_summ":   500,
              "halueval_dialog": 1000}

DATASETS = {}
DATASETS["truthfulqa"] = safe_load(
    lambda: load_dataset("truthful_qa", "generation", split="validation"),
    "truthfulqa", SUBSAMPLES["truthfulqa"])
DATASETS["triviaqa"] = safe_load(
    lambda: load_dataset("trivia_qa", "rc.nocontext", split="validation"),
    "triviaqa", SUBSAMPLES["triviaqa"])
DATASETS["coqa"] = safe_load(
    lambda: load_dataset("stanfordnlp/coqa", split="validation"),
    "coqa", SUBSAMPLES["coqa"])
DATASETS["tydiqa"] = safe_load(
    lambda: load_dataset("tydiqa", "secondary_task", split="validation")
             .filter(lambda x: "english" in x.get("id", "").lower()),
    "tydiqa", SUBSAMPLES["tydiqa"])
for split, label in [("qa","halueval_qa"), ("summarization","halueval_summ"),
                     ("dialogue","halueval_dialog")]:
    DATASETS[label] = safe_load(
        lambda s=split: load_dataset("pminervini/HaluEval", s, split="data"),
        label, SUBSAMPLES[label])

for k, v in DATASETS.items():
    print(f"  {k:18s} -> {len(v) if v else 'unavailable'}")
'''

CELL_12_SCORER = '''# =============================================================================
# BLOCK 12: LIGHTWEIGHT GOLD-VS-GENERATION SCORER (sentence-transformers cosine)
# =============================================================================
# For the open-ended QA datasets (TruthfulQA, TriviaQA, CoQA, TydiQA-GP) we need
# to label each generated answer as "matches gold" (label 0) or "doesn't" (1).
# BLEURT is heavy; sentence-transformers MiniLM (22M params) is plenty.
from sentence_transformers import SentenceTransformer
import numpy as np

scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device=str(device))

def score_match(gen: str, golds, threshold: float = 0.5) -> int:
    """0 = match (non-hallucination), 1 = no match (hallucination)."""
    if not golds:
        return 0
    if isinstance(golds, str):
        golds = [golds]
    embs = scorer.encode([gen] + list(golds), convert_to_numpy=True,
                          normalize_embeddings=True)
    sims = embs[0] @ embs[1:].T
    return 0 if float(sims.max()) >= threshold else 1

print("\\u2713 sentence-transformers scorer ready")
'''

CELL_13_MULTI_EVAL = '''# =============================================================================
# BLOCK 13: MULTI-TASK EVALUATION
# =============================================================================
@torch.no_grad()
def features_at_last_token(prompt: str, generation: str = ""):
    """Run the LLM on prompt+generation, extract canonical + D + V at last token."""
    text = (prompt + " " + generation).strip()
    canon, D, V = extract_features(text)
    # H_mean — for evaluation samples without re-generating, approximate by
    # forward-pass entropy of the last-token logits.
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc)
        p = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        H = float(-(p * p.clamp_min(1e-12).log()).sum().item())
    return canon, D, V, H


@torch.no_grad()
def generate_short_answer(prompt: str, max_new: int = 64) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new,
                          do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()


def classify_features(canon, D, V, H):
    """Run the trained MLP on a single sample's features."""
    scalars = scaler.transform([[D, V, H]]).astype(np.float32)
    x = np.concatenate([np.array(canon, dtype=np.float32), scalars[0]])
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    logits = mlp(x)
    p_hall = float(torch.softmax(logits, dim=1)[0, 1].item())
    pred = int(logits.argmax(1).item())
    return pred, p_hall


def eval_open_qa(ds, prompt_fn, gold_fn, n_max=None, dsname=""):
    """For datasets where we generate an answer + score it against gold."""
    preds, probs, golds = [], [], []
    iterable = ds if n_max is None else ds.select(range(min(n_max, len(ds))))
    for s in tqdm(iterable, desc=dsname):
        try:
            prompt = prompt_fn(s)
            gen    = generate_short_answer(prompt, max_new=48)
            gold   = gold_fn(s)
            label  = score_match(gen, gold)        # 0/1
            canon, D, V, H = features_at_last_token(prompt, gen)
            pred, prob = classify_features(canon, D, V, H)
            preds.append(pred); probs.append(prob); golds.append(label)
        except Exception:
            continue
    return np.array(golds), np.array(preds), np.array(probs)


def eval_halueval(ds, prompt_fn, right_key, wrong_key, n_max=None, dsname=""):
    """HaluEval: each sample provides both right_answer (0) and hallucinated (1)."""
    preds, probs, golds = [], [], []
    iterable = ds if n_max is None else ds.select(range(min(n_max, len(ds))))
    for s in tqdm(iterable, desc=dsname):
        try:
            prompt = prompt_fn(s)
            for ans_key, gold_label in [(right_key, 0), (wrong_key, 1)]:
                ans = s[ans_key]
                canon, D, V, H = features_at_last_token(prompt, ans)
                pred, prob = classify_features(canon, D, V, H)
                preds.append(pred); probs.append(prob); golds.append(gold_label)
        except Exception:
            continue
    return np.array(golds), np.array(preds), np.array(probs)


def compute_metrics(y, p, pr):
    if len(y) == 0:
        return {"n": 0}
    acc  = accuracy_score(y, p)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary",
                                                         zero_division=0)
    try:
        auc = roc_auc_score(y, pr)
    except ValueError:
        auc = float("nan")
    try:
        brier = brier_score_loss(y, pr)
    except ValueError:
        brier = float("nan")
    cm = confusion_matrix(y, p, labels=[0,1])
    return {"n": int(len(y)),
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "auc_roc": float(auc), "brier": float(brier),
            "cm": {"tn": int(cm[0,0]), "fp": int(cm[0,1]),
                   "fn": int(cm[1,0]), "tp": int(cm[1,1])}}


# -----------------------------------------------------------------------------
# Per-dataset evaluation
# -----------------------------------------------------------------------------
multitask = {}

if DATASETS["truthfulqa"] is not None:
    y, p, pr = eval_open_qa(DATASETS["truthfulqa"],
        prompt_fn=lambda s: f"Answer the question concisely. Q: {s['question']} A:",
        gold_fn=lambda s: s["correct_answers"],
        dsname="truthfulqa")
    multitask["truthfulqa"] = compute_metrics(y, p, pr)

if DATASETS["triviaqa"] is not None:
    y, p, pr = eval_open_qa(DATASETS["triviaqa"],
        prompt_fn=lambda s: f"Answer the question concisely. Q: {s['question']} A:",
        gold_fn=lambda s: s["answer"]["aliases"] + [s["answer"]["value"]],
        dsname="triviaqa")
    multitask["triviaqa"] = compute_metrics(y, p, pr)

if DATASETS["coqa"] is not None:
    def coqa_first_turn(s):
        # Use the first question of each conversation as a single-turn slice
        ctx = s["story"][:800]
        q = s["questions"][0] if s["questions"] else ""
        return f"Answer the question concisely based on the context: Context: {ctx} Q: {q} A:"
    def coqa_gold(s):
        a = s["answers"]
        return a["input_text"][0] if a["input_text"] else ""
    y, p, pr = eval_open_qa(DATASETS["coqa"],
        prompt_fn=coqa_first_turn, gold_fn=coqa_gold, dsname="coqa")
    multitask["coqa"] = compute_metrics(y, p, pr)

if DATASETS["tydiqa"] is not None:
    y, p, pr = eval_open_qa(DATASETS["tydiqa"],
        prompt_fn=lambda s: (
            f"Answer the question concisely based on the context: "
            f"Context: {s['context'][:800]} Q: {s['question']} A:"),
        gold_fn=lambda s: s.get("answers", {}).get("text", []),
        dsname="tydiqa")
    multitask["tydiqa"] = compute_metrics(y, p, pr)

if DATASETS["halueval_qa"] is not None:
    y, p, pr = eval_halueval(DATASETS["halueval_qa"],
        prompt_fn=lambda s: (f"Answer the question concisely based on the context: "
                             f"Context: {s['knowledge'][:600]} Q: {s['question']} A:"),
        right_key="right_answer", wrong_key="hallucinated_answer",
        dsname="halueval_qa")
    multitask["halueval_qa"] = compute_metrics(y, p, pr)

if DATASETS["halueval_summ"] is not None:
    y, p, pr = eval_halueval(DATASETS["halueval_summ"],
        prompt_fn=lambda s: f"{s['document'][:800]} Please summarise the above article concisely. A:",
        right_key="right_summary", wrong_key="hallucinated_summary",
        dsname="halueval_summ")
    multitask["halueval_summ"] = compute_metrics(y, p, pr)

if DATASETS["halueval_dialog"] is not None:
    y, p, pr = eval_halueval(DATASETS["halueval_dialog"],
        prompt_fn=lambda s: (f"You are an assistant. Knowledge: {s['knowledge'][:400]}\\n"
                             f"Dialogue: {s['dialogue_history'][:400]}\\n[Assistant]:"),
        right_key="right_response", wrong_key="hallucinated_response",
        dsname="halueval_dialog")
    multitask["halueval_dialog"] = compute_metrics(y, p, pr)

print("\\n=== MULTI-TASK RESULTS ===")
for k, v in multitask.items():
    if v.get("n", 0) > 0:
        print(f"  {k:18s}  n={v['n']:5d}  Acc={v['accuracy']:.3f}  "
              f"F1={v['f1']:.3f}  AUROC={v['auc_roc']:.3f}")
'''

CELL_14_FINAL = '''# =============================================================================
# BLOCK 14: FINAL RESULTS DUMP (HalluShift-style JSON)
# =============================================================================
results = {
    "schema_version": "2.0",
    "timestamp_utc":  datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "host":           ("kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ
                       else ("colab" if "COLAB_GPU" in os.environ else "local")),
    "model": {
        "name": MODEL_NAME,
        "tag":  MODEL_TAG,
        "hidden_dim": int(model.config.hidden_size),
        "n_layers":   int(model.config.num_hidden_layers),
    },
    "config": {
        "seed": SEED, "target_samples_per_class": TARGET_SAMPLES,
        "topk": TOPK_FIRST_TOKEN, "windows": WINDOWS, "use_4bit": USE_4BIT,
        "dtype": str(DTYPE),
    },
    "features": [
        "canonical_mind (last token, last layer)",
        "D_mean (mean cosine distance between adjacent layers)",
        "V_last (L2 cross-layer variance of last-token activations)",
        "H_mean (mean Shannon entropy of per-step token distribution)",
    ],
    "feature_dim": int(input_dim),
    "scaler": {
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": ["D_mean", "V_last", "H_mean"],
    },
    "wikipedia_eval": wiki_metrics,
    "multitask": multitask,
}
out_path = f"{MODEL_TAG}_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\\n\\u2713 Wrote {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

# Print a single-row HalluShift-style summary table
print("\\n" + "=" * 90)
print(f"FINAL TABLE — {MODEL_TAG}")
print("=" * 90)
print(f"{'Dataset':22s} {'N':>6s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUROC':>7s} {'Brier':>7s}")
print("-" * 90)
m = wiki_metrics
print(f"{'wikipedia (mind)':22s} {m['n']:6d} {m['accuracy']:7.3f} "
      f"{m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
      f"{m['auc_roc']:7.3f} {m['brier']:7.3f}")
for k, v in multitask.items():
    if v.get("n", 0) == 0: continue
    print(f"{k:22s} {v['n']:6d} {v['accuracy']:7.3f} "
          f"{v['precision']:7.3f} {v['recall']:7.3f} {v['f1']:7.3f} "
          f"{v['auc_roc']:7.3f} {v['brier']:7.3f}")
print("=" * 90)
print(f"\\nPaste {out_path} back to the assistant for verification.")
'''

# -----------------------------------------------------------------------------
# Build the notebook structure (list of cells)
# -----------------------------------------------------------------------------
def build_cells():
    return [
        cell_md(CELL_INTRO),
        cell_code(CELL_0_INSTALL),
        cell_code(CELL_1_LOAD),
        cell_code(CELL_2_ENT),
        cell_code(CELL_3_TOK),
        cell_code(CELL_4_FEATURES),
        cell_code(CELL_5_SANITY),
        cell_code(CELL_6_DATAGEN),
        cell_code(CELL_7_SPLIT),
        cell_code(CELL_8_MLP),
        cell_code(CELL_9_TRAIN),
        cell_code(CELL_10_WIKI_EVAL),
        cell_code(CELL_11_DOWNLOAD_DATASETS),
        cell_code(CELL_12_SCORER),
        cell_code(CELL_13_MULTI_EVAL),
        cell_code(CELL_14_FINAL),
    ]


def substitute(cells, mapping):
    """Apply per-model substitutions to every cell's source."""
    out = []
    for c in cells:
        new = json.loads(json.dumps(c))  # deep copy
        src = "".join(new["source"])
        for k, v in mapping.items():
            src = src.replace(k, str(v))
        new["source"] = src.splitlines(keepends=True)
        out.append(new)
    return out


def write_notebook(path, cells):
    nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path}")


def main():
    base = build_cells()
    for tag, model_name, env, n_