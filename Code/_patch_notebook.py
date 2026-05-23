"""
One-shot patcher: rewrite project.ipynb to fix MIND-paper deviations
and swap Llama-3.2-1B / Mistral-7B → Qwen2.5-0.5B.

Run from: /sessions/festive-tender-galileo/mnt/Dessertation/Code/
"""
import json
import copy
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "project.ipynb"

with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# ----------------------------------------------------------------------
# helper to build a code cell
def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


# ----------------------------------------------------------------------
# NEW CELL 0: pip installs (Qwen-compatible)
cell0 = code(r"""# =============================================================================
# BLOCK 0: PIP INSTALLS (Qwen 2.5 0.5B + MIND pipeline deps)
# =============================================================================
# Notes:
#  * bitsandbytes kept for users who still want 4-bit quant, but for a 0.5B model
#    we use bf16 by default — quantisation overhead exceeds the memory saving.
#  * tokenizers >= 0.19 is needed for Qwen2.5.
!pip install -q -U "transformers>=4.45" "tokenizers>=0.19" "accelerate>=0.30"
!pip install -q -U datasets spacy nltk scikit-learn tqdm
!pip install -q -U bitsandbytes  # optional, only used if USE_4BIT=True
!python -m spacy download en_core_web_sm
""")

# NEW CELL 1: model loading
cell1 = code(r'''# =============================================================================
# BLOCK 1: IMPORTS & MODEL LOADING  (Qwen 2.5 0.5B, MIND-style probing)
# =============================================================================
import os, gc, json, random
import numpy as np
import torch
import spacy, nltk
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

print("=" * 80)
print("BLOCK 1: MODEL LOADING")
print("=" * 80)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Qwen 2.5 0.5B is small enough for any free Colab/Kaggle GPU and follows the
# MIND base-model recipe. We use the BASE (non-Instruct) variant because MIND
# probes raw next-token prediction, not chat behaviour.
MODEL_NAME = "Qwen/Qwen2.5-0.5B"           # ← swapped from Llama-3.2-1B / Mistral-7B
SEED = 42
TOPK_FIRST_TOKEN = 4                       # MIND default
WINDOWS = 16                               # search window for next_token
TARGET_SAMPLES = 1000                      # per class
USE_4BIT = False                           # 4-bit hurts speed at 0.5B; keep available
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Output-file prefix follows the model so we never get the llama32_* drift again.
MODEL_TAG = MODEL_NAME.split("/")[-1].replace(".", "").lower()   # → "qwen25-05b"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# Tokeniser
# -----------------------------------------------------------------------------
print(f"\nLoading {MODEL_NAME} (4-bit={USE_4BIT}, dtype={DTYPE}) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
load_kwargs = dict(
    trust_remote_code=True,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)
if USE_4BIT and torch.cuda.is_available():
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_quant_type="nf4",
    )

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()
for p in model.parameters():
    p.requires_grad = False  # we only do inference here; saves memory

# -----------------------------------------------------------------------------
# spaCy
# -----------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    print(f"✓ Loaded on {torch.cuda.get_device_name(0)}")
    print(f"✓ Free VRAM    : {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
print(f"✓ Hidden dim   : {model.config.hidden_size}")
print(f"✓ #layers (N)  : {model.config.num_hidden_layers}")
print(f"✓ Model tag    : {MODEL_TAG}")
print("✓ spaCy loaded : en_core_web_sm")
''')

# NEW CELL 2: entity extraction (mostly unchanged, lowercase title check)
cell2 = code(r'''# =============================================================================
# BLOCK 2: ENTITY EXTRACTION  (unchanged from MIND open-source code)
# =============================================================================
print("\n" + "=" * 80)
print("BLOCK 2: ENTITY EXTRACTION FUNCTIONS")
print("=" * 80)


def delete_substrings(lst):
    """Remove strings that are substrings of others."""
    substrings = []
    lst = list(set(lst))
    for s in lst:
        if any(s in o for o in lst if o != s):
            substrings.append(s)
    for s in substrings:
        lst.remove(s)
    return lst


def find_boundaries(text, words):
    """Expand entity matches outward to whitespace boundaries."""
    boundaries = []
    for word in words:
        ntext = text
        while True:
            start = ntext.find(word)
            if start == -1:
                break
            end = start + len(word) - 1
            while start > 0 and ntext[start - 1] != " ":
                start -= 1
            while end < len(ntext) - 1 and ntext[end + 1] != " ":
                end += 1
            boundaries.append("".join(ntext[i] for i in range(start, end + 1)))
            ntext = ntext[end + 1 :]
    return boundaries


def get_entities(text):
    """Return every (entity, char_index) occurrence in `text`."""
    entities_ = list({str(e) for e in nlp(text).ents})
    entities_ = find_boundaries(text, entities_)
    entities = delete_substrings(entities_)
    all_entities = []
    for i in range(len(text)):
        for e in entities:
            if text[i:].startswith(e):
                all_entities.append((e, i))
    return all_entities


print("✓ Entity extraction functions defined")
''')

# NEW CELL 3: token search (now takes the model/tokenizer via globals; identical logic)
cell3 = code(r'''# =============================================================================
# BLOCK 3: TOKEN-LEVEL GENERATION  (MIND Section 3.1 — "Continue Generation")
# =============================================================================
print("\n" + "=" * 80)
print("BLOCK 3: TOKEN-LEVEL GENERATION")
print("=" * 80)


def find_first_and_next_token(text, e, idx, input_id, prompt=""):
    """
    Recover [first_entity_token, post-entity_anchor_token, entity_len, suffix_ids]
    by inserting '@' immediately after the entity and re-tokenising.
    """
    new_text = f"{text[:idx].strip()} {text[idx:].replace(e, e + ' @', 1).strip()}"
    new_input_id = tokenizer(
        prompt + new_text.strip(), return_tensors="pt"
    )["input_ids"].tolist()[0]

    # Verify the prefix matches token-for-token (sanity: tokeniser shouldn't shift)
    for i in range(len(input_id[0])):
        if input_id[0][i] != new_input_id[i]:
            return []

    first_token = new_input_id[len(input_id[0])]

    # Find the '@' marker (may tokenise as " @" or "@")
    at_token_candidates = tokenizer("@", add_special_tokens=False)["input_ids"]
    at_token_candidates += tokenizer(" @", add_special_tokens=False)["input_ids"]

    at_position = None
    for at_tok in at_token_candidates:
        try:
            at_position = new_input_id.index(at_tok, len(input_id[0]))
            break
        except ValueError:
            continue

    if at_position is None or at_position >= len(new_input_id) - 1:
        return []

    next_token = new_input_id[at_position + 1]
    entity_len = at_position - len(input_id[0])
    last_id = new_input_id[at_position + 1 :]
    return [first_token, next_token, entity_len, last_id]


print("✓ Token detection function defined")
''')

# NEW CELL 4: get_hd — adds canonical MIND embedding (last_token, last_layer)
cell4 = code(r'''# =============================================================================
# BLOCK 4: EMBEDDING EXTRACTION  (MIND-canonical + 3 ablation views)
# =============================================================================
# MIND paper Sec 3.2.2 (verbatim):
#     "We choose the contextualized embedding of the last token of last
#      Transformer layer as the input of the MIND classifier."
# i.e.  H = hidden_states[-1][0][-1]   ←  what the trained MLP must consume.
# We additionally keep three "ablation" views from the original notebook so the
# user can compare in Section 6 of the report.
print("\n" + "=" * 80)
print("BLOCK 4: EMBEDDING EXTRACTION")
print("=" * 80)


@torch.no_grad()
def get_hd(text, start_at: int = 2):
    """
    Returns
    -------
    canonical_mind   : list[float]  — H_N^n  (last token, last layer)  ← MIND default
    hds_all_layers   : list[float]  — mean over layers of the last-token embedding
    mean1_first_lyr  : list[float]  — mean over tokens of the first layer
    mean2_last_lyr   : list[float]  — mean over tokens of the last  layer
    """
    enc = tokenizer(text.strip(), return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                  # tuple length = N+1 (incl. embedding layer)
    last_layer = hs[-1][0]                  # [seq_len, hidden_dim]

    # --- canonical MIND embedding ---------------------------------------------
    canonical_mind = last_layer[-1].float().cpu().tolist()

    # --- ablation views (kept for reproducibility with the original notebook) -
    # mean of last-token across all layers (skip embedding layer at index 0)
    last_token_per_layer = torch.stack([h[0][-1] for h in hs[1:]], dim=0)
    hds_all_layers = last_token_per_layer.mean(dim=0).float().cpu().tolist()
    mean1_first_lyr = hs[1][0][start_at - 1:].mean(dim=0).float().cpu().tolist()
    mean2_last_lyr = last_layer[start_at - 1:].mean(dim=0).float().cpu().tolist()
    return canonical_mind, hds_all_layers, mean1_first_lyr, mean2_last_lyr


print("✓ get_hd() returns 4 views; index 0 is the MIND-canonical embedding")
print("  [0] canonical_mind  ← used by the classifier (MIND Eq. 1)")
print("  [1] hds_all_layers  — ablation: avg last-token across layers")
print("  [2] mean1_first_lyr — ablation: mean of first layer tokens")
print("  [3] mean2_last_lyr  — ablation: mean of last  layer tokens")
''')

# NEW CELL 5: sanity test on Einstein paragraph
cell5 = code(r'''# =============================================================================
# BLOCK 5: SANITY TEST ON A SAMPLE PARAGRAPH
# =============================================================================
print("\n" + "=" * 80)
print("BLOCK 5: TESTING ON SAMPLE TEXT")
print("=" * 80)

sample_text = (
    "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the "
    "German Empire, on 14 March 1879. His parents were Hermann Einstein, a "
    "salesman and engineer, and Pauline Koch."
)
print(f"\nSample: {sample_text[:100]}…")

entities = get_entities(sample_text)
print(f"✓ Entity occurrences: {len(entities)}")

title = "Albert Einstein"
valid = [(e, i) for e, i in entities if i != 0 and e.lower() not in title.lower()]
print(f"✓ Valid (post-first-sentence) entities: {len(valid)}")

if valid:
    entity, char_idx = valid[0]
    print(f"\n✓ Trying entity '{entity}' @ char {char_idx}")
    prompt = sample_text[:char_idx].strip()

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    input_id_list = input_ids.tolist()

    toks = find_first_and_next_token(sample_text, entity, char_idx, input_id_list)
    if toks:
        first_, next_, entity_len, last_id = toks
        print(f"✓ first_token={first_} next_token={next_} entity_len={entity_len}")

        # ----  MIND continuation: ONE generate() call instead of N --------------
        gen = model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=entity_len + WINDOWS,
            return_dict_in_generate=True, output_scores=True,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        # step-0 score = first-token probability
        top_first = torch.topk(gen.scores[0], k=TOPK_FIRST_TOKEN).indices[0].tolist()
        if first_ in top_first:
            print("→ Model already knows the entity; this article would be skipped.")
        else:
            # find earliest step at which next_token is in the top-K
            found_step = None
            for step, sc in enumerate(gen.scores):
                if next_ in torch.topk(sc, k=TOPK_FIRST_TOKEN).indices[0].tolist():
                    found_step = step
                    break
            if found_step is None:
                print("✗ next_token not found within window")
            else:
                print(f"✓ next_token reappears at step {found_step + 1}")
                seq = gen.sequences[0, : input_ids.shape[1] + found_step]
                hall_ids = seq.tolist() + last_id
                hall_text = tokenizer.decode(hall_ids, skip_special_tokens=True)
                print(f"\nHallucinated text: {hall_text[:150]}…")
                emb_hall, *_ = get_hd(hall_text)
                emb_orig, *_ = get_hd(sample_text)
                print(f"✓ canonical-MIND emb dim: hall={len(emb_hall)}, orig={len(emb_orig)}")
    else:
        print("✗ Token detection failed")

print("\n" + "=" * 80)
print("BLOCK 5 COMPLETE")
print("=" * 80)
''')

# NEW CELL 6: dataset generation — single generate call, fixed filter
cell6 = code(r'''# =============================================================================
# BLOCK 6: DATASET GENERATION  (MIND-style, efficient single-call decoding)
# =============================================================================
print("\n" + "=" * 80)
print("BLOCK 6: DATASET GENERATION")
print("=" * 80)


def generate_sample(text, entity, idx, title):
    """
    Returns a dict with `embedding` set to MIND canonical (last-token,last-layer)
    for both the hallucinated and the non-hallucinated case, plus the 3 ablation
    views, or None if the article does not yield a usable hallucination.

    Key efficiency change vs. original: a SINGLE model.generate() call with
    `max_new_tokens = entity_len + WINDOWS` instead of one call per token.
    """
    enc = tokenizer(text[:idx].strip(), return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    input_id_list = input_ids.tolist()

    toks = find_first_and_next_token(text, entity, idx, input_id_list)
    if not toks:
        return None
    first_, next_, entity_len, last_id = toks

    # One forward pass returns scores at every generated step
    gen = model.generate(
        input_ids, attention_mask=attn,
        max_new_tokens=entity_len + WINDOWS,
        return_dict_in_generate=True, output_scores=True,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )

    # Step 0: does the model already predict the gold first_token? → skip article
    if first_ in torch.topk(gen.scores[0], k=TOPK_FIRST_TOKEN).indices[0].tolist():
        return None

    # Find earliest step where next_token (post-entity anchor) re-appears
    found_step = None
    for step, sc in enumerate(gen.scores):
        if next_ in torch.topk(sc, k=TOPK_FIRST_TOKEN).indices[0].tolist():
            found_step = step
            break
    if found_step is None:
        return None

    new_seq = gen.sequences[0, : input_ids.shape[1] + found_step].tolist()
    new_entity_ids = new_seq[input_ids.shape[1]:]
    all_new_text_ids = input_id_list[0] + new_entity_ids + last_id
    hallucinated_text = tokenizer.decode(all_new_text_ids, skip_special_tokens=True)

    new_entity = tokenizer.decode(new_entity_ids, skip_special_tokens=True).strip().lower()

    # ------------------ STRICTER but FAIR filter --------------------------------
    # Reject only if (a) empty, (b) literally contains the original entity, or
    # (c) the FULL new entity (not its individual common words) appears verbatim
    # in the original text — i.e. the "hallucinated" entity is just a copy.
    if (
        not new_entity
        or entity.lower() in new_entity
        or new_entity in text.lower()        # whole-phrase containment
    ):
        return None

    # -------- Embeddings (4-tuple, canonical first) -----------------------------
    can_h, hds_h, m1_h, m2_h = get_hd(hallucinated_text)
    can_o, hds_o, m1_o, m2_o = get_hd(text)

    return {
        "text_hall": hallucinated_text,  "entity_hall": new_entity,
        "emb_hall":  can_h,
        "hds_hall":  hds_h, "mean1_hall": m1_h, "mean2_hall": m2_h,
        "label_hall": 1,
        "text_orig": text, "entity_orig": entity,
        "emb_orig":  can_o,
        "hds_orig":  hds_o, "mean1_orig": m1_o, "mean2_orig": m2_o,
        "label_orig": 0,
        "title": title,
    }


# ---------------------------------------------------------------------------
# Streaming loop over Wikipedia
# ---------------------------------------------------------------------------
print(f"\nLoading Wikipedia (streaming) …")
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

dataset_hall, dataset_non_hall = [], []
processed = 0
print(f"\nGenerating up to {TARGET_SAMPLES*2} samples ({TARGET_SAMPLES}/class)\n")

pbar = tqdm(total=TARGET_SAMPLES * 2, desc="Samples")
for row in wiki:
    if len(dataset_hall) >= TARGET_SAMPLES and len(dataset_non_hall) >= TARGET_SAMPLES:
        break
    processed += 1
    try:
        sents = sent_tokenize(row["text"])
        if len(sents) < 2:
            continue
        text = " ".join(sents[:2])
        title = row.get("title", "")

        ents = get_entities(text)
        if not ents:
            continue
        seen, ents_filt = set(), []
        for e, i in ents:
            if i == 0 or e.lower() in title.lower():
                continue
            if i not in seen:
                seen.add(i)
                ents_filt.append((e, i))
        if not ents_filt:
            continue

        entity, char_idx = random.choice(ents_filt)
        result = generate_sample(text, entity, char_idx, title)
        if result is None:
            continue

        if len(dataset_hall) < TARGET_SAMPLES:
            dataset_hall.append({
                "label":     result["label_hall"],
                "text":      result["text_hall"],
                "entity":    result["entity_hall"],
                "embedding": result["emb_hall"],   # MIND canonical
                "ablation": {
                    "hds":   result["hds_hall"],
                    "mean1": result["mean1_hall"],
                    "mean2": result["mean2_hall"],
                },
                "title":     result["title"],
            })
            pbar.update(1)
        if len(dataset_non_hall) < TARGET_SAMPLES:
            dataset_non_hall.append({
                "label":     result["label_orig"],
                "text":      result["text_orig"],
                "entity":    result["entity_orig"],
                "embedding": result["emb_orig"],   # MIND canonical
                "ablation": {
                    "hds":   result["hds_orig"],
                    "mean1": result["mean1_orig"],
                    "mean2": result["mean2_orig"],
                },
                "title":     result["title"],
            })
            pbar.update(1)

        pbar.set_postfix(H1=len(dataset_hall), H0=len(dataset_non_hall), Proc=processed)

        if (len(dataset_hall) + len(dataset_non_hall)) % 500 == 0:
            with open(f"{MODEL_TAG}_checkpoint.json", "w") as f:
                json.dump(dataset_hall + dataset_non_hall, f)
    except Exception as e:
        if processed % 1000 == 0:
            print(f"\n[Warn at {processed}]: {e}")
        continue
    if processed % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

pbar.close()
print(f"\n✓ Generation complete.")
print(f"  Hallucinated (label=1)     : {len(dataset_hall)}")
print(f"  Non-hallucinated (label=0) : {len(dataset_non_hall)}")
print(f"  Total                      : {len(dataset_hall) + len(dataset_non_hall)}")
''')

# NEW CELL 7: split & save with MODEL_TAG-based file names
cell7 = code(r'''# =============================================================================
# BLOCK 7: TRAIN / TEST SPLIT AND SAVE
# =============================================================================
print("\n" + "=" * 80)
print("BLOCK 7: TRAIN/TEST SPLIT")
print("=" * 80)

dataset = dataset_hall + dataset_non_hall
random.shuffle(dataset)

split_idx = int(0.8 * len(dataset))
train_data, test_data = dataset[:split_idx], dataset[split_idx:]

TRAIN_PATH = f"{MODEL_TAG}_train.json"
TEST_PATH  = f"{MODEL_TAG}_test.json"

with open(TRAIN_PATH, "w") as f:
    json.dump(train_data, f, indent=2)
with open(TEST_PATH, "w") as f:
    json.dump(test_data, f, indent=2)

print(f"\n✓ Saved {TRAIN_PATH}  ({len(train_data)} samples)")
print(f"✓ Saved {TEST_PATH}   ({len(test_data)} samples)")

for name, d in [("Train", train_data), ("Test", test_data)]:
    h0 = sum(1 for x in d if x["label"] == 0)
    h1 = sum(1 for x in d if x["label"] == 1)
    print(f"  {name:5s}: H=0 {h0:4d}, H=1 {h1:4d}")

ex = dataset[0]
print(f"\n✓ Example sample:  label={ex['label']}  entity='{ex['entity']}'  "
      f"emb-dim={len(ex['embedding'])}")
print(f"  Text: {ex['text'][:80]}…")

# Free GPU before training the tiny MLP
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("BLOCKS 6-7 COMPLETE — DATA READY FOR TRAINING")
print("=" * 80)
''')

# NEW CELL 8: MLP classifier (matches MIND Eq. 1 exactly — keep)
cell8 = code(r'''# =============================================================================
# BLOCK 8: MLP HALLUCINATION CLASSIFIER  (MIND Eq. 1 + Eq. 2)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)

print("=" * 80)
print("BLOCK 8: MLP CLASSIFIER DEFINITION")
print("=" * 80)


class MINDClassifier(nn.Module):
    """
    Multilayer Perceptron, MIND paper Eq. 1:  P = MLP(ReLU(W · H + b))
    Input  : H  ∈ R^{1 × n}       (last token, last layer)
    Output : logits ∈ R^{1 × 2}   (binary: non-hall / hall)
    Loss   : BCE / CrossEntropy   (Eq. 2)
    """

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, H):
        return self.layers(H)


class MINDDataset(Dataset):
    def __init__(self, data):
        self.embeddings = torch.FloatTensor([d["embedding"] for d in data])
        self.labels = torch.LongTensor([d["label"] for d in data])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


print("✓ MINDClassifier: 5 linear layers (in → 512 → 256 → 128 → 64 → 2)")
print("✓ Activation : ReLU,  Dropout 0.2 after layer 1")
print("✓ Loss       : CrossEntropyLoss (equivalent to BCE for 2-class softmax)")
''')

# NEW CELL 9: training loop
cell9 = code(r'''# =============================================================================
# BLOCK 9: LOAD DATA AND TRAIN
# =============================================================================
print("\n" + "=" * 80)
print("BLOCK 9: TRAINING")
print("=" * 80)

with open(TRAIN_PATH) as f:
    train_data = json.load(f)
with open(TEST_PATH) as f:
    test_data = json.load(f)
print(f"✓ Train : {len(train_data)}   Test : {len(test_data)}")

BATCH_SIZE = 32
train_loader = DataLoader(MINDDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(MINDDataset(test_data),  batch_size=BATCH_SIZE)

input_dim = len(train_data[0]["embedding"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = MINDClassifier(input_dim).to(device)

print(f"\n[Step 2] Model setup:")
print(f"  Input dim   : {input_dim}")
print(f"  Parameters  : {sum(p.numel() for p in mlp.parameters()):,}")
print(f"  Device      : {device}")

EPOCHS, LR, WD = 10, 5e-4, 1e-5
optim = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=WD)
loss_fn = nn.CrossEntropyLoss()

print(f"\n[Step 3] Training {EPOCHS} epochs (lr={LR}, weight_decay={WD})\n")
print("=" * 80)

BEST_PATH = f"{MODEL_TAG}_mind_best.pth"
best_acc = 0.0

for epoch in range(EPOCHS):
    mlp.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        out = mlp(x)
        l = loss_fn(out, y)
        l.backward()
        optim.step()
        tr_loss += l.item()
        tr_correct += (out.argmax(1) == y).sum().item()
        tr_total += y.size(0)
    tr_loss /= max(1, len(train_loader))
    tr_acc = tr_correct / max(1, tr_total)

    mlp.eval()
    te_loss, te_correct, te_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = mlp(x)
            l = loss_fn(out, y)
            te_loss += l.item()
            te_correct += (out.argmax(1) == y).sum().item()
            te_total += y.size(0)
    te_loss /= max(1, len(test_loader))
    te_acc = te_correct / max(1, te_total)

    star = ""
    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(mlp.state_dict(), BEST_PATH)
        star = " ★"
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"test loss {te_loss:.4f} acc {te_acc:.4f}{star}")

print("=" * 80)
print(f"✓ Training complete. Best test acc = {best_acc:.4f}  → {BEST_PATH}")
''')

# NEW CELL 10: comprehensive evaluation
cell10 = code(r'''# =============================================================================
# BLOCK 10: COMPREHENSIVE EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("BLOCK 10: COMPREHENSIVE EVALUATION")
print("=" * 80)

mlp.load_state_dict(torch.load(BEST_PATH))
mlp.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = mlp(x)
        probs = F.softmax(out, dim=1)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

import numpy as np
all_preds = np.array(all_preds);  all_labels = np.array(all_labels);  all_probs = np.array(all_probs)

acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)
print(f"{'Metric':<15}{'Value':<10}")
print("-" * 25)
print(f"{'Accuracy':<15}{acc:.4f}")
print(f"{'Precision':<15}{prec:.4f}")
print(f"{'Recall':<15}{rec:.4f}")
print(f"{'F1':<15}{f1:.4f}")
print(f"{'AUC-ROC':<15}{auc:.4f}")

print("\nConfusion matrix")
print(f"             Pred-NonH  Pred-Hall")
print(f"Actual-NonH   {cm[0,0]:6d}     {cm[0,1]:6d}")
print(f"Actual-Hall   {cm[1,0]:6d}     {cm[1,1]:6d}")

print("\nClassification report")
print(classification_report(
    all_labels, all_preds,
    target_names=["Non-hallucination (0)", "Hallucination (1)"],
    digits=4,
))

# Random qualitative examples
print("=" * 80)
print("QUALITATIVE EXAMPLES")
print("=" * 80)
n_examples = min(20, len(test_data))
for i, idx in enumerate(random.sample(range(len(test_data)), n_examples), 1):
    s = test_data[idx]
    emb = torch.FloatTensor(s["embedding"]).unsqueeze(0).to(device)
    with torch.no_grad():
        out = mlp(emb)
        p_hall = F.softmax(out, dim=1)[0, 1].item()
        pred = int(out.argmax(1).item())
    flag = "✓" if pred == s["label"] else "✗"
    print(f"[{i:2d}] {flag} true={s['label']} pred={pred} P(hall)={p_hall:.2%} "
          f"entity='{s['entity']}'")
    print(f"     {s['text'][:120]}…\n")

print("=" * 80)
print("ALL BLOCKS COMPLETE")
print("=" * 80)
print(f"Model        : {MODEL_NAME}")
print(f"Train + test : {len(train_data) + len(test_data)} samples")
print(f"Accuracy     : {acc:.4f}")
print(f"AUC-ROC      : {auc:.4f}")
print(f"F1           : {f1:.4f}")
print(f"\nArtifacts:")
print(f"  - {TRAIN_PATH}")
print(f"  - {TEST_PATH}")
print(f"  - {BEST_PATH}")
print("=" * 80)
''')

# Optional doc cell at top: keep, but add a short markdown banner about changes
banner = md(r"""# `project.ipynb` — MIND-style hallucination detection (Qwen 2.5 0.5B)

**Pipeline:** Wikipedia → entity-substitution probe → hidden-state extraction → MLP classifier.

**Model:** `Qwen/Qwen2.5-0.5B` (base, bf16). MIND probes raw next-token prediction, so we use the base
variant rather than the Instruct one. The 4-bit quantisation step from the earlier Llama / Mistral
configuration was removed — for a 0.5B model the dequant overhead exceeds the memory saving.

**Embedding (canonical MIND, Sec. 3.2.2):** `H = hidden_states[-1][0][-1]` — the contextualised embedding
of the **last token at the last Transformer layer**. The dataset stores this as `embedding` and the
classifier consumes it directly. Three ablation views (`hds`, `mean1`, `mean2`) are kept under `ablation/`
for the report (these reproduce other rows of MIND's Table in Sec. 3.2.1).

**File naming:** all artefacts are prefixed with `MODEL_TAG`, e.g. `qwen25-05b_train.json`,
`qwen25-05b_test.json`, `qwen25-05b_mind_best.pth`. Switching `MODEL_NAME` automatically renames outputs.

**Reproducibility:** `SEED = 42` is propagated to Python `random`, NumPy and PyTorch.
""")

# ----------------------------------------------------------------------
# Now stitch everything back into the notebook
new_cells = [banner, cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10]
nb["cells"] = new_cells

# Preserve metadata block; only set kernel/lang if not present
nb.setdefault("metadata", {})
nb["metadata"].setdefault("language_info", {"name": "python"})
nb.setdefault("nbformat", 4)
nb.setdefault("nbformat_minor", 5)

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {NB_PATH} with {len(new_cells)} cells")
