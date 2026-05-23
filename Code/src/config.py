"""
config.py — project-wide constants.

Edit values here instead of scattering them across notebooks.
Anything you'd want to tweak per experiment should live in this file.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Canonical model (decided 2026-05-23): Llama-3.2-1B — small enough for free
# Colab/Kaggle GPUs and matches the existing llama32_* artefact naming.
MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Trust remote model code (required for some checkpoints — Llama-3.2 is fine
# either way, but the flag keeps the loader portable).
TRUST_REMOTE_CODE = True

# ---------------------------------------------------------------------------
# Token-level generation (Block 3 / Block 5 / Block 6)
# ---------------------------------------------------------------------------
TOPK_FIRST_TOKEN = 4   # top-k threshold for "model already knows the entity"
WINDOWS = 16           # extra tokens beyond entity_len in which to find next_token

# ---------------------------------------------------------------------------
# Dataset generation (Block 6)
# ---------------------------------------------------------------------------
TARGET_SAMPLES = 1000      # samples per class (hallucinated + non-hallucinated)
CHECKPOINT_EVERY = 500     # write a partial dump every N total samples
SENTENCES_PER_DOC = 2      # how many leading sentences of each wiki article to use
CACHE_CLEAR_EVERY = 100    # call torch.cuda.empty_cache() every N articles

# ---------------------------------------------------------------------------
# Embedding extraction (Block 4)
# ---------------------------------------------------------------------------
# Start position for the "mean" embeddings. 2 matches the original notebook.
EMBED_START_AT = 2

# Which embedding goes into the dataset that the classifier sees.
# Options: "mean2" (last-layer mean — MIND default), "mean1" (first-layer mean),
# "hds" (avg of all layers' last token).
EMBEDDING_KEY = "mean2"

# ---------------------------------------------------------------------------
# Train / test split (Block 7)
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.8

# ---------------------------------------------------------------------------
# Classifier (Block 8) — MIND paper Eq. (1): P = MLP(ReLU(W · H + b))
# ---------------------------------------------------------------------------
HIDDEN_LAYERS = (512, 256, 128, 64)  # widths between input and output
DROPOUT_RATE = 0.2                   # applied after first hidden layer
NUM_CLASSES = 2                      # binary: hallucinated / not

# ---------------------------------------------------------------------------
# Training (Block 9)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# File paths (relative to Code/)
# ---------------------------------------------------------------------------
TRAIN_PATH = "llama32_train.json"
TEST_PATH = "llama32_test.json"
CHECKPOINT_PATH = "llama32_checkpoint.json"
BEST_MODEL_PATH = "llama32_mind_best.pth"

# ---------------------------------------------------------------------------
# Wikipedia dataset
# ---------------------------------------------------------------------------
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"
