"""
model_loader.py — Block 1: load the quantized LLM and spaCy NLP.

Two functions are exposed:

    load_llm(model_name)  -> (tokenizer, model)
    load_nlp()            -> spaCy nlp pipeline

Both are imported lazily inside the functions so that:
  * unit tests that don't touch the LLM can import this module on CPU-only
    machines without pulling in torch / transformers / bitsandbytes;
  * IDEs don't choke if heavy deps are missing during development.
"""

from __future__ import annotations

import logging

from .config import MODEL_NAME, TRUST_REMOTE_CODE, SEED

log = logging.getLogger(__name__)


def load_llm(model_name: str = MODEL_NAME):
    """
    Load the causal LM with 4-bit NF4 quantization (Block 1 of project.ipynb).

    Returns
    -------
    tokenizer, model
        Both ready for inference (model.eval() already called).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    log.info("Loading %s with 4-bit NF4 quantization...", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=TRUST_REMOTE_CODE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.eval()

    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        log.info(
            "Loaded on %s | free VRAM: %.2f GB | hidden_size=%d",
            torch.cuda.get_device_name(0),
            free_mem,
            model.config.hidden_size,
        )

    return tokenizer, model


def load_nlp():
    """Load spaCy's en_core_web_sm pipeline and the required NLTK tokenizers."""
    import nltk
    import spacy

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    return spacy.load("en_core_web_sm")


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
