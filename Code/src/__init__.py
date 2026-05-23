"""
src/ — refactored MIND-style hallucination detection pipeline.

Modules
-------
config         project-wide constants (model name, hyperparams, paths, seed)
model_loader   load the 4-bit quantized LLM + spaCy NLP
entities       Block 2: entity extraction with spaCy
tokens         Block 3: token-level generation helpers
embeddings     Block 4: hidden-state extraction (get_hd)
dataset_gen    Block 6: build the hallucinated / non-hallucinated dataset from Wikipedia
classifier     Block 8: MINDClassifier (5-layer MLP) + MINDDataset
train          Block 9: training loop
evaluate       Block 10: metrics + qualitative examples

The original project.ipynb is preserved unchanged as the canonical reference.
notebook_refactored.ipynb is the thin notebook that drives this package.
"""

__version__ = "0.1.0"
