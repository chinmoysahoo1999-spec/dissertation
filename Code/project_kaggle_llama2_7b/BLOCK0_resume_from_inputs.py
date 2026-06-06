# =============================================================================
# BLOCK 0  (RESUME / OFFLINE STAGING)  --  paste this as the VERY FIRST code cell
# -----------------------------------------------------------------------------
# Purpose: you already ran the pipeline once, downloaded its outputs, and re-
# uploaded them as Kaggle input datasets (any names, e.g. "datgagj", "dfdfsdfs").
# This cell copies those files into the working dir by FILENAME so the notebook
# RESUMES instead of regenerating:
#
#   *_dataset_full.json            -> Stage 2 (data-gen)      SKIPS
#   *_dataset_with_features.json   -> Stage 3 (F1..F10)       SKIPS
#   *_features_NEW.json            -> Stage 3.6 (F11..F16)    SKIPS
#   *_variant_*_best.pth           -> trained probes (kept; Stage 4 re-creates them deterministically)
#   eval_*.parquet                 -> Stage 5 loads them LOCALLY (no internet)
#   eval_datasets*.zip             -> auto-unzipped if present
#
# It is pattern-based, so it works for ANY model tag and ANY input-dataset name,
# and is a harmless no-op when /kaggle/input is empty (e.g. running locally).
#
# NOTE: Stage 6 (multi-task eval) STILL loads the base model to score eval
# prompts -- there is no cached substitute. Enable internet once for the model
# download (Llama-2 is gated -> set HF_TOKEN in Block 0/1), or attach the model
# weights as an input dataset. After the model is loaded the run is fully offline.
# =============================================================================
import os, glob, fnmatch, shutil, zipfile

WORKDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.makedirs(WORKDIR, exist_ok=True)

# filename patterns we are willing to pull in from the uploaded input datasets
WANT_PATTERNS = [
    "*_dataset_full.json",
    "*_dataset_with_features.json",
    "*_features_NEW.json",
    "*_datagen_checkpoint.json",
    "*_variant_*_best.pth",
    "eval_*.parquet",
]

def _all_input_files():
    files = []
    for root in sorted(glob.glob("/kaggle/input/*")):
        files += glob.glob(os.path.join(root, "**", "*"), recursive=True)
    return [f for f in files if os.path.isfile(f)]

src_files = _all_input_files()
print(f"[block0] scanning {len(src_files)} file(s) under /kaggle/input ...")

# 0a) auto-unzip any eval_datasets*.zip into the working dir
for f in src_files:
    if fnmatch.fnmatch(os.path.basename(f), "eval_datasets*.zip"):
        try:
            with zipfile.ZipFile(f) as zf:
                zf.extractall(WORKDIR)
            print(f"[block0] unzipped {os.path.basename(f)} -> {WORKDIR}")
        except Exception as e:
            print(f"[block0] could not unzip {f}: {e}")

# 0b) stage wanted files into the working dir (copy by basename, idempotent)
def _stage(src):
    b = os.path.basename(src)
    dst = os.path.join(WORKDIR, b)
    if os.path.abspath(src) == os.path.abspath(dst):
        return b, "in-place"
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return b, "exists"
    shutil.copy2(src, dst)
    return b, "copied"

staged = {}
for f in src_files:
    b = os.path.basename(f)
    if any(fnmatch.fnmatch(b, pat) for pat in WANT_PATTERNS):
        name, how = _stage(f)
        staged[name] = how

if staged:
    print(f"[block0] staged {len(staged)} file(s) into {WORKDIR}:")
    for b in sorted(staged):
        mb = os.path.getsize(os.path.join(WORKDIR, b)) / 1e6
        print(f"   {staged[b]:8s} {b:42s} {mb:8.2f} MB")
else:
    print("[block0] nothing matched -- inputs empty or already in place.")

# 0c) resume plan (what the rest of the notebook will now do)
def _ex(suffix):
    return any(b.endswith(suffix) for b in os.listdir(WORKDIR))

n_parq = len(glob.glob(os.path.join(WORKDIR, "eval_*.parquet")))
print("\n[block0] resume plan")
print(f"  Stage 2  data-gen        : {'SKIP (dataset_full present)'      if _ex('_dataset_full.json')          else 'WILL RUN (needs model + internet)'}")
print(f"  Stage 3  F1..F10         : {'SKIP (features present)'          if _ex('_dataset_with_features.json') else 'WILL RUN (needs model)'}")
print(f"  Stage 3.6 F11..F16       : {'SKIP (NEW features present)'      if _ex('_features_NEW.json')          else 'WILL RUN (needs model)'}")
print(f"  Stage 4  train 16 probes : runs from cached features (no model, no internet)")
print(f"  Stage 5  eval datasets   : {n_parq}/10 local parquet(s) -> OFFLINE" if n_parq else "  Stage 5  eval datasets   : 0 local -> needs internet")
print(f"  Stage 6  multi-task eval : LOADS the base model to score eval prompts (model required)")
print("[block0] done.\n")
