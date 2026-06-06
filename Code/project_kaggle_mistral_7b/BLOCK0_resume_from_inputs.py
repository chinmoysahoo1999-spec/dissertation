# =============================================================================
# BLOCK 0  (RESUME / OFFLINE STAGING)  --  baselines_sota  (Mistral-7B-v0.1)
# Paste as the VERY FIRST code cell. Copies your uploaded Kaggle inputs into the
# working dir BY FILENAME so the notebook RESUMES instead of regenerating:
#
#   *_dataset_full.json            -> Stage 1 loads it (03 has NO data-gen)
#   *_baseline_feature_cache.json  -> Stage 2 feature extraction SKIPS (no model)
#   *_best.pth (saplma/haloscope/hallushift/mind) -> trained probes (kept)
#   eval_*.parquet                 -> downstream eval loads LOCALLY (no internet)
#   eval_datasets*.zip             -> auto-unzipped if present
#
# Robust to ANY input-dataset name (jshggd, shjjjjjj, ...). No-op if input empty.
# =============================================================================
import os, glob, fnmatch, shutil, zipfile

WORKDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.makedirs(WORKDIR, exist_ok=True)

WANT_PATTERNS = [
    "*_dataset_full.json",
    "*_dataset_with_features.json",
    "*_features_NEW.json",
    "*_baseline_feature_cache.json",
    "*_datagen_checkpoint.json",
    "*_best.pth",                       # saplma / haloscope / hallushift / mind / variant_*
    "eval_*.parquet",
]

def _input_files():
    fs = []
    for root in sorted(glob.glob("/kaggle/input/*")):
        fs += glob.glob(os.path.join(root, "**", "*"), recursive=True)
    return [f for f in fs if os.path.isfile(f)]

src = _input_files()
print(f"[block0] scanning {len(src)} file(s) under /kaggle/input ...")

# auto-unzip an eval bundle if you uploaded the .zip instead of loose parquets
for f in src:
    if fnmatch.fnmatch(os.path.basename(f), "eval_datasets*.zip"):
        try:
            zipfile.ZipFile(f).extractall(WORKDIR)
            print(f"[block0] unzipped {os.path.basename(f)} -> {WORKDIR}")
        except Exception as e:
            print(f"[block0] unzip failed {f}: {e}")

def _stage(srcpath):
    b = os.path.basename(srcpath); dst = os.path.join(WORKDIR, b)
    if os.path.abspath(srcpath) == os.path.abspath(dst): return b, "in-place"
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(srcpath): return b, "exists"
    shutil.copy2(srcpath, dst); return b, "copied"

staged = {}
for f in src:
    b = os.path.basename(f)
    if any(fnmatch.fnmatch(b, p) for p in WANT_PATTERNS):
        name, how = _stage(f); staged[name] = how

print(f"[block0] staged {len(staged)} file(s) into {WORKDIR}:")
for b in sorted(staged):
    print(f"   {staged[b]:8s} {b:44s} {os.path.getsize(os.path.join(WORKDIR, b))/1e6:8.2f} MB")

# resume plan
def _has(suffix): return any(x.endswith(suffix) for x in os.listdir(WORKDIR))
n_parq = len(glob.glob(os.path.join(WORKDIR, "eval_*.parquet")))
n_pth  = len(glob.glob(os.path.join(WORKDIR, "*_best.pth")))
print("\n[block0] resume plan -- baselines_sota")
print(f"  Stage 1  dataset_full       : {'present -> loads'                       if _has('_dataset_full.json')           else 'MISSING -> Stage 1 errors'}")
print(f"  Stage 2  feature cache      : {'present -> SKIP extraction (no model!)' if _has('_baseline_feature_cache.json') else 'absent -> WILL RUN (needs model)'}")
print(f"  Probes SAPLMA/HaloScope/HalluShift/MIND : retrain from cache -- fast, no model, no internet ({n_pth} .pth staged)")
print(f"  EigenScore + Perplexity (Wiki held-out) : LOADS the model (sampling) -- required")
print(f"  Downstream eval datasets    : {n_parq}/10 local parquet(s) -> OFFLINE")
print(f"  Downstream multi-task eval  : LOADS the model to score prompts -- required")
print("[block0] done.\n")
