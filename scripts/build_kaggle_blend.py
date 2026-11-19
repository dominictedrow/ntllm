"""Build the Kaggle kernel for the Chunk 5 blend: plain average of SVM + NB + encoder.

    python scripts/build_kaggle_blend.py --dataset ntllm-encoder-blend-v1 --mode submit
    python scripts/build_kaggle_blend.py --dataset ntllm-encoder-blend-v1 --mode timing  # dummy 4,000 rows
    python scripts/build_kaggle_blend.py --dataset ntllm-encoder-blend-v1 --mode check   # local-test rows

The finalist picked in scripts/blend.py (val CV 0.363, local test 0.367): each base model's
per-class scores are z-scored per row, averaged, argmax. TF-IDF + SVM (C=0.3) and TF-IDF +
Complement NB (alpha=10) are refit on the train split inside the kernel (seconds); encoder
weights come from the dataset (scripts/export_encoder.py). Inlines ntllm/features/tfidf.py,
ntllm/models/linear.py and ntllm/models/blend.py.

check mode predicts the local-test rows and writes check_predictions.csv (no labels, no score)
so the port can be compared with runs/blend/<ts>_test/predictions.csv pred_average.

Run it locally first:
    NTLLM_INPUT=<dir with train.csv, test.csv and the weights folder> NTLLM_OUTPUT=<out> \\
        python notebooks/kaggle_blend_submit/kernel.py
Push (needs Dom's OK): kaggle kernels push -p notebooks/kaggle_blend_<mode>
"""
import argparse
import json

from ntllm.data.load import LABELS
from ntllm.data.splits import load_split
from ntllm.paths import ROOT

KAGGLE_USER = "dominictedrow"
COMPETITION = "name-that-llm"

MAIN = r'''

# ---------------------------------------------------------------- kernel main
import glob
import json
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

INPUT = os.environ.get("NTLLM_INPUT", "/kaggle/input")
OUTPUT = os.environ.get("NTLLM_OUTPUT", "/kaggle/working")


def find(name):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", name), recursive=True))
    assert hits, f"{name} not found under {INPUT}"
    return hits[0]


@torch.no_grad()
def encoder_logits(model, tok, texts, device, max_len, batch_size=64):
    order = np.argsort([len(t) for t in texts])  # length-sorted batches: less padding
    out = np.empty((len(texts), len(LABELS)), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out[idx] = model(**enc).logits.float().cpu().numpy()
    return out


def main():
    t0 = time.time()
    model_dir = os.path.dirname(find("ntllm.json"))
    meta = json.load(open(os.path.join(model_dir, "ntllm.json")))
    assert meta["labels"] == LABELS, "label order differs from training"
    assert not meta.get("family"), "blend finalist uses the encoder without the gzip family prefix"

    full = pd.read_csv(find("train.csv"))
    train = full[full.prompt_id.isin(TRAIN_PROMPTS)].assign(split="train").reset_index(drop=True)
    assert train.prompt_id.nunique() == len(TRAIN_PROMPTS)
    if MODE == "timing":
        pool = full[~full.id.isin(train.id)]
        test = pool.sample(n=DUMMY_ROWS, replace=True, random_state=0)[["text"]].reset_index(drop=True)
        test.insert(0, "id", [f"dummy{i:05d}" for i in range(len(test))])
    elif MODE == "check":
        test = full[full.prompt_id.isin(TEST_PROMPTS)][["id", "text"]].reset_index(drop=True)
    else:
        test = pd.read_csv(find("test.csv"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"mode={MODE} device={device} train={len(train)} test={len(test)}", flush=True)

    feats = TfidfFeatures(**FEATURES).fit(train)
    Xtr, Xte = feats.transform(train.text), feats.transform(test.text)
    svm = make_model("svm", C=C).fit(Xtr, train.model)
    nb = make_model("nb", alpha=ALPHA, nb_kind=NB_KIND).fit(Xtr, train.model)
    t1 = time.time()
    print(f"tfidf + svm + nb fit {t1 - t0:.1f}s  vocab: {feats.stats_}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    s_enc = encoder_logits(enc, tok, test.text.tolist(), device, meta["max_len"])
    t2 = time.time()

    X = np.hstack([standardize(s) for s in (scores(svm, Xte), scores(nb, Xte), s_enc)])
    pred = average_predict(X)
    sub = pd.DataFrame({"id": test.id, "model": pred})
    assert sub.id.is_unique and set(sub.id) == set(test.id)
    assert set(sub.model) <= set(LABELS), set(sub.model) - set(LABELS)
    os.makedirs(OUTPUT, exist_ok=True)
    if MODE == "check":
        sub.to_csv(os.path.join(OUTPUT, "check_predictions.csv"), index=False)
        name = "check_predictions.csv"
    else:
        name = "submission.parquet" if MODE == "submit" else "timing_submission.parquet"
        sub.to_parquet(os.path.join(OUTPUT, name), index=False)

    per_row = (t2 - t1) / max(len(test), 1)
    print(f"encoder {t2 - t1:.1f}s ({per_row * 1000:.1f} ms/row)  total {time.time() - t0:.1f}s", flush=True)
    print(f"projected for 4,000 rows: {per_row * 4000 / 60:.1f} min + setup (limit 540 min)", flush=True)
    print(f"wrote {name}: {len(sub)} rows, {sub.model.nunique()} labels predicted", flush=True)


if __name__ == "__main__":
    main()
'''


def inline(path) -> str:
    """Module source minus its ntllm imports (the kernel defines LABELS itself)."""
    lines = (ROOT / path).read_text().splitlines()
    return "\n".join(l for l in lines if not l.startswith("from ntllm")) + "\n"


def build(mode: str, C: float, alpha: float, min_prompts: int, dummy_rows: int) -> str:
    train_prompts = sorted(load_split("train").prompt_id.unique())
    test_prompts = sorted(load_split("test").prompt_id.unique()) if mode == "check" else []
    # Same settings as scripts/train.py defaults, which produced the measured runs.
    features = dict(analyzer="both", word_ngram=[1, 2], char_ngram=[2, 5], min_df=2,
                    max_features=None, lowercase=False, min_prompts=min_prompts)
    head = (f"# AUTO-GENERATED by scripts/build_kaggle_blend.py; do not edit by hand.\n"
            f"# ntllm blend kernel (average of SVM + NB + encoder), mode={mode} (splits v1).\n\n"
            f"import numpy as np\nimport pandas as pd\n\nLABELS = {LABELS!r}\n\n")
    consts = (f"\n\n# ---------------------------------------------------------------- config\n"
              f"MODE = {mode!r}\nC = {C}\nALPHA = {alpha}\nNB_KIND = 'complement'\n"
              f"FEATURES = {features!r}\nDUMMY_ROWS = {dummy_rows}\n"
              f"TRAIN_PROMPTS = {train_prompts!r}\nTEST_PROMPTS = {test_prompts!r}\n")
    mods = ["ntllm/features/tfidf.py", "ntllm/models/linear.py", "ntllm/models/blend.py"]
    body = "".join(f"# ---------------------------------------------------------------- {m}\n{inline(m)}\n"
                   for m in mods)
    return head + body + consts + MAIN


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True, help="encoder weights dataset slug (scripts/export_encoder.py)")
    ap.add_argument("--mode", choices=["timing", "submit", "check"], default="submit")
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--min-prompts", type=int, default=3)
    ap.add_argument("--dummy-rows", type=int, default=4000)
    a = ap.parse_args()

    out = ROOT / "notebooks" / f"kaggle_blend_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(build(a.mode, a.C, a.alpha, a.min_prompts, a.dummy_rows))
    slug = f"ntllm-blend-{a.mode}"
    meta = {
        "id": f"{KAGGLE_USER}/{slug}", "title": slug, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{KAGGLE_USER}/{a.dataset}"], "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/kernel.py + kernel-metadata.json  (kaggle id {meta['id']}, dataset {a.dataset})")


if __name__ == "__main__":
    main()
