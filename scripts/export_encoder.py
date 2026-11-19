"""Package a trained encoder checkpoint as a Kaggle Dataset folder (Chunk 4).

    python scripts/export_encoder.py --checkpoint checkpoints/encoder/<run>/final --slug ntllm-encoder-v1

Writes checkpoints/kaggle_datasets/<slug>/ (weights + tokenizer + ntllm.json + dataset-metadata.json,
private). Then upload it (needs Dom's OK; first time `create`, later `version`):
    kaggle datasets create  -p checkpoints/kaggle_datasets/<slug>
    kaggle datasets version -p checkpoints/kaggle_datasets/<slug> -m "..."
and build the kernel: python scripts/build_kaggle_encoder.py --dataset <slug>
"""
import argparse
import json
import shutil
from pathlib import Path

from ntllm.paths import CHECKPOINTS, ROOT

KAGGLE_USER = "dominictedrow"
NEEDED = ["config.json", "model.safetensors", "ntllm.json", "tokenizer.json", "tokenizer_config.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a .../final (or .../best) folder")
    ap.add_argument("--slug", required=True, help="Kaggle dataset slug, e.g. ntllm-encoder-v1")
    a = ap.parse_args()

    src = Path(a.checkpoint)
    src = src if src.is_absolute() else ROOT / src
    missing = [f for f in NEEDED if not (src / f).exists()]
    assert not missing, f"{src} is missing {missing}"
    meta = json.loads((src / "ntllm.json").read_text())
    assert meta.get("trained_on") == "train split v1", "export only weights trained on the train split"

    out = CHECKPOINTS / "kaggle_datasets" / a.slug
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if not f.name.startswith("._"):
            shutil.copy2(f, out / f.name)
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": a.slug, "id": f"{KAGGLE_USER}/{a.slug}", "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }, indent=2))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"wrote {out} ({size:.0f} MB); family input: {meta.get('family')}; "
          f"backbone {meta.get('backbone')}")
    print(f"upload (needs OK): kaggle datasets create -p {out}")


if __name__ == "__main__":
    main()
