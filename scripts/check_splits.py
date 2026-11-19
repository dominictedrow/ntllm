"""Load every split, re-check prompt separation, and print label balance.

    python scripts/check_splits.py [--version v1]
"""
import argparse

import pandas as pd

from ntllm.data.load import LABELS
from ntllm.data.splits import DEFAULT_VERSION, SPLIT_NAMES, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_VERSION)
    a = ap.parse_args()

    parts = {n: load_split(n, a.version) for n in SPLIT_NAMES}
    prompts = {n: set(p["prompt_id"]) for n, p in parts.items()}
    for i, x in enumerate(SPLIT_NAMES):
        for y in SPLIT_NAMES[i + 1:]:
            shared = prompts[x] & prompts[y]
            assert not shared, f"{x} and {y} share prompts: {sorted(shared)}"
    print(f"splits {a.version}: no prompt shared between splits")

    counts = pd.DataFrame({n: p["model"].value_counts() for n, p in parts.items()}).reindex(LABELS)
    print(counts.fillna(0).astype(int).to_string())
    print("min/max rows per label:", {n: (int(counts[n].min()), int(counts[n].max()))
                                      for n in SPLIT_NAMES})


if __name__ == "__main__":
    main()
