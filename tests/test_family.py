import numpy as np
import pandas as pd

from ntllm.data.load import LABELS
from ntllm.features.family import FAMILIES, FAMILY, GzipFamily, add_prefix


def test_every_label_has_a_family():
    assert set(FAMILY) == set(LABELS) and set(FAMILY.values()) == set(FAMILIES)
    assert FAMILY["gpt-6-astra"] == "gpt" and FAMILY["glm-5.3-flash"] == "glm"


def test_leave_prompt_out_ignores_prompt_mates():
    # Each prompt's two rows are near-identical, so without masking a row would match its
    # prompt-mate. With masking it must fall back to the other prompts' rows.
    rows = []
    for p, topic in enumerate(["lava flows from the volcano", "quantum spin states of electrons",
                               "moss grows on the forest floor"]):
        rows.append({"id": f"a{p}", "prompt_id": f"p{p}", "split": "train",
                     "text": f"Certainly! Let us delve into it. {topic}. " * 3, "model": "claude-opus-5"})
        rows.append({"id": f"b{p}", "prompt_id": f"p{p}", "split": "train",
                     "text": f"Here is the answer, plainly. {topic}. " * 3, "model": "gpt-5.5"})
    gf = GzipFamily(k=1, n_jobs=1).fit(pd.DataFrame(rows))
    fam = gf.predict_fit_rows()
    assert list(fam) == ["claude", "gpt"] * 3   # style match across prompts, not the prompt-mate


def test_prefix_format_is_stable():
    # The Kaggle kernel inlines this function; training and inference must agree exactly.
    assert add_prefix(["hi"], ["gpt"]) == ["family: gpt | hi"]
