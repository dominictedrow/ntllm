"""Load the raw competition files."""
import pandas as pd

from ntllm.paths import RAW

# The 18 labels, sorted. Fixed order so confusion matrices and model outputs line up.
LABELS = sorted([
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
])


def load_train() -> pd.DataFrame:
    """train.csv: id, prompt_id, text, model."""
    df = pd.read_csv(RAW / "train.csv")
    assert list(df.columns) == ["id", "prompt_id", "text", "model"], df.columns
    assert df["id"].is_unique, "duplicate ids in train.csv"
    unknown = set(df["model"]) - set(LABELS)
    assert not unknown, f"labels not in LABELS: {unknown}"
    return df


def load_test() -> pd.DataFrame:
    """Kaggle's test.csv: id, text. Only for submissions, never for local scoring."""
    return pd.read_csv(RAW / "test.csv")
