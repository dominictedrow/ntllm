# Name That LLM: Project Guide

Kaggle competition: given AI-generated text, predict which LLM wrote it. Owner: Dom.
Design is locked (v2, 2026-11-18). Read `docs/DESIGN.md` and `docs/BUILD_BRIEF.md` before building anything.

## Competition Lab rules
- If `docs/DESIGN_LOCK.md` does not exist, do not write pipeline, model, training, feature or inference code. EDA and data-inspection scripts are fine. Say: "Design isn't locked yet. Finish the defense in Competition Lab."
- Build only what `docs/DESIGN.md` describes. If a task needs a material change (new model family, validation scheme or representation), stop and tell Dom to run "kag change" first.
- Before starting any training run, ask Dom for a one-line hypothesis and a predicted CV change, and print them in the run log. After the run, print the actual change so he can log it with "kag result".
- If Dom asks what a block of code does, ask him what he thinks first.
- Build in the chunk order of `docs/BUILD_BRIEF.md`, one chunk at a time.

## Competition facts
- Slug: `name-that-llm` (https://www.kaggle.com/competitions/name-that-llm). Featured, $50,000. Final deadline **2027-01-14 23:59 UTC**. Dom has accepted the rules. [Kaggle CLI, 2026-11-18]
- Metric: balanced accuracy over 18 labels. Submission file: `submission.parquet` with columns `id, model`. [sample_submission.parquet]
- Submissions run as a Kaggle notebook, ≤9 h. Hidden test ~4,000 rows; public LB is a slice, private is ~60% and differs from public. [docs/BUILD_BRIEF.md, docs/DESIGN.md; re-check the Overview page]

## Data (verified 2026-11-18)
- `data/raw/train.csv`: 2,699 rows, columns `id, prompt_id, text, model`. **150 prompts, 18 labels**, ~150 rows each (`claude-fable-5` has 149). One row per (prompt, model).
- **No prompt text, only an anonymized `prompt_id`** (answers open check 1: synthetic-prompt overlap must be checked against the passages, not the prompts).
- `data/raw/test.csv`: `id, text` (stub; replaced at scoring).
- Labels: claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5, gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, glm-5.3, glm-5.3-flash, gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, kimi-k3, qwen3.8-max.

## Validation (from the design; binding)
- Three splits of train.csv, all grouped by `prompt_id`: train / val / local test. Saved once to disk and reused. Assert no prompt crosses splits.
- Every fitted thing (TF-IDF vocab, kNN index, encoder, meta-model) sees only the train split. Meta-model trains on val predictions, scored once on local test. Kaggle's test.csv is only for submissions.
- Synthetic data goes into the train split only.
- **Splits v1 (made 2026-11-18, Chunk 1):** `data/splits/v1/` (assignments.csv + manifest.json with sha256 of train.csv and the split file). 60/20/20 by prompt, seed 42: train 90 prompts / 1,620 rows, val 30 / 540, local test 30 / 539. Every split is label-balanced (90 / 30 / 29–30 per label) because every prompt has all 18 models (except p002, missing one).
- `fold` column: 4 prompt-grouped folds inside train (k-fold tuning of single models) and inside val (meta-model selection, so local test is touched once); local test has fold = -1.
- Load data only through `ntllm.data.splits.load_split("train"|"val"|"test", fold=None)`; it re-checks hashes and prompt separation. Score only through `ntllm.data.scoring.score/report/save`. Never regenerate v1; make v2 with `scripts/make_splits.py --version v2` (a design-level change).

## Repo layout
- `ntllm/` library code: `data/` (loading, splits, scoring), `features/`, `models/`, `kaggle/` (notebook + submission).
- `scripts/` simple entry points with flags (`python scripts/train.py --model svm`). `experiments/` one script per experiment that prints the comparison.
- `data/raw/` competition files, `data/splits/` saved splits, `checkpoints/`, `runs/`, `logs/` (all gitignored). Run results go to files Claude can read.

## Environment
- Conda env `ntllm` (Python 3.12.16, matches the Kaggle image), `~/miniconda3/envs/ntllm`. Pinned packages in `requirements.txt`; Kaggle CLI 2.2.5 (+`six`, a missing dep). Credentials in `~/.kaggle/kaggle.json`.
- Mac (dev): torch 2.15.0 with MPS, for smoke tests only. Windows box (training, 24/7): RTX 5070 12 GB, i5, 16 GB RAM; install the cu132 torch wheel there (cu128 stops at torch 2.11).
- Encoder (Chunk 4): load HF weights with `dtype=torch.float32`. transformers 5 keeps the checkpoint's stored dtype, deberta-v3 is stored fp16, and AdamW then NaNs every weight after one step. The Mac's MPS runs out of memory on deberta-v3-base, so smoke tests use `--device cpu`. Longest train passage is 351 deberta tokens (`max_len` 384).
- Repo lives on an external FAT32 drive: macOS writes `._*` AppleDouble files everywhere, including inside `.git/` (which breaks git with "non-monotonic index"). They are gitignored; before git operations run `find . -name '._*' -delete` (safe: git never creates `._` names).

## Git
- Remote: https://github.com/dominictedrow/ntllm (public). Never commit `data/`, weights, or `kaggle.json`. Follow Dom's global git checklist.
