# ntllm: Name That LLM (Kaggle 2026)

> **How it's built: 🤖 vibe coded.** AI coding agents (mostly Claude Code) write the code. I pick the goal, make the design calls, and steer and test the work.
>
> **When:** 2026-11 → 2027-01 · Companion projects: [journey](https://github.com/dominictedrow/journey) (`vibe/knees`, `vibe/envenda`, `vibe/kaggriculture`)

An entry for the [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) Kaggle competition: given a passage of AI-generated text, predict which of 18 LLMs wrote it. Scored on balanced accuracy.

Planned models: gzip + kNN baseline, TF-IDF with Naive Bayes and SVM, a fine-tuned encoder, and a blend on top.

## Where things are
- `CLAUDE.md`: project rules and competition facts (read first).
- `docs/DESIGN.md`: the locked design (v2). `docs/BUILD_BRIEF.md`: build order and ablation list. `docs/DESIGN_LOCK.md`: lock record.
- `ntllm/`: library code (data, features, models, kaggle). `scripts/`: entry points (`python scripts/train.py ...`). `experiments/`: experiment comparison scripts.
- `data/`: competition files (gitignored).

## Setup
```bash
conda env create -f environment.yml   # or: conda create -n ntllm python=3.12 && pip install -r requirements.txt
conda activate ntllm
kaggle competitions download -c name-that-llm -p data/raw && unzip data/raw/*.zip -d data/raw
```
On the Windows RTX 5070 box, install the CUDA torch wheel first (see `requirements.txt`).
