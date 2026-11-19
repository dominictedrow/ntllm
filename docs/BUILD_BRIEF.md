# Build brief: Name That LLM (design v2, locked 2026-11-18)

Hand these to Claude Code one chunk at a time, in order. Every training run needs a one-line hypothesis and a predicted change logged first (`kag exp ...`), and the result after (`kag result ...`).

## Chunk 1: Data and splits
- Builds: loading train.csv; three prompt-grouped splits (train / val / local test), saved once to disk and reused; scoring utility (balanced accuracy, per-label recall, confusion matrix).
- Implements: Validation Scheme, Leakage Risks.
- Works when: no prompt appears in more than one split (assert it); the split files load in every script.
- Enables: every experiment. Answer open check 1 (prompt text?) and 3 (prompts per split) here.

## Chunk 2: Gzip + kNN baseline and a Kaggle submission notebook
- Builds: gzip+kNN using only the train split; a Kaggle notebook that writes submission.parquet; a timing run on Kaggle with a dummy test set the size of the real one (~4,000 rows).
- Implements: Baseline Model Submission; Inference and compute budget.
- Works when: local-test balanced accuracy is printed; the Kaggle timing run finishes well under 9h.
- Enables: Experiment 1 (gzip column) and your first leaderboard entry.

## Chunk 3: TF-IDF + Naive Bayes + SVM
- Builds: TF-IDF fit on the train split only, plus the prompt-frequency topic-word filter; NB and SVM; ~5 tuning runs each.
- Implements: Features; Models.
- Works when: the vectorizer is never fit on val or local test (assert it); per-label recall is printed.
- Enables: Experiment 1 (NB/SVM columns); Experiment 3 (SVM + gzip vs SVM).

## Chunk 4: Encoder
- Builds: fine-tuning on the train split at home (RTX 5070); weights exported as a Kaggle Dataset; inference in the notebook.
- Implements: Models (encoder); Training Setup.
- Works when: notebook inference runs offline; local-test score is printed.
- Enables: Experiment 1 (encoder column); Experiment 2 (encoder with vs without gzip family input).

## Chunk 5: Blending
- Builds: base models trained on train predict val; meta-model trained on the val predictions; meta-model choices compared inside val (folds), then scored once on local test; plain averaging as the comparison.
- Implements: Addenda > Ensembling / Validation.
- Works when: local test is used once per finalist (log it); blend vs averaging is reported.
- Enables: the blend submission. Answer open check 3.

## Chunk 6: Synthetic data
- Builds: a small pilot first (run app outputs for each reachable label through the trained classifier, plus a style comparison); prompt-overlap filter (gzip similarity + threshold); generation; synthetic rows in the train split only; class weights for unreachable labels.
- Implements: Data.
- Works when: the pilot shows app outputs look like the competition's passages for that label; no flagged prompts remain.
- Enables: the with/without experiment on local test (real data only), with per-label recall. Answer open checks 1, 2, 5 and 6.

## Chunk 7: Final selection
- Builds: a table of each candidate's public score, local-test score, and approach.
- Implements: Leaderboard; Addenda > Final submissions.
- Works when: your "generalized" rule (open check 7) is written down before January.

## Ablation list (fill in your predicted change before each run)
| # | Hypothesis | Predicted change | Beats noise bar? |
|---|---|---|---|
| 1 | SVM beats NB, gzip and the encoder on local test | | |
| 2 | Gzip family input helps the encoder | | |
| 3 | SVM + gzip beats SVM alone | | |
| 4 | Topic-word filter helps TF-IDF models | | |
| 5 | Blend (meta-model) beats plain averaging | | |
| 6 | Synthetic data helps on real local test | | |
| 7 | Class weights protect unreachable labels' recall | | |
