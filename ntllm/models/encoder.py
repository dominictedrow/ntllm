"""Fine-tuned transformer encoder classifier (DESIGN.md Models; Chunk 4).

Default backbone: microsoft/deberta-v3-base (~184M params), the usual strong choice for
authorship / AI-text classification at a size that fits a 12 GB RTX 5070.
Every passage fits whole: the longest train passage is 351 DeBERTa tokens, max_len is 384.

Per epoch it logs what the design asks for: runtime, balanced accuracy, macro precision,
perplexity (exp of mean cross-entropy on the eval rows). Checkpoints are save_pretrained
folders (weights + tokenizer + label map), which is also what the Kaggle Dataset ships.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from ntllm.data.load import LABELS
from ntllm.data.scoring import score

L2I = {l: i for i, l in enumerate(LABELS)}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Rows(torch.utils.data.Dataset):
    def __init__(self, enc, labels=None):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


class EncoderClassifier:
    def __init__(self, backbone="microsoft/deberta-v3-base", max_len=384, lr=2e-5,
                 epochs=4, batch_size=16, grad_accum=1, warmup=0.1, weight_decay=0.01,
                 seed=42, device="auto", bf16=True, log=print):
        self.cfg = dict(backbone=backbone, max_len=max_len, lr=lr, epochs=epochs,
                        batch_size=batch_size, grad_accum=grad_accum, warmup=warmup,
                        weight_decay=weight_decay, seed=seed)
        self.device = pick_device(device)
        # bf16 autocast only on CUDA (the 5070); MPS/CPU smoke tests stay fp32.
        self.amp = bf16 and self.device.type == "cuda"
        self.log = log
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.tok = AutoTokenizer.from_pretrained(backbone)
        # dtype=float32 matters: transformers 5 loads the checkpoint's stored dtype by default,
        # and deberta-v3 is stored in fp16, where AdamW's eps=1e-8 underflows to 0 and every
        # weight turns NaN after one step. Master weights stay fp32; bf16 is autocast only.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I,
            dtype=torch.float32,
        ).to(self.device)
        assert next(self.model.parameters()).dtype == torch.float32

    # ------------------------------------------------------------------ data
    def _loader(self, texts, labels=None, shuffle=False):
        enc = self.tok(list(texts), truncation=True, max_length=self.cfg["max_len"])
        ys = None if labels is None else [L2I[l] for l in labels]
        g = torch.Generator().manual_seed(self.cfg["seed"])
        return DataLoader(_Rows(enc, ys), batch_size=self.cfg["batch_size"], shuffle=shuffle,
                          generator=g, collate_fn=DataCollatorWithPadding(self.tok))

    # ------------------------------------------------------------------ train
    def fit(self, texts, labels, eval_texts=None, eval_labels=None, ckpt_dir=None,
            max_steps=None, on_epoch=None):
        """Train for cfg epochs. If eval rows are given, score them after every epoch and
        keep the best epoch's weights in ckpt_dir/best (by balanced accuracy).
        Returns the per-epoch history."""
        c = self.cfg
        dl = self._loader(texts, labels, shuffle=True)
        no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm.weight")
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": c["weight_decay"]},
            {"params": [p for n, p in self.model.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=c["lr"])
        total = math.ceil(len(dl) / c["grad_accum"]) * c["epochs"]
        if max_steps:
            total = min(total, max_steps)
        sched = get_linear_schedule_with_warmup(opt, int(c["warmup"] * total), total)

        history, best, step = [], -1.0, 0
        for epoch in range(1, c["epochs"] + 1):
            self.model.train()
            t0, run_loss, n = time.time(), 0.0, 0
            for i, batch in enumerate(dl):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    loss = self.model(**batch).loss / c["grad_accum"]
                loss.backward()
                run_loss += loss.item() * c["grad_accum"]
                n += 1
                if (i + 1) % c["grad_accum"] == 0 or i + 1 == len(dl):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if max_steps and step >= max_steps:
                        break
            rec = {"epoch": epoch, "train_loss": run_loss / max(n, 1), "epoch_s": time.time() - t0}
            if eval_texts is not None:
                logits = self.predict_logits(eval_texts)
                rec.update(evaluate(logits, eval_labels))
                if ckpt_dir and rec["balanced_accuracy"] > best:
                    best = rec["balanced_accuracy"]
                    self.save(Path(ckpt_dir) / "best", extra={"epoch": epoch, **_short(rec)})
            history.append(rec)
            self.log(fmt_epoch(rec))
            if on_epoch:
                on_epoch(rec)
            if max_steps and step >= max_steps:
                break
        return history

    # ------------------------------------------------------------------ infer
    @torch.no_grad()
    def predict_logits(self, texts) -> np.ndarray:
        self.model.eval()
        out = []
        for batch in self._loader(texts):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out)

    def save(self, path, extra=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        (path / "ntllm.json").write_text(json.dumps({"labels": LABELS, **self.cfg, **(extra or {})}, indent=2))


def evaluate(logits: np.ndarray, labels) -> dict:
    """Balanced accuracy, macro precision, perplexity (exp mean CE of the true label)."""
    y = np.array([L2I[l] for l in labels])
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    ppl = float(np.exp(-logp[np.arange(len(y)), y].mean()))
    r = score(labels, np.array(LABELS)[logits.argmax(1)])
    return {"balanced_accuracy": r["balanced_accuracy"], "macro_precision": r["macro_precision"],
            "perplexity": ppl, "_score": r}


def _short(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def fmt_epoch(rec) -> str:
    s = f"epoch {rec['epoch']}: {rec['epoch_s']:.1f}s  train_loss {rec['train_loss']:.4f}"
    if "balanced_accuracy" in rec:
        s += (f"  bal_acc {rec['balanced_accuracy']:.4f}  macro_prec {rec['macro_precision']:.4f}"
              f"  ppl {rec['perplexity']:.3f}")
    return s
