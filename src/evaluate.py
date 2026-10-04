"""Compare the trained RNN / GRU / LSTM models on the TEST split.

    python -m src.evaluate                 # all models that have a checkpoint
    python -m src.evaluate --cells gru lstm
"""
import argparse
import json
import os
from functools import partial

import sacrebleu
import torch
import torch.nn as nn

from src.config import Config
from src.data import clean_english, clean_hindi
from src.iitb import load_iitb_pairs
from src.dataset import TranslationDataset, collate_fn, split_pairs
from src.inference import load_model, translate_batch
from src.train import CHECKPOINT_DIR, run_epoch
from src.utils import get_device

N_QUALITATIVE = 15     # fixed test sentences shown side by side
SHORT_MAX = 5          # English tokens <= 5  -> "short"
LONG_MIN = 10          # English tokens >= 10 -> "long"


def bleu(hyps, refs):
    """Corpus BLEU. Both sides are already tokenized by our cleaning."""
    if not hyps:
        return float("nan")
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score


def evaluate_model(cell_type, test_pairs, device):
    path = os.path.join(CHECKPOINT_DIR, f"{cell_type}_best.pt")
    if not os.path.exists(path):
        return None

    model, cfg, src_vocab, tgt_vocab = load_model(path, device)

    # Test loss (teacher-forced, same definition as the validation loss)
    test_ds = TranslationDataset(test_pairs, src_vocab, tgt_vocab)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=partial(collate_fn, src_pad_idx=src_vocab.pad_idx,
                           tgt_pad_idx=tgt_vocab.pad_idx),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    test_loss = run_epoch(model, loader, criterion, device, tgt_vocab.pad_idx,
                          teacher_forcing_ratio=1.0)

    # Greedy translations of the whole test set
    sources = [en for en, _ in test_pairs]
    refs = [clean_hindi(hi) for _, hi in test_pairs]
    hyps = []
    for i in range(0, len(sources), cfg.batch_size):
        hyps += translate_batch(model, sources[i:i + cfg.batch_size],
                                src_vocab, tgt_vocab, device, cfg.max_decode_len)

    # BLEU overall, and separately for short and long source sentences
    src_lens = [len(clean_english(s).split()) for s in sources]
    short = [i for i, n in enumerate(src_lens) if n <= SHORT_MAX]
    long_ = [i for i, n in enumerate(src_lens) if n >= LONG_MIN]

    hist_path = os.path.join(CHECKPOINT_DIR, f"{cell_type}_history.json")
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else {}
    best_epoch = min(hist.get("history", [{}]),
                     key=lambda h: h.get("val_loss", float("inf")), default={})

    return {
        "cell_type": cell_type,
        "parameters": hist.get("parameters"),
        "train_seconds": hist.get("total_seconds"),
        "best_epoch": best_epoch.get("epoch"),
        "train_loss": best_epoch.get("train_loss"),
        "val_loss": best_epoch.get("val_loss"),
        "test_loss": test_loss,
        "bleu": bleu(hyps, refs),
        "bleu_short": bleu([hyps[i] for i in short], [refs[i] for i in short]),
        "bleu_long": bleu([hyps[i] for i in long_], [refs[i] for i in long_]),
        "n_short": len(short),
        "n_long": len(long_),
        "hyps": hyps,
    }


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test set")
    parser.add_argument("--cells", nargs="+", default=["rnn", "gru", "lstm"])
    args = parser.parse_args()

    device = get_device()
    cfg = Config()

    # Same split as training (same seed, same function)
    pairs = load_iitb_pairs(cfg.data_dir, cfg.iitb_max_len, cfg.iitb_max_pairs, cfg.seed)
    _, _, test_pairs = split_pairs(pairs, seed=cfg.seed)

    results = [r for r in (evaluate_model(c, test_pairs, device) for c in args.cells) if r]
    if not results:
        print("No checkpoints found. Train first: python -m src.train --cell_type gru")
        return

    # ---- Comparison table (markdown) ----
    n_short, n_long = results[0]["n_short"], results[0]["n_long"]
    header = ["Metric"] + [r["cell_type"].upper() for r in results]
    rows = [
        ["Parameters"] + [fmt(r["parameters"], ",") for r in results],
        ["Training time (s)"] + [fmt(r["train_seconds"], ".0f") for r in results],
        ["Best epoch"] + [fmt(r["best_epoch"], "d") for r in results],
        ["Train loss (best epoch)"] + [fmt(r["train_loss"], ".3f") for r in results],
        ["Val loss (best)"] + [fmt(r["val_loss"], ".3f") for r in results],
        ["Test loss"] + [fmt(r["test_loss"], ".3f") for r in results],
        ["Test BLEU"] + [fmt(r["bleu"], ".2f") for r in results],
        [f"BLEU short (<= {SHORT_MAX} words, n={n_short})"]
        + [fmt(r["bleu_short"], ".2f") for r in results],
        [f"BLEU long (>= {LONG_MIN} words, n={n_long})"]
        + [fmt(r["bleu_long"], ".2f") for r in results],
    ]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    table = "\n".join(lines)
    print(table)

    # ---- Fixed qualitative examples ----
    print("\n=== Qualitative comparison (first", N_QUALITATIVE, "test sentences) ===")
    qual_lines = []
    for i in range(min(N_QUALITATIVE, len(test_pairs))):
        en, hi = test_pairs[i]
        block = [f"\n[{i + 1}] EN : {en}", f"     REF: {clean_hindi(hi)}"]
        for r in results:
            block.append(f"     {r['cell_type'].upper():4s}: {r['hyps'][i]}")
        qual_lines += block
    print("\n".join(qual_lines))

    # Save everything next to the checkpoints
    out_path = os.path.join(CHECKPOINT_DIR, "results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n" + "\n".join(qual_lines) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
