"""Train one model.  Usage (from the repo root):

    python -m src.train --cell_type gru
    python -m src.train --cell_type lstm --num_epochs 15
"""
import argparse
import dataclasses
import json
import math
import os
import time

import torch
import torch.nn as nn

from src.config import Config
from src.dataset import build_data
from src.model import build_model, count_parameters
from src.utils import get_device, set_seed

CHECKPOINT_DIR = "checkpoints"   # gitignored; lives on Drive next to the code


def run_epoch(model, loader, criterion, device, pad_idx,
              optimizer=None, clip_max_norm=None, teacher_forcing_ratio=1.0):
    """
    One pass over a loader.
    optimizer given  -> training mode (updates weights)
    optimizer None   -> evaluation mode (no gradients, no updates)
    Returns the average loss per real (non-<PAD>) target token.
    """
    is_train = optimizer is not None
    model.train(is_train)          # train(): dropout etc. on; eval(): off

    total_loss = 0.0
    total_tokens = 0

    # No graph is built during evaluation -> faster, less memory
    with torch.set_grad_enabled(is_train):
        for src, src_lengths, tgt in loader:
            # src_lengths stays on the CPU (pack_padded_sequence needs that)
            src = src.to(device)
            tgt = tgt.to(device)

            # logits: [B, T-1, V]; labels: [B, T-1] (target shifted by one)
            logits = model(src, src_lengths, tgt, teacher_forcing_ratio)
            labels = tgt[:, 1:]

            # CrossEntropyLoss wants [N, C] and [N]: flatten batch and time
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

            if is_train:
                optimizer.zero_grad()                  # clear old gradients
                loss.backward()                        # compute new gradients
                torch.nn.utils.clip_grad_norm_(        # cap the gradient norm
                    model.parameters(), clip_max_norm
                )
                optimizer.step()                       # update the weights

            # Weight by the number of real tokens so the epoch average is exact
            n_tokens = (labels != pad_idx).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Train an English->Hindi Seq2Seq model")
    parser.add_argument("--cell_type", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--teacher_forcing_ratio", type=float)
    parser.add_argument("--no_packing", action="store_true",
                        help="disable pack_padded_sequence in the encoder")
    return parser.parse_args()


def main():
    args = parse_args()

    # Start from the defaults and override only what was passed on the command line
    overrides = {}
    if args.cell_type is not None:
        overrides["cell_type"] = args.cell_type
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.teacher_forcing_ratio is not None:
        overrides["teacher_forcing_ratio"] = args.teacher_forcing_ratio
    cfg = dataclasses.replace(Config(), **overrides)
    use_packing = not args.no_packing

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device} | cell_type: {cfg.cell_type} | packing: {use_packing}")

    # Data pipeline (split -> train-only vocab -> loaders)
    data = build_data(cfg)
    src_vocab, tgt_vocab = data["en_vocab"], data["hi_vocab"]
    print(f"Vocab sizes: en={len(src_vocab)} hi={len(tgt_vocab)}")

    # Re-seed right before creating the model so every cell type starts from
    # the same RNG state (fair comparison)
    set_seed(cfg.seed)
    model = build_model(cfg, src_vocab, tgt_vocab, use_packing).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # Raw logits go in; <PAD> positions in the labels are ignored
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # e.g. "gru" or "gru_nopack" (the ablation does not overwrite the main run)
    run_name = cfg.cell_type if use_packing else f"{cfg.cell_type}_nopack"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pt")
    hist_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_history.json")

    best_val = math.inf
    history = []
    start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model, data["train_loader"], criterion, device, tgt_vocab.pad_idx,
            optimizer=optimizer, clip_max_norm=cfg.clip_max_norm,
            teacher_forcing_ratio=cfg.teacher_forcing_ratio,
        )
        # Validation: teacher-forced next-word loss (same for every model)
        val_loss = run_epoch(
            model, data["val_loader"], criterion, device, tgt_vocab.pad_idx,
            teacher_forcing_ratio=1.0,
        )
        secs = time.time() - t0

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            # Save weights + everything needed to rebuild the model later
            torch.save({
                "model_state": model.state_dict(),
                "config": dataclasses.asdict(cfg),
                "use_packing": use_packing,
                "src_idx2word": src_vocab.idx2word,
                "tgt_idx2word": tgt_vocab.idx2word,
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "seconds": secs})
        print(f"epoch {epoch:2d} | train {train_loss:.3f} | val {val_loss:.3f} "
              f"| {secs:5.1f}s {'| saved' if improved else ''}")

    total_time = time.time() - start
    with open(hist_path, "w") as f:
        json.dump({"cell_type": cfg.cell_type, "parameters": n_params,
                   "total_seconds": total_time, "best_val_loss": best_val,
                   "history": history}, f, indent=2)

    print(f"Done in {total_time:.0f}s. Best val loss {best_val:.3f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
