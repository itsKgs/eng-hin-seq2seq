"""Translate English sentences with a trained checkpoint (greedy decoding).

    python -m src.inference --cell_type gru --sentence "I am happy."
"""
import argparse
import os

import torch

from src.config import Config
from src.data import Vocab, clean_english
from src.model import build_model
from src.train import CHECKPOINT_DIR
from src.utils import get_device


def vocab_from_idx2word(idx2word):
    """Rebuild a Vocab from a saved id -> word list (exact same ids as training)."""
    vocab = Vocab([], min_freq=1)            # only the 4 special tokens
    vocab.idx2word = list(idx2word)
    vocab.word2idx = {word: i for i, word in enumerate(vocab.idx2word)}
    # Special-token ids are unchanged: they are always the first 4 entries
    return vocab


def load_model(checkpoint_path, device):
    """Rebuild config, vocabularies and model from a checkpoint."""
    # map_location: a GPU-trained checkpoint also loads on a CPU-only machine
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = Config(**ckpt["config"])
    src_vocab = vocab_from_idx2word(ckpt["src_idx2word"])
    tgt_vocab = vocab_from_idx2word(ckpt["tgt_idx2word"])

    model = build_model(cfg, src_vocab, tgt_vocab, ckpt["use_packing"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, src_vocab, tgt_vocab


def translate_batch(model, sentences, src_vocab, tgt_vocab, device, max_len):
    """Translate a list of RAW English sentences. Returns cleaned Hindi strings."""
    ids = [src_vocab.encode(clean_english(s), add_sos_eos=False) for s in sentences]
    # An empty sentence would break packing; give it a single <UNK>
    ids = [x if len(x) > 0 else [src_vocab.unk_idx] for x in ids]

    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    src = torch.full((len(ids), int(lengths.max())), src_vocab.pad_idx, dtype=torch.long)
    for i, x in enumerate(ids):
        src[i, :len(x)] = torch.tensor(x, dtype=torch.long)

    preds = model.greedy_decode(src.to(device), lengths, max_len)   # [B, <=max_len]
    return [tgt_vocab.decode(row.tolist()) for row in preds]


def translate(model, sentence, src_vocab, tgt_vocab, device, max_len):
    """Translate one RAW English sentence."""
    return translate_batch(model, [sentence], src_vocab, tgt_vocab, device, max_len)[0]


def main():
    parser = argparse.ArgumentParser(description="Translate English -> Hindi")
    parser.add_argument("--cell_type", default="gru", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--sentence", required=True)
    args = parser.parse_args()

    device = get_device()
    path = os.path.join(CHECKPOINT_DIR, f"{args.cell_type}_best.pt")
    model, cfg, src_vocab, tgt_vocab = load_model(path, device)
    print(translate(model, args.sentence, src_vocab, tgt_vocab, device, cfg.max_decode_len))


if __name__ == "__main__":
    main()
