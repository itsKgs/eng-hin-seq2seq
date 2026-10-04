"""PyTorch side of the data pipeline: split, Dataset, padding, DataLoaders.

src/data.py    -> plain-text work (download, load, clean, Vocab); no torch needed.
src/dataset.py -> turns that text into batched tensors for the model.
"""
import random
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data import Vocab, clean_english, clean_hindi
from src.iitb import load_iitb_pairs


def split_pairs(pairs, seed, train_frac=0.8, val_frac=0.1):
    """Shuffle a COPY of pairs reproducibly and split into train / val / test."""
    # Make a copy so the original list is not modified (no side effects)
    data = pairs.copy()

    # Dedicated generator: same seed -> same split, whatever ran before
    rng = random.Random(seed)
    rng.shuffle(data)

    # Calculate the train and validation sizes; test gets the remainder
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


class TranslationDataset(Dataset):
    """Cleans and encodes every pair ONCE; __getitem__ just returns tensors."""

    def __init__(self, pairs, en_vocab, hi_vocab):
        # Store all encoded examples
        self.examples = []

        for en, hi in pairs:
            # Clean both sentences
            en = clean_english(en)
            hi = clean_hindi(hi)

            # English (encoder input): no <SOS>/<EOS>
            src = en_vocab.encode(en, add_sos_eos=False)
            # Hindi (decoder target): with <SOS> ... <EOS>
            tgt = hi_vocab.encode(hi, add_sos_eos=True)

            # Token ids must be integer (int64) tensors for nn.Embedding
            src = torch.tensor(src, dtype=torch.long)
            tgt = torch.tensor(tgt, dtype=torch.long)

            self.examples.append((src, tgt))

    def __len__(self):
        # Number of examples
        return len(self.examples)

    def __getitem__(self, i):
        # Return the i-th (src, tgt) pair
        return self.examples[i]


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    """
    Turn a list of (src, tgt) examples into padded batch tensors.

    Returns:
        src         : [B, max_src_len]  padded source ids
        src_lengths : [B]               real (unpadded) source lengths (keep on CPU)
        tgt         : [B, max_tgt_len]  padded target ids
    """
    # Separate source and target sequences
    src_list = []
    tgt_list = []
    for src, tgt in batch:
        src_list.append(src)
        tgt_list.append(tgt)

    # Real source lengths (needed later for pack_padded_sequence)
    src_lengths = []
    for src in src_list:
        src_lengths.append(len(src))
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)

    # Pad each side with ITS OWN vocabulary's <PAD> id, only up to the
    # longest sequence in this batch (dynamic padding)
    src = pad_sequence(src_list, batch_first=True, padding_value=src_pad_idx)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad_idx)

    return src, src_lengths, tgt


def build_data(cfg):
    """
    Full data pipeline in one call:
        load -> split -> vocab (TRAIN only) -> datasets -> loaders

    Returns a dict with the three loaders and the two vocabularies.
    """
    # 1. Load raw pairs and split them (split BEFORE building the vocab)
    pairs = load_iitb_pairs(cfg.data_dir, cfg.iitb_max_len, cfg.iitb_max_pairs, cfg.seed)
    train, val, test = split_pairs(pairs, seed=cfg.seed)

    # 2. Vocabularies from the TRAIN split only (no leakage from val/test)
    en_vocab = Vocab([clean_english(en) for en, _ in train], cfg.min_freq)
    hi_vocab = Vocab([clean_hindi(hi) for _, hi in train], cfg.min_freq)

    # 3. Datasets
    train_ds = TranslationDataset(train, en_vocab, hi_vocab)
    val_ds = TranslationDataset(val, en_vocab, hi_vocab)
    test_ds = TranslationDataset(test, en_vocab, hi_vocab)

    # 4. One collate function for all loaders; pad ids passed BY NAME
    collate = partial(collate_fn,
                      src_pad_idx=en_vocab.pad_idx,
                      tgt_pad_idx=hi_vocab.pad_idx)

    # 5. Seeded generator -> reproducible shuffle order for training
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, collate_fn=collate)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "en_vocab": en_vocab,
        "hi_vocab": hi_vocab,
    }
