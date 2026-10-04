"""IIT Bombay English-Hindi parallel corpus (~1.66M pairs, CC BY-NC 4.0).

Hugging Face: cfilt/iitb-english-hindi  (splits: train / validation / test)

The full corpus has many long, formal and noisy pairs. For a basic word-level
Seq2Seq without attention we keep only short, clean pairs, then sample a subset.
The result is cached as a TSV in data/ so the 190 MB download and the filtering
happen only once.
"""
import os
import random

from src.data import clean_english, clean_hindi

HF_NAME = "cfilt/iitb-english-hindi"


def _devanagari_ratio(text):
    """Fraction of letters in text that are Devanagari (U+0900-U+097F)."""
    letters = [ch for ch in text if ch.isalpha() or "\u0900" <= ch <= "\u097f"]
    if not letters:
        return 0.0
    return sum("\u0900" <= ch <= "\u097f" for ch in letters) / len(letters)


def _ascii_ratio(text):
    """Fraction of letters in text that are plain ASCII (English)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isascii() for ch in letters) / len(letters)


def keep_pair(en, hi, max_len, min_len=3):
    """Filters: non-empty, min_len..max_len tokens, sensible length ratio, correct script."""
    if not en or not hi:
        return False

    # Cheap pre-check on raw whitespace tokens before the (slower) cleaning
    if len(en.split()) > max_len + 5 or len(hi.split()) > max_len + 5:
        return False

    n_en = len(clean_english(en).split())
    n_hi = len(clean_hindi(hi).split())
    # min_len drops dictionary-style entries ('progeny' -> 'सन्तान') that are not sentences
    if not (min_len <= n_en <= max_len and min_len <= n_hi <= max_len):
        return False

    # Badly aligned pairs usually have very different lengths
    if not (0.5 <= n_hi / n_en <= 2.5):
        return False

    # Hindi side must be Devanagari (drops UI strings like "OK बटन"), English ASCII
    if _devanagari_ratio(hi) < 0.95 or _ascii_ratio(en) < 0.95:
        return False

    return True


def _clean_field(text):
    # Tabs/newlines would break the TSV cache
    return " ".join(text.replace("\t", " ").split())


def load_iitb_pairs(data_dir, max_len=15, max_pairs=100_000, seed=42, min_len=3):
    """
    Return up to max_pairs filtered (english, hindi) RAW pairs from the IITB
    training split. Cached in data_dir after the first call.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"iitb_min{min_len}_len{max_len}_n{max_pairs}_seed{seed}.tsv")

    if os.path.exists(cache):
        pairs = []
        with open(cache, encoding="utf-8") as f:
            for line in f:
                en, hi = line.rstrip("\n").split("\t")
                pairs.append((en, hi))
        return pairs

    # Imported here so the rest of the project works without `datasets`
    from datasets import load_dataset

    print(f"Downloading / loading {HF_NAME} (first time only) ...")
    ds = load_dataset(HF_NAME, split="train")

    seen = set()
    kept = []
    for i, ex in enumerate(ds):
        en = _clean_field(ex["translation"]["en"])
        hi = _clean_field(ex["translation"]["hi"])
        if not keep_pair(en, hi, max_len, min_len):
            continue
        key = (en.lower(), hi)                 # drop exact duplicates
        if key in seen:
            continue
        seen.add(key)
        kept.append((en, hi))
        if (i + 1) % 200_000 == 0:
            print(f"  scanned {i + 1:,} / {len(ds):,} | kept {len(kept):,}")

    print(f"Kept {len(kept):,} of {len(ds):,} pairs after filtering")

    # Reproducible random subset
    rng = random.Random(seed)
    rng.shuffle(kept)
    kept = kept[:max_pairs]

    with open(cache, "w", encoding="utf-8") as f:
        for en, hi in kept:
            f.write(f"{en}\t{hi}\n")
    print(f"Cached {len(kept):,} pairs -> {cache}")
    return kept


# --------------------------------------------------------------------------
# python -m src.iitb  ->  build the cache and print statistics about the data
# --------------------------------------------------------------------------
def main():
    from collections import Counter

    from src.config import Config
    from src.dataset import split_pairs

    cfg = Config()
    pairs = load_iitb_pairs(cfg.data_dir, cfg.iitb_max_len, cfg.iitb_max_pairs, cfg.seed)
    train, val, test = split_pairs(pairs, seed=cfg.seed)

    print(f"\nPairs: {len(pairs):,} -> train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print("avg tokens  en %.1f | hi %.1f" % (
        sum(len(clean_english(e).split()) for e, _ in pairs) / len(pairs),
        sum(len(clean_hindi(h).split()) for _, h in pairs) / len(pairs)))

    for side, clean, idx in [("English", clean_english, 0), ("Hindi", clean_hindi, 1)]:
        counts = Counter()
        for p in train:
            counts.update(clean(p[idx]).split())
        test_tokens = [w for p in test for w in clean(p[idx]).split()]
        print(f"\n--- {side} (vocab from TRAIN only) ---")
        for k in [1, 2, 3, 5]:
            vocab = {w for w, n in counts.items() if n >= k}
            cov = 100 * sum(w in vocab for w in test_tokens) / len(test_tokens)
            mark = "  <- current min_freq" if k == cfg.min_freq else ""
            print(f"min_freq={k}: vocab {len(vocab):>7,} | test tokens covered {cov:5.1f}%{mark}")

    print("\n--- 8 random examples ---")
    for en, hi in random.Random(0).sample(pairs, min(8, len(pairs))):
        print(" EN:", en)
        print(" HI:", hi)


if __name__ == "__main__":
    main()
