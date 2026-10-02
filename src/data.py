import os
import re
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections import Counter

DATA_URL = "https://www.manythings.org/anki/hin-eng.zip"

# ManyThings rejects non-browser requests (HTTP 406), so look like a browser.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def download_dataset(data_dir: str) -> str:
    """Return the path to hin.txt, extracting/downloading only if needed."""
    txt_path = os.path.join(data_dir, "hin.txt")
    zip_path = os.path.join(data_dir, "hin-eng.zip")

    if os.path.exists(txt_path):
        return txt_path

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        try:
            request = urllib.request.Request(DATA_URL, headers=BROWSER_HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Automatic download failed ({e}).\n"
                f"Manual fix: download {DATA_URL} in your browser and upload it to\n"
                f"  {os.path.abspath(zip_path)}\n"
                f"then run this again."
            ) from e
        with open(zip_path, "wb") as f:
            f.write(content)

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        raise RuntimeError(f"{zip_path} is not a valid zip; download it manually.")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"hin.txt not found after extracting {zip_path}")
    return txt_path


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")       # line ke end se \n hatao
            parts = line.split("\t")
            if len(parts) < 2:
                continue                   # kharab line → chhodo
            en = parts[0]
            hi = parts[1]
            pairs.append((en, hi))
    return pairs


def clean_english(s):
    # 1. NFC normalize
    s = unicodedata.normalize("NFC", s)

    # 2. tedhe apostrophe → seedha
    s = s.replace("\u2019", "'")
    s = s.replace("\u2018", "'")            # ← SUDHAAR: ulta apostrophe bhi

    # 3. lowercase
    s = s.lower()

    # 4. punctuation ke dono taraf space
    s = re.sub(r'([?.!,":])', r" \1 ", s)   # ← SUDHAAR: " aur : jode (data se)

    # 5. extra spaces saaf
    s = re.sub(r"\s+", " ", s).strip()

    return s


def clean_hindi(s):
    # 1. NFC normalize
    s = unicodedata.normalize("NFC", s)

    # 2. | ko Hindi danda mein badlo
    s = s.replace("|", "।")

    # 3. punctuation ke dono taraf space
    s = re.sub(r'([।?!,".()])', r" \1 ", s)  # ← SUDHAAR: \- hataya (एक-दूसरे ek word)

    # 4. extra spaces saaf
    s = re.sub(r"\s+", " ", s).strip()

    return s


class Vocab:
    """
    Vocabulary for one language: maps word <-> integer id.

    Build:  vocab = Vocab(train_sentences, min_freq=2)   # from TRAIN data only, once
    Use:    vocab.encode(...)   -> list of ids for the model
            vocab.decode(...)   -> readable sentence from model output ids
            len(vocab)          -> vocabulary size (for Embedding / Linear layers)
    """

    # Special tokens always come first, in a fixed order:
    # <PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3
    SPECIALS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    def __init__(self, sentences, min_freq):
        """
        Runs ONCE, when the Vocab object is created. The vocabulary is built here.

        sentences : list of cleaned strings, e.g. ["i am happy .", ...]
        min_freq  : words appearing fewer times than this are left out (-> <UNK>)
        """

        # 1. Count how often each word appears.
        counts = Counter()
        for sentence in sentences:
            counts.update(sentence.split())   # add this sentence's words to the counts

        # 2. Start the id -> word list with the special tokens.
        #    .copy() matters: without it, idx2word and SPECIALS would be the SAME
        #    list, and appending words would modify the class-level SPECIALS.
        self.idx2word = self.SPECIALS.copy()

        # 3. Sort words into a FIXED order, so every run gives the same ids:
        #    higher count first (-counts[word]); ties broken alphabetically (word).
        words = sorted(counts.keys(), key=lambda word: (-counts[word], word))

        # 4. Keep only words that appear at least min_freq times.
        #    Rarer words are not added, so encode() will map them to <UNK>.
        for word in words:
            if counts[word] >= min_freq:
                self.idx2word.append(word)

        # 5. Build the reverse mapping: word -> id.
        #    enumerate gives (index, word); the index IS the word's id.
        self.word2idx = {}
        for i, word in enumerate(self.idx2word):
            self.word2idx[word] = i

        # 6. Shortcuts for the special token ids (looked up, never hard-coded).
        self.pad_idx = self.word2idx["<PAD>"]
        self.unk_idx = self.word2idx["<UNK>"]
        self.sos_idx = self.word2idx["<SOS>"]
        self.eos_idx = self.word2idx["<EOS>"]

    def encode(self, sentence, add_sos_eos):
        """
        Cleaned sentence -> list of ids.

        add_sos_eos=True  for the target (Hindi / decoder):  [<SOS>, ..., <EOS>]
        add_sos_eos=False for the source (English / encoder): no special tokens
        """
        ids = []
        for word in sentence.split():
            if word in self.word2idx:
                ids.append(self.word2idx[word])   # known word -> its id
            else:
                ids.append(self.unk_idx)          # unknown word -> <UNK>

        if add_sos_eos:
            ids.insert(0, self.sos_idx)   # <SOS> at the start
            ids.append(self.eos_idx)      # <EOS> at the end
        return ids

    def decode(self, ids):
        """
        List of ids -> readable sentence.
        Stops at the first <EOS>; skips <SOS> and <PAD>.
        """
        words = []
        for idx in ids:                   # 'idx', not 'id': avoid shadowing the built-in id()
            word = self.idx2word[idx]
            if word == "<EOS>":
                break                     # everything after <EOS> is meaningless
            if word == "<SOS>" or word == "<PAD>":
                continue                  # skip, but keep reading
            words.append(word)
        return " ".join(words)

    def __len__(self):
        """Vocabulary size, including the 4 special tokens. Enables len(vocab)."""
        return len(self.idx2word)
