import os
import re
import unicodedata
import urllib.error
import urllib.request
import zipfile

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

# Punctuation we split off as separate tokens.
# Note: we TARGET punctuation instead of selecting "word characters" with \w,
# because \w drops Devanagari vowel signs (category Mn/Mc).
EN_PUNCT = r'([?.!,;:"])'
HI_PUNCT = r'([?.!,;:"।॥])'


def download_dataset(data_dir: str) -> str:
    """Return the path to hin.txt, extracting/downloading only if needed.

    Order of attempts:
      1. data_dir/hin.txt already exists  -> use it
      2. data_dir/hin-eng.zip already exists  -> extract it (manual-upload path)
      3. otherwise                            -> download the zip, then extract
    """
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
        except urllib.error.URLError as e:  # HTTPError is a subclass
            raise RuntimeError(
                f"Automatic download failed ({e}).\n"
                f"Manual fix: download {DATA_URL} in your browser and upload it to\n"
                f"  {os.path.abspath(zip_path)}\n"
                f"then run this again."
            ) from e
        # Write only after a successful read, so a failed request never leaves
        # a broken file behind.
        with open(zip_path, "wb") as f:
            f.write(content)

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        raise RuntimeError(
            f"{zip_path} is not a valid zip (the server probably returned a web "
            f"page). It has been deleted; download it manually from {DATA_URL}."
        )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"hin.txt not found after extracting {zip_path}")
    return txt_path


def load_pairs(path: str) -> list[tuple[str, str]]:
    """Return raw (english, hindi) pairs, dropping the attribution field."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue  # malformed line
            en, hi = fields[0].strip(), fields[1].strip()
            if en and hi:
                pairs.append((en, hi))
    return pairs


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_english(s: str) -> str:
    """NFC -> unify apostrophes -> lowercase -> split punctuation -> collapse spaces."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")  # curly -> straight apostrophe
    s = s.lower()
    s = re.sub(EN_PUNCT, r" \1 ", s)
    return _normalize_whitespace(s)


def clean_hindi(s: str) -> str:
    """NFC -> split punctuation (incl. danda) -> collapse spaces. No lowercasing."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(HI_PUNCT, r" \1 ", s)
    return _normalize_whitespace(s)
