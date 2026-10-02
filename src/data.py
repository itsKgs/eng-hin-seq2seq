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