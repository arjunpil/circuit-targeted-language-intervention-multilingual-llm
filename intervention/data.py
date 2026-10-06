import csv
import json
import os
import tarfile
import zipfile
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

DATA = Path(os.environ.get("CTLI_DATA", Path(__file__).resolve().parents[1] / ".data"))

URLS = {
    "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "flores200_dataset.tar.gz": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
    "ted_talks_code_switching_second_half.jsonl":
        "https://raw.githubusercontent.com/fxlrnrpt/language-steering-in-latent-space/HEAD/data/"
        "ted_talks_code_switching_second_half.jsonl",
    "lcb_test_sets.zip": "https://raw.githubusercontent.com/for-ai/language-confusion/HEAD/test_sets.zip",
    "lcb_words": "https://raw.githubusercontent.com/for-ai/language-confusion/HEAD/words",
}
EXTRACT_TO = {"flores200_dataset.tar.gz": "flores200_dataset", "lcb_test_sets.zip": "lcb"}

LANGS = {
    "en": ("eng_Latn", "eng_Latn", "en"),
    "es": ("spa_Latn", "spa_Latn", "es"),
    "ru": ("rus_Cyrl", "rus_Cyrl", "ru"),
    "zh": ("zho_Hans", "cmn_Hans", "zh"),
    "hi": ("hin_Deva", "hin_Deva", "hi"),
    "ko": ("kor_Hang", None, "ko"),
}


def ensure(name):
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urlretrieve(URLS[name], tmp)
        tmp.rename(path)
    target = EXTRACT_TO.get(name)
    if target and not (DATA / target).exists():
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                z.extractall(DATA / target)
        else:
            with tarfile.open(path) as t:
                t.extractall(DATA, filter="data")
    return path


def lid_path():
    return ensure("lid.176.bin")


@lru_cache(maxsize=None)
def flores(lang, split="dev"):
    ensure("flores200_dataset.tar.gz")
    code = LANGS[lang][0]
    with open(DATA / "flores200_dataset" / split / f"{code}.{split}", encoding="utf-8") as f:
        return tuple(line.rstrip("\n") for line in f)


@lru_cache(maxsize=None)
def ted_code_switch(lang):
    col = LANGS[lang][1]
    with open(ensure("ted_talks_code_switching_second_half.jsonl"), encoding="utf-8") as f:
        return tuple(json.loads(line)[col] for line in f)


@lru_cache(maxsize=None)
def lcb(lang, task):
    ensure("lcb_test_sets.zip")
    rows = []
    for path in sorted((DATA / "lcb" / "test_sets" / task).glob(f"*/{lang}.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            rows += [(path.parent.name, r["prompt"]) for r in csv.DictReader(f)]
    return tuple(rows)


@lru_cache(maxsize=None)
def lcb_en_words():
    with open(ensure("lcb_words"), encoding="utf-8") as f:
        return frozenset(w for w in (line.strip() for line in f) if w.islower() and len(w) > 3)
