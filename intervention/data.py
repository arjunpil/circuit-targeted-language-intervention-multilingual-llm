import json
import os
import tarfile
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
}

LANGS = {
    "en": ("eng_Latn", "eng_Latn", "en"),
    "es": ("spa_Latn", "spa_Latn", "es"),
    "ru": ("rus_Cyrl", "rus_Cyrl", "ru"),
    "zh": ("zho_Hans", "cmn_Hans", "zh"),
    "hi": ("hin_Deva", "hin_Deva", "hi"),
}


def ensure(name):
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urlretrieve(URLS[name], tmp)
        tmp.rename(path)
    if name.endswith(".tar.gz") and not (DATA / "flores200_dataset").exists():
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
