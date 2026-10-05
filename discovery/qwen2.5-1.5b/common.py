import json
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_DIR = ROOT / "discovery"
MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = DISCOVERY_DIR / "data"
RESULTS_DIR = MODEL_DIR / "results"

FLORES_ARCHIVE = DATA_DIR / "flores200_dataset.tar.gz"
FLORES_DIR = DATA_DIR / "flores200_dataset"

FLORES_URL = (
    "https://dl.fbaipublicfiles.com/"
    "nllb/flores200_dataset.tar.gz"
)

LANG_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
}


def ensure_flores():
    """Download/extract FLORES-200 when it is not already present."""
    if FLORES_DIR.exists():
        return FLORES_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FLORES_ARCHIVE.exists():
        tmp = FLORES_ARCHIVE.with_suffix(
            FLORES_ARCHIVE.suffix + ".part"
        )

        print("Downloading FLORES-200...")
        urlretrieve(FLORES_URL, tmp)
        tmp.rename(FLORES_ARCHIVE)

    print("Extracting FLORES-200...")

    with tarfile.open(FLORES_ARCHIVE) as tar:
        try:
            tar.extractall(DATA_DIR, filter="data")
        except TypeError:
            # Compatibility with Python versions without the filter arg.
            tar.extractall(DATA_DIR)

    return FLORES_DIR


def flores(lang, split="devtest"):
    """Return FLORES sentences for a supported language."""
    if lang not in LANG_CODES:
        raise ValueError(
            f"Unsupported language: {lang}. "
            f"Supported: {sorted(LANG_CODES)}"
        )

    root = ensure_flores()
    code = LANG_CODES[lang]

    path = root / split / f"{code}.{split}"

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return [
            line.rstrip("\n")
            for line in f
        ]


def load_metric_tokens(path=None):
    """
    Load the frozen EN/ES token sets used by the EN→ES experiment.

    Keeping the token IDs frozen makes the circuit-discovery run
    reproduce the metric used for the committed results rather than
    silently rebuilding a slightly different objective.
    """
    if path is None:
        path = (
            RESULTS_DIR
            / "en_es_language_metric.json"
        )

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return {
        "english_token_ids": data["english_token_ids"],
        "spanish_token_ids": data["spanish_token_ids"],
        "metadata": data,
    }


def language_metric(
    logits,
    english_token_ids,
    spanish_token_ids,
):
    """
    Spanish-vs-English next-token score.

    Positive values indicate more Spanish-specific next-token mass;
    negative values indicate more English-specific next-token mass.
    """
    en = torch.as_tensor(
        english_token_ids,
        device=logits.device,
        dtype=torch.long,
    )

    es = torch.as_tensor(
        spanish_token_ids,
        device=logits.device,
        dtype=torch.long,
    )

    en_score = torch.logsumexp(
        logits.index_select(-1, en),
        dim=-1,
    )

    es_score = torch.logsumexp(
        logits.index_select(-1, es),
        dim=-1,
    )

    return es_score - en_score


def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-1.5B",
    device=None,
):
    """Load the model in the configuration used by the pilot."""
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("Loading tokenizer:", model_name)
    tok = AutoTokenizer.from_pretrained(
        model_name
    )

    print("Loading model:", model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
    except TypeError:
        # Compatibility with older Transformers versions.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )

    model = model.to(device)
    model.eval()
    model.config.use_cache = False

    print("Device:", next(model.parameters()).device)
    print("Layers:", model.config.num_hidden_layers)
    print("Query heads:", model.config.num_attention_heads)
    print("KV heads:", model.config.num_key_value_heads)

    return model, tok


def head_dim(model):
    return (
        getattr(
            model.config,
            "head_dim",
            None,
        )
        or model.config.hidden_size
        // model.config.num_attention_heads
    )


def build_position_matched_examples(
    tok,
    start,
    n_examples,
    max_len=32,
    min_len=6,
):
    """
    Build EN/ES aligned examples using position-matched token prefixes.

    Each pair is independently tokenized, then both sides are truncated
    to the same token length. This keeps final-position RoPE indices
    aligned while preserving the aligned FLORES sentence pair.
    """
    en_texts = flores("en", "devtest")
    es_texts = flores("es", "devtest")

    stop = min(
        start + n_examples,
        len(en_texts),
        len(es_texts),
    )

    examples = []

    for index in range(start, stop):
        en_ids = tok(
            en_texts[index],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        es_ids = tok(
            es_texts[index],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        length = min(
            len(en_ids),
            len(es_ids),
            max_len,
        )

        if length < min_len:
            continue

        examples.append(
            {
                "index": index,
                "length": int(length),
                "en_ids": en_ids[:length],
                "es_ids": es_ids[:length],
            }
        )

    return examples


def model_inputs(ids, device):
    ids = ids.unsqueeze(0).to(device)

    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
    }


@torch.no_grad()
def capture_head_outputs(
    model,
    inputs,
    layers=None,
):
    """
    Capture z, the concatenated attention-head outputs immediately
    before each layer's o_proj.
    """
    if layers is None:
        layers = range(
            model.config.num_hidden_layers
        )

    store = {}
    handles = []

    for layer in layers:
        o_proj = (
            model.model.layers[layer]
            .self_attn
            .o_proj
        )

        def hook(
            module,
            hook_inputs,
            layer=layer,
        ):
            store[layer] = (
                hook_inputs[0]
                .detach()
                .clone()
            )

        handles.append(
            o_proj.register_forward_pre_hook(
                hook
            )
        )

    try:
        model(
            **inputs,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    return store


def forward_capture_head_outputs(
    model,
    inputs,
    layers=None,
):
    """
    Forward pass that keeps captured z tensors attached to autograd.
    """
    if layers is None:
        layers = range(
            model.config.num_hidden_layers
        )

    store = {}
    handles = []

    for layer in layers:
        o_proj = (
            model.model.layers[layer]
            .self_attn
            .o_proj
        )

        def hook(
            module,
            hook_inputs,
            layer=layer,
        ):
            store[layer] = hook_inputs[0]

        handles.append(
            o_proj.register_forward_pre_hook(
                hook
            )
        )

    try:
        outputs = model(
            **inputs,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    return outputs, store


@torch.no_grad()
def exact_head_patch_metric(
    model,
    inputs,
    layer,
    head,
    clean_z,
    english_token_ids,
    spanish_token_ids,
):
    """
    Replace one final-position attention-head output with its clean
    counterpart and evaluate the exact language-metric change.
    """
    dh = head_dim(model)

    start = head * dh
    end = start + dh

    replacement = (
        clean_z[layer][
            0,
            -1,
            start:end,
        ]
        .detach()
        .clone()
    )

    o_proj = (
        model.model.layers[layer]
        .self_attn
        .o_proj
    )

    def hook(
        module,
        hook_inputs,
    ):
        z = hook_inputs[0].clone()

        z[
            :,
            -1,
            start:end,
        ] = replacement.to(
            device=z.device,
            dtype=z.dtype,
        )

        return (z,) + tuple(
            hook_inputs[1:]
        )

    handle = (
        o_proj.register_forward_pre_hook(
            hook
        )
    )

    try:
        logits = model(
            **inputs,
            use_cache=False,
        ).logits

        value = language_metric(
            logits[:, -1, :],
            english_token_ids,
            spanish_token_ids,
        )[0]
    finally:
        handle.remove()

    return float(value)
