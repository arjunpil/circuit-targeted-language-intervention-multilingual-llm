import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_common(model_dir, module_name):
    path = REPO / "discovery" / model_dir / "common.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


QWEN_COMMON = _load_common("qwen2.5-1.5b", "_test_qwen_common")
LLAMA_COMMON = _load_common("llama-3.2-1b", "_test_llama_common")

EXPECTED_LANGS = {"en", "es", "ru", "zh", "hi", "ko"}


def test_both_model_dirs_support_the_same_languages():
    assert set(QWEN_COMMON.LANG_CODES) == EXPECTED_LANGS
    assert set(LLAMA_COMMON.LANG_CODES) == EXPECTED_LANGS


def test_flores_codes_match_between_model_dirs():
    assert QWEN_COMMON.LANG_CODES == LLAMA_COMMON.LANG_CODES


def test_default_paths_preserve_the_original_es_layout():
    results_dir = REPO / "discovery" / "qwen2.5-1.5b" / "results"

    assert (
        QWEN_COMMON.default_metric_json("es")
        == results_dir / "en_es_language_metric.json"
    )
    assert (
        QWEN_COMMON.default_heads_json("es", "qwen25")
        == results_dir / "qwen25_en_es_heads.json"
    )


def test_default_paths_are_lang_specific_for_new_languages():
    results_dir = REPO / "discovery" / "qwen2.5-1.5b" / "results"

    assert (
        QWEN_COMMON.default_metric_json("ru")
        == results_dir / "en_ru_language_metric.json"
    )
    assert (
        QWEN_COMMON.default_heads_json("ru", "qwen25")
        == results_dir / "qwen25_en_ru_heads.json"
    )


def test_load_metric_tokens_reads_the_committed_legacy_es_file():
    data = QWEN_COMMON.load_metric_tokens(lang="es")

    assert len(data["english_token_ids"]) > 0
    assert data["target_token_ids"] == data["spanish_token_ids"]


def test_language_metric_is_antisymmetric_in_the_two_token_sets():
    import torch

    logits = torch.zeros(1, 10)
    logits[0, [1, 2]] = 5.0
    logits[0, [7, 8]] = -5.0

    forward = QWEN_COMMON.language_metric(logits, [1, 2], [7, 8])
    backward = QWEN_COMMON.language_metric(logits, [7, 8], [1, 2])

    assert torch.allclose(forward, -backward)


def test_freeze_heads_produces_the_compact_heads_format(tmp_path):
    discovery_dump = {
        "model": "Qwen/Qwen2.5-1.5B",
        "language_pair": "en->ru",
        "activation_site": "attention head output before o_proj",
        "n_examples": 30,
        "example_indices": [0, 29],
        "heads": {"16": [9], "25": [10]},
        "attribution_rows": [],
        "exact_rows": [],
    }

    discovery_path = tmp_path / "discovery.json"
    discovery_path.write_text(json.dumps(discovery_dump))

    out_path = tmp_path / "heads.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "discovery" / "freeze_heads.py"),
            "--discovery",
            str(discovery_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=REPO,
    )

    frozen = json.loads(out_path.read_text())

    assert frozen["heads"] == {"16": [9], "25": [10]}
    assert frozen["model"] == "Qwen/Qwen2.5-1.5B"
    assert frozen["language_pair"] == "en->ru"
    assert frozen["discovery_examples"] == 30
    assert frozen["heldout_examples"] == 50
    assert frozen["discovery_indices"] == [0, 29]
