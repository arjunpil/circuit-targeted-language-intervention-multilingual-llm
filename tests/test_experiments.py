import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from experiments import aggregate, run
from experiments import config as cfgmod

REPO = Path(__file__).resolve().parents[1]

FAKE = textwrap.dedent('''
    import argparse, json, random
    from pathlib import Path

    p = argparse.ArgumentParser()
    for flag in ["model", "lang", "dtype", "n-fit", "n-eval", "n-ppl", "max-new-tokens", "batch-size", "steer-layers",
                 "resid-coefs", "resid-own-coefs", "head-coefs", "heads-json", "top-k", "head-min-layer", "n-random",
                 "n-nearby", "seed", "out"]:
        p.add_argument("--" + flag, default=None)
    a = p.parse_args()
    counter = Path(a.out).parent / "calls.txt"
    counter.parent.mkdir(parents=True, exist_ok=True)
    with open(counter, "a") as f:
        f.write(a.out + "\\n")
    if a.lang == "ru" and "FAIL_RU" in open(counter.parent / "mode.txt").read():
        raise SystemExit(3)
    rng = random.Random(int(a.seed or 0))
    out = Path(a.out) / (a.model.split("/")[-1] + "_" + a.lang)
    out.mkdir(parents=True, exist_ok=True)
    n = int(a.n_eval)
    base = [rng.uniform(0.5, 1.0) for _ in range(n)]
    plans = {("none", 0.0): 0.0, ("residual", -2.8): 0.1, ("heads", -10.0): -0.15}
    for i in range(4):
        plans[("random%d" % i, -10.0)] = -0.03
    for i in range(2):
        plans[("nearby%d" % i, -10.0)] = 0.0
    samples, rows = [], []
    for (cond, coef), shift in plans.items():
        for split in ("cs", "en"):
            for i in range(n):
                v = min(1.0, max(0.0, base[i] + shift + rng.gauss(0, 0.03))) if split == "cs" else 0.0
                samples.append({"condition": cond, "coef": coef, "split": split, "ted_idx": i, "csi_pair": v})
        rows.append({"condition": cond, "coef": coef, "ppl_en": 23.0 * (1.01 if cond == "heads" else 1.0), "ppl_" + a.lang: 15.0})
    (out / "samples.jsonl").write_text("\\n".join(json.dumps(s) for s in samples))
    (out / "rows.jsonl").write_text("\\n".join(json.dumps(r) for r in rows))
    (out / "summary.json").write_text(json.dumps({"ok": True}))
''')


def make_repo(tmp_path, langs=("es", "ru")):
    runner = tmp_path / "fake_pilot.py"
    runner.write_text(FAKE)
    heads = tmp_path / "heads_qwen25_en_es.json"
    heads.write_text(json.dumps({"heads": {"16": [9], "25": [10]}, "model": "Qwen/Qwen2.5-1.5B", "language_pair": "en->es"}))
    cfg = {
        "name": "t",
        "output_root": str(tmp_path / "out"),
        "runner": str(runner),
        "models": {"qwen25": "Qwen/Qwen2.5-1.5B"},
        "defaults": {"n_eval": 20, "seed": 0, "on_missing_heads": "standin",
                     "heads_json": str(tmp_path / "heads_{model}_en_{lang}.json")},
        "matrix": {"model": ["qwen25"], "lang": list(langs)},
        "overrides": [{"match": {"lang": "ru"}, "set": {"resid_coefs": [-2.2, -1, 1]}}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_expand_ids_overrides_and_head_sources(tmp_path):
    _, cfg = make_repo(tmp_path)
    runs = cfgmod.expand(cfg, tmp_path)
    assert [r["run_id"] for r in runs] == ["qwen25_en_es", "qwen25_en_ru"]
    assert runs[0]["head_source"] == "circuit" and runs[1]["head_source"] == "standin"
    assert runs[1]["resid_coefs"] == [-2.2, -1, 1] and "resid_coefs" not in runs[0]


def test_seed_in_matrix_changes_ids(tmp_path):
    _, cfg = make_repo(tmp_path, langs=("es",))
    cfg["matrix"]["seed"] = [0, 1]
    assert [r["run_id"] for r in cfgmod.expand(cfg, tmp_path)] == ["qwen25_en_es_seed0", "qwen25_en_es_seed1"]


def test_validation_errors(tmp_path):
    _, cfg = make_repo(tmp_path)
    bad = json.loads(json.dumps(cfg))
    bad["defaults"]["bogus_key"] = 1
    with pytest.raises(cfgmod.ConfigError, match="unknown keys"):
        cfgmod.expand(bad, tmp_path)
    mismatch = json.loads(json.dumps(cfg))
    mismatch["defaults"]["heads_json"] = str(tmp_path / "heads_qwen25_en_es.json")
    with pytest.raises(cfgmod.ConfigError, match="heads file is for en->es"):
        cfgmod.expand(mismatch, tmp_path)
    missing = json.loads(json.dumps(cfg))
    missing["defaults"]["on_missing_heads"] = "error"
    with pytest.raises(cfgmod.ConfigError, match="not found"):
        cfgmod.expand(missing, tmp_path)
    dup = json.loads(json.dumps(cfg))
    dup["runs"] = [{"model": "qwen25", "lang": "es"}]
    with pytest.raises(cfgmod.ConfigError, match="duplicate"):
        cfgmod.expand(dup, tmp_path)
    skip = json.loads(json.dumps(cfg))
    skip["defaults"]["on_missing_heads"] = "skip"
    assert [r["head_source"] for r in cfgmod.expand(skip, tmp_path)] == ["circuit", "skip"]


def real_flags():
    text = (REPO / "intervention" / "run_pilot.py").read_text()
    return set(re.findall(r'add_argument\("(--[a-z-]+)"', text))


def test_generated_flags_exist_in_run_pilot_and_negative_values_parse(tmp_path):
    _, cfg = make_repo(tmp_path)
    runs = cfgmod.expand(cfg, tmp_path)
    args = cfgmod.cli_args(runs[1], runs[1]["heads_path"])
    assert {a.split("=")[0] for a in args} <= real_flags()
    parser = argparse.ArgumentParser()
    for flag in real_flags():
        parser.add_argument(flag)
    cfg["defaults"]["head_coefs"] = [-1, -3, -10]
    parsed = parser.parse_args(cfgmod.cli_args(cfgmod.expand(cfg, tmp_path)[1], None))
    assert parsed.resid_coefs == "-2.2,-1,1" and parsed.head_coefs == "-1,-3,-10"


def test_run_caches_reruns_on_change_and_records_failures(tmp_path, capsys):
    path, cfg = make_repo(tmp_path)
    out = tmp_path / "out" / "t"
    out.mkdir(parents=True)
    (out / "mode.txt").write_text("OK")
    assert run.main([str(path)]) == 0
    calls = lambda: len((out / "calls.txt").read_text().split())
    assert calls() == 2
    manifest = json.loads((out / "qwen25_en_es" / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["head_source"] == "circuit" and manifest["seconds"] >= 0
    assert run.main([str(path)]) == 0 and calls() == 2
    assert run.main([str(path), "--force", "--only", "_ru"]) == 0 and calls() == 3
    cfg["defaults"]["n_eval"] = 21
    path.write_text(yaml.safe_dump(cfg))
    assert run.main([str(path)]) == 0 and calls() == 5
    (out / "mode.txt").write_text("FAIL_RU")
    assert run.main([str(path), "--force"]) == 1
    status = json.loads((out / "status.json").read_text())
    assert status == {"qwen25_en_es": "done", "qwen25_en_ru": "failed"}
    assert json.loads((out / "qwen25_en_ru" / "manifest.json").read_text())["returncode"] == 3


def test_aggregate_across_runs(tmp_path):
    path, _ = make_repo(tmp_path)
    out = tmp_path / "out" / "t"
    out.mkdir(parents=True)
    (out / "mode.txt").write_text("OK")
    assert run.main([str(path), "--aggregate"]) == 0
    agg = out / "aggregate"
    runs_csv = (agg / "runs.csv").read_text().splitlines()
    cond_csv = (agg / "conditions.csv").read_text().splitlines()
    assert len(runs_csv) == 3 and any("standin" in line for line in runs_csv)
    assert len(cond_csv) == 1 + 2 * 2
    result = aggregate.analyse_run(out / "qwen25_en_es" / "Qwen2.5-1.5B_es", "csi_pair", 1000)
    heads = next(e for e in result["entries"] if e["condition"] == "heads")
    assert -0.2 < heads["cs_delta"]["mean"] < -0.1 and heads["cs_delta"]["hi"] < 0
    assert heads["controls"]["random"]["p_one_sided"] == pytest.approx(1 / 5)
    assert heads["controls"]["random"]["heads_minus_control_mean"]["mean"] < -0.08
    assert heads["ppl_ratio"]["ppl_en"] == pytest.approx(1.01)
    resid = next(e for e in result["entries"] if e["condition"] == "residual")
    assert resid["cs_delta"]["lo"] > 0
    text = (agg / "report.md").read_text()
    assert "stand-in" in text and "qwen25_en_es" in text and "qwen25_en_ru" in text
    assert "different code versions" not in text


def test_code_hash_tracks_package_sources(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("x = 1\n")
    first = cfgmod.code_hash("pkg.mod", tmp_path)
    (tmp_path / "other.py").write_text("y = 2\n")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "mod.py").write_text("junk")
    assert cfgmod.code_hash("pkg.mod", tmp_path) == first
    (pkg / "mod.py").write_text("x = 2\n")
    assert cfgmod.code_hash("pkg.mod", tmp_path) != first
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.code_hash("missing.mod", tmp_path)


def test_code_change_invalidates_cache_and_report_warns(tmp_path):
    path, _ = make_repo(tmp_path)
    out = tmp_path / "out" / "t"
    out.mkdir(parents=True)
    (out / "mode.txt").write_text("OK")
    calls = lambda: len((out / "calls.txt").read_text().split())
    assert run.main([str(path), "--only", "_es"]) == 0 and calls() == 1
    assert run.main([str(path), "--only", "_es"]) == 0 and calls() == 1
    first = json.loads((out / "qwen25_en_es" / "manifest.json").read_text())["code_hash"]
    runner_file = tmp_path / "fake_pilot.py"
    runner_file.write_text(runner_file.read_text() + "\n")
    assert run.main([str(path), "--only", "_ru", "--aggregate"]) == 0 and calls() == 2
    assert run.main([str(path), "--only", "_es"]) == 0 and calls() == 3
    second = json.loads((out / "qwen25_en_es" / "manifest.json").read_text())["code_hash"]
    assert first != second
    assert run.main([str(path), "--only", "_ru"]) == 0 and calls() == 3


def test_aggregate_warns_on_mixed_code_versions(tmp_path):
    path, _ = make_repo(tmp_path)
    out = tmp_path / "out" / "t"
    out.mkdir(parents=True)
    (out / "mode.txt").write_text("OK")
    assert run.main([str(path), "--only", "_es"]) == 0
    runner_file = tmp_path / "fake_pilot.py"
    runner_file.write_text(runner_file.read_text() + "\n")
    assert run.main([str(path), "--only", "_ru", "--aggregate"]) == 0
    assert "different code versions" in (out / "aggregate" / "report.md").read_text()


def test_aggregate_lists_failed_runs(tmp_path):
    path, _ = make_repo(tmp_path)
    out = tmp_path / "out" / "t"
    out.mkdir(parents=True)
    (out / "mode.txt").write_text("FAIL_RU")
    assert run.main([str(path), "--aggregate"]) == 1
    assert "Not aggregated: qwen25_en_ru (failed)" in (out / "aggregate" / "report.md").read_text()


def test_repo_heads_files_resolve_to_circuits():
    cfg = cfgmod.load(REPO / "experiments" / "configs" / "en_es_circuit.yaml")
    runs = cfgmod.expand(cfg, REPO)
    assert {r["head_source"] for r in runs} == {"circuit"}
    assert all(Path(r["heads_path"]).exists() for r in runs)
    grid = cfgmod.expand(cfgmod.load(REPO / "experiments" / "configs" / "language_grid.yaml"), REPO)
    by_id = {r["run_id"]: r["head_source"] for r in grid}
    assert by_id["qwen25_en_es"] == "circuit" and by_id["llama32_en_es"] == "circuit"
    for lang in ("ru", "zh", "hi"):
        assert by_id[f"qwen25_en_{lang}"] == "circuit"
        assert by_id[f"llama32_en_{lang}"] == "circuit"
    assert len(by_id) == 8


def test_new_pilot_flags_are_supported():
    assert {"resid_gated_coefs", "head_dir"} <= set(cfgmod.PILOT_KEYS)
    assert {"--resid-gated-coefs", "--head-dir"} <= real_flags()


def test_repo_configs_expand(tmp_path):
    for name in ("en_es_circuit", "language_grid", "en_es_head_dir"):
        cfg = cfgmod.load(REPO / "experiments" / "configs" / f"{name}.yaml")
        runs = cfgmod.expand(cfg, REPO)
        if name == "en_es_circuit":
            assert runs[0]["head_coefs"] == [-1, -3, -10, -20]
        assert runs and all(r["run_id"] for r in runs)
        for r in runs:
            if r["head_source"] != "skip":
                flags = {a.split("=")[0] for a in cfgmod.cli_args(r, r["heads_path"])}
                assert flags <= real_flags()
