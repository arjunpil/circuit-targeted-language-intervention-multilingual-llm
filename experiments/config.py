import hashlib
import itertools
import json
import re
from pathlib import Path

import yaml

PILOT_KEYS = (
    "dtype", "n_fit", "n_eval", "n_ppl", "max_new_tokens", "batch_size", "steer_layers", "resid_coefs",
    "resid_gated_coefs", "resid_own_coefs", "head_coefs", "head_dir", "top_k", "head_min_layer", "n_random", "n_nearby", "seed",
)
AUX_KEYS = ("model", "lang", "model_dir", "heads_json", "tag", "on_missing_heads", "allow_head_mismatch", "model_name")
LANGS = ("es", "fr", "ru", "zh", "hi")
POLICIES = ("error", "standin", "skip")


class ConfigError(ValueError):
    pass


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    cfg = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
    if not isinstance(cfg, dict) or "name" not in cfg:
        raise ConfigError(f"{path}: config must be a mapping with a 'name'")
    return cfg


def short(model_name):
    return model_name.split("/")[-1]


def slug(value):
    return re.sub(r"[^A-Za-z0-9.\-]", "-", str(value))


def points(cfg):
    out = []
    matrix = cfg.get("matrix") or {}
    if matrix:
        keys = list(matrix)
        for combo in itertools.product(*[matrix[k] for k in keys]):
            out.append((dict(zip(keys, combo)), set(keys)))
    for run in cfg.get("runs") or []:
        out.append((dict(run), set()))
    if not out:
        raise ConfigError("config defines neither 'matrix' nor 'runs'")
    return out


def run_id(params, matrix_keys):
    parts = [params["model"], "en", params["lang"]]
    for key in sorted(matrix_keys - {"model", "lang"}):
        parts.append(f"{key}{params[key]}")
    if params.get("tag"):
        parts.append(params["tag"])
    return "_".join(slug(p) for p in parts)


def resolve(cfg, point, matrix_keys):
    models = cfg.get("models") or {}
    params = {**(cfg.get("defaults") or {}), **point}
    for ov in cfg.get("overrides") or []:
        if all(params.get(k) == v for k, v in ov["match"].items()):
            params.update(ov["set"])
    params.update(point)
    for required in ("model", "lang"):
        if required not in params:
            raise ConfigError(f"run is missing '{required}': {point}")
    if params["lang"] not in LANGS:
        raise ConfigError(f"lang must be one of {LANGS}, got {params['lang']!r}")
    params["model_name"] = models.get(params["model"], params["model"])
    unknown = set(params) - set(PILOT_KEYS) - set(AUX_KEYS)
    if unknown:
        raise ConfigError(f"unknown keys {sorted(unknown)} in run {point}")
    policy = params.get("on_missing_heads", "error")
    if policy not in POLICIES:
        raise ConfigError(f"on_missing_heads must be one of {POLICIES}")
    if isinstance(params.get("heads_json"), str):
        params["heads_json"] = params["heads_json"].format(**params)
    params["run_id"] = run_id(params, matrix_keys)
    return params


def read_heads(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = data.get("heads")
    if not isinstance(heads, dict) or not heads:
        raise ConfigError(f"{path}: expected a non-empty 'heads' mapping")
    for layer, hs in heads.items():
        if not str(layer).isdigit() or not isinstance(hs, list) or not hs or not all(isinstance(h, int) for h in hs):
            raise ConfigError(f"{path}: bad entry for layer {layer!r}")
    return data


def check_heads(params, repo_root):
    path = params.get("heads_json")
    if not path:
        return None, "standin"
    full = Path(path) if Path(path).is_absolute() else Path(repo_root) / path
    if not full.exists():
        policy = params.get("on_missing_heads", "error")
        if policy == "error":
            raise ConfigError(f"{params['run_id']}: heads file not found: {path}")
        return None, policy
    data = read_heads(full)
    if not params.get("allow_head_mismatch"):
        if data.get("model") and data["model"] != params["model_name"]:
            raise ConfigError(f"{params['run_id']}: heads file is for {data['model']}, run uses {params['model_name']}")
        pair = data.get("language_pair")
        if pair and pair != f"en->{params['lang']}":
            raise ConfigError(f"{params['run_id']}: heads file is for {pair}, run uses en->{params['lang']}")
    return str(full), "circuit"


def cli_args(params, heads_path):
    args = [f"--model={params['model_name']}", f"--lang={params['lang']}"]
    for key in PILOT_KEYS:
        value = params.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        args.append(f"--{key.replace('_', '-')}={value}")
    if heads_path:
        args.append(f"--heads-json={heads_path}")
    return args


def code_hash(runner, repo_root):
    root = Path(repo_root)
    if runner.endswith(".py"):
        path = Path(runner) if Path(runner).is_absolute() else root / runner
        base, files = path.parent, [path] if path.exists() else []
    else:
        base = root / runner.split(".")[0]
        files = sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts) if base.is_dir() else []
    if not files:
        raise ConfigError(f"cannot find the source of runner {runner!r} under {root}")
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(base)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def fingerprint(args, runner, heads_path, code):
    h = hashlib.sha256()
    h.update(json.dumps({"args": args, "runner": runner, "code": code}, sort_keys=True).encode())
    if heads_path:
        h.update(Path(heads_path).read_bytes())
    return h.hexdigest()[:16]


def expand(cfg, repo_root="."):
    runs, seen = [], set()
    for point, matrix_keys in points(cfg):
        params = resolve(cfg, point, matrix_keys)
        if params["run_id"] in seen:
            raise ConfigError(f"duplicate run id {params['run_id']}; add a 'tag' to distinguish runs")
        seen.add(params["run_id"])
        heads_path, source = check_heads(params, repo_root)
        params["head_source"] = source
        params["heads_path"] = heads_path
        runs.append(params)
    return runs
