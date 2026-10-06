import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from experiments import config as cfgmod

REPO = Path(__file__).resolve().parents[1]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def command(runner, args):
    base = [sys.executable, runner] if runner.endswith(".py") else [sys.executable, "-m", runner]
    return base + args


def result_subdir(params):
    return f"{cfgmod.short(params['model_name'])}_{params['lang']}"


def plan(cfg, root, runner):
    runs = cfgmod.expand(cfg, REPO)
    code = cfgmod.code_hash(runner, REPO)
    out = []
    for params in runs:
        run_dir = Path(root) / cfg["name"] / params["run_id"]
        if params["head_source"] == "skip":
            out.append(dict(params=params, run_dir=run_dir, skip=True))
            continue
        base_args = cfgmod.cli_args(params, params["heads_path"])
        args = base_args + [f"--out={run_dir}"]
        fp = cfgmod.fingerprint(base_args, runner, params["heads_path"], code)
        out.append(dict(params=params, run_dir=run_dir, args=args, fingerprint=fp, code_hash=code,
                        command=command(runner, args), skip=False))
    return out


def read_manifest(run_dir):
    path = Path(run_dir) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_manifest(run_dir, manifest):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def is_cached(item):
    manifest = read_manifest(item["run_dir"])
    return bool(manifest and manifest.get("status") == "done" and manifest.get("fingerprint") == item["fingerprint"]
                and (Path(item["run_dir"]) / manifest["result_subdir"] / "summary.json").exists())


def execute(item):
    params, run_dir = item["params"], Path(item["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": params["run_id"],
        "params": {k: v for k, v in params.items() if k not in ("heads_path",)},
        "head_source": params["head_source"],
        "heads_path": params["heads_path"],
        "fingerprint": item["fingerprint"],
        "code_hash": item["code_hash"],
        "command": item["command"],
        "result_subdir": result_subdir(params),
        "status": "running",
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(run_dir, manifest)
    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(run_dir / "run.log", "w") as log:
        proc = subprocess.Popen(item["command"], cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    done = rc == 0 and (run_dir / manifest["result_subdir"] / "summary.json").exists()
    manifest.update(status="done" if done else "failed", returncode=rc, seconds=round(time.time() - t0, 1),
                    finished=datetime.now(timezone.utc).isoformat())
    write_manifest(run_dir, manifest)
    return manifest["status"]


def select(items, only):
    if not only:
        return items
    return [i for i in items if any(token in i["params"]["run_id"] for token in only)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--root", default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args(argv)

    cfg = cfgmod.load(a.config)
    root = a.root or cfg.get("output_root", "results/experiments")
    runner = cfg.get("runner", "intervention.run_pilot")
    items = select(plan(cfg, root, runner), a.only)

    if a.list or a.dry_run:
        for i in items:
            status = "skip (no heads file)" if i["skip"] else ("cached" if is_cached(i) else "pending")
            print(f"{i['params']['run_id']:40s} {i['params']['head_source']:8s} {status}")
            if a.dry_run and not i["skip"]:
                print("    " + " ".join(i["command"]))
        return 0

    statuses = {}
    for n, i in enumerate(items, 1):
        rid = i["params"]["run_id"]
        if i["skip"]:
            statuses[rid] = "skipped"
            print(f"[{n}/{len(items)}] {rid}: skipped, heads file missing")
            continue
        if not a.force and is_cached(i):
            statuses[rid] = "cached"
            print(f"[{n}/{len(items)}] {rid}: cached")
            continue
        print(f"[{n}/{len(items)}] {rid}: running", flush=True)
        statuses[rid] = execute(i)
        print(f"[{n}/{len(items)}] {rid}: {statuses[rid]}", flush=True)
        if statuses[rid] == "failed" and a.stop_on_error:
            break

    summary_path = Path(root) / cfg["name"] / "status.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(statuses, indent=2))
    failed = [r for r, s in statuses.items() if s == "failed"]
    print(json.dumps({s: sum(v == s for v in statuses.values()) for s in set(statuses.values())}))
    if a.aggregate:
        from experiments import aggregate
        aggregate.main([str(Path(root) / cfg["name"])])
    if failed:
        print("failed runs:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
