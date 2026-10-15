"""Provenance: every script run writes a manifest so results can be traced and reproduced.

Each call to `record(...)`:
  - writes <out_dir>/manifest.json (or <out_path>.manifest.json for a single file)
  - appends one line to logs/runs.jsonl (the canonical run log)

Fields: run id, UTC time, script + argv, git commit + dirty flag + diff hash, config,
model id + resolved Hugging Face revision, data sources with sha256, machine, Python and
package versions, device, outputs, wall time, and any result summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "runs.jsonl"
_T0 = time.time()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _git_state():
    diff = _git("diff", "HEAD")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(diff) or bool(_git("status", "--porcelain", "--untracked-files=no")),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


# Captured when the script starts: this is the code that actually ran.
_GIT_AT_START = _git_state()


def sha256(path, limit_mb: int = 512):
    p = Path(path)
    if not p.exists() or p.is_dir():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "sae_lens", "sklearn", "numpy", "pandas", "fastmcp"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def _model_revision(model_id):
    """Resolve the exact model snapshot (commit hash) from the local HF cache or a local path."""
    if not model_id:
        return None
    p = Path(model_id)
    if p.exists():
        return {"local_path": str(p), "snapshot": p.name}
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id:
                revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                return {"repo": model_id, "revision": revs[0].commit_hash if revs else None}
    except Exception:
        pass
    return {"repo": model_id, "revision": None}


def _machine(device=None):
    m = {"hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine()}
    if platform.system() == "Darwin":
        try:
            m["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            m["ram_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30)
        except Exception:
            pass
    if device:
        m["device"] = device
    return m


def record(out, config=None, model_id=None, device=None, data=None, outputs=None, results=None, note=None):
    """Write a manifest next to `out` (dir or file) and append to logs/runs.jsonl. Returns the manifest."""
    out = Path(out)
    cfg = asdict(config) if is_dataclass(config) else (config or {})
    man = {
        "run_id": RUN_ID,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(sys.argv[0]).name,
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "git": _GIT_AT_START,
        "git_at_end": _git_state(),
        "config": cfg,
        "model": _model_revision(model_id or cfg.get("model_id")),
        "data": {str(k): {"path": str(v), "sha256": sha256(v)} for k, v in (data or {}).items()},
        "machine": _machine(device),
        "versions": _versions(),
        "outputs": [str(o) for o in (outputs or [out])],
        "wall_seconds": round(time.time() - _T0, 1),
        "results": results,
        "note": note,
    }
    dest = out / "manifest.json" if (out.is_dir() or not out.suffix) else out.with_suffix(out.suffix + ".manifest.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(man, indent=2, default=str))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(man, default=str) + "\n")
    print(f"[provenance] run {RUN_ID} -> {dest} (git {str(man['git']['commit'])[:7]}"
          f"{' dirty' if man['git']['dirty'] else ''})")
    return man
