"""Small reproducibility record for qualitative demonstrations."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def write_provenance(args):
    root = Path(__file__).resolve().parents[2]
    files = {}
    for path in sorted((args.asset_root / args.env).rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                files[str(path.relative_to(args.asset_root))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "gymnasium",
            "isaacsim",
            "isaaclab",
            "isaaclab-rl",
            "uwlab",
            "rsl-rl-lib",
        ]
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    code = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src/resync_isaac").rglob("*.py"))
    }
    data = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "python": platform.python_version(),
        "versions": versions,
        "commit": commit,
        "demo_source_sha256": code,
        "assets_sha256": files,
    }
    (args.output / "provenance.json").write_text(json.dumps(data, indent=2) + "\n")
