"""Verify restored source files against the upstream snapshot."""

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release/source_manifest.json").read_text())
    exceptions = {"README.md", ".gitignore", "TEST_README.md", "CLAUDE.md"}
    failures = []
    checked = 0
    for name, record in manifest["files"].items():
        if name in exceptions:
            continue
        path = root / name
        if not path.is_file():
            failures.append(f"Missing: {name}; download simulation assets first.")
            continue
        data = path.read_bytes()
        digest = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if digest != record["git_blob"]:
            failures.append(f"Changed: {name}")
        if record["mode"] == "100755" and not path.stat().st_mode & 0o111:
            failures.append(f"Lost executable mode: {name}")
        checked += 1
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {checked} files match source main {manifest['source_commit']}")
    print("Release documentation and ignore rules are excluded.")


if __name__ == "__main__":
    main()
