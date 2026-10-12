"""Build deterministic, checksum-verified Isaac release archives (maintainer tool)."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    cache = ROOT / ".release-cache/isaac"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "archives": []}
    for domain, folders in [
        ("isaac_runtime", ["runtime", "simulation"]),
        ("isaac_flipped_stacking", ["flipped_stacking"]),
        ("isaac_corner_stacking", ["corner_stacking"]),
    ]:
        files = []
        for folder in folders:
            files += [
                p
                for p in (ROOT / "inference_assets/isaac" / folder).rglob("*")
                if p.is_file()
                and not any(
                    x in p.parts for x in ["__pycache__", ".git", ".pytest_cache"]
                )
                and not any(x.endswith(".egg-info") for x in p.parts)
                and p.suffix not in (".pyc", ".bak")
            ]
        files = sorted(files)
        entries = [
            {
                "path": str(p.relative_to(ROOT)),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in files
        ]
        contents = ROOT / "release/isaac" / f"{domain}_contents.json"
        contents.write_text(json.dumps(entries, indent=2) + "\n")
        archive = cache / f"resync_{domain}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in files:
                info = zipfile.ZipInfo(
                    str(p.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, p.read_bytes())
        parts = []
        with archive.open("rb") as f:
            i = 1
            while data := f.read(90 * 1024 * 1024):
                part = archive.with_suffix(f".zip.part{i:03d}")
                part.write_bytes(data)
                parts.append(
                    {"filename": part.name, "bytes": len(data), "sha256": digest(part)}
                )
                i += 1
        manifest["archives"].append(
            {
                "domain": domain,
                "filename": archive.name,
                "sha256": digest(archive),
                "bytes": archive.stat().st_size,
                "file_count": len(files),
                "contents_manifest": str(contents.relative_to(ROOT)),
                "parts": parts,
            }
        )
        print(domain, len(files), archive.stat().st_size, flush=True)
    target = ROOT / "release/isaac/assets_manifest.json"
    if target.exists():
        old = json.loads(target.read_text())
        by_hash = {p["sha256"]: p for a in old["archives"] for p in a["parts"]}
        for a in manifest["archives"]:
            for p in a["parts"]:
                for key in ["drive_file_id", "drive_url"]:
                    if key in by_hash.get(p["sha256"], {}):
                        p[key] = by_hash[p["sha256"]][key]
    target.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
