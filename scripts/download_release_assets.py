"""Download and verify simulation assets and pretrained models from Google Drive."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=[
            "simulation_assets",
            "blocked_stacking",
            "cluttered_drawer",
            "icy_transport",
            "cluttered_room",
        ],
        choices=[
            "simulation_assets",
            "blocked_stacking",
            "cluttered_drawer",
            "icy_transport",
            "cluttered_room",
        ],
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify already extracted files without downloading",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release/assets_manifest.json").read_text())
    source_paths = set(
        json.loads((root / "release/source_manifest.json").read_text())["files"]
    )
    cache = root / ".release-cache"
    cache.mkdir(exist_ok=True)
    for archive in manifest["archives"]:
        if archive["domain"] not in args.domains:
            continue
        metadata = root / archive.get(
            "contents_manifest",
            f"release/asset_contents/{archive['domain']}_seed0.json",
        )
        if not args.verify_only:
            path = cache / archive["filename"]
            if not path.exists():
                import gdown

                for part in archive["parts"]:
                    part_path = cache / part["filename"]
                    if not part_path.exists():
                        if not part.get("drive_file_id"):
                            raise SystemExit(
                                f"No published download ID for {part['filename']}"
                            )
                        temporary = part_path.with_suffix(part_path.suffix + ".partial")
                        result = gdown.download(
                            id=part["drive_file_id"],
                            output=str(temporary),
                            quiet=False,
                            use_cookies=False,
                        )
                        if not result:
                            raise SystemExit(f"Download failed: {part['filename']}")
                        temporary.replace(part_path)
                    with part_path.open("rb") as stream:
                        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                    if checksum != part["sha256"]:
                        raise SystemExit(
                            f"Checksum mismatch: {part_path}. Remove this part and retry."
                        )
                temporary = path.with_suffix(".zip.partial")
                with temporary.open("wb") as destination:
                    for part in archive["parts"]:
                        with (cache / part["filename"]).open("rb") as source:
                            shutil.copyfileobj(source, destination)
                temporary.replace(path)
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != archive["sha256"]:
                raise SystemExit(
                    f"Checksum mismatch: {path}. Remove this archive and retry."
                )
            print(f"Verified {path.name}; extracting...", flush=True)
            with zipfile.ZipFile(path) as zipped:
                for member in zipped.infolist():
                    relative = Path(member.filename)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or (
                            relative.as_posix() in source_paths
                            and not (
                                archive["domain"] == "simulation_assets"
                                and relative.as_posix().startswith(
                                    "src/skill_refactor/assets/"
                                )
                            )
                        )
                    ):
                        raise SystemExit(
                            f"Unsafe or source-overwriting archive member: {relative}"
                        )
                    target = root / relative
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        zipped.open(member) as source,
                        target.open("wb") as destination,
                    ):
                        shutil.copyfileobj(source, destination)
        if not metadata.exists():
            raise SystemExit(
                f"Asset manifest missing: {metadata}; download this domain first."
            )
        entries = json.loads(metadata.read_text())
        for entry in entries:
            path = root / entry["path"]
            if not path.is_file():
                raise SystemExit(f"Missing asset: {path}")
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != entry["sha256"]:
                raise SystemExit(f"Asset checksum mismatch: {path}")
        print(f"PASS: {archive['domain']}, {len(entries)} files verified")


if __name__ == "__main__":
    main()
