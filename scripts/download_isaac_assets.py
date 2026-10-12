"""Download the optional Isaac demo assets without changing reproduction assets."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.as_posix().startswith("inference_assets/isaac/")
    ):
        raise ValueError(f"Unexpected archive path: {name}")
    target = root / relative
    if not target.resolve().is_relative_to((root / "inference_assets/isaac").resolve()):
        raise ValueError(f"Archive path escapes asset directory: {name}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Extraction destination (for clean-install checks)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((ROOT / "release/isaac/assets_manifest.json").read_text())
    cache = root / ".release-cache/isaac"
    cache.mkdir(parents=True, exist_ok=True)
    for archive in manifest["archives"]:
        if not args.verify_only:
            path = cache / archive["filename"]
            if not path.exists():
                import gdown

                for part in archive["parts"]:
                    part_path = cache / part["filename"]
                    if not part_path.exists():
                        tmp = part_path.with_suffix(".partial")
                        if not gdown.download(
                            id=part["drive_file_id"],
                            output=str(tmp),
                            quiet=False,
                            use_cookies=False,
                        ):
                            raise RuntimeError(f'Download failed: {part["filename"]}')
                        tmp.replace(part_path)
                    if sha256(part_path) != part["sha256"]:
                        raise ValueError(
                            f"Checksum mismatch: {part_path}; remove it and retry"
                        )
                with path.with_suffix(".partial").open("wb") as dst:
                    for part in archive["parts"]:
                        with (cache / part["filename"]).open("rb") as src:
                            shutil.copyfileobj(src, dst)
                path.with_suffix(".partial").replace(path)
            if sha256(path) != archive["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            with zipfile.ZipFile(path) as z:
                for member in z.infolist():
                    target = checked_path(root, member.filename)
                    if member.is_dir():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        entries = json.loads((ROOT / archive["contents_manifest"]).read_text())
        for entry in entries:
            path = checked_path(root, entry["path"])
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Missing or changed asset: {path}")
        print(f'PASS {archive["domain"]}: {len(entries)} files', flush=True)


if __name__ == "__main__":
    main()
