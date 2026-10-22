"""Download the preprocessed SONICOM arrays and GeoAtt checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


PAPER_REPO_ID = "xingyuchen-8/GeoAtt-HRTF"
PAPER_REVISION = "afe312913ae2f793939f1484d46e69f0bb70148c"
# LFS SHA256 values from the pinned Hub model API (verified against file bytes).
PAPER_SHA256 = {
    "dataset/H_log.npy": "c10924b29b33659e5c7dbb08198067c293d03833a5ab0268df85c7bc5bf52e9a",
    "dataset/locs.npy": "d9f0439d35ea332a199624df4028d454062d6395ad65b0223b0c82e3ffbfb621",
    "model/geoatt_final_ep440.pt": "99205b4529c223c4186be370b9939b9a20fd570848d5c3c73ee4849db094d23e",
}


ASSETS = {
    "dataset/H_log.npy": Path("data/sonicom/H_log.npy"),
    "dataset/locs.npy": Path("data/sonicom/locs.npy"),
    "model/geoatt_final_ep440.pt": Path("weights/geoatt_final_ep440.pt"),
}


def resolve_assets(repo_id: str, revision: str) -> tuple[str, dict[str, str]]:
    if (repo_id, revision) == (PAPER_REPO_ID, PAPER_REVISION):
        return PAPER_REVISION, PAPER_SHA256

    # Resolve a branch/tag once so all three files come from the same commit.
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Hub did not return an immutable commit SHA")
    files = {file.rfilename: file for file in info.siblings or []}
    checksums = {}
    for source in ASSETS:
        file = files.get(source)
        digest = file.lfs.sha256 if file is not None and file.lfs else None
        if not digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Missing LFS SHA256 for {source} at {repo_id}@{info.sha}")
        checksums[source] = digest
    return info.sha, checksums


def copy_verified(cached: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Verify the copied bytes before atomically replacing any existing output.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(cached, temporary_path)
        digest = hashlib.sha256()
        with temporary_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"SHA256 mismatch for {target}: expected {expected_sha256}, got {actual}. "
                f"Output was not replaced; remove the corrupt cached file {cached.resolve()} and retry."
            )
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=PAPER_REPO_ID)
    parser.add_argument("--revision", default=PAPER_REVISION,
                        help="Hub commit, tag, or branch (default: pinned paper release)")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    revision, checksums = resolve_assets(args.repo_id, args.revision)
    print(f"Assets: {args.repo_id}@{revision}")
    for source, destination in ASSETS.items():
        cached = Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=source,
            repo_type="model",
            revision=revision,
            cache_dir=cache_dir,
        ))
        target = output_dir / destination
        copy_verified(cached, target, checksums[source])
        print(f"{source} -> {target} (SHA256 verified)")


if __name__ == "__main__":
    main()
