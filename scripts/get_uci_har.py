"""Fetch and safely extract UCI's Human Activity Recognition dataset.

The source is UCI ML Repository dataset 240, CC BY 4.0.  This script keeps
the 58 MB source archive out of git and rejects unsafe zip member paths.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        if destination not in target.parents and target != destination:
            raise RuntimeError(f"unsafe zip path: {info.filename}")
    archive.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/uci_har"))
    args = parser.parse_args()
    destination = args.destination.resolve()
    marker = destination / "UCI HAR Dataset" / "train" / "X_train.txt"
    if marker.exists():
        print(f"Dataset already present: {destination}")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        archive_path = Path(temporary) / "uci_har.zip"
        print("Downloading UCI HAR dataset (CC BY 4.0)…")
        request = urllib.request.Request(URL, headers={"User-Agent": "StrataCompute-demo/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        if archive_path.stat().st_size < 10_000_000:
            raise RuntimeError("download is unexpectedly small; refusing to extract it")
        staging = Path(temporary) / "extract"
        staging.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, staging)
        # UCI currently wraps the actual dataset zip inside a small outer zip
        # alongside its README.  Support that official packaging without
        # assuming a flat archive.
        nested_archives = list(staging.glob("*.zip"))
        if nested_archives:
            with zipfile.ZipFile(nested_archives[0]) as nested:
                safe_extract(nested, staging)
        extracted = staging / "UCI HAR Dataset"
        if not (extracted / "test" / "X_test.txt").exists():
            raise RuntimeError("download did not contain the expected UCI HAR files")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staging), str(destination))
    print(f"Downloaded and verified: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # clear command-line failure, no partial dataset
        print(f"get_uci_har: {error}", file=sys.stderr)
        raise SystemExit(1)
