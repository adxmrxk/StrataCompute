#!/usr/bin/env python3
"""Download one release artifact with verified TLS, atomically."""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: download_with_python.py URL DESTINATION", file=sys.stderr)
        return 2

    url, destination = sys.argv[1:]
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="strata-ort-", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(temporary, destination)
        return 0
    except Exception as error:
        print(f"download failed: {error}", file=sys.stderr)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
