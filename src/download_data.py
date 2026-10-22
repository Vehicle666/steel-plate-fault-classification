"""Tải dataset Steel Plates Faults từ UCI ML Repository.

Nguồn: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
Chạy: python -m src.download_data [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ZIP_PATH = RAW_DIR / "steel+plates+faults.zip"
EXTRACT_DIR = RAW_DIR / "steel_plates_faults"

EXPECTED_FILES = ["Faults.NNA", "Faults27x7_var"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if force and EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)

    if not ZIP_PATH.exists() or force:
        print(f"Đang tải {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
    else:
        print(f"Đã có sẵn {ZIP_PATH.name}, bỏ qua tải.")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)

    missing = [name for name in EXPECTED_FILES if not (EXTRACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu file sau khi giải nén: {missing}")

    n_rows = sum(1 for _ in (EXTRACT_DIR / "Faults.NNA").open(encoding="utf-8"))
    print(f"Xong. {EXTRACT_DIR} | {n_rows} dòng dữ liệu")
    print(f"SHA-256(zip): {sha256(ZIP_PATH)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Tải lại và giải nén đè")
    download(parser.parse_args().force)
