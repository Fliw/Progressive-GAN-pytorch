#!/usr/bin/env python3
"""
download_acne_dataset.py
------------------------
Meng‑automation:

1. Memastikan paket `kaggle` ter‑install.
2. Mem‑copy kredensial `kaggle.json` ke ~/.kaggle dan mengatur permission 600.
3. Meng‑download dataset 'rafliindrawan/acne-dataset'.
4. Mengekstrak arsip ZIP ke ./merged_dataset/Acne/data

Opsi dapat diatur via CLI – lihat `python download_acne_dataset.py -h`.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def ensure_kaggle_installed() -> None:
    """Pasang library Kaggle kalau belum ada."""
    try:
        import kaggle  # noqa: F401  pylint: disable=unused-import
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])


def copy_credentials(src: Path) -> None:
    """Salin kaggle.json ke ~/.kaggle dengan permission 600."""
    if not src.exists():
        sys.exit(f"[ERROR] File kredensial tidak ditemukan: {src}")

    target_dir = Path.home() / ".kaggle"
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / "kaggle.json"
    shutil.copy(src, dest)
    dest.chmod(0o600)
    print(f"[INFO] Kaggle credential tersalin ke {dest}")


def download_dataset(slug: str, download_dir: Path) -> Path:
    """Unduh dataset via Kaggle CLI; kembalikan path ZIP‑nya."""
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Mengunduh {slug} ke {download_dir}")
    subprocess.check_call(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(download_dir)]
    )
    zip_name = slug.split("/")[-1] + ".zip"
    zip_path = download_dir / zip_name
    if not zip_path.exists():
        sys.exit("[ERROR] ZIP tidak ditemukan setelah download ‑‑ cek slug & kredensial.")
    return zip_path


def unzip(zip_path: Path, dest_dir: Path) -> None:
    """Ekstrak ZIP ke dest_dir."""
    print(f"[INFO] Mengekstrak {zip_path} ke {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print("[INFO] Ekstraksi selesai")


def main() -> None:
    parser = argparse.ArgumentParser(description="Downloader Acne Dataset Kaggle")
    parser.add_argument(
        "--cred",
        type=Path,
        default=Path("kaggle.json"),
        help="Lokasi file kaggle.json (default: ./kaggle.json)",
    )
    parser.add_argument(
        "--dataset",
        default="rafliindrawan/acne-dataset",
        help="Slug dataset Kaggle",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("merged_dataset/Acne/data"),
        help="Folder akhir tempat ekstraksi dataset",
    )
    args = parser.parse_args()

    ensure_kaggle_installed()
    copy_credentials(args.cred.expanduser())
    zip_path = download_dataset(args.dataset, args.out.parent)
    unzip(zip_path, args.out)


if __name__ == "__main__":
    main()
