"""Blender binary downloader/extractor and environment validator for Colab (Linux x64).

Supports the portable ``.tar.xz`` archives published at download.blender.org.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

DEFAULT_BLENDER_VERSION = "4.2.3"
DEFAULT_INSTALL_DIR = Path("/content/blender")

# Shared libraries the portable Blender Linux build needs on a headless Colab VM.
RUNTIME_APT_PACKAGES = [
    "libx11-6",
    "libxi6",
    "libxxf86vm1",
    "libxfixes3",
    "libxrender1",
    "libxkbcommon0",
    "libxext6",
    "libsm6",
    "libice6",
    "libgl1",
]

_VERSION_RE = re.compile(r"Blender (\d+\.\d+\.\d+)")


def blender_archive_url(version: str = DEFAULT_BLENDER_VERSION) -> str:
    """Return the download URL for a portable Linux x64 Blender archive.

    ``4.2.3`` -> ``https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz``
    """
    parts = version.split(".")
    if len(parts) < 2 or not all(p.isdigit() for p in parts[:2]):
        raise ValueError(f"Invalid Blender version: {version!r}")
    major_minor = f"{parts[0]}.{parts[1]}"
    return (
        f"https://download.blender.org/release/Blender{major_minor}/"
        f"blender-{version}-linux-x64.tar.xz"
    )


def find_blender_binary(install_dir: Path) -> Optional[Path]:
    """Locate the ``blender`` executable inside an install directory."""
    install_dir = Path(install_dir)
    candidates = [install_dir / "blender", *install_dir.glob("*/blender")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def validate_blender(binary: Path, timeout: int = 180) -> str:
    """Run ``blender --version`` and return the version string (e.g. ``4.2.3``)."""
    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    match = _VERSION_RE.search(result.stdout or "")
    if result.returncode != 0 or not match:
        raise RuntimeError(
            f"Blender binary at {binary} failed validation "
            f"(exit {result.returncode}): {(result.stderr or result.stdout)[:500]}"
        )
    return match.group(1)


def _download(url: str, dest: Path, progress: Callable[[str], None]) -> None:
    progress(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as fh:
        total = int(response.headers.get("Content-Length", 0) or 0)
        done = 0
        next_report = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if done >= next_report:
                pct = f" ({done * 100 // total}%)" if total else ""
                progress(f"  {done / 1e6:.0f} MB{pct}")
                next_report = done + 100 * 1024 * 1024


def _extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, mode="r:xz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")  # Python 3.12+/backports: blocks unsafe paths
        else:
            tar.extractall(dest)


def install_runtime_libs(progress: Callable[[str], None] = print) -> None:
    """Install the apt packages Blender needs. Requires root (true on Colab)."""
    progress("Installing runtime libraries with apt-get")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", *RUNTIME_APT_PACKAGES],
        check=True,
    )


def ensure_blender(
    version: str = DEFAULT_BLENDER_VERSION,
    install_dir: Path = DEFAULT_INSTALL_DIR,
    progress: Callable[[str], None] = print,
) -> Path:
    """Return a working Blender binary, downloading and extracting it if needed."""
    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    existing = find_blender_binary(install_dir)
    if existing is not None:
        try:
            found = validate_blender(existing)
            progress(f"Blender {found} already installed at {existing}")
            return existing
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            progress(f"Existing install is unusable, reinstalling: {exc}")
            shutil.rmtree(install_dir, ignore_errors=True)
            install_dir.mkdir(parents=True, exist_ok=True)

    archive = install_dir / "blender.tar.xz"
    _download(blender_archive_url(version), archive, progress)
    progress("Extracting archive (this takes a minute)")
    _extract(archive, install_dir)
    archive.unlink(missing_ok=True)

    binary = find_blender_binary(install_dir)
    if binary is None:
        raise RuntimeError(f"Blender executable not found after extracting to {install_dir}")
    os.chmod(binary, 0o755)
    found = validate_blender(binary)
    progress(f"Blender {found} ready at {binary}")
    return binary


def nvidia_gpu_summary() -> str:
    """Return a one-line description of the attached NVIDIA GPU, or a warning."""
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found - no NVIDIA GPU runtime detected"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"nvidia-smi failed: {result.stderr.strip()[:200]}"
    return result.stdout.strip() or "No GPU reported by nvidia-smi"


def check_environment(blender_bin: Optional[Path] = None) -> dict:
    """Summarise whether this machine can run GPU renders. Used by app startup and the notebook."""
    report = {"gpu": nvidia_gpu_summary(), "blender": None, "blender_version": None}
    if blender_bin is None and os.environ.get("BLENDER_BIN"):
        blender_bin = Path(os.environ["BLENDER_BIN"])
    if blender_bin is not None and Path(blender_bin).is_file():
        report["blender"] = str(blender_bin)
        try:
            report["blender_version"] = validate_blender(Path(blender_bin))
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            report["blender_version"] = f"error: {exc}"
    return report
