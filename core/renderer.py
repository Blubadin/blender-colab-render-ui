"""Headless Blender CLI wrapper.

* :func:`build_command` / :func:`build_env` create the exact argv and environment (no shell).
* :class:`RenderRunner` runs the process, streams stdout line by line, and supports cancel.
* Parsing helpers turn Blender log lines into progress and saved-file events.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Mapping, Optional, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPU_SCRIPT = PROJECT_ROOT / "core" / "gpu_setup.py"

ENGINES = ("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
FORMATS = ("PNG", "JPEG", "OPEN_EXR")
GPU_BACKENDS = ("OPTIX", "CUDA")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")  # EXR is not displayable in the browser
FRAME_PATTERN = "frame_####"  # Blender replaces #### with the zero-padded frame number

_FORMAT_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "OPEN_EXR": ".exr"}


@dataclass
class RenderJob:
    blend_file: str
    output_dir: str
    engine: str = "CYCLES"
    output_format: str = "PNG"
    frame: Optional[int] = None  # set for a single-frame render
    start: Optional[int] = None  # animation range
    end: Optional[int] = None
    step: int = 1
    samples: int = 0  # 0 = scene default
    resolution_pct: int = 100
    tile_size: int = 0  # 0 = Blender default
    gpu_backend: str = "OPTIX"

    @property
    def is_animation(self) -> bool:
        return self.frame is None

    def validate(self) -> None:
        """Raise ``ValueError`` for anything Blender would reject or that is unsafe."""
        if not str(self.blend_file).lower().endswith(".blend"):
            raise ValueError("Scene file must end with .blend")
        if not self.output_dir:
            raise ValueError("Output directory is required")
        if self.engine not in ENGINES:
            raise ValueError(f"Unsupported engine {self.engine!r}; expected one of {ENGINES}")
        if self.output_format not in FORMATS:
            raise ValueError(f"Unsupported format {self.output_format!r}; expected one of {FORMATS}")
        if self.gpu_backend not in GPU_BACKENDS:
            raise ValueError(f"Unsupported GPU backend {self.gpu_backend!r}")
        if self.samples < 0:
            raise ValueError("Samples must be 0 (scene default) or a positive integer")
        if not 1 <= self.resolution_pct <= 400:
            raise ValueError("Resolution percentage must be between 1 and 400")
        if self.tile_size < 0:
            raise ValueError("Tile size must be >= 0")
        if self.is_animation:
            if self.start is None or self.end is None:
                raise ValueError("Animation renders need start and end frames")
            if self.start < 0 or self.end < self.start:
                raise ValueError(f"Invalid frame range {self.start}-{self.end}")
            if self.step < 1:
                raise ValueError("Frame step must be >= 1")
        elif self.frame < 0:
            raise ValueError("Frame number must be >= 0")


def build_command(blender_bin: str | os.PathLike, job: RenderJob, gpu_script: Path = GPU_SCRIPT) -> List[str]:
    """Return the Blender argv for a job. Every element is a separate argument, so paths
    with spaces or quotes are passed through unchanged."""
    job.validate()
    output_pattern = str(Path(job.output_dir) / FRAME_PATTERN)
    argv = [
        str(blender_bin),
        "-b", str(job.blend_file),
        "-P", str(gpu_script),
        "-E", job.engine,
        "-o", output_pattern,
        "-F", job.output_format,
    ]
    if job.is_animation:
        argv += ["-s", str(job.start), "-e", str(job.end), "-j", str(job.step), "-a"]
    else:
        argv += ["-f", str(job.frame)]
    return argv


def build_env(job: RenderJob, base_env: Optional[Mapping[str, str]] = None) -> dict:
    """Environment for the Blender process; read by ``core/gpu_setup.py``."""
    env = dict(base_env if base_env is not None else os.environ)
    env["BLENDER_GPU_BACKEND"] = job.gpu_backend
    env["BLENDER_SAMPLES"] = str(job.samples)
    env["BLENDER_RESOLUTION_PCT"] = str(job.resolution_pct)
    env["BLENDER_TILE_SIZE"] = str(job.tile_size)
    return env


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_FRAME_RE = re.compile(r"Fra:(?P<frame>\d+)")
_TIME_RE = re.compile(r"Time:(?P<elapsed>[\d:.]+)")
_REMAINING_RE = re.compile(r"Remaining:(?P<remaining>[\d:.]+|-)")
_MEM_RE = re.compile(r"Mem:(?P<mem>[\d.]+[KMGkmg]?)")
_SAMPLE_RE = re.compile(r"Sample (?P<sample>\d+)/(?P<total>\d+)")
_SAVED_RE = re.compile(r"Saved:\s*'(?P<path>[^']+)'")


@dataclass
class ProgressInfo:
    frame: int
    elapsed: Optional[str] = None
    remaining: Optional[str] = None
    memory: Optional[str] = None
    sample: Optional[int] = None
    total_samples: Optional[int] = None

    @property
    def percent(self) -> Optional[float]:
        if self.sample is None or not self.total_samples:
            return None
        return round(100.0 * self.sample / self.total_samples, 1)


def parse_progress_line(line: str) -> Optional[ProgressInfo]:
    """Parse a Blender progress line such as
    ``Fra:3 Mem:45.67M (Peak 50.00M) | Time:00:01.23 | Remaining:00:04.00 | Sample 64/128``.
    Returns ``None`` when the line is not a progress line."""
    frame = _FRAME_RE.search(line)
    if not frame:
        return None
    sample = _SAMPLE_RE.search(line)
    time_m = _TIME_RE.search(line)
    remaining = _REMAINING_RE.search(line)
    mem = _MEM_RE.search(line)
    return ProgressInfo(
        frame=int(frame.group("frame")),
        elapsed=time_m.group("elapsed") if time_m else None,
        remaining=remaining.group("remaining") if remaining else None,
        memory=mem.group("mem") if mem else None,
        sample=int(sample.group("sample")) if sample else None,
        total_samples=int(sample.group("total")) if sample else None,
    )


def parse_saved_path(line: str) -> Optional[str]:
    """Return the file path from a ``Saved: '...'`` line, or ``None``."""
    match = _SAVED_RE.search(line)
    return match.group("path") if match else None


def iter_output_lines(stream: TextIO) -> Iterator[str]:
    """Yield lines from a process stream. Blender redraws progress with carriage
    returns, so ``\\r`` and ``\\n`` both end a line."""
    buf: List[str] = []
    while True:
        ch = stream.read(1)
        if ch == "":
            break
        if ch in "\r\n":
            if buf:
                yield "".join(buf)
                buf = []
        else:
            buf.append(ch)
    if buf:
        yield "".join(buf)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    status: str  # "done", "failed" or "cancelled"
    returncode: Optional[int] = None
    saved_files: List[str] = field(default_factory=list)
    error: Optional[str] = None


class RenderRunner:
    """Runs one command, streams its output and can terminate it from another thread."""

    def __init__(
        self,
        argv: Iterable[str],
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
        log_lines: int = 500,
    ) -> None:
        self.argv = list(argv)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.log = deque(maxlen=log_lines)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def run(
        self,
        on_log: Callable[[str], None] = lambda _line: None,
        on_progress: Callable[[ProgressInfo], None] = lambda _p: None,
        on_saved: Callable[[str], None] = lambda _path: None,
    ) -> RenderResult:
        saved: List[str] = []
        with self._lock:
            if self._cancelled:
                return RenderResult(status="cancelled")
            try:
                self._proc = subprocess.Popen(
                    self.argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=self.env,
                    cwd=self.cwd,
                )
            except OSError as exc:
                return RenderResult(status="failed", error=f"Could not start Blender: {exc}")
            proc = self._proc

        for line in iter_output_lines(proc.stdout):
            self.log.append(line)
            on_log(line)
            progress = parse_progress_line(line)
            if progress is not None:
                on_progress(progress)
            path = parse_saved_path(line)
            if path is not None:
                saved.append(path)
                on_saved(path)

        returncode = proc.wait()
        if self._cancelled:
            return RenderResult("cancelled", returncode, saved)
        if returncode == 0:
            return RenderResult("done", returncode, saved)
        return RenderResult("failed", returncode, saved, error=f"Blender exited with code {returncode}")

    def cancel(self, grace_seconds: float = 10.0) -> None:
        """Terminate the running process. Safe to call before, during or after ``run``."""
        with self._lock:
            self._cancelled = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def list_rendered_images(output_dir: str | os.PathLike, limit: int = 200) -> List[Path]:
    """Newest-first list of displayable images under ``output_dir`` (recursive)."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def create_zip(files: Iterable[str | os.PathLike], zip_path: str | os.PathLike) -> Path:
    """Write the given files into a ZIP archive (stored under their base names)."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=Path(f).name)
    return zip_path


def output_suffix(output_format: str) -> str:
    return _FORMAT_SUFFIX[output_format]
