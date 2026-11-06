"""Gradio Web UI for the Blender Colab render farm.

Run on Colab with ``python app.py``. Set ``GRADIO_SHARE=0`` to disable the public share link.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

from core.installer import DEFAULT_BLENDER_VERSION, check_environment, ensure_blender
from core.renderer import (
    ENGINES,
    FORMATS,
    GPU_BACKENDS,
    ProgressInfo,
    RenderJob,
    RenderResult,
    RenderRunner,
    build_command,
    build_env,
    create_zip,
    list_rendered_images,
)

DRIVE_ROOT = Path("/content/drive/MyDrive")
DEFAULT_OUTPUT = DRIVE_ROOT / "Blender_Renders" if DRIVE_ROOT.is_dir() else Path("/content/Blender_Renders")

_state_lock = threading.Lock()
_active_runner: Optional[RenderRunner] = None
_blender_bin: Optional[Path] = None


def scan_drive_blends(limit: int = 300) -> list[str]:
    """List .blend files under the mounted Google Drive (empty if not mounted)."""
    if not DRIVE_ROOT.is_dir():
        return []
    return sorted(str(p) for p in DRIVE_ROOT.rglob("*.blend"))[:limit]


def _resolve_blender() -> Path:
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    return ensure_blender(os.environ.get("BLENDER_VERSION", DEFAULT_BLENDER_VERSION))


def _format_progress(p: ProgressInfo) -> str:
    parts = [f"Frame {p.frame}"]
    if p.percent is not None:
        parts.append(f"samples {p.sample}/{p.total_samples} ({p.percent}%)")
    if p.remaining:
        parts.append(f"remaining {p.remaining}")
    if p.memory:
        parts.append(f"mem {p.memory}")
    return " | ".join(parts)


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _worker(runner: RenderRunner, events: "queue.Queue") -> None:
    global _active_runner
    try:
        result = runner.run(
            on_log=lambda line: events.put(("log", line)),
            on_progress=lambda p: events.put(("progress", p)),
            on_saved=lambda path: events.put(("saved", path)),
        )
    except Exception as exc:  # keep the UI alive even if the runner crashes
        result = RenderResult(status="failed", error=str(exc))
    finally:
        with _state_lock:
            _active_runner = None
    events.put(("done", result))


def _render(
    mode: str,
    upload: Optional[str],
    drive_path: Optional[str],
    engine: str,
    gpu_backend: str,
    resolution: float,
    samples: float,
    tile_size: float,
    output_format: str,
    output_dir: str,
    frame: float,
    start: float,
    end: float,
    step: float,
):
    """Generator behind the Start buttons. Yields (status, log, preview, zip) tuples."""
    global _active_runner

    blend = upload or drive_path
    if not blend:
        yield "Upload a .blend file or choose one from Google Drive.", "", None, None
        return

    stem = Path(blend).stem
    out_root = Path(output_dir or DEFAULT_OUTPUT).expanduser()
    job_dir = out_root / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"

    try:
        job = RenderJob(
            blend_file=str(blend),
            output_dir=str(job_dir),
            engine=engine,
            output_format=output_format,
            frame=int(frame) if mode == "single" else None,
            start=int(start) if mode == "animation" else None,
            end=int(end) if mode == "animation" else None,
            step=int(step) if mode == "animation" else 1,
            samples=int(samples or 0),
            resolution_pct=int(resolution or 100),
            tile_size=int(tile_size or 0),
            gpu_backend=gpu_backend,
        )
        job.validate()
        job_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(_blender_bin, job)
    except (ValueError, OSError) as exc:
        yield f"Cannot start render: {exc}", "", None, None
        return

    with _state_lock:
        busy = _active_runner is not None
        if not busy:
            runner = RenderRunner(command, env=build_env(job))
            _active_runner = runner
    if busy:  # yield outside the lock so cancel_render() can never block on a suspended generator
        yield "A render is already running. Stop it first or wait for it to finish.", "", None, None
        return

    events: queue.Queue = queue.Queue()
    threading.Thread(target=_worker, args=(runner, events), daemon=True).start()

    logs: deque = deque(maxlen=300)
    progress_text = "Starting Blender..."
    preview: Optional[str] = None
    started = time.monotonic()

    while True:
        try:
            kind, payload = events.get(timeout=0.5)
        except queue.Empty:
            kind, payload = None, None

        if kind == "log":
            logs.append(payload)
        elif kind == "progress":
            progress_text = _format_progress(payload)
        elif kind == "saved":
            preview = payload
        elif kind == "done":
            result: RenderResult = payload
            zip_path = None
            images = list_rendered_images(job_dir)
            if images:
                zip_path = str(create_zip(images, job_dir / f"{job_dir.name}.zip"))
            count = len(result.saved_files)
            if result.status == "done":
                status = f"Finished: {count} file(s) saved to `{job_dir}`"
            elif result.status == "cancelled":
                status = f"Cancelled after {count} file(s). Saved files are in `{job_dir}`"
            else:
                status = f"Failed: {result.error}. Check the log below."
            preview_out = preview if preview and Path(preview).suffix.lower() in (".png", ".jpg", ".jpeg") else None
            yield status, "\n".join(logs), preview_out, zip_path
            return

        elapsed = _fmt_elapsed(time.monotonic() - started)
        yield f"**{progress_text}**  \nElapsed {elapsed}", "\n".join(logs), preview, None


def start_single(*args):
    yield from _render("single", *args)


def start_animation(*args):
    yield from _render("animation", *args)


def cancel_render():
    with _state_lock:
        runner = _active_runner
    if runner is None:
        return "Nothing is rendering right now."
    runner.cancel()
    return "Stop requested. Blender is being terminated; frames already written are kept."


def refresh_gallery(output_dir: str):
    return [str(p) for p in list_rendered_images(Path(output_dir or DEFAULT_OUTPUT).expanduser())]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Blender Cloud Render Farm") as demo:
        gr.Markdown(
            "# Blender Cloud Render Farm\n"
            "Upload a `.blend` file or pick one from Google Drive, choose settings, and render on the Colab GPU. "
            "Frames are written to Google Drive as they finish."
        )

        with gr.Row():
            with gr.Column(scale=1):
                blend_upload = gr.File(label="Upload .blend", file_types=[".blend"], type="filepath")
                blend_drive = gr.Dropdown(
                    label="Or pick a .blend from Google Drive",
                    choices=scan_drive_blends(),
                    value=None,
                    allow_custom_value=True,
                )
                refresh_btn = gr.Button("Refresh Drive list")
            with gr.Column(scale=1):
                engine = gr.Dropdown(list(ENGINES), value="CYCLES", label="Render engine")
                gpu_backend = gr.Dropdown(list(GPU_BACKENDS), value="OPTIX", label="GPU compute device")
                resolution = gr.Slider(25, 200, value=100, step=5, label="Resolution scale (%)")
                output_format = gr.Radio(list(FORMATS), value="PNG", label="Output format")
                output_dir = gr.Textbox(value=str(DEFAULT_OUTPUT), label="Output directory")
                with gr.Accordion("Advanced", open=False):
                    samples = gr.Number(value=0, precision=0, minimum=0, label="Samples (0 = scene default)")
                    tile_size = gr.Number(value=0, precision=0, minimum=0, label="Tile size (0 = automatic)")

        common_inputs = [
            blend_upload, blend_drive, engine, gpu_backend, resolution, samples, tile_size,
            output_format, output_dir,
        ]

        with gr.Tabs():
            with gr.Tab("Single Frame Render"):
                frame = gr.Number(value=1, precision=0, minimum=0, label="Frame number")
                single_btn = gr.Button("Start Render", variant="primary")
            with gr.Tab("Animation Render"):
                with gr.Row():
                    start = gr.Number(value=1, precision=0, minimum=0, label="Start frame")
                    end = gr.Number(value=10, precision=0, minimum=0, label="End frame")
                    step = gr.Number(value=1, precision=0, minimum=1, label="Frame step")
                animation_btn = gr.Button("Start Animation Render", variant="primary")
            with gr.Tab("Output Gallery"):
                gallery = gr.Gallery(label="Rendered images in output directory", columns=4, height="auto")
                gallery_btn = gr.Button("Refresh gallery")

        stop_btn = gr.Button("Stop / Cancel Render", variant="stop")
        status = gr.Markdown("Idle")
        log = gr.Textbox(label="Blender log", lines=14, max_lines=14, interactive=False, autoscroll=True)
        preview = gr.Image(label="Latest completed frame", type="filepath", interactive=False)
        zip_out = gr.File(label="Download all frames (ZIP) when finished")

        render_outputs = [status, log, preview, zip_out]
        single_btn.click(start_single, inputs=common_inputs + [frame, start, end, step], outputs=render_outputs)
        animation_btn.click(start_animation, inputs=common_inputs + [frame, start, end, step], outputs=render_outputs)
        stop_btn.click(cancel_render, outputs=[status])
        refresh_btn.click(lambda: gr.update(choices=scan_drive_blends()), outputs=[blend_drive])
        gallery_btn.click(refresh_gallery, inputs=[output_dir], outputs=[gallery])

    return demo


def main() -> None:
    global _blender_bin
    report = check_environment()
    print(f"GPU: {report['gpu']}")
    _blender_bin = _resolve_blender()
    print(f"Blender: {_blender_bin}")
    if "NVIDIA" not in report["gpu"].upper():
        print("WARNING: no NVIDIA GPU detected. Set Colab Runtime > Change runtime type > GPU.")

    demo = build_ui()
    share = os.environ.get("GRADIO_SHARE", "1") != "0"
    demo.queue().launch(share=share)


if __name__ == "__main__":
    main()
