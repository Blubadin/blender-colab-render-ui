"""Unit tests for Blender CLI construction, path handling, output parsing and cancellation."""

import io
import sys
import time
from pathlib import Path

import pytest

from core import gpu_setup
from core.renderer import (
    GPU_SCRIPT,
    ProgressInfo,
    RenderJob,
    RenderRunner,
    build_command,
    build_env,
    create_zip,
    iter_output_lines,
    list_rendered_images,
    parse_progress_line,
    parse_saved_path,
)

BLENDER = "/opt/blender/blender"


def make_job(**overrides):
    base = dict(
        blend_file="/content/drive/MyDrive/scene.blend",
        output_dir="/content/drive/MyDrive/Blender_Renders/scene_1",
        engine="CYCLES",
        output_format="PNG",
        frame=1,
    )
    base.update(overrides)
    return RenderJob(**base)


# ---------------------------------------------------------------- command building

def test_single_frame_command_matches_spec():
    argv = build_command(BLENDER, make_job(frame=12))
    assert argv == [
        BLENDER,
        "-b", "/content/drive/MyDrive/scene.blend",
        "-P", str(GPU_SCRIPT),
        "-E", "CYCLES",
        "-o", str(Path("/content/drive/MyDrive/Blender_Renders/scene_1") / "frame_####"),
        "-F", "PNG",
        "-f", "12",
    ]


def test_animation_command_uses_range_step_and_animation_flag():
    job = make_job(frame=None, start=1, end=250, step=2, engine="BLENDER_EEVEE_NEXT", output_format="OPEN_EXR")
    argv = build_command(BLENDER, job)
    assert argv[argv.index("-E") + 1] == "BLENDER_EEVEE_NEXT"
    assert argv[argv.index("-F") + 1] == "OPEN_EXR"
    assert argv[argv.index("-s") + 1] == "1"
    assert argv[argv.index("-e") + 1] == "250"
    assert argv[argv.index("-j") + 1] == "2"
    assert "-a" in argv
    assert "-f" not in argv


def test_paths_with_spaces_and_quotes_stay_single_arguments():
    job = make_job(
        blend_file='/content/drive/MyDrive/My Scenes/"hero" shot.blend',
        output_dir="/content/drive/MyDrive/Blender Renders/final run",
    )
    argv = build_command(BLENDER, job)
    assert argv[argv.index("-b") + 1] == '/content/drive/MyDrive/My Scenes/"hero" shot.blend'
    assert str(Path("/content/drive/MyDrive/Blender Renders/final run") / "frame_####") in argv


def test_gpu_script_path_is_absolute_and_exists():
    assert Path(GPU_SCRIPT).is_absolute()
    assert Path(GPU_SCRIPT).name == "gpu_setup.py"
    assert Path(GPU_SCRIPT).exists()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"blend_file": "scene.fbx"}, ".blend"),
        ({"engine": "WORKBENCH"}, "engine"),
        ({"output_format": "BMP"}, "format"),
        ({"gpu_backend": "METAL"}, "GPU backend"),
        ({"resolution_pct": 0}, "Resolution"),
        ({"samples": -1}, "Samples"),
        ({"frame": -3}, "Frame"),
        ({"frame": None, "start": 10, "end": 5}, "frame range"),
        ({"frame": None, "start": 1, "end": 5, "step": 0}, "step"),
        ({"frame": None, "start": None, "end": 5}, "start and end"),
    ],
)
def test_invalid_jobs_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_command(BLENDER, make_job(**overrides))


def test_build_env_passes_gpu_settings():
    job = make_job(gpu_backend="CUDA", samples=64, resolution_pct=50, tile_size=256)
    env = build_env(job, base_env={"PATH": "/usr/bin"})
    assert env["PATH"] == "/usr/bin"
    assert env["BLENDER_GPU_BACKEND"] == "CUDA"
    assert env["BLENDER_SAMPLES"] == "64"
    assert env["BLENDER_RESOLUTION_PCT"] == "50"
    assert env["BLENDER_TILE_SIZE"] == "256"


# ---------------------------------------------------------------- stdout parsing

PROGRESS = (
    "Fra:3 Mem:45.67M (Peak 50.00M) | Time:00:01.23 | Remaining:00:04.00 | "
    "Mem:45.67M | Sample 64/128"
)


def test_parse_progress_line_extracts_all_fields():
    info = parse_progress_line(PROGRESS)
    assert isinstance(info, ProgressInfo)
    assert info.frame == 3
    assert info.elapsed == "00:01.23"
    assert info.remaining == "00:04.00"
    assert info.memory == "45.67M"
    assert info.sample == 64
    assert info.total_samples == 128
    assert info.percent == 50.0


def test_parse_progress_line_without_samples_has_no_percent():
    info = parse_progress_line("Fra:7 Mem:10.00M | Time:00:00.50 | Remaining:- | Loading")
    assert info.frame == 7
    assert info.sample is None
    assert info.percent is None
    assert info.remaining == "-"


@pytest.mark.parametrize("line", ["", "Blender 4.2.3 (hash) built", "Color management: sRGB"])
def test_non_progress_lines_return_none(line):
    assert parse_progress_line(line) is None


def test_parse_saved_path_handles_spaces():
    line = "Saved: '/content/drive/MyDrive/Blender Renders/frame_0007.png'"
    assert parse_saved_path(line) == "/content/drive/MyDrive/Blender Renders/frame_0007.png"
    assert parse_saved_path("Time: 00:01.00") is None


def test_iter_output_lines_splits_on_carriage_returns():
    stream = io.StringIO("Fra:1 Sample 1/4\rFra:1 Sample 2/4\r\nSaved: 'x.png'\n")
    assert list(iter_output_lines(stream)) == [
        "Fra:1 Sample 1/4",
        "Fra:1 Sample 2/4",
        "Saved: 'x.png'",
    ]


# ---------------------------------------------------------------- GPU helpers (gpu_setup.py)

def test_backend_order_prefers_optix_then_cuda():
    assert gpu_setup.backend_order("OPTIX") == ("OPTIX", "CUDA")
    assert gpu_setup.backend_order("CUDA") == ("CUDA", "OPTIX")
    assert gpu_setup.backend_order("nonsense") == ("OPTIX", "CUDA")


def test_pick_backend_falls_back_to_cuda_when_optix_missing():
    assert gpu_setup.pick_backend("OPTIX", ["CUDA"]) == "CUDA"
    assert gpu_setup.pick_backend("OPTIX", ["CUDA", "OPTIX"]) == "OPTIX"
    assert gpu_setup.pick_backend("OPTIX", []) is None


def test_read_overrides_defaults_and_values():
    defaults = gpu_setup.read_overrides({})
    assert defaults == {"backend": "OPTIX", "samples": 0, "resolution_pct": 0, "tile_size": 0}
    values = gpu_setup.read_overrides(
        {"BLENDER_GPU_BACKEND": "cuda", "BLENDER_SAMPLES": "32", "BLENDER_RESOLUTION_PCT": "75", "BLENDER_TILE_SIZE": "64"}
    )
    assert values == {"backend": "CUDA", "samples": 32, "resolution_pct": 75, "tile_size": 64}


def test_read_overrides_rejects_bad_numbers():
    with pytest.raises(ValueError, match="BLENDER_SAMPLES"):
        gpu_setup.read_overrides({"BLENDER_SAMPLES": "lots"})


# ---------------------------------------------------------------- process runner

def test_runner_streams_lines_and_reports_done(tmp_path):
    script = (
        "import sys\n"
        "print('Fra:1 Mem:1.00M | Time:00:00.10 | Sample 1/2', flush=True)\n"
        "print(\"Saved: '\" + sys.argv[1] + \"'\", flush=True)\n"
    )
    runner = RenderRunner([sys.executable, "-c", script, "out frame.png"])
    progress, saved, logs = [], [], []
    result = runner.run(on_log=logs.append, on_progress=progress.append, on_saved=saved.append)
    assert result.status == "done"
    assert result.returncode == 0
    assert progress and progress[0].frame == 1
    assert saved == ["out frame.png"]
    assert len(logs) == 2


def test_runner_reports_failure_exit_code():
    runner = RenderRunner([sys.executable, "-c", "import sys; sys.exit(3)"])
    result = runner.run()
    assert result.status == "failed"
    assert result.returncode == 3


def test_runner_reports_missing_binary():
    result = RenderRunner(["/definitely/not/a/blender-binary"]).run()
    assert result.status == "failed"
    assert "Could not start Blender" in result.error


def test_cancel_terminates_running_process():
    script = "import time\nprint('Fra:1 Sample 1/100', flush=True)\ntime.sleep(60)\n"
    runner = RenderRunner([sys.executable, "-c", script])
    results = {}

    import threading

    thread = threading.Thread(target=lambda: results.setdefault("r", runner.run()))
    thread.start()
    time.sleep(1.0)  # let the child start
    runner.cancel(grace_seconds=5)
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert results["r"].status == "cancelled"


def test_cancel_before_run_skips_process():
    runner = RenderRunner([sys.executable, "-c", "print('should not run')"])
    runner.cancel()
    assert runner.run().status == "cancelled"


# ---------------------------------------------------------------- output helpers

def test_list_rendered_images_filters_and_sorts(tmp_path):
    (tmp_path / "sub").mkdir()
    a = tmp_path / "frame_0001.png"
    b = tmp_path / "sub" / "frame_0002.jpg"
    c = tmp_path / "frame_0003.exr"
    for f in (a, b, c):
        f.write_bytes(b"x")
    images = list_rendered_images(tmp_path)
    assert c not in images  # EXR is not browser-displayable
    assert set(images) == {a, b}


def test_create_zip_contains_files_by_basename(tmp_path):
    import zipfile

    f1 = tmp_path / "a" / "frame_0001.png"
    f1.parent.mkdir()
    f1.write_bytes(b"png")
    zip_path = create_zip([f1], tmp_path / "out.zip")
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["frame_0001.png"]
