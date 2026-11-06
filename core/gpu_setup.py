"""Blender-side GPU setup, executed inside Blender via ``-P``::

    blender -b scene.blend -P core/gpu_setup.py -E CYCLES -o out/frame_#### -F PNG -f 1

Configuration is read from environment variables (the renderer sets them):

* ``BLENDER_GPU_BACKEND``   - preferred compute backend, ``OPTIX`` (default) or ``CUDA``
* ``BLENDER_SAMPLES``       - Cycles/EEVEE sample override, ``0`` keeps the scene default
* ``BLENDER_RESOLUTION_PCT``- resolution percentage override, ``0`` keeps the scene default
* ``BLENDER_TILE_SIZE``     - Cycles tile size override, ``0`` keeps the scene default

The pure helpers at the top do not need ``bpy`` and are unit-tested in ``tests/``.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping, Optional, Tuple

try:  # bpy only exists inside the Blender interpreter
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - exercised only outside Blender
    bpy = None

GPU_BACKENDS: Tuple[str, ...] = ("OPTIX", "CUDA")


def backend_order(preferred: Optional[str]) -> Tuple[str, ...]:
    """Return backends to try, starting with ``preferred`` and falling back to the rest.

    An unknown or empty preference uses the default order (OPTIX, then CUDA).
    """
    preferred = (preferred or "").strip().upper()
    if preferred not in GPU_BACKENDS:
        return GPU_BACKENDS
    return (preferred,) + tuple(b for b in GPU_BACKENDS if b != preferred)


def pick_backend(preferred: Optional[str], available: Iterable[str]) -> Optional[str]:
    """Pick the first backend in :func:`backend_order` that is available."""
    available_set = {a.upper() for a in available}
    for backend in backend_order(preferred):
        if backend in available_set:
            return backend
    return None


def _int_env(environ: Mapping[str, str], key: str, default: int = 0) -> int:
    raw = environ.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def read_overrides(environ: Mapping[str, str]) -> Dict[str, object]:
    """Parse render overrides from the environment. ``0`` means "keep scene value"."""
    return {
        "backend": (environ.get("BLENDER_GPU_BACKEND") or "OPTIX").strip().upper(),
        "samples": _int_env(environ, "BLENDER_SAMPLES"),
        "resolution_pct": _int_env(environ, "BLENDER_RESOLUTION_PCT"),
        "tile_size": _int_env(environ, "BLENDER_TILE_SIZE"),
    }


# ---------------------------------------------------------------------------
# bpy-dependent functions (only callable inside Blender)
# ---------------------------------------------------------------------------

def configure_gpu(preferred: str = "OPTIX") -> str:
    """Enable GPU devices for Cycles. Returns the backend used, or ``"CPU"``.

    When a GPU backend works, only its GPU devices are enabled and the scene
    device is set to GPU, so the CPU is not used as a render fallback.
    """
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in backend_order(preferred):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue  # this Blender build does not offer the backend
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for device in prefs.devices:
                device.use = device.type == backend
            bpy.context.scene.cycles.device = "GPU"
            print(f"[gpu_setup] using {backend}: {', '.join(d.name for d in gpus)}")
            return backend

    bpy.context.scene.cycles.device = "CPU"
    print("[gpu_setup] no GPU backend available, rendering on CPU")
    return "CPU"


def apply_overrides(scene, overrides: Mapping[str, object]) -> None:
    """Apply sample, resolution and tile overrides to a scene."""
    samples = int(overrides["samples"])
    if samples > 0:
        if hasattr(scene, "cycles"):
            scene.cycles.samples = samples
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = samples

    resolution = int(overrides["resolution_pct"])
    if resolution > 0:
        scene.render.resolution_percentage = resolution

    tile = int(overrides["tile_size"])
    if tile > 0 and hasattr(scene, "cycles") and hasattr(scene.cycles, "tile_size"):
        scene.cycles.tile_size = tile


def main() -> None:
    overrides = read_overrides(os.environ)
    scene = bpy.context.scene
    backend = configure_gpu(str(overrides["backend"]))
    apply_overrides(scene, overrides)
    print(
        f"[gpu_setup] backend={backend} samples={overrides['samples'] or 'scene'} "
        f"resolution={overrides['resolution_pct'] or 'scene'}% tile={overrides['tile_size'] or 'auto'}"
    )


if bpy is not None:  # pragma: no branch - runs when Blender executes this file via -P
    main()
